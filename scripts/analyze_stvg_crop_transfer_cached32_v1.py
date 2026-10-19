"""Read saved scalar qualities and prediction geometry; no GT/model/scoring calls.

This is a descriptive chain decomposition, not a new prediction or causal
intervention. All support masks are disclosed and never select online outputs.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import hashlib
import json
from pathlib import Path
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / 'artifacts/stvg_crop_transfer_cached32_v1'
P = ROOT / 'results/stvg_error_state/2026-10-19/crop_transfer_cached32'
OLD = ROOT / 'artifacts/stvg_observation_factorial32_v1'
PRE = ROOT / 'artifacts/stvg_dino_endpoint_confirm32_v1'
PUBLIC = ROOT / 'results/stvg_error_state/2026-10-19/observation_factorial32'


def read(p):
    return json.loads(p.read_text())


def put(p, v):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(v, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def chart(b):
    b = np.asarray(b, dtype=float)
    assert b.ndim == 2 and b.shape[1] == 4 and (b[:, 2:] > 0).all()
    return np.column_stack((b[:, :2], np.log(b[:, 2:])))


def stats(v):
    v = np.asarray(v, dtype=float)
    if not len(v):
        return dict(parents=0, mean=None, CI95=None)
    ids = np.random.default_rng(20261018).integers(len(v), size=(10000, len(v)))
    return dict(parents=len(v), mean=float(v.mean()),
                CI95=np.quantile(v[ids].mean(1), [.025, .975]).tolist(),
                positive=int((v > 1e-10).sum()), negative=int((v < -1e-10).sum()),
                neutral=int((np.abs(v) <= 1e-10).sum()), per_parent_pp=v.tolist())


def main():
    import torch
    torch.set_num_threads(2)
    started = time.perf_counter()
    assert not torch.cuda.is_initialized()
    assert not D.exists() and not P.exists(), 'Preserve completed analysis; use a new version.'
    plan = read(OLD / 'PLAN_LOCK.json')
    ref = read(PUBLIC / 'REFERENCE_SEPARATION_PUBLIC.json')
    main_scores = read(PUBLIC / 'RESULTS_PUBLIC.json')
    scores = {r['alias']: r for r in main_scores['rows']}
    calibration = read(ROOT / 'artifacts/stvg_joint_robust_correction32_v1/SOURCE_CALIBRATION.json')['source_domains']
    inputs = [OLD / 'PLAN_LOCK.json', PUBLIC / 'REFERENCE_SEPARATION_PUBLIC.json',
              PUBLIC / 'RESULTS_PUBLIC.json', ROOT / 'artifacts/stvg_joint_robust_correction32_v1/SOURCE_CALIBRATION.json']
    inputs += [ROOT / 'methods' / n for n in ['CURRENT_METHOD.json', 'MAIN_METHOD.json', 'PAPER_METHOD.json']]
    for r in plan['records']:
        inputs += [OLD / 'predictions' / (r['alias'] + '.pt'),
                   PRE / 'source' / (r['alias'] + '.pt'), PRE / 'E' / (r['alias'] + '.pt')]
    pins = {str(p.relative_to(ROOT)): sha(p) for p in inputs}
    put(D / 'INPUT_LOCK.json', dict(scope='saved evidence only, no new task scoring', pins=pins,
        new_models=0, new_GT_annotation_reads=0, new_predictions=0, new_student_updates=0))
    byframe = {(f['alias'], f['position']): f for f in ref['frames']}
    frame_rows, query_rows = [], []
    max_uniform_error = 0.
    for r in plan['records']:
        a, ds = r['alias'], r['dataset']
        z = torch.load(OLD / 'predictions' / (a + '.pt'), map_location='cpu', weights_only=False)
        s = torch.load(PRE / 'source' / (a + '.pt'), map_location='cpu', weights_only=False)
        e = torch.load(PRE / 'E' / (a + '.pt'), map_location='cpu', weights_only=False)
        source = np.asarray(s['Source'], float)
        E = np.asarray(e['E'], float)
        S = np.asarray(z['outputs']['S_crop'], float)
        shift = chart(S) - chart(E)
        uniform_error = float(np.max(np.abs(shift - shift[0])))
        assert uniform_error < 2e-12
        max_uniform_error = max(max_uniform_error, uniform_error)
        choice = dict(zip(z['spatial']['pool_positions'], z['spatial']['estimate']['selected']))
        scale = np.asarray(calibration['vidstg' if ds == 'hc2' else 'hc2']['scale'], float)
        selected_residuals = []
        top_residuals = []
        fs = []
        for ob in z['spatial']['observations']:
            pos = ob['position']
            f = dict(byframe[a, pos])
            cp = ob.get('probe', {})
            accepted = bool(cp.get('accepted', False))
            f.update(crop_top_index=None, selected_index=choice.get(pos, -1),
                     selected_is_crop_top=None, selected_anchor_center_mismatch_in_ref_widths=None,
                     selected_anchor_log_size_mismatch=None)
            if accepted:
                b = np.asarray(cp['boxes'], float)
                top = int(np.argmax(cp['target_scores']))
                k = choice.get(pos, -1)
                f['crop_top_index'] = top
                top_residuals.append((chart(b[[top]])[0] - chart(source[[pos]])[0]) / scale)
                if k >= 0:
                    f['selected_is_crop_top'] = (k == top)
                    selected_residuals.append((chart(b[[k]])[0] - chart(source[[pos]])[0]) / scale)
                    f['selected_anchor_center_mismatch_in_ref_widths'] = float(np.linalg.norm((S[pos, :2] - b[k, :2]) / b[k, 2:]))
                    f['selected_anchor_log_size_mismatch'] = float(np.linalg.norm(np.log(S[pos, 2:] / b[k, 2:])))
            common = bool(f['GT_known'] and accepted and f['new_E_selected_reference_IoU'] is not None)
            f['common_chain_support'] = common
            f['chain_pp'] = None
            if common:
                base, raw, selected, propagated = [f[k] for k in ['E_IoU', 'crop_raw_top_IoU', 'new_E_selected_reference_IoU', 'propagated_S_IoU']]
                terms = dict(raw_minus_E=100*(raw-base), selection=100*(selected-raw),
                             propagation=100*(propagated-selected), final_minus_E=100*(propagated-base))
                assert abs(terms['raw_minus_E']+terms['selection']+terms['propagation']-terms['final_minus_E']) < 1e-10
                f['chain_pp'] = terms
                f['raw_gain_lost_in_selection'] = raw > base+1e-12 and selected <= base+1e-12
                f['selected_gain_lost_in_propagation'] = selected > base+1e-12 and propagated <= base+1e-12
                f['raw_beneficial'] = raw > base+1e-12
                f['selected_beneficial'] = selected > base+1e-12
            frame_rows.append(f)
            fs.append(f)
        common_fs = [f for f in fs if f['common_chain_support']]
        chain = {key: float(np.mean([f['chain_pp'][key] for f in common_fs])) for key in ['raw_minus_E', 'selection', 'propagation', 'final_minus_E']} if common_fs else None
        def scatter(values):
            v = np.asarray(values, float)
            return float(np.sqrt(np.square(v-v.mean(0)).sum(1).mean())) if len(v) else None
        query_rows.append(dict(alias=a, dataset=ds, planned_frames=len(fs),
            known_accepted_crop_frames=sum(f['GT_known'] and f['crop_admitted'] for f in fs),
            common_chain_frames=len(common_fs), chain_pp=chain,
            crop_selected_frames=sum(f['selected_is_crop_top'] is not None for f in fs),
            selected_crop_non_top_frames=sum(f['selected_is_crop_top'] is False for f in fs),
            top_residual_source_scale_RMS=scatter(top_residuals),
            selected_crop_residual_source_scale_RMS=scatter(selected_residuals),
            selected_all_pool_count=z['spatial']['estimate']['support_frames'],
            rejected_all_pool_count=z['spatial']['estimate']['rejected_frames'],
            geometry_objective=z['spatial']['estimate']['geometry_objective'],
            score_objective=z['spatial']['estimate']['score_objective'],
            uniform_E_to_S_chart_shift_max_error=uniform_error,
            saved_full_v_change_pp=scores[a]['changes']['S_crop']['v'],
            saved_unobserved_actual_s_change_pp=scores[a]['changes']['S_crop']['unobserved_actual_s']))
    groups = {}
    for ds in ['hc2', 'vidstg']:
        qs = [q for q in query_rows if q['dataset'] == ds]
        fs = [f for f in frame_rows if f['dataset'] == ds]
        common = [f for f in fs if f['common_chain_support']]
        groups[ds] = dict(main_parents=16,
            accepted_known_frames=sum(f['crop_admitted'] and f['GT_known'] for f in fs),
            accepted_known_rejected_by_E=sum(f['crop_admitted'] and f['GT_known'] and f['new_E_selected_reference_IoU'] is None for f in fs),
            common_chain_frames=len(common), common_chain_parents=sum(q['chain_pp'] is not None for q in qs),
            chain_pp={key: stats([q['chain_pp'][key] for q in qs if q['chain_pp'] is not None]) for key in ['raw_minus_E', 'selection', 'propagation', 'final_minus_E']},
            raw_beneficial=sum(f['raw_beneficial'] for f in common),
            raw_gain_lost_in_selection=sum(f['raw_gain_lost_in_selection'] for f in common),
            selected_beneficial=sum(f['selected_beneficial'] for f in common),
            selected_gain_lost_in_propagation=sum(f['selected_gain_lost_in_propagation'] for f in common),
            all_selected_crop_frames=sum(q['crop_selected_frames'] for q in qs),
            all_selected_crop_non_top_frames=sum(q['selected_crop_non_top_frames'] for q in qs),
            selected_crop_inconsistent_queries=sum(q['selected_crop_residual_source_scale_RMS'] is not None and q['selected_crop_residual_source_scale_RMS'] > 1e-6 for q in qs),
            saved_main_full_v_change_pp=main_scores['groups'][ds]['paired']['S_crop_minus_E_MBR']['v'],
            saved_main_actual_s_change_pp=main_scores['groups'][ds]['paired']['S_crop_minus_E_MBR']['unobserved_actual_s'])
    result = dict(status='completed_cached_descriptive_diagnosis', count=32, groups=groups,
        rows=query_rows, frames=frame_rows,
        interpretation='Telescope of saved qualities on common support; not causal intervention or a new output. Selection and propagation losses may interact.',
        support='Known admitted crop frames with a selected reference; within-query then parent mean; unknown never filled with a quality value.',
        no_new_GT_scoring=True, no_new_predictions=True, OPD_not_run=True,
        uniform_shift_max_error=max_uniform_error)
    put(P / 'CHAIN_PUBLIC.json', result)
    assert not torch.cuda.is_initialized()
    assert all(sha(ROOT/f) == value for f, value in pins.items())
    cost = dict(CPU_seconds=time.perf_counter()-started, new_GPU=0, new_GT_annotation_reads=0,
        new_task_score_calls=0, new_model_or_teacher_calls=0, new_training=0, new_student_updates=0,
        outputs_replaced=0, historical_cache_cost_excluded=True)
    put(P / 'COST_BREAKDOWN.json', cost)
    put(D / 'STATUS.json', dict(status='completed_cached_analysis_pending_review', count=32,
        no_automatic_experiment=True, registries_unchanged=True))
    print(json.dumps(dict(groups=groups, cost=cost, uniform_shift_max_error=max_uniform_error), ensure_ascii=False))


if __name__ == '__main__':
    main()
