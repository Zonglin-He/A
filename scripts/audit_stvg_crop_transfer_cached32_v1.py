"""Independent saved-quality arithmetic and no-GT selection objective audit."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / 'artifacts/stvg_crop_transfer_cached32_v1'
P = ROOT / 'results/stvg_error_state/2026-10-19/crop_transfer_cached32'


def read(p):
    return json.loads(p.read_text())


def put(p, x):
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def main():
    import torch
    torch.set_num_threads(2)
    assert not torch.cuda.is_initialized()
    source = read(ROOT / 'results/stvg_error_state/2026-10-19/observation_factorial32/REFERENCE_SEPARATION_PUBLIC.json')
    result = read(P / 'CHAIN_PUBLIC.json')
    original = {(f['alias'], f['position']): f for f in source['frames']}
    checks = dict(saved_scalar_frames=0, stage_terms=0, bootstrap_endpoints=0, uniform_transforms=0)
    for f in result['frames']:
        old = original[f['alias'], f['position']]
        assert all(f[key] == value for key, value in old.items())
        checks['saved_scalar_frames'] += 1
    for ds, group in result['groups'].items():
        records = [f for f in source['frames'] if f['dataset'] == ds and f['GT_known'] and f['crop_admitted'] and f['new_E_selected_reference_IoU'] is not None]
        for key, hi, lo in [('raw_minus_E', 'crop_raw_top_IoU', 'E_IoU'),
                            ('selection', 'new_E_selected_reference_IoU', 'crop_raw_top_IoU'),
                            ('propagation', 'propagated_S_IoU', 'new_E_selected_reference_IoU'),
                            ('final_minus_E', 'propagated_S_IoU', 'E_IoU')]:
            aliases = sorted({f['alias'] for f in records})
            values = np.array([np.mean([100*(f[hi]-f[lo]) for f in records if f['alias']==a]) for a in aliases])
            expected = group['chain_pp'][key]
            assert len(aliases)==expected['parents'] and abs(values.mean()-expected['mean']) < 1e-12
            ids = np.random.default_rng(20261018).integers(len(values), size=(10000, len(values)))
            assert np.max(abs(np.quantile(values[ids].mean(1), [.025, .975])-expected['CI95'])) < 1e-12
            checks['bootstrap_endpoints'] += 1
            checks['stage_terms'] += len(records)
        assert len(records)==group['common_chain_frames']
        assert sum(f['crop_raw_top_IoU'] > f['E_IoU']+1e-12 and f['new_E_selected_reference_IoU'] <= f['E_IoU']+1e-12 for f in records)==group['raw_gain_lost_in_selection']
        assert sum(f['new_E_selected_reference_IoU'] > f['E_IoU']+1e-12 and f['propagated_S_IoU'] <= f['E_IoU']+1e-12 for f in records)==group['selected_gain_lost_in_propagation']
    cal = read(ROOT / 'artifacts/stvg_joint_robust_correction32_v1/SOURCE_CALIBRATION.json')['source_domains']
    behavior = {ds: dict(selected_crop_frames=0, selected_below_frame_admission_score=0, selected_non_top=0) for ds in ['hc2', 'vidstg']}
    cases = []
    for q in result['rows']:
        a, ds = q['alias'], q['dataset']
        z = torch.load(ROOT/'artifacts/stvg_observation_factorial32_v1/predictions'/(a+'.pt'), map_location='cpu', weights_only=False)
        s = torch.load(ROOT/'artifacts/stvg_dino_endpoint_confirm32_v1/source'/(a+'.pt'), map_location='cpu', weights_only=False)
        e = torch.load(ROOT/'artifacts/stvg_dino_endpoint_confirm32_v1/E'/(a+'.pt'), map_location='cpu', weights_only=False)
        def enc(b):
            b=np.asarray(b, dtype=float)
            return np.concatenate((b[..., :2], np.log(b[..., 2:])), axis=-1)
        diff=enc(z['spatial']['boxes'])-enc(e['E'])
        assert np.max(abs(np.ptp(diff, axis=0))) < 1e-12
        checks['uniform_transforms'] += 1
        selected=dict(zip(z['spatial']['pool_positions'], z['spatial']['estimate']['selected']))
        frames=[]; top_res=[]
        for ob in z['spatial']['observations']:
            probe=ob.get('probe', {})
            if not probe.get('accepted', False): continue
            pos=ob['position'];k=selected.get(pos,-1);scores=np.asarray(probe['target_scores'], float)
            j=int(scores.argmax());boxes=np.asarray(probe['boxes'],float)
            top_res.append(enc(boxes[j])-enc(np.asarray(s['Source'])[pos]))
            if k < 0: continue
            behavior[ds]['selected_crop_frames'] += 1
            behavior[ds]['selected_below_frame_admission_score'] += int(scores[k] < .35)
            behavior[ds]['selected_non_top'] += int(k!=j)
            frames.append(dict(position=pos, top_index=j, selected_index=k,
                               raw_top_target_score=float(scores[j]), selected_target_score=float(scores[k])))
        if a=='NV016':
            settings=cal['hc2'];scale=np.asarray(settings['scale']);res=np.asarray(top_res)
            # A descriptive forced-all-top mean objective, NOT the best top-only
            # subset, not a new output and not a GT-optimal candidate selection.
            geo=np.minimum(settings['truncation'], np.square((res-res.mean(0))/scale).sum(1))
            cases.append(dict(alias=a, dataset=ds, frames=frames,
                chosen_E_total=z['spatial']['estimate']['objective'],
                chosen_E_geometry=z['spatial']['estimate']['geometry_objective'],
                chosen_E_score_penalty=z['spatial']['estimate']['score_objective'],
                forced_all_top_mean_truncated_geometry=float(geo.sum()),
                forced_all_top_relative_score_penalty=0.,
                not_top_subset_optimum=True, GT_not_used_in_objective=True))
    for ds in behavior:
        assert behavior[ds]['selected_crop_frames']==result['groups'][ds]['all_selected_crop_frames']
        assert behavior[ds]['selected_non_top']==result['groups'][ds]['all_selected_crop_non_top_frames']
    pins=read(D/'INPUT_LOCK.json')['pins']
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==value for p,value in pins.items())
    assert not torch.cuda.is_initialized()
    put(P/'SELECTION_OBJECTIVE_PUBLIC.json', dict(status='cached_no_GT_geometry_diagnostic', behavior=behavior, cases=cases,
        frame_gate_not_per_candidate_gate=True, no_new_output_or_scoring=True))
    audit=dict(status='pass', checks=checks, pins_preserved=len(pins),
        no_new_GT_annotation_reads=True, no_new_task_score_calls=True,
        no_new_model_or_GPU_or_training_or_student_updates=True,
        scope='independent saved scalar/conditional telescope/bootstrap; exact global-transform geometry; selected-score behavior')
    put(P/'ROOT_INDEPENDENT_AUDIT.json',audit)
    print(json.dumps(dict(audit=audit, behavior=behavior, cases=cases)))


if __name__=='__main__': main()
