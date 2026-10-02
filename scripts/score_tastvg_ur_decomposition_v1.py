"""Post-global-seal CPU metric, arithmetic and paired functional audit."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import sys
import time
import collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scripts.tastvg_ur_decomposition_common_v1 import *

BRANCHES = ['U', 'R', 'Specific']
FIELDS = ['pre_v', 'pre_s'] + [b+'_'+m for b in BRANCHES for m in ['v', 's']]
FIELDS += ['delta_'+b+'_'+m for b in BRANCHES for m in ['v', 's', 'free_v', 'free_t']]
FIELDS += ['delta_R_U_v', 'delta_R_U_s', 'cosine', 'lag']


def summarize(rows):
    from scripts.score_tastvg_best_quick_v1 import source_summary
    out = source_summary(rows, FIELDS)
    out['effects'] = {}
    for branch in BRANCHES:
        field = 'delta_'+branch+'_v'
        take = [{**r, 'gross_gain': max(r[field], 0.), 'gross_loss': max(-r[field], 0.)} for r in rows]
        out['effects'][branch] = dict(positive=sum(r[field] > 1e-12 for r in rows),
            negative=sum(r[field] < -1e-12 for r in rows), zero=sum(abs(r[field]) <= 1e-12 for r in rows),
            harm_gt5pp=sum(r[field] < -.05 for r in rows),
            intervals_changed=sum(r[branch+'_interval'] != r['pre_interval'] for r in rows),
            gross_source_macro=source_summary(take, ['gross_gain', 'gross_loss']),
            correctness={str(t): dict(rescued=sum(r['pre_v'] <= t < r[branch+'_v'] for r in rows),
                                     destroyed=sum(r[branch+'_v'] <= t < r['pre_v'] for r in rows)) for t in [.3, .5]})
    out['coverage'] = dict(target_cells=len(rows), both_nonempty=sum(r['both_nonempty'] for r in rows),
                          donor_sources=len({r['source_id'] for r in rows}),
                          target_sources=len({r['target_source_id'] for r in rows}))
    return out


def role_rows(rows, role):
    return [r for r in rows if (r['broader'] if role == 'broader' else role in r['roles'])]


def contrasts(rows):
    from scripts.score_tastvg_best_quick_v1 import source_summary
    buckets = collections.defaultdict(list)
    for row in rows:
        buckets[row['condition'], row['order'], row['donor_arrival']].append(row)
    result = []
    for key, selected in buckets.items():
        by_role = {role: next(r for r in selected if role in r['roles']) for role in ['self', 'next', 'near', 'far']}
        broad = [r for r in selected if r['broader']]
        record = {k: selected[0][k] for k in ['condition', 'order', 'donor_arrival', 'source_id']}
        for branch in BRANCHES:
            field = 'delta_'+branch+'_v'; average = float(np.mean([r[field] for r in broad]))
            for role in ['self', 'near', 'next', 'far']:
                record['delta_'+branch+'_'+role+'_broader_v'] = by_role[role][field]-average
            record['delta_'+branch+'_self_far_v'] = by_role['self'][field]-by_role['far'][field]
            record['delta_'+branch+'_near_far_v'] = by_role['near'][field]-by_role['far'][field]
        result.append(record)
    fields = [k for k in result[0] if k.startswith('delta_')]
    return result, source_summary(result, fields)


def aggregate(rows, conditions):
    from scripts.score_tastvg_best_quick_v1 import source_summary
    result = {}
    for group in ['corruption', 'clean', *conditions[1:]]:
        selected = [r for r in rows if (r['condition'] != 'clean' if group == 'corruption' else r['condition'] == group)]
        result[group] = dict(roles={}, contrasts={})
        for role in ['self', 'next', 'near', 'far', 'broader']:
            take = role_rows(selected, role)
            result[group]['roles'][role] = summarize(take)
            result[group]['roles'][role]['both_nonempty_only'] = summarize([r for r in take if r['both_nonempty']])
            result[group]['roles'][role]['target_clustered_sensitivity'] = source_summary(
                [{**r, 'source_id': r['target_source_id']} for r in take],
                ['delta_'+b+'_v' for b in BRANCHES]+['delta_R_U_v'])
        _, result[group]['contrasts'] = contrasts(selected)
    return result


def run():
    import torch
    torch.set_num_threads(2)
    from scripts.audit_tastvg_ur_write_v1 import validate_write
    from methods.decota_final_simplified_v1.tensors import state_hash
    from vg_tta.tastvg_paper48_metrics_v1 import xyxy
    from vg_tta.tastvg_paper48_hc2_metrics_v1 import HC2DenseMetric
    from vg_tta.tastvg_paper_readouts_v1 import dense_official_metrics
    from scripts.diagnose_tastvg_pipeline_cpu_v1 import evaluator
    from vg_tta.tastvg_reference_selection_v1 import pairwise
    verify(); cohort = read(BASE / 'COHORT.json'); config = read(BASE / 'CONFIG.json')
    barrier = read(BASE / 'GLOBAL_PREDICTION_BARRIER.json')
    assert barrier['status'] == 'sealed' and barrier['donors'] == 96 and barrier['targets'] == 1392
    assert not barrier['GT_read'] and sha(BASE / 'COHORT.json') == barrier['cohort_sha256']
    for relative, digest in barrier['files'].items():
        path = BASE / relative
        assert sha(path) == digest and read(path)['time'] < barrier['time']
        receipt(path.with_suffix('.pt'))
    context = load(CONTEXT); plan = read(OLD / 'hc2/PLAN.json'); checks = collections.Counter()
    for row in cohort['rows']:
        sequence = cohort['orders'][row['order']]; i = row['donor_arrival']; parent = row['donor_parent']
        eligible = [j for j in range(i+1, 32) if j % 4 != 0]
        similarities = {j: float(context['cosine'][context['parents'].index(parent), context['parents'].index(sequence[j])]) for j in eligible}
        expected = dict(self=i, next=eligible[0], near=min(eligible, key=lambda j: (-similarities[j], j)),
                        far=min(eligible, key=lambda j: (similarities[j], j)))
        assert row['roles'] == expected
        assert [t['arrival'] for t in row['targets']] == [i, *eligible]
        for target in row['targets']:
            assert target['parent'] == sequence[target['arrival']]
            assert target['roles'] == [role for role, j in expected.items() if j == target['arrival']]
            assert target['broader'] == (target['arrival'] in eligible)
            if target['broader']:
                assert target['source_id'] != row['source_id'] and target['cosine'] == similarities[target['arrival']]
            checks['target_selection'] += 1
    label_path = POOL / 'hc2/GT_LABELS_search.json'
    write(BASE / 'SCORING_LOCK.json', dict(barrier_sha256=sha(BASE / 'GLOBAL_PREDICTION_BARRIER.json'),
        labels_sha256=sha(label_path), pins={str(Path(__file__).relative_to(ROOT)): sha(Path(__file__)),
        'scripts/audit_tastvg_ur_write_v1.py': sha(ROOT / 'scripts/audit_tastvg_ur_write_v1.py'),
        'scripts/score_tastvg_best_quick_v1.py': sha(ROOT / 'scripts/score_tastvg_best_quick_v1.py'),
        'vg_tta/tastvg_paper48_hc2_metrics_v1.py': sha(ROOT / 'vg_tta/tastvg_paper48_hc2_metrics_v1.py')}, time=time.time()))
    write(BASE / 'GT_EXPOSURE.json', dict(after_global_barrier=True, historically_exposed=True,
        scope='diagnostic scoring only; no threshold/parameter/target reselection', time=time.time()))
    labels = read(label_path); metric = HC2DenseMetric(); max_error = 0.
    def evaluate(pred, parent, indices):
        nonlocal max_error
        row = plan['rows'][parent]; gt = labels[str(parent)]
        truth = {int(k): v for k, v in gt['truth'].items()}; span = gt['span']; ids = row['frame_ids']
        boxes = np.maximum(xyxy(pred['boxes'], row['input']['width'], row['input']['height']), 0)
        interval = [ids[indices[0]], ids[indices[1]]+1]
        actual = metric(boxes, ids, interval, truth, span)
        independent = dense_official_metrics(boxes, ids, interval, truth, span)
        for field in ['m_vIoU', 'm_tIoU', 'vIoU@0.3', 'vIoU@0.5']:
            err = abs(actual[field]-independent[field]); assert err < 1e-10
            max_error = max(max_error, err); checks['dense_independent_scalars'] += 1
        small = evaluator(pred['boxes'], row, truth, span, True)(indices)
        assert abs(small['v']-actual['m_vIoU']) < 1e-10
        dense_ids = np.asarray(sorted(truth))
        interpolated = np.column_stack([np.interp(dense_ids, ids, boxes[:, k]) for k in range(4)])
        target = np.asarray([truth[int(fid)] for fid in dense_ids])
        inter = np.maximum(np.minimum(interpolated[:, 2:], target[:, 2:])-np.maximum(interpolated[:, :2], target[:, :2]), 0).prod(-1)
        union = np.maximum(interpolated[:, 2:]-interpolated[:, :2], 0).prod(-1)+np.maximum(target[:, 2:]-target[:, :2], 0).prod(-1)-inter
        iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        iou[(dense_ids < ids[0]) | (dense_ids > ids[-1])] = 0
        err = abs(float(iou.mean())-actual['sIoU_dense_GT']); assert err < 1e-10
        max_error = max(max_error, err); checks['dense_s_independent'] += 1
        return dict(v=float(actual['m_vIoU']), t=float(actual['m_tIoU']),
                    s=float(actual['sIoU_dense_GT']), correct30=float(actual['vIoU@0.3']), correct50=float(actual['vIoU@0.5']))
    rows, writes = [], []; arithmetic_errors = collections.defaultdict(float)
    for row in cohort['rows']:
        wp = BASE / 'writes' / row['condition'] / row['order'] / f'{row["donor_arrival"]:05}.pt'
        value = load(wp); assert value['row'] == row
        cs, es = validate_write(value, config); checks.update(cs)
        for name, error in es.items():
            arithmetic_errors[name] = max(arithmetic_errors[name], error)
        probe_utilities = [evaluate(p, row['donor_parent'], value['donor_predictions']['pre']['indices'])['v'] for p in value['probes']]
        wr = {k: row[k] for k in ['condition', 'order', 'donor_arrival', 'source_id']}
        wr.update(geometry=value['geometry'], both_nonempty=all(value['metadata'][b]['available'] for b in ['U', 'R']),
                  U=value['metadata']['U'], R=value['metadata']['R'], probe_v=probe_utilities)
        for branch in ['U', 'R']:
            wr[branch] = {k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in wr[branch].items()}
            wr[branch]['critic_pairwise'] = pairwise(wr[branch]['rewards'], probe_utilities)
            selected = 0 if wr[branch]['rewards'] is None else int(np.argmax(wr[branch]['rewards']))
            wr[branch]['selected_index'] = selected
            wr[branch]['selected_v'] = probe_utilities[selected]
        writes.append(wr)
        for target in row['targets']:
            path = BASE / 'predictions' / row['condition'] / row['order'] / f'{row["donor_arrival"]:05}' / f'{target["arrival"]:05}.pt'
            x = load(path)
            assert x['target'] == target and x['write_payload_sha256'] == sha(wp)
            assert x['states_sha256'] == value['state_hashes']
            assert not x['GT_read'] and x['target_updates'] == x['new_experts'] == x['new_backbone_forwards'] == 0
            cr = read(POOL / 'hc2/capture' / row['condition'] / f'{target["parent"]:05}.json')
            assert cr['sha256'] == x['capture_sha256'] and cr['pixel_sha256'] == x['pixel_sha256']
            assert x['interval'] == x['predictions']['pre']['indices']
            checks['prediction_write_capture_bindings'] += 1
            out = dict(source_id=row['source_id'], donor_source_id=row['source_id'], target_source_id=target['source_id'],
                condition=row['condition'], order=row['order'], donor_arrival=row['donor_arrival'],
                target_arrival=target['arrival'], roles=target['roles'], broader=target['broader'],
                cosine=target['cosine'], lag=target['arrival']-row['donor_arrival'], both_nonempty=wr['both_nonempty'])
            for branch in ['pre', *BRANCHES]:
                pred = x['predictions'][branch]; fixed = evaluate(pred, target['parent'], x['interval'])
                free = fixed if pred['indices'] == x['interval'] else evaluate(pred, target['parent'], pred['indices'])
                out.update({branch+'_'+k: v for k, v in fixed.items()})
                out[branch+'_free_v'] = free['v']; out[branch+'_free_t'] = free['t']; out[branch+'_interval'] = pred['indices']
            for branch in BRANCHES:
                for field in ['v', 's', 'free_v', 'free_t']:
                    out['delta_'+branch+'_'+field] = out[branch+'_'+field]-out['pre_'+field]
            out['delta_R_U_v'] = out['R_v']-out['U_v']; out['delta_R_U_s'] = out['R_s']-out['U_s']
            rows.append(out)
    assert len(rows) == 1392 and len(writes) == 96
    summaries = aggregate(rows, cohort['conditions']); contrast_rows, _ = contrasts(rows)
    write(PUBLIC / 'ROWS.json', rows); write(PUBLIC / 'WRITE_ROWS.json', writes)
    write(PUBLIC / 'SUMMARY.json', summaries); write(PUBLIC / 'CONTRAST_ROWS.json', contrast_rows)
    write(PUBLIC / 'CASES.json', {branch: {role: dict(
        positive=sorted(role_rows([r for r in rows if r['condition'] != 'clean'], role), key=lambda r: -r['delta_'+branch+'_v'])[:5],
        negative=sorted(role_rows([r for r in rows if r['condition'] != 'clean'], role), key=lambda r: r['delta_'+branch+'_v'])[:5])
        for role in ['self', 'next', 'near', 'far', 'broader']} for branch in BRANCHES})
    resources = dict(new_experts=0, new_backbone_forwards=0)
    rr = [read(BASE / (mode+'_RESOURCES.json')) for mode in ['SMOKE', 'FULL']]
    for field in ['suffix_replays', 'backward_calls', 'new_writes', 'capture_cache_reads', 'expert_cache_reads', 'worker_wall_seconds']:
        resources[field] = sum(r[field] for r in rr)
    resources.update(logical_state_predictions=1392*4, peak_allocated_vram_bytes=max(r['peak_allocated_vram_bytes'] for r in rr),
                     checkpoint_restored=all(r['checkpoint_restored'] for r in rr), wall_includes_loading_IO=True)
    write(PUBLIC / 'RESOURCES.json', resources); write(PUBLIC / 'CONFIG.json', config)
    write(PUBLIC / 'SMOKE_ROOT_ACCEPTANCE.json', read(BASE / 'SMOKE_ROOT_ACCEPTANCE.json'))
    audit = dict(status='pass', checks=dict(checks), max_metric_error=max_error,
        max_arithmetic_errors=dict(arithmetic_errors), donors=96, targets=1392,
        after_prediction_barrier=True, historically_exposed=True,
        CUDA_initialized=torch.cuda.is_initialized(), time=time.time())
    assert not audit['CUDA_initialized']
    write(BASE / 'ROOT_READBACK.json', audit); write(PUBLIC / 'ROOT_READBACK.json', audit)
    status(BASE / 'STATUS.json', dict(status='scored_pending_publication', done=96, total=96, time=time.time()))
    print('ROOT audit pass', len(rows), 'target cells', dict(checks))


if __name__ == '__main__':
    run()
