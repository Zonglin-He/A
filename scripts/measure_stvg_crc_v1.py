"""Independent saved arithmetic readback, then separate post-seal official scores."""
import math
import os
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.stvg_crc_common_v1 import *


def iou(a, b):
    import numpy as np
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    alo, ahi = a[..., :2]-a[..., 2:]/2, a[..., :2]+a[..., 2:]/2
    blo, bhi = b[..., :2]-b[..., 2:]/2, b[..., :2]+b[..., 2:]/2
    intersection = np.maximum(0., np.minimum(ahi, bhi)-np.maximum(alo, blo)).prod(-1)
    union = np.maximum(a[..., 2:], 0).prod(-1)+np.maximum(b[..., 2:], 0).prod(-1)-intersection
    return intersection/np.maximum(union, 1e-12)


def root_readback():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    import numpy as np
    from scripts import stvg_target_teacher_student_v1 as core
    verify_runtime()
    sys.addaudithook(guard)
    bar = read(BASE / 'STUDENT_GLOBAL_BARRIER.json')
    assert bar['status'] == 'complete' and bar['count'] == 32 and not bar['GT_read']
    expected_paths = {str((BASE/'student'/(r['probe_alias']+'.pt')).relative_to(ROOT)) for r in roster()}
    assert len(bar['files']) == 32 and {r['path'] for r in bar['files']} == expected_paths
    refs = {r['case_id']: r for r in read(BASE / 'REFERENCE_LOCK.json')['records']}
    cases = {r['probe_alias']: r for r in roster()}
    max_error = 0.
    max_q_error = 0.
    source_pins = []
    seen_cases = set()
    checks = fit_calls = accepted_steps = 0
    totals = {v: dict(cases_with_reference=0, fit_invocations=0,
        accepted_steps=0, no_update_cases=0) for v in ['Pair']}
    for item in bar['files']:
        path = ROOT / item['path']
        assert sha(path) == item['sha256']
        d = torch.load(path, map_location='cpu', weights_only=False)
        cid = d['case_id']
        assert cid not in seen_cases and path == BASE/'student'/(cid+'.pt')
        seen_cases.add(cid)
        case, source_path = load_case(cases[cid])
        source_pins.extend(dict(path=str(p.relative_to(ROOT)), sha256=sha(p)) for p in
                           [source_path, source_path.with_suffix('.receipt.json')])
        assert sha(source_path) == d['source_file_sha256']
        assert d['source_case_sha256'] == case['source_case_sha256']
        assert not d['GT_read'] and d['source_restored']
        assert np.array_equal(d['arms']['F']['boxes'], native_arm(case)['boxes'])
        parent_path=PARENT/'student'/(cid+'.pt')
        parent_bar=read(PARENT/'STUDENT_GLOBAL_BARRIER.json')
        expected_parent=next(x for x in parent_bar['files'] if x['path']==str(parent_path.relative_to(ROOT)))
        assert sha(parent_path)==expected_parent['sha256']
        parent=torch.load(parent_path,map_location='cpu',weights_only=False)
        for arm,old_arm in [('S_historical','S_new_crop'),('TS_historical','TS_new_crop')]:
            assert np.array_equal(d['arms'][arm]['boxes'],parent['arms'][old_arm]['boxes'])
            assert d['arms'][arm]['interval']==parent['arms'][old_arm]['interval']
        source_pins.append(dict(path=str(parent_path.relative_to(ROOT)),sha256=sha(parent_path)))
        assert np.array_equal(d['arms']['F']['boxes'], d['arms']['T']['boxes'])
        interval = d['arms']['T']['interval']
        for v in ['Pair']:
            aa = refs[cid][v]['anchors']
            assert d['reference_records'][v]['anchors'] == aa
            assert np.array_equal(d['arms']['S_'+v]['boxes'], d['arms']['TS_'+v]['boxes'])
            assert d['arms']['TS_'+v]['interval'] == interval
            totals[v]['cases_with_reference'] += bool(aa)
            totals[v]['fit_invocations'] += d['receipts'][v]['actual_fit_count']
            fitted = d['fits'].get(v)
            if fitted is None:
                assert not aa and np.array_equal(d['arms']['S_'+v]['boxes'], d['arms']['F']['boxes'])
                totals[v]['no_update_cases'] += 1
                continue
            assert fitted['active_parameters'] == 1792
            assert fitted['config'] == read(BASE/'DESIGN_LOCK.json')['spatial_configs'][d['dataset']]
            assert torch.equal(d['readouts'][v]['boxes'].cpu(), fitted['final_cached_decoder_boxes'].cpu())
            totals[v]['accepted_steps'] += fitted['optimizer_steps']
            totals[v]['no_update_cases'] += fitted['optimizer_steps'] == 0
            fit_calls += d['receipts'][v]['actual_fit_count']
            accepted_steps += fitted['optimizer_steps'] * d['receipts'][v]['actual_fit_count']
            if fitted['optimizer_steps'] == 0:
                assert core._tree_equal(fitted['initial_state'], fitted['final_state'])
                assert np.array_equal(d['arms']['S_'+v]['boxes'], d['arms']['F']['boxes'])
            positions = [a['position'] for a in aa]
            evidence = np.array([a['box_cxcywh'] for a in aa])
            weights = np.array([a['weight'] for a in aa])
            for row in fitted['path']:
                if 'rollout' not in row:
                    assert row.get('skipped')
                    continue
                policy = row['rollout']
                samples = policy['samples'].numpy().astype(np.float64)
                sampled_boxes = 1/(1+np.exp(-samples))
                computed = iou(sampled_boxes, evidence[:, None, :])
                stored = policy['rewards'].numpy()
                error = float(abs(computed-stored).max())
                max_error = max(max_error, error)
                assert error < 5e-6
                checks += int(computed.size)
                logits=computed/fitted['config']['tau']
                logits-=logits.max(axis=1,keepdims=True)
                q=np.exp(logits);q/=q.sum(axis=1,keepdims=True)
                q_error=float(abs(q-policy['weights'].numpy()).max())
                max_q_error=max(max_q_error,q_error)
                assert q_error<5e-6
                assert policy['samples'].shape[1]==32
                before = iou(row['boxes'].numpy()[positions], evidence)
                if row.get('proposed', False):
                    after = iou(row['proposed_boxes'].numpy()[positions], evidence)
                    assert float((weights*(after-before)).sum()/weights.sum()) >= -2e-6 or not row['accepted']
                if row.get('proposed') and not row['accepted']:
                    assert core._tree_equal(row['optimizer_before'], row['optimizer_after'])
                    restored = row.get('update_state', row.get('state_after'))
                    assert core._tree_equal(row['state'], restored)
            boxes = fitted['final_cached_decoder_boxes'].numpy()
            xyxy = np.concatenate([boxes[:, :2]-boxes[:, 2:]/2, boxes[:, :2]+boxes[:, 2:]/2], axis=-1)
            assert np.allclose(xyxy, d['arms']['S_'+v]['boxes'], atol=1e-7, rtol=0)
    assert seen_cases == set(cases)
    readback = dict(status='pass', case_count=32, fit_invocations=fit_calls,
        accepted_optimizer_steps=accepted_steps, independent_reward_scalar_checks=checks,
        max_reward_iou_error=max_error, max_softmax_weight_error=max_q_error, per_variant=totals, GT_read=False, CPU_only=True,
        per_variant_step_scope='arm-level outcomes include exact reuse; aggregate accepted_optimizer_steps counts actual invocations only',
        scope='independent sampled rewards, reference acceptance, saved state/Adam rollback and actual full-vs-cached decoder; no independent backward replay')
    write(BASE / 'ROOT_STUDENT_READBACK.json', readback)
    pins = list(bar['files']) + source_pins
    for p in ['DESIGN_LOCK.json', 'RUNTIME.json', 'ROSTER_LOCK.json', 'REFERENCE_LOCK.json',
              'TEACHER_GLOBAL_BARRIER.json', 'STUDENT_GLOBAL_BARRIER.json', 'inputs_teacher/INPUT_BARRIER.json',
              'inputs_teacher/TEACHER_JOB_LOCK.json', 'ROOT_STUDENT_READBACK.json']:
        pins.append(dict(path=str((BASE/p).relative_to(ROOT)), sha256=sha(BASE/p)))
    write(BASE/'GLOBAL_PREDICTION_BARRIER.json', dict(status='complete', count=32,
        files=pins, scorer_sha256=sha(__file__), GT_read=False, time=time.time()))
    print('root', readback, flush=True)


def score():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    from scripts import review_stvg_sparse_r1_v2 as scorer
    from scripts import score_stvg_native_instance_nativeclock005 as metric
    verify_runtime()
    seal = read(BASE/'GLOBAL_PREDICTION_BARRIER.json')
    assert seal['status'] == 'complete' and seal['count'] == 32 and not seal['GT_read']
    assert seal['scorer_sha256'] == sha(__file__)
    for p in seal['files']:
        assert sha(ROOT/p['path']) == p['sha256']
    assert read(BASE/'ROOT_STUDENT_READBACK.json')['status'] == 'pass'
    bar = read(BASE/'STUDENT_GLOBAL_BARRIER.json')
    expected_paths = {str((BASE/'student'/(r['probe_alias']+'.pt')).relative_to(ROOT)) for r in roster()}
    assert bar['count'] == 32 and len(bar['files']) == 32
    assert {r['path'] for r in bar['files']} == expected_paths
    assert expected_paths <= {r['path'] for r in seal['files']}
    assert str((BASE/'STUDENT_GLOBAL_BARRIER.json').relative_to(ROOT)) in {r['path'] for r in seal['files']}
    cases, outputs = {}, {}
    for rec in roster():
        case, source_path = load_case(rec)
        cid = rec['probe_alias']
        cases[cid] = case
        outputs[cid] = torch.load(BASE/'student'/(cid+'.pt'), map_location='cpu', weights_only=False)
        assert outputs[cid]['case_id'] == cid and not outputs[cid]['GT_read']
        assert case['source_case_sha256'] == outputs[cid]['source_case_sha256']
        assert sha(source_path) == outputs[cid]['source_file_sha256']
        assert str(source_path.relative_to(ROOT)) in {r['path'] for r in seal['files']}
    write(BASE/'GT_EXPOSURE.json', dict(status='after_all_predictions_sealed',
        seal_sha256=sha(BASE/'GLOBAL_PREDICTION_BARRIER.json'), GT_read=True, time=time.time()))
    truth, provenance = scorer._truths_after_seal(cases)
    official, independent, _ = scorer._metric_functions()
    rows = []
    for cid, d in outputs.items():
        support = {a['position'] for v in ['Pair'] for a in d['reference_records'][v]['anchors']}
        for arm, pred in d['arms'].items():
            values, audit = metric._score_student_arm(pred, cases[cid], truth[cid], support, official, independent)
            rows.append(dict(case_id=cid, dataset=d['dataset'], arm=arm, metrics=values,
                             support=audit['support'], metric_audit=audit['metric_audit']))
    summary, paired = [], []
    for ds in ['hc2', 'vidstg']:
        rr = [r for r in rows if r['dataset'] == ds]
        ids = sorted({r['case_id'] for r in rr})
        assert len(ids) == 16
        lookup = {(r['case_id'], r['arm']): r for r in rr}
        names = sorted({r['arm'] for r in rr})
        for arm in names:
            for key in rr[0]['metrics']:
                values = [lookup[c, arm]['metrics'][key] for c in ids
                          if lookup[c, arm]['metrics'][key] is not None]
                summary.append(dict(dataset=ds, arm=arm, metric=key, count=len(values),
                    mean=math.fsum(values)/len(values) if values else None))
        comparisons = [(a, 'F') for a in names if a != 'F']
        comparisons += [('TS_'+v, 'T') for v in ['Pair']]
        comparisons += [('TS_Pair', 'TS_historical'), ('S_Pair', 'S_historical')]
        for left, right in comparisons:
            delta = {c: (lookup[c,left]['metrics']['m_vIoU']-lookup[c,right]['metrics']['m_vIoU'])*100 for c in ids}
            vv = list(delta.values())
            paired.append(dict(dataset=ds, left=left, right=right, metric='m_vIoU',
                mean_delta_pp=math.fsum(vv)/16, positive_sources=sum(v>1e-10 for v in vv),
                negative_sources=sum(v< -1e-10 for v in vv), neutral_sources=sum(abs(v)<=1e-10 for v in vv),
                worst_delta_pp=min(vv), best_delta_pp=max(vv), per_source_delta_pp=delta))
    reference_diagnostics(cases, truth)
    write(BASE/'POSTSEAL_ROWS.json', dict(status='complete', rows=rows, GT_read=True))
    write(BASE/'POSTSEAL_SUMMARY.json', dict(status='complete', summary=summary, GT_read=True,
        development_selection=True, independent_confirmation=False))
    write(BASE/'PARENT_PAIRED_DELTAS.json', dict(status='complete', records=paired,
        GT_read=True, bootstrap_executed=False))
    write(BASE/'STATUS.json', dict(status='measurement_complete_pending_review_publication',
        case_count=32, GT_read=True, full_experiment_started=False, time=time.time()), mutable=True)
    print('postseal vIoU', [r for r in summary if r['metric']=='m_vIoU'], flush=True)


def reference_diagnostics(cases, truth):
    import numpy as np
    from scripts.score_stvg_native_instance_nativeclock005 import _truth_map
    parent = {r['case_id']:r for r in read(PARENT/'REFERENCE_LOCK.json')['records']}
    ref = {r['case_id']: r for r in read(BASE/'REFERENCE_LOCK.json')['records']}
    rows = []
    def overlap(a, b):
        a, b = np.array(a,dtype=float), np.array(b,dtype=float)
        inter=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod()
        return float(inter/max(1e-12,np.maximum(a[2:]-a[:2],0).prod()+np.maximum(b[2:]-b[:2],0).prod()-inter))
    for cid, case in cases.items():
        aa = parent[cid]['new_crop']['anchors']
        kept = {(a['position'], a['candidate_id']) for a in ref[cid]['Pair']['anchors']}
        gt = _truth_map(truth[cid]); width,height=case['input']['width'],case['input']['height']
        for a in aa:
            fid=a['physical_frame_id']; row=dict(case_id=cid,dataset=case['dataset'],center_position=a['position'],retained=(a['position'],a['candidate_id']) in kept,gt_observed=fid in gt,reference_delta_iou_pp=None)
            if fid in gt:
                box=np.asarray(a['box_cxcywh'],float); xy=np.r_[box[:2]-box[2:]/2,box[:2]+box[2:]/2]*[width,height,width,height]
                source=np.array(case['native_boxes_xyxy_normalized'][a['position']])*[width,height,width,height]
                row['reference_delta_iou_pp']=100*(overlap(xy,gt[fid])-overlap(source,gt[fid]))
            rows.append(row)
    assert len(rows)==49
    summary=[]
    for ds in ['hc2','vidstg']:
        for retained in [False,True]:
            rr=[r for r in rows if r['dataset']==ds and r['retained']==retained]
            values=[r['reference_delta_iou_pp'] for r in rr if r['gt_observed']]
            summary.append(dict(dataset=ds,retained=retained,count=len(rr),observed_count=len(values),unknown_count=len(rr)-len(values),positive=sum(v>1e-8 for v in values),negative=sum(v< -1e-8 for v in values),neutral=sum(abs(v)<=1e-8 for v in values),mean_delta_pp=math.fsum(values)/len(values) if values else None))
    write(BASE/'POSTSEAL_REFERENCE_DIAGNOSTICS.json',dict(status='complete',rows=rows,summary=summary,GT_read=True,scope='same common observed anchor locations only; not full-tube vIoU or reference construction'))


if __name__ == '__main__':
    {'root': root_readback, 'score': score}[sys.argv[1]]()
