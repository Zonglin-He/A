"""Post-seal fixed256 evaluation. Inert imports; contracts never load GT."""
from pathlib import Path
import gc, hashlib, json, math, sys, time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_stvg_spatial_controls256_v1 import BASE, read, put, sha
from scripts.run_stvg_spatial_controls256_student_v1 import ARM_FROM, check

def aggregate(rows):
    summaries, pairs = [], []
    for ds in ('hc2', 'vidstg'):
        rr = [r for r in rows if r['dataset'] == ds]
        ids = sorted({r['case_id'] for r in rr})
        arms = list(dict.fromkeys(r['arm'] for r in rr))
        idx = {(r['case_id'], r['arm']): r['metrics'] for r in rr}
        assert len(ids) == 128 and len(idx) == len(rr) == 128 * len(arms)
        metrics = tuple(idx[ids[0], arms[0]])
        assert all(tuple(v) == metrics for v in idx.values())
        assert all(v is None or math.isfinite(v) for m in idx.values() for v in m.values())
        for arm in arms:
            for metric in metrics:
                v = [idx[c, arm][metric] for c in ids if idx[c, arm][metric] is not None]
                summaries.append(dict(dataset=ds, arm=arm, metric=metric, valid=len(v),
                    unknown=128-len(v), mean=math.fsum(v)/len(v) if v else None))
        for arm in arms:
            for ref in arms:
                if arm == ref:
                    continue
                for metric in metrics:
                    v = {c: 100*(idx[c, arm][metric]-idx[c, ref][metric]) for c in ids
                         if idx[c, arm][metric] is not None and idx[c, ref][metric] is not None}
                    pairs.append(dict(dataset=ds, comparison=arm+'_minus_'+ref, metric=metric,
                        valid=len(v), unknown=128-len(v), mean_delta_pp=math.fsum(v.values())/len(v) if v else None,
                        positive=sum(x > 1e-6 for x in v.values()), negative=sum(x < -1e-6 for x in v.values()),
                        neutral=sum(abs(x) <= 1e-6 for x in v.values()), per_source_delta_pp=v,
                        worst_sources=sorted(v, key=lambda c: (v[c], c))[:8]))
    return summaries, pairs

def validate_barrier(arm):
    check(arm)
    b = BASE/'trials'/arm
    root = read(b/'ROOT_READBACK.json')
    bar = read(b/'GLOBAL_PREDICTION_BARRIER.json')
    student = read(b/'STUDENT_GLOBAL_BARRIER.json')
    assert root['status'] == 'pass' and root['count'] == 256 and root['GT_read'] is False
    assert bar['status'] == student['status'] == 'complete'
    assert bar['count'] == student['count'] == 256 and not bar['GT_read'] and not student['GT_read']
    assert bar['root_sha256'] == sha(b/'ROOT_READBACK.json')
    assert root['barrier_sha256'] == sha(b/'STUDENT_GLOBAL_BARRIER.json')
    assert bar['files'] == student['files']
    ids = {r['probe_alias'] for r in read(BASE/'ROSTER_LOCK.json')['records']}
    assert len(ids) == 256
    files = {f['case_id']: f for f in bar['files']}
    assert len(files) == len(bar['files']) == 256 and set(files) == ids
    for f in files.values():
        assert sha(ROOT/f['path']) == f['sha256']
        assert sha(ROOT/f['receipt_path']) == f['receipt_sha256']
    return files

def main(arm):
    assert arm in ARM_FROM
    lock = read(BASE/'EVALUATION_LOCK.json')
    for p, h in lock['pins'].items():
        assert sha(ROOT/p) == h, p
    start = time.perf_counter()
    maps = {a: validate_barrier(a) for a in dict.fromkeys(('Full_E1', arm))}
    tr = read(BASE/'temporal/ROOT_READBACK.json')
    assert tr['status'] == 'pass' and tr['count'] == 256 and not tr['GT_read']
    # GT access is below all complete256 ROOT, file-digest and prediction checks.
    import torch, numpy as np
    from scripts.stvg_soft_ep_coordinate_common_v1 import load_case
    from scripts import review_stvg_sparse_r1_v2 as truth
    from scripts import score_stvg_native_instance_nativeclock005 as metric
    from scripts.score_stvg_B_transfer_v1 import geom
    roster = read(BASE/'ROSTER_LOCK.json')['records']
    assert len({(r['dataset'], r['parent_sha256']) for r in roster}) == 256
    cases = {r['probe_alias']: load_case(r)[0] for r in roster}
    targets, bindings = truth._truths_after_seal(cases)
    official, independent, _ = truth._metric_functions()
    refs = {r['case_id']: r for r in read(BASE/'trials'/arm/'REFERENCE_LOCK.json')['records']}
    rows, support_rows, usage, geometry, audits = [], [], [], [], []
    for rec in roster:
        cid, ds = rec['probe_alias'], rec['dataset']
        case = cases[cid]
        d = {a: torch.load(ROOT/maps[a][cid]['path'], map_location='cpu', weights_only=False) for a in maps}
        current, full = d[arm], d['Full_E1']
        assert current['temporal_record'] == full['temporal_record']
        poolpath = BASE/'spatial/E1/proposals'/f'{cid}.json'
        pool = read(poolpath) if poolpath.exists() else dict(windows=[])
        common = [w['source_grid_position'] for w in pool['windows']]
        adopted = [a['position'] for a in refs[cid]['anchors']]
        preds = {'Frozen': current['arms']['F'], 'Temporal': current['arms']['T'],
                 'Full_E1': full['arms']['TS_M2'], 'Full_E1_Spatial': full['arms']['S_M2'],
                 arm: current['arms']['TS_M2'], arm+'_Spatial': current['arms']['S_M2']}
        for name, pred in preds.items():
            s, audit = metric._score_student_arm(pred, case, targets[cid], common, official, independent)
            if name in ('Frozen', 'Temporal'):
                other, _ = metric._score_student_arm(full['arms']['F' if name == 'Frozen' else 'T'],
                    case, targets[cid], common, official, independent)
                assert s == other, (cid, name)
            rows.append(dict(case_id=cid, dataset=ds, arm=name, metrics=s, GT_read=True))
            audits.append(dict(case_id=cid, arm=name, common=audit))
            s, a = metric._score_student_arm(pred, case, targets[cid], adopted, official, independent)
            support_rows.append(dict(case_id=cid, dataset=ds, arm=name, metrics=s,
                adopted_anchor_count=len(adopted), known_supervised=a['support']['supervised_gt_known_frame_count'],
                known_unsupervised=a['support']['unsupervised_gt_known_frame_count'], GT_read=True))
        fit = current['fits']['M2']
        steps = []
        for r in fit['path'] if fit else []:
            before = r.get('reference_iou_before')
            candidates = [dict(scale=z['scale'], finite=z['finite'], reference_pass=z['reference_pass'],
                accepted=z['accepted'], agreement_gain=float(z['anchor_IoU'].double().mean()-before.double().mean()))
                for z in r.get('candidates', [])]
            steps.append(dict(step=r['step'], accepted=r['accepted'], accepted_scale=r['accepted_scale'],
                skipped=r['skipped'], reason=r.get('reason'), rollback=r['rollback'], candidates=candidates))
        usage.append(dict(case_id=cid, dataset=ds, anchors=len(adopted), actual_fit=fit is not None,
            accepted_steps=fit['optimizer_steps'] if fit else 0,
            nonzero_final=any(not torch.equal(v.cpu(), fit['final_state'][k].cpu())
                for k, v in fit['initial_state'].items()) if fit else False,
            fit_seconds=fit.get('instance_study_fit_seconds') if fit else None,
            decisions=refs[cid]['decisions'], steps=steps, GT_read=False))
        target = metric._truth_map(targets[cid])
        scale = np.array([case['input']['width'], case['input']['height']]*2)
        anchors = {a['position']: a for a in refs[cid]['anchors']}
        for w in pool['windows']:
            fid = w['center_physical_frame_id']
            g = np.asarray(target[fid])/scale if fid in target else None
            adopted_anchor = anchors.get(w['source_grid_position'])
            geometry.append(dict(case_id=cid, dataset=ds, slot=w['slot'], known_GT=g is not None,
                candidates=[dict(candidate_id=t['candidate_id'], presence=t['presence'],
                    diagnostic=geom(t['center_box'], g) if g is not None and t['center_box'] is not None else None)
                    for t in w['tracks']],
                adopted_candidate_id=adopted_anchor['candidate_id'] if adopted_anchor else None,
                adopted_geometry=geom(adopted_anchor['box_cxcywh'], g) if adopted_anchor and g is not None else None,
                GT_read=True))
        del d, current, full
        gc.collect()
    summaries, pairs = aggregate(rows)
    sa, pa = aggregate(support_rows)
    result = dict(status='complete', count=256, per_domain=128, arm=arm, rows=rows,
        summary=summaries, paired=pairs, actual_support_rows=support_rows, actual_support_summary=sa,
        actual_support_paired=pa, usage=usage, geometry=geometry, metric_audits=audits, GT_bindings=bindings,
        semantic_status='separate fixed512-window arm-blind audit; no numeric identity claim from these scores',
        GT_read=True, bootstrap=False, evaluation_lock_sha256=sha(BASE/'EVALUATION_LOCK.json'),
        prediction_sha256=sha(BASE/'trials'/arm/'GLOBAL_PREDICTION_BARRIER.json'),
        support='same prospective E1 positions for every comparison; actual adopted support reported separately; '
                'actual-support values across recipes can use different supervised sets',
        geometry_limit='postseal diagnostic, fixed extent is not certified full-object correction')
    b = BASE/'trials'/arm
    put(b/'POSTSEAL_RESULTS.json', result)
    put(b/'CPU_COMPLETION.json', dict(status='complete', count=256, row_count=len(rows), GT_read=True,
        result_sha256=sha(b/'POSTSEAL_RESULTS.json'), wall_seconds=time.perf_counter()-start))
    print(json.dumps(dict(status='complete', count=256, arm=arm, row_count=len(rows))))

def lock():
    roster = read(BASE/'ROSTER_LOCK.json')
    assert roster['count'] == 256
    assert not (BASE/'EVALUATION_LOCK.json').exists()
    files = [Path(__file__), BASE/'ROSTER_LOCK.json', BASE/'DESIGN_LOCK.json',
             ROOT/'scripts/score_stvg_native_instance_nativeclock005.py',
             ROOT/'scripts/review_stvg_sparse_r1_v2.py', ROOT/'scripts/score_stvg_B_transfer_v1.py']
    put(BASE/'EVALUATION_LOCK.json', dict(status='locked', count=256,
        pins={str(p.relative_to(ROOT)): sha(p) for p in files}, GT_read=False,
        before_any_new_group_GT=True, no_bootstrap=True, no_configuration_selection=True))
    panel = [dict(case_id=r['probe_alias'], dataset=r['dataset'], parent_sha256=r['parent_sha256'], slot=s)
             for r in roster['records'] for s in (0, 3)]
    put(BASE/'SEMANTIC_PANEL_LOCK.json', dict(status='locked', count=256, window_count=512, records=panel,
        selection='all fixed256 parents, slots0 and3, fixed before any new WHO choices; no score filtering',
        annotation_status='pending_actual_arm_blind_pixel_review', GT_read=False,
        protocol='label all candidate IDs from original query and unselected videos/panels; hide arm names, '
                 'chosen IDs, Teacher replies, GT and Student outcomes until labels are locked',
        fields=['identity_known', 'target_full', 'target_partial', 'other_entity', 'mixed', 'unknown',
                'role_supported', 'role_contradicted', 'role_unknown'],
        missing_policy='missing window remains unknown, never replace it with a more favorable time',
        limitation='project-exposed visual review, not an independent human benchmark or latent reasoning proof'))
    print('EVALUATION_AND_SEMANTIC_PANEL_LOCKED_256_512')

def contracts():
    rows = [dict(case_id=ds+str(i), dataset=ds, arm=a,
        metrics={'m_vIoU': i/256+(0.001 if a == 'new' else 0), 'sparse': None if i%2 else .2})
        for ds in ('hc2', 'vidstg') for i in range(128) for a in ('old', 'new')]
    s, p = aggregate(rows)
    assert all(r['valid'] == 64 and r['unknown'] == 64 for r in s if r['metric'] == 'sparse')
    assert all(abs(r['mean_delta_pp']-.1) < 1e-10 and r['positive'] == 128
               for r in p if r['comparison'] == 'new_minus_old' and r['metric'] == 'm_vIoU')
    for bad in (rows[:-1], rows+[rows[0]], [{**r, 'metrics': {**r['metrics'], 'm_vIoU': float('nan')}} for r in rows]):
        try:
            aggregate(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError('invalid cohort/duplicate/nonfinite accepted')
    # Guard must reject the actual unsealed cohort before touching any annotation loader.
    try:
        validate_barrier('Full_E1')
    except (AssertionError, FileNotFoundError):
        pass
    else:
        raise AssertionError('preseal cohort unexpectedly qualified')
    print('CONTRACTS_PASS_128_PER_DOMAIN_NULL_DUPLICATE_MISSING_NONFINITE_PRESEAL_NO_GT')

if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'contracts':
        contracts()
    elif mode == 'lock':
        lock()
    else:
        main(mode)
