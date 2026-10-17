"""Matched fixed-support Student32; GT only in explicit post-seal score."""
from pathlib import Path
import ast, copy, gc, inspect, json, math, sys, time, types
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.stvg_native_instance_common_v1 import read, write, sha, guard
BASE = ROOT / 'artifacts/stvg_semantic_preference_student32_v1'
PREF = ROOT / 'artifacts/stvg_semantic_preference_r1_32_v1'
OLD = ROOT / 'artifacts/stvg_qwen35_geometry32_v1'

def verify():
    r = read(BASE / 'EXECUTION_RUNTIME.json')
    for p, h in r['pins'].items():
        assert sha(ROOT / p) == h, p
    if (BASE / 'ACTIVE_EXECUTION.json').exists():
        assert time.time() < read(BASE / 'ACTIVE_EXECUTION.json')['deadline_epoch'], 'bounded Student budget expired'
    return r

def setup():
    assert not (BASE / 'SUPPORT_LOCK.json').exists()
    assert read(PREF / 'R1_ROOT_READBACK.json')['status'] == 'pass'
    BASE.mkdir(exist_ok=True)
    probes = read(PREF / 'QUALIFICATION_PLAN.json')['probe_jobs']
    jobs = {j['job_id']: j for j in read(PREF / 'INPUT_JOB_LOCK.json')['records']}
    rr = read(PREF / 'ROSTER_LOCK.json')
    picked = [r for r in rr['records'] if any((jobs[j]['case_id'] == r['probe_alias'] for j in probes))]
    assert len(picked) == 32
    roster = dict(rr, count=32, records=picked, selection='same prelocked R1 interface8; no selection by preference or GT')
    source = read(OLD / 'SOURCE_CAPTURE_BARRIER.json')
    ids = {r['probe_alias'] for r in picked}
    source = dict(source, count=32, files=[f for f in source['files'] if f['case_id'] in ids])
    temp = read(OLD / 'TEMPORAL_LOCK.json')
    temp = dict(temp, count=32, records=[r for r in temp['records'] if r['case_id'] in ids])
    support = []
    refs = []
    for rec in picked:
        j = next((jobs[x] for x in probes if jobs[x]['case_id'] == rec['probe_alias']))
        f = PREF / 'teachers/R1' / (j['job_id'] + '.json')
        d = read(f)
        p = d['permutation_averaged_preference']
        boxes = []
        for c in j['candidates']:
            x, y, u, v = c['box_xyxy']
            boxes.append([(x + u) / 2, (y + v) / 2, u - x, v - y])
        reason = 'admitted'
        chosen = max(range(len(p)), key=p.__getitem__)
        if chosen == 0:
            reason = 'Teacher_abstention'
        elif not all((all((0 < v < 1 and math.isfinite(v) for v in b)) for b in boxes)):
            reason = 'fixed_candidate_not_in_open_logit_chart_whole_Query_noop'
        from scripts.stvg_soft_ep_coordinate_common_v1 import load_case
        case,_=load_case(rec)
        native=case['zero']['boxes'][j['position']]
        if reason=='admitted' and not all(0<float(v)<1 and math.isfinite(float(v)) for v in native):reason='Source_anchor_not_in_open_logit_chart_whole_Query_noop'
        del case
        anchors = []
        if reason == 'admitted':
            anchors = [dict(position=j['position'], physical_frame_id=j['physical_frame_id'], box_cxcywh=boxes[chosen - 1], weight=1.0, candidate_id='semantic_preference_argmax', teacher_output_sha256=sha(f), GT_read=False)]
        refs.append(dict(case_id=rec['probe_alias'], dataset=rec['dataset'], source_case_sha256=j['source_case_sha256'], anchors=anchors, admission_reason=reason, GT_read=False))
        support.append(dict(case_id=rec['probe_alias'], dataset=rec['dataset'], job_id=j['job_id'], position=j['position'], boxes_cxcywh=boxes, posterior=p, common_reference=anchors, admission_reason=reason, Teacher_sha256=sha(f), GT_read=False))
    write(BASE / 'ROSTER_LOCK.json', roster)
    write(BASE / 'SOURCE_CAPTURE_BARRIER.json', source)
    write(BASE / 'TEMPORAL_LOCK.json', temp)
    write(BASE / 'SUPPORT_LOCK.json', dict(status='locked', count=32, records=support, GT_read=False))
    write(BASE / 'D1_REFERENCE_LOCK.json', dict(status='locked', count=32, records=[dict(case_id=r['case_id'], anchors=r['anchors']) for r in refs], GT_read=False))
    write(BASE / 'INPUT_JOB_LOCK.json', dict(status='locked', count=32, records=[jobs[x] for x in probes], GT_read=False))
    oldbase = read(OLD / 'BASELINE_BARRIER.json')
    write(BASE / 'BASELINE_BARRIER.json', dict(oldbase, count=32, files=[f for f in oldbase['files'] if f['case_id'] in ids]))
    for arm in ['GEOM', 'PREF']:
        a = BASE / 'arms' / arm
        a.mkdir(parents=True)
        write(a / 'ROSTER_LOCK.json', roster)
        write(a / 'SOURCE_CAPTURE_BARRIER.json', source)
        write(a / 'SCOPE_CONFIG.json', dict(recipe='lite_fullstep', label='fixed_support_R1_' + arm, GT_read=False))
        write(a / 'teacher/G32/ROOT_READBACK.json', dict(status='pass', count=32, GT_read=False))
        t = a / 'trials/G32'
        write(t / 'REFERENCE_LOCK.json', dict(status='locked', count=32, records=refs, GT_read=False))
        write(t / 'TEMPORAL_LOCK.json', temp)
        d = copy.deepcopy(read(ROOT / 'artifacts/stvg_paper_t2_b_v1/arms/B/trials/G32/DESIGN_LOCK.json'))
        d.update(count=32, trial_id='G32', scope='fixed_support ' + arm, cohort='exposed interface8, slot0', family_reuse_from={ds: {m: None for m in v} for ds, v in d['configs'].items()})
        for ds in d['configs']:
            for cfg in d['configs'][ds].values():
                cfg['samples'] = 4
        write(t / 'DESIGN_LOCK.json', d)
    print('SUPPORT_LOCKED32', [(r['case_id'], r['admission_reason']) for r in support])

def policy(mean, evidence, key, step, *, config, support, arm):
    import torch
    from vg_tta.tastvg_decota_critic_p0_v1 import aligned_iou
    boxes = mean.new_tensor(support['boxes_cxcywh'])
    assert boxes.shape == (4, 4) and ((boxes > 0) & (boxes < 1)).all()
    samples = torch.logit(boxes)[None].detach()
    p = mean.new_tensor(support['posterior'])
    r = aligned_iou(boxes[None], evidence[:, None, :]).detach()
    weights = (p[1:][None] if arm == 'PREF' else (1 - p[0]) * torch.softmax(r / config['tau'], -1)).detach()
    return dict(mean=mean.detach().clone(), samples=samples, rewards=r, used_rewards=(p[1:][None] if arm == 'PREF' else r).detach(), weights=weights, permutations=None, informative=bool((weights != weights[:, :1]).any()), seed=None, sample_origin='fixed_existing_candidate_atoms_not_Gaussian_draws', feedback_arm=arm)

def worker(arm):
    import torch
    from scripts import stvg_overnight_student_v1 as s, stvg_spatial_lite_fit_v1 as lite, stvg_pipeline_speed_v1 as speed
    from scripts.resume_stvg_r1_geometry32_dino_v1 import anchors
    from scripts.run_stvg_table1_student004 import stable_fit_key
    anchors()
    records = read(BASE / 'ROSTER_LOCK.json')['records']
    supports = {r['case_id']: r for r in read(BASE / 'SUPPORT_LOCK.json')['records']}
    bykey = {stable_fit_key(r['dataset'], r['query_key_sha256']): supports[r['probe_alias']] for r in records}
    oldfit = lite.fit
    original_cfg = lite._validate_config

    def fit(base, initial, aa, ids, key, cfg, mode, settings, qualification=False, checkpoint=None, **kw):
        support = bykey[key]
        ns = dict(lite.__dict__)
        ns['_validate_config'] = lambda c, m: original_cfg(dict(c, samples=32), m)
        ns['rollout'] = lambda mean, evidence, key, step, config: policy(mean, evidence, key, step, config=config, support=support, arm=arm)
        exec(compile(inspect.getsource(oldfit), __file__, 'exec'), ns)
        f = ns['fit'](base, initial, aa, ids, key, cfg, mode, settings, qualification=qualification, checkpoint=checkpoint, **kw)
        f.update(policy_family='fixed_support_distillation', feedback_arm=arm, on_policy=False)
        return f
    s.BASE = BASE / 'arms' / arm
    s.CORE = ROOT / 'artifacts/stvg_paper_t2_b_v1'
    s.N = 32
    s.CFG = read(s.BASE / 'SCOPE_CONFIG.json')
    s.common = types.SimpleNamespace(BASE=BASE, verify=verify, read=read, write=write, sha=sha)
    import scripts.run_stvg_controls256_speed_student_v1 as source_module
    original_patch = s.source_patch

    def patch_source(text, enabled):
        z = original_patch(text, enabled)
        old = '        validated_configs[dataset] = {mode: _validate_config(configs[dataset][mode], dataset, mode) for mode in MODES}'
        return z
    source = inspect.getsource(s.worker)
    needle = 'w.__dict__.update(speed_engine=engine, speed_enqueue=enqueue, speed_flush=flush)'
    needle = 'w.__dict__.update(speed_engine=engine,speed_enqueue=enqueue,speed_flush=flush)'
    assert source.count(needle) == 1
    source = source.replace(needle, needle + ';oldcfg=w._validate_config;w._validate_config=lambda c,d,m:(oldcfg(dict(c,samples=32),d,m) and dict(c))')
    ns = dict(s.__dict__)
    ns['speed'] = types.SimpleNamespace(BASE=speed.BASE, Engine=lambda label, enabled: speed.Engine('semantic_pref32_' + arm, enabled))
    lite.fit = fit
    try:
        exec(compile(source, __file__, 'exec'), ns)
        ns['worker']()
    finally:
        lite.fit = oldfit

def rollout_audit(row, evidence, cfg, positions, key, step, support, arm):
    import numpy as np
    from scripts import audit_stvg_spatial_lite_fit002 as audit
    a = lambda x: np.asarray(audit._arr(x), dtype=float)
    roll = row['rollout']
    mean = a(roll['mean'])
    samples = a(roll['samples'])
    w = a(roll['weights'])
    boxes = np.asarray(support['boxes_cxcywh'], float)
    p = np.asarray(support['posterior'], float)
    expected_samples = np.log(boxes / (1 - boxes))[None]
    assert np.max(abs(samples - expected_samples)) < 3e-06
    reward = audit._iou(boxes[None], evidence[:, None, :])
    logits = reward / cfg['tau']
    q = np.exp(logits - logits.max(-1, keepdims=True))
    q = q / q.sum(-1, keepdims=True) * (1 - p[0])
    expected_q = p[1:][None] if arm == 'PREF' else q
    qe = float(np.max(abs(w - expected_q)))
    assert qe < 3e-06
    logp = -np.square(samples - mean[:, None, :]).sum(-1) / (2 * cfg['sigma'] ** 2) - 2 * math.log(2 * math.pi * cfg['sigma'] ** 2)
    per = -(w * logp).sum(-1)
    loss = float(per.mean())
    le = abs(loss - row['loss'])
    assert le < 0.001 + 3e-05 * abs(loss)
    info = bool(np.any(w != w[:, :1]))
    ge = 0.0
    if info:
        g = (w[:, :, None] * (mean[:, None, :] - samples)).sum(1) / (cfg['sigma'] ** 2 * len(positions))
        ge = float(np.max(abs(a(row['mean_gradient']) - g)))
        assert ge < 0.001 + 3e-05 * float(np.max(abs(g)))
    assert bool(roll['informative']) == info and roll['seed'] is None
    return dict(mean=mean, samples=samples, weights=w, evidence=evidence, expected_loss=loss, informative=info, loss_error=le, mean_gradient_error=ge, reward_error=float(np.max(abs(a(roll['rewards']) - reward))), q_error=qe)

def root(arm):
    from scripts import run_stvg_instance_student32_v1 as original, stvg_overnight_student_v1 as s, audit_stvg_spatial_lite_fit002 as audit
    from scripts.run_stvg_table1_student004 import stable_fit_key
    supports = {r['case_id']: r for r in read(BASE / 'SUPPORT_LOCK.json')['records']}
    rr = read(BASE / 'ROSTER_LOCK.json')['records']
    bykey = {stable_fit_key(r['dataset'], r['query_key_sha256']): supports[r['probe_alias']] for r in rr}

    def fit_audit(f, ref, recipe, response_positions=None):
        ns = dict(audit.__dict__)
        ns['_rollout_and_loss'] = lambda row, evidence, cfg, pos, key, step: rollout_audit(row, evidence, cfg, pos, key, step, bykey[key], arm)
        exec(compile(inspect.getsource(audit.fit_audit), __file__, 'exec'), ns)
        return ns['fit_audit'](f, ref, recipe, response_positions)
    source = inspect.getsource(original.root).replace('    from scripts.audit_stvg_spatial_lite_fit002 import fit_audit', '    # fixed-support audit supplied by this module')

    class Count(ast.NodeTransformer):

        def visit_Constant(self, n):
            return ast.copy_location(ast.Constant(32), n) if type(n.value) is int and n.value == 32 else n
    source = ast.unparse(ast.fix_missing_locations(Count().visit(ast.parse(source))))
    source = source.replace("bb = arr(o['boxes'])", "bb = arr(o['boxes']).astype(float)")
    ns = dict(original.__dict__)
    ns.update(BASE=BASE / 'arms' / arm, check=lambda tid: verify(), modes=lambda _: ('M2',), fit_audit=fit_audit)
    exec(compile(source, __file__, 'exec'), ns)
    ns['root']('G32')

def propagate():
    from scripts import stvg_r1_dino_complement32_v1 as old
    source = inspect.getsource(old.propagate)

    class Count(ast.NodeTransformer):

        def visit_Constant(self, n):
            return ast.copy_location(ast.Constant(32), n) if type(n.value) is int and n.value == 32 else n
    source = ast.unparse(ast.fix_missing_locations(Count().visit(ast.parse(source))))
    ns = dict(old.__dict__)
    ns.update(BASE=BASE, verify=verify)
    exec(compile(source, __file__, 'exec'), ns)
    ns['propagate']()

def score():
    import torch, numpy as np
    verify()
    bars = {a: read(BASE / 'arms' / a / 'trials/G32/GLOBAL_PREDICTION_BARRIER.json') for a in ['GEOM', 'PREF']}
    bars['D2'] = read(BASE / 'D2_GLOBAL_PREDICTION_BARRIER.json')
    for a, b in bars.items():
        assert b['status'] == 'complete' and b['count'] == 32
        for f in b['files']:
            assert sha(ROOT / f['path']) == f['sha256']
    write(BASE / 'EXPERIMENT_GLOBAL_PREDICTION_BARRIER.json', dict(status='complete', count=32, all_new_predictions_sealed=True, GT_read=False))
    from scripts import review_stvg_sparse_r1_v2 as truth, score_stvg_native_instance_nativeclock005 as metric
    from scripts.stvg_soft_ep_coordinate_common_v1 import load_case
    from scripts.score_stvg_temporal_t2_v1 import subset_aggregate
    rr = read(BASE / 'ROSTER_LOCK.json')['records']
    cases = {r['probe_alias']: load_case(r)[0] for r in rr}
    targets, bindings = truth._truths_after_seal(cases)
    off, ind, _ = truth._metric_functions()
    rows = []
    usage = []
    maps = {a: {f['case_id']: f for f in b['files']} for a, b in bars.items()}
    jobs = read(BASE / 'INPUT_JOB_LOCK.json')['records']
    ss = {r['case_id']: r for r in read(BASE / 'SUPPORT_LOCK.json')['records']}
    for r in rr:
        cid = r['probe_alias']
        case = cases[cid]
        preds = {}
        support = {j['position'] for j in jobs if j['case_id'] == cid}
        for a, m in maps.items():
            d = torch.load(ROOT / m[cid]['path'], map_location='cpu', weights_only=False)
            if a == 'D2':
                preds[a] = d['arm']
            else:
                preds[a] = d['arms']['TS_M2']
                preds['Frozen'] = d['arms']['F']
                preds['T2'] = d['arms']['T']
                f = d['fits']['M2']
                usage.append(dict(case_id=cid, dataset=r['dataset'], arm=a, fit=f is not None, accepted=0 if f is None else f['optimizer_steps'], proposed=0 if f is None else f['proposed_optimizer_steps'], admission=ss[cid]['admission_reason']))
        for a in ['Frozen', 'T2', 'GEOM', 'PREF', 'D2']:
            v, _ = metric._score_student_arm(preds[a], case, targets[cid], support, off, ind)
            v.update(R_vIoU_at_0p3=float(v['m_vIoU'] >= 0.3), R_vIoU_at_0p5=float(v['m_vIoU'] >= 0.5))
            rows.append(dict(case_id=cid, dataset=r['dataset'], arm=a, metrics=v))
            assert a == 'Frozen' or preds[a]['interval'] == preds['T2']['interval']
    summary, paired = subset_aggregate(rows)
    write(BASE / 'POSTSEAL_RESULTS.json', dict(status='complete', count=32, rows=rows, summary=summary, paired=paired, usage=usage, GT_bindings=bindings, GT_read=True, scope='exposed32 one-anchor fixed-support matched comparison, not4-anchor Direct or independent confirmation'))
    write(BASE / 'CPU_COMPLETION.json', dict(status='complete', count=32, result_sha256=sha(BASE / 'POSTSEAL_RESULTS.json'), GT_read=True))
    print('SCORED32')
if __name__ == '__main__':
    stage = sys.argv[1]
    if stage == 'setup':
        setup()
    elif stage in ['worker', 'root']:
        globals()[stage](sys.argv[2])
    else:
        globals()[stage]()
