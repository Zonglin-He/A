"""CPU postseal readout of the fixed OPD QR x LN-history factorial.

Import and summary() do not access predictions or labels.  Real label access
requires every formal fit hash and the independent deployment receipt first.
No scientific model, optimizer, teacher, or GPU operation is implemented here.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import collections
import gzip
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.stvg_opd_p7a_component_common_v1 import (
    BASE, OUT, ARMS, BOOT, SEED, EPS, read, write, status, sha, load, verify)

DATASETS = ('hc2', 'vidstg')
ORDERS = ('order1', 'order2')
READOUTS = ('I1', 'I0')
STATES = ('Frozen', 'Before', 'After')
METRICS = ('v', 't', 's', 'R30', 'R50')
CONTRASTS = {
    'remove_QR_carry': {'LN_carry': 1., 'joint_carry': -1.},
    'remove_carry_joint': {'joint_reset': 1., 'joint_carry': -1.},
    'remove_carry_LN': {'LN_reset': 1., 'LN_carry': -1.},
    'remove_QR_reset': {'LN_reset': 1., 'joint_reset': -1.},
    'remove_both': {'LN_reset': 1., 'joint_carry': -1.},
    'interaction': {'LN_reset': 1., 'joint_reset': -1.,
                    'LN_carry': -1., 'joint_carry': 1.},
}
FIELDS = [f'{iv}_{st}_{m}' for iv in READOUTS for st in STATES for m in METRICS]
FIELDS += [f'{iv}_delta_{part}_{m}' for iv in READOUTS
           for part in ('total', 'current', 'inherited') for m in METRICS]
FIELDS += [f'{iv}_{origin}_{kind}_R{th}' for iv in READOUTS
           for origin in ('Frozen', 'Before')
           for kind in ('correct_to_wrong', 'wrong_to_correct', 'correct_retained')
           for th in (30, 50)]
COST_FIELDS = ('active_parameters', 'gradient_calls', 'empty', 'rounds',
               'no_information_rounds', 'observed_frames')
PUBLIC_KEYS = {'index', 'parent_id', 'dataset', 'order', 'arrival', 'arm',
               'historically_exposed', 'compute', *FIELDS, *COST_FIELDS}


def array(value):
    if hasattr(value, 'detach'):
        assert value.device.type == 'cpu'
        value = value.detach().numpy()
    out = np.asarray(value, dtype=np.float64)
    assert np.isfinite(out).all()
    return out


def near(a, b):
    """Comparison tolerance only; never changes correctness thresholds."""
    if isinstance(a, dict):
        assert set(a) == set(b)
        return sum(near(v, b[k]) for k, v in a.items())
    if isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        return sum(near(x, y) for x, y in zip(a, b))
    if a is None or isinstance(a, (str, bool, int)):
        assert a == b, (a, b)
    else:
        assert math.isfinite(a) and math.isfinite(b)
        assert abs(a-b) <= 2e-10 * (1+abs(a)), (a, b)
    return 1


def rows_write(path, rows):
    path = Path(path)
    assert not path.exists(), 'Preserve existing public rows'
    with path.open('xb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as f:
            for r in rows:
                f.write((json.dumps(r, ensure_ascii=False, allow_nan=False,
                                   sort_keys=True, separators=(',', ':'))+'\n').encode())


def rows_read(path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def _number(value):
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, (bool, np.bool_))


def public_rows_check(rows):
    """Exhaustive allowlist; identifiers and full mathematical witnesses stay private."""
    for r in rows:
        assert set(r) == PUBLIC_KEYS, set(r) ^ PUBLIC_KEYS
        assert r['dataset'] in DATASETS and r['order'] in ORDERS and r['arm'] in ARMS
        assert r['historically_exposed'] is True
        for k in FIELDS + list(COST_FIELDS):
            assert r[k] is None or (_number(r[k]) and math.isfinite(r[k])), k
        assert isinstance(r['compute'], dict)
        for k, v in r['compute'].items():
            assert isinstance(k, str) and k.replace('_', '').isalnum()
            assert not any(t in k.lower() for t in ('query', 'frame_id', 'ordinal', 'path', 'sha', 'box', 'state'))
            assert _number(v) and math.isfinite(v)
    return len(rows)


def estimate(values):
    """Equal-parent resampling, including all signs, missingness, and severe tails."""
    vals = [float(v) for v in values if v is not None]
    assert all(math.isfinite(v) for v in vals)
    n = len(vals)
    out = dict(defined_parents=n, unknown=len(values)-n,
               positive=sum(v > EPS for v in vals), negative=sum(v < -EPS for v in vals),
               neutral=sum(abs(v) <= EPS for v in vals),
               harm_gt5pp=sum(v < -.05 for v in vals),
               harm_gt20pp=sum(v < -.20 for v in vals),
               gain_gt5pp=sum(v > .05 for v in vals),
               gain_gt20pp=sum(v > .20 for v in vals))
    if not n:
        return dict(out, mean=None, ci95=None, mean_pp=None, ci95_pp=None,
                    gross_gain_pp=None, gross_loss_pp=None, min=None, max=None,
                    parent_values=list(values))
    a = np.asarray(vals, np.float64)
    gen = np.random.default_rng(SEED)
    boot = np.concatenate([a[gen.integers(n, size=(min(250, BOOT-i), n))].mean(1)
                           for i in range(0, BOOT, 250)])
    ci = np.quantile(boot, [.025, .975]).tolist()
    return dict(out, mean=float(a.mean()), ci95=ci,
                mean_pp=float(a.mean()*100), ci95_pp=[v*100 for v in ci],
                gross_gain_pp=float(np.maximum(a, 0).mean()*100),
                gross_loss_pp=float(-np.minimum(a, 0).mean()*100),
                min=float(a.min()), max=float(a.max()), parent_values=list(values))


def parent_rows(rows, *, order=None):
    """Both orders first average within each source; missingness is fail closed."""
    public_rows_check(rows)
    groups = collections.defaultdict(list)
    for r in rows:
        if order is None or r['order'] == order:
            groups[r['dataset'], r['parent_id'], r['arm']].append(r)
    out = []
    for (ds, parent, arm), rr in sorted(groups.items()):
        expected_orders = set(ORDERS) if order is None else {order}
        assert {r['order'] for r in rr} == expected_orders and len(rr) == len(expected_orders)
        assert {r['index'] for r in rr} == {parent}
        def avg(k):
            return math.fsum(r[k] for r in rr)/len(rr) if all(r[k] is not None for r in rr) else None
        costs = sorted({k for r in rr for k in r['compute']})
        out.append(dict(index=parent, parent_id=parent, dataset=ds, arm=arm,
            orders=sorted(expected_orders), historically_exposed=True,
            **{k: avg(k) for k in FIELDS + list(COST_FIELDS)},
            compute={k: math.fsum(r['compute'][k] for r in rr)/len(rr)
                     if all(k in r['compute'] for r in rr) else None for k in costs}))
    return out


def _block(rows, ds=None, order=None):
    selected = [r for r in rows if ds is None or r['dataset'] == ds]
    pp = parent_rows(selected, order=order)
    parents = sorted({(r['dataset'], r['parent_id']) for r in pp})
    table = {(r['dataset'], r['parent_id'], r['arm']): r for r in pp}
    assert len(table) == len(parents)*4
    fields = {}
    for field in FIELDS:
        arms = {a: estimate([table[d, p, a][field] for d, p in parents]) for a in ARMS}
        contrasts = {}
        for name, coeff in CONTRASTS.items():
            values = [math.fsum(c*table[d, p, a][field] for a, c in coeff.items())
                      if all(table[d, p, a][field] is not None for a in coeff) else None
                      for d, p in parents]
            contrasts[name] = estimate(values)
        fields[field] = dict(arms=arms, contrasts=contrasts)
    fitrows = [r for r in selected if order is None or r['order'] == order]
    compute = {}
    for arm in ARMS:
        ar = [r for r in fitrows if r['arm'] == arm]
        costs = sorted({k for r in ar for k in r['compute']})
        compute[arm] = dict(actual_complete_fits=len(ar),
            actual_complete_rounds=sum(r['rounds'] for r in ar),
            actual_backwards=sum(r['gradient_calls'] for r in ar),
            no_information_rounds=sum(r['no_information_rounds'] for r in ar),
            empty_fits=sum(r['empty'] for r in ar),
            active_parameters=sorted({r['active_parameters'] for r in ar}),
            instrumented={k: dict(defined_fits=sum(k in r['compute'] for r in ar),
                unknown_fits=sum(k not in r['compute'] for r in ar),
                total=math.fsum(r['compute'][k] for r in ar if k in r['compute']),
                mean=math.fsum(r['compute'][k] for r in ar if k in r['compute'])/
                     sum(k in r['compute'] for r in ar)) for k in costs})
    return dict(parents=len(parents), complete_fits=len(fitrows),
                aggregation='equal parents; orders averaged within parent before resampling',
                fields=fields, compute=compute)


def summary(rows):
    """Pure anonymous arithmetic; usable by an independent public checker."""
    public_rows_check(rows)
    assert len(rows) == 512
    assert len({(r['dataset'], r['parent_id']) for r in rows}) == 64
    assert len({(r['dataset'], r['parent_id'], r['arm'], r['order']) for r in rows}) == 512
    assert {r['parent_id'] for r in rows} == set(range(64))
    for ds in DATASETS:
        assert len({r['parent_id'] for r in rows if r['dataset'] == ds}) == 32
    return dict(status='postseal_exploratory_development_factorial',
        primary='I1_After_v', secondary='I0_After_v', metric_units='fractions; *_pp fields are percentage points',
        bootstrap=BOOT, seed=SEED, arms=ARMS, contrast_coefficients=CONTRASTS,
        parents=64, complete_fits=512, complete_rounds=sum(r['rounds'] for r in rows),
        historically_exposed_development=True, confirmation_used=False,
        cost_accounting='Each fit is instrumented; shared frozen capture cost/count is allocated once using capture_owner.',
        automatic_promotion=False, equivalence_claim=False,
        limitation='Unadjusted exploratory CI; a CI crossing zero is not equivalence or proof a component is dispensable.',
        pooled=_block(rows),
        datasets={ds: dict(paired_both_orders=_block(rows, ds),
                           orders={o: _block(rows, ds, o) for o in ORDERS}) for ds in DATASETS})


def require_seal():
    """Every original hash is checked before any label-reading module is called."""
    runtime = verify()
    bp = BASE/'GLOBAL_PREDICTION_BARRIER.json'
    b = read(bp)
    assert b['status'] == 'sealed' and b['GT_read'] is False
    assert b['actual_complete_fits'] == 512 and b['actual_complete_rounds'] == 12800
    assert b['runtime_sha256'] == sha(BASE/'RUNTIME.json')
    d = read(BASE/'DESIGN_LOCK.json')
    expected = {str((BASE/'formal'/ds/order/arm/f'{at:05}.pt').relative_to(ROOT))
                for ds, stage in d['stages'].items() for order, seq in stage['orders'].items()
                for arm in ARMS for at in range(len(seq))}
    assert len(expected) == 512 and set(b['files']) == expected
    for rel, h in b['files'].items():
        assert not Path(rel).is_absolute() and '..' not in Path(rel).parts
        assert sha(ROOT/rel) == h, rel
    captures = {str((BASE/'formal/captures'/ds/order/f'{at:05}_repeat0.pt').relative_to(ROOT))
                for ds, stage in d['stages'].items() for order, seq in stage['orders'].items()
                for at in range(len(seq))}
    assert len(captures) == 128 and set(b['capture_files']) == captures
    assert b['actual_frozen_captures'] == 128 and b['actual_new_DINO_calls'] == 0
    for rel, h in b['capture_files'].items():
        assert sha(ROOT/rel) == h, rel
    rp = BASE/'ROOT_DEPLOYMENT_READBACK.json'
    r = read(rp)
    assert r['status'] == 'pass' and r['barrier_sha256'] == sha(bp)
    assert r['GT_read'] is False and r['time'] >= b['time']
    assert r['counts']['fits'] == 512 and r['counts']['rounds'] == 12800
    assert runtime['GT_read'] is False
    return b


def _exposure(barrier):
    path = BASE/'GT_EXPOSURE.json'
    bound = sha(BASE/'GLOBAL_PREDICTION_BARRIER.json')
    if path.exists():
        old = read(path)
        assert old['barrier_sha256'] == bound and old['time'] >= barrier['time']
        return
    write(path, dict(status='actual_GT_only_after_all512_complete_fits_and_label_free_root',
        barrier_sha256=bound,
        root_deployment_readback_sha256=sha(BASE/'ROOT_DEPLOYMENT_READBACK.json'),
        historically_exposed_development=True, GT_for_selection=False,
        GT_for_update=False, CPU_only=True, time=time.time()))


def assemble(independent=False, *, return_provenance=False):
    """Return all 512 anonymous rows; independent uses separate dense geometry.

    Only explicit execution calls this label-reading path.  The same sealed
    deployment witnesses supply both modes; neither mode reruns a model.
    """
    import torch
    torch.set_num_threads(2)
    assert not torch.cuda.is_initialized()
    barrier = require_seal()
    _exposure(barrier)
    from scripts.score_stvg_opd_p1_v1 import truths
    from scripts.run_decota_paper_main_v1 import read_row
    from vg_tta.tastvg_oracle_event5_v1 import official
    from scripts.review_stvg_opd_revised_p1_v2 import dense_metric
    d = read(BASE/'DESIGN_LOCK.json')
    byquery = {(r['dataset'], r['query_ordinal']): r for r in d['records']}
    assert len(byquery) == 64
    targets = {ds: truths(ds, dict(parents=d['stages'][ds]['parents'])) for ds in DATASETS}
    provenance = {}
    rows = []
    checks = 0
    for ds in DATASETS:
        truth, spans, binding = targets[ds]
        provenance.update(binding)
        stage = d['stages'][ds]
        for order, seq in stage['orders'].items():
            for at, q in enumerate(seq):
                rec = byquery[ds, q]
                raw = read_row(ds, q)
                for arm in ARMS:
                    p = BASE/'formal'/ds/order/arm/f'{at:05}.pt'
                    rel = str(p.relative_to(ROOT))
                    assert sha(p) == barrier['files'][rel]
                    z = load(p)
                    f = z['fit']
                    assert z['dataset'] == ds and z['arm'] == arm and z['order'] == order
                    assert z['parent_id'] == rec['parent_id'] == rec['index'] and z['query_ordinal'] == q
                    assert z['GT_read'] is False and f['GT_used'] is False
                    assert z['frame_ids'] == raw['frame_ids']
                    expected_scope = 'LN_only' if arm.startswith('LN_') else 'joint'
                    assert f.get('scope', 'joint') == expected_scope
                    assert f['active_parameters'] == (1536 if expected_scope == 'LN_only' else 1792)
                    cfg = stage['configs'][arm]
                    assert f['config'] == cfg and f['selected_step'] == cfg['steps']
                    assert len(f['rounds']) == cfg['steps'] and len(f['path']) == cfg['steps']+1
                    assert f['empty'] == (not f['positions'])
                    assert f['gradient_calls'] == sum(bool(rd['updated']) for rd in f['rounds'])
                    ib = z['original_input_binding']
                    assert ib['input_path'] == rec['originals'][order]['input_path']
                    assert ib['input_sha256'] == rec['originals'][order]['input_sha256']
                    ip = ROOT/ib['input_path']
                    assert sha(ip) == ib['input_sha256']
                    inp = load(ip)
                    assert inp['GT_read'] is False and inp['frame_ids'] == z['frame_ids']
                    assert inp['interval'] == z['Native_interval']
                    assert inp['pixel_sha256'] == rec['pixel_sha256']
                    rr = dict(index=rec['index'], parent_id=rec['parent_id'], dataset=ds,
                        order=order, arrival=at, arm=arm, historically_exposed=True,
                        active_parameters=int(f['active_parameters']), gradient_calls=int(f['gradient_calls']),
                        empty=int(f['empty']), rounds=len(f['rounds']),
                        no_information_rounds=sum(bool(rd['no_information']) for rd in f['rounds']),
                        observed_frames=len(f['positions']), compute={})
                    owner = z['compute']['capture_owner']
                    assert type(owner) is bool and owner == (arm == ARMS[0])
                    for k, value in z['compute'].items():
                        if isinstance(value, (bool, np.bool_)):
                            value = int(value)
                        if _number(value):
                            assert math.isfinite(value)
                            if k in ('frozen_capture_seconds', 'actual_frozen_capture_forwards', 'decoded_frame_reuse'):
                                value = value if owner else 0
                            rr['compute'][k] = int(value) if isinstance(value, (int, np.integer)) else float(value)
                    boxes = {'Frozen': inp['native_boxes'], 'Before': f['before'], 'After': f['final']}
                    for iv, interval in [('I1', z['Projection_interval']), ('I0', z['Native_interval'])]:
                        assert len(interval) == 2 and interval[1] > interval[0]
                        for st, box in boxes.items():
                            b = array(box)
                            assert b.shape == (len(raw['frame_ids']), 4)
                            vals = (dense_metric(b, raw, truth[q], spans[q], interval, ds)[0]
                                    if independent else official(b, raw, truth[q], spans[q], interval, ds))
                            if independent:
                                checks += near(vals, official(b, raw, truth[q], spans[q], interval, ds))
                            for met in ('v', 't', 's'):
                                value = float(vals[met])
                                assert math.isfinite(value) and -EPS <= value <= 1+EPS
                                rr[f'{iv}_{st}_{met}'] = value
                            for th in (30, 50):
                                rr[f'{iv}_{st}_R{th}'] = float(vals['v'] > th/100)
                        assert rr[f'{iv}_Frozen_t'] == rr[f'{iv}_Before_t'] == rr[f'{iv}_After_t']
                        for met in METRICS:
                            frozen, before, after = [rr[f'{iv}_{st}_{met}'] for st in STATES]
                            rr[f'{iv}_delta_total_{met}'] = after-frozen
                            rr[f'{iv}_delta_current_{met}'] = after-before
                            rr[f'{iv}_delta_inherited_{met}'] = before-frozen
                            checks += near(after-frozen, (after-before)+(before-frozen))
                        for origin in ('Frozen', 'Before'):
                            for th in (30, 50):
                                pre = bool(rr[f'{iv}_{origin}_R{th}'])
                                post = bool(rr[f'{iv}_After_R{th}'])
                                rr[f'{iv}_{origin}_correct_to_wrong_R{th}'] = float(pre and not post)
                                rr[f'{iv}_{origin}_wrong_to_correct_R{th}'] = float(not pre and post)
                                rr[f'{iv}_{origin}_correct_retained_R{th}'] = float(pre and post)
                    for st in STATES:
                        checks += near(rr[f'I1_{st}_s'], rr[f'I0_{st}_s'])
                    if arm.endswith('reset'):
                        for iv in READOUTS:
                            for met in METRICS:
                                checks += near(rr[f'{iv}_delta_inherited_{met}'], 0.)
                    assert sha(p) == barrier['files'][rel] and sha(ip) == ib['input_sha256']
                    rows.append(rr)
                status(BASE/('ROOT_SCALAR_STATUS.json' if independent else 'CPU_STATUS.json'),
                    dict(status='postseal_independent_scalar_readback' if independent else 'postseal_CPU_scoring',
                         done=len(rows), total=512, CPU_only=True, GT_read=True, time=time.time()))
    assert len(rows) == public_rows_check(rows) == 512
    assert sum(r['rounds'] for r in rows) == 12800
    if return_provenance:
        return rows, provenance, checks
    return rows


def score():
    require_seal()
    assert not (BASE/'CPU_COMPLETION.json').exists(), 'Existing score is immutable; use a new explicit revision'
    assert OUT.is_dir()
    start = time.time()
    rows, provenance, checks = assemble(return_provenance=True)
    pp = parent_rows(rows)
    assert len(pp) == 256
    rows_write(OUT/'ROWS.jsonl.gz', rows)
    rows_write(OUT/'PARENT_ROWS.jsonl.gz', pp)
    write(OUT/'SUMMARY.json', summary(rows))
    files = {str(p.relative_to(ROOT)): sha(p) for p in
             [OUT/'ROWS.jsonl.gz', OUT/'PARENT_ROWS.jsonl.gz', OUT/'SUMMARY.json']}
    write(BASE/'CPU_COMPLETION.json', dict(status='scored_pending_independent_root_report_publication',
        barrier_sha256=sha(BASE/'GLOBAL_PREDICTION_BARRIER.json'), rows=512, parent_arm_rows=256,
        complete_rounds=12800, GT_provenance=provenance, metric_identity_checks=checks,
        files=files, CPU_only=True, GT_read=True, CPU_seconds=time.time()-start, time=time.time()))
    print('P7A_COMPONENT_CPU_SCORE_COMPLETE', flush=True)


def root():
    """Only an explicit root invocation publishes this independent scalar receipt."""
    require_seal()
    complete = read(BASE/'CPU_COMPLETION.json')
    for rel, h in complete['files'].items():
        assert sha(ROOT/rel) == h
    expected, provenance, checks = assemble(independent=True, return_provenance=True)
    actual = rows_read(OUT/'ROWS.jsonl.gz')
    assert len(actual) == len(expected) == 512
    checks += near(expected, actual)
    checks += near(parent_rows(expected), rows_read(OUT/'PARENT_ROWS.jsonl.gz'))
    checks += near(summary(expected), read(OUT/'SUMMARY.json'))
    checks += near(provenance, complete['GT_provenance'])
    receipt = dict(status='pass',
        scope='all512 independent dense geometry/strict R thresholds/256 parent means/10000 bootstrap and all signed outcomes',
        barrier_sha256=sha(BASE/'GLOBAL_PREDICTION_BARRIER.json'),
        deployment_readback_sha256=sha(BASE/'ROOT_DEPLOYMENT_READBACK.json'),
        scoring_completion_sha256=sha(BASE/'CPU_COMPLETION.json'),
        files=complete['files'], scalar_checks=checks, rows=512, parent_arm_rows=256,
        independent_dense_geometry=True, independent_deployment_math=False,
        summary_reuses_declared_pure_function=True, CPU_only=True, GT_read=True, time=time.time())
    deployment = read(BASE/'ROOT_DEPLOYMENT_READBACK.json')
    write(BASE/'ROOT_READBACK.json', dict(receipt,
        scope='existing independent label-free deployment mathematics plus exhaustive postseal independent dense scalar readback',
        deployment_math_counts=deployment['counts'],
        independent_deployment_math_in_separate_receipt=True))
    write(BASE/'ROOT_ANONYMOUS_SCALAR_READBACK.json', receipt)
    print('P7A_COMPONENT_SCALAR_ROOT_PASS', checks, flush=True)


def selftest():
    """Synthetic geometry, sign, order-pairing, strict-threshold and unknown contracts."""
    from vg_tta.tastvg_oracle_event5_v1 import official
    from scripts.review_stvg_opd_revised_p1_v2 import dense_metric
    raw = dict(frame_ids=[0, 2, 4], input=dict(width=100, height=80))
    boxes = np.tile([.5, .5, .2, .3], (3, 1))
    gt = {i: [40., 28., 60., 52.] for i in range(5)}
    for ds in DATASETS:
        for interval in ([0, 4], [1, 3]):
            near(official(boxes, raw, gt, [0, 4], interval, ds),
                 dense_metric(boxes, raw, gt, [0, 4], interval, ds)[0])
    assert not .3 > .3 and not .5 > .5
    e = estimate([.06, -.21, 0., None])
    assert (e['positive'], e['negative'], e['neutral'], e['unknown'], e['harm_gt20pp']) == (1, 1, 1, 1, 1)
    rr = []
    for index in range(64):
        for order in ORDERS:
            for arm in ARMS:
                r = dict(index=index, parent_id=index, dataset=DATASETS[index//32], order=order,
                    arrival=index % 32, arm=arm, historically_exposed=True,
                    compute=dict(fit_seconds=1.), active_parameters=1536 if arm.startswith('LN') else 1792,
                    gradient_calls=0, empty=1, rounds=40 if index < 32 else 10,
                    no_information_rounds=40 if index < 32 else 10, observed_frames=0)
                r.update({k: 0. for k in FIELDS})
                r['I1_After_v'] = .2 + (0.1 if order == 'order2' else 0.) + ARMS.index(arm)*.01
                rr.append(r)
    s = summary(rr)
    assert s['complete_rounds'] == 12800 and len(parent_rows(rr)) == 256
    c = s['pooled']['fields']['I1_After_v']['contrasts']
    near(c['remove_QR_carry']['mean'], .01)
    near(c['interaction']['mean'], 0.)
    return dict(status='pass', synthetic_only=True, actual_GT_read=False,
                model_calls=0, GPU_calls=0, rows=512, complete_rounds=12800)


if __name__ == '__main__':
    action = sys.argv[1] if len(sys.argv) > 1 else 'score'
    if action == 'selftest':
        print(json.dumps(selftest(), sort_keys=True))
    else:
        {'score': score, 'root': root}[action]()
