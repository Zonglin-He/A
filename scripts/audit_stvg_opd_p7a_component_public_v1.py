"""Portable, read-only anonymous audit of the fixed P7-A OPD factorial.

run(checkout) reads public JSON/gzip tables, projected completion receipts and
the public source files explicitly named by those receipts.  It never opens a
fit, checkpoint, input cache, annotation, image, video or scientific model.
All parent arithmetic and the 10,000-draw count-based bootstrap are independent
of the scorer.  Source custody is a binding of completed root receipts, not an
independent re-read of their private mathematical witnesses.  Nothing is written.
"""
import ast
import collections
import functools
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import numpy as np

OUT_REL = 'results/stvg_opd_p7a_component_simplification/2026-10-13'
ARMS = ['joint_carry', 'LN_carry', 'joint_reset', 'LN_reset']
DATASETS = ('hc2', 'vidstg')
ORDERS = ('order1', 'order2')
READOUTS = ('I1', 'I0')
STATES = ('Frozen', 'Before', 'After')
METRICS = ('v', 't', 's', 'R30', 'R50')
BOOT, SEED, EPS = 10000, 20261013, 1e-12
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
ROW_KEYS = {'index', 'parent_id', 'dataset', 'order', 'arrival', 'arm',
            'historically_exposed', 'compute', *FIELDS, *COST_FIELDS}
PARENT_KEYS = {'index', 'parent_id', 'dataset', 'arm', 'orders',
               'historically_exposed', 'compute', *FIELDS, *COST_FIELDS}
COMPUTE_KEYS = {'instrumented_fit_seconds', 'frozen_capture_seconds',
                'decoded_frame_reuse', 'actual_new_DINO_calls',
                'actual_frozen_capture_forwards', 'capture_owner',
                'CUDA_peak_allocated', 'actual_active_parameters'}
SCIENCE = [
    'protocols/stvg_opd_p7a_component_simplification_v1.md',
    'scripts/stvg_opd_p7a_component_common_v1.py',
    'scripts/run_stvg_opd_p7a_component_v1.py',
    'scripts/score_stvg_opd_p7a_component_v1.py',
    'vg_tta/stvg_opd_p7a_component_state_audit_v1.py',
]
REPORTING = ['scripts/report_stvg_opd_p7a_component_v1.py',
             'scripts/audit_stvg_opd_p7a_component_public_v1.py',
             'scripts/publish_stvg_opd_p7a_component_v1.py',
             'scripts/close_stvg_opd_p7a_component_v1.py']
SHARED_SCIENCE = ['scripts/run_decota_paper_main_v1.py',
                  'scripts/stvg_opd_legacy_input_bridge003.py',
                  'vg_tta/decota_spatial_opd_tunable_v1.py',
                  'vg_tta/stvg_opd_paper_ablations_v1.py',
                  'vg_tta/stvg_opd_p3_scope_precision001.py',
                  'vg_tta/stvg_opd_common_state_geometry_arithmetic004.py',
                  'methods/decota_final_simplified_v1/objectives.py',
                  'scripts/stvg_opd_p6_existing_common001.py',
                  'methods/decota_final_simplified_v1/config.py']
RECEIPTS = ('RUNTIME.json', 'CPU_CONTRACTS.json',
            'SOURCE_STATE_CUSTODY_READBACK.json', 'QUALIFICATION_BARRIER.json',
            'ROOT_QUALIFICATION.json', 'GLOBAL_PREDICTION_BARRIER.json',
            'ROOT_DEPLOYMENT_READBACK.json', 'GT_EXPOSURE.json',
            'CPU_COMPLETION.json', 'ROOT_ANONYMOUS_SCALAR_READBACK.json',
            'ROOT_READBACK.json', 'ROOT_PUBLIC_SOURCE_BINDINGS_READBACK.json')
SCALAR_FILES = ('ROWS.jsonl.gz', 'PARENT_ROWS.jsonl.gz', 'SUMMARY.json')
FORBIDDEN = {'query', 'query_id', 'query_ordinal', 'qid', 'caption', 'text',
             'prompt', 'sentence', 'frame_ids', 'frame_id', 'physical_frame_id',
             'coordinates', 'boxes', 'box', 'bbox', 'bboxes', 'xyxy', 'cxcywh',
             'samples', 'actions', 'rewards', 'weights', 'gradient', 'gradients',
             'state', 'initial', 'final', 'fit', 'exp_avg', 'exp_avg_sq',
             'features', 'annotations', 'ground_truth', 'GT_provenance',
             'bindings', 'protected', 'processes', 'command', 'private_source_paths'}


def _pairs(items):
    out = {}
    for k, v in items:
        assert k not in out, ('duplicate JSON key', k)
        out[k] = v
    return out


def _decode(value):
    def invalid(v):
        raise AssertionError(('nonfinite JSON literal', v))
    return json.loads(value, object_pairs_hook=_pairs, parse_constant=invalid)


def read(path):
    return _decode(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def _digest(value):
    assert isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value), 'Invalid SHA256'
    return value


def _number(value):
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, (bool, np.bool_))


def _integer(value, low=0, high=None):
    assert isinstance(value, int) and not isinstance(value, bool)
    assert value >= low and (high is None or value <= high)


def near(expected, actual, where='root'):
    """Roundoff tolerance for arithmetic only, never for R30/R50 decisions."""
    if isinstance(expected, dict):
        assert isinstance(actual, dict) and set(expected) == set(actual), (where, 'schema mismatch')
        return sum(near(v, actual[k], where + '.' + str(k)) for k, v in expected.items())
    if isinstance(expected, (list, tuple)):
        assert isinstance(actual, (list, tuple)) and len(expected) == len(actual), where
        return sum(near(x, y, where + f'[{i}]') for i, (x, y) in enumerate(zip(expected, actual)))
    if _number(expected):
        assert _number(actual) and math.isfinite(expected) and math.isfinite(actual), where
        assert abs(expected-actual) <= 2e-10 * (1+abs(expected)), (where, expected, actual)
    else:
        assert type(expected) is type(actual) and expected == actual, (where, expected, actual)
    return 1


def privacy(value, where='root'):
    """Scalar metadata, hash strings, anonymous parent vectors and code paths only."""
    if isinstance(value, dict):
        for k, v in value.items():
            assert isinstance(k, str) and k not in FORBIDDEN, (where, k)
            low = k.lower()
            assert not ('query_ordinal' in low or low.startswith('gt_box') or
                        low.startswith('expert_box') or low.endswith('_vectors')), (where, k)
            # Digests can bind IDs without exposing any actual physical ID.
            assert 'frame_id' not in low or low == 'frame_ids_sha256', (where, k)
            privacy(v, where + '.' + k)
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            privacy(v, where + f'[{i}]')
    elif isinstance(value, str):
        low = value.lower()
        assert not any(s in low for s in ('file://', 'data:image/', 'data:video/',
                                        '/annotations/', '/annos/', '/.codex/',
                                        '.safetensors', '.pth', '.pt', '.npz')), (where, 'private reference')
        assert not value.startswith('/home/') and not value.startswith('/mnt/'), (where, 'absolute path')
    elif _number(value):
        assert math.isfinite(value), where
    else:
        assert value is None or isinstance(value, bool), (where, type(value))


def rows_check(rows):
    """Exact 512-cell factorial, anonymous allowlist and all per-row identities."""
    assert len(rows) == 512
    cells, arrivals, frozen = set(), collections.defaultdict(dict), {}
    checks = 0
    for r in rows:
        assert isinstance(r, dict) and set(r) == ROW_KEYS, ('row keys', set(r) ^ ROW_KEYS)
        assert r['dataset'] in DATASETS and r['arm'] in ARMS and r['order'] in ORDERS
        _integer(r['parent_id'], 0, 63)
        assert r['index'] == r['parent_id'] and type(r['index']) is int
        assert r['dataset'] == DATASETS[r['parent_id']//32]
        assert r['historically_exposed'] is True
        _integer(r['arrival'], 0, 31)
        key = r['dataset'], r['parent_id'], r['order'], r['arm']
        assert key not in cells, ('duplicate cell', key)
        cells.add(key)
        amap = arrivals[r['dataset'], r['order']]
        if r['parent_id'] in amap:
            assert amap[r['parent_id']] == r['arrival']
        amap[r['parent_id']] = r['arrival']
        for k in FIELDS:
            assert r[k] is None or (_number(r[k]) and math.isfinite(r[k])), k
        for k in COST_FIELDS:
            _integer(r[k])
        K = 40 if r['dataset'] == 'hc2' else 10
        assert r['rounds'] == K
        assert r['active_parameters'] == (1536 if r['arm'].startswith('LN_') else 1792)
        assert r['gradient_calls'] + r['no_information_rounds'] == K
        assert 0 <= r['observed_frames'] <= 4 and r['empty'] == int(r['observed_frames'] == 0)
        if r['empty']:
            assert r['gradient_calls'] == 0
        c = r['compute']
        assert isinstance(c, dict) and set(c) <= COMPUTE_KEYS
        for k, v in c.items():
            assert _number(v) and math.isfinite(v) and v >= 0, (k, v)
        if 'actual_active_parameters' in c:
            assert c['actual_active_parameters'] == r['active_parameters']
        if 'actual_new_DINO_calls' in c:
            assert c['actual_new_DINO_calls'] == 0
        if 'capture_owner' in c:
            assert c['capture_owner'] == int(r['arm'] == ARMS[0])
        if 'actual_frozen_capture_forwards' in c:
            assert c['actual_frozen_capture_forwards'] == int(r['arm'] == ARMS[0])
        if r['arm'] != ARMS[0]:
            for k in ('frozen_capture_seconds', 'decoded_frame_reuse', 'actual_frozen_capture_forwards'):
                if k in c:
                    assert c[k] == 0, ('unallocated shared cost', k)
        if 'decoded_frame_reuse' in c:
            assert c['decoded_frame_reuse'] in (0, 1)
        for iv in READOUTS:
            for st in STATES:
                for m in METRICS:
                    v = r[f'{iv}_{st}_{m}']
                    assert v is None or -EPS <= v <= 1+EPS
                v = r[f'{iv}_{st}_v']
                for th in (30, 50):
                    recall = r[f'{iv}_{st}_R{th}']
                    assert recall is None or recall in (0., 1.)
                    if v is not None and recall is not None:
                        assert recall == float(v > th/100), ('strict threshold', iv, st, th)
                        checks += 1
            if all(r[f'{iv}_{st}_t'] is not None for st in STATES):
                checks += near([r[f'{iv}_Frozen_t']]*3, [r[f'{iv}_{st}_t'] for st in STATES])
            for m in METRICS:
                f, b, a = [r[f'{iv}_{st}_{m}'] for st in STATES]
                for part, x, y in (('total', a, f), ('current', a, b), ('inherited', b, f)):
                    d = r[f'{iv}_delta_{part}_{m}']
                    assert d is None or -1-EPS <= d <= 1+EPS
                    if x is not None and y is not None:
                        assert d is not None
                        checks += near(x-y, d)
                    if r['arm'].endswith('reset') and part == 'inherited' and d is not None:
                        checks += near(0., d)
                    if r['empty'] and part == 'current' and d is not None:
                        checks += near(0., d)
                d = [r[f'{iv}_delta_{p}_{m}'] for p in ('total', 'current', 'inherited')]
                if all(v is not None for v in d):
                    checks += near(d[0], d[1]+d[2])
            for origin in ('Frozen', 'Before'):
                for th in (30, 50):
                    pre, post = r[f'{iv}_{origin}_R{th}'], r[f'{iv}_After_R{th}']
                    for kind, flag in [('correct_to_wrong', bool(pre) and not bool(post)),
                                       ('wrong_to_correct', not bool(pre) and bool(post)),
                                       ('correct_retained', bool(pre) and bool(post))]:
                        v = r[f'{iv}_{origin}_{kind}_R{th}']
                        assert v is None or v in (0., 1.)
                        if pre is not None and post is not None:
                            assert v == float(flag)
                            checks += 1
        for st in STATES:
            x, y = r[f'I1_{st}_s'], r[f'I0_{st}_s']
            if x is not None and y is not None:
                checks += near(x, y)
        fk = r['dataset'], r['parent_id'], r['order']
        fv = [r[f'{iv}_Frozen_{m}'] for iv in READOUTS for m in METRICS]
        if fk in frozen:
            checks += near(frozen[fk], fv)
        frozen[fk] = fv
    assert len(cells) == 512 and len(arrivals) == 4 and len(frozen) == 128
    for (ds, _), mapping in arrivals.items():
        assert len(mapping) == 32 and set(mapping.values()) == set(range(32))
        assert set(mapping) == set(range(0 if ds == 'hc2' else 32, 32 if ds == 'hc2' else 64))
    assert sum(r['rounds'] for r in rows) == 12800
    privacy(rows, 'ROWS')
    return checks


def independent_parents(rows, order=None):
    groups = collections.defaultdict(list)
    for r in rows:
        if order is None or r['order'] == order:
            groups[r['dataset'], r['parent_id'], r['arm']].append(r)
    out = []
    for (ds, p, arm), rr in sorted(groups.items()):
        orders = sorted(ORDERS if order is None else (order,))
        assert len(rr) == len(orders) and sorted(r['order'] for r in rr) == orders
        fields = {k: math.fsum(r[k] for r in rr)/len(rr)
                  if all(r[k] is not None for r in rr) else None for k in (*FIELDS, *COST_FIELDS)}
        keys = sorted(set().union(*(r['compute'] for r in rr)))
        costs = {k: math.fsum(r['compute'][k] for r in rr)/len(rr)
                 if all(k in r['compute'] for r in rr) else None for k in keys}
        out.append(dict(index=p, parent_id=p, dataset=ds, arm=arm, orders=orders,
                        historically_exposed=True, compute=costs, **fields))
    return out


@functools.lru_cache(maxsize=65)
def bootstrap_counts(n):
    assert 1 <= n <= 64
    draws = np.random.default_rng(SEED).integers(n, size=(BOOT, n))
    # A count matrix, rather than the scorer's advanced-index/mean calculation.
    offsets = np.arange(BOOT)[:, None]*n
    counts = np.bincount((draws+offsets).ravel(), minlength=BOOT*n).reshape(BOOT, n)
    counts.flags.writeable = False
    return counts


@functools.lru_cache(maxsize=10000)
def _estimate(values):
    defined = [float(x) for x in values if x is not None]
    assert all(math.isfinite(x) for x in defined)
    n = len(defined)
    result = dict(defined_parents=n, unknown=len(values)-n,
                  positive=sum(x > EPS for x in defined), negative=sum(x < -EPS for x in defined),
                  neutral=sum(abs(x) <= EPS for x in defined),
                  harm_gt5pp=sum(x < -.05 for x in defined), harm_gt20pp=sum(x < -.20 for x in defined),
                  gain_gt5pp=sum(x > .05 for x in defined), gain_gt20pp=sum(x > .20 for x in defined))
    if not n:
        return dict(result, mean=None, ci95=None, mean_pp=None, ci95_pp=None,
                    gross_gain_pp=None, gross_loss_pp=None, min=None, max=None,
                    parent_values=list(values))
    a = np.asarray(defined, dtype=np.float64)
    means = (bootstrap_counts(n)*a[None, :]).sum(axis=1)/n
    ci = np.quantile(means, [.025, .975], method='linear').tolist()
    mean = math.fsum(defined)/n
    return dict(result, mean=mean, ci95=ci, mean_pp=100*mean,
                ci95_pp=[100*x for x in ci],
                gross_gain_pp=100*math.fsum(max(x, 0.) for x in defined)/n,
                gross_loss_pp=100*math.fsum(max(-x, 0.) for x in defined)/n,
                min=min(defined), max=max(defined), parent_values=list(values))


def independent_estimate(values):
    return _estimate(tuple(values))


def _block(rows, ds=None, order=None):
    selected = [r for r in rows if (ds is None or r['dataset'] == ds) and
                (order is None or r['order'] == order)]
    pp = independent_parents(selected, order)
    parents = sorted({(r['dataset'], r['parent_id']) for r in pp})
    table = {(r['dataset'], r['parent_id'], r['arm']): r for r in pp}
    assert len(table) == len(parents)*4
    fields = {}
    for field in FIELDS:
        arms = {a: independent_estimate([table[d, p, a][field] for d, p in parents]) for a in ARMS}
        contrasts = {}
        for name, coef in CONTRASTS.items():
            vals = []
            for d, p in parents:
                vals.append(math.fsum(c*table[d, p, a][field] for a, c in coef.items())
                            if all(table[d, p, a][field] is not None for a in coef) else None)
            contrasts[name] = independent_estimate(vals)
        fields[field] = dict(arms=arms, contrasts=contrasts)
    compute = {}
    for a in ARMS:
        rr = [r for r in selected if r['arm'] == a]
        keys = sorted(set().union(*(r['compute'] for r in rr)))
        instrumented = {}
        for k in keys:
            vals = [r['compute'][k] for r in rr if k in r['compute']]
            total = math.fsum(vals)
            instrumented[k] = dict(defined_fits=len(vals), unknown_fits=len(rr)-len(vals),
                                    total=total, mean=total/len(vals))
        compute[a] = dict(actual_complete_fits=len(rr), actual_complete_rounds=sum(r['rounds'] for r in rr),
                          actual_backwards=sum(r['gradient_calls'] for r in rr),
                          no_information_rounds=sum(r['no_information_rounds'] for r in rr),
                          empty_fits=sum(r['empty'] for r in rr),
                          active_parameters=sorted({r['active_parameters'] for r in rr}),
                          instrumented=instrumented)
    return dict(parents=len(parents), complete_fits=len(selected),
                aggregation='equal parents; orders averaged within parent before resampling',
                fields=fields, compute=compute)


def independent_summary(rows):
    return dict(status='postseal_exploratory_development_factorial',
        primary='I1_After_v', secondary='I0_After_v',
        metric_units='fractions; *_pp fields are percentage points',
        bootstrap=BOOT, seed=SEED, arms=ARMS, contrast_coefficients=CONTRASTS,
        parents=64, complete_fits=512, complete_rounds=sum(r['rounds'] for r in rows),
        historically_exposed_development=True, confirmation_used=False,
        cost_accounting='Each fit is instrumented; shared frozen capture cost/count is allocated once using capture_owner.',
        automatic_promotion=False, equivalence_claim=False,
        limitation='Unadjusted exploratory CI; a CI crossing zero is not equivalence or proof a component is dispensable.',
        pooled=_block(rows),
        datasets={ds: dict(paired_both_orders=_block(rows, ds),
                           orders={o: _block(rows, ds, o) for o in ORDERS}) for ds in DATASETS})


def protocol_check(protocol, rows):
    """Bind anonymous parent membership, per-order arrivals and fixed config."""
    privacy(protocol, 'ANONYMOUS_PROTOCOL')
    keys = {'status', 'records', 'arms', 'parents', 'parents_each', 'orders',
            'qualification_complete_GPU_fits', 'formal_complete_GPU_fits',
            'qualification_rounds', 'formal_rounds', 'qualification_captures',
            'formal_captures', 'old_joint_carry_controls', 'joint_active_parameters',
            'LN_active_parameters', 'bootstrap', 'seed', 'historically_exposed_development',
            'independent_confirmation', 'new_DINO_calls', 'primary', 'secondary',
            'Query_reset_each_arrival', 'Adam_reset_each_arrival', 'complete_own_history_each_arm',
            'reset_absolute_source_LN', 'no_alternate_objective', 'parameter_search_executed',
            'production_promoted', 'monitor_recreated', 'paper_main_experiments_executed',
            'excluded_categories'}
    assert set(protocol) == keys
    fixed = dict(status='complete_finite_internal_component_experiment', arms=ARMS,
                 parents=64, parents_each=32, orders=list(ORDERS),
                 qualification_complete_GPU_fits=64, formal_complete_GPU_fits=512,
                 qualification_rounds=1600, formal_rounds=12800,
                 qualification_captures=16, formal_captures=128,
                 old_joint_carry_controls=128, joint_active_parameters=1792,
                 LN_active_parameters=1536, bootstrap=BOOT, seed=SEED,
                 historically_exposed_development=True, independent_confirmation=False,
                 new_DINO_calls=0, primary='I1 parent-macro mvIoU', secondary='I0 parent-macro mvIoU',
                 Query_reset_each_arrival=True, Adam_reset_each_arrival=True,
                 complete_own_history_each_arm=True, reset_absolute_source_LN=True,
                 no_alternate_objective=True, parameter_search_executed=False,
                 production_promoted=False, monitor_recreated=False, paper_main_experiments_executed=False,
                 excluded_categories=['private media', 'annotations', 'weights', 'raw cached input',
                                      'complete fits', 'parameter or gradient vectors', 'personal conversation'])
    checks = near(fixed, {k: protocol[k] for k in fixed})
    assert len(protocol['records']) == 64
    recordkeys = {'parent_id', 'dataset', 'source', 'source_model_sha256', 'pixel_sha256',
                  'frame_ids_sha256', 'evidence_sha256', 'old_complete_fit_sha256',
                  'input_sha256', 'arrival', 'configs'}
    byparent, sourcehashes = {}, collections.defaultdict(set)
    configs = {'hc2': dict(lr=.01, sigma=.025, tau=.05, steps=40, writeback=.0625, actions_per_round=32),
               'vidstg': dict(lr=.03, sigma=.1, tau=.25, steps=10, writeback=.125, actions_per_round=32)}
    for r in protocol['records']:
        assert set(r) == recordkeys
        p, ds = r['parent_id'], r['dataset']
        _integer(p, 0, 63)
        assert p not in byparent and ds == DATASETS[p//32]
        assert r['source'] == ('vidstg' if ds == 'hc2' else 'hcstvg2')
        sourcehashes[ds].add(_digest(r['source_model_sha256']))
        for k in ('pixel_sha256', 'frame_ids_sha256', 'evidence_sha256', 'input_sha256'):
            _digest(r[k])
        assert set(r['old_complete_fit_sha256']) == set(ORDERS) == set(r['arrival'])
        for o in ORDERS:
            _digest(r['old_complete_fit_sha256'][o])
            _integer(r['arrival'][o], 0, 31)
        assert set(r['configs']) == set(ARMS)
        for a in ARMS:
            expected = dict(configs[ds], writeback=0. if a.endswith('reset') else configs[ds]['writeback'])
            checks += near(expected, r['configs'][a])
        byparent[p] = r
    assert set(byparent) == set(range(64)) and all(len(v) == 1 for v in sourcehashes.values())
    for r in rows:
        p = byparent[r['parent_id']]
        assert r['dataset'] == p['dataset'] and r['arrival'] == p['arrival'][r['order']]
        checks += 2
    return checks


def _public_path(checkout, relative, *, source=False):
    assert isinstance(relative, str) and '\\' not in relative
    p = Path(relative)
    assert not p.is_absolute() and '..' not in p.parts and '.' not in p.parts and p.as_posix() == relative
    if source:
        assert (relative in SCIENCE+REPORTING+SHARED_SCIENCE or
                re.fullmatch(r'vg_tta/stvg_opd_frozen_[a-z0-9_]*0[a-z0-9_]*\.py', relative)), ('source whitelist', relative)
    else:
        assert relative in {OUT_REL+'/'+n for n in SCALAR_FILES}
    target = (checkout/p).resolve()
    assert target.is_relative_to(checkout.resolve())
    return target


def _receipt_read(folder):
    receipt, original, files = {}, {}, {}
    for name in RECEIPTS:
        p = folder/'receipts'/name
        value = read(p)
        assert set(value) == {'original_private_receipt_sha256', 'projection'}
        original[name] = _digest(value['original_private_receipt_sha256'])
        assert isinstance(value['projection'], dict)
        privacy(value['projection'], 'receipt.'+name)
        def scalar_metadata(v):
            if isinstance(v, dict):
                for x in v.values():
                    scalar_metadata(x)
            else:
                assert not isinstance(v, (list, tuple)), ('receipt vector/container', name)
        scalar_metadata(value['projection'])
        receipt[name] = value['projection']
        files['receipts/'+name] = dict(sha256=sha(p), bytes=p.stat().st_size)
    return receipt, original, files


def receipt_contract(receipts, original, public_hashes, rows=None):
    """Only declared completed-root metadata is inspected; no private path is opened."""
    assert set(receipts) == set(RECEIPTS) == set(original)
    for v in original.values():
        _digest(v)
    checks = 0
    runtime = receipts['RUNTIME.json']
    assert runtime['status'] == 'locked_before_new_GPU_GT' and runtime['GT_read'] is False
    assert runtime['contracts_sha256'] == original['CPU_CONTRACTS.json']
    _digest(runtime['design_sha256'])
    assert receipts['CPU_CONTRACTS.json']['status'] == 'pass'
    assert receipts['CPU_CONTRACTS.json']['CPU_only'] is True and receipts['CPU_CONTRACTS.json']['GT_read'] is False
    custody = receipts['SOURCE_STATE_CUSTODY_READBACK.json']
    assert custody['status'] == 'pass' and custody['CPU_only'] is True and custody['GT_read'] is False
    for k, v in {'old_joint_carry_complete_fits': 128, 'unique_inputs': 64,
                 'parent_sources': 64, 'source_state_coordinates': 3584,
                 'original_development_and_orders_exact': True}.items():
        checks += near(v, custody[k])
    for prefix, n, rounds, captures, controls, rootname in (
            ('QUALIFICATION', 64, 1600, 16, 16, 'ROOT_QUALIFICATION.json'),
            ('GLOBAL_PREDICTION', 512, 12800, 128, 128, 'ROOT_DEPLOYMENT_READBACK.json')):
        bn = prefix+'_BARRIER.json'
        b, root = receipts[bn], receipts[rootname]
        assert b['status'] == ('pass' if n == 64 else 'sealed') and b['GT_read'] is False
        assert b['runtime_sha256'] == original['RUNTIME.json']
        for k, v in {'actual_complete_GPU_fits': n, 'actual_complete_fits': n,
                     'actual_complete_rounds': rounds, 'actual_frozen_captures': captures,
                     'original_joint_carry_full_controls': controls, 'actual_new_DINO_calls': 0}.items():
            checks += near(v, b[k])
        assert root['status'] == 'pass' and root['CPU_only'] is True and root['GT_read'] is False
        assert root['independent_full_decoder_Jacobian'] is False
        assert root['barrier_sha256'] == original[bn]
        counts = root['counts']
        for k, v in {'fits': n, 'rounds': rounds, 'old_complete_controls': controls,
                     'active_state_coordinates': n//2*(1792+1536)}.items():
            checks += near(v, counts[k])
        pairkey = 'whole_repeat_pairs' if n == 64 else 'qualified_formal_pairs'
        checks += near(32, counts[pairkey])
        assert _number(counts['scoped_Adam_scalar_checks']) and counts['scoped_Adam_scalar_checks'] > 0
        if n == 512 and rows is not None:
            checks += near(sum(r['empty'] for r in rows), counts.get('empty_fits', 0))
            checks += near(sum(r['gradient_calls'] for r in rows), counts.get('actual_backward_steps', 0))
    exposure = receipts['GT_EXPOSURE.json']
    assert exposure['status'] == 'actual_GT_only_after_all512_complete_fits_and_label_free_root'
    assert exposure['barrier_sha256'] == original['GLOBAL_PREDICTION_BARRIER.json']
    assert exposure['root_deployment_readback_sha256'] == original['ROOT_DEPLOYMENT_READBACK.json']
    assert exposure['GT_for_selection'] is False and exposure['GT_for_update'] is False
    assert exposure['CPU_only'] is True and exposure['historically_exposed_development'] is True
    complete = receipts['CPU_COMPLETION.json']
    assert complete['status'] == 'scored_pending_independent_root_report_publication'
    assert complete['CPU_only'] is True and complete['GT_read'] is True
    assert complete['barrier_sha256'] == original['GLOBAL_PREDICTION_BARRIER.json']
    for k, v in {'rows': 512, 'parent_arm_rows': 256, 'complete_rounds': 12800}.items():
        checks += near(v, complete[k])
    assert complete['files'] == public_hashes
    for name in ('ROOT_ANONYMOUS_SCALAR_READBACK.json', 'ROOT_READBACK.json'):
        r = receipts[name]
        assert r['status'] == 'pass' and r['CPU_only'] is True and r['GT_read'] is True
        assert r['rows'] == 512 and r['parent_arm_rows'] == 256 and r['scalar_checks'] > 0
        assert r['barrier_sha256'] == original['GLOBAL_PREDICTION_BARRIER.json']
        assert r['deployment_readback_sha256'] == original['ROOT_DEPLOYMENT_READBACK.json']
        assert r['scoring_completion_sha256'] == original['CPU_COMPLETION.json']
        assert r['files'] == public_hashes
        assert r['independent_dense_geometry'] is True and r['independent_deployment_math'] is False
        # The scorer receipt honestly declares its shared summary function.
        assert r['summary_reuses_declared_pure_function'] is True
    assert receipts['ROOT_READBACK.json']['independent_deployment_math_in_separate_receipt'] is True
    chronology = ('CPU_CONTRACTS.json', 'RUNTIME.json', 'SOURCE_STATE_CUSTODY_READBACK.json',
                  'QUALIFICATION_BARRIER.json', 'ROOT_QUALIFICATION.json',
                  'GLOBAL_PREDICTION_BARRIER.json', 'ROOT_DEPLOYMENT_READBACK.json',
                  'GT_EXPOSURE.json', 'CPU_COMPLETION.json',
                  'ROOT_ANONYMOUS_SCALAR_READBACK.json', 'ROOT_READBACK.json')
    times = [receipts[n]['time'] for n in chronology]
    assert all(_number(v) and math.isfinite(v) for v in times)
    assert all(a <= b for a, b in zip(times, times[1:])), 'Broken preparation/seal/root/GT/scoring sequence'
    proof = receipts['ROOT_PUBLIC_SOURCE_BINDINGS_READBACK.json']
    assert proof['status'] == 'pass' and proof['CPU_only'] is True and proof['GT_read'] is False
    assert proof['time'] >= receipts['ROOT_READBACK.json']['time']
    assert set(SCIENCE+REPORTING) <= set(proof['hashes'])
    assert set(SCIENCE+SHARED_SCIENCE) <= set(runtime['pins'])
    assert proof['count'] == len(runtime['pins'])
    for n in SCIENCE:
        assert proof['hashes'][n] == runtime['pins'][n]
    return checks


def source_bindings(checkout, receipts):
    """Read only public source paths; never traverse private source/input hashes."""
    hashes = {}
    runtime = receipts['RUNTIME.json']['pins']
    reporting = receipts['ROOT_PUBLIC_SOURCE_BINDINGS_READBACK.json']['hashes']
    for name, h in {**runtime, **reporting}.items():
        _digest(h)
        if name in runtime and name in reporting:
            assert runtime[name] == reporting[name]
        path = _public_path(checkout, name, source=True)
        assert path.is_file() and sha(path) == h, ('source bytes differ', name)
        hashes[name] = h
    return dict(status='pass', public_source_files=len(hashes),
                runtime_pins=len(runtime), new_science_files=len(SCIENCE),
                code_sha256=hashes, private_inputs_independently_reread=False,
                private_fit_or_checkpoint_opened=False)


def _rows_read(path, expected):
    out = []
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            assert line.strip(), 'Empty JSONL line'
            out.append(_decode(line))
            assert len(out) <= expected, 'Unexpected extra public rows'
    assert len(out) == expected
    return out


def run(checkout):
    """Return JSON-compatible receipt; the caller, not this checker, owns writes."""
    checkout = Path(checkout).resolve()
    folder = checkout/OUT_REL
    receipts, original, files = _receipt_read(folder)  # Root metadata gate first.
    # These references only point to the three explicitly anonymous public files.
    bound = receipts['CPU_COMPLETION.json']['files']
    assert set(bound) == {OUT_REL+'/'+n for n in SCALAR_FILES}
    for name, h in bound.items():
        _digest(h)
        p = _public_path(checkout, name)
        assert p.is_file() and sha(p) == h, ('anonymous public bytes differ', name)
        files[p.name] = dict(sha256=h, bytes=p.stat().st_size)
    checks = receipt_contract(receipts, original, bound)
    sources = source_bindings(checkout, receipts)
    rows = _rows_read(folder/'ROWS.jsonl.gz', 512)
    checks += rows_check(rows)
    checks += receipt_contract(receipts, original, bound, rows)
    parents = _rows_read(folder/'PARENT_ROWS.jsonl.gz', 256)
    for p in parents:
        assert set(p) == PARENT_KEYS
        assert isinstance(p['compute'], dict) and set(p['compute']) <= COMPUTE_KEYS
    privacy(parents, 'PARENT_ROWS')
    checks += near(independent_parents(rows), parents, 'PARENT_ROWS')
    actual_summary = read(folder/'SUMMARY.json')
    privacy(actual_summary, 'SUMMARY')
    checks += near(independent_summary(rows), actual_summary, 'SUMMARY')
    protocol = read(folder/'ANONYMOUS_PROTOCOL.json')
    checks += protocol_check(protocol, rows)
    pp = folder/'ANONYMOUS_PROTOCOL.json'
    files[pp.name] = dict(sha256=sha(pp), bytes=pp.stat().st_size)
    return dict(status='pass', scope='anonymous512 rows/256 parent-arm means/84 fields/six contrasts/paired orders and each order/independent count bootstrap/root receipt chain/public source exact bytes',
                checks=checks, files=files, rows=512, parent_arm_rows=256, parents=64,
                complete_rounds=12800, fields=len(FIELDS), contrasts=6,
                bootstrap=BOOT, seed=SEED, primary='I1_After_v', secondary='I0_After_v',
                independent_parent_arithmetic=True, independent_bootstrap=True,
                summary_reuses_scorer=False, root_scalar_summary_shared_function_declared=True,
                source_binding=sources, original_root_receipt_sha256=original,
                private_root_witnesses_reverified=False,
                annotation_or_prediction_geometry_recomputed=False,
                historically_exposed_development=True, independent_confirmation=False,
                automatic_promotion=False, equivalence_claim=False,
                CPU_only=True, GT_read=False, model_calls=0, GPU_calls=0)


def _synthetic_rows():
    rows = []
    for p in range(64):
        ds, K = DATASETS[p//32], (40 if p < 32 else 10)
        for oi, order in enumerate(ORDERS):
            at = p % 32 if oi == 0 else 31-p % 32
            for ai, arm in enumerate(ARMS):
                empty = int(p % 11 == 0)
                r = dict(index=p, parent_id=p, dataset=ds, order=order, arrival=at,
                         arm=arm, historically_exposed=True,
                         active_parameters=1536 if arm.startswith('LN_') else 1792,
                         gradient_calls=0 if empty else K-2, empty=empty, rounds=K,
                         no_information_rounds=K if empty else 2, observed_frames=0 if empty else 4,
                         compute=dict(instrumented_fit_seconds=1.+p/100, frozen_capture_seconds=.2 if ai == 0 else 0.,
                                      actual_new_DINO_calls=0, actual_frozen_capture_forwards=int(ai == 0),
                                      decoded_frame_reuse=int(ai == 0 and oi == 1), capture_owner=int(ai == 0),
                                      CUDA_peak_allocated=1024, actual_active_parameters=1536 if arm.startswith('LN_') else 1792))
                for iv in READOUTS:
                    offset = .02 if iv == 'I1' else 0.
                    fv = (.3, .5, .08, .64)[p % 4]+offset
                    bv = fv if arm.endswith('reset') or at == 0 else fv+(ai-1)*.01
                    av = bv if empty else bv+(ai-1.5)*.02+(.005 if oi else 0.)
                    fs = .35+p/200
                    bs = fs if arm.endswith('reset') or at == 0 else fs+(ai-1)*.005
                    ass = bs if empty else bs+(ai-1.5)*.01
                    t = .6+offset
                    for st, v, s in zip(STATES, (fv, bv, av), (fs, bs, ass)):
                        for m, val in (('v', v), ('s', s), ('t', t), ('R30', float(v > .3)), ('R50', float(v > .5))):
                            r[f'{iv}_{st}_{m}'] = val
                    for m in METRICS:
                        f, b, a = [r[f'{iv}_{st}_{m}'] for st in STATES]
                        for part, v in (('total', a-f), ('current', a-b), ('inherited', b-f)):
                            r[f'{iv}_delta_{part}_{m}'] = v
                    for origin in ('Frozen', 'Before'):
                        for th in (30, 50):
                            pre, post = bool(r[f'{iv}_{origin}_R{th}']), bool(r[f'{iv}_After_R{th}'])
                            for kind, val in (('correct_to_wrong', pre and not post),
                                              ('wrong_to_correct', not pre and post), ('correct_retained', pre and post)):
                                r[f'{iv}_{origin}_{kind}_R{th}'] = float(val)
                rows.append(r)
    return rows


def _synthetic_receipts(rows):
    """Public projection fixtures, never private files or scientific witnesses."""
    h = lambda s: hashlib.sha256(s.encode()).hexdigest()
    original = {n: h('synthetic-original-'+n) for n in RECEIPTS}
    public = {OUT_REL+'/'+n: h('synthetic-public-'+n) for n in SCALAR_FILES}
    pins = {n: h('synthetic-source-'+n) for n in SCIENCE+SHARED_SCIENCE}
    hashes = {n: h('synthetic-source-'+n) for n in SCIENCE+REPORTING}
    result = dict.fromkeys(RECEIPTS)
    result['RUNTIME.json'] = dict(status='locked_before_new_GPU_GT', GT_read=False,
        pins=pins, contracts_sha256=original['CPU_CONTRACTS.json'], design_sha256=h('synthetic-design'), time=2.)
    result['CPU_CONTRACTS.json'] = dict(status='pass', CPU_only=True, GT_read=False, time=1.)
    result['SOURCE_STATE_CUSTODY_READBACK.json'] = dict(status='pass', CPU_only=True, GT_read=False,
        old_joint_carry_complete_fits=128, unique_inputs=64, parent_sources=64,
        source_state_coordinates=3584, original_development_and_orders_exact=True, time=3.)
    for prefix, n, rounds, captures, controls, rn, t in (
            ('QUALIFICATION', 64, 1600, 16, 16, 'ROOT_QUALIFICATION.json', 4.),
            ('GLOBAL_PREDICTION', 512, 12800, 128, 128, 'ROOT_DEPLOYMENT_READBACK.json', 6.)):
        bn = prefix+'_BARRIER.json'
        result[bn] = dict(status='pass' if n == 64 else 'sealed', GT_read=False,
            actual_complete_GPU_fits=n, actual_complete_fits=n, actual_complete_rounds=rounds,
            actual_frozen_captures=captures, original_joint_carry_full_controls=controls,
            actual_new_DINO_calls=0, runtime_sha256=original['RUNTIME.json'], time=t)
        counts = dict(fits=n, rounds=rounds, old_complete_controls=controls,
                      active_state_coordinates=n//2*(1792+1536), scoped_Adam_scalar_checks=1)
        counts['whole_repeat_pairs' if n == 64 else 'qualified_formal_pairs'] = 32
        if n == 512:
            counts.update(empty_fits=sum(r['empty'] for r in rows),
                          actual_backward_steps=sum(r['gradient_calls'] for r in rows))
        result[rn] = dict(status='pass', counts=counts, barrier_sha256=original[bn],
                         CPU_only=True, GT_read=False, independent_full_decoder_Jacobian=False, time=t+1.)
    result['GT_EXPOSURE.json'] = dict(status='actual_GT_only_after_all512_complete_fits_and_label_free_root',
        barrier_sha256=original['GLOBAL_PREDICTION_BARRIER.json'],
        root_deployment_readback_sha256=original['ROOT_DEPLOYMENT_READBACK.json'],
        historically_exposed_development=True, GT_for_selection=False, GT_for_update=False, CPU_only=True, time=8.)
    result['CPU_COMPLETION.json'] = dict(status='scored_pending_independent_root_report_publication',
        barrier_sha256=original['GLOBAL_PREDICTION_BARRIER.json'], rows=512, parent_arm_rows=256,
        complete_rounds=12800, files=public, CPU_only=True, GT_read=True, time=9.)
    for name in ('ROOT_READBACK.json', 'ROOT_ANONYMOUS_SCALAR_READBACK.json'):
        result[name] = dict(status='pass', barrier_sha256=original['GLOBAL_PREDICTION_BARRIER.json'],
            deployment_readback_sha256=original['ROOT_DEPLOYMENT_READBACK.json'],
            scoring_completion_sha256=original['CPU_COMPLETION.json'],
            files=public, scalar_checks=1, rows=512, parent_arm_rows=256,
            independent_dense_geometry=True, independent_deployment_math=False,
            summary_reuses_declared_pure_function=True, CPU_only=True, GT_read=True, time=10.)
    result['ROOT_READBACK.json']['independent_deployment_math_in_separate_receipt'] = True
    result['ROOT_PUBLIC_SOURCE_BINDINGS_READBACK.json'] = dict(status='pass', count=len(pins), hashes=hashes,
        CPU_only=True, GT_read=False, time=11.)
    return result, original, public


def _synthetic_publisher_protocol():
    """Exercise only the publisher's anonymous projection on invented records."""
    path = Path(__file__).with_name('publish_stvg_opd_p7a_component_v1.py')
    tree = ast.parse(path.read_text())
    fun = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
    branch = next(n for n in fun.body if isinstance(n, ast.If) and
                  isinstance(n.test, ast.UnaryOp) and isinstance(n.test.op, ast.Not) and
                  isinstance(n.test.operand, ast.Name) and n.test.operand.id == 'closing')
    loop = next(n for n in branch.body if isinstance(n, ast.For) and
                isinstance(n.iter, ast.Subscript) and isinstance(n.iter.value, ast.Name) and
                n.iter.value.id == 'd' and isinstance(n.iter.slice, ast.Constant) and
                n.iter.slice.value == 'records')
    put = next(n for n in branch.body if isinstance(n, ast.Expr) and
               isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name) and
               n.value.func.id == 'put' and isinstance(n.value.args[0], ast.BinOp) and
               isinstance(n.value.args[0].right, ast.Constant) and
               n.value.args[0].right.value == 'ANONYMOUS_PROTOCOL.json')
    h = lambda s: hashlib.sha256(s.encode()).hexdigest()
    configs = {'hc2': dict(lr=.01, sigma=.025, tau=.05, steps=40, writeback=.0625, samples=32),
               'vidstg': dict(lr=.03, sigma=.1, tau=.25, steps=10, writeback=.125, samples=32)}
    stages = {ds: dict(source='vidstg' if ds == 'hc2' else 'hcstvg2',
                      configs={a: dict(configs[ds], writeback=0. if a.endswith('reset') else configs[ds]['writeback'])
                               for a in ARMS}) for ds in DATASETS}
    records = [dict(parent_id=p, dataset=DATASETS[p//32],
                    source_model_sha256=h('source'+DATASETS[p//32]), pixel_sha256=h('pixel'+str(p)),
                    frame_ids_sha256=h('ids'+str(p)), evidence_sha256=h('evidence'+str(p)),
                    input=dict(input_sha256=h('input'+str(p))),
                    originals={o: dict(sha256=h(o+str(p)), arrival=p % 32 if o == 'order1' else 31-p % 32)
                               for o in ORDERS}) for p in range(64)]
    captured = []
    namespace = dict(d=dict(stages=stages, records=records), records=[],
                     OUT=Path('/synthetic/public'), ARMS=ARMS, BOOT=BOOT, SEED=SEED,
                     put=lambda unused, value: captured.append(value))
    # No publisher import, read(), copy(), scientific verify(), or git operation.
    exec(compile(ast.Module(body=[loop, put], type_ignores=[]), '<synthetic anonymous projection>', 'exec'), namespace)
    assert len(captured) == 1
    return captured[0]


def selftest():
    """Synthetic-only contract, optionally compared with AST-isolated pure scorer."""
    import copy
    rows = _synthetic_rows()
    checks = rows_check(rows)
    parent = independent_parents(rows)
    assert len(parent) == 256
    own = independent_summary(rows)
    assert own['complete_rounds'] == 12800 and len(FIELDS) == 84
    e = independent_estimate([.06, -.21, 0., None])
    assert (e['positive'], e['negative'], e['neutral'], e['unknown'], e['harm_gt20pp']) == (1, 1, 1, 1, 1)
    assert independent_estimate([-.05, -.20, .05, .20])['harm_gt5pp'] == 1
    assert independent_estimate([None, None])['mean'] is None
    assert independent_estimate([1., 1.])['ci95'] == [1., 1.]
    receipts, original, public = _synthetic_receipts(rows)
    privacy(receipts)
    checks += receipt_contract(receipts, original, public, rows)
    protocol = _synthetic_publisher_protocol()
    checks += protocol_check(protocol, rows)
    # Missing one order must invalidate the parent, not average only known orders.
    missing = copy.deepcopy(rows)
    missing[0]['I1_After_v'] = None
    assert next(p for p in independent_parents(missing) if p['parent_id'] == 0 and p['arm'] == ARMS[0])['I1_After_v'] is None
    # Checking the existing producer's pure functions does not import its common,
    # torch, label loaders or scientific runner.  Only named pure definitions run.
    path = Path(__file__).with_name('score_stvg_opd_p7a_component_v1.py')
    producer = False
    if path.is_file():
        funcs = {'_number', 'public_rows_check', 'estimate', 'parent_rows', '_block', 'summary'}
        tree = ast.parse(path.read_text())
        selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in funcs]
        assert {n.name for n in selected} == funcs
        ns = dict(np=np, math=math, collections=collections, ARMS=ARMS, BOOT=BOOT, SEED=SEED, EPS=EPS,
                  DATASETS=DATASETS, ORDERS=ORDERS, READOUTS=READOUTS, STATES=STATES,
                  METRICS=METRICS, CONTRASTS=CONTRASTS, FIELDS=FIELDS,
                  COST_FIELDS=COST_FIELDS, PUBLIC_KEYS=ROW_KEYS)
        exec(compile(ast.Module(body=selected, type_ignores=[]), '<isolated anonymous pure functions>', 'exec'), ns)
        checks += near(ns['parent_rows'](rows), parent)
        checks += near(ns['summary'](rows), own)
        checks += near(ns['summary'](missing), independent_summary(missing))
        producer = True
    tamper = 0
    def reject(action):
        nonlocal tamper
        try:
            action()
        except (AssertionError, KeyError):
            tamper += 1
        else:
            raise AssertionError('Synthetic tamper accepted')
    bad = copy.deepcopy(rows); bad[0]['I0_Frozen_R30'] = 1.
    reject(lambda: rows_check(bad))  # Native .3 uses strict > .3.
    bad = copy.deepcopy(rows); bad[0]['query_ordinal'] = 7
    reject(lambda: rows_check(bad))
    bad = copy.deepcopy(rows); bad[0]['compute']['box_area'] = 1.
    reject(lambda: rows_check(bad))
    bad = copy.deepcopy(rows); bad[0]['arrival'] = 1
    reject(lambda: rows_check(bad))
    reject(lambda: _decode('{"v":1,"v":2}'))
    reject(lambda: _decode('{"v":NaN}'))
    reject(lambda: _public_path(Path('/tmp'), '../weights/model.pt', source=True))
    reject(lambda: privacy({'boxes': [1., 2., 3., 4.]}))
    bad = copy.deepcopy(own); bad['pooled']['fields']['I1_After_v']['contrasts']['remove_both']['mean'] += .001
    reject(lambda: near(own, bad))
    bad = copy.deepcopy(receipts); bad['ROOT_DEPLOYMENT_READBACK.json']['barrier_sha256'] = '0'*64
    reject(lambda: receipt_contract(bad, original, public, rows))
    bad = copy.deepcopy(receipts); bad['GT_EXPOSURE.json']['time'] = 6.5
    reject(lambda: receipt_contract(bad, original, public, rows))
    bad = copy.deepcopy(receipts); bad['ROOT_DEPLOYMENT_READBACK.json']['counts']['old_complete_controls'] = 127
    reject(lambda: receipt_contract(bad, original, public, rows))
    bad = copy.deepcopy(receipts); bad['CPU_COMPLETION.json']['files'] = dict(public, **{'private/00000.pt': '0'*64})
    reject(lambda: receipt_contract(bad, original, public, rows))
    bad = copy.deepcopy(receipts); bad['ROOT_ANONYMOUS_SCALAR_READBACK.json']['summary_reuses_declared_pure_function'] = False
    reject(lambda: receipt_contract(bad, original, public, rows))
    bad = copy.deepcopy(receipts); del bad['RUNTIME.json']['pins'][SHARED_SCIENCE[0]]
    reject(lambda: receipt_contract(bad, original, public, rows))
    reject(lambda: _public_path(Path('/tmp'), 'methods/annotations/labels.json', source=True))
    bad = copy.deepcopy(protocol); bad['records'][0]['configs']['LN_carry']['tau'] = .051
    reject(lambda: protocol_check(bad, rows))
    bad = copy.deepcopy(protocol); bad['records'][0]['arrival']['order1'] = 1
    reject(lambda: protocol_check(bad, rows))
    bad = copy.deepcopy(protocol); bad['records'][0]['query_ordinal'] = 9
    reject(lambda: protocol_check(bad, rows))
    return dict(status='pass', synthetic_only=True, rows=512, parent_arm_rows=256,
                fields=84, contrasts=6, complete_rounds=12800, scalar_checks=checks,
                tamper_rejections=tamper, existing_pure_scorer_comparison=producer,
                independent_bootstrap=True, completed_receipt_binding_contract=True, actual_GT_read=False,
                synthetic_publisher_protocol_comparison=True,
                actual_fit_read=False, model_calls=0, GPU_calls=0, writes=0)


if __name__ == '__main__':
    assert len(sys.argv) == 2, 'Usage: checker.py <public-checkout> | --selftest'
    print(json.dumps(selftest() if sys.argv[1] == '--selftest' else run(sys.argv[1]),
                     sort_keys=True, ensure_ascii=False, allow_nan=False))
