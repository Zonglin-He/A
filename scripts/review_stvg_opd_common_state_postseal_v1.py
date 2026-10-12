"""Postseal root review; no model calls, no reads of unsealed results."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse, collections, gzip, json, math, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.decota_matrix_common_v1 import read, write, status, sha, load
BASE = ROOT / 'artifacts/stvg_opd_optimization_geometry_v1'
REC = BASE / 'postseal_independent_root001'
PUB = ROOT / 'results/stvg_opd_optimization_geometry/2026-10-12/common_state'
ARMS = ['Full', 'Direct', 'Direct_capped']
GROUPS = ['observed', 'unobserved_decoder', 'interpolated', 'outside_native']


def require_seal(base=BASE):
    p = base / 'GLOBAL_PREDICTION_BARRIER.json'
    assert p.exists(), 'No complete global seal; refuse result/GT access'
    b = read(p)
    assert b['status'] == 'sealed' and b['logical_fits'] == 5376
    assert not b['GT_read'] and len(b['files']) == 5376
    p = base / 'COMMON_STATE_CPU_COMPLETION.json'
    assert p.exists(), 'Original actual postseal CPU scorer has not completed'
    c = read(p)
    assert c['cells'] == 1792 and c['complete_fits'] == 5376 and c['GT_read']
    return b, c


def parent_statistics(rows, field):
    """Independent parent equal weighting, preserving row encounter order."""
    parents = {}
    for row in rows:
        if row.get(field) is not None:
            parents.setdefault(row['source_id'], []).append(float(row[field]))
    x = np.asarray([math.fsum(v) / len(v) for v in parents.values()])
    if not len(x):
        return dict(parents=0, mean=None, ci95=None)
    rng = np.random.default_rng(20261012)
    bootstrap = np.empty(10000)
    for lo in range(0, 10000, 100):
        ix = rng.integers(len(x), size=(100, len(x)))
        bootstrap[lo:lo+100] = np.sum(x[ix], axis=1) / len(x)
    return dict(parents=len(x), mean=math.fsum(x)/len(x),
                ci95=np.percentile(bootstrap, [2.5, 97.5]).tolist(),
                negative_parent_tail_5pp=int(np.count_nonzero(x < -.05)),
                positive_parent_tail_5pp=int(np.count_nonzero(x > .05)),
                p05=float(np.percentile(x, 5)))


def scalar_equal(a, b, path=''):
    if isinstance(a, dict):
        assert set(a) == set(b), path
        return sum(scalar_equal(a[k], b[k], path+'/'+k) for k in a)
    if isinstance(a, list):
        assert len(a) == len(b), path
        return sum(scalar_equal(x, y, path) for x, y in zip(a, b))
    if a is None or isinstance(a, (str, bool, int)):
        assert a == b, (path, a, b)
    else:
        assert abs(a-b) <= 2e-12*(1+abs(a)), (path, a, b)
    return 1


def contracts():
    rr = [dict(source_id=1, x=0.), dict(source_id=1, x=1.),
          dict(source_id=2, x=1.), dict(source_id=3, x=None)]
    s = parent_statistics(rr, 'x')
    assert s['parents'] == 2 and s['mean'] == .75
    assert parent_statistics([dict(source_id=1, x=None)], 'x')['mean'] is None
    assert parent_statistics(rr, 'x') == s
    assert scalar_equal(s, dict(s)) == 7
    caught = False
    try:
        scalar_equal(dict(x=None), dict(x=0.))
    except AssertionError:
        caught = True
    assert caught
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        try:
            require_seal(Path(tmp))
        except AssertionError:
            pass
        else:
            raise AssertionError('Unsealed gate accepted')
    return dict(status='pass', scope='synthetic aggregation/undefined/repeatability and preseal refusal only',
                valid_contracts=6, CPU_only=True, GT_read=False, new_model_calls=0,
                actual_formal_review_complete=False, time=time.time())


def run():
    barrier, completion = require_seal()
    runtime = read(REC/'RUNTIME.json')
    for p, h in runtime['pins'].items():
        assert sha(ROOT/p) == h, p
    gpu = read(BASE/'GPU_RUNTIME.json')
    for p, h in gpu['pins'].items():
        assert sha(ROOT/p) == h, p
    for p, h in completion['files'].items():
        assert sha(ROOT/p) == h, p
    assert sha(BASE/'COMMON_STATE_COHORT.json') == gpu['cohort_sha256']
    import torch
    torch.set_num_threads(2)
    from scripts.stvg_opd_design_ablation_common_v1 import activate_old, audit
    activate_old()
    from scripts.run_stvg_opd_p1_softmax_recovery_002 import equal
    from vg_tta.stvg_opd_common_state_geometry_arithmetic004 import bare, mathematical_readback
    from scripts.stvg_opd_paper_common_v1 import committed
    from scripts.stvg_opd_paper_later_common_v1 import stage_definition
    from scripts.run_decota_paper_main_v1 import read_row, unpack_expert
    from scripts.stvg_opd_legacy_input_bridge003 import normalized
    from scripts.score_stvg_opd_p1_v1 import truths
    from scripts.review_stvg_opd_revised_p1_v2 import dense_metric
    from scripts.analyze_stvg_opd_mechanism_review_v1 import partition
    cohort = read(BASE/'COMMON_STATE_COHORT.json')['records']
    with gzip.open(PUB/'ROWS.jsonl.gz', 'rt') as f:
        rows = [json.loads(line) for line in f]
    assert len(rows) == len(cohort) == 1792
    count = collections.Counter()
    prior, origin, source, gt, spans = {}, {}, None, None, None
    tick = time.time()
    for r, row in zip(cohort, rows):
        for k in ['dataset', 'source_id', 'stage', 'condition', 'order', 'arrival', 'query_ordinal', 'setting', 'development']:
            assert r[k] == row[k], k
        binding = r['original_Full']
        assert sha(ROOT/binding['path']) == binding['sha256']
        assert sha(ROOT/binding['input_path']) == binding['input_sha256']
        old = load(ROOT/binding['path'])
        inp = load(ROOT/binding['input_path'])
        assert not inp['GT_read'] and old['interval'] == inp['interval']
        inp = normalized(inp)
        expert = unpack_expert(inp['expert'])
        group = (r['dataset'], r['setting'], r['condition'], r['order'])
        initial = old['fit']['initial']
        assert torch.count_nonzero(initial['spatial.query_residual']) == 0
        if group in prior:
            prev, h = prior[group]
            assert old['previous_payload_sha256'] == h
            assert all(torch.equal(v, torch.zeros_like(v) if n == 'spatial.query_residual' else prev[n]) for n, v in initial.items())
        else:
            assert r['arrival'] == 0 and old['previous_payload_sha256'] is None
            if r['dataset'] in origin:
                assert all(torch.equal(v, origin[r['dataset']][n]) for n, v in initial.items())
            else:
                origin[r['dataset']] = {n: v.clone() for n, v in initial.items()}
        expected = committed(initial, old['fit']['state'], old['fit']['config']['writeback'])
        assert all(torch.equal(expected[n], old['committed'][n]) for n in expected)
        prior[group] = (old['committed'], binding['sha256'])
        fits = {}
        for arm in ARMS:
            p = BASE/'common_state'/f"{r['index']:05}"/(arm+'.pt')
            rel = str(p.relative_to(ROOT)); rc = read(p.with_suffix('.json'))
            h = sha(p)
            assert h == barrier['files'][rel] == rc['sha256']
            assert p.stat().st_size == rc['bytes'] and not rc['GT_read']
            assert rc['time'] <= barrier['time']
            z = load(p); fit = z['fit']; fits[arm] = fit
            assert z['index'] == r['index'] and z['arm'] == arm and not z['GT_read']
            assert z['input_sha256'] == binding['input_sha256']
            assert z['common_initial_original_Full_sha256'] == binding['sha256']
            assert z['runtime_sha256'] == sha(BASE/'GPU_RUNTIME.json') and z['Adam_reset']
            assert z['interval'] == old['interval'] and fit['config'] == old['fit']['config']
            assert torch.equal(fit['before'], old['fit']['before'])
            assert fit['positions'] == old['fit']['positions']
            assert all(torch.equal(fit['initial'][n], v) for n, v in initial.items())
            m = mathematical_readback(fit)
            assert m == z['math_audit']['independent']
            if arm != 'Direct_capped':
                assert audit(fit, expert) == z['math_audit']['original_component']
                count['complete_objective_math_dictionaries'] += 1
            count['math_scalar_checks'] += m['scalar_checks']
            count['state_coordinates'] += sum(v.numel() for v in fit['state'].values())
            count['complete_rounds'] += len(fit['rounds'])
            counts = dict(tensors=0, tensor_coordinates=0, scalar_values=0)
            if arm == 'Full':
                equal(bare(fit), bare(old['fit']), counts)
                for k, geom in enumerate(fit['geometry']['rounds']):
                    if 'joint_cosine' not in geom:
                        continue
                    a = geom['OPD_parameter_gradient'].numpy().astype(float)
                    b = geom['Direct_parameter_gradient'].numpy().astype(float)
                    for label, aa, bb in [('joint', a, b), ('query', a[:256], b[:256]), ('LN', a[256:], b[256:])]:
                        den = np.linalg.norm(aa)*np.linalg.norm(bb)
                        value = float(np.dot(aa, bb)/den) if den else None
                        scalar_equal(value, geom[label+'_cosine'], label)
                        count['parameter_geometry_checks'] += 1
                    scalar_equal(float(np.linalg.norm(a)), geom['OPD_norm'])
                    scalar_equal(float(np.linalg.norm(b)), geom['Direct_norm'])
            if r['qualify']:
                q = load(BASE/'qualification'/f"{r['index']:05}"/(arm+'.pt'))
                equal(bare(fit), bare(q['fit']), counts)
                equal(fit['geometry'], q['fit']['geometry'], counts)
                count['qualified_formal_pairs'] += 1
            if arm == 'Direct_capped':
                for k, path in enumerate(fit['path'][:-1]):
                    u = path.get('update')
                    if u is None:
                        continue
                    cap = u['output_cap']
                    ref = old['fit']['path']
                    radius = float(np.linalg.norm(ref[k+1]['boxes'].numpy().astype(float)-ref[k]['boxes'].numpy().astype(float)))
                    scalar_equal(radius, cap['reference_radius'])
                    assert cap['Adam_moments_not_scaled'] and cap['cap_satisfied']
                    count['reference_cap_checks'] += 1
            count['complete_fits'] += 1
        canonical = 'P2_'+r['dataset']+('_cross_clean' if r['setting'].endswith('/cross_clean') else '_same_5percent')
        if source != canonical:
            source = canonical; gt, spans, provenance = truths(r['dataset'], stage_definition(source))
        raw = read_row(r['dataset'], r['query_ordinal']); q = r['query_ordinal']
        assert inp['frame_ids'] == raw['frame_ids']
        curves = {}
        for arm, boxes in [('Before', fits['Full']['before'])]+[(a, fits[a]['final']) for a in ARMS]:
            values, fids, ious = dense_metric(boxes.numpy(), raw, gt[q], spans[q], old['interval'], r['dataset'])
            for k, value in values.items():
                scalar_equal(value, row[arm+'_'+k]); count['dense_scalar_checks'] += 1
            curves[arm] = ious
        masks = partition(fids, raw['frame_ids'], [raw['frame_ids'][p] for p in fits['Full']['positions']], old['interval'])
        event = (fids >= spans[q][0]) & (fids < spans[q][1])
        for arm in ARMS:
            scalar_equal(row[arm+'_v']-row['Before_v'], row[arm+'_current_v'])
            for group_name, mask in masks.items():
                values = (curves[arm]-curves['Before'])[mask & event]
                value = float(np.mean(values)) if len(values) else None
                scalar_equal(value, row[arm+'_'+group_name+'_current_IoU'])
                count['propagation_scalar_checks'] += 1
        for a, b in [('Full', 'Direct'), ('Full', 'Direct_capped'), ('Direct_capped', 'Direct')]:
            scalar_equal(row[a+'_v']-row[b+'_v'], row[a+'_minus_'+b+'_v'])
        cosines = [g['joint_cosine'] for g in fits['Full']['geometry']['rounds'] if g.get('joint_cosine') is not None]
        scalar_equal(math.fsum(cosines)/len(cosines) if cosines else None, row['common_parameter_gradient_cosine'])
        caps = [p['update']['output_cap'] for p in fits['Direct_capped']['path'][:-1] if p.get('update')]
        ratios = [c['budget_utilization'] for c in caps if c['budget_utilization'] is not None]
        scalar_equal(math.fsum(ratios)/len(ratios) if ratios else None, row['Direct_capped_budget_utilization'])
        assert sum(c['extra_native_forwards'] for c in caps) == fits['Direct_capped']['geometry']['extra_native_forwards'] == row['Direct_capped_extra_forwards']
        count['matched_inputs'] += 1
        if r['index'] % 32 == 0:
            status(REC/'STATUS.json', dict(status='actual_full_source_state_dense_review', done=r['index']+1,
                   total=1792, CPU_only=True, GT_read=True, pid=os.getpid(), time=time.time()))
            print('INDEPENDENT_COMMON_STATE_ROOT', r['index']+1, 1792, flush=True)
    assert count['complete_fits'] == 5376 and count['matched_inputs'] == 1792
    assert count['qualified_formal_pairs'] == 84
    summary = read(PUB/'SUMMARY.json')
    recomputed = {}; checks = 0
    for key, block in summary['settings'].items():
        setting, pop = key.rsplit('/', 1)
        selected = [r for r in rows if r['setting'] == setting and (pop == 'all_historical' or not r['development'])]
        assert len(selected) == block['cells']
        output = dict(cells=len(selected), statistics={}, per_order={})
        for field, old_stats in block['statistics'].items():
            s = parent_statistics(selected, field); checks += scalar_equal(s, old_stats, key+'/'+field)
            output['statistics'][field] = s
        for order, old_stats in block['per_order'].items():
            ordered = [r for r in selected if r['order'] == order]
            output['per_order'][order] = {}
            for field, s in old_stats.items():
                result = parent_statistics(ordered, field); checks += scalar_equal(result, s)
                output['per_order'][order][field] = result
        output['propagation'] = {a+'_'+g+'_current_IoU': parent_statistics(selected, a+'_'+g+'_current_IoU') for a in ARMS for g in GROUPS}
        recomputed[key] = output
    # Re-read all bytes after numerical/GT review. No source rewriting on success.
    for r in cohort:
        for p, h in [(r['original_Full']['path'], r['original_Full']['sha256']), (r['original_Full']['input_path'], r['original_Full']['input_sha256'])]:
            assert sha(ROOT/p) == h; count['second_read_hashes'] += 1
    for p, h in barrier['files'].items():
        assert sha(ROOT/p) == h; count['second_read_hashes'] += 1
    write(PUB/'INDEPENDENT_ROOT_STATISTICS.json', dict(status='pass', settings=recomputed,
          bootstrap=10000, seed=20261012, conditional_on_original_Full_history=True,
          historically_exposed=True, unadjusted_CI=True, no_independent_online_stream_claim=True))
    write(REC/'ROOT_READBACK.json', dict(status='pass', scope='all common-state source/input/math/state/Full-bitwise/qualified-formal/dense/propagation/parent paired statistics',
          counts=dict(count), independent_statistics_checks=checks, CPU_only=True, GT_read=True,
          new_model_calls=0, independent_full_decoder_Jacobian=False, actual_PDF_view_complete=False,
          task_complete=False, elapsed_seconds=time.time()-tick, time=time.time()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--contracts', action='store_true')
    args = parser.parse_args()
    if args.contracts:
        print(json.dumps(contracts()))
    else:
        run()
