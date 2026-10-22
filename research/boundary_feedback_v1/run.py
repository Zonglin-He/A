"""Locked HC64 cached boundary feedback study. Never resumes the full queue."""
import os, sys, json, hashlib, time
from pathlib import Path
R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R))
sys.path.insert(0, str(Path(__file__).parent))
D = R/'outputs/boundary_feedback_hc64_v1'
P = R/'results/boundary_feedback_hc64_v1'
OLD = R/'outputs/event_query_hc64_v1'
B = R.parent/'outputs/tube_temporal_hc64'
ARMS = ['NativeG6', 'BoundaryCell', 'BoundaryGlobal', 'ShiftCell', 'RawCell']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def put(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    os.replace(tmp, path)


def records():
    return [r for r in map(json.loads, (R/'data/evaluation/development.jsonl').read_text().splitlines())
            if r['dataset'] == 'hc2'][:64]


def verify():
    p = read(D/'plan.json')
    assert read(R/'outputs/tastvg_paper_v1/state.json')['status'] == 'paused_by_user'
    assert sha(R/'outputs/tastvg_paper_v1/plan.json') == p['paper_plan_sha256']
    for name, digest in p['code'].items():
        assert sha(R/name) == digest, name
    for name, digest in p['paper_code'].items():
        assert sha(R/name) == digest, name
    for name, digest in p['provider_sources'].items():
        assert sha(R.parent/'stvg-opd'/name) == digest, name
    assert sha(OLD/'sealed.json') == p['prior_seal_sha256']
    return p


def prepare():
    assert not (D/'plan.json').exists()
    old = read(OLD/'plan.json'); seal = read(OLD/'sealed.json')
    assert seal['count'] == 64 and seal['GT_read'] is False
    bindings = {}
    for row in records():
        key = row['id']; e = seal['files'][key]
        assert sha(OLD/e['path']) == e['sha256']
        assert sha(OLD/e['trace_path']) == e['trace_sha256']
        assert sha(B/'captures'/f'{key}.pt') == old['captures'][key]['capture_sha256']
        bindings[key] = dict(capture_sha256=old['captures'][key]['capture_sha256'],
                             prior_prediction=e, metadata_sha256=hashlib.sha256(
                                 json.dumps(row, sort_keys=True).encode()).hexdigest())
    paper = read(R/'outputs/tastvg_paper_v1/plan.json')
    code = {str(f.relative_to(R)): sha(f) for f in Path(__file__).parent.glob('*.py')}
    for name in ['stvg/temporal.py', 'stvg/spatial.py', 'stvg/_spatial_math.py',
                 'research/event_query_v1/score.py', 'research/event_query_v1/run.py']:
        code[name] = sha(R/name)
    p = dict(schema='boundary_feedback_hc64_v1', count=64, arms=ARMS,
             primary='BoundaryCell', exposure='same previously exposed HC64 development; not independent',
             hypothesis='dense text-conditioned hidden transitions distinguish boundaries inside identical eight-anchor evidence patterns',
             fixed=dict(window_per_offset=2, layers='all6 equal', offsets='separate contrast then physical interpolation mean',
                        score='centered percentile rank of two-sided mean-embedding cosine contrast; start+end',
                        posterior='q0*exp(score), preserve each native eight-anchor-membership mass for Cell arms',
                        shift='cyclic shift of dense start/end curves by floor(T/4) positions, same values/budget',
                        RawCell='old raw-event contrast with identical mass preservation; no new semantic extraction'),
             scientific_controls='BoundaryGlobal removes mass preservation; ShiftCell removes correct temporal location; RawCell tests protection alone',
             budget=dict(new_model_forwards=0, new_expert_calls=0, query_steps=0, maximum_CPU_seconds=600),
             labels_sha256=old['labels_sha256'], bindings=bindings, code=code,
             paper_code=paper['code'], provider_sources=paper['provider_sources'], paper_plan_sha256=sha(R/'outputs/tastvg_paper_v1/plan.json'),
             prior_seal_sha256=sha(OLD/'sealed.json'), spatial='unchanged complete-box geometry and original interval-conditioned exact DP',
             decode='original physical MBR, all legal intervals; no truncation or GT rule',
             all64_predictions_before_new_GT=True, full_status='paused_by_user', seed=20261022,
             references=['https://arxiv.org/abs/2111.14799', 'https://github.com/musicalOffering/uboco',
                         'https://arxiv.org/html/2609.08850v1'],
             literature_boundary='Uses local self-similarity principle only, not a UBoCo or DSE reproduction. UBoCo trains its feature encoder; no such training here. Generic changes may not identify described events.')
    p['fingerprint'] = hashlib.sha256(json.dumps(p, sort_keys=True).encode()).hexdigest()
    put(D/'plan.json', p)
    put(P/'protocol.json', {k:v for k,v in p.items() if k not in ['bindings', 'paper_code', 'provider_sources']})
    for name in code:
        f = D/'source_snapshot'/name; f.parent.mkdir(parents=True, exist_ok=True); f.write_bytes((R/name).read_bytes())
    put(D/'state.json', dict(status='prepared', fingerprint=p['fingerprint']))
    print(json.dumps(dict(status='prepared', fingerprint=p['fingerprint'])))


def predict():
    import torch, numpy as np
    from core import boundary_curve, groups_for, reweight
    from stvg.spatial import SpatialGeometry, evidence_dp
    from stvg.temporal import final_mbr
    p = verify(); torch.set_num_threads(1)
    assert not torch.cuda.is_initialized() and not (D/'sealed.json').exists()
    def forbid_labels(event,args):
        if event == 'open' and isinstance(args[0],(str,bytes,os.PathLike)):
            name=os.fsdecode(args[0]).lower()
            if 'labels' in name or 'annotation' in name:
                raise PermissionError('Labels/annotations forbidden during prediction: '+name)
    sys.addaudithook(forbid_labels)
    put(D/'state.json', dict(status='predicting_CPU', fingerprint=p['fingerprint'], pid=os.getpid()))
    files = {}; start = time.perf_counter()
    for row in records():
        key = row['id']; binding = p['bindings'][key]; e = binding['prior_prediction']
        assert sha(B/'captures'/f'{key}.pt') == binding['capture_sha256']
        assert sha(OLD/e['trace_path']) == e['trace_sha256'] and sha(OLD/e['path']) == e['sha256']
        cap = torch.load(B/'captures'/f'{key}.pt', map_location='cpu', weights_only=False)
        tr = torch.load(OLD/e['trace_path'], map_location='cpu', weights_only=False)
        ids = cap['frame_ids']; pairs = tr['feedback']['pairs']; q0 = tr['q']['NativeG6']
        tick = time.perf_counter()
        curve, raw = boundary_curve(cap['temporal_cache']['offsets'], ids)
        patterns, group = groups_for(ids, cap['planned_positions'], pairs)
        score = curve[0, pairs[:,0]] + curve[1, pairs[:,1]]
        shifted = np.roll(curve, len(ids)//4, axis=1)
        shift_score = shifted[0, pairs[:,0]] + shifted[1, pairs[:,1]]
        qs = dict(NativeG6=q0, BoundaryCell=reweight(q0,pairs,score,group),
                  BoundaryGlobal=reweight(q0,pairs,score), ShiftCell=reweight(q0,pairs,shift_score,group),
                  RawCell=reweight(q0,pairs,tr['feedback']['raw'],group))
        feedback_seconds = time.perf_counter()-tick
        geom = SpatialGeometry.__new__(SpatialGeometry); geom.__dict__.update(cap['geometry'])
        arrays = {}; intervals = {}; mass_errors = {}
        for arm, q in qs.items():
            intervals[arm] = final_mbr(ids, q)['interval']
            arrays[arm] = evidence_dp(geom.for_interval(intervals[arm]))['final'].numpy()
            if arm.endswith('Cell'):
                mass_errors[arm] = float(np.max(abs(np.bincount(group, weights=q[pairs[:,0],pairs[:,1]])-
                                                         np.bincount(group, weights=q0[pairs[:,0],pairs[:,1]]))))
                assert mass_errors[arm] < 2e-13
        with np.load(OLD/e['path'], allow_pickle=False) as z:
            assert np.array_equal(arrays['NativeG6'], z['NativeG6'])
            assert np.array_equal(intervals['NativeG6'], z['NativeG6_interval'])
            arrays['Source'] = z['Source'].copy(); intervals['Source'] = z['Source_interval'].tolist()
        path = D/'predictions'/f'{key}.npz'; assert not path.exists(); path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays, **{a+'_interval':v for a,v in intervals.items()}, frame_ids=ids, positions8=cap['planned_positions'])
        trace = D/'traces'/f'{key}.pt'; trace.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(q=qs, curve=curve, raw=raw, score=score, groups=group, patterns=patterns,
                        pairs=pairs, mass_errors=mass_errors, feedback_seconds=feedback_seconds, GT_read=False), trace)
        files[key] = dict(path=str(path.relative_to(D)), sha256=sha(path), trace_path=str(trace.relative_to(D)),
                          trace_sha256=sha(trace), seconds=time.perf_counter()-tick, feedback_seconds=feedback_seconds)
        put(D/'progress.json',dict(completed=len(files),total=64,seconds=time.perf_counter()-start))
        assert time.perf_counter()-start < p['budget']['maximum_CPU_seconds']
        print(key, round(files[key]['seconds'],3), flush=True)
    verify()
    put(D/'sealed.json',dict(count=64,files=files,GT_read=False,CUDA_initialized=torch.cuda.is_initialized(),
                            plan_sha256=sha(D/'plan.json'),seconds=time.perf_counter()-start))
    put(D/'state.json',dict(status='sealed',fingerprint=p['fingerprint']))


if __name__ == '__main__':
    globals()[sys.argv[1]]()
