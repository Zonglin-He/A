"""Independent frozen Query/LN diagnostic namespace and CPU-only preparation."""
import collections, hashlib, json, math, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.decota_matrix_common_v1 import read, write, status, sha, save, load
BASE = ROOT/'artifacts/stvg_opd_query_observability_q0_v1'
PUB = ROOT/'results/stvg_opd_query_observability_q0/2026-10-12'
PYTHON = ROOT/'.conda/tubedetr/bin/python'
GROUPS = ['Q_random', 'Q_observable', 'LN_observable']
QUERY = 'spatial.query_residual'


def relative(p):
    return str(Path(p).relative_to(ROOT))


def canonical(v):
    import numpy as np
    v = np.asarray(v, dtype=float).copy()
    for row in v:
        if row[np.argmax(np.abs(row))] < 0:
            row *= -1
    return v


def spectrum(j):
    import numpy as np
    j = np.asarray(j, dtype=float)
    if not len(j):
        return dict(singular_values=[], rank=0, tolerance=0., directions=np.empty((0, j.shape[1])))
    u, s, vh = np.linalg.svd(j, full_matrices=False)
    tolerance = float(max(j.shape)*np.finfo(np.float32).eps*s[0])
    return dict(singular_values=s.tolist(), rank=int(np.count_nonzero(s > tolerance)),
                tolerance=tolerance, directions=canonical(vh))


def weights(rewards, tau):
    import numpy as np
    x = np.asarray(rewards, dtype=float)/tau
    e = np.exp(x-x.max()); return e/e.sum()


def contracts():
    import numpy as np
    s = spectrum(np.diag([3., 2., 0.]))
    assert s['rank'] == 2 and np.array_equal(s['directions'], np.eye(3))
    assert spectrum(np.empty((0, 256)))['rank'] == 0
    assert np.array_equal(weights([0., 0., 0.], .05), np.ones(3)/3)
    w = weights([.1, .2, .3], .25); x = np.array([[-1., 0.], [0., 2.], [1., 1.]])
    mean = (w[:, None]*x).sum(0)
    assert np.max(np.abs(sum(w[i]*(mean-x[i]) for i in range(3)))) < 1e-14
    assert int(np.argmax([1., 1., 1.])) == 0
    assert canonical([[-1., .1]])[0, 0] == 1.
    return dict(status='pass', valid_contracts=6, scope='synthetic SVD/rank/empty/softmax/mean/tie/sign only',
                CPU_only=True, GT_read=False, actual_GPU_qualification=False, time=time.time())


def prepare():
    import numpy as np
    sources = [dict(dataset='hc2', source='vidstg', base=ROOT/'artifacts/stvg_opd_hc2_coordinate_v1',
                    trial='trials/cfg_e8735d3fe1dd4bc2'),
               dict(dataset='vidstg', source='hcstvg2', base=ROOT/'artifacts/decota_spatial_opd_tuning_v1',
                    trial='trials/vidstg/refine_10')]
    records, rms = [], []
    for source in sources:
        b = source['base']; folder = b/source['trial']; barrier = read(folder/'PREDICTION_BARRIER.json')
        assert barrier['status'] == 'sealed' and barrier['cells'] == 64 and not barrier['GT_read']
        paths = sorted((folder/'order1').glob('*.pt')); assert len(paths) == 32
        parents = set()
        for ordinal, p in enumerate(paths):
            z = load(p); rc = read(p.with_suffix('.json')); h = sha(p)
            assert h == rc['sha256'] and not rc['GT_read'] and not z['GT_read']
            assert barrier['files'].get(relative(p), barrier['files'].get(str(p.relative_to(b)))) == h
            assert z['order'] == 'order1' and z['arrival'] == ordinal and z['dataset'] == source['dataset']
            parents.add(z['parent']); ip = b/z['input']['path']; assert sha(ip) == z['input']['sha256']
            initial = z['fit']['path'][0]['boxes'].numpy().astype(float)
            next_ = z['fit']['path'][1]['boxes'].numpy().astype(float)
            value = float(np.sqrt(np.mean((next_-initial)**2))); rms.append(value)
            records.append(dict(index=len(records), dataset=z['dataset'], source=z['source'], query_ordinal=z['parent'],
                development_parent=z['parent'], arrival=ordinal, order='order1',
                original_Full=dict(path=relative(p), sha256=h),
                input=dict(path=relative(ip), sha256=z['input']['sha256']),
                config=z['config'], first_step_output_RMS=value, qualify=ordinal < 2))
        assert len(parents) == 32
    assert len(records) == 64 and sum(r['qualify'] for r in records) == 4
    budget = float(np.median(rms)); assert math.isfinite(budget) and budget >= 0
    BASE.mkdir(parents=True, exist_ok=True)
    write(BASE/'DESIGN_LOCK.json', dict(status='fixed_from_old_development_science_before_new_Q0_GT', records=records,
        cells=64, parents_per_dataset=32, pooled_output_RMS_budget=budget,
        budget_definition='median of all 64 original Full first-step native RMS including zeros',
        groups=GROUPS, directions_max=8, common_direction_count='min(8, Query numerical rank, LN numerical rank)',
        amplitude_max=1., bisections=12, upward_doublings_max=10, rank_tolerance='max(J.shape)*float32_eps*s0',
        qualification_queries=4, qualification_real_repeats=2, candidate_writeback=False,
        optimizer_steps=0, historically_exposed_development=True, GT_read=False, time=time.time()))
    write(BASE/'CPU_CONTRACTS.json', contracts())
    write(BASE/'PRIORITY_AUTHORIZATION.json', dict(status='human_requested_Q0_first', requests=['做这个', '先做一下'],
        scope='bounded Q0 only; reversible saved-boundary priority pause if needed, no kill/restart/source changes',
        previous_tasks_remain_authorized=True, GT_read=False, time=time.time()))
    pins = {}
    own = ['scripts/stvg_opd_query_q0_common_v1.py', 'vg_tta/stvg_opd_query_q0_v1.py',
           'scripts/run_stvg_opd_query_q0_v1.py', 'scripts/continue_stvg_opd_query_q0_v1.py',
           'scripts/score_stvg_opd_query_q0_v1.py', 'protocols/stvg_opd_query_observability_q0_v1.md',
           'scripts/pause_stvg_opd_for_geometry_v1.py', 'scripts/stvg_opd_legacy_input_bridge003.py',
           'scripts/run_spatial_ssl_gpu_v1.py', 'methods/CURRENT_METHOD.json',
           'methods/decota_spatial_opd_v1/configs.json']
    old = read(ROOT/'artifacts/stvg_opd_optimization_geometry_v1/GPU_RUNTIME.json')
    pins.update(old['pins']); pins.update({p:sha(ROOT/p) for p in own})
    write(BASE/'RUNTIME.json', dict(status='locked_before_actual_Q0_GPU', design_sha256=sha(BASE/'DESIGN_LOCK.json'),
          CPU_contract_sha256=sha(BASE/'CPU_CONTRACTS.json'), pins=pins,
          old_O_GPU_runtime_sha256=sha(ROOT/'artifacts/stvg_opd_optimization_geometry_v1/GPU_RUNTIME.json'),
          GT_read=False, source_main_method_changed=False, time=time.time()))
    status(BASE/'STATUS.json', dict(status='prepared_pending_saved_boundary_serial_GPU', formal_done=0,
          total=64, qualification_done=0, GT_read=False, CPU_only=True, time=time.time()))


def verify():
    r = read(BASE/'RUNTIME.json')
    assert sha(BASE/'DESIGN_LOCK.json') == r['design_sha256']
    assert sha(BASE/'CPU_CONTRACTS.json') == r['CPU_contract_sha256']
    for p, h in r['pins'].items():
        assert sha(ROOT/p) == h, p
    return r


if __name__ == '__main__':
    prepare()
