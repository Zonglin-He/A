"""Independent finite full-target corruption protocol, no old runtime mutations."""
import os, sys, json, time, hashlib, csv, shutil, gc
from pathlib import Path
from functools import lru_cache
R = Path(__file__).resolve().parents[2]
S = Path(__file__).parent
sys.path[:0] = [str(R/'studies/no_shrink_main_v1'), str(R), str(R/'tools')]
import ns_common as ns
from full import read, sha, atomic, identity, check, RUN, PROVIDER, binding
from operators import conditions
D = R/'outputs/corruption_full_v1'
P = R/'results/corruption_full_v1'
COUNTS = ns.COUNTS
digest = ns.digest
exited = ns.exited
records = ns.records
labels = ns.labels
fixed = lru_cache(None)(read)

def validate():
    p = read(D/'plan.json')
    assert p['fingerprint'] == digest({k:v for k,v in p.items() if k != 'fingerprint'})
    for f,h in p['code_sha256'].items():
        assert sha(R/f) == h, f
    for f,h in p['input_sha256'].items():
        assert sha(f) == h, f
    check()
    return p

def unit(ds, name):
    assert ds in COUNTS and name in {c['id'] for c in conditions()}
    return D/'units'/name/ds

def condition(name):
    return next(c for c in conditions() if c['id'] == name)

def resources():
    assert shutil.disk_usage(R).free >= 80*2**30
    assert sum(f.stat().st_size for f in D.rglob('*') if f.is_file()) < 160*2**30
    assert int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024 >= 8*2**30

def save(path, value):
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    torch.save(value, temp)
    os.replace(temp, path)

def legacy(ds, key, name):
    if not name.startswith('legacy_'):
        return None
    path = ns.D/f'corruption_{ds}_sealed.json'
    e = fixed(path)['files'].get(key+'.'+name.removeprefix('legacy_'))
    return e

def clean(ds, row):
    import numpy as np
    ep = fixed(RUN/ds/'sealed.json')['files'][row['id']]
    pp = RUN/ds/ep['path']
    rp = RUN/ds/ep['receipt_path']
    assert sha(pp) == ep['sha256'] and sha(rp) == ep['receipt_sha256']
    assert read(rp)['input_binding'] == binding(row, fixed(RUN/'plan.json')['provider_runtime_fingerprint'])
    npth, proof = ns.no_shrink(ds, row['id'])
    with np.load(npth) as z, np.load(pp) as old:
        assert np.array_equal(z['Source'], old['Source'])
        a = dict(Source=old['Source'].copy(), Source_interval=z['Source_interval'].copy(),
                 FullMAP=z['NoShrink'].copy(), FullMAP_interval=z['NoShrink_interval'].copy(),
                 frame_ids=old['frame_ids'].copy(), positions8=old['positions8'].copy())
    return a, dict(Source_prediction=str(pp), Source_prediction_sha256=ep['sha256'],
                   Source_receipt=str(rp), Source_receipt_sha256=ep['receipt_sha256'], no_shrink=proof,
                   inherited_nonzero_acquisition=True)

def replay(c):
    import numpy as np, torch
    if c['kind'] == 'clean_identity':
        row = next(r for r in records(c['dataset']) if r['id'] == c['query'])
        a, proof = clean(c['dataset'], row)
        assert proof == c['provenance']
        return a
    if c['kind'] == 'legacy_reused':
        e = c['entry']
        p = ns.D/e['path']
        rp = ns.D/e['receipt_path']
        assert sha(p) == e['sha256'] and sha(rp) == e['receipt_sha256']
        from core import derive
        row = next(r for r in records(c['dataset']) if r['id'] == c['query'])
        a, _ = derive('corruption', c['dataset'], dict(row=row, key=row['id'], view=c['view']))
        with np.load(p) as z:
            assert all(np.array_equal(a[k], z[k]) for k in a)
        return a
    from stvg.prototype import one_step
    from stvg.current import map_interval
    from stvg.spatial import SpatialGeometry, evidence_dp
    q, _ = one_step(c['content_features'], c['native_prior'], kappa=0, beta=1)
    iv = map_interval(c['frame_ids'], q)
    g = SpatialGeometry.from_cached_geometry(c['geometry'])
    return dict(Source=c['Source'], Source_interval=np.asarray(c['Source_interval']),
                FullMAP=evidence_dp(g.for_interval(iv))['final'].numpy(), FullMAP_interval=np.asarray(iv),
                frame_ids=np.asarray(c['frame_ids']), positions8=np.asarray(c['positions8']))
