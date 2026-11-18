"""Fixed cohort and sealed prediction access for the paired single-frame comparison."""
from pathlib import Path
from functools import lru_cache
import hashlib
import json
import os

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
WORK = ROOT / 'crossplatform/round160'
R158 = ROOT / 'crossplatform/round158'
R159 = ROOT / 'crossplatform/round159'
SPEC = HERE / 'experiment/PROTOCOL.json'
PLATFORMS = ('waymo', 'drone', 'quad')
ORDER = tuple((s,t) for s in PLATFORMS for t in PLATFORMS if s != t)
DEV, EXTRA, REC = 'development1798', 'ordinary5942', 'other_recordings3072'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def array_sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def json_new(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False, ensure_ascii=False)
        f.write('\n')


def npz_atomic(path, **data):
    tmp = path.with_suffix('.tmp')
    with tmp.open('wb') as f:
        np.savez(f, **data)
    os.replace(tmp, path)


def canonical(text):
    return ' '.join(text.replace(',', ' ,').split())


def check_protocol():
    spec = read(SPEC)
    for path, expected in spec['pins'].items():
        if sha(ROOT/path) != expected:
            raise ValueError('Prerequisite changed: '+path)
    return spec


def rows():
    result = []
    for cohort, path, total in (
        (DEV, ROOT/'research/round145/INPUT_REGISTRY_PRIVATE.json', 1798),
        (EXTRA, R158/'EXTRA_REGISTRY_PRIVATE.json', 5942)):
        registry = read(path)
        assert len(registry) == total
        for n, e in enumerate(registry):
            a, r = e['A_sensor'], e['row']
            assert e['target'] == r['platform'] == a['platform'] and (e['source'], e['target']) in ORDER
            assert not e['protected_alias_fallback'] and a['eligible_for_extra_media'] and not a['protected_image_content_alias']
            assert r['caption'] == canonical(r['caption']) and Path(r['pointpath']).samefile(a['pointpath'])
            result.append(dict(cohort=cohort, ordinal=n+1, source=e['source'], target=e['target'], r=r,
                               point_sha=a['point_sha256'], image_sha=a['image_sha256'],
                               sampling_weight=float(e.get('sampling_weight',1)), group=a['point_sha256']))
    registry = read(R159/'RECORDING_INPUTS_PRIVATE.json')
    assert len(registry)==1536
    p_seal=read(ROOT/'research/round159/RECORDING_PREDICTION_SEAL.json')
    n=0
    for source,item in p_seal['sources'].items():
        prior=R159/(source+'_PRIVATE.npz')
        assert sha(prior)==item['sha256']
        with np.load(prior,allow_pickle=False) as p:
            idx = p['rows'].tolist()
        for j,ii in enumerate(idx):
            r=dict(registry[ii],seed=0)
            assert source!=r['platform']
            n+=1
            result.append(dict(cohort=REC, ordinal=n, source=source, target=r['platform'], r=r,
                               prior_offset=j,point_sha=r['point_sha256'], image_sha=r['image_sha256'],
                               sampling_weight=1.,group=r['point_sha256']))
    assert n==3072 and len(set((e['cohort'],e['source'],e['r']['id']) for e in result))==len(result)
    return result


@lru_cache(maxsize=8)
def prior_arrays(cohort,source):
    if cohort==DEV:
        with np.load(R158/'WORKING_PRIVATE.npz',allow_pickle=False) as f:
            keys=f['arms'].tolist()
            ii=[keys.index('frozen_A'),keys.index('overlap_A_only')]
            return f['indices'][:,ii].copy(), f['boxes'][:,ii,:].copy()
    if cohort==EXTRA:
        with np.load(R158/'EXTRA_PRIVATE.npz',allow_pickle=False) as f:
            return f['indices'],f['boxes']
    with np.load(R159/(source+'_PRIVATE.npz'),allow_pickle=False) as f:
        return f['winners'],f['boxes'][:,1:].copy()


def reference(e):
    cohort=e['cohort'];source=e['source'] if cohort==REC else ''
    q,b=prior_arrays(cohort,source)
    i=e['prior_offset'] if cohort==REC else e['ordinal']-1
    return q[i],b[i]


def prediction_file(i):
    return WORK/'representations_PRIVATE'/f'{i:05}.npz'
