"""Generate physically and linguistically crossed observations with frozen single-source v1."""
import argparse
from dataclasses import replace
from functools import lru_cache
from hashlib import sha256
from time import perf_counter
from unittest.mock import patch
import numpy as np
import torch
from scipy.special import softmax

from registry_swap import HERE, ROOT, WORK, OBS, PARENT, SOURCES, SPEC, digest, read, save, key, parent_key
from collect import Grounder, annulus, profiles, original_pool, role_mass, converse, archive_npz
from roles_joint import affinity
from joint_coverage import bearings
from physical_swap import select_exchange, neutral_text


def view(model, inp, swap=None):
    import interface66
    frontend = model._frontend
    native, original = interface66.capture, frontend.score_frame
    recorded = {}
    def capture(net, pts, indices, text):
        changed = np.ascontiguousarray(swap.move(pts), dtype=np.float32) if swap is not None else pts
        recorded['ids'] = sha256(indices.tobytes()).hexdigest()
        recorded['points'] = changed
        return native(net, changed, indices, text)
    def score(**kwargs):
        return replace(original(**kwargs), sampled_points=recorded['points'])
    with patch.object(interface66, 'capture', capture), patch.object(frontend, 'score_frame', score):
        cap = {}
        scene, ev = model._read_observation(inp, capture=cap)
    return scene, ev, cap, recorded['ids']


def collect_one(model, rec):
    inp = rec['frame']
    scene, ev, cap, ids = view(model, inp)
    result = model._decoder(ev)
    i, j = result['choices'][0].cpu().tolist()
    a = original_pool(scene)
    s = ev['subject_boxes'][0].double().cpu().numpy()
    r = ev['reference_boxes'][0,j].double().cpu().numpy()
    if not ev['has_reference'][0]:
        r = np.full(7, np.nan)
    old = dict(a=a, slate=s, base4=softmax(result['subject_logits'][0].double().cpu().numpy()),
               before=i, ref=r, ma=role_mass(scene), active=False, inverse_error=0.)
    phrase = inp['text'].strip()
    poses = converse(phrase, cap) if ev['has_reference'][0] else None
    neutral = neutral_text(phrase, cap['graph']) if poses else None
    if neutral is None:
        return dict(**old, reason='language')
    sign,text = poses
    if not rec['panel'].startswith('ordinary_'):
        name = parent_key(rec)
        with np.load(PARENT/'private'/name, allow_pickle=False) as prev:
            np.testing.assert_array_equal(a, prev['a'])
            assert int(prev['before'][0]) == i and bool(prev['active'])
            d, dm, hd = prev['d'], prev['mass_d'], prev['hd']
    else:
        dscene, _, _, did = view(model, dict(inp, text=text))
        assert did == ids
        np.testing.assert_array_equal(dscene.sampled_points, scene.sampled_points)
        d, dm = original_pool(dscene), role_mass(dscene)
        pts,col,width = annulus(scene, inp)
        hd = profiles(d,pts,col,width)
    dscore = affinity(d,d) @ dm
    candidate = select_exchange(d, dscore, hd.sum(1))
    if candidate is None:
        return dict(**old, reason='disjoint_landmarks', d=d)
    ix, swap = candidate
    before_points = scene.sampled_points
    after_points = np.ascontiguousarray(swap.move(before_points), dtype=np.float32)
    both = swap.masks(before_points).sum(1)
    assert both.min() >= 4
    inverse_error = float(np.max(abs(swap.move(after_points)-before_points)))
    if inverse_error > 5e-5:
        return dict(**old, reason='noninvertible', d=d)
    pools = [a]
    masses = [old['ma']]
    refs = d[ix]
    moved_refs = refs.copy()
    moved_refs[:,:3] = refs[::-1,:3]
    ha, hda = None, None
    for changed, words in ((False,neutral),(True,phrase),(True,neutral)):
        current,_,_,cid = view(model,dict(inp,text=words),swap if changed else None)
        assert cid == ids
        np.testing.assert_array_equal(current.sampled_points, after_points if changed else before_points)
        pools.append(original_pool(current))
        masses.append(role_mass(current))
        when = int(changed)
        if when==0 or ha is None:
            pts,col,width = annulus(current,inp)
            h = profiles(a,pts,col,width)
            ref_hist = profiles(moved_refs if changed else refs,pts,col,width)
            if when:
                ha,hda = h,ref_hist
            else:
                original_h,original_d = h,ref_hist
    assert ha is not None
    available = (original_h.sum(1)>0)[:,None] & (ha.sum(1)>0)[:,None]
    available = available & ((original_d.sum(1)>0) & (hda.sum(1)>0))[None,:]
    return dict(old, active=True, reason='applied', d=d, views=np.stack(pools), masses=np.stack(masses),
                landmarks=refs, q=dscore[ix]/dscore[ix].sum(), cells=swap.centers, half=swap.half,
                looks=np.stack([bearings(original_h,original_d,sign),bearings(ha,hda,sign)]),
                available=available, moved_points=both, inverse_error=inverse_error)


def pins():
    spec = read(SPEC)
    assert digest(ROOT/'crossplatform/single_frame_sources.json') == spec['source_manifest']
    assert digest(PARENT/'private/OBSERVED.json') == spec['baseline_observations']
    return dict(protocol=digest(SPEC), roster=digest(WORK/'roster.json'),
                code={n:digest(HERE/n) for n in ('observe_swap.py','physical_swap.py','registry_swap.py')})


@torch.inference_mode()
def run(source):
    pin = pins()
    prior = read(PARENT/'private/OBSERVED.json')['files']
    jobs = [r for r in read(WORK/'roster.json') if r['source']==source]
    OBS.mkdir(mode=0o700, exist_ok=True)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    model = Grounder(source)
    model._frontend.parser = lru_cache(maxsize=1024)(model._frontend.parser)
    files = {}
    start = perf_counter()
    for i,rec in enumerate(jobs,1):
        name = key(rec)
        dst = OBS/name
        receipt = dst.with_suffix('.json')
        if not receipt.exists():
            if not rec['panel'].startswith('ordinary_'):
                p = parent_key(rec)
                assert digest(PARENT/'private'/p) == prior[p]
            obs = collect_one(model,rec)
            archive_npz(dst, **obs)
            save(receipt, dict(pin=pin,sha256=digest(dst)))
        saved = read(receipt)
        assert saved['pin']==pin and digest(dst)==saved['sha256']
        files[name] = saved['sha256']
        if i%40==0 or i==len(jobs):
            print(source,i,len(jobs),round(perf_counter()-start,1),flush=True)
    save(OBS/f'{source}.seal.json', dict(pin=pin,files=files,seconds=perf_counter()-start))


def seal():
    pin = pins()
    files = {}
    for src in SOURCES:
        pack = read(OBS/f'{src}.seal.json')
        assert pack['pin']==pin and not files.keys() & pack['files'].keys()
        files.update(pack['files'])
    assert set(files)=={key(r) for r in read(WORK/'roster.json')}
    assert all(digest(OBS/n)==v for n,v in files.items())
    save(OBS/'SEALED.json', dict(pin=pin,n=len(files),files=files,labels=False,FINAL_access=False))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',choices=(*SOURCES,'seal'))
    arg=p.parse_args().source
    seal() if arg=='seal' else run(arg)
