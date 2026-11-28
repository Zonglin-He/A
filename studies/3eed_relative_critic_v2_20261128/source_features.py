"""Build frozen candidate tensors from real Waymo and pre-Bind perturbations."""
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter, sleep
from unittest.mock import patch

import cv2
import numpy as np
import torch

from compact_pool import digest, read, write
from descriptors import describe
from prepare_blind import prepare
from score_paradigms import ious
from source import BATCH, SOURCE, gate, order, runtime, source_guard

PROBABILITIES = [.30,.30,.40]
VIEW_NAMES = ('clean','high','low','scan')


def target_band(frame, replies, points):
    from minimal import geometry
    from geometry import sensor_pixels

    window = replies['2d']['value']
    if window is None:
        return np.zeros(len(points),bool)
    spec, size, transform = geometry.camera(frame)
    camera = geometry.rigid_xyz(points[:,:3],transform)
    idx = np.flatnonzero((camera[:,2] > .2) & np.isfinite(camera).all(1))
    if not len(idx):
        return np.zeros(len(points),bool)
    uv = sensor_pixels(camera[idx],spec)
    rect = np.asarray(window,float)*np.tile(size,2)/1000.
    roi = geometry.inside(uv,rect)
    center = geometry.inside(uv,rect,.5)
    depths = camera[idx,2]
    strong = depths[center] if center.any() else depths[roi]
    mask = np.zeros(len(points),bool)
    if not len(strong):
        return mask
    span = replies['3d']['value']
    limit = max(.4,.75*max(span[3:6])) if span is not None else 1.
    mask[idx[roi & (abs(depths-np.median(strong)) <= limit)]] = True
    return mask


def perturbed(points, band, rng, mode):
    """Use only measured points and the first Qwen window to localize thinning."""
    keep = np.ones(len(points),bool)
    ground_shift = 0.
    ids = np.flatnonzero(band)
    if mode in ('high','low') and len(ids) >= 3:
        cloud = points[ids,:3]
        z0,z1 = np.quantile(cloud[:,2],[.10,.90])
        cut = z0+rng.uniform(.25,.65)*(z1-z0)
        if mode == 'high':
            keep[ids] = (cloud[:,2]>=cut) & (rng.random(len(ids))>=rng.uniform(.75,.95))
        else:
            local = cloud[:,:2]-np.median(cloud[:,:2],axis=0)
            yaw = np.arctan2(local[:,1],local[:,0])
            azimuth = rng.uniform(-np.pi,np.pi)
            width = np.deg2rad(rng.uniform(30,120))
            dist = (yaw-azimuth+np.pi)%(2*np.pi)-np.pi
            keep[ids] = (cloud[:,2]<=cut) & (abs(dist)>width/2)
    if mode == 'low':
        ground_shift = float(rng.uniform(-.15,.15))

    xyz = points[:,:3]
    elevation = np.arctan2(xyz[:,2],np.linalg.norm(xyz[:,:2],axis=1))
    bands = np.floor(elevation/np.deg2rad(.25)).astype(int)
    levels,assigned = np.unique(bands,return_inverse=True)
    number = round(len(levels)*rng.uniform(.50,.85))
    distinct = np.ones(len(levels),bool)
    distinct[rng.choice(len(levels),number,replace=False)] = False
    result = points[keep & distinct[assigned]].copy()
    sig = rng.uniform(.02,.08)
    result[:,:3] += rng.normal(0.,sig,(len(result),3))
    return result,ground_shift


def perturbed_reply(replies, rng):
    original = replies['2d']['value']
    if original is None:
        return replies
    first = np.asarray(original,float)
    size = first[2:]-first[:2]
    center = (first[:2]+first[2:])/2 + rng.uniform(-.05,.05,2)*size
    half = size*rng.uniform(.90,1.20,2)/2
    value = np.clip(np.r_[center-half,center+half],0,1000).tolist()
    return dict(replies, **{'2d':dict(replies['2d'],value=value)})


def pool_features(case, frame, replies):
    """Union the three eligible binding supports for the shared local point statistics."""
    import compact_pool
    from minimal import geometry

    original = compact_pool.ranked_regions
    regions = {}

    def capture(*args):
        found = original(*args)
        regions['H' if args[-1] is None else 'C_ray'] = found
        return found

    with patch.object(compact_pool,'ranked_regions',capture):
        pack = compact_pool.make(case,frame,replies)
    selected = regions.get(pack['branch'],[])
    if selected:
        spec, _, transform = geometry.camera(frame)
        mat = np.linalg.inv(transform)@geometry.model_to_camera_matrix(spec)
        points = np.unique(np.concatenate([v.points for v in selected]),axis=0)
        case = dict(case,measured=geometry.rigid_xyz(points,mat))
    return pack,describe(case,pack['pool'])


def observation(frame, replies, truth, ground_shift):
    import minimal

    geometry = minimal.geometry
    terrain = geometry.regions.terrain
    cache = {}
    with ExitStack() as ctx:
        for key in ('camera','raw_returns','fit_sample','model_sample'):
            func = getattr(geometry,key)
            def reuse(*args, key=key, func=func):
                if key not in cache:
                    cache[key] = func(*args)
                return cache[key]
            ctx.enter_context(patch.object(geometry,key,reuse))

        if ground_shift:
            def shifted(*args):
                value = np.asarray(terrain(*args)).copy()
                value[2] += ground_shift
                return value
            ctx.enter_context(patch.object(geometry.regions,'terrain',shifted))
        base = minimal.predict(frame,replies)
        row = dict(frame=frame,raw=replies,target='waymo')
        case = prepare((frame['uid'],row,dict(outputs=dict(axis_2d=base))))
        pack, (nodes,edges,valid) = pool_features(case,frame,replies)
    values = np.zeros(len(valid),np.float32)
    values[:valid.sum()] = ious([p['box'] for p in pack['pool']],truth)
    return nodes,edges,valid,values,pack['branch']


def job(value):
    row, replies = value
    frame = row['frame']
    points = np.load(frame['points'],allow_pickle=False)
    points = points[np.isfinite(points[:,:3]).all(1)]
    band = target_band(frame,replies,points)
    rng = np.random.default_rng(int(order(row['uid']+'|degrade')[:16],16))
    output = [observation(frame,replies,row['box'],0.) + (0,)]
    with TemporaryDirectory(prefix='view-',dir=SOURCE/'scratch') as directory:
        for i in range(4):
            mode = rng.choice(VIEW_NAMES[1:],p=PROBABILITIES)
            degraded,shift = perturbed(points,band,rng,mode)
            new_path = str(Path(directory)/f'{i}.npy')
            np.save(new_path,degraded,allow_pickle=False)
            view = dict(frame,points=new_path)
            raw = perturbed_reply(replies,rng)
            output.append(observation(view,raw,row['box'],shift) + (VIEW_NAMES.index(mode),))
    return output


def verified_batches(part, rows, start, seen):
    for j in range(0,len(rows),BATCH):
        batch = rows[j:j+BATCH]
        reply = SOURCE/'qwen'/f'{part}-{start+j:04d}.json'
        while not reply.is_file():
            sleep(4)
        raw = read(reply)
        assert [r['uid'] for r in batch] == raw['pins']['ids']
        assert raw['pins']['roster'] == digest(SOURCE/'roster-seal.json')
        seen[reply.name] = digest(reply)
        yield from zip(batch,raw['replies'],strict=True)


def materialize(part, workers, seen):
    roster = SOURCE/f'{part}-rows-private.json'
    rows = read(roster)
    assert read(SOURCE/'roster-seal.json')['population'][part]['bank'] == digest(roster)
    chunks = SOURCE/'feature-parts'
    chunks.mkdir(exist_ok=True)
    block_size = 96
    banks = []
    for j in range(0,len(rows),block_size):
        path = chunks/f'{part}-{j:04d}.npz'
        block = list(verified_batches(part,rows[j:j+block_size],j,seen))
        if not path.exists():
            records = [r for group in workers.map(job,block,chunksize=2) for r in group]
            nodes,edges,valid,labels,branches,modes = map(np.asarray,zip(*records,strict=True))
            tmp = path.with_suffix('.tmp')
            with tmp.open('wb') as fp:
                np.savez(fp,nodes=nodes,edges=edges,valid=valid,labels=labels,branches=branches,modes=modes,
                         uid=np.repeat([b[0]['uid'] for b in block],5),
                         recording=np.repeat([b[0]['recording'] for b in block],5))
            tmp.replace(path)
        banks.append(path)
        if (j//block_size)%5 == 0 or j+block_size>=len(rows):
            print('Waymo tensor bank',part,min(j+block_size,len(rows)),'/',len(rows),flush=True)
    return dict(queries=len(rows), views=5*len(rows), files={p.name:digest(p) for p in banks})


def main():
    source_guard()
    gate()
    runtime()
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    started = perf_counter()
    (SOURCE/'scratch').mkdir(exist_ok=True)
    seen = {}
    with ProcessPoolExecutor(10) as workers:
        result = {part:materialize(part,workers,seen) for part in ('val','train')}
    seal = read(SOURCE/'qwen-seal.json')
    assert seal['roster'] == digest(SOURCE/'roster-seal.json') and seal['source_only']
    assert seen == seal['checksums']
    names = ('source_features.py','descriptors.py','compact_pool.py')
    here = Path(__file__).parent
    write(SOURCE/'features-seal.json',dict(banks=result, source_only=True,
          decoded=digest(SOURCE/'qwen-seal.json'), code={name:digest(here/name) for name in names},
          seconds=perf_counter()-started))


if __name__ == '__main__':
    main()
