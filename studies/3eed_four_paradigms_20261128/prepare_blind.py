"""Prepare shared sensor observations and a fixed, label-free crop cohort."""
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from pathlib import Path
import sys
from unittest.mock import patch

import cv2
import numpy as np

from sim_train import HERE, RUN, ROOT, digest, write

sys.path.insert(0, str(HERE.parent/'3eed_minimal_pipeline_20261128'))
from replay_minimal import sealed_inputs, read

TARGETS=('waymo','drone','quad','kitti','mono')
QWEN=ROOT/'assets/models/Qwen3-VL-4B-Instruct/ebb281ec70b05090aa6165b016eac8ec08e71b17'


def order(value):
    return sha256(('four-paradigms-fovea-v1|'+value).encode()).hexdigest()


def cohort(source):
    out=[]
    for target in TARGETS:
        buckets=defaultdict(list)
        image_seen=set()
        for uid in sorted((u for u,r in source.items() if r['target']==target),key=order):
            row=source[uid]
            rgb=row['frame']['rgb']
            if rgb in image_seen:
                continue
            image_seen.add(rgb)
            buckets[row.get('recording') or rgb].append(uid)
        keys=sorted(buckets,key=order)
        ranked=[buckets[k][i] for i in range(max(map(len,buckets.values()))) for k in keys if i<len(buckets[k])]
        assert len(ranked)>=32
        out.extend(ranked[:32])
    return out


def sampled(points, n):
    return points if len(points)<=n else points[np.linspace(0,len(points)-1,n,dtype=int)]


def normalize_window(values, size):
    if values is None:
        return None
    from live_2d import visible_rectangle
    return visible_rectangle(np.asarray(values,float)*np.tile(size,2)/1000.,size)


def prepare(job):
    import minimal
    from geometry import pixel_rays, sensor_pixels
    from shared import rect

    uid,row,previous=job
    frame,replies=row['frame'],row['raw']
    g=minimal.geometry
    captured=[]
    choose=minimal.choose_region
    def trace(*args):
        support,used=choose(*args)
        if support is not None:
            captured.append(support)
        return support,used
    with patch.object(minimal,'choose_region',trace):
        frozen,aux=minimal.predict(frame,replies,audit=True)
    assert frozen==previous['outputs']['axis_2d'],uid

    spec,size,transform=g.camera(frame)
    inverse=np.linalg.inv(transform)
    eye=inverse[:3,3]
    box=frozen['box']
    K,_,dist,_=spec.validated()
    window=normalize_window(replies['2d']['value'],size)
    first_window=window is not None
    if window is None and box is not None:
        projected=rect(np.asarray(box),transform,K,dist)
        if projected is not None:
            window=normalize_window(projected/np.tile(size,2)*1000.,size)
    raw=replies['3d']['value']
    if raw is None:
        prior=np.array([1.5,1.,1.5])
        yaw=0.
    else:
        upright,rotation=g.to_sensor(np.asarray(raw,float),transform)
        vertical=int(np.argmax(np.abs(rotation[2])))
        prior=np.r_[np.sort(np.delete(raw[3:6],vertical))[::-1],raw[3+vertical]]
        yaw=float(upright[6])

    selection=np.empty((0,3)); audit=selection; support=selection; candidate_refs=[]; fits=[]
    if aux is not None:
        evidence,candidates,_=aux
        support,selection,audit=evidence.support_sensor,evidence.selection,evidence.audit
        fits=[fitted.tolist() for fitted,_,_,_ in candidates]
        candidate_refs=[reference.tolist() for _,reference,_,_ in candidates]
    elif window is not None:
        uv,returns=g.raw_returns(frame,spec,size)
        picks,cells=g.z_buffer(uv,returns,np.flatnonzero(g.inside(uv,window)))
        partition=g.reserve(cells)
        if captured:
            support=g.rigid_xyz(captured[-1],inverse@g.model_to_camera_matrix(spec))
            cam_support=g.rigid_xyz(support,transform)
            depths=cam_support[:,2]
        else:
            central=returns[picks[g.inside(uv[picks],window,.55)&(partition==0)],2]
            depths=central if len(central) else returns[picks,2]
        if len(depths):
            mid=np.median(depths)
            band=max(.4,float(prior.max())*.75)
            close=abs(returns[picks,2]-mid)<=band
            if not len(support):
                support=g.rigid_xyz(returns[picks[close&(partition==0)]],inverse)
            selection=sampled(returns[picks[close&(partition==1)]],128)
            audit=sampled(returns[picks[close&(partition==2)]],128)

    support=sampled(support,320)
    observed=np.r_[support,g.rigid_xyz(selection,inverse)]
    if box is None and window is not None and len(observed):
        # An evidence-bearing case may be estimated without a decoded 3D box.
        center=np.median(observed,axis=0)
    elif box is not None:
        center=np.asarray(box[:3])
    else:
        center=None
    points=sampled(observed,64)
    marks=[]
    if first_window and len(support):
        proj=sensor_pixels(g.rigid_xyz(support,transform),spec)
        if proj is not None:
            hit=g.inside(proj,window,.85)
            marks=proj[hit].tolist()
    if window is not None:
        guides=pixel_rays(np.asarray([window[:2],window[2:]]),spec)
        tangents=np.r_[guides[0,:2]/guides[0,2],guides[1,:2]/guides[1,2]].tolist()
    else:
        tangents=[0.]*4
    return dict(uid=uid,target=row['target'],base=frozen,window=None if window is None else window.tolist(),
                first_window=first_window,size=np.asarray(size).tolist(),transform=transform.tolist(),K=K.tolist(),dist=dist.tolist(),
                eye=eye.tolist(),prior=prior.tolist(),yaw=yaw,initial=None if center is None else center.tolist(),
                measured=support.tolist(),points=points.tolist(),selection=selection.tolist(),audit=audit.tolist(),
                reference=candidate_refs,fitted=fits,marks=marks,tangents=tangents)


def main():
    cv2.setNumThreads(1)
    RUN.mkdir(exist_ok=True,parents=True)
    source,_,history,pins,_=sealed_inputs()
    fovea=cohort(source)
    with ProcessPoolExecutor(8) as workers:
        cases=[]
        for record in workers.map(prepare,((u,r,history[u]) for u,r in source.items()),chunksize=3):
            cases.append(record)
            if len(cases)%150==0 or len(cases)==len(source):
                print('Blind evidence',len(cases),'/',len(source),flush=True)
    assert len(cases)==873
    write(RUN/'inputs-private.json',cases)
    write(RUN/'fovea-roster-private.json',fovea)
    write(RUN/'input-seal.json',dict(source=pins,replayed_baseline=873,labels_read=False,
          evidence=digest(RUN/'inputs-private.json'),fovea=digest(RUN/'fovea-roster-private.json'),
          crop_counts=dict(Counter(source[u]['target'] for u in fovea)),code=digest(__file__)))


if __name__=='__main__':
    main()
