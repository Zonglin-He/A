"""Lock test membership, complete-scene proposals, depth prefixes and ICL donor banks."""
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter
import sys

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'3eed_nullspace_solver_20261128'))
from run_experiment import FrameEvidence, ROOT, check, digest, load_blind, marks, publish, rays_for, read, rect_for
from shared import canonical, pixels, rect, stable, slab, ray_geometry

RUN=ROOT/'runs/inference-time-vlm-20261128'
PLATFORMS=('waymo','drone','quad')
TEST_COUNT=32
KMAX=8


def overlap(a,b):
    if a is None or b is None:
        return 0.
    a,b=np.asarray(a),np.asarray(b)
    inter=np.prod(np.maximum(0.,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])))
    total=np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-inter
    return float(inter/total) if total>0 else 0.


def clipped(projected, size):
    if projected is None:
        return None
    bounds=np.clip(projected,0,np.tile(size,2))
    if min(bounds[2:]-bounds[:2])>=2:
        return bounds
    return None


def locked_members(src):
    result=[]
    used=set()
    for platform in PLATFORMS:
        segments=defaultdict(list)
        for uid,r in src.items():
            if r['target']==platform:
                segments[r['recording']].append(uid)
        for item in segments.values():
            item.sort(key=lambda u:stable('inference-compute-v1|'+u))
        levels=sorted(segments,key=lambda tag:stable('inference-rec-v1|'+tag))
        for i in range(max(map(len,segments.values()))):
            for record in levels:
                if i>=len(segments[record]):
                    continue
                u=segments[record][i]
                rgb=src[u]['frame']['rgb']
                if rgb in used:
                    continue
                result.append(u)
                used.add(rgb)
                if sum(src[q]['target']==platform for q in result)==TEST_COUNT:
                    break
            if sum(src[q]['target']==platform for q in result)==TEST_COUNT:
                break
    assert len(result)==len(used)==TEST_COUNT*len(PLATFORMS)
    return result


def raw_box(box, ev):
    vec=np.asarray(box,dtype=float)
    heading=Rotation.from_euler('z',vec[6]).as_matrix()
    rotation=Rotation.from_matrix(ev.mat[:3,:3]@heading)
    aa=rotation.as_euler('xyz')/np.pi
    center=ev.module.rigid_xyz(vec[None,:3],ev.mat)[0]
    return np.r_[center,vec[3:6],aa].tolist()


def center_forms(row, diag, formal, ev):
    raw = row['raw']['3d']['value']
    win = rect_for(row,ev)
    if raw is None or formal is None:
        return None
    native=np.asarray(raw[:3],float)
    anchor=ev.module.rigid_xyz(np.asarray(formal)[None,:3],ev.mat)[0]
    if win is None or not np.isfinite(anchor).all() or anchor[2]<=.3:
        return None
    observed=ev.observed(win)['fit']
    held=diag['selection']
    if len(held):
        # Center supervision stays fixed. Dispersion is determined without GT.
        ranges=np.asarray(held)[:,2]
    elif len(observed):
        ranges=np.asarray(observed)[:,2]
    else:
        return None
    med=np.median(ranges)
    spread=max(1.,float(1.4826*np.median(abs(ranges-med))))
    wrong=anchor*1.4
    interpolated=[(1-t)*native+t*anchor for t in (.25,.50,.75)]
    candidates=[native,anchor,*interpolated,anchor*.85,anchor*1.15]
    c={tuple(float(f'{v:.2f}') for v in x) for x in candidates if x[2]>.2}
    grid=sorted(c)
    logits=[]
    for x in grid:
        delta=np.asarray(x)-anchor
        logits.append(float(-min(20.,np.dot(delta,delta)/(2*spread**2))))
    return dict(native=native.tolist(),metric=anchor.tolist(),wrong=wrong.tolist(),grid=grid,log_prior=logits)


def proposal_bank(row, ev):
    """Build all visible proposals using only the calibrated full-view LiDAR."""
    import cray_simple

    context=ev.get('fit_sample',ev.frame,ev.spec,ev.size)[0]
    if len(context)<9:
        return [],0
    terrain=ev.module.regions.terrain(context)
    _, assigned, births=cray_simple.region_catalog(context,terrain,cray_simple.FOUR_SCALES)
    K,_,dist,_=ev.spec.validated()
    world=ev.module.rigid_xyz(context,ev.model_to_raw)
    cam=ev.module.rigid_xyz(context,ev.model_to_cam)
    valid=cam[:,2]>.2
    xy=np.full((len(cam),2),0.)
    xy[valid]=pixels(cam[valid],K,dist)
    seen=valid&(xy>=0).all(1)&(xy<ev.size).all(1)
    raw=row['raw']['3d']['value']
    prior=ev.module.to_sensor(np.array(raw,dtype=float),ev.mat)[0][3:6] if raw is not None else None
    proposals=[]
    # Only points and calibration decide which proposals exist and their order.
    groups={}
    for level,ids in enumerate(assigned):
        for j in np.unique(ids[ids>=0]):
            if births[j]==level:
                groups[int(j)]=np.flatnonzero(ids==j)
    for number,rows in groups.items():
        vis=rows[seen[rows]]
        if len(vis)<6:
            continue
        points=context[rows]
        measured=ev.module.regions.measured_box(points,terrain)
        if max(measured[3:5])>18 or measured[5]>6 or min(measured[3:6])<.08:
            continue
        near=np.quantile(xy[vis],[.05,.95],axis=0)
        rectangle=clipped(near.reshape(-1),ev.size)
        if rectangle is None or min(rectangle[2:]-rectangle[:2])<5:
            continue
        obs=canonical(ev.module.sensor_frame_box(measured,ev.frame['target'],ev.frame['suffix']))
        completed=cray_simple.complete_h(points,terrain,prior) if prior is not None else measured
        complete=canonical(ev.module.sensor_frame_box(completed,ev.frame['target'],ev.frame['suffix']))
        view=np.median(cam[vis],axis=0)
        area=float(np.prod(rectangle[2:]-rectangle[:2]))
        if not 0.2<view[2]<100:
            continue
        # Saturation prevents one very dense region from crowding out others.
        priority=float(np.sqrt(min(len(vis),80))*np.power(min(area,50000.),.15))
        ids=rows if len(rows)<=128 else rows[np.linspace(0,len(rows)-1,128,dtype=int)]
        mid=(rectangle[:2]+rectangle[2:])/2
        H=float(np.median(points[:,2]-points[:,:2]@terrain[:2]-terrain[2])) if np.isfinite(terrain).all() else None
        proposals.append(dict(region=number,priority=priority,rect=rectangle.tolist(),mark=mid.tolist(),visible=int(len(vis)),
                center_camera=view.tolist(), box_camera=raw_box(complete,ev), measured=obs.tolist(),completed=complete.tolist(),
                height=H,points=world[ids].tolist()))
    proposals.sort(key=lambda x:(-x['priority'],x['region']))
    distinct=[]
    for p in proposals:
        if any(overlap(p['rect'],q['rect'])>.65 and abs(p['center_camera'][2]-q['center_camera'][2])<1.5 for q in distinct):
            continue
        distinct.append(p)
    for i,v in enumerate(distinct,1):
        v['id']=i
    return distinct,len(proposals)


def one(job):
    uid,row,diag,formal=job
    ev=FrameEvidence(row['frame'])
    proposed,n_raw=proposal_bank(row,ev)
    w=rect_for(row,ev)
    shortlist=proposed[:KMAX]
    baseline=[overlap(w,v['rect']) for v in shortlist]
    window_pick=1+int(np.argmax(baseline)) if baseline and max(baseline)>0 else 0
    permutation=list(np.random.default_rng(int(stable('som-permute|'+uid)[:16],16)).permutation(len(shortlist))+1)
    if len(permutation)>1 and permutation==list(range(1,len(permutation)+1)):
        permutation=permutation[1:]+permutation[:1]
    return dict(uid=uid,platform=row['target'],recording=row['recording'],n_raw=n_raw,
        proposals=proposed,window_pick=window_pick,permutation=[int(i) for i in permutation],
        center_forms=center_forms(row,diag,formal,ev),formal=formal)


def donor(job):
    uid,row,diag,formal=job
    ev=FrameEvidence(row['frame'])
    raw=row['raw']['3d']['value']
    w=rect_for(row,ev)
    if raw is None or w is None:
        return None
    obs=ev.observed(w)
    audited=diag['audit'] if diag is not None and diag['route']=='C_ray' and len(diag['audit']) else obs['audit']
    if len(audited)<5 or len(obs['fit'])<5:
        return None
    if diag is not None and diag['route']=='C_ray' and len(diag['audit']):
        candidate=canonical(formal)
        origin='frozen_cray'
    else:
        from geometry import front_offset,pixel_rays
        guide=pixel_rays((w[:2]+w[2:])/2,ev.spec)[0]
        measured=np.asarray(raw,dtype=float).copy()
        measured[:3]=guide*(np.median(obs['fit'][:,2])+front_offset(measured,guide))
        candidate=canonical(ev.module.to_sensor(measured,ev.mat)[0])
        origin='fit_only_depth'
    camera=raw_box(candidate,ev)
    if camera[2]<=.1:
        return None
    K,_,dist,_=ev.spec.validated()
    intersect=overlap(w,clipped(rect(candidate,ev.mat,K,dist),ev.size))
    proof=marks(candidate,rays_for(ev,audited,w))
    if proof is None:
        return None
    strict=intersect>=.85 and proof['truncated_m']<=.25 and proof['hit']>=.60
    value=intersect*proof['hit']/(.2+proof['truncated_m'])
    return dict(uid=uid,platform=row['target'],recording=row['recording'],
        rgb=row['frame']['rgb'],text=row['frame']['text'],raw=camera,strict=strict,
        quality=value,iou2d=intersect,audit=proof,source=origin)


def prepare():
    cv2.setNumThreads(1)
    RUN.mkdir(parents=True,exist_ok=True)
    src,old,inv,selections,base=load_blind()
    picked=locked_members(src)
    occupied={(src[u]['target'],src[u]['recording']) for u in picked}
    images={src[u]['frame']['rgb'] for u in picked}
    from metric_common import input_frames

    calibration=ROOT/'runs/pedestrian-paired-recheck-20261128'
    manifest=calibration/'manifest.json'
    p_input=check(manifest,'manifest',calibration/'input-seal.json')
    legacy=HERE.parent/'3eed_qwen_extent_height_only_20261122/LEGACY_REGISTRATION.json'
    assert p_input['code']['original']['registration']==digest(legacy)
    ped_path=calibration/'predictions.json'
    ped_seal=check(ped_path,'predictions',calibration/'prediction-seal.json')
    assert ped_seal['input']==digest(calibration/'input-seal.json')
    assert not ped_seal['target_labels_read'] and not ped_seal['training']
    frames=input_frames(manifest)
    for f in frames.values():
        assert all(digest(Path(f[k]))==p_input['inputs'][f[k]] for k in ('rgb','points','calibration'))
    ped=read(ped_path)
    assert list(frames)==[x['uid'] for x in ped] and len(ped)==288
    donor_src={}
    for u,s in src.items():
        if (s['target'],s['recording']) not in occupied and s['frame']['rgb'] not in images and s['target'] in PLATFORMS:
            box=inv[u]['pool']['cray']['boxes']
            formal=box[selections[u]['selected']['cray']['cached']] if box else None
            if formal is not None:
                donor_src[u]=(u,s,old[u],formal)
    for r in ped:
        u=r['uid']; frame=frames[u]
        record=Path(frame['rgb']).parent.parent.name
        if (r['target'],record) not in occupied and frame['rgb'] not in images and r['official'] is not None:
            donor_src[u]=(u,dict(frame=frame,raw=r['replies'],target=r['target'],recording=record),None,r['official'])
    unique=set()
    donor_jobs=[]
    for platform in PLATFORMS:
        ids=sorted((u for u,v in donor_src.items() if v[1]['target']==platform),
            key=lambda u:stable('inference-donor-v1|'+u))
        kept=0
        for u in ids:
            v=donor_src[u]
            if v[1]['frame']['rgb'] in unique:
                continue
            unique.add(v[1]['frame']['rgb'])
            donor_jobs.append(v)
            kept+=1
            if kept==100:
                break
    start=perf_counter()
    with ProcessPoolExecutor(10) as pool:
        region_jobs=[]
        for u in picked:
            box=inv[u]['pool']['cray']['boxes']
            formal=box[selections[u]['selected']['cray']['cached']] if box else None
            region_jobs.append((u,src[u],old[u],formal))
        proposals=list(pool.map(one,region_jobs,chunksize=3))
        examples=[q for q in pool.map(donor,donor_jobs,chunksize=4) if q is not None]
    donor_by={}
    for platform in PLATFORMS:
        valid=[e for e in examples if e['platform']==platform]
        assert len(valid)>=2,(platform,len(valid),len(donor_jobs))
        verified=sorted(valid,key=lambda v:(not v['strict'],-v['quality'],v['uid']))[:2]
        random=sorted(valid,key=lambda v:stable('same-prompt-donor-v1|'+v['uid']))[:2]
        cross=sorted((e for e in examples if e['platform']!=platform),
                     key=lambda v:(not v['strict'],-v['quality'],v['uid']))[:2]
        for e in (*verified,*random,*cross):
            assert (e['platform'],e['recording']) not in occupied and e['rgb'] not in images
        donor_by[platform]=dict(verified=verified,random=random,cross=cross,eligible=len(valid),
                                strict_eligible=sum(e['strict'] for e in valid))
    target=RUN/'plan-private.json'
    publish(target,dict(source=base,units=proposals,donors=donor_by,test_recordings=sorted(occupied),
               donor_input=ped_seal['predictions'],planned_protocol=digest(HERE/'PROTOCOL_ZH.md')))
    publish(RUN/'plan-seal.json',dict(input=digest(target),code=digest(Path(__file__)),
                protocol=digest(HERE/'PROTOCOL_ZH.md'),test_queries=len(proposals),labels_read=False,
                donor_eligible=len(examples),cpu_wall_s=perf_counter()-start))
    print('Prepared',len(proposals),'queries and',len(examples),'donors',flush=True)
    for p,m in donor_by.items():
        print(p,'donor eligibility',m['eligible'],m['strict_eligible'],'selected:',[(x['iou2d'],x['audit']['truncated_m'],x['strict']) for x in m['verified']],flush=True)
    print('proposal count',Counter(min(len(x['proposals']),KMAX) for x in proposals),
          'prefix eligibility',sum(bool(x['center_forms']) for x in proposals),flush=True)


if __name__=='__main__':
    prepare()
