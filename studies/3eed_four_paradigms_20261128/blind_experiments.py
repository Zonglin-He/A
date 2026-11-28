"""Run the four independent paradigms on sealed sensor observations."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter
import argparse
import sys

import cv2
import numpy as np
from safetensors.torch import load_file
import torch

from geometry_models import Flow, Critic, critic_features
from physical_readout import (Silhouette, angle, anchored, coordinates, first_hit_counts, flow_intervals,
                              headings, interval, iou_mbr, lattice, make_scene, matched_strict_pair,
                              new_bind, ray_selection_cost, rectangle_iou)
from prepare_blind import HERE, RUN, digest, read, sampled, sealed_inputs, write


ARMS={'flow':('flow_free','flow_project'), 'vit':('vit_select','vit_bind_select'),
      'critic':('grid_aabb','grid_ray','critic_geometry','critic_silhouette'),
      'fovea':('crop_single','crop_mbr','crop_mbr_view','crop_mbr_hard')}
FLOW_PARTICLES=12
FLOW_STEPS=8


def load_network(name,cls,stamp):
    path=RUN/f'{name}.safetensors'
    assert stamp['checkpoints'][name]['sha256']==digest(path)
    model=cls().eval().requires_grad_(False)
    model.load_state_dict(load_file(str(path)))
    return model


def project_step(boxes,intervals):
    references=np.array([c.reference for c in intervals])
    gap=abs(angle(boxes[:,6,None]-references[None,:,6]))
    choices=gap.argmin(1)
    constrained=np.array([intervals[i].project(box) for box,i in zip(boxes,choices,strict=True)])
    return constrained,references[choices]


def flow_generate(flow,scene,case):
    batch=scene.repeat(FLOW_PARTICLES)
    embedding=flow.embed(batch)
    gen=torch.Generator().manual_seed(int(case['uid'][:12],16))
    noise=torch.randn((FLOW_PARTICLES,7),generator=gen)
    free=noise.clone(); x=noise.clone()
    intervals=flow_intervals(case)
    refs=None
    for loop in range(FLOW_STEPS):
        t=torch.full((FLOW_PARTICLES,1),loop/FLOW_STEPS)
        free=free+flow(free,t,embedding)/FLOW_STEPS
        x=x+flow(x,t,embedding)/FLOW_STEPS
        step=batch.decode(x).numpy()
        if intervals:
            step,refs=project_step(step,intervals)
            x=batch.encode(torch.tensor(step,dtype=torch.float32))
    particles=batch.decode(free).numpy()
    pick,conviction=iou_mbr(step)
    alternative,_=iou_mbr(particles)
    audit={k:dict(n=0,changed=0) for k in ('selection','audit')}
    if refs is not None:
        for reference,proposal in zip(refs,step,strict=True):
            for keys in audit:
                counts=first_hit_counts(reference,proposal,case,keys)
                for name in ('n','changed'):
                    audit[keys][name]+=counts[name]
    return dict(flow_free=particles[alternative].tolist(),flow_project=step[pick].tolist()),dict(
        proposed=FLOW_PARTICLES,projected=0 if refs is None else FLOW_PARTICLES,confidence=conviction,
        nonempty_support=bool(len(case['points'])),first_hits=audit)


def critic_read(critic,scene,case,field):
    pool=lattice(case)
    boxes=torch.tensor(pool,dtype=torch.float32)
    props=scene.repeat(len(pool))
    with torch.inference_mode():
        geom=critic_features(props,boxes)
        appearance=np.asarray([field.score(box) for box in pool] if field is not None else [0.]*len(pool))
        blind=torch.zeros(len(pool))
        rgb=torch.tensor(appearance,dtype=torch.float32)
        energy=critic(geom,blind).numpy()
        rgb_energy=critic(geom,rgb).numpy()
    image=np.array([rectangle_iou(box,case) for box in pool])
    cost=np.array([ray_selection_cost(box,case) for box in pool])
    reads=dict(grid_aabb=int(image.argmax()),grid_ray=int(cost.argmin()),
               critic_geometry=int(energy.argmax()),critic_silhouette=int(rgb_energy.argmax()))
    return {k:pool[v].tolist() for k,v in reads.items()},pool.tolist(),dict(
        selected=reads,energy_geometry=energy.tolist(),energy_silhouette=rgb_energy.tolist())


def coherent_crop_points(frame,case):
    import minimal

    g=minimal.geometry
    spec,size,mat=g.camera(frame)
    uv,camera=g.raw_returns(frame,spec,size)
    return uv,camera,np.linalg.inv(mat),g


def crop_evidence(case,bank,rectangle,prior):
    uv,camera,inverse,g=bank
    rectangle=np.asarray(rectangle,float)
    all_ids,cells=g.z_buffer(uv,camera,np.flatnonzero(g.inside(uv,rectangle)))
    split=g.reserve(cells)
    central=g.inside(uv[all_ids],rectangle,.55)
    where=all_ids[central&(split==0)]
    if not len(where):
        where=all_ids[split==0]
    if not len(where):
        return coordinates(case['measured']),coordinates(case['selection'])
    depth=np.median(camera[where,2])
    keep=abs(camera[all_ids,2]-depth)<=max(.45,float(prior.max())*.7)
    points=g.rigid_xyz(camera[all_ids[(split==0)&keep]],inverse)
    rays=camera[all_ids[(split==1)&keep]]
    return sampled(points,240),sampled(rays,96)


def view_filter(proposals, probabilities, viewpoint, seed):
    # Merge directed front/rear and left/right when the box yaw is axial.
    end=sum(probabilities[:2]); side=sum(probabilities[2:4])
    if max(end,side)<=.55:
        return np.ones(len(proposals),dtype=bool)
    look=np.arctan2(viewpoint[1]-seed[1],viewpoint[0]-seed[0])
    delta=np.array([abs(angle(b[6]-look)) for b in proposals])
    allow=(delta<np.pi/4) if end>side else (delta>=np.pi/4)
    return allow if allow.any() else np.ones(len(proposals),dtype=bool)


def foveated(case,raw,frame):
    base=case['base']['box']
    initial=None if case['initial'] is None else np.asarray(case['initial'])
    if not raw:
        return dict.fromkeys(ARMS['fovea'],base),dict(crops=0,particles=0,view_drops=0)
    bank=coherent_crop_points(frame,case)
    eye=np.asarray(case['eye'])
    soft,hard,picks,weights,plain_weights=[],[],[],[],[]
    candidates_dropped=valid=0
    for crop in raw:
        for sample in crop['samples']:
            window=sample['window']
            if window is None:
                continue
            prior=np.asarray(sample['prior'] if sample['prior'] is not None else case['prior'])
            prior=np.r_[np.sort(prior[:2])[::-1],prior[2]]
            support,rays=crop_evidence(case,bank,window,prior)
            if not len(support) and initial is None:
                continue
            seed=np.median(support,axis=0) if len(support) else initial
            pose=headings(case,support)
            # Always include the alternate perpendicular yaw to expose end/side flips.
            yaws=[pose[0],pose[1],float(angle(pose[0]+np.pi/2))]
            observed=dict(case,measured=support.tolist(),selection=rays.tolist())
            above,above_hard=[],[]
            for yaw in yaws:
                for anchors,q in ((above,.10),(above_hard,0.)):
                    reference=anchored(support,eye,seed,1.18*prior,yaw,q)
                    proposal=np.r_[seed,prior,yaw]
                    anchors.append(interval(reference,observed).project(proposal))
            quality=np.asarray([rectangle_iou(b,case,window) for b in above])
            if not valid:
                picks=above[int(quality.argmax())].tolist()
            accept=view_filter(above,crop['view'],eye,seed)
            candidates_dropped+=int((~accept).sum())
            soft.extend(above); hard.extend(above_hard)
            plain_weights.extend((.5+quality).tolist())
            weights.extend(((.5+quality)*accept).tolist())
            valid+=1
    if not valid:
        return dict.fromkeys(ARMS['fovea'],base),dict(crops=len(raw),particles=0,view_drops=0)
    if base is not None:
        soft.append(base);hard.append(base)
        weights.append(1.);plain_weights.append(1.)
    # Decisions and their losses use the same nonzero-weight posterior support.
    subset=np.flatnonzero(np.asarray(weights)>0)
    selected=np.asarray(soft)[subset]
    hard_selected=np.asarray(hard)[subset]
    weight=np.asarray(weights)[subset]
    result=dict(crop_single=picks,crop_mbr=np.asarray(soft)[iou_mbr(soft,plain_weights)[0]].tolist(),
                crop_mbr_view=selected[iou_mbr(selected,weight)[0]].tolist(),
                crop_mbr_hard=hard_selected[iou_mbr(hard_selected,weight)[0]].tolist())
    return result,dict(crops=len(raw),particles=len(soft),valid_samples=valid,view_drops=candidates_dropped)


WORK={}


@torch.inference_mode()
def one(row):
    case,mode=row
    state=WORK
    uid=case['uid']; want=state['source'][uid]
    if mode=='fovea' and uid not in state['crop_roster']:
        return None
    base=case['base']['box']
    out=dict.fromkeys(ARMS[mode],base)
    record=dict(uid=uid,target=case['target'],base=base,boxes=out)
    # Construct mask only for modes that use image evidence.
    data=state['vit'].get(uid)
    field=Silhouette(data,case) if data is not None else None
    if mode=='fovea':
        record['boxes'],record['detail']=foveated(case,state['crops'].get(uid),want['frame'])
    elif mode=='vit' and field is not None:
        if len(case['fitted'])==2:
            scores=[field.score(x) for x in case['fitted']]
            if scores[0]!=scores[1]:
                out['vit_select']=case['fitted'][int(scores[1]>scores[0])]
        out['vit_bind_select']=new_bind(want['frame'],want['raw'],field)
        record['strict_pair']=matched_strict_pair(case,field)
        record['data']=dict(grid=data['grid'],foreground_seeds=data['foreground_seeds'],background_seeds=data['background_seeds'])
    elif mode in ('flow','critic'):
        scene=make_scene(case)
        if scene is not None:
            if mode=='flow':
                record['boxes'],record['detail']=flow_generate(state['flow'],scene,case)
            else:
                record['boxes'],record['pool'],record['detail']=critic_read(state['critic'],scene,case,field)
    return record


def verified(stem,receipt,key='payload'):
    record=read(RUN/receipt)
    assert record[key]==digest(RUN/stem),stem
    return read(RUN/stem)


def run(mode,limit):
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    stamp=read(RUN/'training-seal.json')
    assert not stamp['used_target_data'] and not stamp['qwen_updates']
    assert all(digest(HERE/f)==sha for f,sha in stamp['code'].items())
    source,*_=sealed_inputs()
    cases=verified('inputs-private.json','input-seal.json','evidence')
    WORK.clear()
    WORK.update(source=source,vit={},crops={},crop_roster=set())
    if mode in ('vit','critic'):
        WORK['vit']=verified('vit-masks-private.json','vit-seal.json')
    if mode=='fovea':
        WORK['crops']=verified('fovea-private.json','fovea-seal.json')
        WORK['crop_roster']=set(verified('fovea-roster-private.json','input-seal.json','fovea'))
        cases=[r for r in cases if r['uid'] in WORK['crop_roster']]
    if mode=='critic':
        WORK['critic']=load_network('critic',Critic,stamp)
    if mode=='flow':
        WORK['flow']=load_network('flow',Flow,stamp)
    start=perf_counter()
    if limit:
        cases=cases[:limit]
    results=[]
    with ProcessPoolExecutor(8) as workers:
        for out in workers.map(one,((r,mode) for r in cases),chunksize=2):
            assert out is not None
            results.append(out)
            if len(results)%100==0 or len(results)==len(cases):
                print(mode,len(results),'/',len(cases),flush=True)
    assert len(results)==len(cases)
    if limit:
        print('Smoke passed for',mode,flush=True)
        return
    record=RUN/f'{mode}-blind-private.json'
    write(record,results)
    write(RUN/f'{mode}-prediction-seal.json',dict(arms=ARMS[mode],counts=dict(Counter(r['target'] for r in cases)),
          labels_read=False,seconds=perf_counter()-start,payload=digest(record),
          training=digest(RUN/'training-seal.json') if mode in ('flow','critic') else None,
          code={name:digest(HERE/name) for name in ('blind_experiments.py','physical_readout.py','geometry_models.py')}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=ARMS)
    parser.add_argument('--smoke',type=int)
    args=parser.parse_args()
    run(args.mode,args.smoke)
