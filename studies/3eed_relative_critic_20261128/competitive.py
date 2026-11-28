"""Compete visibility-aligned, frozen visual prototypes over existing regions."""
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import torch
from torch.nn.functional import normalize

from source_geometry import HERE

sys.path.insert(0,str(HERE.parent/'3eed_four_paradigms_20261128'))
from physical_readout import Silhouette


MAX_REGIONS=4


def catalogue(sample,terrain,scales,fit_returns,in_window,selection):
    import cray_simple as physical

    g=physical.geometry
    radii,assigned,births=physical.region_catalog(sample,terrain,scales)
    if not len(births) or not in_window.any():
        return None
    mapped=g.transfer(sample,fit_returns,radii,assigned,births)
    scene=mapped[:,:len(in_window)]
    central=mapped[:,len(in_window):]
    view=scene[:,in_window]
    n=len(births)
    scene_counts=np.bincount(scene[scene>=0],minlength=n)
    hits=np.bincount(view[view>=0],minlength=n)
    ncentral=np.bincount(central[central>=0],minlength=n)
    good=(ncentral>=physical.MIN_RETURNS)&(hits>0)
    if not good.any():
        return None
    weight=hits.astype(float)**2/np.maximum(1,scene_counts)
    winner=int(np.argmax(np.where(good,weight,-1.)))
    keep=[winner]
    mask=lambda i:assigned[births[i]]==i
    for j in np.argsort(-weight,kind='stable'):
        j=int(j)
        if j==winner or hits[j]<3:
            continue
        bits=mask(j)
        if any((bits & mask(i)).sum()>.50*min(bits.sum(),mask(i).sum()) for i in keep):
            continue
        keep.append(j)
        if len(keep)==MAX_REGIONS:
            break
    return dict(sample=sample,radii=radii,assigned=assigned,births=births,selected=selection,
                fit=fit_returns[:len(in_window)][in_window],view=view,ids=keep,valid=good,weight=weight)


def stage_key(selection):
    return 'H' if selection is None else 'C'


def saved_regions(frame,replies):
    import minimal
    from geometry import sensor_pixels
    from prepare_blind import sampled

    geom=minimal.geometry
    spec,_,tf=geom.camera(frame)
    model_to_camera=geom.model_to_camera_matrix(spec)
    to_sensor=np.linalg.inv(tf)@model_to_camera
    storage={}
    vanilla=minimal.choose_region
    def track(sample,terrain,scales,fit_returns,in_window,selection=None):
        result=vanilla(sample,terrain,scales,fit_returns,in_window,selection)
        cat=catalogue(sample,terrain,scales,fit_returns,in_window,selection)
        if cat is None:
            assert result[0] is None
            return result
        first=cat['ids'][0]
        truth=cat['assigned'][cat['births'][first]]==first
        np.testing.assert_array_equal(sample[truth],result[0])
        votes=sensor_pixels(geom.rigid_xyz(cat['fit'],model_to_camera),spec)
        if votes is None:
            return result
        regions=[]
        for index in cat['ids']:
            level=cat['births'][index]
            cloud=sample[cat['assigned'][level]==index]
            cloud=sampled(cloud,96)
            p=sensor_pixels(geom.rigid_xyz(cloud,model_to_camera),spec)
            if p is None:
                continue
            regions.append(dict(id=index,seeds=p.tolist(),xyz=geom.rigid_xyz(cloud,to_sensor).tolist(),
                                fit_votes=sampled(votes[cat['view'][level]==index],72).tolist(),
                                allowed=bool(cat['valid'][index]),weight=float(cat['weight'][index])))
        if regions and regions[0]['id']==first:
            storage[stage_key(selection)]=regions
        return result
    with patch.object(minimal,'choose_region',track):
        minimal.predict(frame,replies)
    return storage


def patch_ids(uv,size,shape):
    h,w=shape
    xy=(np.asarray(uv).reshape(-1,2)/np.asarray(size)*np.array([w,h])).astype(int)
    xy=xy.clip([0,0],[w-1,h-1])
    return xy[:,1]*w+xy[:,0]


def probability_fields(features,case,regions,*,matched_primary):
    h,w,_=features.shape
    yy,xx=np.mgrid[:h,:w]
    grid=np.stack(((xx+.5)/w,(yy+.5)/h),-1).reshape(-1,2)
    rect=np.asarray(case['window'])/np.tile(case['size'],2)
    distance=(abs(grid-(rect[:2]+rect[2:])/2)/((rect[2:]-rect[:2])/2)).max(1)
    ring=np.flatnonzero((distance>1)&(distance<=1.55))
    if len(ring)<4:
        ring=np.flatnonzero(distance>1.05)
    if len(ring)<4:
        return None
    ids=[]
    good=[]
    for i,region in enumerate(regions):
        p=np.asarray(case['marks'] if i==0 and matched_primary else region['seeds'])
        if not len(p):
            continue
        half=(np.asarray(case['window'][2:])-case['window'][:2])/2
        center=np.asarray(case['window'][:2])+half
        q=p[(abs(p-center)/half).max(1)<=1.35]
        if not len(q):
            continue
        g=np.unique(patch_ids(q,case['size'],(h,w)))
        ids.append(g);good.append(region)
    if len(good)<2 or good[0]['id']!=regions[0]['id']:
        return None
    em=features.flatten(0,1).float()
    em=normalize(em-em.mean(0),dim=-1)
    prototypes=torch.stack([normalize(em[g].mean(0),dim=0) for g in (*ids,ring)],1)
    logits=8*(em@prototypes)
    same=logits.softmax(1)
    compact=same[:,0].masked_fill(torch.as_tensor(distance>1.15,device=em.device),0.)
    wide=same[:,:len(good)].masked_fill(torch.as_tensor(distance[:,None]>1.50,device=em.device),0.)
    admission=[]
    for col,region in enumerate(good):
        px=patch_ids(region['fit_votes'],case['size'],(h,w))
        mass=float(wide[px,col].mean()) if len(px) else 0.
        admission.append([region['id'],region['weight']*(.25+.75*mass)**2,region['allowed']])
    # Keep every returned field per candidate separate from background.
    return dict(ids=[r['id'] for r in good],compact=compact.reshape(h,w).cpu().numpy().round(5).tolist(),
                masks=wide.T.reshape(len(good),h,w).cpu().numpy().round(5).tolist(),admission=admission)


def heads(readout,stages):
    wanted='C' if readout=='C_ray' else 'H'
    return stages.get(wanted) or stages.get('C') or stages.get('H')


def silhouettes(case,visual):
    stages=visual.get('stages',{})
    chosen=heads(case['base']['readout'],stages)
    legacy=visual.get('legacy')
    old=Silhouette(legacy,case) if legacy is not None else None
    if chosen is None:
        return old,None,[]
    compact=Silhouette(dict(mask=chosen['compact']),case)
    return old,compact,[Silhouette(dict(mask=value),case) for value in chosen['masks']]


def rebind(frame,replies,stages):
    """Change the existing region vote and leave the original pose readout active."""
    import minimal
    old=minimal.choose_region
    def choose(sample,terrain,scales,fit_returns,in_window,selection=None):
        stage=stages.get(stage_key(selection))
        if stage is None:
            return old(sample,terrain,scales,fit_returns,in_window,selection)
        cat=catalogue(sample,terrain,scales,fit_returns,in_window,selection)
        if cat is None:
            return old(sample,terrain,scales,fit_returns,in_window,selection)
        weights={idx:weight for idx,weight,allowed in stage['admission'] if allowed and cat['valid'][idx]}
        if not weights or not set(weights)<=set(cat['ids']):
            return old(sample,terrain,scales,fit_returns,in_window,selection)
        i=max(weights,key=weights.get)
        level=cat['births'][i]
        region=sample[cat['assigned'][level]==i]
        return (region,None) if selection is None else (region,minimal.geometry.transfer(
             sample,selection,cat['radii'],cat['assigned'],cat['births'])[level]==i)
    with patch.object(minimal,'choose_region',choose):
        return minimal.predict(frame,replies)
