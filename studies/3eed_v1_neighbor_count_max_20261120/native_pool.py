"""One branch intervention with explicit raw-return provenance and an exact frozen suffix."""
from contextlib import ExitStack
import hashlib
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
RAY=HERE.parent/'3eed_v1_language_displacement_rays_20261120'
sys.path.insert(0,str(RAY))
import capture as legacy
from scope183 import atomic, digest, read

WORK=HERE/'run_PRIVATE'
OUT=HERE/'results'
CFG=read(HERE/'PROTOCOL.json')
PLATFORMS=('waymo','drone','quad')
MASKS=('native','n4_211','n4_947','n8_211','n8_947','structured4')


def save_npz(path,**values):
    with path.with_suffix('.tmp').open('wb') as fp:
        np.savez_compressed(fp,**values)
    path.with_suffix('.tmp').replace(path)


def pinned_rows(src):
    folder=legacy.SUP/'inputs_PRIVATE'
    manifest=read(folder/'SEAL.json')['files']
    assert digest(folder/'SEAL.json')==CFG['source_inventory_sha256']
    fit,hold=[],[]
    for dest,split in ((fit,'fit'),(hold,'calibration')):
        name=f'{src}_{split}.json'
        assert digest(folder/name)==manifest[name]
        dest.extend(read(folder/name))
        assert all(r['platform']==src for r in dest)
    rec=lambda r: str(Path(r['metapath']).resolve().parent.parent)
    assert not ({rec(r) for r in fit}&{rec(r) for r in hold})
    assert not ({r['point_sha256'] for r in fit}&{r['point_sha256'] for r in hold})
    return fit,hold


def sample_ids(row,example):
    from runner_helpers import sample_points
    from utils.transform_waymo import transform_to_front_view
    assert digest(Path(row['pointpath']))==row['point_sha256']
    xyz, ids, raw_size=sample_points(dict(platform=row['platform'],pointpath=row['pointpath'],seed=example['seed']))
    if row['platform']=='waymo':
        xyz,_=transform_to_front_view(xyz,np.empty((0,7)),example['waymo_view'])
    assert len(ids)==16384 and min(ids)>=0 and max(ids)<raw_size
    return torch.as_tensor(np.ascontiguousarray(xyz,dtype=np.float32),device='cuda'),torch.as_tensor(ids,device='cuda')


class NativePool:
    def __init__(self,model,points,ids):
        self.model,self.points,self.ids=model,points,ids
        self.sa=model.backbone_net.SA_modules[0]
        grouper=self.sa.groupers[0]
        assert grouper.nsample==16 and grouper.radius==.2 and self.sa.pool_method=='max_pool'
        assert len(self.sa.groupers)==2 and all(not m.training for m in model.modules())
        self.positions=[]

    def hooks(self,stack):
        from ops.teed_pointnet.pointnet2_batch import pointnet2_utils as util
        fps,ball=util.furthest_point_sample,util.ball_query
        def actual_fps(xyz,n):
            result=fps(xyz,n)
            if n==2048 and xyz.shape[1]==16384:
                assert not hasattr(self,'centers')
                torch.testing.assert_close(xyz[0],self.points,atol=0,rtol=0)
                self.centers=result[0].long()
            return result
        def actual_ball(radius,n,xyz,centers):
            result=ball(radius,n,xyz,centers)
            if radius==.2 and n==16 and centers.shape[1]==2048:
                assert not hasattr(self,'indices')
                torch.testing.assert_close(centers[0],self.points[self.centers],atol=0,rtol=0)
                self.indices=result[0].long()
            return result
        def record_feature(_,args,out):
            self.activation=out[0].permute(1,2,0).contiguous()
        def record_pool(_,args):
            self.pooled=args[0][:,:32].contiguous()
        def position(_,args):
            if len(self.positions)<6:
                self.positions.append(args[3])
        def record_raw(_,args,out):
            self.input=args[0]
            self.raw=out
        stack.enter_context(patch.object(util,'furthest_point_sample',actual_fps))
        stack.enter_context(patch.object(util,'ball_query',actual_ball))
        stack.enter_context(self.sa.mlps[0].register_forward_hook(record_feature))
        stack.enter_context(self.sa.aggregation_layer.register_forward_pre_hook(record_pool))
        for layer in self.model.decoder:
            stack.enter_context(layer.register_forward_pre_hook(position))
        stack.enter_context(self.model.register_forward_hook(record_raw))

    def identify(self):
        group=self.ids[self.indices]
        center=self.ids[self.centers]
        sorted_ids=group.sort(1).values
        self.count=1+(sorted_ids[:,1:]!=sorted_ids[:,:-1]).sum(1)
        self.center_slots=group==center[:,None]
        self.has_center=self.center_slots.any(1)
        self.full=(self.count==16)&self.has_center
        self.native=self.activation.amax(1)
        assert self.activation.shape==(2048,16,32) and len(self.positions)==6
        torch.testing.assert_close(self.pooled,self.native.T.unsqueeze(0),atol=0,rtol=0)
        # The first occurrence of each actual return and a literal extra copy
        # of the first return both have exactly the same native maximum.
        order=torch.arange(16,device=group.device)
        seen=((group[:,:,None]==group[:,None,:])&(order[None,:]<order[:,None])).any(2)
        unique=self.activation.masked_fill(seen[:,:,None],-torch.inf).amax(1)
        torch.testing.assert_close(unique,self.native,atol=0,rtol=0)
        torch.testing.assert_close(torch.maximum(unique,self.activation[:,0]),self.native,atol=0,rtol=0)

    def thinning(self,mode,frame_hash):
        if mode=='native':
            return self.native,self.count
        n=4 if '4' in mode.split('_')[0] else 8
        groups=torch.nonzero(self.full).flatten()
        output=self.native.clone()
        counts=self.count.clone()
        if not len(groups):
            return output,counts
        if mode.startswith('structured'):
            # Source views only rotate xy for Waymo and translate z for Drone;
            # use the physical sensor origin before that z translation.
            p=self.points[self.indices[groups]].clone()
            p[...,2]-=self.origin_z
            priority=p.square().sum(-1)
        else:
            seed=int.from_bytes(hashlib.sha256((frame_hash+'|'+mode.split('_')[1]).encode()).digest()[:8],'little')
            priority=torch.as_tensor(np.random.default_rng(seed).random((len(groups),16)),device=groups.device)
        priority=priority.masked_fill(self.center_slots[groups],-torch.inf)
        keep=torch.argsort(priority,dim=1,stable=True)[:,:n]
        voxels=self.activation[groups[:,None],keep]
        output[groups]=voxels.amax(1)
        counts[groups]=n
        assert (output<=self.native).all()
        return output,counts

    def odds(self,raw,slot_ids,mask):
        from model51 import native_matching
        return native_matching(raw['last_proj_queries'],raw['proj_tokens'],mask)[1][slot_ids]

    def replay(self,new_pool,slate,mask):
        raw=self.raw
        def first32(_,args):
            baseline=args[0]
            torch.testing.assert_close(baseline[:,:32],self.pooled,atol=0,rtol=0)
            return (torch.cat((new_pool.T[None],baseline[:,32:]),dim=1),)
        def fix_ids(_,args):
            torch.testing.assert_close(args[0],raw['fp2_xyz'],atol=0,rtol=0)
            return args[0],args[1],raw['query_points_sample_inds']
        def set_position(pos):
            return lambda _,args: (*args[:3],pos,*args[4:])
        with ExitStack() as stack:
            stack.enter_context(self.sa.aggregation_layer.register_forward_pre_hook(first32))
            stack.enter_context(self.model.gsample_module.register_forward_pre_hook(fix_ids))
            for layer,pos in zip(self.model.decoder,self.positions):
                stack.enter_context(layer.register_forward_pre_hook(set_position(pos)))
            result=self.model(self.input)
        torch.testing.assert_close(result['query_points_sample_inds'],raw['query_points_sample_inds'],atol=0,rtol=0)
        torch.testing.assert_close(result['query_points_xyz'],raw['query_points_xyz'],atol=0,rtol=0)
        return self.odds(result,slate,mask)


@torch.inference_mode()
def capture_case(grounder,row,ranking):
    from model51 import firstnoun_mask
    example=legacy.observation(row)
    points,ids=sample_ids(row,example)
    model=grounder._frontend.base
    pool=NativePool(model,points,ids)
    pool.origin_z=1.8 if row['platform']=='drone' else 0.
    with ExitStack() as stack:
        pool.hooks(stack)
        result=grounder.collect(example,ranking=ranking)
    pool.identify()
    slate=torch.as_tensor(result['slate'],device=points.device,dtype=torch.long)
    mask=firstnoun_mask(model,pool.input['text'][0])[0].to(points.device)
    odds=pool.odds(pool.raw,slate,mask)
    assert len(torch.unique(slate))==16
    return pool,result,slate,mask,odds


def membership(centers,boxes):
    delta=centers[None]-boxes[:,None,:3]
    cos,sin=np.cos(boxes[:,6:7]),np.sin(boxes[:,6:7])
    local=np.stack((delta[:,:,0]*cos+delta[:,:,1]*sin,delta[:,:,1]*cos-delta[:,:,0]*sin,delta[:,:,2]),axis=-1)
    return (abs(local)<=boxes[:,None,3:6]/2+1e-7).all(-1)
