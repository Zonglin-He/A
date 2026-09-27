"""Versioned diagnostic interventions. Released DeCoTA code is not imported/mutated.

All queries/frames/position encodings remain present. Only selected attention
keys are perturbed. GT arguments denote diagnostic oracles, never legal TTA.
"""
import math
from contextlib import ExitStack
import numpy as np
import torch
from vg_tta.expert_space_time_probe_v1 import tree


class KeyIntervention:
    def __init__(self, modules, *, allowed=None, roi=None, visual_start=0,
                 gain=4., mode='hard', zero=False):
        self.modules=list(modules); self.allowed=allowed; self.roi=roi
        self.visual_start=visual_start; self.gain=gain; self.mode=mode
        self.zero=zero; self.handles=[]; self.audit=[]

    def pre(self,m,args,kw):
        if self.zero:return
        q=args[0] if args else kw['query']; k=args[1] if len(args)>1 else kw['key']
        out=dict(kw); assert kw.get('attn_mask') is None
        if self.allowed is not None:
            allowed=torch.as_tensor(self.allowed,device=q.device,dtype=torch.bool)
            assert q.shape[:2]==(len(allowed),1) and k.shape==q.shape and allowed.any()
            n=len(allowed)
            if self.mode=='self': mask=~torch.eye(n,device=q.device,dtype=torch.bool)
            elif self.mode=='soft': mask=(~allowed).float()[None].expand(n,n)*-math.log(self.gain)
            else: mask=(~allowed)[None].expand(n,n)
            out['attn_mask']=mask
            if mask.dtype!=torch.bool and kw.get('key_padding_mask') is not None:
                out['key_padding_mask']=torch.zeros_like(kw['key_padding_mask'],dtype=mask.dtype).masked_fill(kw['key_padding_mask'],-torch.inf)
            self.audit.append(dict(kind='temporal_keys',queries=n,allowed=int(allowed.sum()),mode=self.mode))
        else:
            roi=torch.as_tensor(self.roi,device=q.device,dtype=torch.bool)
            t,v=roi.shape; assert q.shape[:2]==(1,t) and k.shape[1]==t
            start=self.visual_start if self.visual_start>=0 else k.shape[0]+self.visual_start
            assert 0<=start and start+v<=k.shape[0]
            pad=kw.get('key_padding_mask'); assert pad is not None and pad.dtype==torch.bool
            assert not pad[:,start:start+v].any()
            bias=torch.zeros_like(pad,dtype=torch.float32)
            add=roi.float()*math.log(self.gain)
            if self.mode=='uniform':add=torch.log1p(roi.float().mean(1,keepdim=True)*(self.gain-1)).expand_as(add)
            bias[:,start:start+v]=add
            out['key_padding_mask']=bias.masked_fill(pad,-torch.inf)
            self.audit.append(dict(kind='spatial_keys',frames=t,active=int(roi.any(1).sum()),tokens=int(roi.sum()),visual_start=start,gain=self.gain))
        return args,out

    def __enter__(self):
        for m in self.modules:self.handles.append(m.register_forward_pre_hook(self.pre,with_kwargs=True))
        return self

    def __exit__(self,*args):
        for h in self.handles:h.remove()


def choose_layers(layers,stage):
    if stage=='all':return list(layers)
    n=len(layers); groups={'early':range(0,max(1,n//3)), 'middle':range(n//3,2*n//3), 'late':range(2*n//3,n)}
    return [layers[i] for i in groups[stage]]


def center_roi(centers,grid,fraction=.25):
    h,w=grid; centers=torch.as_tensor(centers,dtype=torch.float32)
    yy,xx=torch.meshgrid((torch.arange(h)+.5)/h,(torch.arange(w)+.5)/w,indexing='ij')
    xy=torch.stack([xx.flatten(),yy.flatten()],1)
    k=max(1,math.ceil(h*w*fraction)); indices=((xy[None]-centers[:,None])**2).sum(-1).argsort(dim=1,stable=True)[:,:k]
    out=torch.zeros(len(centers),h*w,dtype=torch.bool);out.scatter_(1,indices,True)
    return out


class TubeReplay:
    def __init__(self,m,raw,ids,q):
        from vg_tta import tubedetr_runtime as rt
        from datasets.video_transforms import make_video_transforms
        from scripts.decota_matrix_common_v1 import ROOT
        self.m=m; self.ids=ids; self.q=q; self.rt=rt
        video,_=make_video_transforms('val',cautious=True,resolution=224)(raw,None)
        shapes=[];h=m.input_proj.register_forward_hook(lambda _,a,o:shapes.append(tuple(o.shape[-2:])))
        try:
            with torch.no_grad():self.memory=rt.encode_video(m,video,q['caption'],repo=ROOT/'external/TubeDETR',stride=2,device='cuda:0')
        finally:h.remove()
        self.grid=shapes[-1];self.grids=[self.grid];self.offsets=[list(range(len(ids)))];self.base=self.run()

    def run(self,spec=None):
        spec=spec or {};layers=choose_layers(self.m.transformer.decoder.layers,spec.get('stage','all'))
        cm=ExitStack();audit=[]
        if 'allowed' in spec:
            i=cm.enter_context(KeyIntervention([x.self_attn for x in layers],allowed=spec['allowed'],mode=spec.get('mode','hard'),zero=spec.get('zero',False)));audit=i.audit
        if 'roi' in spec:
            i=cm.enter_context(KeyIntervention([x.cross_attn_image for x in layers],roi=spec['roi'][0],gain=spec.get('gain',4.),mode=spec.get('mode','hard'),zero=spec.get('zero',False)));audit=i.audit
        try:
            with torch.no_grad():o,h=self.rt.decode_video_with_temporal_head_input(self.m,self.memory,duration=len(self.ids),caption=self.q['caption'],device='cuda:0')
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                zs=self.m.sted_embed(h)[:,0];bs=self.m.bbox_embed(h).sigmoid()[:,0]
        finally:cm.close()
        from vg_tta.metrics import interval_from_logits
        z=o['pred_sted'][0].float().cpu();b=o['pred_boxes'].float().cpu()
        return dict(boxes=b,logits=z,indices=list(interval_from_logits(z)),layer_boxes=bs.float().cpu(),layer_logits=zs.float().cpu(),audit=audit)


class TAReplay:
    def __init__(self,m,raw,ids,q,subject):
        from vg_tta.decota_tastvg_episode_v1 import make_batch,forward
        self.m=m;self.ids=ids;batch=make_batch(raw,ids,q,subject,m);caps=[];outs=[]
        h=m.ground_decoder.register_forward_pre_hook(lambda _,a,kw:caps.append(tree(kw)),with_kwargs=True)
        ho=m.ground_decoder.register_forward_hook(lambda _,a,o:outs.append(tree(o)))
        try:base,_,records=forward(m,batch)
        finally:h.remove();ho.remove()
        assert len(caps)==4
        self.caps=[caps[1],caps[3]];self.records=records
        self.offsets=[list(range(o,len(ids),2)) for o in [0,1]]
        self.grids=[tuple(c['encoded_info']['fea_map_size']) for c in self.caps]
        self.base=self.run()
        assert torch.equal(self.base['boxes'],base['raw_boxes'].float())
        assert torch.equal(self.base['logits'],base['temporal_logits'].float())
        assert self.base['indices']==list(base['predicted_indices'])

    def run(self,spec=None):
        from vg_tta.expert_space_time_probe_v1 import combine
        spec=spec or {};zs=[];bs=[];lzs=[];lbs=[];audits=[]
        for off,(cap,positions,grid) in enumerate(zip(self.caps,self.offsets,self.grids)):
            cm=ExitStack();branch=spec.get('branch','spatial')
            if 'allowed' in spec:
                ls=self.m.ground_decoder.decoder.layers if branch=='spatial' else self.m.ground_decoder.time_decoder.layers
                layers=choose_layers(ls,spec.get('stage','all'));allowed=np.asarray(spec['allowed'])[positions]
                if allowed.any():
                    i=cm.enter_context(KeyIntervention([x.self_attn for x in layers],allowed=allowed,mode=spec.get('mode','hard'),zero=spec.get('zero',False)));audits.append(i.audit)
                else:audits.append([dict(skipped='no allowed key in offset',offset=off)])
            if 'roi' in spec:
                layers=choose_layers(self.m.ground_decoder.time_decoder.layers,spec.get('stage','all'))
                i=cm.enter_context(KeyIntervention([x.cross_attn_image for x in layers],roi=spec['roi'][off],visual_start=-grid[0]*grid[1],gain=spec.get('gain',4.),mode=spec.get('mode','hard'),zero=spec.get('zero',False)));audits.append(i.audit)
            try:
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
                    bp,hh=self.m.ground_decoder(**cap);zz=self.m.temp_embed(hh)[:,0]
                    bb=bp[:,0]
                zs.append(zz[-1][None].detach().cpu());bs.append(bb[-1].detach().float().cpu())
                lzs.append(zz.detach().float().cpu());lbs.append(bb.detach().float().cpu())
            finally:cm.close()
        z,ij=combine(zs,self.records,self.ids)
        merge=lambda v:torch.stack([v[t%2][t//2] for t in range(len(self.ids))])
        return dict(boxes=merge(bs),logits=z,indices=ij,layer_boxes=torch.stack([merge([x[l] for x in lbs]) for l in range(lbs[0].shape[0])]),
                    layer_logits=torch.stack([merge([x[l] for x in lzs]) for l in range(lzs[0].shape[0])]),audit=audits)
