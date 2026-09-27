"""F23 isolated temporal capacity diagnostic. No dataset/GT reader here.

Labels, when explicitly provided for an oracle cell, are endpoint targets only.
No production module is changed. Native temporal graph is replayed twice.
"""
import copy
import math
import time

import numpy as np
import torch

from vg_tta.decota_tastvg_episode_v1 import fitted_merge, native_view_indices
from vg_tta.temporal_optimizer_probe_v1 import cpu_state, logits as head_logits
from methods.decota_v1.api import decode as final_decode


def detached(x):
    if torch.is_tensor(x): return x.detach().clone()
    if isinstance(x, dict): return {k: detached(v) for k,v in x.items()}
    if isinstance(x, tuple): return tuple(detached(v) for v in x)
    if isinstance(x, list): return [detached(v) for v in x]
    return copy.deepcopy(x)


def capture(model, batch):
    from vg_tta.decota_tastvg_episode_v1 import forward
    calls=[];grounds=[];all_outputs=[]
    hooks=[model.ground_decoder.time_decoder.register_forward_pre_hook(
        lambda m,a,k: calls.append(detached(k)), with_kwargs=True),
        model.ground_decoder.register_forward_pre_hook(
        lambda m,a,k: grounds.append(detached(k)), with_kwargs=True),
        model.register_forward_hook(lambda m,a,o: all_outputs.append(detached(o['att_sequences'])))]
    try: base,inputs,records=forward(model,batch)
    finally:
        for h in hooks: h.remove()
    assert len(calls)==len(grounds)==4 and len(all_outputs)==2
    views=[]
    for j in range(2):
        kw=calls[2*j];second=calls[2*j+1];enc=grounds[2*j]['encoded_info']
        for k in kw:
            if k!='query_tgt':
                assert torch.equal(kw[k],second[k]), ('non-query temporal input changed',k)
        h,w=enc['fea_map_size'];n=kw['query_tgt'].shape[0];features=enc['encoded_feature'];l=h*w
        fm=features[-l:].permute(1,2,0).reshape(n,256,h,w).detach()
        text=features[l:-l].mean(1).unsqueeze(0).detach()
        fallback=torch.nonzero(all_outputs[j].squeeze()>0).squeeze().tolist()
        if isinstance(fallback,int): fallback=[fallback]
        views.append(dict(kwargs=kw,second_kwargs=second,motion=fm,text=text,
            encoded_motion=features[-l:].permute(1,0,2),fallback=fallback))
    return base,inputs,records,views


class Replay:
    def __init__(self, model, inputs, views, scope):
        assert scope in ('head','head_ln')
        self.scope=scope;self.model=model;self.inputs=inputs;self.views=views
        self.head=copy.deepcopy(model.temp_embed).eval().requires_grad_(True)
        self.decoder=None
        self.named=[('temp_embed.'+n,p) for n,p in self.head.named_parameters()]
        if scope=='head_ln':
            self.decoder=copy.deepcopy(model.ground_decoder.time_decoder).eval().requires_grad_(False)
            i=len(self.decoder.layers)-1
            for n,m in self.decoder.layers[-1].named_modules():
                if isinstance(m,torch.nn.LayerNorm):
                    m.requires_grad_(True)
                    for k,p in m.named_parameters(recurse=False):
                        self.named.append((f'ground_decoder.time_decoder.layers.{i}.{n}.{k}',p))
            assert len(self.named)==len(list(self.head.parameters()))+6
        self.initial={n:p.detach().clone() for n,p in self.named}
        self.source=cpu_state(model.temp_embed)
        self.gates=[]

    def state(self): return {n:p.detach().cpu().clone() for n,p in self.named}

    def restore(self,state):
        with torch.no_grad():
            for n,p in self.named: p.copy_(state[n].to(p.device))

    def reset(self):
        self.restore(self.initial)
        for _,p in self.named: p.grad=None

    def values(self):
        if self.scope=='head': return head_logits(self.head,self.inputs)
        values=[];self.gates=[]
        for view in self.views:
            with torch.autocast('cuda',dtype=torch.float16):
                # Native gate has no differentiable path. Recompute its VALUE.
                with torch.no_grad():
                    first=self.decoder(**view['kwargs'])
                    action=self.model.action_embed(first)[-1].squeeze().sigmoid()
                    chosen=torch.nonzero((action>0.5).int()).squeeze().tolist()
                    if isinstance(chosen,int): chosen=[chosen]
                    chosen=chosen or list(view['fallback'])
                    _,att=self.model.t_spatial_clas(view['motion'][chosen],view['text'][:,:1])
                    itq=(view['encoded_motion'][chosen]*att.unsqueeze(2)).mean((0,1))
                kw=dict(view['kwargs']);n=kw['query_tgt'].shape[0]
                kw['query_tgt']=itq[None,None,:].expand(n,1,256)
                hidden=self.decoder(**kw)
                values.append(self.head(hidden)[-1]);self.gates.append(chosen)
        assert all(torch.isfinite(z).all() for z in values),'nonfinite logits'
        return values


def quantize(interval,records,ids):
    """Inward nearest legal offset pairs; quantify unavoidable grid mismatch."""
    start,end=interval;assert end>start
    pairs=[];audit=[];ms=[];me=[]
    for r in records:
        a=np.asarray(r['frame_ids']);n=len(a)
        legal=np.triu(np.ones((n,n),bool),1)
        inside=(a>=start)&(a+1<=end)
        restricted=bool(inside.sum()>=2)
        mask=legal&(inside[:,None]&inside[None,:]) if restricted else legal
        err=np.abs(a[:,None]-start)+np.abs(a[None,:]+1-end)
        s,e=np.unravel_index(np.argmin(np.where(mask,err,np.inf)),err.shape)
        pairs.append([int(s),int(e)]);ms.append(ids.index(int(a[s])));me.append(ids.index(int(a[e])))
        audit.append(dict(offset=r['offset'],indices=[int(s),int(e)],interval=[int(a[s]),int(a[e]+1)],
            signed_error=[float(a[s]-start),float(a[e]+1-end)],inward_restricted=restricted))
    merged=[min(ms),max(me)]
    return pairs,dict(target=list(map(float,interval)),offsets=audit,merged=merged,
        merged_interval=[ids[merged[0]],ids[merged[1]]+1])


def objective(zs,targets,kind):
    loss=[]
    for z,(s,e) in zip(zs,targets):
        lp=z.float().log_softmax(1)
        loss.append(-(lp[0,s,0]+lp[0,e,1])*(1. if kind=='f5_fit' else .5))
    # Same order and differentiable zero as production for coverage.
    return torch.stack(loss).mean()


def readout(zs,records,ids,base):
    tmp=list(fitted_merge(zs,records,ids))
    fin=list(final_decode(base['temporal_logits'],tmp,ids,native_indices=base['predicted_indices'])['indices'])
    return dict(temporary=tmp,final=fin,offsets=[list(native_view_indices(z)) for z in zs])


def run_trajectory(replay,records,ids,base,kind,targets,lr,steps=20):
    replay.reset();opt=torch.optim.AdamW([p for _,p in replay.named],lr=lr,eps=1e-4,weight_decay=0.)
    rows=[];states={};bests={};best_loss=math.inf;best=None;failure=None
    started=time.perf_counter();base_gates=None;backwards=0
    try:
        for step in range(steps+1):
            zs=replay.values();loss=objective(zs,targets,kind)+next(replay.head.parameters()).float().sum()*0.
            value=float(loss.detach());assert math.isfinite(value),'nonfinite loss'
            decoded=readout([z.detach() for z in zs],records,ids,base)
            relative={}
            for tag in ('head','ln'):
                pp=[(n,p) for n,p in replay.named if n.startswith('temp_embed.')==(tag=='head')]
                if pp:
                    d=sum(float(((p.detach()-replay.initial[n])**2).sum()) for n,p in pp)
                    denom=sum(float((replay.initial[n]**2).sum()) for n,p in pp)
                    relative[tag]=math.sqrt(d/max(denom,1e-30))
            if base_gates is None: base_gates=copy.deepcopy(replay.gates)
            row=dict(step=step,loss=value,**decoded,relative_displacement=relative,
                gate_changed=replay.gates!=base_gates,seconds=time.perf_counter()-started)
            rows.append(row)
            if value<best_loss:
                best_loss=value;best=step;best_state=replay.state()
            if step in (5,20) or step==steps:
                bests[str(step)]=best;states[str(step)]=copy.deepcopy(best_state)
            if step<steps:
                opt.zero_grad(set_to_none=True);loss.backward();backwards+=1
                grads={n:float(p.grad.norm()) if p.grad is not None else None for n,p in replay.named}
                row['gradient_norms']=grads
                assert all(v is not None and math.isfinite(v) for v in grads.values()),'missing/nonfinite gradient'
                opt.step()
                assert all(torch.isfinite(p).all() for _,p in replay.named),'nonfinite parameter'
    except (AssertionError,RuntimeError,ValueError) as exc:
        # Shape/graph failures must not be swallowed as optimizer instability.
        if not any(k in str(exc) for k in ('nonfinite','finite floating','logits must be finite')): raise
        failure=dict(step=step,message=str(exc))
        for budget in (5,20):
            if str(budget) not in bests:
                bests[str(budget)]=best;states[str(budget)]=copy.deepcopy(best_state)
    restored={}
    for budget,state in states.items():
        replay.restore(state)
        with torch.no_grad():
            zs=replay.values();v=float(objective(zs,targets,kind))
        reference=rows[bests[budget]];assert v==reference['loss'],('restore loss',v,reference['loss'])
        assert readout(zs,records,ids,base)=={k:reference[k] for k in ('temporary','final','offsets')}
        assert all(torch.equal(replay.state()[k],v) for k,v in state.items())
        restored[budget]=True
    replay.reset()
    assert all(torch.equal(v,cpu_state(replay.model.temp_embed)[k]) for k,v in replay.source.items())
    return dict(kind=kind,scope=replay.scope,lr=lr,states=rows,best_steps=bests,
        saved_best_states=states,restored_exact=restored,failure=failure,backwards=backwards,
        seconds=time.perf_counter()-started,GT_used=kind=='gt_oracle')
