"""Conditional A/B/C CPU-tested building blocks. Not wired into a live model.

No expert, target, optimizer launch, best-step selection, or data access here.
All magnitudes are relative to ||F||, not absolute token-space lengths.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

STATE_DIM=33

def compressed_native_state(*,time_logits,interval,positions,boxes_xyxy,geometry_valid,
                            coordinate_logits,coordinate_token_ids,evidence_boxes,evidence_known,frames):
    """Fixed [T,33] state, full-vocabulary statistics, explicit missing support.

    Inputs are detached native observations at their real frame anchors. The
    external/native caller must supply the actual generated coordinate IDs;
    neither restricted-1001 probabilities nor GT IDs may be substituted.
    Evidence boxes may be source oracle or external, with missing known=False.
    No vocabulary logits leave this function. Unknown anchors have validity0.
    """
    if frames<1:raise ValueError('Empty time support')
    device=boxes_xyxy.device;out=torch.zeros(frames,STATE_DIM,device=device)
    with torch.no_grad():
        if time_logits is not None:
            if time_logits.shape!=(2,frames) or not torch.isfinite(time_logits).all():raise ValueError('Native endpoint support')
            p=time_logits.detach().float().softmax(-1);out[:,0:2]=p.T
            top=p.topk(min(2,frames),-1).values
            margins=top[:,0]-top[:,-1] if frames>1 else torch.ones(2,device=device)
            entropy=-(p*p.clamp_min(1e-30).log()).sum(-1)/max(math.log(frames),1)
            out[:,5:7]=margins;out[:,7:9]=entropy;out[:,9]=1
        if interval is not None:
            s,e=interval
            if not 0<=s<=e<frames:raise ValueError('Native interval out of support')
            out[s:e+1,2]=1;out[:,3]=s/max(frames-1,1);out[:,4]=e/max(frames-1,1)
        n=len(positions)
        if boxes_xyxy.shape!=(n,4) or geometry_valid.shape!=(n,) or evidence_boxes.shape!=(frames,4) or evidence_known.shape!=(frames,):raise ValueError('Physical anchor/evidence support')
        if len(set(positions))!=n or any(i<0 or i>=frames for i in positions):raise ValueError('Nonunique native anchors')
        for x in (boxes_xyxy,evidence_boxes):
            if not torch.isfinite(x).all():raise ValueError('Nonfinite boxes')
        if n:
            if coordinate_logits is None or coordinate_token_ids is None:raise ValueError('Need actual coordinate policy support, not fabricated zeros')
            if coordinate_logits.shape[:2]!=(n,4) or coordinate_token_ids.shape!=(n,4):raise ValueError('Coordinate action support')
            if not torch.isfinite(coordinate_logits).all():raise ValueError('Nonfinite native logits')
            c=coordinate_logits.detach().float();ids=coordinate_token_ids.detach().long()
            if (ids<0).any() or (ids>=c.shape[-1]).any():raise ValueError('Generated token outside full vocabulary')
            # The actual selected token, not argmax over unrestricted support.
            probs=c.softmax(-1);chosen=probs.gather(-1,ids[...,None]).squeeze(-1)
            other=probs.clone();other.scatter_(-1,ids[...,None],-1)
            margins=chosen-other.max(-1).values
            entropy=-(probs*probs.clamp_min(1e-30).log()).sum(-1)/max(math.log(c.shape[-1]),1)
            out[positions,10:14]=boxes_xyxy;out[positions,14:18]=chosen
            out[positions,18:22]=margins;out[positions,22:26]=entropy
            out[positions,26]=1;out[positions,27]=geometry_valid.float()
            known=evidence_known[positions]&geometry_valid.bool()
            out[positions,28:32]=(evidence_boxes[positions]-boxes_xyxy)*known[:,None]
            out[positions,32]=evidence_known[positions].float()
    return out.detach()

def coefficient_direction_loss(prediction,oracle):
    """Per-query unit-vector loss. Coef loss=2*cosine loss, not two signals.

    Zero-oracle queries have no direction target and remain in outcome scores.
    Nonzero output initialization is required; normalizing exact zero is not a
    finite-magnitude direction. Labels are never used to select an update.
    """
    p=prediction.flatten(1);o=oracle.detach().flatten(1)
    pn=p.norm(dim=1);on=o.norm(dim=1);valid=on>0
    pu=p/pn[:,None].clamp_min(1e-12);ou=o/on[:,None].clamp_min(1e-12)
    cosine=1-(pu*ou).sum(-1);coef=(pu-ou).square().sum(-1)
    return dict(cosine=cosine[valid].mean() if valid.any() else p.sum()*0,
                coefficient=coef[valid].mean() if valid.any() else p.sum()*0,valid=valid)

def direction_field(coefficients,basis,stock,radius):
    if coefficients.shape[:-1]!=stock.shape[:-1] or coefficients.shape[-1]!=basis.shape[1]:raise ValueError('Union support')
    field=F.linear(coefficients,basis.detach());n=field.flatten(1).norm(dim=1)
    fn=stock.detach().flatten(1).norm(dim=1)
    scale=torch.where(n>0,radius*fn/n.clamp_min(1e-12),torch.zeros_like(n))
    return field*scale.reshape(-1,*([1]*(field.ndim-1)))

class StateAwareDirectionMixer(nn.Module):
    """A: fixed magnitude, state-aware direction only. Source GT direction target."""
    def __init__(self,basis,hidden=128,radius=.13545580427763146):
        super().__init__();self.radius=radius;self.register_buffer('basis',basis.detach().float().clone())
        self.input=nn.Linear(3*hidden+8+STATE_DIM,hidden)
        self.local=nn.Conv3d(hidden,hidden,3,padding=1,groups=hidden)
        self.mix=nn.Linear(hidden,hidden);self.output=nn.Linear(hidden,basis.shape[1])
        # Fixed seeded small nonzero initialization avoids undefined normalized zero.
        nn.init.normal_(self.output.weight,std=1e-3);nn.init.zeros_(self.output.bias)
    def forward(self,z,qT,qS,evidence,state,stock):
        shape=z.shape[:-1]
        if state.shape!=(shape[0],shape[1],STATE_DIM) or evidence.shape!=(*shape,8):raise ValueError('State/THW support')
        qT=qT.detach()[:,None,None,None].expand(*shape,-1);qS=qS.detach()[:,None,None,None].expand(*shape,-1)
        st=state.detach()[:,:,None,None].expand(*shape,-1)
        h=F.silu(self.input(torch.cat((z.detach(),qT,qS,evidence.detach(),st),-1)))
        h=h+F.silu(self.local(h.movedim(-1,1)).movedim(1,-1))
        coefficients=self.output(F.silu(self.mix(h)))
        return direction_field(coefficients,self.basis,stock,self.radius),coefficients

class TrustGatedCorrection(nn.Module):
    """B: freeze the original direction generator, train only query-specific gate.

    Gate maps mean/max native state and evidence to alpha/r in [0,1].
    Initialization .01 has nonzero derivative; explicit clamp can represent0.
    No straight-through surrogate and no global radius search.
    """
    def __init__(self,direction_generator,radius=.13545580427763146):
        super().__init__();self.direction_generator=direction_generator.eval().requires_grad_(False)
        self.radius=radius;self.gate=nn.Sequential(nn.Linear(2*(STATE_DIM+8),16),nn.SiLU(),nn.Linear(16,1))
        nn.init.zeros_(self.gate[-1].weight);nn.init.constant_(self.gate[-1].bias,.01)
    def train(self,mode=True):
        super().train(mode);self.direction_generator.eval();return self
    def forward(self,args,state,evidence):
        with torch.no_grad():field=self.direction_generator(*args).detach()
        st=state.detach();ev=evidence.detach().flatten(1,-2)
        pooled=torch.cat((st.mean(1),st.amax(1),ev.mean(1),ev.amax(1)),-1)
        fraction=self.gate(pooled).clamp(0,1).flatten();stock=args[-1]
        n=field.flatten(1).norm(dim=1);fn=stock.detach().flatten(1).norm(dim=1)
        scale=torch.where(n>0,self.radius*fn*fraction/n.clamp_min(1e-12),torch.zeros_like(n))
        return field*scale.reshape(-1,*([1]*(field.ndim-1))),self.radius*fraction

def rescue_ladder(objective,stock,basis,*,mode,steps=20,radius=.13545580427763146):
    """C fixed-support normalized projected descent; no model launch here.

    objective(field,branch) must use the SAME frozen B1 native support for every
    step and both arms. Unit full-field branch gradients are summed, exactly as
    C1; the direction is projected into union or the full merger space. Step
    length is fixed at r*||stock||, then project the accumulated correction onto
    that same ball. Twenty iterations, zero initialization, final state only.
    Consequently union step1 is the existing analytic C1 (not a changed LR).
    C2/C3 differ only in feasible subspace; all model parameters stay frozen.
    """
    from vg_tta.desta3d_v3_decomposition import project,unit_to,norm
    if mode not in ('union','free'):raise ValueError(mode)
    parameter=torch.zeros((*stock.shape[:-1],basis.shape[1]) if mode=='union' else stock.shape,
                          device=stock.device,dtype=stock.dtype)
    history=[];cap=radius*norm(stock)
    for k in range(steps):
        delta=F.linear(parameter,basis.detach()) if mode=='union' else parameter
        field=(stock.detach()+delta).detach().requires_grad_();balanced=torch.zeros_like(stock);losses={}
        for branch in ('event','spatial'):
            loss=objective(field,branch)
            if loss is None:losses[branch]=None;continue
            g,=torch.autograd.grad(loss,field)
            if not torch.isfinite(g).all() or not torch.isfinite(loss):raise ValueError('Nonfinite rescue step')
            balanced+=unit_to(g.detach(),1.);losses[branch]=float(loss.detach())
        direction=project(balanced,basis) if mode=='union' else balanced
        update=unit_to(-direction,cap)
        with torch.no_grad():
            if mode=='union':
                parameter+=(update.double()@basis.detach().double()).float()
            else:parameter+=update
            # Measure the realized field, not nominal coefficient norm.
            realized=F.linear(parameter,basis.detach()) if mode=='union' else parameter
            n=norm(realized)
            if n>cap:parameter*=cap/n
        history.append(dict(step=k+1,loss_before=losses,norm=min(n,cap)))
    delta=F.linear(parameter,basis.detach()) if mode=='union' else parameter
    return delta.detach(),history
