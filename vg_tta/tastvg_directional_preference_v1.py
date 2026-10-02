"""Rank-direction actuation with state-local counterfactual RKL magnitude.

Arm A retains the sealed v3 mathematics. Candidate compatibility is a geometry
surrogate, not a native tube policy. Targets are frozen at each inner step.
"""
import numpy as np
import torch
from methods.decota_final_simplified_v1.tensors import detached,state_hash
from methods.decota_final_simplified_v1.objectives import prediction
from vg_tta.tastvg_corruption_c0c1_v1 import candidates
from vg_tta.tastvg_temporal_qualification_v1 import critic_scores
from vg_tta.tastvg_spatial_rank_s11_v1 import SpatialActor,average_ranks,update_scale
from vg_tta.tastvg_native_spatial_rollout_s05_v1 import predict
from vg_tta.tastvg_spatial_critic_s06_v1 import rewards


def combine_layers(boxes,logits,records,ids):
    assert len(boxes)==len(logits)==2 and all(b.shape[0]==z.shape[0]==6 for b,z in zip(boxes,logits))
    return [prediction([z[layer] for z in logits],torch.stack([boxes[i%2][layer,i//2] for i in range(len(ids))]),records,ids) for layer in range(6)]


@torch.no_grad()
def central_with_candidates(actor,data):
    """Observe native second-pass layer outputs; hooks do not change computation."""
    boxes=[];logits=[];calls=[0]
    def decoder_hook(mod,args,out):
        calls[0]+=1
        if calls[0]%2==0:boxes.append(out[0].flatten(1,2).detach().cpu())
    def temporal_hook(mod,args,out):logits.append(out.detach().cpu())
    hs=[actor.model.ground_decoder.register_forward_hook(decoder_hook),actor.model.temp_embed.register_forward_hook(temporal_hook)]
    try:ev,b,p=actor.values(data)
    finally:
        for h in hs:h.remove()
    assert calls[0]==4
    layers=combine_layers(boxes,logits,data['records'],data['frame_ids'])
    assert torch.equal(layers[-1]['boxes'],p['boxes']) and layers[-1]['indices']==p['indices']
    assert all(torch.equal(a,z) for a,z in zip(layers[-1]['logits'],p['logits']))
    return ev,b,p,layers,candidates(layers,data['records'],data['frame_ids'])['temporal']


def fast_rerank(pre,support,expert):
    assert support[0]['indices']==pre['indices']
    score=critic_scores([x['physical_interval'] for x in support],expert['proposals'],expert['proposal_confidence'])
    selected=int(np.argmax(score));c=support[selected]
    out={**pre,'indices':list(c['indices']),'physical_interval':list(c['physical_interval'])}
    return out,dict(scores=score.tolist(),selected=selected,candidates=support,teacher_interval_as_output=False)


def reverse_kl(central,candidates,rewards,coeff,temperature=1.,student_temperature=1.):
    from vg_tta.tastvg_spatial_online_opd_s1_v1 import geometry
    if not torch.isfinite(central).all() or not torch.isfinite(candidates).all():raise FloatingPointError('nonfinite spatial boxes')
    distances=geometry(central,candidates,*coeff);logp=(-distances/student_temperature).log_softmax(0)
    ranks=average_ranks(rewards.detach().cpu().numpy())
    logq=(-torch.as_tensor(ranks,device=central.device,dtype=central.dtype)/temperature).log_softmax(0)
    loss=(logp.exp()*(logp-logq)).sum()
    if not torch.isfinite(loss):raise FloatingPointError('nonfinite rank KL')
    return loss,logp.exp(),logq.exp(),distances


def norm(state):
    return float(torch.sqrt(sum(v.double().square().sum() for v in state.values())))


def target_loss(central, targets, scores, coeff, teacher_temperature,
                student_temperature, mode, s_ref, fixed_lambda=None):
    from vg_tta.tastvg_spatial_online_opd_s1_v1 import geometry
    distances=geometry(central,targets,*coeff)
    logp=(-distances/student_temperature).log_softmax(0)
    ranks=average_ranks(scores.detach().cpu().numpy())
    logrank=(-torch.as_tensor(ranks,device=central.device,dtype=central.dtype)
             /teacher_temperature).log_softmax(0)
    spread=float((scores.max()-scores.min()).detach())
    strength=1. if mode=='rank' else (float(fixed_lambda) if fixed_lambda is not None
                                     else spread/(spread+s_ref))
    # Exact no-op is part of the requested objective semantics.
    flat_noop=mode!='rank' and spread==0. and fixed_lambda is None
    if strength==1.: logtarget=logrank.detach()
    elif strength==0.: logtarget=logp.detach()
    else:
        logtarget=torch.logaddexp(logp.detach()+np.log1p(-strength),
                                 logrank.detach()+np.log(strength))
    loss=central.sum()*0. if strength==0. else (logp.exp()*(logp-logtarget)).sum()
    if not torch.isfinite(loss):raise FloatingPointError('nonfinite proximal KL')
    return loss,logp.exp(),logtarget.exp(),distances,logtarget,dict(
        reward_spread=spread,strength=strength,s_ref=s_ref,mode=mode,
        fixed_lambda=fixed_lambda,flat_noop=flat_noop,p_ref=logp.detach().exp().cpu(),
        q_rank=logrank.detach().exp().cpu(),frozen_log_target=logtarget.cpu())


def project_arrival(actor, center, radius):
    delta={n:p.detach()-center[n] for n,p in actor.named}
    proposed_norm=norm(delta)
    factor=min(1.,radius/(proposed_norm+1e-30)) if radius is not None else 1.
    with torch.no_grad():
        if factor<1.:
            for n,p in actor.named:p.copy_(center[n]+factor*delta[n])
    actual=norm({n:p.detach()-center[n] for n,p in actor.named})
    if radius is not None:assert actual<=radius+2e-6
    return dict(proposed_arrival_norm=proposed_norm,projection_factor=factor,
                actual_arrival_norm=actual,arrival_radius=radius,
                arrival_center_sha256=state_hash(center))


def direction_from_ranks(basis,ranks,actuation):
    """Candidate order: center, +u0,-u0, +u1,-u1, ...; rank zero is best."""
    c=np.asarray(ranks,dtype=np.float64)[2::2]-np.asarray(ranks,dtype=np.float64)[1::2]
    assert basis.shape[0]==len(c)==4
    top=int(np.argmax(np.abs(c)))
    coeff=c if actuation!='top_directional' else np.eye(4,dtype=np.float64)[top]*np.sign(c[top])
    vector=torch.as_tensor(coeff,device=basis.device,dtype=basis.dtype)@basis
    return vector,c,top


@torch.no_grad()
def apply_direction(named,gradients,basis,ranks,scale,actuation):
    flat=torch.cat([g.detach().double().flatten() for g in gradients])
    ub=basis.to(device=flat.device,dtype=torch.float64)
    vec,c,top=direction_from_ranks(ub,ranks,actuation)
    gn=float(torch.sqrt(sum(g.double().square().sum() for g in gradients)));vn=float(vec.norm());desired=float(scale*gn)
    usable=vn>0.;signed=-flat
    cosine=float(torch.dot(signed,vec)/(gn*vn)) if gn>0 and usable else None
    before=[p.detach().clone() for _,p in named]
    delta=(desired/(vn+1e-12))*vec if usable else torch.zeros_like(vec)
    if actuation=='rkl':
        for (_,p),g in zip(named,gradients):p.add_(g,alpha=-scale)
        actuated=gn>0 and scale>0;reason=None if actuated else 'zero_rkl_magnitude'
    else:
        pos=0
        for _,p in named:
            q=delta[pos:pos+p.numel()].reshape(p.shape).to(p);p.add_(q);pos+=p.numel()
        assert pos==len(delta)
        actuated=usable and desired>0
        reason=None if actuated else ('no_rank_direction' if not usable else 'zero_rkl_magnitude')
    actual=float(torch.sqrt(sum((p.detach()-v).double().square().sum() for (_,p),v in zip(named,before))))
    proj=ub@signed
    return dict(actuation=actuation,rank_contrasts=c.tolist(),top_pair=top,
                preferred_direction_norm=vn,counterfactual_rkl_step_norm=desired,
                actual_step_norm=actual,rkl_preference_cosine=cosine,
                rkl_probe_axis_components=proj.detach().cpu().tolist(),
                rkl_probe_subspace_norm=float(proj.norm()),actuated=bool(actuated),
                no_op_reason=reason,magnitude_reference='RKL evaluated at this arm current state',
                magnitude_match_applicable=bool(actuation=='rkl' or usable))


class SingleStep:
    """One persistent actor per independent stream; evidence providers are lazy."""
    def __init__(self,model,deltas,*,lr=.005,teacher_temperature=1.,student_temperature=1.,fast=True,slow=True,target_mode="rank",s_ref=1.,arrival_radius=None,fixed_lambda=None,actuation="rkl",basis=None):
        assert lr>0 and teacher_temperature>0 and student_temperature>0
        self.student_temperature=float(student_temperature);self.lr=float(lr);self.teacher_temperature=float(teacher_temperature)
        self.actor=SpatialActor(model);self.model=model;self.deltas=deltas;self.fast=fast;self.slow=slow
        self.coeff=[model.cfg.SOLVER.BBOX_COEF,model.cfg.SOLVER.GIOU_COEF]
        assert target_mode in ["rank","proximal"] and s_ref>0
        self.target_mode=target_mode;self.s_ref=float(s_ref);self.arrival_radius=arrival_radius;self.fixed_lambda=fixed_lambda
        assert actuation in ["rkl","rank_directional","top_directional"] and basis is not None
        assert target_mode=="rank" and arrival_radius is None and fixed_lambda is None
        self.actuation=actuation;self.basis=basis.detach().clone()
    def reset(self):self.actor.restore(self.actor.initial)
    def close(self):self.actor.close()
    def arrive(self,data,scheduled,temporal_provider=None,spatial_provider=None):
        actor=self.actor;before=actor.state()
        with torch.no_grad():
            if scheduled and self.fast:
                ev,boxes,pre,layers,tc=central_with_candidates(actor,data)
            else:ev,boxes,pre=actor.values(data);layers=tc=None
        out=pre;result=dict(prediction=detached(pre,'cpu'),pre_state=detached(before,'cpu'),pre_state_sha256=state_hash(before),updated=False,temporal_expert_read=False,spatial_expert_read=False,GT_read=False)
        if scheduled and self.fast:
            expert=temporal_provider();assert expert['pixel_sha256']==data['pixel_sha256']
            out,td=fast_rerank(pre,tc,expert);result.update(temporal=td,temporal_layers=layers,temporal_expert_read=True)
        # Seal current output in memory BEFORE any spatial write.
        result['output_prediction']=detached(out,'cpu');post_ev=ev;post=pre
        if scheduled and self.slow:
            cs=[]
            for k,delta in enumerate(self.deltas):
                _,_,cp=predict(self.model,data,{n:before[n]+delta[n] for n in before})
                if k==0:assert torch.equal(cp['boxes'],pre['boxes'])
                cs.append(dict(prediction=detached(cp,'cpu')))
            expert=spatial_provider();assert expert['pixel_sha256']==data['pixel_sha256']
            score=rewards([c['prediction']['boxes'].numpy() for c in cs],expert['boxes'],expert['valid'])
            result.update(candidates=cs,rewards=score.tolist() if score is not None else None,valid_expert_frames=int(expert['valid'].sum()),spatial_expert_read=True)
            if score is not None:
                target=torch.stack([c['prediction']['boxes'] for c in cs]).cuda().detach();rt=torch.tensor(score,device='cuda',dtype=torch.float64).detach()
                evg,bg,pg=actor.values(data);assert torch.equal(bg.detach().cpu(),pre['boxes'])
                loss,pi,qi,dist,logtarget,meta=target_loss(bg,target,rt,self.coeff,self.teacher_temperature,self.student_temperature,self.target_mode,self.s_ref,self.fixed_lambda);grad=torch.autograd.grad(loss,[v for _,v in actor.named]);
                if not all(torch.isfinite(g).all() for g in grad):raise FloatingPointError('nonfinite spatial gradient')
                scale,gn=update_scale(grad,'rank',self.lr,0.)
                result['update']=dict(rank=average_ranks(score).tolist(),update_scale=scale,global_gradient_norm=gn,loss_before=float(loss.detach()),p=pi.detach().cpu(),q=qi.detach().cpu(),distances=dist.detach().cpu(),gradients={n:g.detach().cpu() for (n,_),g in zip(actor.named,grad)},gradient_norm=float(torch.sqrt(sum(g.square().sum() for g in grad))),query_gradient_norm=float(grad[0].norm()),LN_gradient_norm=float(torch.sqrt(sum(g.square().sum() for g in grad[1:]))),coefficients=self.coeff,lr=self.lr,teacher_temperature=self.teacher_temperature,student_temperature=self.student_temperature,candidate_targets_detached=not target.requires_grad,reward_detached=not rt.requires_grad)
                result["update"].update(meta)
                with torch.no_grad():
                    result["update"].update(apply_direction(actor.named,grad,self.basis,average_ranks(score),scale,self.actuation))
                    result["update"].update(project_arrival(actor,self.arrival_center,self.arrival_radius))
                    post_ev,bpost,post=actor.values(data);from vg_tta.tastvg_spatial_online_opd_s1_v1 import geometry
                    lp_after=(-geometry(bpost,target,*self.coeff)/self.student_temperature).log_softmax(0)
                    after_loss=(lp_after.exp()*(lp_after-logtarget)).sum()
                result['update']['loss_after']=float(after_loss);result['updated']=result['update']['actuated']
        after=actor.state();result.update(post_prediction=detached(post,'cpu'),post_state=detached(after,'cpu'),post_state_sha256=state_hash(after),parameter_displacement=float(torch.sqrt(sum((after[n]-before[n]).square().sum() for n in before))),displacement_from_source=float(torch.sqrt(sum((after[n]-actor.initial[n]).square().sum() for n in before))))
        assert torch.equal(result['output_prediction']['boxes'],result['prediction']['boxes'])
        return result,post_ev


def nested_directions(size=1792, count=4):
    """Preserve the old four directions exactly, then extend a fixed basis."""
    from vg_tta.tastvg_native_spatial_rollout_s05_v1 import directions
    assert 1 <= count <= 16 and size >= 16
    first = directions(size, 4)
    if count <= 4:
        return first[:count].clone()
    vectors = list(first.unbind())
    generator = torch.Generator(device='cpu').manual_seed(20261001)
    for _ in range(12):
        v = torch.randn(size, generator=generator, dtype=torch.float64)
        for repeat in range(2):
            for q in vectors:
                v = v - torch.dot(q, v)*q
        v = v/v.norm()
        vectors.append(v)
    return torch.stack(vectors[:count])


def rollout_states(center, rho=.05, direction_count=4):
    flat = torch.cat([x.detach().cpu().double().flatten() for x in center.values()])
    basis = nested_directions(len(flat), direction_count)
    radius = rho*flat.norm()
    states = []
    for z in [flat]+[flat+sign*radius*u for u in basis for sign in [1, -1]]:
        state, offset = {}, 0
        for key, value in center.items():
            state[key] = z[offset:offset+value.numel()].reshape(value.shape).to(value)
            offset += value.numel()
        states.append(state)
    return states, dict(rho=rho, radius=float(radius), central_norm=float(flat.norm()),
        dimensions=len(flat), direction_count=direction_count, candidates=2*direction_count+1,
        seed=20260929, extension_seed=20261001), basis


class OnlineMethod(SingleStep):
    """Seal one current output, then refresh on-policy candidates at each inner step."""
    def __init__(self, model, deltas, *, steps=1, **kwargs):
        super().__init__(model, deltas, **kwargs)
        assert isinstance(steps, int) and 1 <= steps <= 10
        self.steps = steps

    def arrive(self, data, scheduled, temporal_provider=None, spatial_provider=None):
        self.arrival_center=self.actor.state()
        cache = {}
        def spatial_once():
            if 'spatial' not in cache:
                cache['spatial'] = spatial_provider()
            return cache['spatial']
        result, ev = super().arrive(data, scheduled, temporal_provider, spatial_once)
        traces = []
        def record(x):
            return {k:x.get(k) for k in ['pre_state', 'post_state', 'pre_state_sha256',
                'post_state_sha256', 'updated', 'update', 'prediction', 'post_prediction',
                'candidates', 'rewards', 'valid_expert_frames']}
        if scheduled:
            traces.append(record(result))
            for step in range(1, self.steps):
                if traces[-1]['update'] is None:
                    break
                saved_fast = self.fast
                self.fast = False
                try:
                    nxt, ev = super().arrive(data, True, None, spatial_once)
                finally:
                    self.fast = saved_fast
                assert nxt['pre_state_sha256'] == traces[-1]['post_state_sha256']
                traces.append(record(nxt))
                result['post_prediction'] = nxt['post_prediction']
                result['post_state'] = nxt['post_state']
                result['post_state_sha256'] = nxt['post_state_sha256']
                result['displacement_from_source'] = nxt['displacement_from_source']
        result['update_steps'] = traces
        result['updated'] = any(x['updated'] for x in traces)
        result['parameter_displacement'] = float(torch.sqrt(sum(
            (result['post_state'][n]-v).square().sum() for n,v in result['pre_state'].items())))
        # Each actor.values/predict is one exact downstream replay, comprising two offsets.
        result['compute'] = dict(inner_steps=len(traces),
            spatial_candidate_replays=len(traces)*len(self.deltas),
            native_replays=1+max(0,len(traces)-1)+len(traces)*len(self.deltas)
                +2*sum(x['update'] is not None for x in traces),
            backward_calls=sum(x['update'] is not None for x in traces),
            spatial_provider_calls=len(cache), temporal_provider_calls=int(scheduled and self.fast))
        return result, ev
