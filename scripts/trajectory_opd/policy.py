"""Reward-tilted mixture projection and matched continuous-gradient controls.

Each action is an entire content state, executed by the frozen decoder.  The
OPD update is a finite conditional KL projection, not a decoder gradient or a
selection of the best sampled output.  Components are never averaged together.
"""
from __future__ import annotations
import copy, gc, hashlib, math, time
import torch
from scripts.trajectory_opd.replay import FrozenTrajectoryReplay

ARMS = ('TrajectoryOPD', 'Direct', 'DirectMulti', 'DirectMC')
SCALES = (1., .5, .25, .125, .0625)


def cpu(x):
    if torch.is_tensor(x): return x.detach().cpu().clone()
    if isinstance(x,dict): return {k:cpu(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [cpu(v) for v in x]
    return x


def seed_for(key,round_index):
    return int(hashlib.sha256(f'trajectory-opd-v1|41|{key}|{round_index}'.encode()).hexdigest()[:15],16)


def initial_states(key,sigma):
    g=torch.Generator().manual_seed(seed_for(key,-1))
    e=torch.randn(8,256,generator=g)*sigma
    return torch.stack((torch.zeros_like(e),e,-e))


def noise_for(key,r,counts):
    g=torch.Generator().manual_seed(seed_for(key,r));out=[]
    assert sum(counts)==32 and all(n%2==0 and n>=4 for n in counts)
    for n in counts:
        e=torch.randn(n//2,8,256,generator=g)
        out.append(torch.cat((e,-e)))
    return out


def allocations(probabilities):
    """Six antithetic exploration samples/component, remaining pairs by mass."""
    p=probabilities.detach().cpu().double();counts=[6,6,6]
    raw=7*p;whole=torch.floor(raw).long()
    for c in range(3): counts[c]+=2*int(whole[c])
    remainder=7-int(whole.sum())
    order=torch.argsort(raw-whole,descending=True,stable=True).tolist()
    for c in order[:remainder]:counts[c]+=2
    assert sum(counts)==32
    return counts


def conditional_projection(mean,actions,utilities,tau):
    """Empirical weighted cross-entropy optimum, approximating conditional KL.

    The atomic empirical measure is not assigned a finite KL to a continuous
    Gaussian; only the model-dependent cross-entropy is fitted from samples.
    """
    w=(utilities/tau).softmax(0)
    return mean+(w[:,None,None]*(actions-mean)).sum(0).to(mean),w


def calibrated_temperature(utilities,groups,target_fraction=.5):
    """Conditional ESS control, using current rewards only; no GT/scale sweep.

    Center within each component so between-mode differences cannot conceal a
    completely uninformative local cloud. Constant rewards give uniform weights.
    """
    centered=torch.cat([u-u.mean() for u in utilities]).double()
    assert len(centered)==sum(groups)==32
    spread=float(centered.abs().max())
    if spread<1e-12:return 1.,dict(flat=True,ESS=32.)
    def ess(t):
        w=(centered/t).softmax(0);return float(w.square().sum().reciprocal())
    lo=max(spread*1e-10,1e-15);hi=spread*1e4
    target=32*target_fraction
    for _ in range(64):
        mid=math.sqrt(lo*hi)
        if ess(mid)<target:lo=mid
        else:hi=mid
    return hi,dict(flat=False,ESS=ess(hi))


def brief(u):
    keys=('U','mask','link','guard','guard_mean','guard_max','obs','unobs','d_mean','d_max')
    z={k:float(u[k].detach()) for k in keys}
    assert all(math.isfinite(v) for v in z.values())
    return z


def fit(model,cache,evidence,key,arm,cfg):
    assert arm in ARMS
    device=next(model.ground_decoder.decoder.parameters()).device
    assert device.type=='cuda'
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();begin=time.perf_counter()
    replay=FrozenTrajectoryReplay(model,cache,evidence.W,full_reference=arm!='TrajectoryOPD')
    starts=initial_states(key,cfg['sigma']).to(device)
    if arm=='Direct': starts=starts[:1]
    means=[torch.nn.Parameter(x.clone()) for x in starts]
    opts=[torch.optim.Adam([m],lr=cfg['lr'],betas=(.9,.999),eps=1e-8) for m in means]
    current=[];current_boxes=[]
    for m in means:
        with torch.no_grad(): b=replay.boxes(m);u=evidence.utility(b,replay.E)
        current.append(brief(u));current_boxes.append(b.detach())
    initial=cpu(torch.stack(means));initial_values=copy.deepcopy(current)
    initial_boxes=cpu(torch.stack(current_boxes))
    probabilities=torch.ones(len(means),device=device,dtype=torch.float64)/len(means)
    rounds=[];path=[];decoder_backwards=accepted=rejected=0
    inactive=set()
    limit=cfg['rounds'] if arm in ('TrajectoryOPD','DirectMC') else cfg['direct_steps']
    initial_weight_hash=hashlib.sha256(b''.join(p.detach().cpu().contiguous().numpy().tobytes() for p in replay.decoder.parameters())).hexdigest()
    try:
        for r in range(limit):
            sampled=arm in ('TrajectoryOPD','DirectMC')
            needed=35 if sampled else 2
            if replay.evals+needed>cfg['forward_budget']:break
            before=torch.stack([m.detach() for m in means]).clone();opt_before=[copy.deepcopy(o.state_dict()) for o in opts]
            rr=dict(round=r,before=copy.deepcopy(current),probabilities=cpu(probabilities),accepted_components=[],backtracking=[])
            proposals=before.clone()
            for o in opts:o.zero_grad(set_to_none=True)
            if sampled:
                counts=allocations(probabilities);noise=noise_for(key,r,counts)
                clouds=[];values=[];gradients=[]
                for c,(n,e) in enumerate(zip(counts,noise)):
                    actions=before[c,None]+cfg['sigma']*e.to(before)
                    vals=[];grads=[]
                    for a in actions:
                        if arm=='TrajectoryOPD':
                            with torch.no_grad():
                                b=replay.boxes(a);u=evidence.utility(b,replay.E)
                        else:
                            point=means[c]+(a-before[c]).detach()
                            b=replay.boxes(point);u=evidence.utility(b,replay.E)
                            grads.append(torch.autograd.grad(u['U'],means[c])[0].detach())
                            decoder_backwards+=1
                        vals.append(u['U'].detach());del b,u
                    clouds.append(actions);values.append(torch.stack(vals));gradients.append(grads)
                tau,ess=calibrated_temperature(values,counts)
                logmass=[];conditional=[]
                grad=torch.zeros_like(before)
                for c,(actions,U) in enumerate(zip(clouds,values)):
                    proposal,w=conditional_projection(before[c],actions,U,tau);conditional.append(w)
                    if arm=='TrajectoryOPD':
                        # Exact weighted Gaussian conditional maximum likelihood.
                        # Centering preserves a bit-exact zero delta for flat rewards.
                        proposals[c]=proposal
                    else:
                        grad[c]=-(w[:,None,None]*torch.stack(gradients[c])).sum(0).to(before)/tau
                    logmass.append(torch.log(probabilities[c])+torch.logsumexp(U/tau,0)-math.log(len(U)))
                target_probabilities=torch.stack(logmass).softmax(0).clamp_min(1e-12)
                target_probabilities=target_probabilities/target_probabilities.sum()
                if arm=='DirectMC':
                    for c in range(3):
                        means[c].grad=grad[c];opts[c].step()
                    proposals=torch.stack([m.detach() for m in means])
                actual_ess=[float(w.square().sum().reciprocal()) for w in conditional]
                rr.update(counts=counts,tau=tau,centered_pool_ESS=ess,actual_conditional_ESS=actual_ess,candidate_states=cpu(clouds),candidate_U=cpu(values),conditional_weights=cpu(conditional),target_probabilities=cpu(target_probabilities))
                del clouds,values,gradients
                components=range(3)
            else:
                available=[c for c in range(len(means)) if c not in inactive]
                if not available:break
                c=available[r%len(available)];b=replay.boxes(means[c]);u=evidence.utility(b,replay.E)
                grad=torch.autograd.grad(-u['U'],means[c])[0];decoder_backwards+=1
                means[c].grad=grad;opts[c].step();proposals=before.clone();proposals[c]=means[c].detach()
                rr.update(gradient_norm=float(grad.norm()),gradient_evaluation=brief(u))
                if r==0:rr['initial_loss_gradient_state']=cpu(grad)
                del b,u,grad
                components=(c,);target_probabilities=probabilities
            with torch.no_grad():
                for c,m in enumerate(means):m.copy_(before[c])
            all_failed=True
            for c in components:
                delta=proposals[c]-before[c];found=False;attempts=[]
                for scale in SCALES:
                    if replay.evals>=cfg['forward_budget']:break
                    with torch.no_grad():
                        trial=before[c]+scale*delta;b=replay.boxes(trial);u=evidence.utility(b,replay.E)
                    z=brief(u);attempts.append(dict(scale=scale,**z))
                    if z['U']>current[c]['U']+1e-8:
                        with torch.no_grad():means[c].copy_(trial)
                        current[c]=z;current_boxes[c]=b.detach();accepted+=1;found=True;all_failed=False
                        rr['accepted_components'].append(c);break
                if not found:
                    rejected+=1
                    if arm!='TrajectoryOPD':opts[c].load_state_dict(opt_before[c])
                    if not sampled:inactive.add(c)
                rr['backtracking'].append(dict(component=c,attempts=attempts,accepted=found))
            probabilities=target_probabilities.detach()
            rr.update(after=copy.deepcopy(current),state_norms=torch.stack(means).detach().norm(dim=(1,2)).tolist(),full_trajectory_evals=replay.evals)
            rounds.append(rr)
            best=max(range(len(means)),key=lambda c:current[c]['U'])
            path.append(dict(step=r+1,best_component=best,states=cpu(torch.stack(means)),values=copy.deepcopy(current)))
            if not sampled and len(inactive)==len(means):break
        best=max(range(len(means)),key=lambda c:current[c]['U'])
        assert replay.evals<=cfg['forward_budget']
        replay.assert_frozen()
        assert all(not p.requires_grad and p.grad is None for p in replay.decoder.parameters())
        final_weight_hash=hashlib.sha256(b''.join(p.detach().cpu().contiguous().numpy().tobytes() for p in replay.decoder.parameters())).hexdigest()
        assert initial_weight_hash==final_weight_hash
        torch.cuda.synchronize()
        initial_best=max(range(len(initial_values)),key=lambda c:initial_values[c]['U'])
        return dict(arm=arm,final=cpu(current_boxes[best]),final_z=cpu(means[best]),final_eta=cpu(replay.expand(means[best])),initial_states=initial,initial_values=initial_values,initial_final=initial_boxes[initial_best],initial_best_component=initial_best,final_states=cpu(torch.stack(means)),best_component=best,path=path,rounds=rounds,initial_checks=replay.initial_checks,evidence=evidence.metadata,
                    costs=dict(seconds=time.perf_counter()-begin,full_trajectory_evals=replay.evals,native_offset_decoder_calls=2*replay.evals,setup_full_trajectory_evals=replay.setup_evals,setup_native_offset_decoder_calls=2*replay.setup_evals,decoder_backwards=decoder_backwards,state_only_backwards=0,updates=len(rounds),accepted=accepted,rejected=rejected,accepted_updates=accepted,allowance=cfg['forward_budget'],peak_GPU_bytes=torch.cuda.max_memory_allocated(),stop_reason='budget_or_fixed_limit_or_rejected'))
    finally:
        for o in opts:o.zero_grad(set_to_none=True)
        del replay,opts;gc.collect();torch.cuda.empty_cache()
