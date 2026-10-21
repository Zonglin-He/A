"""Isolated engineering replay of the frozen trajectory policy.

Each action is an entire content state, executed by the frozen decoder.  The
OPD update is a finite conditional KL projection, not a decoder gradient or a
selection of the best sampled output. Components are never averaged together.
Only OPD candidate reward evaluation is batched; decoder calls stay sequential.
"""
from __future__ import annotations
import copy, gc, hashlib, math, time
from contextlib import contextmanager
import torch
from scripts.trajectory_opd.replay import FrozenTrajectoryReplay

ARMS = ('TrajectoryOPD', 'Direct', 'DirectMulti', 'DirectMC')
SCALES = (1., .5, .25, .125, .0625)


class EventRecorder:
    """Deferred CUDA stream timing, with no synchronization per operation.

    The CPU timings measure host elapsed time, including any implicit waits in
    the unchanged algorithm. CUDA events measure stream intervals, which can
    include launch/host gaps; they are not isolated kernel execution times.
    ``arm_stream`` encloses every category and must never be added to them.
    """
    def __init__(self, device):
        self.enabled = device.type == 'cuda' and torch.cuda.is_available()
        self.records = []
        self.arm_events = None
        if self.enabled:
            self.arm_events = (torch.cuda.Event(enable_timing=True),
                               torch.cuda.Event(enable_timing=True))
            self.arm_events[0].record()

    @contextmanager
    def record(self, category, evaluations=1):
        events = None
        if self.enabled:
            events = (torch.cuda.Event(enable_timing=True),
                      torch.cuda.Event(enable_timing=True))
            events[0].record()
        begin = time.perf_counter()
        try:
            yield
        finally:
            if events is not None:
                events[1].record()
            self.records.append((category, events, time.perf_counter() - begin,
                                 evaluations))

    def end_arm(self):
        if self.arm_events is not None:
            self.arm_events[1].record()

    def summary_after_synchronize(self, arm_wall_seconds):
        categories = {}
        for category, events, wall, evaluations in self.records:
            row = categories.setdefault(category, dict(calls=0, evaluations=0,
                host_wall_seconds=0., CUDA_stream_seconds=0. if self.enabled else None))
            row['calls'] += 1
            row['evaluations'] += evaluations
            row['host_wall_seconds'] += wall
            if events is not None:
                row['CUDA_stream_seconds'] += events[0].elapsed_time(events[1]) / 1000.
        categorized_wall = sum(x['host_wall_seconds'] for x in categories.values())
        arm_stream = (self.arm_events[0].elapsed_time(self.arm_events[1]) / 1000.
                      if self.arm_events is not None else None)
        categorized_stream = (sum(x['CUDA_stream_seconds'] for x in categories.values())
                              if self.enabled else None)
        return dict(categories=categories, arm_wall_seconds=arm_wall_seconds,
            arm_CUDA_stream_seconds=arm_stream,
            other_host_wall_seconds=max(0., arm_wall_seconds-categorized_wall),
            other_CUDA_stream_seconds=(max(0., arm_stream-categorized_stream)
                                       if self.enabled else None),
            CUDA_events_enabled=self.enabled,
            scope='Disjoint setup, decoder forward, reward forward, backward, and policy-fit intervals; arm totals enclose these intervals and are not additional costs. Host and CUDA intervals overlap and must not be added. CUDA intervals include stream launch gaps; setup separately contains its numerical checks. Other is the residual, including event bookkeeping, trace transfer, and scalar synchronization.',
            backward_scope='Autograd of the complete reward through the frozen decoder to content state; includes reward and decoder backward')


class RecorderReplay:
    """Observe the unmodified replay, including its original setup checks."""
    def __init__(self, model, cache, W, recorder, full_reference):
        self.recorder = recorder
        with recorder.record('replay_setup'):
            self.replay = FrozenTrajectoryReplay(model, cache, W,
                                                  full_reference=full_reference)

    def __getattr__(self, name):
        return getattr(self.replay, name)

    def boxes(self, *args, **kwargs):
        with self.recorder.record('decoder_forward'):
            return self.replay.boxes(*args, **kwargs)


class RecorderEvidence:
    def __init__(self, evidence, recorder):
        self.evidence, self.recorder = evidence, recorder

    def __getattr__(self, name):
        return getattr(self.evidence, name)

    def utility(self, boxes, E):
        with self.recorder.record('reward_forward'):
            return self.evidence.utility(boxes, E)

    def utility_batch(self, boxes, E):
        with self.recorder.record('reward_forward', evaluations=len(boxes)):
            return self.evidence.utility_batch(boxes, E)


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
    recorder=EventRecorder(device)
    replay=RecorderReplay(model,cache,evidence.W,recorder,full_reference=arm!='TrajectoryOPD')
    evidence=RecorderEvidence(evidence,recorder)
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
                clouds=[];values=[];gradients=[];candidate_boxes=[]
                for c,(n,e) in enumerate(zip(counts,noise)):
                    actions=before[c,None]+cfg['sigma']*e.to(before)
                    vals=[];grads=[]
                    for a in actions:
                        if arm=='TrajectoryOPD':
                            with torch.no_grad():
                                b=replay.boxes(a)
                            candidate_boxes.append(b.detach())
                            del b
                        else:
                            point=means[c]+(a-before[c]).detach()
                            b=replay.boxes(point);u=evidence.utility(b,replay.E)
                            with recorder.record('decoder_reward_backward'):
                                grads.append(torch.autograd.grad(u['U'],means[c])[0].detach())
                            decoder_backwards+=1
                            vals.append(u['U'].detach());del b,u
                    clouds.append(actions)
                    if arm!='TrajectoryOPD':values.append(torch.stack(vals))
                    gradients.append(grads)
                if arm=='TrajectoryOPD':
                    # Exactly 32 real sequential decoder evaluations above.
                    # Only the deterministic reward computation is vectorized.
                    with torch.no_grad():
                        batch_utility=evidence.utility_batch(torch.stack(candidate_boxes),replay.E)
                    values=list(batch_utility['U'].detach().split(counts))
                    del candidate_boxes,batch_utility
                else:
                    del candidate_boxes
                with recorder.record('policy_fit'):
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
                with recorder.record('decoder_reward_backward'):
                    grad=torch.autograd.grad(-u['U'],means[c])[0]
                decoder_backwards+=1
                with recorder.record('policy_fit'):
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
        initial_best=max(range(len(initial_values)),key=lambda c:initial_values[c]['U'])
        result=dict(arm=arm,final=cpu(current_boxes[best]),final_z=cpu(means[best]),final_eta=cpu(replay.expand(means[best])),initial_states=initial,initial_values=initial_values,initial_final=initial_boxes[initial_best],initial_best_component=initial_best,final_states=cpu(torch.stack(means)),best_component=best,path=path,rounds=rounds,initial_checks=replay.initial_checks,evidence=evidence.metadata,
                    costs=dict(full_trajectory_evals=replay.evals,native_offset_decoder_calls=2*replay.evals,setup_full_trajectory_evals=replay.setup_evals,setup_native_offset_decoder_calls=2*replay.setup_evals,decoder_backwards=decoder_backwards,state_only_backwards=0,updates=len(rounds),accepted=accepted,rejected=rejected,accepted_updates=accepted,allowance=cfg['forward_budget'],peak_GPU_bytes=torch.cuda.max_memory_allocated(),stop_reason='budget_or_fixed_limit_or_rejected'))
        recorder.end_arm()
        torch.cuda.synchronize()
        result['costs']['seconds']=time.perf_counter()-begin
        result['profiling']=recorder.summary_after_synchronize(result['costs']['seconds'])
        return result
    finally:
        for o in opts:o.zero_grad(set_to_none=True)
        del replay,opts;gc.collect();torch.cuda.empty_cache()
