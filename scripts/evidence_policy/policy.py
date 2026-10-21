"""Eight-feature episodic evidence policy: immediate A and exact-lookahead B.

The immutable full-box bank, physical timeline, Gumbel stream, prior, memory
write, optimizer and hard fixed-score acceptance are shared across all arms.
Hard-history OPD caches each outer acquisition for ten inner supervised updates;
Direct recomputes its full soft history and differentiable teacher every inner.
"""
from __future__ import annotations
import copy
import time
import torch
import torch.nn.functional as F

from scripts.propagation_v0.policy import (
    ARMS, _require, _constant, _prepare, _prior, _log_probs, _kl, _detach,
    _optimizer_hash, _score, _fixed_loss, _batch_loss, gumbel_noise, evidence_dp)

FEATURE_NAMES = ('memory_cosine','previous_selected_feature_cosine','previous_candidate_link',
                 'E_feature_cosine','negative_guard','is_E','delta_unary','confidence_delta_unary')
MODES = ('student_sample','teacher_sample','student_greedy','teacher_greedy','direct_soft')


def _settings(cfg):
    values = dict(seed=int(cfg.get('seed',41)),rounds=int(cfg.get('rounds',cfg.get('outer_steps',3))),
        inner_steps=int(cfg.get('inner_steps',10)),paths=int(cfg.get('paths',2)),
        lr=float(cfg.get('lr',.1)),teacher_tau=float(cfg.get('teacher_tau',.2)),
        prior_weight=float(cfg.get('prior_weight',.05)),memory_rate=float(cfg.get('memory_write_max',.2)),
        grad_clip=float(cfg.get('gradient_clip_norm',1.)),prior_E=float(cfg.get('original_prior',.7)))
    _require(values==dict(seed=41,rounds=3,inner_steps=10,paths=2,lr=.1,teacher_tau=.2,
        prior_weight=.05,memory_rate=.2,grad_clip=1.,prior_E=.7),'Evidence-policy configuration differs from the locked A/B protocol')
    initial=cfg.get('initial_weights',[0.]*8)
    _require(tuple(initial)==(0.,)*8,'Eight temporary weights must reset to zero')
    _require(cfg.get('feedback_weights',dict(support=1.,link=.25,memory=.1,guard=.1))==
             dict(support=1.,link=.25,memory=.1,guard=.1),'Feedback weights differ from the locked reference')
    _require(float(cfg.get('Direct_relaxation_temperature',1.))==1.,
             'Direct uses the locked unit Gumbel relaxation temperature')
    return values


def _weights(phi):
    phi=torch.as_tensor(phi)
    _require(phi.shape==(8,) and phi.dtype==torch.float64 and phi.device.type=='cpu'
        and bool(torch.isfinite(phi).all()),'Weights must be finite FP64 CPU [8]')
    return phi


def exact_lookahead(*args,**kwargs):
    # Lazy import lets pure policy tests substitute a synthetic exact feedback
    # function without importing or rebuilding a real candidate bank.
    from scripts.evidence_policy.lookahead import exact_lookahead as implementation
    return implementation(*args,**kwargs)


def _query(prepared,nodes,memory,previous,variant,meter,prev_features=None):
    result=exact_lookahead(prepared,nodes,memory,previous,horizon=1 if variant=='A' else 4,
                           prev_features=prev_features)
    _require(isinstance(result,dict) and result['q'].shape==(len(nodes),3),
             'Lookahead must return q[N,3] and its actual costs')
    meter['q_calls']=meter.get('q_calls',0)+1
    meter['q_input_states']=meter.get('q_input_states',0)+len(nodes)
    for key,value in result.get('costs',{}).items():
        tensor=torch.as_tensor(value).detach()
        if tensor.numel()==1:
            meter['lookahead_'+key]=meter.get('lookahead_'+key,0)+int(tensor)
        elif key.endswith('_by_depth'):
            counts=tensor.tolist();previous=meter.setdefault('lookahead_'+key,[0]*len(counts))
            _require(len(previous)==len(counts),'Lookahead counter horizon changed within one fit')
            meter['lookahead_'+key]=[a+int(b) for a,b in zip(previous,counts)]
        else:
            # Per-input/per-action continuation tables have different N in
            # teacher sequential calls and student batches; aggregate their
            # actual counts, never attempt to add mismatched arrays.
            name='lookahead_'+key+'_total'
            meter[name]=meter.get(name,0)+int(tensor.sum())
    return result


def _finish_feedback(paths,prepared,variant,settings,meter,detach_teacher):
    T=prepared['T'];N=len(paths)*T
    nodes=torch.arange(T).repeat(len(paths))
    memories=torch.cat([p['memory_before'] for p in paths])
    previous=torch.cat([p['previous_action_probs'] for p in paths])
    prev_features=torch.cat([p['previous_features'] for p in paths])
    queried=_query(prepared,nodes,memories,previous,variant,meter,prev_features)
    _require(queried['q'].shape==(N,3),'Batched teacher output changed state order')
    for index,path in enumerate(paths):
        Q=queried['q'][index*T:(index+1)*T]
        if detach_teacher:Q=Q.detach()
        _attach(path,Q,prepared,variant,settings,detach_teacher)
    return paths


def _attach(path,Q,prepared,variant,settings,detach_teacher):
    logq=_log_probs(Q/settings['teacher_tau'],path['effective_valid'])
    if detach_teacher:logq=logq.detach()
    instant=(prepared['u']+.25*path['features'][...,2]+
             .1*path['features'][...,0]+.1*path['features'][...,4])
    instant=instant.masked_fill(~path['effective_valid'],0.)
    instant=torch.where(prepared['active'][:,None],instant,torch.zeros_like(instant))
    KL=_kl(logq,path['student_log_probs'],path['effective_valid'])
    prior_KL=_kl(path['student_log_probs'],path['prior_log_probs'],path['effective_valid'])
    c=path['confidence'];feedback=(c*KL).sum()/(c.sum()+1e-12);prior=prior_KL.mean()
    path.update(Q=Q,teacher_scores=Q,teacher_log_probs=logq,teacherq=logq.exp(),
        teacher_probs=logq.exp(),r=instant,immediate_scores=instant,KL_terms=KL,
        prior_KL_terms=prior_KL,feedback_loss=feedback,prior_loss=prior,
        loss=feedback+settings['prior_weight']*prior,
        teacher_target_detached=detach_teacher,variant=variant)


def rollout(bank,phi,cfg,mode='student_greedy',generator=None,noise=None,
            variant='A',collect_only=False,prepared=None,meter=None):
    settings=_settings(cfg);_require(variant in ('A','B') and mode in MODES,'Unknown variant/mode')
    b=_prepare(bank) if prepared is None else prepared;phi=_weights(phi);T=b['T']
    meter={} if meter is None else meter
    stochastic=mode in ('student_sample','teacher_sample','direct_soft')
    if stochastic:
        if noise is None:
            from scripts.propagation_v0.policy import _gumbels
            noise=_gumbels((T,3),generator)
        noise=_constant(noise,'noise')
        _require(noise.shape==(T,3) and bool(torch.isfinite(noise).all()),'Noise must be finite [T,3]')
    memory=b['initial_memory'];previous=torch.tensor([1.,0.,0.],dtype=torch.float64)
    previous_feature=b['features'][0,0];is_E=previous.clone()
    fields=('features','student_log_probs','studentq','prior_log_probs','priorq','effective_valid',
        'memory_before','memory_after','previous_action_probs','previous_features','actions','action_probs','omega','selected_u','boxes')
    rows={k:[] for k in fields};teacher_Q=[]
    teacher_mode=mode.startswith('teacher_')
    for t in range(T):
        valid=b['valid'][t] if bool(b['active'][t]) else torch.tensor([True,False,False])
        prior=_prior(valid,settings['prior_E']);logprior=prior.log()
        memcos=(memory[None]*b['features'][t]).sum(-1)
        prevcos=(F.normalize(previous_feature,dim=-1,eps=1e-12)[None]*b['features'][t]).sum(-1)
        link=previous@b['link'][t-1] if t else torch.zeros(3,dtype=torch.float64)
        ecos=(b['features'][t,0,None]*b['features'][t]).sum(-1)
        delta=b['u'][t]-b['u'][t,0];c=b['confidence'][t]*b['active'][t]
        features=torch.stack((memcos,prevcos,link,ecos,-b['guard'][t],is_E,delta,c*delta),dim=-1)
        logits=logprior+features@phi;logp=_log_probs(logits,valid)
        if teacher_mode:
            query=_query(b,torch.tensor([t]),memory[None],previous[None],variant,meter,previous_feature[None])
            Q=query['q'][0].detach();teacher_Q.append(Q)
            selection=Q/settings['teacher_tau']
        else:selection=logits
        if not bool(b['active'][t]):
            action_probs,action=is_E,0
        elif mode=='direct_soft':
            action_probs=(logits+noise[t]).masked_fill(~valid,-torch.inf).softmax(-1)
            action=int(action_probs.detach().argmax())
        else:
            if mode.endswith('_sample'):selection=selection+noise[t]
            action=int(selection.masked_fill(~valid,-torch.inf).detach().argmax())
            action_probs=F.one_hot(torch.tensor(action),3).double()
        selected=action_probs@b['features'][t];u=action_probs@b['u'][t]
        omega=settings['memory_rate']*c*u
        before=memory
        if bool(b['active'][t]) and bool(omega.detach()!=0):
            memory=F.normalize((1-omega)*memory+omega*selected,dim=-1,eps=1e-12)
        values=dict(features=features,student_log_probs=logp,studentq=logp.exp(),prior_log_probs=logprior,
            priorq=prior,effective_valid=valid,memory_before=before,memory_after=memory,
            previous_action_probs=previous,previous_features=previous_feature,actions=torch.tensor(action),
            action_probs=action_probs,omega=omega,selected_u=u,boxes=action_probs@b['boxes'][t])
        for key in fields:rows[key].append(values[key])
        previous,previous_feature=action_probs,selected
    path={key:torch.stack(values) for key,values in rows.items()}
    path.update(active=b['active'],observed=b['observed'],confidence=b['confidence']*b['active'],
        mode=mode,variant=variant,noise=noise,feature_names=FEATURE_NAMES,Q=None,teacherq=None,
        teacher_scores=None,teacher_log_probs=None)
    if teacher_mode:_attach(path,torch.stack(teacher_Q),b,variant,settings,True)
    elif not collect_only:_finish_feedback([path],b,variant,settings,meter,mode!='direct_soft')
    return path


def _margin(paths,phi):
    margins=[];alternative_gaps=[];learned_gaps=[];prior_barriers=[]
    for path in paths:
        logits=(path['prior_log_probs']+path['features']@phi).masked_fill(~path['effective_valid'],-torch.inf)
        top=logits.topk(2,dim=-1).values;valid=torch.isfinite(top[:,1])
        if bool(valid.any()):
            margins.append((top[valid,0]-top[valid,1]).detach())
            alt_logits,alt_indices=logits[:,1:].max(-1);alt_indices=alt_indices+1
            nodes=torch.nonzero(valid,as_tuple=True)[0];alts=alt_indices[valid]
            alternative_gaps.append((alt_logits[valid]-logits[valid,0]).detach())
            learned_gaps.append(((path['features'][nodes,alts]-path['features'][nodes,0])@phi).detach())
            prior_barriers.append((path['prior_log_probs'][nodes,0]-path['prior_log_probs'][nodes,alts]).detach())
    if not margins:return dict(valid_nodes=0,mean=None,minimum=None,maximum=None)
    values=torch.cat(margins)
    gaps=torch.cat(alternative_gaps);learned=torch.cat(learned_gaps);barriers=torch.cat(prior_barriers)
    return dict(valid_nodes=len(values),mean=float(values.mean()),minimum=float(values.min()),maximum=float(values.max()),
        best_alternative_minus_E=gaps,learned_alternative_minus_E=learned,prior_E_barrier=barriers,
        positive_alternative_gap_nodes=int((gaps>0).sum()),alternative_gap_mean=float(gaps.mean()),
        alternative_gap_minimum=float(gaps.min()),alternative_gap_maximum=float(gaps.max()),
        learned_gap_mean=float(learned.mean()),prior_barrier_mean=float(barriers.mean()))


def fit(bank,key,arm,cfg,variant='A'):
    _require(arm in ARMS and variant in ('A','B'),'Unknown arm/variant')
    settings=_settings(cfg);b=_prepare(bank);begin=time.perf_counter();meter={}
    phi=torch.nn.Parameter(torch.zeros(8,dtype=torch.float64))
    optimizer=torch.optim.Adam([phi],lr=settings['lr'],betas=(.9,.999),eps=1e-8)
    with torch.no_grad():current=_detach(rollout(bank,phi,cfg,variant=variant,collect_only=True,prepared=b))
    _require(torch.equal(current['boxes'],b['E']),'Zero initialization must preserve E exactly')
    initial=_detach(current);current_score=initial_score=_score(bank,current)
    fixed=[];setup_begin=time.perf_counter()
    if arm=='FixedKD':
        with torch.no_grad():
            fixed=[_detach(rollout(bank,phi,cfg,'teacher_sample',noise=gumbel_noise(key,0,p,b['T']),
                variant=variant,prepared=b,meter=meter)) for p in range(settings['paths'])]
    setup_seconds=time.perf_counter()-setup_begin
    rounds=[];training=[];acquisitions=[];validations=[];all_actual=[initial,*fixed]
    student_acquisitions=direct_recomputations=accepted=0;stop=False
    for outer in range(settings['rounds']):
        noises=[gumbel_noise(key,outer,p,b['T']) for p in range(settings['paths'])]
        cached=fixed
        if arm=='OPD':
            with torch.no_grad():
                cached=[rollout(bank,phi,cfg,'student_sample',noise=n,variant=variant,
                    collect_only=True,prepared=b) for n in noises]
                _finish_feedback(cached,b,variant,settings,meter,True)
                cached=_detach(cached)
            student_acquisitions+=2;all_actual.extend(cached)
        acquisitions.append(cached if arm!='Direct' else [])
        for inner in range(settings['inner_steps']):
            before=phi.detach().clone();opt_before=copy.deepcopy(optimizer.state_dict())
            before_hash=_optimizer_hash(opt_before);optimizer.zero_grad(set_to_none=True)
            if arm=='Direct':
                paths=[rollout(bank,phi,cfg,'direct_soft',noise=n,variant=variant,
                    collect_only=True,prepared=b) for n in noises]
                _finish_feedback(paths,b,variant,settings,meter,False);direct_recomputations+=2
            else:
                paths=[]
                for state in cached:
                    loss,KL,pKL,feedback,prior=_fixed_loss(state,phi,settings)
                    paths.append({**state,'loss':loss+phi.sum()*0,'KL_terms':KL,'prior_KL_terms':pKL,
                                  'feedback_loss':feedback,'prior_loss':prior})
            loss,feedback,prior=_batch_loss(paths,phi,settings)
            if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite evidence loss')
            loss.backward()
            if phi.grad is None or not bool(torch.isfinite(phi.grad).all()):raise FloatingPointError('Nonfinite evidence gradient')
            gradient_before=phi.grad.detach().clone();torch.nn.utils.clip_grad_norm_([phi],settings['grad_clip'],error_if_nonfinite=True)
            gradient=phi.grad.detach().clone();optimizer.step();proposal=phi.detach().clone()
            if not bool(torch.isfinite(proposal).all()):raise FloatingPointError('Nonfinite evidence weights')
            proposal_hash=_optimizer_hash(optimizer.state_dict());before_score=current_score
            with torch.no_grad():validation=_detach(rollout(bank,phi,cfg,variant=variant,collect_only=True,prepared=b))
            proposal_score=_score(bank,validation);rejected=proposal_score<before_score-1e-12
            if rejected:
                with torch.no_grad():phi.copy_(before)
                optimizer.load_state_dict(opt_before);stop=True
            else:current,current_score=validation,proposal_score;accepted+=1
            optimizer.zero_grad(set_to_none=True);after_hash=_optimizer_hash(optimizer.state_dict())
            if rejected:_require(torch.equal(phi.detach(),before) and after_hash==before_hash,'Incomplete Adam rollback')
            rounds.append(dict(round=len(rounds),outer=outer,inner=inner,before=before,proposal=proposal,
                after=phi.detach().clone(),gradient_before_clip=gradient_before,gradient=gradient,
                loss=float(loss.detach()),feedback_loss=float(feedback.detach()),prior_loss=float(prior.detach()),
                logit_margins=_margin(paths,before),before_score=before_score,proposal_score=proposal_score,
                after_score=current_score,accepted=not rejected,rolled_back=rejected,
                optimizer_before_sha256=before_hash,optimizer_proposal_sha256=proposal_hash,optimizer_after_sha256=after_hash))
            if arm=='Direct':saved=_detach(paths)
            else:
                # Share immutable acquisition tensors across ten inner traces.
                # Only current-loss tensors are new; torch.save preserves the
                # shared storage rather than cloning each hard history ten times.
                dynamic=('loss','KL_terms','prior_KL_terms','feedback_loss','prior_loss')
                saved=[{**state,**{name:_detach(path[name]) for name in dynamic},
                    'student_probabilities_at_acquisition':True} for state,path in zip(cached,paths)]
            training.append(saved);validations.append(validation);all_actual.append(validation)
            if arm=='Direct':all_actual.extend(saved)
            if stop:break
        if stop:break
    costs=dict(seconds=time.perf_counter()-begin,updates=len(rounds),backward=len(rounds),
        accepted_updates=accepted,rejected_updates=len(rounds)-accepted,state_only_backwards=len(rounds),
        heavy_forwards=0,decoder_calls=0,decoder_backwards=0,encoder_calls=0,expert_calls=0,
        outer_iterations=len(acquisitions),outer_student_acquisition_rollouts=student_acquisitions,
        direct_soft_recomputation_rollouts=direct_recomputations,teacher_setup_rollouts=len(fixed),
        teacher_setup_seconds=setup_seconds,cached_state_loss_passes=2*len(rounds) if arm!='Direct' else 0,
        initial_student_greedy_rollouts=1,postupdate_student_greedy_rollouts=len(rounds),
        fixed_score_evaluations=1+len(rounds),total_sequence_rollouts=len(all_actual),**meter)
    return dict(arm=arm,variant=variant,final=current['boxes'],finaltheta=phi.detach().clone(),
        initialtheta=torch.zeros(8,dtype=torch.float64),initial_score=initial_score,final_score=current_score,
        rounds=rounds,costs=costs,trace=dict(initial_rollout=initial,training_paths=training,
            inner_paths=training,outer_acquisitions=acquisitions,fixed_teacher_paths=fixed,
            validation_paths=validations,final_rollout=current,all_rollouts=all_actual),
        metadata=dict(feature_names=FEATURE_NAMES,CPU_only=True,precision='FP64',GT_read=False,
            hard_history_OPD='Supervised semigradient on each outer current-student acquisition; ten fixed-state inner steps',
            Direct='Recomputed soft histories each inner; full memory, exact Q, future reference probabilities and teacher-target gradients; not exact hard-history expected gradient',
            teacher_horizon=1 if variant=='A' else 4,student_logit_temperature=None,
            final_readout='last accepted greedy student; no sampled trajectory selection'))
