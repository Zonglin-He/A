"""Four positive weights and a nine-state episodic source commitment.

All boxes and admission/reset evidence are fixed external tensors. There is no
appearance EMA, model, GT, RGB or decoder in this module. Hard histories retain
discrete source commitments; Direct differentiates the probability mixture of
exactly the same transitions, including all finite-horizon student probabilities.
"""
from __future__ import annotations

import copy
from contextlib import contextmanager
import math
import time
from collections import defaultdict

import torch
import torch.nn.functional as F

from scripts.propagation_v0.policy import _optimizer_hash,gumbel_noise

ARMS=('OPD','FixedKD','Direct')
MODES=('student_sample','teacher_sample','student_greedy','direct_soft')


def require(condition,message):
    if not condition:raise ValueError(message)


def detached(value):
    if torch.is_tensor(value):return value.detach().clone()
    if isinstance(value,dict):return {k:detached(v)for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [detached(v)for v in value]
    return value


def _settings(cfg):
    settings=dict(seed=int(cfg.get('seed',41)),rounds=int(cfg.get('rounds',3)),
        paths=int(cfg.get('paths',2)),inner_steps=int(cfg.get('inner_steps',10)),
        lr=float(cfg.get('lr',.1)),teacher_tau=float(cfg.get('teacher_tau',.2)),
        prior_weight=float(cfg.get('prior_weight',.05)),prior_E=float(cfg.get('original_prior',.7)),
        grad_clip=float(cfg.get('gradient_clip_norm',1.)),horizon=int(cfg.get('horizon',4)),
        route_strength=float(cfg.get('route_strength',2.)))
    require(settings==dict(seed=41,rounds=3,paths=2,inner_steps=10,lr=.1,teacher_tau=.2,
        prior_weight=.05,prior_E=.7,grad_clip=1.,horizon=4,route_strength=2.),'Locked commitment configuration differs')
    for key in ('initial_raw','initial_weights'):
        if key in cfg:require(tuple(cfg[key])==(0.,)*4,'Four raw parameters must reset to zero')
    return settings


def positive_weights(raw):
    require(raw.shape==(4,) and raw.dtype==torch.float64 and raw.device.type=='cpu',
            'Four FP64 CPU raw parameters required')
    require(bool(torch.isfinite(raw).all()),'Nonfinite raw weights')
    return 4*torch.sigmoid(raw-math.log(3.))


def _constant(value,dtype):
    require(torch.is_tensor(value) and value.device.type=='cpu','Inputs must already be CPU tensors')
    return value.detach().to(dtype)


def _prepare(bank,extra):
    b={k:_constant(getattr(bank,k),torch.bool if k in ('valid','active','observed')else torch.float64)
       for k in ('boxes','valid','active','observed','confidence','guard','link','E')}
    b.update({k:_constant(extra[k],torch.bool if k=='commit'else torch.long if k=='slot_routes'else torch.float64)
        for k in ('delta_support','external_support','candidate_trust','slot_routes','commit','route_compat','reset_matrices')})
    T=len(b['E']);b['T']=T
    shapes=dict(boxes=(T,3,4),valid=(T,3),active=(T,),observed=(T,),confidence=(T,),guard=(T,3),
        link=(T-1,3,3),E=(T,4),delta_support=(T,3),external_support=(T,3),candidate_trust=(T,3),
        slot_routes=(T,3),commit=(T,3),route_compat=(T,9,3),reset_matrices=(T,9,9))
    require(T>0,'Empty trajectory')
    for key,shape in shapes.items():
        require(b[key].shape==shape,'Incompatible '+key)
        if b[key].is_floating_point():require(bool(torch.isfinite(b[key]).all()),'Nonfinite '+key)
    require(bool(b['valid'][:,0].all()) and torch.equal(b['boxes'][:,0],b['E']),'E must be unchanged slot zero')
    require(bool((b['boxes'][...,2:]>0).all()),'Positive candidate sizes required')
    require(bool(((b['confidence']>=0)&(b['confidence']<=1)).all()),'Invalid confidence')
    require(bool(((b['slot_routes']>=0)&(b['slot_routes']<=8)).all()),'Source route out of range')
    require(bool(b['slot_routes'][:,0].eq(0).all()) and not bool(b['commit'][:,0].any()),'E cannot create a commitment')
    require(not bool((b['commit']&(~b['valid']|~b['active'][:,None]|b['slot_routes'].eq(0))).any()),'Invalid candidate admission')
    require(bool((b['delta_support'][:,0]==0).all()),'Delta support must be E-relative')
    R=b['reset_matrices']
    require(bool(((R==0)|(R==1)).all()) and bool(R.sum(-1).eq(1).all()),'Reset must be a deterministic row transition')
    require(torch.equal(R[:,0],F.one_hot(torch.zeros(T,dtype=torch.long),9).double()),'E source cannot be reset elsewhere')
    require(bool((b['route_compat'].abs()<=1).all()) and bool(b['route_compat'][:,0].eq(0).all())
        and bool(b['route_compat'][:,:,0].eq(0).all()),'Route compatibility must be bounded and E-neutral')
    zero=b['link'].new_zeros((1,3,3));incoming=torch.cat((zero,b['link']),0)
    edge=torch.zeros(T,dtype=torch.bool)
    edge[1:]=b['active'][:-1]&b['active'][1:]
    b['incoming_link']=incoming*edge[:,None,None]
    b['effective_valid']=b['valid']&b['active'][:,None]
    b['effective_valid'][:,0]=True
    alternatives=b['effective_valid'][:,1:].sum(-1)
    prior=torch.zeros(T,3,dtype=torch.float64)
    prior[:,0]=torch.where(alternatives>0,.7,1.)
    prior[:,1:]=b['effective_valid'][:,1:].double()*(.3/alternatives.clamp_min(1))[:,None]
    b['prior']=prior;b['logprior']=prior.log();b['c']=b['confidence']*b['active']
    return b


def _logp(logits,valid):return F.log_softmax(logits.masked_fill(~valid,-torch.inf),dim=-1)


def _kl(logp,logq,valid):
    return (logp.exp()*(logp.masked_fill(~valid,0)-logq.masked_fill(~valid,0))).sum(-1)


def _node(b,raw,nodes,history,previous):
    """Evaluate one batched visited state, applying the external reset first."""
    reset=torch.bmm(history[:,None],b['reset_matrices'][nodes]).squeeze(1)
    route=torch.bmm(reset[:,None],b['route_compat'][nodes]).squeeze(1)*b['commit'][nodes]
    incoming=b['incoming_link'][nodes]
    link=torch.bmm(previous[:,None],incoming).squeeze(1)-incoming[:,0,0,None]
    du=b['delta_support'][nodes];G=b['guard'][nodes];c=b['c'][nodes]
    features=torch.stack((2*du,2*c[:,None]*du,.25*link,-.1*G),-1)
    logits=b['logprior'][nodes]+features@positive_weights(raw)+2*route
    valid=b['effective_valid'][nodes];logp=_logp(logits,valid)
    rho=(du+.25*link-.1*G)*b['active'][nodes,None]
    return dict(reset=reset,route=route,features=features,logits=logits,logp=logp,valid=valid,rho=rho)


def _advance(b,nodes,reset,probs):
    admission=b['commit'][nodes].double()
    mass=(probs*admission).sum(-1)
    sources=F.one_hot(b['slot_routes'][nodes],9).double()
    commits=((probs*admission)[...,None]*sources).sum(1)
    return (1-mass[:,None])*reset+commits,mass


def exact_q(b,raw,nodes,history,previous,horizon=4,meter=None):
    """Force the current action, then exactly integrate the current student.

    No reward depends on route compatibility. All future policy probabilities,
    incoming soft states and reset/commit transitions retain their derivatives.
    Invalid/inactive starting candidates have Q zero and are masked by caller.
    """
    require(isinstance(horizon,int) and 1<=horizon<=4,'Invalid finite horizon')
    N=len(nodes);info=_node(b,raw,nodes,history,previous)
    eligible=info['valid']&b['active'][nodes,None]
    q=torch.where(eligible,info['rho'],torch.zeros_like(info['rho'])).reshape(-1)
    row,action=torch.nonzero(eligible,as_tuple=True)
    root_index=row*3+action;terminal=(nodes[row]+1>=b['T'])|(horizon==1)
    leaves=torch.zeros(N*3,dtype=torch.long)
    leaves.index_add_(0,root_index[terminal],torch.ones_like(root_index[terminal]))
    keep=~terminal;row,action,origin=row[keep],action[keep],root_index[keep]
    current_nodes=nodes[row]
    probs=F.one_hot(action,3).double()
    h,_=_advance(b,current_nodes,info['reset'][row],probs)
    prev=probs;current_nodes=current_nodes+1;mass=history.new_ones(len(row))
    states=[N];branches=[int(eligible.sum())];transitions=[len(row)]
    for depth in range(1,horizon):
        states.append(len(current_nodes))
        if len(current_nodes)==0:
            branches.append(0);transitions.append(0);continue
        branch=_node(b,raw,current_nodes,h,prev);p=branch['logp'].exp()
        q=q.index_add(0,origin,mass*(p*branch['rho']).sum(-1))
        parent,action=torch.nonzero(branch['valid'],as_tuple=True)
        branches.append(len(parent));roots=origin[parent]
        terminal=(current_nodes[parent]+1>=b['T'])|(depth==horizon-1)
        leaves.index_add_(0,roots[terminal],torch.ones_like(roots[terminal]))
        keep=~terminal;parent,action,origin=parent[keep],action[keep],roots[keep]
        transitions.append(len(parent));mass=mass[parent]*p[parent,action]
        forced=F.one_hot(action,3).double()
        h,_=_advance(b,current_nodes[parent],branch['reset'][parent],forced)
        prev=forced;current_nodes=current_nodes[parent]+1
    if meter is not None:
        meter['q_calls']+=1;meter['q_input_states']+=N
        meter['q_branch_state_evaluations']+=sum(states)
        meter['q_candidate_evaluations']+=3*sum(states)
        meter['q_valid_branches']+=sum(branches);meter['q_transitions']+=sum(transitions)
        meter['q_terminal_continuations']+=int(leaves.sum())
        for name,values in (('q_states_by_depth',states),('q_branches_by_depth',branches),('q_transitions_by_depth',transitions)):
            if name not in meter:meter[name]=[0]*horizon
            meter[name]=[a+c for a,c in zip(meter[name],values)]
    return q.reshape(N,3)


class _Clock:
    def __init__(self):self.stack=[];self.last=None;self.start=None;self.end=None;self.ns=defaultdict(int)
    @contextmanager
    def scope(self,name):
        now=time.perf_counter_ns()
        if self.stack:self.ns[self.stack[-1]]+=now-self.last
        else:self.start=now
        self.stack.append(name);self.last=now
        try:yield
        finally:
            now=time.perf_counter_ns();self.ns[self.stack[-1]]+=now-self.last
            self.stack.pop();self.last=now
            if not self.stack:self.end=now
    def report(self):
        require(not self.stack and self.start is not None and self.end is not None,'Incomplete timing')
        total=self.end-self.start;require(total==sum(self.ns.values()),'Timing overlap')
        return dict(seconds=total/1e9,exclusive_seconds={k:v/1e9 for k,v in self.ns.items()},
            exclusive_nanoseconds=dict(self.ns),exclusive_residual_nanoseconds=0)


def _scope(clock,name):
    if clock is None:
        from contextlib import nullcontext
        return nullcontext()
    return clock.scope(name)


def _attach(paths,b,raw,settings,meter,clock,detach_target):
    T=b['T'];nodes=torch.arange(T).repeat(len(paths))
    h=torch.cat([p['route_before']for p in paths]);previous=torch.cat([p['previous_action_probs']for p in paths])
    with _scope(clock,'feedback_Q'):
        Q=exact_q(b,raw,nodes,h,previous,horizon=settings['horizon'],meter=meter)
    for i,path in enumerate(paths):
        scores=Q[i*T:(i+1)*T]
        if detach_target:scores=scores.detach()
        _target(path,scores,settings,detach_target)
    return paths


def _target(path,Q,settings,detach_target):
    logits=path['prior_log_probs']+(Q-Q[:,0,None])/settings['teacher_tau']
    target=_logp(logits,path['effective_valid'])
    if detach_target:target=target.detach()
    path.update(Q=Q,teacher_scores=Q,teacher_log_probs=target,teacherq=target.exp(),
        teacher_probs=target.exp(),teacher_target_detached=detach_target)


def rollout(bank,extra,raw,cfg,mode='student_greedy',noise=None,prepared=None,
            collect_only=True,meter=None,clock=None):
    settings=_settings(cfg);require(mode in MODES,'Unknown rollout mode')
    b=_prepare(bank,extra)if prepared is None else prepared;positive_weights(raw);T=b['T']
    meter=defaultdict(int)if meter is None else meter
    if mode in ('student_sample','teacher_sample','direct_soft'):
        require(torch.is_tensor(noise) and noise.dtype==torch.float64 and noise.device.type=='cpu'
                and noise.shape==(T,3) and bool(torch.isfinite(noise).all()),'Explicit FP64 [T,3] Gumbels required')
    else:noise=None
    h=F.one_hot(torch.tensor(0),9).double();previous=torch.tensor([1.,0.,0.],dtype=torch.float64)
    names=('student_features','route_bias','student_log_probs','studentq','prior_log_probs','effective_valid',
        'route_before','route_after_reset','route_after','previous_action_probs','action_probs','actions','boxes',
        'rho','commit_mass','commit_event','switch_event','reset_mass','reset_event','committed_route','source_age')
    rows={k:[]for k in names};teacher=[];age=0;last_source=0
    for t in range(T):
        nodes=torch.tensor([t]);info=_node(b,raw,nodes,h[None],previous[None]);logp=info['logp'][0]
        valid=info['valid'][0];reset=info['reset'][0]
        if mode=='teacher_sample':
            with _scope(clock,'feedback_Q'):
                q=exact_q(b,raw,nodes,h[None],previous[None],settings['horizon'],meter)[0].detach()
            target=_logp(b['logprior'][t]+(q-q[0])/settings['teacher_tau'],valid)
            selection=target;teacher.append(q)
        else:selection=logp
        if mode=='direct_soft':
            probs=(info['logits'][0]+noise[t]).masked_fill(~valid,-torch.inf).softmax(-1)
            action=int(probs.detach().argmax())
        else:
            if mode.endswith('_sample'):selection=selection+noise[t]
            action=int(selection.detach().masked_fill(~valid,-torch.inf).argmax())
            probs=F.one_hot(torch.tensor(action),3).double()
        after,mass=_advance(b,nodes,reset[None],probs[None]);after=after[0];mass=mass[0]
        source=int(after.detach().argmax());incoming_source=int(reset.detach().argmax())
        reset_mass=(reset[0]-h[0]).clamp_min(0)
        commit_event=bool(b['commit'][t,action]);switch=commit_event and source!=incoming_source
        age=age+1 if source==last_source and source!=0 else 0;last_source=source
        values=dict(student_features=info['features'][0],route_bias=2*info['route'][0],
            student_log_probs=logp,studentq=logp.exp(),prior_log_probs=b['logprior'][t],effective_valid=valid,
            route_before=h,route_after_reset=reset,route_after=after,previous_action_probs=previous,
            action_probs=probs,actions=torch.tensor(action),boxes=probs@b['boxes'][t],rho=info['rho'][0],
            commit_mass=mass,commit_event=torch.tensor(commit_event),switch_event=torch.tensor(switch),
            reset_mass=reset_mass,reset_event=reset_mass.detach()>1e-12,committed_route=torch.tensor(source),source_age=torch.tensor(age))
        for key in names:rows[key].append(values[key])
        h,previous=after,probs
    path={k:torch.stack(v)for k,v in rows.items()}
    path.update(active=b['active'],observed=b['observed'],confidence=b['c'],noise=noise,mode=mode,
        candidate_commit_allowed=b['commit'],candidate_trust=b['candidate_trust'],slot_routes=b['slot_routes'],
        Q=None,teacherq=None,teacher_log_probs=None,teacher_scores=None,
        state_semantics='nine source provenance states; no appearance EMA',raw=raw.detach().clone())
    meter['policy_rollouts']+=1;meter['policy_nodes']+=T
    if mode=='teacher_sample':_target(path,torch.stack(teacher),settings,True)
    elif not collect_only:_attach([path],b,raw,settings,meter,clock,mode!='direct_soft')
    return path


def utility(bank,extra,actions,prepared=None):
    b=_prepare(bank,extra)if prepared is None else prepared
    require(actions.shape==(b['T'],) and actions.dtype==torch.long,'Hard action trajectory required')
    require(bool(b['effective_valid'][torch.arange(b['T']),actions].all()),'Invalid utility path')
    unary=((b['delta_support']-.1*b['guard'])*b['active'][:,None])[torch.arange(b['T']),actions].sum()
    if b['T']>1:
        edge=b['active'][:-1]&b['active'][1:]
        pairs=b['link'][torch.arange(b['T']-1),actions[:-1],actions[1:]]-b['link'][:,0,0]
        unary=unary+.25*(pairs*edge).sum()
    return unary


def exact_dp(bank,extra,cfg=None):
    """Exact maximizer of the external additive utility, not a KL optimum."""
    b=_prepare(bank,extra);start=time.perf_counter();T=b['T']
    unary=(b['delta_support']-.1*b['guard'])*b['active'][:,None]
    values=unary[0].masked_fill(~b['effective_valid'][0],-torch.inf);back=[]
    for t in range(1,T):
        pair=.25*(b['incoming_link'][t]-b['incoming_link'][t,0,0])
        scores=values[:,None]+pair;best,parent=scores.max(0)
        values=(best+unary[t]).masked_fill(~b['effective_valid'][t],-torch.inf);back.append(parent)
    action=int(values.argmax());actions=[action]
    for parent in reversed(back):action=int(parent[action]);actions.append(action)
    actions=torch.tensor(list(reversed(actions)),dtype=torch.long)
    return dict(final=b['boxes'][torch.arange(T),actions],actions=actions,score=float(utility(bank,extra,actions,b)),
        trace=dict(actions=actions),costs=dict(seconds=time.perf_counter()-start,updates=0,backward=0,
        heavy_forwards=0,decoder_calls=0,expert_calls=0,encoder_calls=0),
        metadata=dict(objective='exact maximum external sum utility; no route reward or prior/KL optimum'))


evidence_dp=exact_dp
fixed_dp=exact_dp


def initial(bank,extra,cfg):
    """Unadapted raw-zero policy, kept distinct from the E trajectory."""
    begin=time.perf_counter();raw=torch.zeros(4,dtype=torch.float64)
    with torch.no_grad():path=detached(rollout(bank,extra,raw,cfg))
    return dict(final=path['boxes'],trace=path,initialtheta=raw.clone(),finaltheta=raw,
        score=float(utility(bank,extra,path['actions'])),costs=dict(seconds=time.perf_counter()-begin,
        updates=0,backward=0,heavy_forwards=0,decoder_calls=0,expert_calls=0,encoder_calls=0,
        policy_forward_calls=1,policy_forward_nodes=len(path['actions'])))


def _loss(paths,raw,settings,fixed):
    KL=[];prior_KL=[];probabilities=[]
    for path in paths:
        logp=_logp(path['prior_log_probs']+path['student_features']@positive_weights(raw)+path['route_bias'],
                   path['effective_valid'])if fixed else path['student_log_probs']
        KL.append(_kl(path['teacher_log_probs'],logp,path['effective_valid']))
        prior_KL.append(_kl(logp,path['prior_log_probs'],path['effective_valid']))
        probabilities.append(logp)
    c=torch.stack([p['confidence']for p in paths]);KL=torch.stack(KL);prior_KL=torch.stack(prior_KL)
    feedback=(c*KL).sum()/(c.sum()+1e-12);prior=prior_KL.mean()
    return feedback+settings['prior_weight']*prior+raw.sum()*0,feedback,prior,KL,prior_KL,probabilities


def fit(bank,extra,key,arm,cfg):
    require(arm in ARMS,'Unknown commitment arm');settings=_settings(cfg)
    clock=_Clock();meter=defaultdict(int)
    with clock.scope('bookkeeping'):
        with clock.scope('bank_preparation'):b=_prepare(bank,extra)
        raw=torch.nn.Parameter(torch.zeros(4,dtype=torch.float64));optimizer=torch.optim.Adam([raw],lr=.1,betas=(.9,.999),eps=1e-8)
        with torch.no_grad(),clock.scope('greedy_validation'):
            current=detached(rollout(bank,extra,raw,cfg,prepared=b,meter=meter))
            score=float(utility(bank,extra,current['actions'],b))
        initial=current;initial_score=score;fixed=[]
        if arm=='FixedKD':
            with torch.no_grad(),clock.scope('history_construction'):
                fixed=[detached(rollout(bank,extra,raw,cfg,'teacher_sample',
                    noise=gumbel_noise(key,0,p,b['T']),prepared=b,meter=meter,clock=clock))for p in range(2)]
        rounds=[];training=[];histories=[];validations=[];all_rollouts=[initial,*fixed];accepted=0
        for outer in range(3):
            before=raw.detach().clone();opt_before=copy.deepcopy(optimizer.state_dict());hash_before=_optimizer_hash(opt_before)
            before_score=score;noises=[gumbel_noise(key,outer,p,b['T'])for p in range(2)]
            cached=[]
            if arm=='OPD':
                with torch.no_grad(),clock.scope('history_construction'):
                    cached=[rollout(bank,extra,raw,cfg,'student_sample',noise=n,prepared=b,meter=meter)for n in noises]
                all_rollouts.extend(cached);meter['student_acquisition_rollouts']+=2
            elif arm=='FixedKD':cached=[{**p}for p in fixed]
            if arm!='Direct':
                with torch.no_grad(),clock.scope('feedback_targets'):
                    _attach(cached,b,raw,settings,meter,clock,True)
                # These states are fixed for ten inner steps; targets refresh
                # once every outer for BOTH OPD and the fixed-history control.
                cached=detached(cached)
            histories.append(cached);inner_rows=[];outer_training=[]
            for inner in range(10):
                optimizer.zero_grad(set_to_none=True)
                if arm=='Direct':
                    with clock.scope('history_construction'):
                        paths=[rollout(bank,extra,raw,cfg,'direct_soft',noise=n,prepared=b,meter=meter)for n in noises]
                    with clock.scope('feedback_targets'):_attach(paths,b,raw,settings,meter,clock,False)
                    meter['direct_soft_rollouts']+=2
                else:paths=cached
                with clock.scope('objective'):
                    loss,feedback,prior,KL,pKL,logs=_loss(paths,raw,settings,arm!='Direct')
                require(bool(torch.isfinite(loss)),'Nonfinite commitment objective')
                inner_before=raw.detach().clone()
                with clock.scope('backward'):
                    loss.backward();require(raw.grad is not None and bool(torch.isfinite(raw.grad).all()),'Nonfinite commitment gradient')
                unscaled=raw.grad.detach().clone()
                with clock.scope('parameter_update'):
                    torch.nn.utils.clip_grad_norm_([raw],1.,error_if_nonfinite=True);gradient=raw.grad.detach().clone();optimizer.step()
                require(bool(torch.isfinite(raw).all()),'Nonfinite commitment parameters')
                meter['updates']+=1;meter['backward']+=1
                inner_rows.append(dict(inner=inner,before=inner_before,after=raw.detach().clone(),
                    loss=float(loss.detach()),feedback_loss=float(feedback.detach()),prior_loss=float(prior.detach()),
                    gradient_before_clip=unscaled,gradient=gradient))
                if arm=='Direct':
                    saved=detached(paths);all_rollouts.extend(saved)
                else:saved=[{**p}for p in cached]
                for p,path in enumerate(saved):
                    path.update(KL_terms=KL[p].detach().clone(),prior_KL_terms=pKL[p].detach().clone(),
                        evaluated_student_log_probs=logs[p].detach().clone(),student_probabilities_at_acquisition=arm!='Direct')
                outer_training.append(saved)
            proposal=raw.detach().clone();proposal_hash=_optimizer_hash(optimizer.state_dict())
            with torch.no_grad(),clock.scope('greedy_validation'):
                validation=detached(rollout(bank,extra,raw,cfg,prepared=b,meter=meter))
                proposal_score=float(utility(bank,extra,validation['actions'],b))
            reject=proposal_score<before_score-1e-12
            if reject:
                with clock.scope('rollback'):
                    with torch.no_grad():raw.copy_(before)
                    optimizer.load_state_dict(opt_before)
            else:current=validation;score=proposal_score;accepted+=1
            optimizer.zero_grad(set_to_none=True);after_hash=_optimizer_hash(optimizer.state_dict())
            if reject:require(torch.equal(raw.detach(),before) and after_hash==hash_before,'Incomplete outer rollback')
            rounds.append(dict(round=outer,outer=outer,before=before,proposal=proposal,after=raw.detach().clone(),
                before_score=before_score,proposal_score=proposal_score,after_score=score,accepted=not reject,
                rolled_back=reject,inner=inner_rows,optimizer_before_sha256=hash_before,
                optimizer_proposal_sha256=proposal_hash,optimizer_after_sha256=after_hash))
            training.append(outer_training);validations.append(validation);all_rollouts.append(validation)
        # The last accepted greedy state already exists. Add precisely one
        # batched final feedback evaluation, equally for all three arms.
        current={**current}
        with torch.no_grad(),clock.scope('final_feedback'):_attach([current],b,raw,settings,meter,clock,True)
        final_weights=positive_weights(raw).detach().clone()
    timing=clock.report()
    costs=dict(**timing,**meter,accepted_outer=accepted,rejected_outer=3-accepted,
        outer_iterations=3,inner_optimizer_steps=30,state_only_backwards=30,
        heavy_forwards=0,decoder_calls=0,decoder_backwards=0,encoder_calls=0,expert_calls=0,
        teacher_setup_rollouts=len(fixed),postupdate_student_greedy_rollouts=3,initial_student_greedy_rollouts=1,
        fixed_utility_evaluations=4,final_feedback_batches=1,total_sequence_rollouts=len(all_rollouts))
    return dict(arm=arm,final=current['boxes'],initialtheta=torch.zeros(4,dtype=torch.float64),
        finaltheta=raw.detach().clone(),finalweights=final_weights,initial_score=initial_score,final_score=score,
        rounds=rounds,costs=costs,trace=dict(initial_rollout=initial,final_rollout=current,
            training_paths=training,outer_histories=histories,fixed_teacher_paths=fixed,
            validation_paths=validations,all_rollouts=all_rollouts),
        metadata=dict(GT_read=False,CPU_only=True,precision='FP64',appearance_EMA=False,
            state='nine discrete evidence-source commitments; Direct uses the exact transition probability mixture',
            teacher='Q4 under current student; external rho and identical prior; no route self reward',
            loss='confidence-normalized forward KL plus .05 exact mean prior KL',
            derivative='Direct full soft history, raw weights, branch probabilities and teacher target gradients; not an exact hard-history expected gradient',
            final_readout='last accepted greedy after three outers; no GT or sampled-best choice',
            outer_rejection='restore full raw+Adam, then continue subsequent outer refresh'))
