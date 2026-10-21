"""Evidence-gated matching adaptation on a fixed three-candidate bank.

This module reads no files, models, annotations or device assets. Every query
resets six temporary linear matching weights to zero. Student features exclude
the teacher unary score. Direct is a full-gradient Gumbel-soft relaxation;
hard-history OPD uses fixed conditional teacher targets, not a policy-gradient
estimator of the complete hard-history expected loss.
"""
from __future__ import annotations

import copy
import hashlib
import math
import time

import torch
import torch.nn.functional as F


ARMS = ('OPD', 'FixedKD', 'Direct')
MODES = ('student_sample','teacher_sample','student_greedy','teacher_greedy',
         'no_memory_teacher_greedy','direct_soft')
FEATURE_NAMES = ('memory_cosine','previous_selected_feature_cosine',
                 'previous_candidate_link','E_feature_cosine','negative_guard','is_E')


def _require(condition,message):
    if not condition:
        raise ValueError(message)


def _settings(cfg):
    def value(canonical,aliases,default):
        present = [cfg[key] for key in (canonical,*aliases) if key in cfg]
        _require(not present or all(float(x)==float(present[0]) for x in present),
                 'Conflicting canonical/legacy values for '+canonical)
        return float(present[0]) if present else default
    values = dict(seed=int(cfg.get('seed',41)),rounds=int(cfg.get('rounds',3)),
        paths=int(cfg.get('paths',2)),lr=float(cfg.get('lr',.1)),
        teacher_tau=value('teacher_tau',('teacher_temperature',),.2),
        prior_weight=value('prior_weight',('prior_kl_weight',),.05),
        memory_rate=value('memory_write_max',('memory_rate',),.2),
        grad_clip=value('gradient_clip_norm',('grad_clip',),1.),
        prior_E=value('original_prior',('prior_E',),.7),relaxation_temperature=float(
            cfg.get('Direct_relaxation_temperature',1.)))
    expected = dict(seed=41,rounds=3,paths=2,lr=.1,teacher_tau=.2,
        prior_weight=.05,memory_rate=.2,grad_clip=1.,prior_E=.7,
        relaxation_temperature=1.)
    _require(values==expected,'The isolated v0 protocol uses its locked configuration')
    _require(tuple(cfg.get('initial_weights',(0.,)*6))==(0.,)*6,
             'Initial matching weights must all be zero')
    _require(cfg.get('feedback_weights',dict(support=1.,link=.25,memory=.1,guard=.1))==
             dict(support=1.,link=.25,memory=.1,guard=.1),'Teacher feedback weights must match v0')
    return values


def _constant(value,name,dtype=torch.float64):
    value = torch.as_tensor(value)
    _require(value.device.type=='cpu',name+' must already reside on CPU')
    return value.detach().to(dtype=dtype)


def _prepare(bank):
    result = {name:_constant(getattr(bank,name),name,
        torch.bool if name in ('valid','active','observed') else torch.float64)
        for name in ('boxes','valid','features','anchor_features','confidence',
                     'active','observed','E','guard','link')}
    result['u'] = _constant(getattr(bank,'u',getattr(bank,'projection',None)),'u')
    T = len(result['E'])
    expected = dict(boxes=(T,3,4),valid=(T,3),features=(T,3,256),
        anchor_features=(T,256),confidence=(T,),active=(T,),observed=(T,),
        E=(T,4),guard=(T,3),link=(max(T-1,0),3,3),u=(T,3))
    _require(T>0,'An empty path is unsupported')
    for name,shape in expected.items():
        _require(result[name].shape==shape,name+' has an incompatible shape')
        if result[name].is_floating_point():
            _require(bool(torch.isfinite(result[name]).all()),name+' must be finite')
    _require(bool(result['valid'][:,0].all()),'E must remain valid everywhere')
    _require(torch.equal(result['boxes'][:,0],result['E']),'Candidate slot zero must equal E')
    _require(bool((result['boxes'][...,2:]>0).all()),'All stored boxes need positive sizes')
    _require(bool(((result['confidence']>=0)&(result['confidence']<=1)).all()),
             'Confidence must lie in [0,1]')
    _require(bool(((result['u']>=0)&(result['u']<=1)).all()),'Teacher unary must lie in [0,1]')
    result['features'] = F.normalize(result['features'],dim=-1,eps=1e-12)
    initial = getattr(bank,'initial_memory',result['anchor_features'][0])
    result['initial_memory'] = F.normalize(_constant(initial,'initial_memory'),dim=-1,eps=1e-12)
    _require(result['initial_memory'].shape==(256,), 'Initial teacher-seed memory needs 256 channels')
    result['T'] = T
    return result


def _weights(phi):
    phi = torch.as_tensor(phi)
    _require(phi.device.type=='cpu' and phi.dtype==torch.float64 and phi.shape==(6,),
             'Matching weights must be an FP64 CPU vector of length six')
    _require(bool(torch.isfinite(phi).all()),'Matching weights must be finite')
    return phi


def _gumbels(shape,generator):
    _require(generator is not None and generator.device.type=='cpu',
             'Stochastic paths require explicit private CPU noise')
    u = torch.rand(shape,dtype=torch.float64,generator=generator)
    f = torch.finfo(torch.float64)
    return -(-u.clamp(min=f.tiny,max=1-f.eps).log()).log()


def gumbel_noise(key,round_index,path_index,n_frames,seed=41):
    _require(seed==41 and round_index>=0 and path_index in (0,1) and n_frames>0,
             'Noise must obey seed41 and the two-path v0 protocol')
    token = f'propagation-v0|{seed}|{key}|{round_index}|{path_index}'
    generator = torch.Generator(device='cpu').manual_seed(
        int(hashlib.sha256(token.encode()).hexdigest()[:15],16))
    return _gumbels((n_frames,3),generator)


def _prior(valid,prior_E):
    prior = torch.zeros(3,dtype=torch.float64)
    alternatives = int(valid[1:].sum())
    if alternatives:
        prior[0] = prior_E
        prior[1:] = valid[1:].to(torch.float64)*(.3/alternatives)
    else:
        prior[0] = 1.
    return prior


def _log_probs(logits,valid):
    return F.log_softmax(logits.masked_fill(~valid,-torch.inf),dim=-1)


def _kl(log_p,log_q,valid):
    safe_p = log_p.masked_fill(~valid,0.)
    safe_q = log_q.masked_fill(~valid,0.)
    return (log_p.exp()*(safe_p-safe_q)).sum(-1)


def _detach(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value,dict):
        return {k:_detach(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [_detach(v) for v in value]
    return value


def _loss(teacher_log_probs,student_log_probs,prior_log_probs,valid,confidence,prior_weight):
    feedback_KL = _kl(teacher_log_probs,student_log_probs,valid)
    prior_KL = _kl(student_log_probs,prior_log_probs,valid)
    denominator = confidence.sum()+1e-12
    feedback_loss = (confidence*feedback_KL).sum()/denominator
    prior_loss = prior_KL.mean()
    return feedback_loss+prior_weight*prior_loss,feedback_KL,prior_KL,feedback_loss,prior_loss


def rollout(bank,phi,cfg,mode='student_greedy',generator=None,noise=None):
    settings = _settings(cfg)
    _require(mode in MODES,'Unsupported v0 rollout mode')
    b,phi = _prepare(bank),_weights(phi)
    T = b['T']
    if mode in ('student_sample','teacher_sample','direct_soft'):
        noise = _gumbels((T,3),generator) if noise is None else _constant(noise,'noise')
        _require(noise.shape==(T,3) and bool(torch.isfinite(noise).all()),
                 'Gumbel noise must be finite [T,3]')
    memory_disabled = mode=='no_memory_teacher_greedy'
    memory = torch.zeros_like(b['initial_memory']) if memory_disabled else b['initial_memory']
    previous = torch.tensor([1.,0.,0.],dtype=torch.float64)
    previous_feature = b['features'][0,0]
    is_E = torch.tensor([1.,0.,0.],dtype=torch.float64)
    fields = ['features','teacher_log_probs','student_log_probs','prior_log_probs',
        'teacherq','studentq','priorq','r','effective_valid','actions','action_probs',
        'previous_action_probs','previous_features','memory_before','memory_after',
        'selected_u','omega','boxes']
    collected = {field:[] for field in fields}
    for t in range(T):
        valid = b['valid'][t]
        if not bool(b['active'][t]):
            valid = torch.tensor([True,False,False])
        prior = _prior(valid,settings['prior_E'])
        prior_log = prior.log()
        memory_cos = (memory[None]*b['features'][t]).sum(-1)
        previous_cos = (F.normalize(previous_feature,dim=-1,eps=1e-12)[None]*b['features'][t]).sum(-1)
        previous_link = previous@b['link'][t-1] if t else torch.zeros(3,dtype=torch.float64)
        E_cos = (b['features'][t,0,None]*b['features'][t]).sum(-1)
        features = torch.stack((memory_cos,previous_cos,previous_link,E_cos,-b['guard'][t],is_E),dim=-1)
        student_logits = prior_log+features@phi  # No student temperature, no unary input.
        teacher_scores = b['u'][t]+.25*previous_link+.1*memory_cos-.1*b['guard'][t]
        teacher_log = _log_probs(teacher_scores/settings['teacher_tau'],valid)
        student_log = _log_probs(student_logits,valid)
        if mode!='direct_soft':
            teacher_log = teacher_log.detach()
        if not bool(b['active'][t]):
            action_probs,action = is_E,0
        elif mode=='direct_soft':
            action_probs = F.softmax((student_logits+noise[t]).masked_fill(~valid,-torch.inf)
                                      /settings['relaxation_temperature'],dim=-1)
            action = int(action_probs.detach().argmax())  # Diagnostic only, never used in state updates.
        else:
            selection = teacher_scores/settings['teacher_tau'] if mode in (
                'teacher_sample','teacher_greedy','no_memory_teacher_greedy') else student_logits
            if mode.endswith('_sample'):
                selection = selection+noise[t]
            action = int(selection.masked_fill(~valid,-torch.inf).detach().argmax())
            action_probs = F.one_hot(torch.tensor(action),num_classes=3).to(torch.float64)
        selected_feature = action_probs@b['features'][t]
        selected_u = action_probs@b['u'][t]
        omega = (settings['memory_rate']*b['confidence'][t]*selected_u
                 if bool(b['active'][t]) and not memory_disabled else selected_u*0)
        before = memory
        if bool(b['active'][t]) and not memory_disabled and bool(omega.detach()!=0):
            memory = F.normalize((1-omega)*memory+omega*selected_feature,dim=-1,eps=1e-12)
        values = dict(features=features,teacher_log_probs=teacher_log,student_log_probs=student_log,
            prior_log_probs=prior_log,teacherq=teacher_log.exp(),studentq=student_log.exp(),
            priorq=prior,r=teacher_scores,effective_valid=valid,actions=torch.tensor(action),
            action_probs=action_probs,previous_action_probs=previous,previous_features=previous_feature,
            memory_before=before,memory_after=memory,selected_u=selected_u,omega=omega,
            boxes=action_probs@b['boxes'][t])
        for field in fields:
            collected[field].append(values[field])
        previous,previous_feature = action_probs,selected_feature
    result = {field:torch.stack(values) for field,values in collected.items()}
    confidence = b['confidence']*b['active'].to(torch.float64)
    loss,KL,prior_KL,feedback_loss,prior_loss = _loss(result['teacher_log_probs'],
        result['student_log_probs'],result['prior_log_probs'],result['effective_valid'],
        confidence,settings['prior_weight'])
    result.update(loss=loss+phi.sum()*0,KL_terms=KL,prior_KL_terms=prior_KL,
        feedback_loss=feedback_loss,prior_loss=prior_loss,confidence=confidence,
        active=b['active'],observed=b['observed'],noise=noise,mode=mode,
        feature_names=FEATURE_NAMES,teacher_scores=result['r'],teacher_probs=result['teacherq'],
        teacher_target_detached=mode!='direct_soft',memory_disabled=memory_disabled,
        Direct_scope='Soft previous feature, link, selected unary, memory and teacher target all differentiate; this is a relaxation of hard histories')
    return result


def _fixed_loss(states,phi,settings):
    log_probs = _log_probs(states['prior_log_probs']+states['features']@phi,
                          states['effective_valid'])
    return _loss(states['teacher_log_probs'],log_probs,states['prior_log_probs'],
        states['effective_valid'],states['confidence'],settings['prior_weight'])


def _batch_loss(paths,phi,settings):
    # The attachment defines one normalization over both sampled paths, with
    # epsilon added once. Prior KL is the exact mean over all K*T states.
    confidence = torch.stack([p['confidence'] for p in paths])
    KL = torch.stack([p['KL_terms'] for p in paths])
    prior_KL = torch.stack([p['prior_KL_terms'] for p in paths])
    feedback = (confidence*KL).sum()/(confidence.sum()+1e-12)
    prior = prior_KL.mean()
    return feedback+settings['prior_weight']*prior+phi.sum()*0,feedback,prior


def _optimizer_hash(state):
    digest = hashlib.sha256()
    def add(value):
        if torch.is_tensor(value):
            x = value.detach().cpu().contiguous()
            digest.update(str((str(x.dtype),tuple(x.shape))).encode())
            digest.update(x.numpy().tobytes())
        elif isinstance(value,dict):
            for key in sorted(value,key=lambda k:(str(type(k)),str(k))):
                digest.update(repr(key).encode());add(value[key])
        elif isinstance(value,(list,tuple)):
            for item in value:add(item)
        else:digest.update(repr(value).encode())
    add(state)
    return digest.hexdigest()


def _score(bank,path):
    score = torch.as_tensor(bank.score_path(path['actions']),dtype=torch.float64)
    _require(score.device.type=='cpu' and score.ndim==0 and bool(torch.isfinite(score)),
             'The fixed evidence path score must be a finite CPU scalar')
    return float(score)


def initial_diagnostics(bank,key,cfg):
    settings = _settings(cfg)
    begin = time.perf_counter()
    phi = torch.zeros(6,dtype=torch.float64)
    with torch.no_grad():
        initial = _detach(rollout(bank,phi,cfg,'student_greedy'))
        _require(torch.equal(initial['boxes'],_constant(bank.E,'E')),
                 'The zero-initialized greedy student must exactly preserve E')
        paths = [_detach(rollout(bank,phi,cfg,'student_sample',noise=gumbel_noise(
            key,0,path_index,len(bank.E),settings['seed']))) for path_index in range(settings['paths'])]
    return dict(initialphi=phi,diagnostic_paths=paths,paths=paths,
        initial_student=initial,initial_final=initial['boxes'],
        costs=dict(seconds=time.perf_counter()-begin,initial_greedy_rollouts=1,
                   diagnostic_student_rollouts=2,total_sequence_rollouts=3,
                   updates=0,backward=0,heavy_forwards=0,decoder_calls=0))


def fit(bank,key,arm,cfg):
    _require(arm in ARMS,'Unsupported v0 adaptation arm')
    settings = _settings(cfg)
    begin = time.perf_counter()
    phi = torch.nn.Parameter(torch.zeros(6,dtype=torch.float64))
    optimizer = torch.optim.Adam([phi],lr=settings['lr'],betas=(.9,.999),eps=1e-8)
    with torch.no_grad():
        current = _detach(rollout(bank,phi,cfg,'student_greedy'))
    _require(torch.equal(current['boxes'],_constant(bank.E,'E')),
             'Initial greedy output must exactly equal E')
    initial = _detach(current)
    current_score = _score(bank,current)
    initial_score = current_score
    fixed = []
    teacher_begin = time.perf_counter()
    if arm=='FixedKD':
        with torch.no_grad():
            fixed = [_detach(rollout(bank,phi,cfg,'teacher_sample',noise=gumbel_noise(
                key,0,p,len(bank.E),settings['seed']))) for p in range(settings['paths'])]
    teacher_seconds = time.perf_counter()-teacher_begin
    rounds,training_paths,validation_paths = [],[],[]
    accepted = 0
    for r in range(settings['rounds']):
        before = phi.detach().clone()
        opt_before = copy.deepcopy(optimizer.state_dict())
        opt_before_hash = _optimizer_hash(opt_before)
        optimizer.zero_grad(set_to_none=True)
        if arm=='FixedKD':
            paths = []
            for states in fixed:
                loss,KL,prior_KL,feedback_loss,prior_loss = _fixed_loss(states,phi,settings)
                paths.append({**states,'loss':loss+phi.sum()*0,'KL_terms':KL,
                    'prior_KL_terms':prior_KL,'feedback_loss':feedback_loss,'prior_loss':prior_loss})
        else:
            paths = [rollout(bank,phi,cfg,'direct_soft' if arm=='Direct' else 'student_sample',
                noise=gumbel_noise(key,r,p,len(bank.E),settings['seed'])) for p in range(settings['paths'])]
        loss,feedback_loss,prior_loss = _batch_loss(paths,phi,settings)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Nonfinite v0 loss')
        loss.backward()
        if phi.grad is None or not bool(torch.isfinite(phi.grad).all()):
            raise FloatingPointError('Nonfinite v0 gradient')
        gradient_before_clip = phi.grad.detach().clone()
        torch.nn.utils.clip_grad_norm_([phi],settings['grad_clip'],error_if_nonfinite=True)
        gradient = phi.grad.detach().clone()
        optimizer.step()
        if not bool(torch.isfinite(phi).all()):
            raise FloatingPointError('Nonfinite updated v0 weights')
        proposal = phi.detach().clone()
        opt_proposal_hash = _optimizer_hash(optimizer.state_dict())
        with torch.no_grad():
            validation = _detach(rollout(bank,phi,cfg,'student_greedy'))
        proposal_score = _score(bank,validation)
        rejected = proposal_score < current_score-1e-12
        if rejected:
            with torch.no_grad():phi.copy_(before)
            optimizer.load_state_dict(opt_before)
        else:
            current,current_score = validation,proposal_score
            accepted += 1
        optimizer.zero_grad(set_to_none=True)
        opt_after_hash = _optimizer_hash(optimizer.state_dict())
        if rejected:
            _require(torch.equal(phi.detach(),before) and opt_after_hash==opt_before_hash,
                     'Rejected proposals must restore weights and full Adam state')
        rounds.append(dict(round=r,before=before,proposal=proposal,after=phi.detach().clone(),
            gradient_before_clip=gradient_before_clip,gradient=gradient,loss=float(loss.detach()),
            feedback_loss=float(feedback_loss.detach()),prior_loss=float(prior_loss.detach()),
            before_score=initial_score if r==0 else rounds[-1]['after_score'],
            proposal_score=proposal_score,after_score=current_score,accepted=not rejected,
            rolled_back=rejected,optimizer_before_sha256=opt_before_hash,
            optimizer_proposal_sha256=opt_proposal_hash,optimizer_after_sha256=opt_after_hash))
        training_paths.append(_detach(paths));validation_paths.append(validation)
        if rejected:break
    costs = dict(seconds=time.perf_counter()-begin,updates=len(rounds),backward=len(rounds),
        accepted_updates=accepted,rejected_updates=len(rounds)-accepted,state_only_backwards=len(rounds),
        heavy_forwards=0,decoder_backwards=0,decoder_calls=0,encoder_calls=0,expert_calls=0,
        teacher_setup_rollouts=len(fixed),teacher_setup_seconds=teacher_seconds,
        student_training_rollouts=0 if arm=='FixedKD' else len(rounds)*settings['paths'],
        cached_teacher_state_loss_passes=len(rounds)*settings['paths'] if arm=='FixedKD' else 0,
        initial_student_greedy_rollouts=1,postupdate_student_greedy_rollouts=len(rounds),
        fixed_score_evaluations=len(rounds)+1,
        total_sequence_rollouts=len(fixed)+1+len(rounds)+(0 if arm=='FixedKD' else len(rounds)*settings['paths']))
    all_rollouts = [initial,*fixed]
    if arm!='FixedKD':
        all_rollouts.extend(path for paths in training_paths for path in paths)
    all_rollouts.extend(validation_paths)
    return dict(arm=arm,final=current['boxes'],finalphi=phi.detach().clone(),finaltheta=phi.detach().clone(),
        initialphi=torch.zeros(6,dtype=torch.float64),initialtheta=torch.zeros(6,dtype=torch.float64),
        initial_score=initial_score,final_score=current_score,
        rounds=rounds,costs=costs,trace=dict(initial_rollout=initial,training_paths=training_paths,
            fixed_teacher_paths=fixed,validation_paths=validation_paths,final_rollout=current,
            all_rollouts=all_rollouts),
        metadata=dict(feature_names=FEATURE_NAMES,unary_is_student_feature=False,
            final_readout='last accepted greedy student; reject restores weights and Adam and stops',
            loss='confidence-normalized forward KL + .05 all-node mean KL(student||prior)',
            Direct_scope='Full soft previous feature/link/unary gate/memory/teacher-target gradient; relaxed, not exact hard-history expectation',
            CPU_only=True,precision='FP64',GT_read=False,network_weights_updated=False))


def fixed_propagation(bank,cfg,mode='teacher_greedy'):
    _require(mode in ('teacher_greedy','no_memory_teacher_greedy'),
             'Fixed propagation is the memory or no-memory teacher greedy control')
    begin = time.perf_counter()
    with torch.no_grad():
        path = _detach(rollout(bank,torch.zeros(6,dtype=torch.float64),cfg,mode))
    return dict(final=path['boxes'],score=_score(bank,path),trace=path,
        costs=dict(seconds=time.perf_counter()-begin,updates=0,backward=0,state_only_backwards=0,
            heavy_forwards=0,decoder_backwards=0,decoder_calls=0,encoder_calls=0,expert_calls=0,
            teacher_greedy_rollouts=1,total_sequence_rollouts=1,fixed_score_evaluations=1))


def evidence_dp(bank):
    """Exact maximum of the fixed additive evidence score, excluding memory."""
    begin = time.perf_counter()
    b = _prepare(bank)
    unary = _constant(bank.score_unary,'score_unary')
    pair = _constant(bank.score_pair,'score_pair')
    _require(unary.shape==(b['T'],3) and pair.shape==(max(0,b['T']-1),3,3),
             'DP requires additive unary and pair score tables')
    _require(bool(torch.isfinite(unary).all()) and bool(torch.isfinite(pair).all()),
             'DP score tables must be finite')
    valid = b['valid'].clone()
    valid[~b['active'],1:] = False
    values = unary[0].masked_fill(~valid[0],-torch.inf)
    back = []
    for t in range(1,b['T']):
        scores = values[:,None]+pair[t-1]+unary[t,None,:]
        values,previous = scores.max(dim=0)
        values = values.masked_fill(~valid[t],-torch.inf)
        back.append(previous)
    action = int(values.argmax())
    actions = [action]
    for previous in reversed(back):
        action = int(previous[action]);actions.append(action)
    actions = torch.tensor(list(reversed(actions)),dtype=torch.long)
    final = b['boxes'][torch.arange(b['T']),actions]
    score = _score(bank,dict(actions=actions))
    _require(abs(score-float(values.max()))<1e-10,'DP tables must exactly decompose score_path')
    return dict(final=final,score=score,trace=dict(actions=actions,boxes=final,score=score,
        objective='fixed additive score; exact global maximum, no memory or GT'),
        costs=dict(seconds=time.perf_counter()-begin,updates=0,backward=0,heavy_forwards=0,
            decoder_calls=0,total_sequence_rollouts=0,fixed_score_evaluations=1,
            dynamic_program_states=3*b['T']))
