"""CPU-only episodic propagation policy and matched supervision controls.

Only six temporary matching weights are updated. Candidate trajectories and
all visual evidence are fixed cached inputs. The on-policy arm fits teacher
feedback on hard states visited by its current policy; FixedKD reuses two
teacher histories, and Direct differentiates the same KL through a Gumbel-soft
history. That relaxation is not the exact gradient of the hard-path expectation.
"""
from __future__ import annotations

import hashlib
import math
import time

import torch
import torch.nn.functional as F


ARMS = ('OPD', 'FixedKD', 'Direct')
INITIAL_WEIGHTS = (1., 0., 0., 0., 0., 0.)
MODES = ('student_sample', 'teacher_sample', 'student_greedy',
         'teacher_greedy', 'direct_soft')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _settings(cfg):
    settings = dict(seed=int(cfg.get('seed', 41)), rounds=int(cfg.get('rounds', 3)),
        paths=int(cfg.get('paths', cfg.get('maxpaths', 2))), lr=float(cfg.get('lr', .1)),
        tau=float(cfg.get('tau', .1)), prior_other_logit=float(cfg.get('prior_other_logit', -2.)),
        memory_rate=float(cfg.get('memory_rate', .5)),
        relaxation_temperature=float(cfg.get('Direct_relaxation_temperature', 1.)))
    expected = dict(seed=41, rounds=3, paths=2, lr=.1, tau=.1,
        prior_other_logit=-2., memory_rate=.5, relaxation_temperature=1.)
    _require(settings == expected, 'This isolated protocol uses the locked propagation configuration')
    _require(tuple(cfg.get('initial_weights', INITIAL_WEIGHTS)) == INITIAL_WEIGHTS,
             'Each query must reset the six matching weights to theta0')
    return settings


def _constant(value, name, dtype=torch.float64):
    tensor = torch.as_tensor(value)
    _require(tensor.device.type == 'cpu', name+' must already be on CPU')
    return tensor.detach().to(dtype=dtype)


def _prepare(bank):
    boxes = _constant(bank.boxes, 'boxes')
    _require(boxes.ndim == 3 and boxes.shape[1:] == (4, 4),
             'Candidate boxes must have shape [T,4,4]')
    T = len(boxes)
    prepared = dict(boxes=boxes, valid=_constant(bank.valid, 'valid', torch.bool),
        features=_constant(bank.features, 'features'),
        anchor_features=_constant(bank.anchor_features, 'anchor_features'),
        confidence=_constant(bank.confidence, 'confidence'),
        active=_constant(bank.active, 'active', torch.bool), E=_constant(bank.E, 'E'),
        projection=_constant(bank.projection, 'projection'), guard=_constant(bank.guard, 'guard'),
        link=_constant(bank.link, 'link'), observed=_constant(bank.observed, 'observed', torch.bool))
    expected = dict(valid=(T,4), features=(T,4,256), anchor_features=(T,256),
        confidence=(T,), active=(T,), E=(T,4), projection=(T,4), guard=(T,4),
        link=(max(0,T-1),4,4), observed=(T,))
    _require(T > 0, 'A propagation trajectory cannot be empty')
    for name, shape in expected.items():
        _require(prepared[name].shape == shape, f'{name} has an invalid shape')
    for name, tensor in prepared.items():
        if tensor.is_floating_point():
            _require(bool(torch.isfinite(tensor).all()), name+' must be finite')
    _require(bool(prepared['valid'][:,0].all()), 'E must be valid at every physical node')
    _require(torch.equal(boxes[:,0], prepared['E']), 'Candidate zero must be exactly E')
    _require(bool((boxes[...,2:] > 0).all()), 'Candidate dimensions must be positive')
    _require(bool(((prepared['confidence'] >= 0)&(prepared['confidence'] <= 1)).all()),
             'Confidence must lie in [0,1]')
    prepared['features'] = F.normalize(prepared['features'], dim=-1, eps=1e-12)
    prepared['anchor_features'] = F.normalize(prepared['anchor_features'], dim=-1, eps=1e-12)
    prepared['anchor_cosine'] = (prepared['anchor_features'][:,None]*prepared['features']).sum(-1)
    prepared['T'] = T
    return prepared


def _theta(theta):
    theta = torch.as_tensor(theta)
    _require(theta.device.type == 'cpu' and theta.dtype == torch.float64 and theta.shape == (6,),
             'Temporary matching weights must be an FP64 CPU vector of length six')
    _require(bool(torch.isfinite(theta).all()), 'Temporary matching weights must be finite')
    return theta


def _gumbels(shape, generator):
    _require(generator is not None and generator.device.type == 'cpu',
             'Stochastic rollouts require explicit CPU noise or a private generator')
    u = torch.rand(shape, generator=generator, dtype=torch.float64)
    tiny, eps = torch.finfo(torch.float64).tiny, torch.finfo(torch.float64).eps
    return -torch.log(-torch.log(u.clamp(min=tiny, max=1-eps)))


def gumbel_noise(key, round_index, path_index, n_frames, seed=41):
    _require(seed == 41 and round_index >= 0 and path_index in (0,1) and n_frames > 0,
             'Noise must follow the fixed seed41 / two-path protocol')
    token = f'propagation-opd-v1|{seed}|{key}|{round_index}|{path_index}'
    private_seed = int(hashlib.sha256(token.encode()).hexdigest()[:15],16)
    generator = torch.Generator(device='cpu').manual_seed(private_seed)
    return _gumbels((n_frames,4), generator)


def _log_probs(logits, valid):
    return F.log_softmax(logits.masked_fill(~valid, -torch.inf), dim=-1)


def _kl(target_log_probs, student_log_probs, valid):
    # Replacing invalid log entries before subtraction prevents 0*(-inf- -inf)
    # and preserves finite gradients for both student and Direct teacher paths.
    target_safe = target_log_probs.masked_fill(~valid, 0.)
    student_safe = student_log_probs.masked_fill(~valid, 0.)
    return (target_log_probs.exp()*(target_safe-student_safe)).sum(-1)


def _detach(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key:_detach(item) for key,item in value.items()}
    if isinstance(value, (list,tuple)):
        return [_detach(item) for item in value]
    return value


def rollout(bank, theta, cfg, mode='student_greedy', generator=None, noise=None):
    """Execute one complete history, using either hard actions or soft history.

    A teacher target never uses memory cosine. Its previous-action link still
    depends on the same visited history. Only Direct keeps that target/history
    derivative; hard actions in OPD/FixedKD produce fixed state supervision.
    """
    settings = _settings(cfg)
    _require(mode in MODES, 'Unknown propagation rollout mode')
    prepared = _prepare(bank)
    theta = _theta(theta)
    T = prepared['T']
    if mode in ('student_sample','teacher_sample','direct_soft'):
        noise = _gumbels((T,4),generator) if noise is None else _constant(noise,'noise')
        _require(noise.shape == (T,4) and bool(torch.isfinite(noise).all()),
                 'Each path needs finite Gumbel noise [T,4]')
    memory = prepared['anchor_features'][0]
    previous = torch.tensor([1.,0.,0.,0.], dtype=torch.float64)
    base = torch.tensor([0.,settings['prior_other_logit'],settings['prior_other_logit'],
                         settings['prior_other_logit']], dtype=torch.float64)
    is_E = torch.tensor([1.,0.,0.,0.], dtype=torch.float64)
    collected = {key:[] for key in ['features','teacher_log_probs','student_log_probs',
        'effective_valid','memory_before','memory_after','previous_action_probs',
        'action_probs','actions','boxes','KL_terms']}
    for t in range(T):
        valid = prepared['valid'][t]
        memory_cos = (memory[None]*prepared['features'][t]).sum(-1)
        previous_link = (previous@prepared['link'][t-1] if t else torch.zeros(4,dtype=torch.float64))
        features = torch.stack((memory_cos, prepared['anchor_cosine'][t],
            prepared['projection'][t], previous_link, -prepared['guard'][t], is_E), dim=-1)
        student_logits = base + (features@theta)/settings['tau']
        teacher_score = (prepared['projection'][t] + .1*prepared['anchor_cosine'][t]
                         + .1*previous_link - .1*prepared['guard'][t])
        teacher_logits = base + teacher_score/settings['tau']
        if not bool(prepared['active'][t]):
            valid = torch.tensor([True,False,False,False])
        student_log_probs = _log_probs(student_logits,valid)
        teacher_log_probs = _log_probs(teacher_logits,valid)
        if mode != 'direct_soft':
            teacher_log_probs = teacher_log_probs.detach()
        if not bool(prepared['active'][t]):
            action_probs = is_E
            action = 0
        elif mode == 'direct_soft':
            action_probs = F.softmax((student_logits+noise[t]).masked_fill(~valid,-torch.inf)
                                      /settings['relaxation_temperature'], dim=-1)
            action = int(action_probs.detach().argmax())
        else:
            selection_logits = teacher_logits if mode.startswith('teacher_') else student_logits
            if mode.endswith('_sample'):
                selection_logits = selection_logits+noise[t]
            action = int(selection_logits.masked_fill(~valid,-torch.inf).detach().argmax())
            action_probs = F.one_hot(torch.tensor(action),num_classes=4).to(torch.float64)
        KL = _kl(teacher_log_probs,student_log_probs,valid)
        before = memory
        if bool(prepared['active'][t]):
            if bool(prepared['observed'][t]):
                memory = prepared['anchor_features'][t]
            else:
                selected_feature = action_probs@prepared['features'][t]
                memory = F.normalize((1-settings['memory_rate'])*memory +
                    settings['memory_rate']*selected_feature,dim=-1,eps=1e-12)
        collected['features'].append(features)
        collected['teacher_log_probs'].append(teacher_log_probs)
        collected['student_log_probs'].append(student_log_probs)
        collected['effective_valid'].append(valid)
        collected['memory_before'].append(before)
        collected['memory_after'].append(memory)
        collected['previous_action_probs'].append(previous)
        collected['action_probs'].append(action_probs)
        collected['actions'].append(torch.tensor(action))
        collected['boxes'].append(action_probs@prepared['boxes'][t])
        collected['KL_terms'].append(KL)
        previous = action_probs
    result = {key:torch.stack(value) for key,value in collected.items()}
    weights = prepared['confidence']*prepared['active'].to(torch.float64)
    denominator = max(1,int(prepared['active'].sum()))
    result.update(loss=(weights*result['KL_terms']).sum()/denominator+theta.sum()*0,
        weights=weights, active=prepared['active'], observed=prepared['observed'],
        denominator=denominator, mode=mode, base_logits=base, noise=noise,
        teacher_target_detached=mode!='direct_soft',
        history='Gumbel-soft relaxation' if mode=='direct_soft' else 'hard selected candidates')
    return result


def _fixed_loss(states, theta, settings):
    logits = states['base_logits'] + (states['features']@theta)/settings['tau']
    log_probs = _log_probs(logits,states['effective_valid'])
    KL = _kl(states['teacher_log_probs'],log_probs,states['effective_valid'])
    return (states['weights']*KL).sum()/states['denominator']+theta.sum()*0, KL


def _path_stats(paths):
    actions = torch.cat([p['actions'] for p in paths])
    active_actions = torch.cat([p['actions'][p['active']] for p in paths])
    drift = []
    for path in paths:
        baseline = path['memory_before'][0]
        drift.append((path['memory_before']-baseline[None]).norm(dim=-1).mean())
    return dict(visited_actions=torch.bincount(actions,minlength=4).tolist(),
        visited_active_actions=torch.bincount(active_actions,minlength=4).tolist(),
        memory_drift_from_initial=float(torch.stack(drift).mean().detach()),
        memory_drift_metric='mean L2 distance of normalized memory from its initial value',
        mean_KL=float(torch.stack([p['KL_terms'].mean() for p in paths]).mean().detach()),
        fixed_teacher_memory_drift_scope='Computed offline by pairing saved histories with the two FixedKD teacher histories; no extra diagnostic rollouts')


def fit(bank, key, arm, cfg):
    _require(arm in ARMS, 'Unknown propagation arm')
    settings = _settings(cfg)
    begin = time.perf_counter()
    theta = torch.nn.Parameter(torch.tensor(INITIAL_WEIGHTS,dtype=torch.float64))
    optimizer = torch.optim.Adam([theta],lr=settings['lr'],betas=(.9,.999),eps=1e-8)
    T = len(bank.E)
    fixed_states = []
    teacher_setup_begin = time.perf_counter()
    if arm == 'FixedKD':
        for path_index in range(settings['paths']):
            noise = gumbel_noise(key,0,path_index,T,seed=settings['seed'])
            with torch.no_grad():
                fixed_states.append(_detach(rollout(bank,theta,cfg,'teacher_sample',noise=noise)))
    teacher_setup_seconds = time.perf_counter()-teacher_setup_begin
    rounds, traces = [], []
    for round_index in range(settings['rounds']):
        optimizer.zero_grad(set_to_none=True)
        before = theta.detach().clone()
        paths = []
        if arm == 'FixedKD':
            losses = []
            for states in fixed_states:
                loss, KL = _fixed_loss(states,theta,settings)
                losses.append(loss)
                paths.append({**states,'loss':loss,'KL_terms':KL})
        else:
            for path_index in range(settings['paths']):
                noise = gumbel_noise(key,round_index,path_index,T,seed=settings['seed'])
                paths.append(rollout(bank,theta,cfg,
                    'direct_soft' if arm=='Direct' else 'student_sample',noise=noise))
            losses = [path['loss'] for path in paths]
        loss = torch.stack(losses).mean()
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Nonfinite propagation KL objective')
        loss.backward()
        if theta.grad is None or not bool(torch.isfinite(theta.grad).all()):
            raise FloatingPointError('Nonfinite propagation weight gradient')
        gradient = theta.grad.detach().clone()
        optimizer.step()
        if not bool(torch.isfinite(theta).all()):
            raise FloatingPointError('Nonfinite updated propagation weights')
        rounds.append(dict(round=round_index,before=before,after=theta.detach().clone(),
            gradient=gradient,loss=float(loss.detach()),**_path_stats(paths)))
        traces.append(_detach(paths))
    with torch.no_grad():
        final = _detach(rollout(bank,theta,cfg,'student_greedy'))
    optimizer.zero_grad(set_to_none=True)
    costs = dict(seconds=time.perf_counter()-begin,updates=settings['rounds'],
        backward=settings['rounds'],state_only_backwards=settings['rounds'],decoder_backwards=0,
        heavy_forwards=0,decoder_calls=0,encoder_calls=0,expert_calls=0,
        student_training_rollouts=0 if arm=='FixedKD' else settings['rounds']*settings['paths'],
        teacher_setup_rollouts=len(fixed_states),teacher_setup_seconds=teacher_setup_seconds,
        cached_teacher_state_loss_passes=settings['rounds']*settings['paths'] if arm=='FixedKD' else 0,
        final_student_greedy_rollouts=1,
        total_sequence_rollouts=len(fixed_states)+1+(0 if arm=='FixedKD' else settings['rounds']*settings['paths']))
    return dict(arm=arm,final=final['boxes'],finaltheta=theta.detach().clone(),
        initialtheta=torch.tensor(INITIAL_WEIGHTS,dtype=torch.float64),rounds=rounds,costs=costs,
        trace=dict(training_paths=traces,fixed_teacher_paths=fixed_states,final_rollout=final),
        metadata=dict(device='CPU',precision='FP64',GT_read=False,
            final_readout='one greedy current student path; no best-trajectory selection',
            loss='mean over two paths of confidence-weighted forward KL divided by active node count',
            teacher_memory_cosine_used=False,
            Direct_gradient_scope='Full gradient through soft memory, previous-action link, and teacher target; a relaxation, not the exact hard-path expected gradient',
            candidate_bank_fixed=True,network_weights_updated=False))


def fixed_propagation(bank,cfg):
    begin = time.perf_counter()
    theta = torch.tensor(INITIAL_WEIGHTS,dtype=torch.float64)
    with torch.no_grad():
        trace = _detach(rollout(bank,theta,cfg,'teacher_greedy'))
    return dict(final=trace['boxes'],finaltheta=theta,trace=trace,
        costs=dict(seconds=time.perf_counter()-begin,updates=0,backward=0,
            state_only_backwards=0,decoder_backwards=0,heavy_forwards=0,
            decoder_calls=0,encoder_calls=0,expert_calls=0,teacher_setup_rollouts=0,
            teacher_setup_seconds=0.,teacher_greedy_rollouts=1,total_sequence_rollouts=1))
