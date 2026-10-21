"""Synthetic CPU integration of native-box banks with the immutable v0 policy."""
import hashlib
import importlib
from pathlib import Path

import pytest
import torch

from scripts.box_propagation import policy
from scripts.propagation_v0 import policy as original
from scripts.propagation_v0.test_bank import fixture


CFG = dict(seed=41,rounds=3,paths=2,lr=.1,gradient_clip_norm=1.,teacher_tau=.2,
    prior_weight=.05,original_prior=.7,memory_write_max=.2,
    feedback_weights=dict(support=1.,link=.25,memory=.1,guard=.1),
    initial_weights=[0.]*6,Direct_relaxation_temperature=1.)


def bank(kind='native'):
    module = importlib.import_module('scripts.box_propagation.bank' if kind=='native'
                                    else 'scripts.propagation_v0.bank')
    evidence,cache = fixture()
    return module.CandidateBank(evidence,cache,{}),cache


def test_same_policy_objects_and_original_source_are_frozen():
    assert policy.IMPLEMENTATION_MODULE=='scripts.propagation_v0.policy'
    assert hashlib.sha256(Path(original.__file__).read_bytes()).hexdigest()==policy.FROZEN_IMPLEMENTATION_SHA256
    for name in ('ARMS','FEATURE_NAMES','MODES','fit','rollout','fixed_propagation',
                 'initial_diagnostics','evidence_dp','gumbel_noise'):
        assert getattr(policy,name) is getattr(original,name)
    settings = original._settings(CFG)
    assert settings['rounds']==3 and settings['paths']==2 and settings['lr']==.1
    assert settings['grad_clip']==1 and settings['prior_weight']==.05 and settings['teacher_tau']==.2


@pytest.mark.parametrize('kind',['native','component'])
@pytest.mark.parametrize('arm',policy.ARMS)
def test_both_banks_complete_same_budget_fit_and_final_accepted_greedy(kind,arm):
    candidates,_ = bank(kind)
    initial = policy.rollout(candidates,torch.zeros(6,dtype=torch.float64),CFG)
    assert torch.equal(initial['boxes'],candidates.E)
    out = policy.fit(candidates,'same-synthetic-query',arm,CFG)
    assert out['final'].shape==candidates.E.shape and out['final'].dtype==torch.float64
    assert 1<=len(out['rounds'])<=3
    assert out['costs']['updates']==out['costs']['backward']==len(out['rounds'])
    assert out['costs']['heavy_forwards']==out['costs']['decoder_calls']==0
    assert out['costs']['accepted_updates']+out['costs']['rejected_updates']==len(out['rounds'])
    assert len(out['trace']['all_rollouts'])==out['costs']['total_sequence_rollouts']
    assert out['costs']['postupdate_student_greedy_rollouts']==len(out['rounds'])
    assert out['costs']['fixed_score_evaluations']==1+len(out['rounds'])
    assert torch.isfinite(out['finalphi']).all() and torch.isfinite(out['final']).all()
    assert torch.equal(out['final'],policy.rollout(candidates,out['finalphi'],CFG)['boxes'])
    for step in out['rounds']:
        assert torch.isfinite(step['gradient']).all() and step['gradient'].norm()<=1+1e-12
        if step['accepted']:
            assert step['after_score']>=step['before_score']-1e-12
        else:
            assert step['rolled_back'] and step['optimizer_before_sha256']==step['optimizer_after_sha256']
    if arm=='FixedKD':
        assert out['costs']['teacher_setup_rollouts']==2
        assert out['costs']['cached_teacher_state_loss_passes']==2*len(out['rounds'])
    else:
        assert out['costs']['teacher_setup_rollouts']==0
        assert out['costs']['student_training_rollouts']==2*len(out['rounds'])
    assert torch.equal(out['final'][~candidates.active],candidates.E[~candidates.active])


def test_native_bank_keeps_native_observed_extents_and_fits_without_reboxing():
    candidates,cache = bank()
    for j,node in enumerate(cache['planned_positions']):
        native = cache['teacher']['boxes'][j]
        assert any(torch.equal(candidates.boxes[node,k],native)
                   for k in (1,2) if bool(candidates.valid[node,k]))
    out = policy.fit(candidates,'native-extent','OPD',CFG)
    for node,box in enumerate(out['final']):
        assert any(torch.equal(box,candidates.boxes[node,k])
                   for k in range(3) if bool(candidates.valid[node,k]))


def test_direct_has_full_native_bank_memory_and_teacher_gradient():
    candidates,_ = bank()
    phi = torch.nn.Parameter(torch.tensor([.03,-.02,.01,.02,.01,-.01],dtype=torch.float64))
    noise = policy.gumbel_noise('native-full-gradient',0,0,len(candidates.E))
    out = policy.rollout(candidates,phi,CFG,'direct_soft',noise=noise)
    assert out['memory_after'].requires_grad and out['teacher_log_probs'].requires_grad
    assert not out['teacher_target_detached']
    memory_gradient = torch.autograd.grad(out['memory_after'].square().sum(0)[0],phi,retain_graph=True)[0]
    # Choose a later active valid teacher log probability, preserving its full
    # dependence on the earlier soft history rather than detaching the target.
    target_gradients = []
    for node in range(1,len(candidates.E)):
        available = torch.nonzero(candidates.valid[node,1:]).flatten()
        if bool(candidates.active[node]) and len(available):
            action = int(available[0])+1
            target_gradients.append(torch.autograd.grad(out['teacher_log_probs'][node,action],
                phi,retain_graph=True)[0].norm())
    assert memory_gradient.norm()>1e-12 and target_gradients and max(target_gradients)>1e-12
    assert torch.autograd.gradcheck(lambda p:policy.rollout(candidates,p,CFG,'direct_soft',noise=noise)['loss'],
                                   (phi,),eps=1e-6,atol=2e-6,rtol=2e-5)


def test_same_initial_noise_and_zero_initialization_for_both_bank_types():
    native,_ = bank('native')
    component,_ = bank('component')
    a = policy.initial_diagnostics(native,'matched-input',CFG)
    b = policy.initial_diagnostics(component,'matched-input',CFG)
    assert torch.equal(a['initial_final'],b['initial_final'])
    for index,(new,old) in enumerate(zip(a['diagnostic_paths'],b['diagnostic_paths'])):
        assert torch.equal(new['noise'],old['noise'])
        assert torch.equal(new['noise'],original.gumbel_noise('matched-input',0,index,len(native.E)))
    # Evidence and geometry may differ intentionally; identical noise does not
    # imply that learned paths or outcomes must match across different banks.


@pytest.mark.parametrize('arm',policy.ARMS)
def test_retained_native_outside_event_stays_E_in_every_actual_policy_path(arm):
    module = importlib.import_module('scripts.box_propagation.bank')
    evidence,cache = fixture()
    evidence.event[9:] = False
    cache['interval'] = [0,9]
    candidates = module.CandidateBank(evidence,cache,{})
    assert candidates.observed[10] and candidates.valid[10,1] and not candidates.active[10]
    out = policy.fit(candidates,'inactive-native',arm,CFG)
    assert torch.equal(out['final'][~candidates.active],candidates.E[~candidates.active])
    for path in out['trace']['all_rollouts']:
        assert not bool(path['actions'][~candidates.active].any())
        assert torch.equal(path['memory_before'][~candidates.active],path['memory_after'][~candidates.active])
    dp = policy.evidence_dp(candidates)
    assert not bool(dp['trace']['actions'][~candidates.active].any())
