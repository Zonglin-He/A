"""CPU-only mechanism tests for hard-history OPD and its two controls."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from scripts.propagation_opd import policy


def bank_example():
    T = 5
    basis = torch.eye(256,dtype=torch.float64)
    features = torch.stack((basis[0],basis[1],basis[2],
                            (basis[0]+basis[1])/2**.5))[None].repeat(T,1,1)
    anchors = basis[0][None].repeat(T,1)
    anchors[3] = basis[1]
    E = torch.tensor([.4,.5,.2,.25],dtype=torch.float64)[None].repeat(T,1)
    boxes = E[:,None].repeat(1,4,1)
    boxes[:,1,0] = .6
    boxes[:,2,1] = .7
    boxes[:,3,0] = .45
    valid = torch.ones(T,4,dtype=torch.bool)
    valid[:,3] = False
    return SimpleNamespace(boxes=boxes,valid=valid,features=features,
        anchor_features=anchors,confidence=torch.tensor([1.,.7,1.,.4,.8],dtype=torch.float64),
        active=torch.tensor([True,True,True,False,True]),E=E,
        projection=torch.tensor([.2,.85,.6,.4],dtype=torch.float64)[None].repeat(T,1),
        guard=torch.tensor([0.,.1,.2,.1],dtype=torch.float64)[None].repeat(T,1),
        link=torch.eye(4,dtype=torch.float64)[None].repeat(T-1,1,1)*.7,
        observed=torch.tensor([False,False,True,True,False]))


CFG = dict(seed=41,rounds=3,paths=2,lr=.1,tau=.1,prior_other_logit=-2.,memory_rate=.5,
           Direct_relaxation_temperature=1.,initial_weights=list(policy.INITIAL_WEIGHTS))


def theta(value=None):
    return torch.nn.Parameter(torch.tensor(policy.INITIAL_WEIGHTS if value is None else value,
                                          dtype=torch.float64))


def test_hard_choice_changes_future_memory_and_student_distribution():
    bank = bank_example()
    noise_E = torch.zeros(5,4,dtype=torch.float64)
    noise_other = noise_E.clone()
    noise_other[0,1] = 30
    a = policy.rollout(bank,theta(),CFG,'student_sample',noise=noise_E)
    b = policy.rollout(bank,theta(),CFG,'student_sample',noise=noise_other)
    assert a['actions'][0] == 0 and b['actions'][0] == 1
    assert not torch.equal(a['memory_before'][1],b['memory_before'][1])
    assert not torch.equal(a['student_log_probs'][1],b['student_log_probs'][1])
    assert a['teacher_target_detached'] and not a['memory_after'].requires_grad


def test_observed_reanchor_and_inactive_no_write_override():
    bank = bank_example()
    noise = torch.zeros(5,4,dtype=torch.float64)
    noise[:2,1] = 30
    out = policy.rollout(bank,theta(),CFG,'student_sample',noise=noise)
    assert not torch.equal(out['memory_before'][2],bank.anchor_features[2])
    assert torch.equal(out['memory_after'][2],bank.anchor_features[2])
    # Node3 is observed but inactive: E and no write override reanchoring.
    assert out['actions'][3] == 0
    assert torch.equal(out['boxes'][3],bank.E[3])
    assert torch.equal(out['memory_after'][3],out['memory_before'][3])
    assert not torch.equal(out['memory_after'][3],bank.anchor_features[3])


@pytest.mark.parametrize('mode',policy.MODES)
def test_invalid_candidates_and_inactive_nodes_cannot_be_selected(mode):
    bank = bank_example()
    noise = torch.zeros(5,4,dtype=torch.float64)
    noise[:,3] = 1e3
    out = policy.rollout(bank,theta(),CFG,mode,noise=noise)
    assert not bool((out['actions']==3).any())
    assert torch.equal(out['action_probs'][:,3],torch.zeros(5,dtype=torch.float64))
    assert torch.equal(out['action_probs'][3],torch.tensor([1.,0.,0.,0.],dtype=torch.float64))
    assert out['KL_terms'][3] == 0
    assert torch.isfinite(out['loss'])
    out['loss'].backward()


def test_teacher_does_not_use_wrong_memory_cosine_but_uses_previous_link():
    bank = bank_example()
    no_link = bank_example()
    no_link.link.zero_()
    a = policy.rollout(bank,theta(),CFG,'student_greedy')
    b = policy.rollout(bank,theta(),CFG,'teacher_greedy')
    # Different histories leave different memories, but at the observed node
    # the teacher is determined by fixed evidence and previous action only.
    assert a['actions'][0] != b['actions'][0]
    without = policy.rollout(no_link,theta(),CFG,'teacher_greedy')
    assert not torch.equal(b['teacher_log_probs'][1],without['teacher_log_probs'][1])
    noise = torch.zeros(5,4,dtype=torch.float64)
    altered = noise.clone()
    altered[0,1] = 30
    history_a = policy.rollout(no_link,theta(),CFG,'student_sample',noise=noise)
    history_b = policy.rollout(no_link,theta(),CFG,'student_sample',noise=altered)
    assert not torch.equal(history_a['memory_before'][1],history_b['memory_before'][1])
    assert torch.equal(history_a['teacher_log_probs'][1],history_b['teacher_log_probs'][1])
    # Teacher target at t0 is invariant to a different initial memory weight.
    changed = policy.rollout(bank,theta([9.,0.,0.,0.,0.,0.]),CFG,'student_greedy')
    assert torch.equal(a['teacher_log_probs'][0],changed['teacher_log_probs'][0])


def test_direct_retains_memory_and_teacher_target_gradient():
    bank = bank_example()
    weights = theta([0.,0.,0.,0.,0.,0.])
    out = policy.rollout(bank,weights,CFG,'direct_soft',noise=torch.zeros(5,4,dtype=torch.float64))
    assert not out['teacher_target_detached']
    assert out['memory_after'].requires_grad and out['teacher_log_probs'].requires_grad
    memory_gradient = torch.autograd.grad(out['memory_after'][0,1],weights,retain_graph=True)[0]
    teacher_gradient = torch.autograd.grad(out['teacher_log_probs'][1,1],weights,retain_graph=True)[0]
    assert memory_gradient.norm() > 1e-8
    assert teacher_gradient.norm() > 1e-8
    assert bool((out['action_probs'][0,:3] > 0).all())
    assert bool((out['action_probs'][0,:3] < 1).all())
    out['loss'].backward()
    assert torch.isfinite(weights.grad).all()


def test_full_direct_relaxation_gradient_matches_finite_differences():
    bank = bank_example()
    weights = theta([.05,-.02,.01,.03,-.01,.02])
    noise = policy.gumbel_noise('full-gradient',0,0,5)
    assert torch.autograd.gradcheck(lambda w: policy.rollout(
        bank,w,CFG,'direct_soft',noise=noise)['loss'],(weights,),
        eps=1e-6,atol=2e-6,rtol=2e-5)


def test_fixed_state_KL_matches_replaying_same_teacher_history():
    bank = bank_example()
    noise = policy.gumbel_noise('fixed-objective',0,0,5)
    states = policy._detach(policy.rollout(bank,theta(),CFG,'teacher_sample',noise=noise))
    weights = theta([.8,.1,.2,-.1,.05,.03])
    fixed_loss,_ = policy._fixed_loss(states,weights,policy._settings(CFG))
    replay = policy.rollout(bank,weights,CFG,'teacher_sample',noise=noise)
    assert torch.equal(states['actions'],replay['actions'])
    torch.testing.assert_close(fixed_loss,replay['loss'],atol=2e-15,rtol=2e-15)
    a = torch.autograd.grad(fixed_loss,weights,retain_graph=True)[0]
    b = torch.autograd.grad(replay['loss'],weights)[0]
    torch.testing.assert_close(a,b,atol=2e-14,rtol=2e-14)


@pytest.mark.parametrize('arm',policy.ARMS)
def test_real_updates_deterministic_final_greedy_and_rollout_accounting(arm):
    bank = bank_example()
    result = policy.fit(bank,'synthetic-query',arm,CFG)
    repeated = policy.fit(bank,'synthetic-query',arm,CFG)
    assert torch.equal(result['finaltheta'],repeated['finaltheta'])
    assert torch.equal(result['final'],repeated['final'])
    assert not torch.equal(result['finaltheta'],result['initialtheta'])
    assert len(result['rounds']) == 3
    assert result['costs']['updates'] == result['costs']['backward'] == 3
    assert result['costs']['heavy_forwards'] == result['costs']['decoder_calls'] == 0
    final = policy.rollout(bank,result['finaltheta'],CFG,'student_greedy')
    assert torch.equal(result['final'],final['boxes'])
    for t,box in enumerate(result['final']):
        assert any(torch.equal(box,candidate) for candidate in bank.boxes[t])
    if arm == 'FixedKD':
        assert result['costs']['teacher_setup_rollouts'] == 2
        assert result['costs']['student_training_rollouts'] == 0
        assert result['costs']['cached_teacher_state_loss_passes'] == 6
        assert result['costs']['total_sequence_rollouts'] == 3
    else:
        assert result['costs']['teacher_setup_rollouts'] == 0
        assert result['costs']['student_training_rollouts'] == 6
        assert result['costs']['total_sequence_rollouts'] == 7


def test_fixedKD_teacher_paths_generated_once_and_not_resampled():
    bank = bank_example()
    actual_rollout = policy.rollout
    calls = []

    def recorded(*args,**kwargs):
        calls.append(args[3] if len(args)>3 else kwargs.get('mode','student_greedy'))
        return actual_rollout(*args,**kwargs)

    with patch.object(policy,'rollout',recorded):
        out = policy.fit(bank,'fixed-state-test','FixedKD',CFG)
    assert calls == ['teacher_sample','teacher_sample','student_greedy']
    for paths in out['trace']['training_paths']:
        assert len(paths) == 2
        for states,fixed in zip(paths,out['trace']['fixed_teacher_paths']):
            for key in ['features','teacher_log_probs','actions','memory_before','memory_after']:
                assert torch.equal(states[key],fixed[key])


def test_onpolicy_and_direct_use_same_underlying_noise_without_global_rng_mutation():
    bank = bank_example()
    before = torch.random.get_rng_state().clone()
    opd = policy.fit(bank,'noise-match','OPD',CFG)
    direct = policy.fit(bank,'noise-match','Direct',CFG)
    assert torch.equal(before,torch.random.get_rng_state())
    for opd_paths,direct_paths in zip(opd['trace']['training_paths'],direct['trace']['training_paths']):
        for a,b in zip(opd_paths,direct_paths):
            assert torch.equal(a['noise'],b['noise'])
    assert not torch.equal(policy.gumbel_noise('noise-match',0,0,5),
                           policy.gumbel_noise('noise-match',1,0,5))


@pytest.mark.parametrize('arm',policy.ARMS)
def test_all_inactive_keeps_E_and_state_but_executes_three_finite_updates(arm):
    bank = bank_example()
    bank.active.zero_()
    out = policy.fit(bank,'inactive',arm,CFG)
    assert torch.equal(out['final'],bank.E)
    assert torch.equal(out['finaltheta'],out['initialtheta'])
    for step in out['rounds']:
        assert step['loss'] == 0 and torch.equal(step['gradient'],torch.zeros(6,dtype=torch.float64))
        assert step['memory_drift_from_initial'] == 0
    assert out['costs']['updates'] == out['costs']['backward'] == 3


def test_fixed_propagation_is_greedy_teacher_with_no_weight_updates():
    bank = bank_example()
    result = policy.fixed_propagation(bank,CFG)
    expected = policy.rollout(bank,theta(),CFG,'teacher_greedy')
    assert torch.equal(result['final'],expected['boxes'])
    assert result['costs']['updates'] == result['costs']['backward'] == 0
    assert result['costs']['total_sequence_rollouts'] == 1
