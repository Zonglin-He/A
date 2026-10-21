"""CPU-only reward-factor engineering checks; no model, GT or scoring."""
import pytest
import torch

from scripts.output_opd.feedback import TrackedFeedback, reward_to_go
from scripts.trajectory_fast.evidence import TrajectoryEvidence
from scripts.trajectory_opd.test_evidence import fixture


@pytest.mark.parametrize('padding,interval', [(False, None), (True, None), (False, [100, 110])])
@pytest.mark.parametrize('dtype,tolerance', [(torch.float64, 1e-12), (torch.float32, 2e-6)])
def test_decomposition_and_pathwise_gradients_match_same_sparse_utility(padding, interval, dtype, tolerance):
    cache, pack, E, _ = fixture(padding=padding, interval=interval)
    E = E.to(dtype)
    evidence = TrajectoryEvidence(pack, cache)
    feedback = TrackedFeedback(evidence)
    boxes = E[None].repeat(4, 1, 1)
    boxes[..., :2] += torch.tensor([.037, .17, -.21, .027], dtype=dtype)[:, None, None]
    boxes[..., 2:] *= torch.tensor([1.11, .83, 1.65, .96], dtype=dtype)[:, None, None]
    boxes.requires_grad_()
    original = evidence.utility_batch(boxes, E)
    output = feedback(boxes, E)
    assert torch.equal(output['U'], original['U'])
    assert torch.allclose(output['Psi'].sum(-1), original['U'], atol=tolerance, rtol=tolerance)
    for field in ('Psi', 'unary', 'pair', 'guardfactors', 'reward_to_go', 'obs_unary', 'unobs_unary',
                  'guard_mean_factors', 'guard_max_factors', 'displacement', 'excess'):
        assert output[field].shape == (4, 10)
    assert torch.allclose(output['unary'].sum(-1), original['mask'], atol=tolerance, rtol=tolerance)
    assert torch.allclose(output['pair'].sum(-1), .1 * original['link'], atol=tolerance, rtol=tolerance)
    assert torch.allclose(output['guardfactors'].sum(-1), -.1 * original['guard'], atol=tolerance, rtol=tolerance)
    old = torch.autograd.grad(original['U'].sum(), boxes, retain_graph=True)[0]
    new = torch.autograd.grad(output['Psi'].sum(), boxes)[0]
    assert bool(torch.isfinite(new).all())
    assert torch.allclose(old, new, atol=tolerance, rtol=tolerance)


def test_reward_to_go_keeps_every_later_node_and_returns_terminal_factor():
    factors = torch.tensor([[1., 2., 4., 8.], [-1., 3., -2., 0.]], dtype=torch.float64)
    expected = torch.tensor([[15., 14., 12., 8.], [0., 1., -2., 0.]], dtype=torch.float64)
    result = reward_to_go(factors)
    assert torch.equal(result, expected)
    assert torch.equal(result[:, -1], factors[:, -1])
    assert torch.equal(result[:, 0], factors.sum(-1))
    assert torch.equal(result[:, :-1] - result[:, 1:], factors[:, :-1])


def test_empty_event_still_protects_all_nodes_and_global_guard_is_terminal():
    cache, pack, E, _ = fixture(interval=[100, 110])
    feedback = TrackedFeedback(TrajectoryEvidence(pack, cache))
    boxes = E[None].clone(); boxes[0, 0, 0] += .3
    output = feedback(boxes, E)
    assert torch.count_nonzero(output['unary']) == torch.count_nonzero(output['pair']) == 0
    assert output['guard_mean_factors'][0, 0] < 0
    assert output['guard_max_factors'][0, -1] < 0
    assert torch.count_nonzero(output['guard_max_factors'][:, :-1]) == 0
    assert output['Psi'][0, 0] < 0 and output['Psi'][0, -1] < 0
    assert output['reward_to_go'][0, 5] == output['guard_max_factors'][0, -1]
    assert abs(float(output['Psi'].sum() - output['U'][0])) < 1e-12


def test_unobserved_feedback_remains_positive_and_zero_confidence_is_not_renormalized():
    cache, pack, E, _ = fixture()
    evidence = TrajectoryEvidence(pack, cache)
    feedback = TrackedFeedback(evidence)
    result = feedback(E[None], E)
    unobs = ~evidence.observed
    assert bool((result['unobs_unary'][0, unobs] > 0).all())
    assert torch.count_nonzero(result['obs_unary'][0, unobs]) == 0
    # A separate fixed evidence case with uncertain nodes suppressed; no reward
    # scale is redistributed to the remaining nodes.
    evidence2 = TrajectoryEvidence(pack, cache)
    evidence2.confidence[unobs] = 0
    evidence2._link_confidence = evidence2.confidence[1:] * evidence2.confidence[:-1]
    evidence2._runtime_cache = {}
    feedback2 = TrackedFeedback(evidence2)
    result2 = feedback2(E[None], E)
    assert torch.count_nonzero(result2['unobs_unary']) == 0
    assert torch.equal(result2['obs_unary'], result['obs_unary'])
    assert abs(float(result2['Psi'].sum() - result2['U'][0])) < 1e-12


def test_max_guard_ties_retain_original_split_gradient():
    cache, pack, E, _ = fixture()
    evidence = TrajectoryEvidence(pack, cache)
    feedback = TrackedFeedback(evidence)
    boxes = E[None].clone(); boxes[..., 2:] *= 2
    boxes.requires_grad_()
    result = feedback(boxes, E)
    original = evidence.utility_batch(boxes, E)
    old = torch.autograd.grad(-.1 * original['guard'].sum(), boxes, retain_graph=True)[0]
    new = torch.autograd.grad(result['guardfactors'].sum(), boxes)[0]
    assert torch.allclose(old, new, atol=1e-12, rtol=1e-12)
    assert torch.count_nonzero(new[0, :, 2]) == len(E)


def test_evidence_called_once_and_opd_detach_is_callers_choice():
    cache, pack, E, _ = fixture()
    evidence = TrajectoryEvidence(pack, cache)
    original = evidence.utility_batch
    calls = []
    def counted(*args):
        calls.append(1)
        return original(*args)
    evidence.utility_batch = counted
    feedback = TrackedFeedback(evidence)
    boxes = E[None].repeat(4, 1, 1).requires_grad_()
    result = feedback(boxes, E)
    assert len(calls) == 1 and result['Psi'].requires_grad
    cache_id = id(next(iter(feedback._cache.values())))
    with torch.no_grad():
        detached = feedback(boxes, E)
    assert len(calls) == 2 and not detached['Psi'].requires_grad
    assert id(next(iter(feedback._cache.values()))) == cache_id
    assert feedback.evidence is evidence and feedback.metadata['propagation_builds'] == 0
    assert not torch.cuda.is_initialized()
