"""CPU engineering equivalence tests, with no model, RGB, GT or scoring."""
import pytest
import torch

from scripts.trajectory_fast.evidence import TrajectoryEvidence
from scripts.trajectory_opd.evidence import TrajectoryEvidence as DenseTrajectoryEvidence
from scripts.trajectory_opd.test_evidence import fixture


@pytest.mark.parametrize('padding,ambiguous', [(False, False), (True, False), (False, True)])
def test_sparse_preserves_support_and_both_transport_directions(padding, ambiguous):
    cache, pack, _, _ = fixture(padding=padding, ambiguous=ambiguous)
    dense, fast = DenseTrajectoryEvidence(pack, cache), TrajectoryEvidence(pack, cache)
    for name in ('targets', 'confidence', 'W', 'seeds', 'cycle_scores', 'agreement'):
        assert torch.equal(getattr(dense, name), getattr(fast, name))
    for reverse in (False, True):
        prefix = 'adjacent_reverse' if reverse else 'adjacent_forward'
        matrix = getattr(dense, prefix)
        indices, weights = getattr(fast, prefix + '_indices'), getattr(fast, prefix + '_weights')
        reconstructed = torch.zeros_like(matrix).scatter_add(-1, indices, weights)
        assert torch.equal(reconstructed, matrix)
        values = torch.rand(3, *matrix.shape[:2], dtype=torch.float64)
        expected = torch.einsum('eij,bej->bei', matrix, values)
        assert torch.allclose(fast.transport_adjacent(values, reverse=reverse), expected, atol=3e-16, rtol=1e-14)
    assert not hasattr(fast, 'adjacent_forward') and not hasattr(fast, 'adjacent_reverse')
    private = fast.private_state()
    assert 'adjacent_forward' not in private and 'adjacent_reverse' not in private
    assert private['schema'] == 'trajectory_sparse_evidence_v1'
    assert not fast.metadata['engineering']['persistent_dense_affinities']


@pytest.mark.parametrize('dtype,atol', [(torch.float64, 1e-12), (torch.float32, 2e-5)])
@pytest.mark.parametrize('padding,interval', [(False, None), (True, None), (False, [100, 110])])
def test_all_reward_fields_and_box_gradients_match_dense_and_batch(dtype, atol, padding, interval):
    cache, pack, E, _ = fixture(padding=padding, interval=interval)
    dense, fast = DenseTrajectoryEvidence(pack, cache), TrajectoryEvidence(pack, cache)
    E = E.to(dtype)
    # Avoid grid knots and include active whole-trajectory guards.
    boxes = E[None].repeat(4, 1, 1)
    boxes[:, :, :2] += torch.tensor([.013, .026, -.037, .16], dtype=dtype)[:, None, None]
    boxes[:, :, 2:] *= torch.tensor([1.11, .87, 1.23, 1.56], dtype=dtype)[:, None, None]
    boxes.requires_grad_()
    expected = [dense.utility(item, E) for item in boxes]
    batched = fast.utility_batch(boxes, E)
    scalar = [fast.utility(item, E) for item in boxes]
    assert expected[0].keys() == batched.keys()
    for name in batched:
        target = torch.stack([item[name] for item in expected])
        each = torch.stack([item[name] for item in scalar])
        assert torch.allclose(batched[name], target, atol=atol, rtol=atol), name
        assert torch.allclose(batched[name], each, atol=atol, rtol=atol), name
    old_grad = torch.autograd.grad(sum(item['U'] for item in expected), boxes, retain_graph=True)[0]
    new_grad = torch.autograd.grad(batched['U'].sum(), boxes, retain_graph=True)[0]
    scalar_grad = torch.autograd.grad(sum(item['U'] for item in scalar), boxes)[0]
    assert torch.allclose(old_grad, new_grad, atol=atol, rtol=atol)
    assert torch.allclose(scalar_grad, new_grad, atol=atol, rtol=atol)
    assert bool(torch.isfinite(new_grad).all())


def test_fixed_reference_is_cached_and_cannot_change_or_mutate():
    cache, pack, E, _ = fixture()
    fast = TrajectoryEvidence(pack, cache)
    result = fast.utility(E, E)
    reference = next(iter(fast._reference_runtime.values()))
    support = next(iter(fast._runtime_cache.values()))
    assert torch.equal(fast.utility(E.clone(), E.clone())['U'], result['U'])
    assert next(iter(fast._reference_runtime.values())) is reference
    assert next(iter(fast._runtime_cache.values())) is support
    E[0, 0] += .01
    with pytest.raises(ValueError, match='different fixed E'):
        fast.utility(E, E)


def test_batch_has_one_occupancy_computation_and_all_nodes_in_guard():
    cache, pack, E, _ = fixture(interval=[9, 72])
    fast = TrajectoryEvidence(pack, cache)
    original = fast._occupancy
    calls = []
    def counted(*args):
        calls.append(1)
        return original(*args)
    fast._occupancy = counted
    boxes = E[None].repeat(32, 1, 1)
    boxes[:, 9, 0] += .3  # Outside Strong, still protected by guard.
    result = fast.utility_batch(boxes, E)
    assert len(calls) == 1 and result['U'].shape == (32,)
    assert bool((result['guard'] > 0).all())
    assert result['projection_scores'].shape == (32, 10)
    assert result['link_scores'].shape == (32, 9)
    assert not torch.cuda.is_initialized()


def test_finite_difference_of_batched_reward_away_from_grid_knots():
    cache, pack, E, _ = fixture()
    fast = TrajectoryEvidence(pack, cache)
    boxes = E[None].repeat(2, 1, 1)
    boxes[..., 0] += .037; boxes[..., 2] *= 1.11
    boxes.requires_grad_()
    gradient = torch.autograd.grad(fast.utility_batch(boxes, E)['U'].sum(), boxes)[0]
    plus, minus = boxes.detach().clone(), boxes.detach().clone()
    plus[1, 6, 0] += 1e-6; minus[1, 6, 0] -= 1e-6
    numeric = (fast.utility_batch(plus, E)['U'][1] - fast.utility_batch(minus, E)['U'][1]) / 2e-6
    assert abs(float(numeric - gradient[1, 6, 0])) < 1e-6
