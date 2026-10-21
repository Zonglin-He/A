"""CPU-only algorithm regressions against the original frozen policy.

The differentiable fake replay exercises actual optimizers and budget control;
it does not load assets, initialize CUDA, evaluate GT, or claim GPU equivalence.
"""
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from scripts.trajectory_opd import policy as original
from scripts.trajectory_fast import policy as fast


class FakeReplay:
    instances = []

    def __init__(self, model, cache, W, full_reference=False):
        self.W = W.detach().clone()
        self.E = torch.zeros(8, 4, dtype=torch.float64)
        self.evals = 0
        self.setup_evals = 2 if full_reference else 1
        self.initial_checks = dict(full_reference=full_reference)
        parameter = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.decoder = SimpleNamespace(parameters=lambda: iter([parameter]))
        self.executed_states = []
        self.__class__.instances.append(self)

    def expand(self, knots):
        return self.W @ knots

    def boxes(self, knots):
        self.evals += 1
        self.executed_states.append(knots.detach().clone())
        z = self.expand(knots)[:, :4].double()
        return .3 + .02*z + .003*z.square()

    def assert_frozen(self):
        assert all(not p.requires_grad and p.grad is None
                   for p in self.decoder.parameters())


class FakeEvidence:
    def __init__(self, flat=False):
        self.W = torch.eye(8)
        self.metadata = dict(synthetic=True, GT_read=False)
        self.target = torch.linspace(.32, .37, 32, dtype=torch.float64).reshape(8, 4)
        self.flat = flat
        self.scalar_calls = self.batch_calls = self.batch_evaluations = 0

    def _values(self, boxes):
        U = -((boxes-self.target).square()).sum(dim=(-2, -1))
        if self.flat:
            U = U*0
        zero = U*0
        return dict(U=U, mask=U, link=zero, guard=zero,
                    guard_mean=zero, guard_max=zero, obs=U, unobs=U,
                    d_mean=zero, d_max=zero)

    def utility(self, boxes, E):
        self.scalar_calls += 1
        return self._values(boxes)

    def utility_batch(self, boxes, E):
        self.batch_calls += 1
        self.batch_evaluations += len(boxes)
        return self._values(boxes)


def cpu_fit(module, arm, flat=False, budget=256):
    """Map only the driver's CUDA plumbing to CPU; keep all policy math real."""
    fake_parameter = SimpleNamespace(device=torch.device('cuda'))
    model = SimpleNamespace(ground_decoder=SimpleNamespace(decoder=SimpleNamespace(
        parameters=lambda: iter([fake_parameter]))))
    cfg = dict(sigma=.31, lr=.014, rounds=40, direct_steps=660,
               forward_budget=budget)
    evidence = FakeEvidence(flat=flat)
    tensor_to, ones = torch.Tensor.to, torch.ones

    def to_cpu(tensor, *args, **kwargs):
        args = list(args)
        if args and isinstance(args[0], torch.device) and args[0].type == 'cuda':
            args[0] = torch.device('cpu')
        if isinstance(kwargs.get('device'), torch.device) and kwargs['device'].type == 'cuda':
            kwargs = {**kwargs, 'device': torch.device('cpu')}
        return tensor_to(tensor, *args, **kwargs)

    def ones_cpu(*args, **kwargs):
        if isinstance(kwargs.get('device'), torch.device) and kwargs['device'].type == 'cuda':
            kwargs = {**kwargs, 'device': torch.device('cpu')}
        return ones(*args, **kwargs)

    with ExitStack() as stack:
        stack.enter_context(patch.object(module, 'FrozenTrajectoryReplay', FakeReplay))
        stack.enter_context(patch.object(torch.Tensor, 'to', to_cpu))
        stack.enter_context(patch.object(torch, 'ones', ones_cpu))
        for name in ['synchronize', 'reset_peak_memory_stats', 'empty_cache']:
            stack.enter_context(patch.object(torch.cuda, name, lambda: None))
        stack.enter_context(patch.object(torch.cuda, 'max_memory_allocated', lambda: 0))
        stack.enter_context(patch.object(torch.cuda, 'is_available', lambda: False))
        result = module.fit(model, {}, evidence, 'synthetic-frozen-policy', arm, cfg)
    return result, evidence, FakeReplay.instances[-1]


def assert_same(left, right):
    if torch.is_tensor(left):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        keys = set(left)-{'seconds', 'profiling'}
        assert keys == set(right)-{'seconds', 'profiling'}
        for key in keys:
            assert_same(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_same(a, b)
    else:
        assert left == right


@pytest.mark.parametrize('arm', fast.ARMS)
def test_complete_cpu_policy_matches_original_at_256_allowance(arm):
    old, old_evidence, old_replay = cpu_fit(original, arm)
    new, evidence, replay = cpu_fit(fast, arm)
    assert_same(old, new)
    assert len(old_replay.executed_states) == len(replay.executed_states)
    for a, b in zip(old_replay.executed_states, replay.executed_states):
        assert torch.equal(a, b)
    assert new['costs']['full_trajectory_evals'] <= 256
    assert new['costs']['full_trajectory_evals'] == replay.evals
    assert old_evidence.scalar_calls == replay.evals
    profiling = new['profiling']
    categories = profiling['categories']
    assert categories['decoder_forward']['calls'] == replay.evals
    assert categories['reward_forward']['calls'] == evidence.scalar_calls+evidence.batch_calls
    assert categories['reward_forward']['evaluations'] == replay.evals
    assert not profiling['CUDA_events_enabled']
    if arm == 'TrajectoryOPD':
        assert evidence.batch_calls == new['costs']['updates']
        assert evidence.batch_evaluations == 32*new['costs']['updates']
        assert 'decoder_reward_backward' not in categories
    else:
        assert evidence.batch_calls == 0
        assert categories['decoder_reward_backward']['calls'] == new['costs']['decoder_backwards']


@pytest.mark.parametrize('arm', fast.ARMS)
def test_flat_reward_rejections_and_adam_rollbacks_match_original(arm):
    old, _, _ = cpu_fit(original, arm, flat=True, budget=80)
    new, _, _ = cpu_fit(fast, arm, flat=True, budget=80)
    assert_same(old, new)
    assert new['costs']['accepted'] == 0
    assert torch.equal(new['final_states'], new['initial_states'])
    assert new['best_component'] == new['initial_best_component'] == 0


def test_unchanged_proposal_sampling_and_calibration_helpers():
    starts = original.initial_states('helper-regression', .31)
    assert torch.equal(starts, fast.initial_states('helper-regression', .31))
    for p in [torch.tensor([1/3]*3), torch.tensor([.98, .01, .01])]:
        counts = original.allocations(p)
        assert counts == fast.allocations(p)
        for a, b in zip(original.noise_for('helper-regression', 3, counts),
                        fast.noise_for('helper-regression', 3, counts)):
            assert torch.equal(a, b)
        U = [torch.linspace(-.2, .3, n, dtype=torch.float64) for n in counts]
        assert original.calibrated_temperature(U, counts) == fast.calibrated_temperature(U, counts)
        for mean, n, u in zip(starts, counts, U):
            actions = mean[None]+.31*torch.ones(n, 8, 256)
            assert_same(original.conditional_projection(mean, actions, u, .1),
                        fast.conditional_projection(mean, actions, u, .1))


def test_event_recorder_defers_all_synchronization_and_separates_counts():
    made = []

    class FakeEvent:
        def __init__(self, **kwargs):
            self.recorded = False
            made.append(self)

        def record(self):
            self.recorded = True

        def elapsed_time(self, other):
            assert self.recorded and other.recorded
            return 2.

    with patch.object(torch.cuda, 'is_available', lambda: True), \
         patch.object(torch.cuda, 'Event', FakeEvent), \
         patch.object(torch.cuda, 'synchronize', side_effect=AssertionError('unexpected synchronization')):
        recorder = fast.EventRecorder(torch.device('cuda'))
        with recorder.record('reward_forward', evaluations=32):
            pass
        recorder.end_arm()
        profile = recorder.summary_after_synchronize(.1)
    assert len(made) == 4
    assert profile['categories']['reward_forward']['calls'] == 1
    assert profile['categories']['reward_forward']['evaluations'] == 32
    assert profile['categories']['reward_forward']['CUDA_stream_seconds'] == .002
    assert profile['arm_CUDA_stream_seconds'] == .002
