"""CPU mocks of policy control flow plus exact Gaussian gradient checks.

No model, decoder, RGB, GT or GPU is used.  Mock trajectory calls are counted
explicitly; the Gaussian/Adam implementations themselves remain real CPU code.
"""
from contextlib import contextmanager
from copy import deepcopy
from itertools import product
from types import SimpleNamespace
import math

import numpy as np
import pytest
import torch

from scripts.output_opd import policy
from scripts.output_opd.distribution import JointTrajectoryGaussian, independent_pair_baseline
from scripts.output_opd.feedback import reward_to_go


BRIEF_FIELDS = ('U', 'mask', 'link', 'guard', 'guard_mean', 'guard_max', 'obs', 'unobs', 'd_mean', 'd_max')


def _same(a, b):
    if torch.is_tensor(a):
        return torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


@pytest.fixture
def cpu_policy(monkeypatch):
    """Keep policy.fit's CUDA-only entry guard while redirecting all work to CPU."""
    frames = 10
    E = torch.tensor([.5, .5, .2, .3], dtype=torch.float64).repeat(frames, 1)
    world = SimpleNamespace(replays=[], opts=[], feedback_batches=[], backward_calls=0,
                            E=E, flat=False, reject_on_actual_call=None, invalid_on_actual_call=None)
    zeros = torch.zeros
    backward = torch.Tensor.backward
    Adam = torch.optim.Adam

    def cpu_zeros(*args, **kwargs):
        if str(kwargs.get('device', '')).startswith('cuda'):
            kwargs['device'] = 'cpu'
        return zeros(*args, **kwargs)

    def counted_backward(self, *args, **kwargs):
        world.backward_calls += 1
        return backward(self, *args, **kwargs)

    class TrackedAdam(Adam):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.after_steps = []
            self.restored = []
            world.opts.append(self)

        def step(self, *args, **kwargs):
            result = super().step(*args, **kwargs)
            self.after_steps.append(deepcopy(self.state_dict()))
            return result

        def load_state_dict(self, state):
            result = super().load_state_dict(state)
            self.restored.append(deepcopy(self.state_dict()))
            return result

    class Replay:
        def __init__(self, model, cache, W, full_reference):
            assert W.shape == (frames, 8)
            self.E = world.E.clone()
            self.evals = self.setup_evals = 0
            self.full_reference = full_reference
            self.calls = []
            self.initial_checks = dict(mock_CPU=True, full_reference=full_reference)
            self.frozen_checked = False
            world.replays.append(self)

        def boxes(self, z):
            self.evals += 1
            self.calls.append(dict(state=z.detach().clone(), grad_enabled=torch.is_grad_enabled()))
            direction = self.E.new_tensor([1., 0., 0., 0.])
            return self.E + .1 * z.double().mean() * direction

        def expand(self, z):
            return z.mean(0)[None].expand(frames, -1)

        def assert_frozen(self):
            self.frozen_checked = True

    class Evidence:
        def __init__(self):
            self.W = torch.ones(frames, 8, dtype=torch.float64) / 8
            self.metadata = dict(mock_CPU=True, GT_read=False)
            self.actual_calls = 0

        def utility(self, boxes, E):
            self.actual_calls += 1
            if self.actual_calls == world.invalid_on_actual_call:
                raise FloatingPointError('Mock invalid proposal')
            U = boxes[:, 0].mean()
            if world.flat:
                U = U * 0
            if self.actual_calls == world.reject_on_actual_call:
                U = U - 100
            zero = U * 0
            return {name: U if name in ('U', 'mask', 'obs') else zero for name in BRIEF_FIELDS}

    class Feedback:
        def __init__(self, evidence):
            self.evidence = evidence

        def evaluate(self, boxes, E):
            world.feedback_batches.append(dict(shape=tuple(boxes.shape), replay_calls=world.replays[-1].evals,
                                               grad_enabled=torch.is_grad_enabled()))
            factors = boxes[..., 0] / frames
            if world.flat:
                factors = factors * 0
            return dict(U=factors.sum(-1), Psi=factors, reward_to_go=reward_to_go(factors))

    class Recorder:
        def __init__(self, device):
            self.categories = {}

        @contextmanager
        def record(self, category, evaluations=1):
            row = self.categories.setdefault(category, dict(calls=0, evaluations=0))
            row['calls'] += 1
            row['evaluations'] += evaluations
            yield

        def end_arm(self):
            pass

        def summary_after_synchronize(self, seconds):
            return dict(categories=self.categories, CPU_mock=True)

    def fixed_noise(key, round_index, n_frames, *, device=None, dtype=torch.float64):
        # Deterministic antithetic output noise for control-flow assertions.
        # Distribution mathematics is tested separately with Gaussian quadrature.
        eps = zeros(4, n_frames, 4, dtype=dtype)
        eps[:, -1, 0] = torch.tensor([.5, 1.5, -.5, -1.5], dtype=dtype)
        return eps

    monkeypatch.setattr(policy, 'FrozenTrajectoryReplay', Replay)
    monkeypatch.setattr(policy, 'TrackedFeedback', Feedback)
    monkeypatch.setattr(policy, 'EventRecorder', Recorder)
    monkeypatch.setattr(policy, 'antithetic_noise', fixed_noise)
    monkeypatch.setattr(torch, 'zeros', cpu_zeros)
    monkeypatch.setattr(torch.Tensor, 'backward', counted_backward)
    monkeypatch.setattr(torch.optim, 'Adam', TrackedAdam)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', lambda: None)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda: 0)
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: None)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    world.model = SimpleNamespace(ground_decoder=SimpleNamespace(decoder=SimpleNamespace(
        parameters=lambda: iter([SimpleNamespace(device=torch.device('cuda'))]))))
    world.cache = dict(frame_ids=[0, 3, 8, 14, 21, 29, 38, 48, 59, 71],
                       planned_positions=[0, 1, 2, 3, 4, 5, 8, 9])
    world.cfg = dict(rounds=3, candidates=4, sigma=.1, tau=.01, lr=.014,
                     accept_epsilon=1e-8, forward_budget=6, backward_budget=3)
    world.Evidence = Evidence
    return world


@pytest.mark.parametrize('arm', policy.ARMS)
def test_fit_three_round_budget_and_four_candidates_share_each_student_forward(cpu_policy, arm):
    world = cpu_policy
    result = policy.fit(world.model, world.cache, world.Evidence(), 'CPU-budget', arm, world.cfg)
    replay = world.replays[-1]
    assert result['costs']['full_trajectory_evals'] == replay.evals == 6
    assert result['costs']['decoder_backwards'] == world.backward_calls == 3
    assert result['costs']['setup_full_trajectory_evals'] == replay.setup_evals == 0
    assert result['costs']['native_offset_decoder_calls'] == 12
    assert result['costs']['updates'] == 3
    assert result['costs']['accepted'] + result['costs']['rejected'] == 3
    assert replay.full_reference and replay.frozen_checked
    assert all(call['state'].device.type == 'cpu' for call in replay.calls)
    assert [call['grad_enabled'] for call in replay.calls] == [True, False] * 3
    batches = world.feedback_batches
    if arm == 'Direct':
        assert batches == [] and result['costs']['output_candidates'] == 0
    else:
        assert len(batches) == 3 and all(row['shape'] == (4, 10, 4) for row in batches)
        assert [row['replay_calls'] for row in batches] == [1, 3, 5]
        assert all(row['grad_enabled'] == (arm == 'DirectMC') for row in batches)
        assert result['costs']['output_candidates'] == 12
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('arm', policy.ARMS)
def test_first_rejection_restores_zero_state_and_empty_adam_and_stops(cpu_policy, arm):
    world = cpu_policy
    world.flat = True
    result = policy.fit(world.model, world.cache, world.Evidence(), 'CPU-flat', arm, world.cfg)
    opt = world.opts[-1]
    assert result['costs']['full_trajectory_evals'] == 2
    assert result['costs']['decoder_backwards'] == world.backward_calls == 1
    assert result['costs']['accepted'] == 0 and result['costs']['rejected'] == 1
    assert len(result['rounds']) == 1 and len(opt.after_steps) == len(opt.restored) == 1
    assert opt.after_steps[0]['state'] and opt.state_dict()['state'] == opt.restored[0]['state'] == {}
    assert torch.count_nonzero(result['final_z']) == 0
    assert torch.equal(result['final'], world.E)
    assert result['costs']['stop_reason'] == 'rejected_and_restored'


@pytest.mark.parametrize('arm', policy.ARMS)
@pytest.mark.parametrize('invalid', [False, True])
def test_later_rejection_restores_nonempty_adam_and_last_accepted_state(cpu_policy, arm, invalid):
    world = cpu_policy
    if invalid:
        world.invalid_on_actual_call = 4
    else:
        world.reject_on_actual_call = 4
    result = policy.fit(world.model, world.cache, world.Evidence(), 'CPU-reject-second', arm, world.cfg)
    opt = world.opts[-1]
    assert result['costs']['full_trajectory_evals'] == 4
    assert result['costs']['decoder_backwards'] == world.backward_calls == 2
    assert [row['accepted'] for row in result['rounds']] == [True, False]
    assert len(opt.after_steps) == 2 and len(opt.restored) == 1
    assert _same(opt.after_steps[0], opt.restored[0])
    assert _same(opt.after_steps[0], opt.state_dict())
    assert torch.equal(result['final_z'], result['path'][0]['state'])
    assert torch.equal(result['final'], result['path'][0]['boxes'])
    assert result['rounds'][-1]['proposal_numerically_valid'] == (not invalid)


def test_all_three_arms_use_same_full_reference_and_initial_state_and_matched_output_noise(cpu_policy):
    world = cpu_policy
    outputs = {arm: policy.fit(world.model, world.cache, world.Evidence(), 'same-key', arm, world.cfg)
               for arm in policy.ARMS}
    assert all(replay.full_reference for replay in world.replays)
    assert all(torch.equal(replay.calls[0]['state'], world.replays[0].calls[0]['state']) for replay in world.replays)
    assert all(torch.count_nonzero(replay.calls[0]['state']) == 0 for replay in world.replays)
    assert len({out['rounds'][0]['before']['U'] for out in outputs.values()}) == 1
    for field in ('noise', 'candidate_X', 'candidate_boxes'):
        assert torch.equal(outputs['OutputOPD']['rounds'][0][field], outputs['DirectMC']['rounds'][0][field])
    assert all(out['initial_checks']['full_reference'] for out in outputs.values())


def _law_and_mean():
    E = torch.tensor([.4, .5, .2, .3], dtype=torch.float64).repeat(8, 1)
    law = JointTrajectoryGaussian(E, list(range(8)), list(range(8)), sigma=.3)
    mu = torch.zeros(8, 4, dtype=torch.float64)
    mu[0, 0], mu[1, 0] = .23, -.17
    return law, mu.requires_grad_()


def _quadrature(dimensions, order=3):
    nodes, weights = np.polynomial.hermite.hermgauss(order)
    tuples = list(product(range(order), repeat=dimensions))
    points = torch.tensor([[math.sqrt(2) * nodes[i] for i in row] for row in tuples], dtype=torch.float64)
    probability = torch.tensor([math.prod(weights[i] / math.sqrt(math.pi) for i in row) for row in tuples], dtype=torch.float64)
    return points, probability


def _factors(X):
    a, b = X[..., 0, 0], X[..., 1, 0]
    zero = a * 0
    # A past unary, incoming pair, and all-history factor assigned to terminal.
    return torch.stack((.7 * a, -.3 * b + .2 * a * b + .1 * b.square(),
                        zero, zero, zero, zero, zero, .4 * a.square()), -1)


def test_conditional_score_with_reward_to_go_matches_full_score_and_pathwise_gradient_exactly():
    law, mu = _law_and_mean()
    points, probability = _quadrature(2)
    eps = torch.zeros(len(points), 8, 4, dtype=torch.float64)
    eps[:, :2, 0] = points
    samples = law.sample(mu, eps, detach_mean=True)
    factors = _factors(samples)
    logp = law.log_prob_terms(samples.detach(), mu)
    full = (probability * (factors.sum(-1).detach()[:, None] * logp).sum(-1)).sum()
    causal = (probability * (reward_to_go(factors).detach() * logp).sum(-1)).sum()
    full_grad = torch.autograd.grad(full, mu, retain_graph=True)[0]
    causal_grad = torch.autograd.grad(causal, mu, retain_graph=True)[0]
    pathwise = (probability * _factors(law.sample(mu, eps)).sum(-1)).sum()
    direct_grad = torch.autograd.grad(pathwise, mu)[0]
    expected = mu.detach().new_tensor([.7 + .2 * mu[1, 0] + .8 * mu[0, 0],
                                     -.3 + .2 * mu[0, 0] + .2 * mu[1, 0]])
    assert torch.allclose(causal_grad[:2, 0], expected, atol=1e-12, rtol=1e-12)
    assert torch.allclose(full_grad[:2, 0], causal_grad[:2, 0], atol=1e-12, rtol=1e-12)
    assert torch.allclose(direct_grad[:2, 0], causal_grad[:2, 0], atol=1e-12, rtol=1e-12)


def test_independent_pair_baseline_preserves_causal_score_gradient_and_detaches():
    law, mu = _law_and_mean()
    points, probability = _quadrature(4)
    eps = torch.zeros(len(points), 4, 8, 4, dtype=torch.float64)
    eps[:, 0, :2, 0] = points[:, :2]
    eps[:, 1, :2, 0] = points[:, 2:]
    eps[:, 2] = -eps[:, 0]; eps[:, 3] = -eps[:, 1]
    samples = law.sample(mu, eps.flatten(0, 1), detach_mean=True).reshape(len(points), 4, 8, 4)
    factors = _factors(samples)
    returns = reward_to_go(factors).transpose(0, 1)
    baseline = independent_pair_baseline(returns)
    assert not baseline.requires_grad
    assert torch.equal(baseline[0], (returns[1] + returns[3]) / 2)
    assert torch.equal(baseline[2], baseline[0]) and torch.equal(baseline[3], baseline[1])
    logp = law.log_prob_terms(samples.detach().flatten(0, 1), mu).reshape(len(points), 4, 8)
    ordinary = (probability * (returns.transpose(0, 1).detach() * logp).sum(-1).mean(-1)).sum()
    centered = (probability * ((returns - baseline).transpose(0, 1).detach() * logp).sum(-1).mean(-1)).sum()
    old = torch.autograd.grad(ordinary, mu, retain_graph=True)[0]
    new = torch.autograd.grad(centered, mu)[0]
    assert torch.allclose(old[:2, 0], new[:2, 0], atol=1e-12, rtol=1e-12)


def test_including_dependent_antithetic_partner_in_loo_baseline_can_bias_gradient():
    law, mu = _law_and_mean()
    points, probability = _quadrature(2)
    eps = torch.zeros(len(points), 4, 8, 4, dtype=torch.float64)
    eps[:, 0, 0, 0], eps[:, 1, 0, 0] = points[:, 0], points[:, 1]
    eps[:, 2] = -eps[:, 0]; eps[:, 3] = -eps[:, 1]
    samples = law.sample(mu, eps.flatten(0, 1), detach_mean=True).reshape(len(points), 4, 8, 4)
    returns = samples[:, :, 0, 0].transpose(0, 1)
    good = independent_pair_baseline(returns)
    bad = (returns.sum(0, keepdim=True) - returns) / 3
    logp = law.log_prob_terms(samples.detach().flatten(0, 1), mu).reshape(len(points), 4, 8)[:, :, 0]
    correct = (probability * ((returns - good).T.detach() * logp).mean(-1)).sum()
    biased = (probability * ((returns - bad).T.detach() * logp).mean(-1)).sum()
    good_gradient = torch.autograd.grad(correct, mu, retain_graph=True)[0][0, 0]
    bad_gradient = torch.autograd.grad(biased, mu)[0][0, 0]
    assert abs(float(good_gradient) - 1) < 1e-12
    assert abs(float(bad_gradient) - 4 / 3) < 1e-12
