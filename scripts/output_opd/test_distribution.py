"""Pure CPU distribution and gradient checks; no assets, model, rewards, or GT."""
import math

import pytest
import torch

from scripts.output_opd.distribution import (
    JointTrajectoryGaussian, antithetic_noise, independent_pair_baseline)


def example():
    frames = torch.tensor([0, 1, 3, 4, 6, 9, 13, 14, 18, 25])
    anchors = frames[torch.tensor([0, 1, 2, 3, 4, 6, 8, 9])]
    ref = torch.tensor([[.4, .5, .2, .3]], dtype=torch.float64).repeat(len(frames), 1)
    ref[:, 0] += torch.arange(len(frames), dtype=torch.float64)*.01
    return JointTrajectoryGaussian(ref, frames, anchors, sigma=.1)


def dense_covariance(distribution):
    frame_gaps = (distribution.frame_ids[:, None]-distribution.frame_ids[None]).abs()
    temporal = distribution.sigma**2*torch.exp(-frame_gaps/distribution.anchor_median_gap)
    return torch.kron(temporal, torch.eye(4, dtype=torch.float64))


def test_irregular_physical_time_and_full_rank_normalized_density():
    d = example()
    assert d.anchor_median_gap == 2
    torch.testing.assert_close(d.rho, torch.exp(-(d.frame_ids[1:]-d.frame_ids[:-1])/2),
                               atol=0, rtol=0)
    covariance = dense_covariance(d)
    assert torch.linalg.eigvalsh(covariance).min() > 0
    # Construct the explicit AR innovation operator independently. Its inverse
    # covariance must match the normalized dense multivariate Gaussian.
    A = torch.eye(d.frames, dtype=torch.float64)
    A[torch.arange(1, d.frames), torch.arange(d.frames-1)] = -d.rho
    L = torch.linalg.solve(A, torch.diag(d.innovation_std))
    torch.testing.assert_close(torch.kron(L@L.T, torch.eye(4, dtype=torch.float64)),
                               covariance, atol=2e-17, rtol=2e-15)
    generator = torch.Generator().manual_seed(8)
    mu = .01*torch.randn(d.frames, 4, generator=generator, dtype=torch.float64)
    eps = torch.randn(5, d.frames, 4, generator=generator, dtype=torch.float64)
    X = d.sample(mu, eps)
    expected = torch.distributions.MultivariateNormal(mu.flatten(), covariance_matrix=covariance)
    terms = d.log_prob_terms(X, mu)
    assert terms.shape == (5, d.frames)
    torch.testing.assert_close(terms.sum(-1), expected.log_prob(X.flatten(1)),
                               atol=5e-13, rtol=2e-14)


def test_exact_reference_KL_and_mean_gradients_match_dense_Gaussian():
    d = example()
    covariance = dense_covariance(d)
    mu = torch.linspace(-.04, .05, 4*d.frames, dtype=torch.float64).reshape(d.frames, 4)
    mu.requires_grad_()
    zero = torch.zeros_like(mu)
    q = torch.distributions.MultivariateNormal(mu.flatten(), covariance_matrix=covariance)
    q0 = torch.distributions.MultivariateNormal(zero.flatten(), covariance_matrix=covariance)
    expected = torch.distributions.kl_divergence(q, q0)
    exact = d.kl_reference(mu)
    torch.testing.assert_close(exact, expected, atol=3e-14, rtol=2e-14)
    torch.testing.assert_close(d.kl_reference_terms(mu).sum(), exact, atol=0, rtol=0)
    grad = torch.autograd.grad(exact, mu, retain_graph=True)[0]
    dense_grad = torch.linalg.solve(covariance, mu.flatten()).reshape_as(mu)
    torch.testing.assert_close(grad, dense_grad, atol=2e-13, rtol=2e-14)
    assert d.kl_reference(zero) == 0
    X = d.sample(mu.detach(), antithetic_noise('density-grad', 0, d.frames))
    log_q = d.log_prob_terms(X, mu).sum()
    density_grad = torch.autograd.grad(log_q, mu)[0]
    dense_density_grad = torch.linalg.solve(covariance,
        (X-mu.detach()).flatten(1).sum(0)).reshape_as(mu)
    torch.testing.assert_close(density_grad, dense_density_grad, atol=2e-12, rtol=2e-13)


def test_sampling_innovation_reconstruction_and_pathwise_mean_gradient():
    d = example()
    mu = torch.linspace(-.1, .1, d.frames*4, dtype=torch.float64).reshape(d.frames, 4)
    mu.requires_grad_()
    eps = antithetic_noise('pathwise', 3, d.frames)
    X = d.sample(mu, eps)
    torch.testing.assert_close(d.innovations(X-mu[None]), eps, atol=2e-15, rtol=2e-14)
    pathwise = torch.autograd.grad(X.sum(), mu)[0]
    assert torch.equal(pathwise, torch.full_like(mu, 4))
    detached = d.sample(mu, eps, detach_mean=True)
    assert not detached.requires_grad
    assert torch.equal(detached, X.detach())


def test_sample_covariance_has_fixed_stationary_marginals_and_AR_correlation():
    d = example()
    generator = torch.Generator().manual_seed(300)
    eps = torch.randn(25000, d.frames, 4, generator=generator, dtype=torch.float64)
    mu = torch.zeros(d.frames, 4, dtype=torch.float64)
    X = d.sample(mu, eps).flatten(1)
    torch.testing.assert_close(X.mean(0), torch.zeros_like(X.mean(0)), atol=.0025, rtol=0)
    torch.testing.assert_close(torch.cov(X.T), dense_covariance(d), atol=.0005, rtol=0)


def test_chart_inverse_keeps_unbounded_centers_and_positive_sizes():
    d = example()
    boxes = d.reference.clone()
    boxes[:, 0] = -1.5
    boxes[:, 1] = 2.5
    boxes[:, 2] *= .15
    boxes[:, 3] *= 4
    chart = d.chart(boxes)
    reconstructed = d.inverse(chart)
    torch.testing.assert_close(reconstructed, boxes, atol=2e-15, rtol=2e-15)
    assert (reconstructed[:, 0] < 0).all() and (reconstructed[:, 1] > 1).all()
    assert (reconstructed[:, 2:] > 0).all()
    # Log subtraction remains finite even when the intermediate ratio would
    # overflow. Inversion uses exp(log(Esize)+Xsize), with no ratio product.
    tiny = d.reference.clone()
    tiny[:, 2:] = 1e-200
    small_d = JointTrajectoryGaussian(tiny, d.frame_ids, d.anchor_frame_ids)
    large = tiny.clone()
    large[:, 2:] = 1e200
    assert torch.isinf(large[:, 2:]/tiny[:, 2:]).all()
    z = small_d.chart(large)
    assert torch.isfinite(z).all()
    torch.testing.assert_close(small_d.inverse(z), large, atol=0, rtol=5e-14)
    invalid = torch.zeros_like(chart)
    invalid[:, 2:] = 1e5
    with pytest.raises(FloatingPointError):
        d.inverse(invalid)


def test_chart_inverse_and_reference_KL_preserve_student_autograd():
    d = example()
    boxes = d.reference.clone().requires_grad_()
    reconstructed = d.inverse(d.chart(boxes))
    gradient = torch.autograd.grad(reconstructed.sum(), boxes)[0]
    torch.testing.assert_close(gradient, torch.ones_like(boxes), atol=2e-15, rtol=0)
    shifted = (d.reference+torch.tensor([.02, -.01, .01, .02], dtype=torch.float64)).requires_grad_()
    assert torch.autograd.gradcheck(lambda b: d.kl_reference(d.chart(b)), (shifted,),
                                   eps=1e-6, atol=2e-6, rtol=2e-5)


def test_antithetic_noise_order_determinism_and_no_global_rng_mutation():
    before = torch.random.get_rng_state().clone()
    a = antithetic_noise('same-query', 2, 10)
    assert torch.equal(before, torch.random.get_rng_state())
    assert torch.equal(a, antithetic_noise('same-query', 2, 10))
    assert torch.equal(a[0], -a[2]) and torch.equal(a[1], -a[3])
    assert not torch.equal(a[0], a[1])
    assert not torch.equal(a, antithetic_noise('other-query', 2, 10))
    assert not torch.equal(a, antithetic_noise('same-query', 3, 10))
    assert a.shape == (4, 10, 4) and a.dtype == torch.float64


def test_independent_pair_baseline_uses_only_other_pair_and_is_detached():
    returns = torch.arange(4*7, dtype=torch.float64).reshape(4, 7).requires_grad_()
    baseline = independent_pair_baseline(returns)
    assert not baseline.requires_grad
    assert torch.equal(baseline[0], (returns[1]+returns[3]).detach()/2)
    assert torch.equal(baseline[2], baseline[0])
    assert torch.equal(baseline[1], (returns[0]+returns[2]).detach()/2)
    assert torch.equal(baseline[3], baseline[1])
    altered = returns.detach().clone()
    altered[0] += 1e3
    altered[2] -= 30
    changed = independent_pair_baseline(altered)
    assert torch.equal(changed[[0, 2]], baseline[[0, 2]])
    assert not torch.equal(changed[[1, 3]], baseline[[1, 3]])


def test_invalid_grid_and_nonpositive_box_dimensions_fail_without_clamping():
    d = example()
    with pytest.raises(ValueError):
        JointTrajectoryGaussian(d.reference, d.frame_ids.flip(0), d.anchor_frame_ids)
    with pytest.raises(ValueError):
        JointTrajectoryGaussian(d.reference, d.frame_ids, d.anchor_frame_ids[:7])
    with pytest.raises(ValueError):
        JointTrajectoryGaussian(d.reference, d.frame_ids,
                                torch.tensor([0, 1, 3, 4, 6, 13, 18, 24]))
    invalid = d.reference.clone()
    invalid[:, 2] = 0
    with pytest.raises(ValueError):
        d.chart(invalid)
