import numpy as np
import pytest
from vg_tta.desta3d_v3_shared_channel_basis import (
    normalized_covariance, fit_train_basis, projection_statistics,
    random_basis, norm_matched_projection, energy_gate,
)


def test_query_weight_invariant_to_length_and_scale():
    a = np.array([[2., 0., 0.], [1., 0., 0.]])
    b = np.array([[0., 1., 0.]])
    c = normalized_covariance(a)
    np.testing.assert_allclose(normalized_covariance(np.tile(a * 37., (10, 1))), c)
    mean, _, _ = fit_train_basis([c, normalized_covariance(b)])
    np.testing.assert_allclose(mean, np.diag([.5, .5, 0.]))


def test_shared_vs_query_optimal_and_eval_only():
    train = [normalized_covariance(np.diag([3., 2., 1.]))]
    cov, eig, basis = fit_train_basis(train)
    dev = np.array([[0., 0., 8.]])
    result = projection_statistics(dev, basis, (1, 2, 3))
    assert result['1']['energy'] == 0 and result['3']['energy'] == 1
    cov2, eig2, basis2 = fit_train_basis(train)
    np.testing.assert_array_equal(cov, cov2)
    np.testing.assert_array_equal(basis, basis2)
    assert eig[0] > eig[1] > eig[2]


def test_projection_identity_norm_and_random():
    rng = np.random.default_rng(29)
    a = rng.standard_normal((3, 2, 4, 9));b = random_basis(9, 20260929)[:, :4]
    st = projection_statistics(a, b, (1, 4))['4']
    x = a.reshape(-1, 9);p = (x @ b) @ b.T
    cos = np.sum(x*p) / (np.linalg.norm(x)*np.linalg.norm(p))
    assert abs(cos**2-st['energy']) < 1e-12
    assert abs(cos-st['cosine']) < 1e-12
    matched = norm_matched_projection(a, b)
    assert abs(np.linalg.norm(matched)-np.linalg.norm(a)) < 1e-12
    np.testing.assert_array_equal(random_basis(9, 20260929), random_basis(9, 20260929))


def test_invalid_support_and_locked_gate():
    with pytest.raises(ValueError):normalized_covariance(np.zeros((3, 4)))
    with pytest.raises(ValueError):normalized_covariance(np.full((3, 4), np.nan))
    with pytest.raises(ValueError):norm_matched_projection(np.array([[1.,0.]]), np.array([[0.],[1.]]))
    assert energy_gate(.749999)=='stop_fixed_shared_basis'
    assert energy_gate(.75)=='conditional_dev64_native'
