"""Four-parameter error-state diagnostic. No model, labels or tuning in this module."""
import numpy as np


def encode(boxes):
    b = np.asarray(boxes, dtype=np.float64)
    assert b.ndim == 2 and b.shape[1] == 4 and np.isfinite(b).all()
    assert (b[:, 2:] > 0).all()
    return np.column_stack((b[:, :2], np.log(b[:, 2:])))


def decode(z):
    z = np.asarray(z, dtype=np.float64)
    out = np.column_stack((z[:, :2], np.exp(z[:, 2:])))
    assert np.isfinite(out).all() and (out[:, 2:] > 0).all()
    return out


def correct(boxes, positions, targets):
    """Return a local-node control and one global residual-mean correction.

    Coordinates are normalized cx,cy,w,h. Local changes only the observed actual
    decoder nodes; downstream evaluation retains the original linear xyxy
    interpolation. Global uses a unit-weight least-squares constant in
    (cx,cy,log w,log h), without clipping or reference-quality selection.
    """
    b = np.asarray(boxes, dtype=np.float64)
    z = encode(b)
    p = np.asarray(positions, dtype=np.int64)
    target = np.asarray(targets, dtype=np.float64).reshape(-1, 4)
    assert len(p) == len(target) and len(set(p.tolist())) == len(p)
    assert len(p) == 0 or (p.min() >= 0 and p.max() < len(b))
    if not len(p):
        return dict(local=b.copy(), global_error=b.copy(), delta=np.zeros(4),
                    residuals=np.zeros((0, 4)), reference_count=0,
                    before_z_SSE=0., after_z_SSE=0., mean_residual_error=0.,
                    center_displacement_max_error=0., log_size_change_max_error=0.)
    residuals = encode(target) - z[p]
    delta = residuals.mean(0)
    local = b.copy()
    local[p] = target
    corrected_z = z + delta
    global_error = decode(corrected_z)
    before = float(np.square(residuals).sum())
    after = float(np.square(residuals - delta).sum())
    # Constant LS identity; this is an algebraic contract, not a GT-IoU claim.
    assert abs(before - after - len(p) * float(delta @ delta)) < 1e-9 * (1 + before)
    assert after <= before + 1e-12
    center_error = float(np.max(np.abs(np.diff(global_error[:, :2], axis=0) - np.diff(b[:, :2], axis=0))))
    size_error = float(np.max(np.abs(np.diff(encode(global_error)[:, 2:], axis=0) - np.diff(z[:, 2:], axis=0))))
    assert center_error < 1e-12 and size_error < 1e-12
    return dict(local=local, global_error=global_error, delta=delta,
                residuals=residuals, reference_count=len(p),
                before_z_SSE=before, after_z_SSE=after,
                mean_residual_error=float(np.abs((encode(target) - encode(global_error)[p]).mean(0)).max()),
                center_displacement_max_error=center_error,
                log_size_change_max_error=size_error)


def contract():
    # Nonlinear motion is preserved exactly under a truly constant known error.
    t = np.linspace(0, 1, 31)
    b = np.column_stack((.3 + .2 * t**2, .5 + .1 * np.sin(7*t), .1 + .05*t, .2 + .03*t**2))
    d = np.array([.07, -.03, np.log(1.3), np.log(.8)])
    target = decode(encode(b) + d)
    p = [2, 9, 20, 28]
    c = correct(b, p, target[p])
    assert np.max(np.abs(c['global_error'] - target)) < 1e-13
    other = np.setdiff1d(np.arange(len(b)), p)
    assert np.array_equal(c['local'][other], b[other])
    assert np.array_equal(c['local'][p], target[p])
    assert np.max(np.abs(c['local'][other] - target[other])) > .01
    empty = correct(b, [], [])
    assert np.array_equal(empty['local'], b) and np.array_equal(empty['global_error'], b)
    # Scale ratios, rather than additive width corrections, distinguish the
    # global mode from local cxcywh-node edits and logit residual interpolation.
    ratio = c['global_error'][:, 2:] / b[:, 2:]
    assert np.max(np.abs(ratio - np.exp(d[2:]))) < 1e-13
    return dict(status='pass',constant_error_recovery=True,nonlinear_center_motion_preserved=True,
                log_size_differences_preserved=True,local_nonanchor_nodes_exact=True,
                empty_reference_exact_noop=True,no_clamp=True,not_equal_to_local_control=True)
