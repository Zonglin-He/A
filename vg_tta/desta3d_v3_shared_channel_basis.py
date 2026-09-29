"""Query-balanced, uncentered channel subspaces; CPU arrays only."""
import numpy as np

RANKS = (1, 4, 8, 16, 32, 64)


def normalized_covariance(field):
    a = np.asarray(field, dtype=np.float64)
    if a.ndim < 2 or not np.isfinite(a).all() or min(a.shape) < 1:
        raise ValueError('Complete finite field required')
    x = a.reshape(-1, a.shape[-1])
    energy = float(np.sum(x * x))
    if energy <= 0:
        raise ValueError('Zero field has no defined directional covariance')
    return (x.T @ x) / energy


def fit_train_basis(train_covariances):
    """Caller passes Train only; equal query weight, no token concatenation."""
    c = np.asarray(train_covariances, dtype=np.float64)
    if c.ndim != 3 or len(c) == 0 or c.shape[1] != c.shape[2]:
        raise ValueError('Nonempty covariance stack required')
    if not np.isfinite(c).all() or not np.allclose(np.trace(c, axis1=1, axis2=2), 1, atol=1e-12, rtol=0):
        raise ValueError('Every query covariance must have unit trace')
    mean = c.mean(axis=0)
    eigenvalues, basis = np.linalg.eigh(mean)
    eigenvalues, basis = eigenvalues[::-1], basis[:, ::-1]
    # Fix signs for reproducibility; repeated eigenspaces are compared as projectors.
    for j in range(basis.shape[1]):
        if basis[np.argmax(np.abs(basis[:, j])), j] < 0:
            basis[:, j] *= -1
    return mean, eigenvalues, basis


def random_basis(dimension, seed):
    q, r = np.linalg.qr(np.random.default_rng(seed).standard_normal((dimension, dimension)))
    return q * np.where(np.diag(r) >= 0, 1., -1.)


def projection_statistics(field, basis, ranks=RANKS):
    x = np.asarray(field, dtype=np.float64).reshape(-1, basis.shape[0])
    energy = float(np.sum(x * x))
    if energy <= 0 or not np.isfinite(x).all():
        raise ValueError('Undefined field support')
    if not np.allclose(basis.T @ basis, np.eye(basis.shape[1]), atol=1e-10, rtol=0):
        raise ValueError('Orthonormal columns required')
    reduced = x @ basis[:, :max(ranks)]
    cumulative = np.cumsum(np.sum(reduced * reduced, axis=0)) / energy
    return {str(k): dict(energy=float(cumulative[k-1]), cosine=float(np.sqrt(max(0., cumulative[k-1])))) for k in ranks}


def norm_matched_projection(field, basis):
    """Finite-oracle contract only; not a learned prediction or scale search."""
    a = np.asarray(field, dtype=np.float64)
    x = a.reshape(-1, a.shape[-1])
    if not np.allclose(basis.T @ basis, np.eye(basis.shape[1]), atol=1e-10, rtol=0):
        raise ValueError('Orthonormal columns required')
    p = (x @ basis) @ basis.T
    n = np.linalg.norm(p)
    if n <= 0 or not np.isfinite(n):
        raise ValueError('Zero projection: preserve failure; no arbitrary direction')
    return (p * (np.linalg.norm(x) / n)).reshape(a.shape)


def aggregate(rows):
    out = {}
    for kind in ('shared', 'random', 'optimal'):
        out[kind] = {}
        for k in RANKS:
            out[kind][str(k)] = {}
            keys = ('energy', 'cosine', 'retention_ratio') if kind != 'optimal' else ('energy', 'cosine')
            for metric in keys:
                a = np.asarray([r[kind][str(k)][metric] for r in rows], dtype=np.float64)
                out[kind][str(k)][metric] = dict(defined=len(a), mean=float(a.mean()), median=float(np.median(a)), min=float(a.min()), max=float(a.max()))
    return out


def energy_gate(dev_rank32_median):
    if not np.isfinite(dev_rank32_median):
        raise ValueError('Missing screen measurement')
    return 'conditional_dev64_native' if dev_rank32_median >= .75 else 'stop_fixed_shared_basis'
