"""CPU algebra of an existing residual; no model, labels, or new correction."""
import numpy as np


def dot(a, b):
    return float(np.einsum('ij,ij->', a, b, optimize=True))


def orth(x, rtol=1e-10):
    x = np.asarray(x, dtype=np.float64)
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    return u[:, s > (s[0] * rtol if len(s) else 0)], s


def bases(qt, qs, rtol=1e-10):
    qt, st = orth(qt, rtol); qs, ss = orth(qs, rtol)
    u, sv = orth(np.concatenate([qt, qs], axis=1), rtol)
    # Coordinates in a common orthonormal union. QR removes only FP64 roundoff.
    t = np.linalg.qr(u.T @ qt, mode='reduced')[0]
    s = np.linalg.qr(u.T @ qs, mode='reduced')[0]
    overlap = np.linalg.svd(qt.T @ qs, compute_uv=False)
    cat = np.concatenate([t, s], axis=1)
    return dict(U=u, T=t, S=s, cat=cat, inverse=np.linalg.pinv(cat, rcond=rtol),
                report=dict(T_rank=qt.shape[1], S_rank=qs.shape[1], union_rank=u.shape[1],
                    intersection_dimension=qt.shape[1]+qs.shape[1]-u.shape[1],
                    union_singular_values=sv.tolist(), principal_cosines=overlap.tolist(),
                    principal_angles_degrees=np.degrees(np.arccos(np.clip(overlap, -1, 1))).tolist(),
                    direct_sum_condition=float(sv[0]/sv[u.shape[1]-1]),
                    orthogonality_max=float(np.max(abs(u.T@u-np.eye(u.shape[1]))))))


def proj(x, q):
    return (x @ q) @ q.T


def split(j, b):
    """Decompose union coordinates; perpendicular raw residual handled by caller."""
    t, s = proj(j, b['T']), proj(j, b['S'])
    c = j @ b['inverse'].T
    nt = b['T'].shape[1]
    direct_t, direct_s = c[:, :nt] @ b['T'].T, c[:, nt:] @ b['S'].T
    return {'T_first': {'T': t, 'S_given_T': j-t},
            'S_first': {'S': s, 'T_given_S': j-s},
            'direct_sum': {'T': direct_t, 'S': direct_s}}


def attribution(j, gradients, b):
    parts = split(j, b); result = {}
    for convention, components in parts.items():
        a, c = components.values()
        result[convention] = dict(
            components={name: dict(energy=dot(v, v),
                                   local_descent={k: -dot(g, v) for k, g in gradients.items()})
                        for name, v in components.items()},
            twice_energy_cross_term=2*dot(a, c),
            reconstruction_L2=float(np.linalg.norm(j-a-c)))
    result['naive_sum_relative_error'] = float(np.linalg.norm(j-proj(j,b['T'])-proj(j,b['S'])) / max(np.linalg.norm(j),1e-30))
    return result


def gradient_capacity(gradients, full_energy, b):
    out = {}
    for k, g in gradients.items():
        own = proj(g, b[k]); extra = g-own
        en, on, ex = dot(g,g), dot(own,own), dot(extra,extra)
        out[k] = dict(full_energy=full_energy[k], union_energy=en, own_energy=on,
                      additional_energy=ex, union_fraction_of_full=en/max(full_energy[k],1e-30),
                      own_fraction_of_union=on/max(en,1e-30),
                      additional_fraction_of_union=ex/max(en,1e-30))
    return out
