"""CPU reductions for saved direction audits; no model or CUDA access."""
import math
import numpy as np


def pair_stats(x, y, chunk=131072):
    x, y = np.asarray(x), np.asarray(y)
    if x.shape != y.shape or x.size == 0:
        raise ValueError('Directions require identical, nonempty support')
    a, b = x.reshape(-1), y.reshape(-1)
    xx, yy, xy = [], [], []
    for start in range(0, a.size, chunk):
        u = a[start:start+chunk].astype(np.float64)
        v = b[start:start+chunk].astype(np.float64)
        if not np.isfinite(u).all() or not np.isfinite(v).all():
            raise ValueError('Nonfinite direction')
        xx.append(float(np.dot(u, u)))
        yy.append(float(np.dot(v, v)))
        xy.append(float(np.dot(u, v)))
    nx, ny = math.sqrt(math.fsum(xx)), math.sqrt(math.fsum(yy))
    dot = math.fsum(xy)
    return dict(x_norm=nx, y_norm=ny, dot=dot,
                cosine=dot/(nx*ny) if nx and ny else None)


def sign_gate(span_descent, late_descent, context_descent, tolerance=1e-9):
    """Strict sufficient sign pattern from the attachment; not a universal test."""
    return (span_descent > tolerance and late_descent < -tolerance
            and context_descent < -tolerance)
