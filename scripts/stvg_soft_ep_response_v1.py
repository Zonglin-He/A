"""Matrix-free post-Adam decoder-response filter; no model, GT, or CUDA setup."""
from __future__ import annotations

import argparse
import math
import time
from typing import Sequence

import torch

def _flat(xs):
    return torch.cat([x.reshape(-1) for x in xs]) if xs else torch.empty(0)
def _pieces(v, params):
    v = v.reshape(-1)
    if sum(p.numel() for p in params) != v.numel():
        raise ValueError("direction size does not match parameter list")
    out, i = [], 0
    for p in params:
        out.append(v[i:i + p.numel()].to(device=p.device, dtype=p.dtype).reshape_as(p))
        i += p.numel()
    return out
def _dot(a, b):
    return float(torch.dot(a.reshape(-1).double(), b.reshape(-1).double()).item())
def _vjp(y, params, cotangent):
    grads = torch.autograd.grad(y, params, grad_outputs=cotangent.reshape_as(y),
                                retain_graph=True, allow_unused=True)
    return _flat([torch.zeros_like(p) if g is None else g for p, g in zip(params, grads)])
def graph_jvp(y, params: Sequence[torch.Tensor], v):
    """True native-graph JVP via double backward: D_v y, with no FD estimate."""
    y = y.reshape(-1)
    cot = torch.zeros_like(y, requires_grad=True)
    grads = torch.autograd.grad(y, params, grad_outputs=cot, create_graph=True,
                                retain_graph=True, allow_unused=True)
    terms = [(g * q).sum() for p, g, q in zip(params, grads, _pieces(v, params)) if g is not None]
    if not terms:
        return torch.zeros_like(y)
    pull = sum(terms)
    return torch.autograd.grad(pull, cot, retain_graph=True, allow_unused=False)[0]
def solve_from_graph(y, params: Sequence[torch.Tensor], d, mu=1.0,
                     max_iter=16, rtol=1e-3):
    """Solve (I + lambda J'J)s=d by CG; y is already scalar-mean normalized."""
    start = time.perf_counter()
    y = y.reshape(-1)
    params = list(params)
    if not params or not y.numel() or mu < 0 or max_iter < 1 or rtol <= 0:
        raise ValueError("invalid response solver inputs")
    if any(not p.requires_grad or p.device != y.device or p.dtype != y.dtype for p in params):
        raise ValueError("y and all differentiable parameters must share device and dtype")
    d = d.reshape(-1).to(device=params[0].device, dtype=params[0].dtype)
    if d.numel() != sum(p.numel() for p in params) or not torch.isfinite(d).all():
        raise ValueError("Adam direction has invalid shape or values")

    d2 = _dot(d, d)
    j_d = graph_jvp(y, params, d)
    j_d2 = _dot(j_d, j_d)
    threshold = 1e-12 * max(1.0, math.sqrt(d2))
    if math.sqrt(j_d2) <= threshold:
        stats = {"status": "near_zero_response_identity", "lambda": 0.0, "mu": float(mu),
                 "rho": 1.0, "response_norm_ratio": 1.0, "cos": 1.0, "realcost": j_d2,
                 "response_cost_adam": j_d2, "response_cost_filtered": j_d2,
                 "solver_residual_abs": 0.0, "solver_residual_rel": 0.0,
                 "iterations": 0, "cg_rhs": d.detach().cpu(),
                 "cg_residual": torch.zeros_like(d).cpu(), "Jd": j_d.detach().cpu(),
                 "Jfiltered": j_d.detach().cpu(), "solution": d.detach().cpu(),
                 "elapsed_seconds": time.perf_counter() - start,
                 "near_zero_threshold": threshold}
        return d.clone(), stats
    lam = float(mu) * d2 / j_d2
    if not math.isfinite(lam):
        raise FloatingPointError("non-finite normalized lambda")
    def apply(x):
        jx = graph_jvp(y, params, x)
        return x if lam == 0 else x + lam * _vjp(y, params, jx)
    x = torch.zeros_like(d)
    r = d.clone()
    p = r.clone()
    rr = _dot(r, r)
    iterations = 0
    status = "max_iter"
    for k in range(max_iter):
        ap = apply(p)
        denom = _dot(p, ap)
        if not math.isfinite(denom) or denom <= 0:
            raise FloatingPointError("CG encountered a non-positive/non-finite curvature")
        alpha = rr / denom
        x = x + alpha * p
        r = r - alpha * ap
        rr_new = _dot(r, r)
        iterations = k + 1
        if math.sqrt(rr_new) <= rtol * max(math.sqrt(d2), 1e-30):
            rr = rr_new
            status = "converged"
            break
        beta = rr_new / rr if rr else 0.0
        p = r + beta * p
        rr = rr_new
    j_x = graph_jvp(y, params, x)
    ax = x if lam == 0 else x + lam * _vjp(y, params, j_x)
    residual = d - ax
    res_abs = math.sqrt(_dot(residual, residual))
    res_rel = res_abs / max(math.sqrt(d2), 1e-30)
    response_norm_ratio = math.sqrt(_dot(j_x, j_x) / j_d2)
    rho = math.sqrt(_dot(x, x) / d2)
    cos = _dot(d, x) / max(math.sqrt(d2 * _dot(x, x)), 1e-30)
    stats = {"status": status, "lambda": lam, "mu": float(mu), "rho": rho, "response_norm_ratio": response_norm_ratio, "cos": cos,
             "parameter_norm_adam": math.sqrt(d2), "parameter_norm_filtered": math.sqrt(_dot(x,x)),
             "realcost": j_d2, "response_cost_adam": j_d2,
             "response_cost_filtered": _dot(j_x, j_x),
             "solver_residual_abs": res_abs, "solver_residual_rel": res_rel,
             "iterations": iterations, "cg_rhs": d.detach().cpu(),
             "cg_residual": residual.detach().cpu(), "normal_applied": ax.detach().cpu(),
             "Jd": j_d.detach().cpu(), "Jfiltered": j_x.detach().cpu(),
             "solution": x.detach().cpu(), "elapsed_seconds": time.perf_counter() - start,
             "near_zero_threshold": threshold}
    return x, stats
def _self_test():
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU synthetic test refuses initialized CUDA")
    a = torch.tensor([[1.0, 2.0, -1.0], [0.5, -1.0, 3.0],
                      [2.0, 0.25, 1.0], [-1.0, 1.5, 0.5]], dtype=torch.float32)
    p = torch.nn.Parameter(torch.tensor([0.2, -0.4, 0.7], dtype=torch.float32))
    y = a @ p
    d = torch.tensor([0.3, -0.2, 0.1], dtype=torch.float32)
    jv = graph_jvp(y, [p], d)
    if not torch.allclose(jv, a @ d, atol=1e-6, rtol=1e-6):
        raise AssertionError("double-backward JVP differs from exact linear map")
    direction, stats = solve_from_graph(y, [p], d, mu=1.0, max_iter=16, rtol=1e-6)
    lam = stats["lambda"]
    exact = torch.linalg.solve(torch.eye(3) + lam * (a.T @ a), d)
    if not torch.allclose(direction, exact, atol=2e-5, rtol=2e-5):
        raise AssertionError("matrix-free CG differs from direct linear solve")
    if stats["status"] != "converged" or stats["solver_residual_rel"] > 1e-5:
        raise AssertionError("synthetic CG residual did not converge")
    return {"status": "pass", "checks": 4, "GT_read": False,
            "GPU_used": False, "model_loaded": False,
            "JVP_exact_linear_map": True, "CG_matches_direct_solve": True,
            "relative_residual": stats["solver_residual_rel"]}
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not args.self_test:
        parser.error("this helper is imported by the frozen fitter; use --self-test for CPU validation")
    print(_self_test())
if __name__ == "__main__":
    main()
