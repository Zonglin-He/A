"""Independent CPU state/Adam audit for joint/LN-only x carry/reset OPD.

No I/O, model, optimizer or label operation is performed by the two public
auditors. NumPy reconstructs saved arithmetic. selftest() alone uses a tiny
CPU torch.optim.Adam fixture, never a scientific model or a saved real fit.
The existing P3 objective/action/reward/gradient audit remains separate.
"""
import math

QUERY = "spatial.query_residual"
LN_NAMES = tuple(f"spatial.layers.5.{norm}.{kind}"
                 for norm in ("norm1", "norm3", "norm4")
                 for kind in ("weight", "bias"))
ALL_NAMES = (QUERY,) + LN_NAMES
PROPOSAL_THRESHOLD = 4e-6  # Unchanged existing independent Adam threshold.
MOMENT_THRESHOLD = 2e-5    # Unchanged existing full-path moment threshold.


def _array(value):
    import numpy as np
    if hasattr(value, "detach"):
        assert value.device.type == "cpu", "CPU audit accepts CPU witnesses only"
        value = value.detach().numpy()
    out = np.asarray(value)
    assert np.isfinite(out).all()
    return out


def _float32(value):
    import numpy as np
    out = _array(value)
    assert out.dtype == np.float32
    return out


def _exact(first, second):
    a, b = _array(first), _array(second)
    assert a.shape == b.shape and a.dtype == b.dtype
    assert a.tobytes(order="C") == b.tobytes(order="C"), "Bitwise witness differs"
    return a.size


def _tree_exact(first, second):
    if isinstance(first, dict):
        assert isinstance(second, dict) and first.keys() == second.keys()
        return sum(_tree_exact(v, second[k]) for k, v in first.items())
    if isinstance(first, (list, tuple)):
        assert type(first) is type(second) and len(first) == len(second)
        return sum(_tree_exact(a, b) for a, b in zip(first, second))
    if hasattr(first, "shape") or hasattr(first, "detach"):
        return _exact(first, second)
    assert type(first) is type(second) and first == second
    return 1


def _state(value):
    assert isinstance(value, dict) and tuple(value) == ALL_NAMES
    out = {n: _float32(value[n]) for n in ALL_NAMES}
    assert all(v.shape == (256,) for v in out.values())
    return out


def _zero(value):
    import numpy as np
    return _exact(value, np.zeros_like(_float32(value)))


def _flat(state, names):
    import numpy as np
    return np.concatenate([_float32(state[n]).reshape(-1) for n in names])


def _slots(optimizer, names, state, lr, step):
    """Map actual serialized optimizer identifiers in param-group order."""
    import numpy as np
    assert set(optimizer) == {"state", "param_groups"}
    assert len(optimizer["param_groups"]) == 1
    group = optimizer["param_groups"][0]
    assert float(group["lr"]) == lr and tuple(group["betas"]) == (.9, .999)
    assert float(group["eps"]) == 1e-8 and float(group["weight_decay"]) == 0.
    for option in ("amsgrad", "maximize", "capturable", "differentiable", "decoupled_weight_decay"):
        assert not group.get(option, False), option
    identifiers = group["params"]
    assert len(identifiers) == len(names) and len(set(identifiers)) == len(names)
    assert all(type(k) is int for k in identifiers)
    saved = optimizer["state"]
    assert set(saved) == (set(identifiers) if step else set())
    if not step:
        z = np.zeros(sum(state[n].size for n in names), dtype=np.float32)
        return z, z.copy(), list(identifiers)
    means, variances = [], []
    for identifier, name in zip(identifiers, names):
        slot = saved[identifier]
        assert set(slot) == {"step", "exp_avg", "exp_avg_sq"}
        at = _array(slot["step"])
        assert at.size == 1 and float(at.reshape(-1)[0]) == step
        first, second = _float32(slot["exp_avg"]), _float32(slot["exp_avg_sq"])
        assert first.shape == second.shape == state[name].shape and (second >= 0).all()
        means.append(first.reshape(-1)); variances.append(second.reshape(-1))
    return np.concatenate(means), np.concatenate(variances), list(identifiers)


def _local_moment(actual, expected, scale):
    """One-step FP32 arithmetic bound, in addition to legacy path threshold."""
    import numpy as np
    u = np.finfo(np.float32).eps / 2
    gamma = 32 * u / (1 - 32 * u)
    bound = gamma * scale + np.finfo(np.float32).tiny + 64 * np.finfo(float).eps * np.maximum(1., scale)
    error = np.abs(actual.astype(float) - expected)
    assert np.all(error <= bound), "Saved Adam moment violates one-step FP32 bound"
    return float(error.max()), float((error / bound).max())


def adam_readback(fit):
    """Audit CPU full-fit witnesses; returns scalar receipts, changes nothing.

    Requires geometry.rounds with Adam_before/after for each nonempty-support
    round. Genuine empty-support fits have no optimizer-call trace and are
    accepted only as an exact full-state/full-box no-op. QR must be zero in
    every LN-only path state; source LN is never treated as a zero delta.
    """
    import numpy as np
    scope = fit.get("scope", "joint")
    assert scope in ("joint", "LN_only") and fit["arm"] == "on_policy"
    assert not fit["GT_used"] and not fit.get("direct_objective", False)
    initial, final = _state(fit["initial"]), _state(fit["state"])
    _zero(initial[QUERY])
    names = list(fit.get("optimizer_parameter_names", list(initial)))
    expected_names = list(ALL_NAMES if scope == "joint" else LN_NAMES)
    assert names == expected_names and fit["active_parameters"] == (1792 if scope == "joint" else 1536)
    cfg = fit["config"]; steps = cfg["steps"]; lr = float(cfg["lr"])
    assert type(steps) is int and steps > 0 and math.isfinite(lr) and lr > 0
    assert cfg["samples"] == 32 and fit["selected_step"] == steps
    path, rounds, trace = fit["path"], fit["rounds"], fit["geometry"]["rounds"]
    assert len(path) == steps + 1 and len(rounds) == steps
    assert fit["empty"] == (not fit["positions"])
    checks = _tree_exact(fit["initial"], path[0]["state"]) + _tree_exact(fit["state"], path[-1]["state"])
    checks += _exact(fit["before"], path[0]["boxes"]) + _exact(fit["final"], path[-1]["boxes"])
    for k, entry in enumerate(path):
        assert entry["step"] == k
        values = _state(entry["state"])
        if scope == "LN_only": checks += _zero(values[QUERY])
    if fit["empty"]:
        assert not trace and fit["gradient_calls"] == 0
        for entry, round_ in zip(path[:-1], rounds):
            assert "update" not in entry and not round_["updated"] and round_["no_information"]
        for entry in path:
            checks += _tree_exact(entry["state"], fit["initial"]) + _exact(entry["boxes"], fit["before"])
        return dict(status="pass", scope=scope, active_parameters=fit["active_parameters"],
            scalar_checks=checks, actual_Adam_steps=0, no_update_rounds=steps, optimizer_trace_rounds=0,
            empty_support_exact_noop=True, inactive_QR_exact_zero=scope == "LN_only",
            CPU_only=True, GT_read=False, independent_full_decoder_Jacobian=False)
    assert len(trace) == steps and [x["round"] for x in trace] == list(range(steps))
    moment = np.zeros(fit["active_parameters"]); variance = moment.copy()
    updates = no_updates = 0; previous_after = identifiers = None
    max_proposal = max_moment = max_variance = max_moment_fraction = 0.
    for k, (entry, round_, geometry) in enumerate(zip(path[:-1], rounds, trace)):
        state, next_state = _state(entry["state"]), _state(path[k + 1]["state"])
        before, after = geometry["Adam_before"], geometry["Adam_after"]
        if previous_after is not None: checks += _tree_exact(before, previous_after)
        m0, v0, ids = _slots(before, names, state, lr, updates)
        if identifiers is None: identifiers = ids
        assert ids == identifiers
        checks += _tree_exact(before["param_groups"], after["param_groups"])
        updated = "update" in entry
        assert round_["round"] == k and bool(round_["updated"]) == updated
        assert bool(round_["no_information"]) == (not updated)
        if not updated:
            no_updates += 1
            checks += _tree_exact(before, after) + _tree_exact(entry["state"], path[k + 1]["state"])
            checks += _exact(entry["boxes"], path[k + 1]["boxes"])
            _slots(after, names, next_state, lr, updates)
            previous_after = after; continue
        update = entry["update"]
        assert update["names"] == names and update["optimizer"] == "adam" and float(update["lr"]) == lr
        assert "proposal_raw" not in update and "output_cap" not in update
        gradient, raw = _float32(update["gradient"]), _float32(update["raw"])
        assert gradient.shape == raw.shape == (fit["active_parameters"],)
        g = gradient.astype(float); updates += 1
        ma, va, ids = _slots(after, names, next_state, lr, updates)
        assert ids == identifiers
        first, f1 = _local_moment(ma, .9 * m0.astype(float) + .1 * g, np.abs(m0.astype(float)) + np.abs(g))
        second, f2 = _local_moment(va, .999 * v0.astype(float) + .001 * g * g, np.abs(v0.astype(float)) + .001 * g * g)
        max_moment = max(max_moment, first); max_variance = max(max_variance, second)
        max_moment_fraction = max(max_moment_fraction, f1, f2)
        moment = .9 * moment + .1 * g; variance = .999 * variance + .001 * g * g
        assert np.max(np.abs(ma.astype(float) - moment)) < MOMENT_THRESHOLD * (1 + np.max(np.abs(moment)))
        assert np.max(np.abs(va.astype(float) - variance)) < MOMENT_THRESHOLD * (1 + np.max(np.abs(variance)))
        proposal = -lr * (moment / (1 - .9 ** updates)) / (np.sqrt(variance / (1 - .999 ** updates)) + 1e-8)
        error = float(np.max(np.abs(proposal - raw.astype(float))))
        assert error < PROPOSAL_THRESHOLD, "Independent Adam proposal exceeds unchanged 4e-6"
        max_proposal = max(max_proposal, error)
        # raw is the recorded rounded net displacement, not an unrounded Adam delta.
        displacement = np.subtract(_flat(next_state, names), _flat(state, names), dtype=np.float32)
        checks += _exact(raw, displacement)
        if scope == "LN_only": checks += _exact(state[QUERY], next_state[QUERY])
        checks += fit["active_parameters"] * 5 + len(names)
        previous_after = after
    assert updates == fit["gradient_calls"]
    return dict(status="pass", scope=scope, active_parameters=fit["active_parameters"], scalar_checks=checks,
        actual_Adam_steps=updates, no_update_rounds=no_updates, optimizer_trace_rounds=len(trace),
        maximum_Adam_proposal_error=max_proposal, maximum_one_step_first_moment_error=max_moment,
        maximum_one_step_second_moment_error=max_variance, maximum_one_step_moment_bound_fraction=max_moment_fraction,
        proposal_absolute_threshold=PROPOSAL_THRESHOLD, path_moment_relative_threshold=MOMENT_THRESHOLD,
        one_step_moment_operation_bound=32, actual_optimizer_name_order_checked=True,
        inactive_QR_exact_zero=scope == "LN_only", source_LN_not_assumed_zero=True,
        CPU_only=True, GT_read=False, independent_full_decoder_Jacobian=False)


def lifecycle(source, initial, final, committed, previous, carry, alpha):
    """Check absolute LN source/carry/reset and separate, zeroed QR storage."""
    import numpy as np
    assert type(carry) is bool and not isinstance(alpha, bool)
    alpha = float(alpha); assert math.isfinite(alpha)
    assert (0 < alpha < 1) if carry else alpha == 0
    states = [_state(x) for x in (source, initial, final, committed)]
    src, enter, end, commit = states
    assert any(np.count_nonzero(src[n]) for n in LN_NAMES), "Source LN is an absolute nonzero checkpoint state"
    checks = _zero(src[QUERY]) + _zero(enter[QUERY]) + _zero(commit[QUERY])
    prev = None if previous is None else _state(previous)
    if prev is not None: checks += _zero(prev[QUERY])
    for name in ALL_NAMES:
        assert not np.shares_memory(src[name], enter[name]), "Entering state must clone source/history"
        assert not np.shares_memory(src[name], commit[name]), "Committed state must not alias immutable source"
        if prev is not None: assert not np.shares_memory(prev[name], enter[name]), "History must be cloned"
    for name in LN_NAMES:
        origin = prev[name] if carry and prev is not None else src[name]
        checks += _exact(enter[name], origin)
        if carry:
            expected = np.add(enter[name], np.multiply(np.subtract(end[name], enter[name], dtype=np.float32),
                np.float32(alpha), dtype=np.float32), dtype=np.float32)
        else:
            expected = src[name]
            if prev is not None: checks += _exact(prev[name], src[name])
        checks += _exact(commit[name], expected)
    return dict(status="pass", carry=carry, alpha=alpha, scalar_checks=checks,
        QR_reset_exact_zero=True, source_LN_absolute_values_preserved=True,
        source_LN_nonzero_coordinates=sum(int(np.count_nonzero(src[n])) for n in LN_NAMES),
        reset_source_clone_checked=not carry, independent_history_clone_checked=True,
        CPU_only=True, GT_read=False)


def selftest():
    """Pure synthetic CPU contracts; never calls either scientific fitter."""
    import copy
    import numpy as np
    import torch
    assert not torch.cuda.is_initialized()
    generator = torch.Generator(device="cpu").manual_seed(20261013)
    checks = fits = failures = 0
    def clone(value): return {n: v.clone() for n, v in value.items()}
    source = {n: (torch.zeros(256) if n == QUERY else torch.linspace(-.7, 1.3, 256)) for n in ALL_NAMES}
    def make_fit(scope, steps, empty=False):
        parameters = {n: torch.nn.Parameter(v.clone()) for n, v in source.items()}
        names = list(ALL_NAMES if scope == "joint" else LN_NAMES)
        optimizer = torch.optim.Adam([parameters[n] for n in names], lr=.01, betas=(.9, .999), eps=1e-8, weight_decay=0.)
        snapshot = lambda: {n: v.detach().clone() for n, v in parameters.items()}
        boxes = lambda: torch.stack([parameters[n].detach()[:4] for n in LN_NAMES])
        initial = snapshot(); path = []; rounds = []; geometry = []; count = 0
        for k in range(steps):
            entry = dict(step=k, state=snapshot(), boxes=boxes()); path.append(entry)
            update = not empty and k % 4 != 0
            before = copy.deepcopy(optimizer.state_dict())
            if update:
                optimizer.zero_grad(set_to_none=True)
                gs = [torch.randn(256, generator=generator) * .1 for n in names]
                prior = torch.cat([parameters[n].detach().clone() for n in names])
                for n, g in zip(names, gs): parameters[n].grad = g
                optimizer.step(); count += 1
                after = torch.cat([parameters[n].detach() for n in names])
                entry["update"] = dict(names=names, optimizer="adam", lr=.01,
                    gradient=torch.cat(gs), raw=after - prior)
            rounds.append(dict(round=k, updated=update, no_information=not update))
            if not empty: geometry.append(dict(round=k, Adam_before=before, Adam_after=copy.deepcopy(optimizer.state_dict())))
        path.append(dict(step=steps, state=snapshot(), boxes=boxes()))
        return dict(arm="on_policy", scope=scope, initial=initial, state=snapshot(), path=path, rounds=rounds,
            before=path[0]["boxes"], final=path[-1]["boxes"], positions=[] if empty else [0], empty=empty,
            config=dict(lr=.01, steps=steps, samples=32), selected_step=steps, gradient_calls=count,
            active_parameters=1792 if scope == "joint" else 1536, optimizer_parameter_names=names,
            GT_used=False, geometry=dict(rounds=geometry))
    witnesses = []
    for scope in ("joint", "LN_only"):
        for steps in (1, 10, 40):
            for empty in (False, True):
                fit = make_fit(scope, steps, empty); result = adam_readback(fit)
                checks += result["scalar_checks"]; fits += 1; witnesses.append(fit)
    selected = next(f for f in witnesses if f["scope"] == "LN_only" and f["config"]["steps"] == 10 and not f["empty"])
    def rejects(call):
        nonlocal failures
        try: call()
        except AssertionError: failures += 1
        else: raise AssertionError("Synthetic corrupted witness was accepted")
    for mutation in (
        lambda f: f["path"][2]["state"][QUERY].fill_(1.),
        lambda f: f["geometry"]["rounds"][1]["Adam_after"]["state"][0]["exp_avg"].add_(.1),
        lambda f: f["geometry"]["rounds"][2]["Adam_before"]["state"][0]["step"].add_(1),
        lambda f: f["path"][1]["update"]["raw"].add_(.01),
        lambda f: f["optimizer_parameter_names"].__setitem__(0, QUERY),
        lambda f: f["geometry"]["rounds"][4]["Adam_after"]["state"][0]["exp_avg"].add_(.1),
        lambda f: f["geometry"]["rounds"][1]["Adam_after"]["param_groups"][0].__setitem__("weight_decay", .1),
    ):
        corrupted = copy.deepcopy(selected); mutation(corrupted); rejects(lambda: adam_readback(corrupted))
    for carry, alpha in ((False, 0.), (True, 1/16), (True, 1/8)):
        previous = None
        for j in range(3):
            initial = clone(previous if carry and previous is not None else source)
            final = clone(initial)
            for n in LN_NAMES: final[n] += .01 * (j + 1)
            final[QUERY] += .25  # Joint updates QR; only the submitted QR resets.
            committed = clone(source) if not carry else {n: torch.zeros_like(v) if n == QUERY else v + (final[n] - v) * alpha for n, v in initial.items()}
            checks += lifecycle(source, initial, final, committed, previous, carry, alpha)["scalar_checks"]
            previous = clone(committed)
    initial, final, committed = clone(source), clone(source), clone(source)
    rejects(lambda: lifecycle(source, initial, final, committed, None, False, .1))
    bad = clone(source); bad[LN_NAMES[0]].zero_()
    rejects(lambda: lifecycle(source, bad, final, committed, None, False, 0.))
    rejects(lambda: lifecycle(source, source, final, committed, None, False, 0.))
    bad = clone(committed); bad[QUERY].fill_(1.)
    rejects(lambda: lifecycle(source, initial, final, bad, None, True, 1/16))
    changed = clone(final); changed[LN_NAMES[0]] += .1
    rejects(lambda: lifecycle(source, initial, changed, committed, None, True, 1/16))
    assert not torch.cuda.is_initialized()
    return dict(status="pass", synthetic_Adam_fits=fits, scalar_checks=checks, corrupted_contracts_rejected=failures,
        source_LN_nonzero=True, exact_empty_and_uniform_rounds=True, CPU_only=True, GT_read=False, model_calls=0)


if __name__ == "__main__":
    import json
    print(json.dumps(selftest(), indent=2))
