"""Independent, read-only admission/association and saved-feedback contracts.

No loader, GT, model, optimizer, selector, or fitter is called. ``selector``
accepts the saved selection dictionary. ``feedback`` accepts the saved fit with
``fixed_evidence`` added by its caller (a shallow wrapper is sufficient), plus
the original expert dictionary. Arrays and tensors are accepted. Precision
checks import existing independent CPU auditors only when feedback is checked.
This is not an independent full-decoder Jacobian or parameter-gradient replay.
"""
import collections
import hashlib
import math
import numpy as np

ARMS = ("H_Top1", "H_Path", "R_Top1", "R_Path", "R_Add", "R_Shuffle")
ACTION_FIELD = "actual_rollout_action_readback"
TOL64 = 1e-12  # The selector's existing double-precision decomposition contract.
U32 = float(np.finfo(np.float32).eps / 2)
TINY32 = float(np.finfo(np.float32).tiny)


def _array(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def _finite32(value):
    value = np.asarray(_array(value), dtype=np.float64)
    assert np.isfinite(value).all()
    assert np.array_equal(value, value.astype(np.float32).astype(np.float64))
    return value


def _close(actual, expected, atol=TOL64):
    actual, expected = np.asarray(_array(actual), float), np.asarray(expected, float)
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    assert np.isfinite(actual).all() and np.isfinite(expected).all()
    error = float(np.max(abs(actual - expected))) if actual.size else 0.
    assert error <= atol, (error, atol)
    return error


def _corners(boxes):
    b = np.asarray(_array(boxes), dtype=np.float64)
    assert b.shape[-1] == 4 and np.isfinite(b).all()
    assert (b[..., 2:] > 0).all()
    return np.concatenate((b[..., :2] - .5 * b[..., 2:],
                           b[..., :2] + .5 * b[..., 2:]), axis=-1)


def _edge(left, right):
    """Independent xyxy intersection and areas; retain the locked 1e-7 floor."""
    a, b = _corners(left)[:, None, :], _corners(right)[None, :, :]
    intersection = np.maximum(0, np.minimum(a[..., 2:], b[..., 2:]) -
                               np.maximum(a[..., :2], b[..., :2])).prod(-1)
    area_a = (a[..., 2:] - a[..., :2]).prod(-1)
    area_b = (b[..., 2:] - b[..., :2]).prod(-1)
    return intersection / np.maximum(area_a + area_b - intersection, 1e-7)


def _paths(sizes):
    """Lexicographic DFS, independent of the fitter's Cartesian-product call."""
    out = []
    def visit(prefix, depth):
        if depth == len(sizes):
            out.append(tuple(prefix))
        else:
            for candidate in range(sizes[depth]):
                visit(prefix + [candidate], depth + 1)
    visit([], 0)
    assert len(out) <= 81
    return out


def _path_values(unary, edges, paths):
    # Two separate sums retain the declared raw score's numerical tie rule.
    return [sum(float(unary[i][k]) for i, k in enumerate(path)) +
            sum(float(edges[i][path[i], path[i + 1]]) for i in range(len(edges)))
            for path in paths]


def _first_max(values):
    return max(range(len(values)), key=lambda i: values[i])


def selector(selection, key):
    """Check six-arm pool/support/source bindings and every <=81 path.

    Stored potentials are independently checked geometrically and algebraically,
    then used to verify the exact numerical tie rule. This avoids demanding equal
    argmax indices from two mathematically equivalent floating-point reductions.
    Both exact ties and score gaps <=1e-12 are recorded, without changing choices.
    """
    pools = selection["pools"]
    assert len(pools) <= 4 and len({int(p["position"]) for p in pools}) == len(pools)
    assert len({int(p["frame_id"]) for p in pools}) == len(pools)
    for p in pools:
        n = len(p["scores"])
        assert n == len(p["boxes"]) == len(p["candidate_ids"]) and n <= 3
        assert len(set(p["candidate_ids"])) == n
        scores = np.asarray(p["scores"], float)
        assert np.isfinite(scores).all() and ((scores >= 0) & (scores <= 1)).all()
        if n:
            assert np.asarray(p["boxes"]).shape == (n, 4)
            _corners(p["boxes"])
        else:
            assert not p["accepted"]
    assert set(selection["arms"]) == set(ARMS)
    assert set(selection["decomposition"]) == {"H", "R"}
    counts = collections.Counter()
    panels = {}
    maxima = collections.defaultdict(float)
    for admission in ("H", "R"):
        support = []
        for p in pools:
            keep = bool(p["accepted"])
            if admission == "R" and p["reason"] == "ambiguous_distinct_instances":
                keep |= bool(p["scores"] and max(p["scores"]) >= .35)
            if keep:
                assert p["scores"]
                support.append(p)
        support.sort(key=lambda p: int(p["frame_id"]))
        positions = [int(p["position"]) for p in support]
        assert positions == sorted(positions)
        scores = [np.asarray(p["scores"], float) for p in support]
        d = selection["decomposition"][admission]
        assert len(d["edges"]) == len(d["centered"]) == len(d["shuffled"]) == len(d["permutations"]) == max(0, len(support) - 1)
        assert len(d["unary"]) == len(support)
        expected_u = [s.copy() for s in scores]
        edges, centered, shuffled = [], [], []
        for i, (left, right) in enumerate(zip(support, support[1:])):
            independent = _edge(left["boxes"], right["boxes"])
            edge = np.asarray(d["edges"][i], float)
            maxima["geometry"] = max(maxima["geometry"], _close(edge, independent))
            nr, nc = edge.shape
            row = np.asarray([math.fsum(map(float, e)) / nc for e in edge])
            col = np.asarray([math.fsum(map(float, edge[:, j])) / nr for j in range(nc)])
            grand = math.fsum(map(float, edge.ravel())) / (nr * nc)
            expected_c = edge - row[:, None] - col[None, :] + grand
            c = np.asarray(d["centered"][i], float)
            maxima["centering"] = max(maxima["centering"], _close(c, expected_c))
            _close(c.sum(0), np.zeros(nc)); _close(c.sum(1), np.zeros(nr))
            expected_u[i] += row - grand / 2
            expected_u[i + 1] += col - grand / 2
            seed = int(hashlib.sha256(f"admission-association-shuffle-v1|{key}|{i}".encode()).hexdigest()[:15], 16)
            shift = 1 + seed % (nc - 1) if nc > 1 else 0
            permutation = [(j - shift) % nc for j in range(nc)]
            assert d["permutations"][i] == permutation
            sh = np.asarray(d["shuffled"][i], float)
            _close(sh, c[:, permutation], 0.)
            _close(np.sort(sh.ravel()), np.sort(c.ravel()), 0.)
            _close(sh.sum(0), np.zeros(nc)); _close(sh.sum(1), np.zeros(nr))
            _close(np.linalg.norm(sh), np.linalg.norm(c))
            if min(nr, nc) == 1:
                _close(c, np.zeros_like(c))
            edges.append(edge); centered.append(c); shuffled.append(sh)
            counts["edge_entries"] += edge.size
        for saved, expected in zip(d["unary"], expected_u):
            maxima["additive_unary"] = max(maxima["additive_unary"], _close(saved, expected))
        u = [np.asarray(v, float) for v in d["unary"]]
        paths = _paths([len(v) for v in scores])
        raw = _path_values(scores, edges, paths)
        rebuilt = _path_values(u, centered, paths)
        maxima["path_decomposition"] = max(maxima["path_decomposition"], _close(raw, rebuilt))
        choices = {"Top1": [int(_first_max(s)) for s in scores],
                   "Path": list(paths[_first_max(raw)]) if support else [],
                   "Add": [int(_first_max(v)) for v in u],
                   "Shuffle": list(paths[_first_max(_path_values(u, shuffled, paths))]) if support else []}
        assert d["choices"] == choices
        degenerate = sum(float(np.max(abs(c))) <= TOL64 for c in centered)
        singleton = sum(min(c.shape) == 1 for c in centered)
        assert d["degenerate_edges"] == degenerate and d["singleton_edges"] == singleton
        for kind in (("Top1", "Path") if admission == "H" else ("Top1", "Path", "Add", "Shuffle")):
            fixed = selection["arms"][admission + "_" + kind]
            assert fixed["GT_assisted"] is False and len(fixed["rows"]) == len(support)
            for p, candidate, saved in zip(support, choices[kind], fixed["rows"]):
                expected = dict(position=p["position"], frame_id=p["frame_id"], candidate_index=candidate,
                    candidate_id=p["candidate_ids"][candidate], box=p["boxes"][candidate],
                    score=p["scores"][candidate], old_admitted=p["accepted"], context_active=p["context_active"])
                assert set(saved) == set(expected)
                _close(saved["box"], expected["box"], 0.)
                assert all(saved[k] == expected[k] for k in expected if k != "box")
                counts["selected_source_rows"] += 1
        maximum = max(raw)
        panels[admission] = dict(positions=positions, paths=len(paths),
            exact_raw_maximizers=sum(v == maximum for v in raw),
            near_raw_maximizers_at_1e_minus12=sum(maximum - v <= TOL64 for v in raw),
            unary_exact_top_ties=[sum(v == max(s) for v in s) for s in scores],
            degenerate_edges=degenerate, singleton_edges=singleton)
        counts["enumerated_paths"] += len(paths)
    assert set(panels["H"]["positions"]) <= set(panels["R"]["positions"])
    return dict(status="pass", checks=dict(counts), max_absolute_errors=dict(maxima), supports=panels,
                CPU_only=True, GT_read=False, model_calls=0, independent_full_decoder_Jacobian=False)


def _gamma(n):
    return n * U32 / (1 - n * U32)


def _bounded_reduction(saved, expected, bound, name):
    saved, expected, bound = map(lambda x: np.asarray(_array(x), float), (saved, expected, bound))
    assert saved.shape == expected.shape == bound.shape, name
    assert np.isfinite(saved).all() and np.isfinite(expected).all() and np.isfinite(bound).all()
    error = abs(saved - expected)
    assert (error <= bound).all(), (name, float(error.max()), float(bound.max()))
    return dict(max_absolute_error=float(error.max()) if error.size else 0.,
                max_derived_bound=float(bound.max()) if bound.size else 0.)


def feedback(fit, expert):
    """Recheck saved actual actions, fixed teachers, and per-round feedback.

    Caller supplies ``dict(fit, fixed_evidence=payload['fixed_evidence'])``.
    Geometry and both softmaxes use existing independent precision auditors:
    the unchanged 3e-6 matched-action and 3e-7 matched-softmax checks plus their
    existing interval/sensitivity bounds. Diagnostic reductions have explicit
    float32 operation bounds; no observed-error-based threshold is fitted.
    """
    from vg_tta.decota_spatial_opd_action_readback_precision005 import check as action_check
    from vg_tta.decota_spatial_opd_precision_audit_revision002 import softmax_check
    cfg, fixed = fit["config"], fit["fixed_evidence"]
    assert cfg["samples"] == 32 and cfg["tau"] > 0 and fit["arm"] == "on_policy"
    positions = [int(p) for p in fit["positions"]]
    assert positions == [int(r["position"]) for r in fixed["rows"]]
    assert len(set(positions)) == len(positions)
    metadata = fit["frame_metadata"]
    assert len(metadata) == len(positions)
    lookup = {}
    for (_, pos), observation in expert["observations"].items():
        assert int(pos) not in lookup
        probe = observation["probe"]
        boxes = _finite32(probe["boxes"])
        scores = np.asarray(_array(probe["target_scores"]), np.float32).astype(float)
        assert boxes.shape == (len(scores), 4) and len(scores) <= 3
        if len(boxes): _corners(boxes)
        lookup[int(pos)] = (boxes, scores, list(map(int, probe["candidate_ids"])), probe, observation["receipt"])
    selected, top1 = [], []
    for row, meta in zip(fixed["rows"], metadata):
        boxes, scores, ids, probe, receipt = lookup[int(row["position"])]
        candidate = int(row["candidate_index"])
        assert 0 <= candidate < len(boxes) and ids[candidate] == row["candidate_id"]
        _close(boxes[candidate], row["box"], 0.)
        assert scores[candidate] == row["score"]
        assert row["old_admitted"] == probe["accepted"] == meta["old_admitted"]
        assert row["context_active"] == receipt["context_active"] == meta["context_active"]
        assert meta["position"] == row["position"] and meta["proposals"] == 1
        assert meta["weights"] == [1.] and meta["scores"] == [row["score"]] and meta["evidence_entropy"] == 0.
        selected.append(boxes[candidate]); top1.append(boxes[_first_max(scores)])
    rounds, geometry = fit["rounds"], fit["geometry"]["rounds"]
    traces = fit[ACTION_FIELD]["trace"]
    assert len(rounds) == cfg["steps"]
    if not positions:
        assert fit["empty"] and not geometry and not traces
        assert all(not r["updated"] and r["no_information"] and "rollout" not in r for r in rounds)
        return dict(status="pass", rounds=0, empty=True, uniform_observation_rounds=0,
                    CPU_only=True, GT_read=False, independent_full_decoder_Jacobian=False)
    assert not fit["empty"] and len(geometry) == len(traces) == len(rounds)
    selected, top1 = np.asarray(selected), np.asarray(top1)
    checks, reductions, uniform_count = [], [], 0
    for k, (round_, row, trace) in enumerate(zip(rounds, geometry, traces)):
        assert round_["round"] == row["round"] == trace["round"] == k
        assert trace["same_original_GPU_sigmoid_call"] and trace["original_action_returned_without_modification"]
        rr, saved = round_["rollout"], row["feedback"]
        samples, actions = _finite32(trace["samples"]), _finite32(trace["actions"])
        _close(samples, _array(rr["samples"]), 0.)
        assert samples.shape == actions.shape == (len(positions), 32, 4)
        reward, weights = _finite32(rr["rewards"]), _finite32(rr["weights"])
        shadow_r, shadow_w = _finite32(saved["same_samples_top1_rewards"]), _finite32(saved["same_samples_top1_weights"])
        assert rr["permutations"] is None
        _close(rr["used_rewards"], reward, 0.)
        precision = dict(round=k,
            selected_reward=action_check(samples, selected, reward, actions),
            shadow_Top1_reward=action_check(samples, top1, shadow_r, actions),
            selected_softmax=softmax_check(reward, weights, cfg["tau"]),
            shadow_Top1_softmax=softmax_check(shadow_r, shadow_w, cfg["tau"]))
        # No arithmetic in max; checked rewards retain exact stored float32 values.
        _close(saved["max_reward"], reward.max(1), 0.)
        variance = ((reward - reward.mean(1, keepdims=True)) ** 2).mean(1)
        # At most 256 float32 scalar operations per 32-value Welford reduction;
        # use its uncentered second moment to cover cancellation near uniformity.
        var_bound = _gamma(256) * np.square(reward).mean(1) + 256 * TINY32
        var_bound += 64 * np.finfo(float).eps * np.maximum(1., variance)
        var_check = _bounded_reduction(saved["reward_variance"], variance, var_bound, "population_variance")
        l1 = abs(weights - shadow_w).sum(1)
        # One subtraction per pair and <=31 nonnegative additions, bounded from
        # the saved weights already certified by the two softmax precision checks.
        l1_bound = _gamma(33) * (abs(weights) + abs(shadow_w)).sum(1) + 64 * TINY32
        l1_check = _bounded_reduction(saved["Top1_same_samples_weight_L1"], l1, l1_bound, "weight_L1")
        gradient = _finite32(round_["autograd_mean_gradient"])
        assert gradient.shape == (len(positions), 4)
        norms = np.sqrt(np.square(gradient).sum(1))
        norm_check = _bounded_reduction(saved["mean_gradient_norm"], norms,
                         _gamma(16) * norms + 16 * TINY32, "mean_gradient_norm")
        informative = (reward != reward[:, :1]).any(1)
        assert np.array_equal(_array(saved["per_frame_informative"]), informative)
        assert rr["informative"] == bool(informative.any())
        assert round_["updated"] == bool(informative.any())
        assert round_["no_information"] == (not informative.any())
        _close(round_["reward_variance"], saved["reward_variance"], 0.)
        _close(round_["mean_before"], rr["mean"], 0.)
        for i in np.flatnonzero(~informative):
            _close(weights[i], np.full(32, 1 / 32), 0.)
            _close(gradient[i], np.zeros(4), 0.)
            assert _array(saved["mean_gradient_norm"])[i] == 0.
            uniform_count += 1
        if np.array_equal(selected, top1):
            _close(shadow_r, reward, 0.); _close(shadow_w, weights, 0.)
            _close(saved["Top1_same_samples_weight_L1"], np.zeros(len(positions)), 0.)
        checks.append(precision)
        reductions.append(dict(round=k, population_variance=var_check, weight_L1=l1_check, gradient_norm=norm_check))
    return dict(status="pass", rounds=len(rounds), empty=False,
                uniform_observation_rounds=uniform_count, precision_checks=checks,
                diagnostic_reduction_checks=reductions, canonical_fit_GT_used=bool(fit["GT_used"]),
                fixed_evidence_GT_assisted=bool(fixed["GT_assisted"]), CPU_only=True, GT_read=False,
                independent_full_decoder_Jacobian=False, parameter_gradients_independently_replayed=False,
                fitting_rewards_weights_gradients_or_Adam_changed=False)
