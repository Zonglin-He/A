"""Minimum instance-coordinate correction for <=2 accepted preferences.

Pure NumPy; no model, GPU, task labels, data access, or training. This is a
same-information *output-only control*, not a TTA optimizer and not identity gold.

Each distinct, disjoint query group is one instance hypothesis. Its source score
is the maximum source log-odds in that group. We minimize 1/2 sum_g delta_g**2
subject to positive-group minus negative-group score >= margin. Repeated query
members do not increase a group's objective weight. A uniform group shift keeps
its original source representative fixed. Queries outside every supplied group
have zero raw correction; they are not assigned negative labels and may still win.

For transfer through a query-feature memory, store ``raw_query_correction``.
Retrieve those values for a new observation, then subtract the retrieved
correction at that NEW observation's source q0. Do not transfer old-observation
q0 offsets as if they were candidate evidence. The returned old-observation
``relative_query_correction`` is useful for direct-A scoring/diagnostics only.

Candidate-wise gates must be applied AFTER anchoring both the source scores and
the retrieved correction to the same source q0. ``apply_anchored_correction`` does
this. Source-score/common-correction translations cannot manufacture preferences.

The chosen positive margin is an explicit control convention (default .1 in
log-odds units), not a calibrated probability or a promise of VLM correctness.
It must be fixed before confirmation or selected only on declared development
inputs. Constraints cover only their endpoints, never every other candidate.
"""
from itertools import combinations
import numpy as np


def _scores_1d(values, name):
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or len(x) == 0 or not np.isfinite(x).all():
        raise ValueError(name + ' must be a nonempty finite 1D array')
    return x.copy()


def _source_q0(source, q0):
    if q0 is None:
        return int(np.argmax(source))
    if isinstance(q0, (bool, np.bool_)) or int(q0) != q0:
        raise ValueError('q0 must be an integer query index')
    q0 = int(q0)
    if not 0 <= q0 < len(source):
        raise ValueError('q0 outside candidate bank')
    return q0


def _canonical_group(indices, n):
    x = np.asarray(indices)
    if x.ndim != 1 or not len(x) or x.dtype.kind == 'b':
        raise ValueError('each group must be a nonempty 1D sequence of indices')
    if x.dtype.kind not in 'iu' and (
        x.dtype.kind != 'f' or not np.isfinite(x).all() or np.any(x != np.floor(x))
    ):
        raise ValueError('query group entries must be integer indices')
    x = np.unique(x.astype(np.int64))
    if x[0] < 0 or x[-1] >= n:
        raise ValueError('group query index outside candidate bank')
    return tuple(map(int, x))


def _project_minimum_l2(A, required, atol):
    """Project zero onto A @ delta >= required by enumerating <=4 active sets."""
    m, dimension = A.shape
    candidates = []
    scale = 1.0 + (float(np.max(np.abs(required))) if m else 0.0)
    tol = atol * scale
    for k in range(m + 1):
        for active in combinations(range(m), k):
            dual = np.zeros(m, dtype=np.float64)
            if active:
                ix = np.asarray(active, dtype=int)
                Ai = A[ix]
                gram = Ai @ Ai.T
                lam, _, _, _ = np.linalg.lstsq(gram, required[ix], rcond=None)
                # Singular duplicate constraints are valid only if consistent.
                if np.max(np.abs(gram @ lam - required[ix])) > tol:
                    continue
                if np.min(lam) < -tol:
                    continue
                dual[ix] = np.maximum(lam, 0.0)
            delta = A.T @ dual
            slack = A @ delta - required
            if m and np.min(slack) < -tol:
                continue
            complementarity = float(np.max(np.abs(dual * slack))) if m else 0.0
            if complementarity > tol * (1.0 + np.linalg.norm(dual)):
                continue
            objective = float(0.5 * delta @ delta)
            candidates.append((objective, len(active), active, delta, dual, slack))
    if not candidates:
        raise ValueError('infeasible preference constraints; no label/constraint was silently dropped')
    # The primal projection is unique. Deterministic dual choice handles redundancy.
    best = min(candidates, key=lambda x: (round(x[0], 13), x[1], x[2]))
    objective, _, active, delta, dual, slack = best
    stationarity = float(np.max(np.abs(delta - A.T @ dual))) if dimension else 0.0
    complementarity = float(np.max(np.abs(dual * slack))) if m else 0.0
    return delta, dual, slack, active, objective, max(stationarity, complementarity)


def project_group_preferences(source_scores, preferences, *, margin=0.1, q0=None, representative_scores=None, atol=1e-10):
    """Return a minimum-L2 group-score projection and both raw/anchored shifts.

    Parameters
    ----------
    source_scores: finite vector of source log-odds for all candidates.
    preferences: zero, one, or two (positive_query_indices, negative_query_indices)
        pairs. Identical groups are canonicalized; partially overlapping distinct
        groups raise an error rather than asserting they are the same instance.
    margin: nonnegative scalar, or a vector with one entry per supplied preference.
    q0: optional FIXED official source winner; caller supplies its cached index.
        It is not reselected after an update. We do not assert it is the exact
        floating-point maximum of log-odds: official float32 probabilities can
        tie while their numerically recomputed log-odds differ.
    representative_scores: optional original source scores used ONLY to choose
        the fixed representative inside each group (e.g. official probabilities).
        Group constraint coordinates always use max(source log-odds), independent
        of quantized probability ties. Defaults to the supplied source log-odds.
    atol: numerical KKT feasibility tolerance, not a feedback acceptance threshold.

    No IoU, GT, confidence selection, or extra candidate negatives are used.
    """
    source = _scores_1d(source_scores, 'source_scores')
    if not np.isfinite(atol) or atol <= 0:
        raise ValueError('atol must be finite and positive')
    q0 = _source_q0(source, q0)
    preferences = list(preferences)
    if len(preferences) > 2:
        raise ValueError('this explicit active-set control supports at most two preferences')
    groups = []
    group_lookup = {}
    endpoints = []
    for pair in preferences:
        if len(pair) != 2:
            raise ValueError('a preference must contain exactly positive and negative groups')
        ids = []
        for indices in pair:
            g = _canonical_group(indices, len(source))
            if g not in group_lookup:
                gs = set(g)
                if any(gs.intersection(old) for old in groups):
                    raise ValueError('partially overlapping distinct groups are ambiguous; do not merge them')
                group_lookup[g] = len(groups)
                groups.append(g)
            ids.append(group_lookup[g])
        endpoints.append(tuple(ids))
    margins = np.asarray(margin, dtype=np.float64)
    if margins.ndim == 0:
        margins = np.full(len(preferences), float(margins), dtype=np.float64)
    if margins.shape != (len(preferences),) or not np.isfinite(margins).all() or np.any(margins < 0):
        raise ValueError('margin must be finite, nonnegative, and scalar or one-per-preference')
    rep_reference = source if representative_scores is None else _scores_1d(representative_scores, 'representative_scores')
    if rep_reference.shape != source.shape:
        raise ValueError('representative_scores has different candidate dimensions')
    reps = np.asarray([g[int(np.argmax(rep_reference[list(g)]))] for g in groups], dtype=np.int64)
    group_source = np.asarray([float(np.max(source[list(g)])) for g in groups], dtype=np.float64)
    A = np.zeros((len(preferences), len(groups)), dtype=np.float64)
    for j, (positive, negative) in enumerate(endpoints):
        A[j, positive] += 1.0
        A[j, negative] -= 1.0
    required = margins - A @ group_source
    shifts, dual, slack, active, objective, kkt_error = _project_minimum_l2(A, required, atol)
    raw = np.zeros_like(source)
    query_to_group = np.full(len(source), -1, dtype=np.int64)
    for j, g in enumerate(groups):
        raw[list(g)] = shifts[j]
        query_to_group[list(g)] = j
    projected = source + raw
    source_relative = source - source[q0]
    projected_relative = projected - projected[q0]
    relative_correction = raw - raw[q0]
    unassigned = query_to_group < 0
    if not np.array_equal(projected[unassigned], source[unassigned]):
        raise AssertionError('unassigned source candidates must remain unchanged in raw coordinates')
    if len(preferences) and np.min(A @ (group_source + shifts) - margins) < -atol * (1.0 + np.max(np.abs(required))):
        raise AssertionError('projected constraints not satisfied')
    return {
        'q0': q0,
        'groups': tuple(groups),
        'query_to_group': query_to_group,
        'unassigned_queries': np.flatnonzero(unassigned),
        'representative_queries': reps,
        'group_source_scores': group_source,
        'group_score_correction': shifts,
        'group_projected_scores': group_source + shifts,
        'raw_query_correction': raw,
        'relative_query_correction': relative_correction,
        'source_relative_scores': source_relative,
        'projected_scores': projected,
        'projected_relative_scores': projected_relative,
        'constraint_matrix': A,
        'constraint_margins': margins,
        'margins_before': A @ group_source,
        'margins_after': A @ (group_source + shifts),
        'active_constraints': tuple(active),
        'dual_multipliers': dual,
        'constraint_slack': slack,
        'objective_half_group_l2': objective,
        'kkt_max_abs_error': kkt_error,
    }


def anchor_retrieved_correction(raw_query_correction, source_q0):
    """Anchor retrieved RAW shifts to the new observation's fixed source q0."""
    c = _scores_1d(raw_query_correction, 'raw_query_correction')
    if isinstance(source_q0, (bool, np.bool_)) or int(source_q0) != source_q0 or not 0 <= int(source_q0) < len(c):
        raise ValueError('source_q0 must be a valid query index')
    return c - c[int(source_q0)]


def apply_anchored_correction(source_scores, raw_query_correction, *, gate=None, q0=None):
    """Apply the SAME candidate gate as a learned-readout control, after anchoring.

    Returns relative output scores. It is invariant to separate common offsets
    in source scores and retrieved raw corrections, including with nonuniform
    gates. The source q0 is exactly zero; this is not a class or identity label.
    """
    source = _scores_1d(source_scores, 'source_scores')
    raw = _scores_1d(raw_query_correction, 'raw_query_correction')
    if raw.shape != source.shape:
        raise ValueError('source and correction candidate dimensions differ')
    q0 = _source_q0(source, q0)
    g = np.ones_like(source) if gate is None else np.asarray(gate, dtype=np.float64)
    if g.ndim == 0:
        g = np.full_like(source, float(g))
    if g.shape != source.shape or not np.isfinite(g).all() or np.any(g < 0) or np.any(g > 1):
        raise ValueError('gate must be scalar or candidate vector in [0,1]')
    return (source - source[q0]) + g * anchor_retrieved_correction(raw, q0)


def synthetic_checks():
    """Small mathematical checks only; no real observations or task evaluation."""
    passed = []
    x = project_group_preferences([2., 0., .3], [([1], [0])], margin=1.)
    np.testing.assert_allclose(x['raw_query_correction'], [-1.5, 1.5, 0.], atol=1e-12)
    np.testing.assert_allclose(x['margins_after'], [1.], atol=1e-12)
    assert x['representative_queries'].tolist() == [1, 0]
    passed.append('single_constraint_closed_form_and_unknown_unchanged')
    chain = project_group_preferences([2., 1., 0.], [([2], [0]), ([0], [1])], margin=1.)
    np.testing.assert_allclose(chain['projected_scores'], [1., 0., 2.], atol=1e-12)
    np.testing.assert_allclose(chain['margins_after'], [1., 1.], atol=1e-12)
    passed.append('two_constraint_chain_joint_projection')
    inactive = project_group_preferences([1., 3., .2], [([1], [0])], margin=1.)
    np.testing.assert_array_equal(inactive['raw_query_correction'], np.zeros(3))
    passed.append('already_satisfied_constraint_no_edit')
    redundant = project_group_preferences([2., 0., .3], [([1], [0]), ([1], [0])], margin=1.)
    np.testing.assert_allclose(redundant['raw_query_correction'], x['raw_query_correction'], atol=1e-12)
    passed.append('redundant_singular_constraints')
    dup = project_group_preferences([2., 0., 0., .3], [([1, 2], [0])], margin=1.)
    np.testing.assert_allclose(dup['group_score_correction'], x['group_score_correction'], atol=1e-12)
    np.testing.assert_allclose(dup['raw_query_correction'], [-1.5, 1.5, 1.5, 0.], atol=1e-12)
    assert dup['representative_queries'][0] == 1
    passed.append('repeated_query_count_does_not_weight_instance')
    tied = project_group_preferences([2., .01, .02], [([1, 2], [0])], margin=.1, q0=0, representative_scores=[.99, .5, .5])
    assert tied['representative_queries'][0] == 1
    assert tied['group_source_scores'][0] == .02
    passed.append('official_representative_ties_separate_from_group_coordinate')
    shifted = project_group_preferences(np.array([2., 0., .3]) + 123., [([1], [0])], margin=1.)
    np.testing.assert_allclose(shifted['raw_query_correction'], x['raw_query_correction'], atol=1e-12)
    np.testing.assert_allclose(shifted['projected_relative_scores'], x['projected_relative_scores'], atol=1e-12)
    passed.append('source_common_shift_invariance')
    s = np.array([.7, .6, .5]); c = np.array([.4, -.2, .1]); gate = np.array([0., 1., .5])
    out = apply_anchored_correction(s, c, gate=gate)
    np.testing.assert_allclose(out, apply_anchored_correction(s + 7., c + 19., gate=gate), atol=1e-12)
    np.testing.assert_allclose(apply_anchored_correction(s, np.full(3, 9.), gate=gate), s-s[0], atol=1e-12)
    passed.append('separate_source_correction_gauges_and_nonuniform_gate')
    np.testing.assert_allclose(apply_anchored_correction([2., 0., .3], x['raw_query_correction']), x['projected_relative_scores'], atol=1e-12)
    passed.append('direct_projection_equals_allgate_anchored_control')
    outside = project_group_preferences([3., 0., 2.9], [([1], [0])], margin=.1)
    assert int(np.argmax(outside['projected_scores'])) == 2
    passed.append('unmentioned_candidate_may_win_no_implicit_negative')
    empty = project_group_preferences([1., .5], [], margin=.1)
    np.testing.assert_array_equal(empty['projected_scores'], [1., .5])
    passed.append('empty_preferences_source_preserved')
    for prefs, reason in [([([0], [1]), ([1], [0])], 'contradictory_positive_cycle'), ([([0, 1], [1, 2])], 'partially_overlapping_identity_groups')]:
        try:
            project_group_preferences([2., 1., 0.], prefs, margin=1.)
        except ValueError:
            passed.append(reason + '_explicitly_rejected')
        else:
            raise AssertionError(reason + ' was silently accepted')
    return {'state': 'PASSED', 'synthetic_only': True, 'checks': passed}


if __name__ == '__main__':
    import json
    print(json.dumps(synthetic_checks(), indent=2))
