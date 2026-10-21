"""Label-gated, one-time HC64 diagnostics for the propagation v0 gate.

Pure helpers below do not read assets. ``diagnose`` is wired only after the
stage-one runner schema is finalized; it never constructs or updates a model.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 20261021
TIE_ATOL = 1e-12


def average_ranks(values):
    """One-based average ranks, with exact-value ties and stable ordering."""
    values = np.asarray(values, dtype=float)
    assert values.ndim == 1 and np.isfinite(values).all()
    order = np.argsort(values, kind='stable')
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2
        start = end
    return ranks


def spearman(scores, truth):
    """Return None when fewer than two candidates or either rank is constant."""
    scores, truth = np.asarray(scores, float), np.asarray(truth, float)
    assert scores.shape == truth.shape and scores.ndim == 1
    if len(scores) < 2:
        return None
    x, y = average_ranks(scores), average_ranks(truth)
    x, y = x - x.mean(), y - y.mean()
    denominator = math.sqrt(float(x @ x) * float(y @ y))
    return None if denominator == 0 else float(np.clip((x @ y) / denominator, -1, 1))


def rank_state(scores, candidate_iou, valid, tie_atol=TIE_ATOL):
    """Rank valid same-state candidates; GT ties excluded, score ties get 0.5."""
    scores, candidate_iou, valid = np.asarray(scores, float), np.asarray(candidate_iou, float), np.asarray(valid, bool)
    assert scores.shape == candidate_iou.shape == valid.shape and scores.ndim == 1
    assert np.isfinite(scores[valid]).all() and np.isfinite(candidate_iou[valid]).all()
    x, y = scores[valid], candidate_iou[valid]
    correct = 0.; comparable = 0; tied_scores = 0; tied_truth = 0
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            dy, dx = y[i] - y[j], x[i] - x[j]
            if abs(dy) <= tie_atol:
                tied_truth += 1
                continue
            comparable += 1
            if abs(dx) <= tie_atol:
                correct += .5
                tied_scores += 1
            elif dx * dy > 0:
                correct += 1
    rho = spearman(x, y)
    return dict(correct_pairs=correct, comparable_pairs=comparable,
                pair_accuracy=correct / comparable if comparable else None,
                tied_score_pairs=tied_scores, GT_tied_pairs=tied_truth,
                Spearman=rho, valid_candidates=len(x))


def query_ranking(states):
    """Retain no-pair/fallback states as coverage, without inventing a score."""
    correct = sum(state['correct_pairs'] for state in states)
    pairs = sum(state['comparable_pairs'] for state in states)
    rhos = [state['Spearman'] for state in states if state['Spearman'] is not None]
    identified = [state for state in states if state['Spearman'] is not None]
    return dict(states=len(states), states_with_pairs=sum(state['comparable_pairs'] > 0 for state in states),
        correct_pairs=correct, comparable_pairs=pairs, pair_accuracy=correct / pairs if pairs else None,
        tied_score_pairs=sum(state['tied_score_pairs'] for state in states),
        GT_tied_pairs=sum(state['GT_tied_pairs'] for state in states),
        Spearman=float(np.mean(rhos)) if rhos else None, states_with_Spearman=len(rhos),
        micro_rank_numerator=sum(state['valid_candidates'] * state['Spearman'] for state in identified),
        micro_rank_denominator=sum(state['valid_candidates'] for state in identified))


def oracle_unobserved(candidate_iou, valid, observed, has_GT):
    """All fixed unobserved GT nodes enter the denominator, including E-only."""
    candidate_iou, valid = np.asarray(candidate_iou, float), np.asarray(valid, bool)
    observed, has_GT = np.asarray(observed, bool), np.asarray(has_GT, bool)
    assert candidate_iou.shape == valid.shape and candidate_iou.ndim == 2
    assert observed.shape == has_GT.shape == candidate_iou.shape[:1] and valid[:, 0].all()
    selected = np.zeros(len(observed), dtype=np.int64)
    nodes = (~observed) & has_GT
    best, baseline, improvements = [], [], []
    for node in np.flatnonzero(nodes):
        slots = np.flatnonzero(valid[node])
        assert np.isfinite(candidate_iou[node, slots]).all()
        choice = int(slots[np.argmax(candidate_iou[node, slots])])
        selected[node] = choice
        base, value = float(candidate_iou[node, 0]), float(candidate_iou[node, choice])
        baseline.append(base); best.append(value); improvements.append(value - base)
    count = int(nodes.sum())
    return dict(nodes=count, E_only_nodes=int((nodes & (valid.sum(-1) == 1)).sum()),
        baseline_mean=float(np.mean(baseline)) if count else None,
        oracle_mean=float(np.mean(best)) if count else None,
        mean_improvement=float(np.mean(improvements)) if count else None,
        over5pp_nodes=int(np.sum(np.asarray(improvements) > .05)),
        over5pp_fraction=float(np.mean(np.asarray(improvements) > .05)) if count else None,
        selected_slots=selected, node_mask=nodes)


def distance_bins(frame_ids, positions8):
    """Fixed physical distance to nearest observation, normalized per query."""
    ids = np.asarray(frame_ids, dtype=float)
    positions = np.asarray(positions8, dtype=np.int64)
    assert ids.ndim == 1 and np.all(np.diff(ids) > 0)
    assert positions.shape == (8,) and len(set(positions.tolist())) == 8
    anchors = np.sort(ids[positions])
    median_gap = float(np.median(np.diff(anchors)))
    assert median_gap > 0
    distance = np.abs(ids[:, None] - anchors[None]).min(-1)
    normalized = distance / median_gap
    return dict(distance=distance, normalized_distance=normalized, median_anchor_gap=median_gap,
                near=(normalized > 0) & (normalized <= .25),
                middle=(normalized > .25) & (normalized <= .5), far=normalized > .5)


def _mean_or_none(values):
    values = [value for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(values)) if values else None


def bootstrap_mean(values, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED):
    """Parent/query bootstrap, retaining missing-parent metric coverage."""
    x = np.asarray([np.nan if value is None else value for value in values], dtype=float)
    valid = np.isfinite(x)
    if not valid.any():
        return dict(mean=None, CI95=None, valid_parents=0, all_parents=len(x))
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(x), (draws, len(x)))
    samples = x[indices]
    count = np.isfinite(samples).sum(-1)
    ratio = np.divide(np.nansum(samples, axis=-1), count, out=np.full(draws, np.nan), where=count > 0)
    return dict(mean=float(x[valid].mean()), CI95=np.nanquantile(ratio, [.025, .975]).tolist(),
                valid_parents=int(valid.sum()), all_parents=len(x))


def bootstrap_ratio(numerator, denominator, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED):
    num, den = np.asarray(numerator, float), np.asarray(denominator, float)
    assert num.shape == den.shape and num.ndim == 1 and (den >= 0).all()
    if den.sum() == 0:
        return dict(mean=None, CI95=None, valid_parents=0, all_parents=len(num))
    indices = np.random.default_rng(seed).integers(0, len(num), (draws, len(num)))
    a, b = num[indices].sum(-1), den[indices].sum(-1)
    ratio = np.divide(a, b, out=np.full(draws, np.nan), where=b > 0)
    return dict(mean=float(num.sum() / den.sum()), CI95=np.nanquantile(ratio, [.025, .975]).tolist(),
                valid_parents=int((den > 0).sum()), all_parents=len(num))


def ranking_summary(queries):
    return dict(
        macro_pair_accuracy=bootstrap_mean([q['pair_accuracy'] for q in queries]),
        micro_pair_accuracy=bootstrap_ratio([q['correct_pairs'] for q in queries], [q['comparable_pairs'] for q in queries]),
        macro_Spearman=bootstrap_mean([q['Spearman'] for q in queries]),
        micro_Spearman=bootstrap_ratio([q['micro_rank_numerator'] for q in queries], [q['micro_rank_denominator'] for q in queries]),
        states=sum(q['states'] for q in queries), states_with_pairs=sum(q['states_with_pairs'] for q in queries),
        states_with_Spearman=sum(q['states_with_Spearman'] for q in queries),
        comparable_pairs=sum(q['comparable_pairs'] for q in queries),
        GT_tied_pairs=sum(q['GT_tied_pairs'] for q in queries), tied_score_pairs=sum(q['tied_score_pairs'] for q in queries),
        micro_Spearman_definition='Correlation of concatenated within-state centered standardized average ranks; equivalent to candidate-count weighted within-state Spearman',
        macro_definition='Each query averages its identifiable node/path Spearman; pair accuracy pools its same-state non-GT-tie pairs; then queries receive equal weight',
        missing_definition='E-only/constant states remain in coverage counts and oracle denominators; unidentified ranking metrics are None, not a fabricated 0.5')


def gate_decision(candidate_improvement, pair_accuracy, rho, memory_v_delta, memory_unobs_delta):
    finite = lambda value: value is not None and math.isfinite(value)
    candidate = finite(candidate_improvement) and candidate_improvement > 0
    feedback = finite(pair_accuracy) and finite(rho) and pair_accuracy > .5 and rho > 0
    memory = (finite(memory_v_delta) and finite(memory_unobs_delta)
              and memory_v_delta >= -1e-12 and memory_unobs_delta >= -1e-12
              and (memory_v_delta > 1e-12 or memory_unobs_delta > 1e-12))
    passed = bool(candidate and feedback and memory)
    return dict(passed=passed, decision='eligible_for_complete_HC64_adaptation' if passed else 'stop_no_adaptation',
        candidate_pass=bool(candidate), feedback_pass=bool(feedback), memory_pass=bool(memory),
        point_estimates=dict(candidate_macro_unobserved_oracle_improvement=candidate_improvement,
            feedback_macro_nonGTtie_pair_accuracy=pair_accuracy, feedback_macro_Spearman=rho,
            MemoryGreedy_minus_NoMemoryGreedy_vIoU=memory_v_delta,
            MemoryGreedy_minus_NoMemoryGreedy_unobserved_sIoU=memory_unobs_delta),
        query_selection=False, confidence_interval_sign_required=False,
        configuration_change_after_GT_permitted=False)


def _safe_phase_path(trial, phase, relative):
    path = (trial / phase / relative).resolve()
    assert path.is_relative_to(trial.resolve()), 'Stage artifact escapes its trial'
    return path


def _validate_diagnostic_paths(bank, trace):
    """All initial histories are checked before any label file is opened."""
    assert bank['GT_read'] is False
    boxes, valid = np.asarray(bank['boxes']), np.asarray(bank['valid'], bool)
    active, E = np.asarray(bank['active'], bool), np.asarray(bank['E'])
    assert boxes.ndim == 3 and boxes.shape[1:] == (3, 4) and valid.shape == boxes.shape[:2]
    assert valid[:, 0].all() and np.array_equal(boxes[:, 0], E)
    assert not valid[~active, 1:].any()
    assert len(trace['diagnostic_paths']) == 2
    initial = trace['initial_student']
    assert np.array_equal(np.asarray(initial['boxes']), E)
    assert not np.asarray(initial['actions']).any(), 'The initial student must retain E'
    for path in trace['diagnostic_paths']:
        effective = np.asarray(path['effective_valid'], bool)
        q, reward = np.asarray(path['teacherq'], float), np.asarray(path['r'], float)
        assert effective.shape == q.shape == reward.shape == valid.shape
        assert np.isfinite(q).all() and np.isfinite(reward).all() and (q >= 0).all()
        assert np.allclose(q.sum(-1), 1, atol=1e-12, rtol=0)
        assert not q[~effective].any() and not effective[~valid].any()
        actions = np.asarray(path['actions'], np.int64)
        assert actions.shape == (len(E),) and effective[np.arange(len(E)), actions].all()
        assert np.array_equal(np.asarray(path['boxes']), boxes[np.arange(len(E)), actions])
        assert np.array_equal(np.asarray(path['boxes'])[~active], E[~active])
        assert np.array_equal(np.asarray(path['memory_before'])[~active], np.asarray(path['memory_after'])[~active])


def _node_ious(boxes, record, truth):
    from stvg.metrics import box_iou
    from stvg.metrics_base import xyxy
    boxes = np.asarray(boxes, float)
    meta = record['metadata']
    pixels = np.maximum(xyxy(boxes, meta['width'], meta['height']), 0)
    shape = boxes.shape[:-1]
    result = np.full(shape, np.nan)
    for node, frame in enumerate(meta['frame_ids']):
        if frame not in truth:
            continue
        if boxes.ndim == 2:
            result[node] = box_iou(pixels[node], truth[frame])
        else:
            result[node] = [box_iou(box, truth[frame]) for box in pixels[node]]
    return result


def _distance_summary(queries, names):
    result = {}
    for group in ('near', 'middle', 'far'):
        group_rows = [query[group] for query in queries]
        delta = [row['MemoryGreedy_minus_NoMemoryGreedy'] for row in group_rows]
        result[group] = dict(
            normalized_physical_distance_interval={'near': '(0,.25]', 'middle': '(.25,.5]', 'far': '>.5'}[group],
            nodes=sum(row['nodes'] for row in group_rows),
            queries_with_nodes=sum(row['nodes'] > 0 for row in group_rows),
            means={name: bootstrap_mean([row['means'][name] for row in group_rows]) for name in names},
            MemoryGreedy_minus_NoMemoryGreedy=bootstrap_mean(delta),
            scope='Fixed sampled unobserved nodes with GT; query macro means, parent bootstrap')
    return result


def _render(public):
    gate = public['gate']
    lines = ['# Propagation v0: pre-adaptation HC64 evidence gate', '',
        'Stage one is complete and sealed. No matching-weight adaptation has been run in this diagnostic branch. '
        'The same fixed 64 development parents, one query per parent, are retained. GT is read only after '
        'every stage-one prediction, input, bank, evidence and initial path has passed its hash and schema checks.', '',
        f'**Gate: {gate["decision"]}.** The predeclared point-estimate criteria are applied once to the whole panel; '
        'confidence-interval signs are reported but are not extra pass criteria. No per-query selection is made.', '',
        '| Criterion | Point estimate | Pass |', '|---|---:|:---:|']
    point = gate['point_estimates']
    rows = [('Candidate mean unobserved oracle IoU improvement', point['candidate_macro_unobserved_oracle_improvement'], gate['candidate_pass']),
            ('Same-state feedback macro non-GT-tie pair accuracy', point['feedback_macro_nonGTtie_pair_accuracy'], gate['feedback_pass']),
            ('Same-state feedback macro Spearman', point['feedback_macro_Spearman'], gate['feedback_pass']),
            ('MemoryGreedy − NoMemoryGreedy full vIoU', point['MemoryGreedy_minus_NoMemoryGreedy_vIoU'], gate['memory_pass']),
            ('MemoryGreedy − NoMemoryGreedy unobserved sIoU', point['MemoryGreedy_minus_NoMemoryGreedy_unobserved_sIoU'], gate['memory_pass'])]
    for name, value, passed in rows:
        formatted = 'NA' if value is None else f'{value:.8f}'
        lines.append(f'| {name} | {formatted} | {passed} |')
    lines += ['', '## Saved control predictions', '',
        '| Method | Full vIoU % | Observed sIoU % | Unobserved sIoU % |', '|---|---:|---:|---:|']
    for name in public['arm_order']:
        mean = public['means'][name]
        formatted = ['NA' if mean[field] is None else f'{100*mean[field]:.4f}'
                     for field in ['m_vIoU', 'm_sIoU_observed_actual', 'm_sIoU_unobserved_actual']]
        lines.append('| ' + name + ' | ' + ' | '.join(formatted) + ' |')
    pair = public['memory_comparison']
    lines += ['', '| Memory comparison | Delta pp | 95% parent interval pp |', '|---|---:|---:|']
    for field in ('v', 'unobserved_actual_sIoU'):
        row = pair[field]
        delta = 'NA' if row['mean_pp'] is None else f'{row["mean_pp"]:+.6f}'
        ci = 'NA' if row['CI95_pp'] is None else f'[{row["CI95_pp"][0]:.6f}, {row["CI95_pp"][1]:.6f}]'
        lines.append(f'| {field} | {delta} | {ci} |')
    lines += ['', '## Candidate and feedback checks', '', public['candidate_oracle']['definition'], '',
        '| Feedback quantity | Macro estimate [95% CI] | Micro estimate [95% CI] |', '|---|---:|---:|']
    def interval(row):
        return 'NA' if row['mean'] is None else f'{row["mean"]:.6f} [{row["CI95"][0]:.6f}, {row["CI95"][1]:.6f}]'
    for name in ('r', 'teacherq'):
        ranking = public['feedback_ranking'][name]
        lines.append('| ' + name + ' pair accuracy | ' + interval(ranking['macro_pair_accuracy']) + ' | ' + interval(ranking['micro_pair_accuracy']) + ' |')
        lines.append('| ' + name + ' Spearman | ' + interval(ranking['macro_Spearman']) + ' | ' + interval(ranking['micro_Spearman']) + ' |')
    lines += ['', public['feedback_ranking']['r']['macro_definition'],
        public['feedback_ranking']['r']['micro_Spearman_definition'],
        public['feedback_ranking']['r']['missing_definition'], '',
        '## Fixed physical-distance bins', '',
        'Distance is to the nearest expert observation, divided by the median adjacent gap of the sorted eight '
        'physical anchor frames. The bins were fixed before GT: (0,.25], (.25,.5], and >.5. Observation frames '
        'are separate and are excluded here.', '',
        '| Distance bin | Fixed GT nodes | Valid queries | Memory − NoMemory sIoU pp [95% CI] |', '|---|---:|---:|---:|']
    for group in ('near', 'middle', 'far'):
        row = public['distance'][group]
        difference = row['MemoryGreedy_minus_NoMemoryGreedy']
        text = 'NA' if difference['mean'] is None else f'{100*difference["mean"]:+.6f} [{100*difference["CI95"][0]:.6f}, {100*difference["CI95"][1]:.6f}]'
        lines.append(f'| {row["normalized_physical_distance_interval"]} | {row["nodes"]} | {row["queries_with_nodes"]} | {text} |')
    lines += ['', 'All positive, negative, unchanged, E-only fallback and unavailable-ranking queries remain '
        'in the anonymous JSON. GT selects candidates only for the stated oracle diagnostic, never for an '
        'online prediction or memory update. Subsequent adaptation, if the gate passes, remains a development '
        'experiment on the exposed panel. A passed gate does not establish OPD superiority over Direct.']
    return '\n'.join(lines) + '\n'


def diagnose(trial):
    """Validate all64, read GT once, write one immutable global gate result."""
    import torch
    from scripts.propagation_v0.run import validate_phase
    from scripts.state_opd.evaluate import read, sha, write_json, score_boxes, mean_table, paired_stats

    trial = Path(trial).resolve()
    assert trial == (ROOT / 'outputs/propagation_v0_hc64').resolve()
    lock = read(trial / 'lock.json')
    receipts = validate_phase(trial, 'stage1', lock)
    phase = trial / 'stage1'
    seal_path = phase / 'sealed.json'
    seal_sha = sha(seal_path)
    manifest = ROOT / 'data/evaluation/development.jsonl'
    assert sha(manifest) == lock['manifest_sha256']
    records = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    records = [row for row in records if row['dataset'] == 'hc2']
    assert len(records) == 128
    records = records[:64]
    assert [row['id'] for row in records] == lock['ids']
    seal = read(seal_path)
    assert len(receipts) == len(seal['files']) == 64
    assets = {}
    old = ROOT / 'outputs/layout_reward_dev'
    for record, item in zip(records, lock['inputs']):
        key = record['id']; receipt = receipts[key]
        bank = torch.load(_safe_phase_path(trial, 'stage1', receipt['bank_path']), map_location='cpu', weights_only=False)
        trace = torch.load(_safe_phase_path(trial, 'stage1', receipt['trace_path']), map_location='cpu', weights_only=False)
        _validate_diagnostic_paths(bank, trace)
        with (np.load(_safe_phase_path(trial, 'stage1', seal['files'][key]['path']), allow_pickle=False) as saved,
              np.load(old / item['reference_path'], allow_pickle=False) as reference):
            predictions = {name: saved[name].copy() for name in saved.files}
            assert predictions['frame_ids'].tolist() == record['metadata']['frame_ids']
            for name, old_name in [('SourceStrong', 'Source'), ('ERefStrong', 'E_Ref'), ('GeometryOPD', 'GeometryOPD')]:
                assert np.array_equal(predictions[name], reference[old_name])
        assets[key] = (bank, trace, predictions)
    assert not torch.cuda.is_initialized()
    labels_path = ROOT / 'data/evaluation/development_labels.jsonl'
    output = ROOT / 'results/propagation_v0_hc64'
    gate_path = trial / 'gate.json'
    if gate_path.exists():
        gate = read(gate_path)
        assert gate['fingerprint'] == lock['fingerprint'] and gate['stage1_seal_sha256'] == seal_sha
        assert gate['labels_sha256'] == sha(labels_path)
        assert gate['diagnostics_sha256'] == sha(output / 'diagnostics.json')
        return gate
    # This is the first GT/scored-result access. All64 assets passed above.
    labels_sha = sha(labels_path)
    old_metrics = read(old / 'scope_hc2/metrics.json')
    assert labels_sha == old_metrics['labels_sha256']
    selected = set(lock['ids'])
    labels = {row['id']: row for row in [json.loads(line) for line in labels_path.read_text().splitlines() if line.strip()]
              if row['dataset'] == 'hc2' and row['id'] in selected}
    old_rows = {row['id']: row for row in old_metrics['rows'] if row['id'] in selected}
    assert set(labels) == set(old_rows) == selected
    names = ['ERefStrong', 'GeometryOPD', 'MemoryGreedy', 'NoMemoryGreedy', 'EvidenceDP']
    rows, distance_queries, rank_queries, oracle_queries = [], [], {'r': [], 'teacherq': []}, []
    baseline_error = dense_error = 0.; checks = 0
    for record in records:
        key = record['id']; label = labels[key]
        bank, trace, prediction = assets[key]
        truth = {int(frame): box for frame, box in label['boxes'].items()}
        positions = prediction['positions8'].tolist()
        interval = prediction['Strong_interval'].tolist()
        values = {}
        for name in names:
            values[name], error, support = score_boxes(prediction[name], record, truth, interval, label['interval'], positions)
            dense_error = max(dense_error, error); checks += 1
        for name, prior_name in [('ERefStrong', 'WithoutOPD'), ('GeometryOPD', 'GeometryOPD')]:
            prior = old_rows[key]['values'][prior_name]
            for field in ('v', 't', 's', 'observed_actual_sIoU', 'unobserved_actual_sIoU'):
                if prior[field] is None:
                    assert values[name][field] is None
                else:
                    baseline_error = max(baseline_error, abs(values[name][field] - prior[field]))
        observed = np.asarray(bank['observed'], bool)
        candidate_iou = _node_ious(bank['boxes'], record, truth)
        has_GT = np.asarray([frame in truth for frame in record['metadata']['frame_ids']])
        oracle = oracle_unobserved(candidate_iou, bank['valid'], observed, has_GT)
        selected_boxes = np.asarray(bank['boxes'])[np.arange(len(observed)), oracle['selected_slots']]
        oracle_values, error, _ = score_boxes(selected_boxes, record, truth, interval, label['interval'], positions)
        dense_error = max(dense_error, error); checks += 1
        oracle_public = {k: v for k, v in oracle.items() if k not in ('selected_slots', 'node_mask')}
        oracle_public['GT_nodewise_candidate_readout_vIoU'] = oracle_values['v']
        oracle_queries.append(oracle_public)
        query_rank = {}
        for field in ('r', 'teacherq'):
            states = []
            for path in trace['diagnostic_paths']:
                for node in np.flatnonzero(oracle['node_mask']):
                    states.append(rank_state(np.asarray(path[field])[node], candidate_iou[node],
                                             np.asarray(path['effective_valid'])[node]))
            summary = query_ranking(states)
            rank_queries[field].append(summary); query_rank[field] = summary
        bins = distance_bins(prediction['frame_ids'], positions)
        node_scores = {name: _node_ious(prediction[name], record, truth) for name in names}
        distance_query = {}
        for group in ('near', 'middle', 'far'):
            node_mask = oracle['node_mask'] & bins[group]
            count = int(node_mask.sum())
            distance_query[group] = dict(nodes=count,
                means={name: float(node_scores[name][node_mask].mean()) if count else None for name in names},
                MemoryGreedy_minus_NoMemoryGreedy=float((node_scores['MemoryGreedy'][node_mask] - node_scores['NoMemoryGreedy'][node_mask]).mean()) if count else None)
        distance_queries.append(distance_query)
        rows.append(dict(id=key, parent=label['parent'], values=values, support=support,
                         candidate_oracle=oracle_public, feedback_ranking=query_rank, distance=distance_query))
    assert len(rows) == len({row['parent'] for row in rows}) == 64
    assert baseline_error < 2e-10 and dense_error < 2e-10
    means, metric_counts = mean_table(rows, names)
    comparisons = paired_stats(rows, names)
    memory_pair = comparisons['MemoryGreedy-NoMemoryGreedy']
    ranking = {field: ranking_summary(rank_queries[field]) for field in ('r', 'teacherq')}
    candidate = dict(
        macro_unobserved_IoU_improvement=bootstrap_mean([row['mean_improvement'] for row in oracle_queries]),
        macro_over5pp_node_fraction=bootstrap_mean([row['over5pp_fraction'] for row in oracle_queries]),
        unobserved_GT_nodes=sum(row['nodes'] for row in oracle_queries),
        E_only_fallback_nodes=sum(row['E_only_nodes'] for row in oracle_queries),
        over5pp_nodes=sum(row['over5pp_nodes'] for row in oracle_queries),
        macro_GT_nodewise_candidate_readout_vIoU=_mean_or_none([row['GT_nodewise_candidate_readout_vIoU'] for row in oracle_queries]),
        definition='Every fixed unobserved node with GT remains in the oracle denominator, including E-only fallback. '
            'GT chooses the best valid candidate independently at those nodes; all other nodes retain E. '
            'The resulting interpolated full vIoU is an offline readout, not a method and not the exact global trajectory upper bound.',
        GT_selected=True, used_for_online_predictions=False)
    memory_v = memory_pair['v']['mean_pp']
    memory_s = memory_pair['unobserved_actual_sIoU']['mean_pp']
    gate = gate_decision(candidate['macro_unobserved_IoU_improvement']['mean'],
                         ranking['r']['macro_pair_accuracy']['mean'], ranking['r']['macro_Spearman']['mean'],
                         None if memory_v is None else memory_v / 100,
                         None if memory_s is None else memory_s / 100)
    assert sha(labels_path) == labels_sha and sha(seal_path) == seal_sha
    # Recheck all64 outputs and source/configuration identities after evaluation.
    validate_phase(trial, 'stage1', lock)
    assert not torch.cuda.is_initialized()
    proof = dict(fingerprint=lock['fingerprint'], stage1_seal_sha256=seal_sha,
        lock_sha256=sha(trial / 'lock.json'), manifest_sha256=lock['manifest_sha256'], labels_sha256=labels_sha,
        all64_validated_before_labels=True, all_hashes_rechecked_after_labels=True,
        official_dense_checks=checks, official_dense_max_absolute_error=dense_error,
        old_baseline_metric_max_absolute_error=baseline_error,
        initial_diagnostic_paths_per_query=2, added_rollouts=0, model_calls=0, decoder_calls=0,
        expert_calls=0, RGB_reads=0, CUDA_initialized=False, per_query_GT_selection=False)
    public = dict(status='stage1_complete_not_adapted', dataset='hc2', queries=64, parents=64,
        development_exposure=True, selection='first64_original_manifest_order_one_query_per_parent',
        arm_order=names, means=means, valid_query_counts=metric_counts, paired=comparisons,
        memory_comparison=memory_pair, candidate_oracle=candidate, feedback_ranking=ranking,
        distance=_distance_summary(distance_queries, names), gate=gate, proof=proof,
        ranking_tie_atol=TIE_ATOL, bootstrap_draws=BOOTSTRAP_DRAWS, bootstrap_seed=BOOTSTRAP_SEED,
        queries_anonymous=[dict(query=f'H{index+1:03d}', values=row['values'], candidate_oracle=row['candidate_oracle'],
            feedback_ranking=row['feedback_ranking'], distance=row['distance'],
            MemoryGreedy_minus_NoMemoryGreedy_vIoU_pp=100*(row['values']['MemoryGreedy']['v']-row['values']['NoMemoryGreedy']['v']),
            MemoryGreedy_minus_NoMemoryGreedy_unobserved_sIoU_pp=(None if row['values']['MemoryGreedy']['unobserved_actual_sIoU'] is None or row['values']['NoMemoryGreedy']['unobserved_actual_sIoU'] is None else
                100*(row['values']['MemoryGreedy']['unobserved_actual_sIoU']-row['values']['NoMemoryGreedy']['unobserved_actual_sIoU'])))
            for index, row in enumerate(rows)])
    output.mkdir(parents=True, exist_ok=True)
    assert not (output / 'diagnostics.json').exists(), 'Do not overwrite a prior one-time diagnostic'
    write_json(output / 'diagnostics.json', public)
    (output / 'diagnostics.md').write_text(_render(public))
    decision = dict(gate, fingerprint=lock['fingerprint'], stage1_seal_sha256=seal_sha,
        labels_sha256=labels_sha, lock_sha256=sha(trial / 'lock.json'),
        diagnostics_sha256=sha(output / 'diagnostics.json'), diagnostics_report_sha256=sha(output / 'diagnostics.md'),
        queries=64, all64_sealed_before_GT=True, GT_read=True)
    write_json(gate_path, decision)
    return decision


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trial', type=Path, default=ROOT / 'outputs/propagation_v0_hc64')
    parser.add_argument('--run-after-stage1-seal', action='store_true')
    args = parser.parse_args()
    if not args.run_after_stage1_seal:
        parser.error('This one-time GT diagnostic requires explicit --run-after-stage1-seal')
    result = diagnose(args.trial)
    print(json.dumps(result, indent=2))
