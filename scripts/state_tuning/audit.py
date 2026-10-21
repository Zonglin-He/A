"""Audit sealed full-content HC64 traces and already saved scores on CPU.

This never loads a model, creates candidates, reads GT annotations, or scores
predictions. Explicit --run-after-score and a valid complete seal/score receipt
are required before the new scored result or any full trace set is read.
"""
from __future__ import annotations

import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import argparse
import ast
import collections
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

torch.set_num_threads(1)
ROOT = Path(os.environ.get('STVG_PROJECT_ROOT', Path(__file__).resolve().parents[2])).resolve()
ARMS = ('StateOPD', 'Direct', 'DirectMC', 'DirectFullRef')
STOCHASTIC = ('StateOPD', 'DirectMC')
METRICS = ('m_vIoU', 'vIoU@0.3', 'vIoU@0.5', 'm_tIoU', 'm_sIoU',
           'm_sIoU_fixed_actual', 'm_sIoU_observed_actual', 'm_sIoU_unobserved_actual')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def arr(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def stats(values):
    values = np.asarray(values, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if not len(values):
        return dict(n=0)
    return dict(n=len(values), mean=float(values.mean()), median=float(np.median(values)),
                p10=float(np.quantile(values, .1)), p90=float(np.quantile(values, .9)),
                minimum=float(values.min()), maximum=float(values.max()))


def correlations(x, y):
    """Descriptive associations of saved query metrics, not candidate GT ranking."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y); x, y = x[valid], y[valid]
    if len(x) < 3 or not np.ptp(x) or not np.ptp(y):
        return dict(n=len(x), status='insufficient_variation')
    rank, linear = spearmanr(x, y), pearsonr(x, y)
    return dict(n=len(x), spearman_rho=float(rank.statistic), spearman_p=float(rank.pvalue),
                pearson_r=float(linear.statistic), pearson_p=float(linear.pvalue),
                interpretation='posthoc query-level association; uncorrected descriptive p values')


def cosine(a, b):
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a.reshape(-1), b.reshape(-1)) / denominator) if denominator else None


def corners(boxes):
    return np.concatenate((boxes[..., :2] - boxes[..., 2:] / 2,
                           boxes[..., :2] + boxes[..., 2:] / 2), -1)


def chart_distance(boxes, initial):
    delta = np.concatenate(((boxes[..., :2] - initial[..., :2]) / initial[..., 2:],
                            np.log(boxes[..., 2:]) - np.log(initial[..., 2:])), -1)
    return np.linalg.norm(delta, axis=-1) / 2


def motion(boxes, initial, observed):
    current_corners, initial_corners = corners(boxes), corners(initial)
    intersection = np.maximum(0, np.minimum(current_corners[:, 2:], initial_corners[:, 2:]) -
                              np.maximum(current_corners[:, :2], initial_corners[:, :2])).prod(-1)
    overlap = intersection / (boxes[:, 2:].prod(-1) + initial[:, 2:].prod(-1) - intersection)
    center = np.linalg.norm(boxes[:, :2] - initial[:, :2], axis=-1)
    distance = chart_distance(boxes, initial)
    return dict(corner_abs_coordinate_change_pct_image=float(100 * np.abs(current_corners - initial_corners).mean()),
                center_L2_change_in_normalized_image_units=float(center.mean()),
                center_L2_change_observed=float(center[observed].mean()),
                center_L2_change_unobserved=float(center[~observed].mean()),
                standardized_chart_d_mean=float(distance.mean()),
                standardized_chart_d_max=float(distance.max()),
                same_node_IoU_vs_E_mean=float(overlap.mean()), same_node_IoU_vs_E_min=float(overlap.min()),
                nodes_IoU_vs_E_ge099=float((overlap >= .99).mean()),
                nodes_IoU_vs_E_ge095=float((overlap >= .95).mean()))


def checked_path(root, relative):
    path = (root / relative).resolve()
    require(path.is_relative_to(root.resolve()), 'artifact path escapes its trial')
    return path


def ready(trial, result_path):
    """Fail closed before reading new score values or full traces."""
    require((trial / 'sealed.json').is_file(), 'all64 seal is required; do not audit an active prefix')
    require((trial / 'score_receipt.json').is_file(), 'completed score receipt is required')
    seal, receipt, lock = (read(trial / name) for name in ('sealed.json', 'score_receipt.json', 'lock.json'))
    require(seal['queries'] == lock['queries'] == 64 and len(seal['files']) == 64, 'complete HC64 scope')
    require(seal['GT_read'] is lock['GT_read'] is False, 'pre-scoring label isolation')
    require(seal['lock_sha256'] == sha(trial / 'lock.json'), 'seal binds exact lock')
    require(seal['fingerprint'] == lock['fingerprint'], 'seal fingerprint')
    require(receipt['status'] == 'scored' and receipt['metrics_sha256'] == sha(trial / 'metrics.json'),
            'score receipt binds completed saved metrics')
    proof = receipt['proof']
    require(proof['fingerprint'] == lock['fingerprint'] and proof['seal_sha256'] == sha(trial / 'sealed.json') and
            proof['lock_sha256'] == sha(trial / 'lock.json') and proof['all64_validated_before_labels'] is True,
            'score receipt proof')
    require(set(lock['ids']) == set(seal['files']) and len(set(lock['ids'])) == 64, 'all fixed IDs exactly once')
    require(tuple(lock['arms']) == ARMS, 'locked arm order')
    selected = lock['selected']
    require(selected['basis_dimensions'] == 256 and selected['radius'] is None, 'full identity content, no cap')
    require(selected['sigma'] > 0 and selected['tau'] > 0 and selected['samples'] % 2 == 0, 'locked sampling settings')
    require(sha(trial / 'calibration.json') == lock['calibration_sha256'], 'calibration binding')
    calibration = read(trial / 'calibration.json')
    require(calibration['GT_read'] is False and calibration['selected'] == selected, 'one global label-free calibration')
    source_bad = [name for name, expected in lock['sources'].items() if sha(ROOT / name) != expected]
    require(not source_bad, 'new frozen source changed')
    require(sha(ROOT / 'outputs/formal/state.json') == lock['formal_state_sha256'], 'formal paused state changed')
    # This is the first read of this experiment's scored result.
    result = read(result_path)
    require(result['queries'] == result['parents'] == 64 and result['proof'] == proof, 'saved public result binding')
    return lock, seal, receipt, result


def reward_ast_checks(parent_lock):
    names = ('ProjectionEvidence', 'protected_utility', 'FrozenContentReplay')
    definitions = []
    for relative in ('scripts/state_opd/core.py', 'scripts/state_tuning/core.py'):
        tree = ast.parse((ROOT / relative).read_text())
        definitions.append({n.name: ast.dump(n, include_attributes=False) for n in tree.body
                            if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names})
    require(set(definitions[0]) == set(names) == set(definitions[1]), 'reward/replay definitions present')
    require(definitions[0] == definitions[1], 'reward, evidence and frozen replay computation unchanged')
    bad = [name for name, expected in parent_lock['sources'].items() if sha(ROOT / name) != expected]
    require(not bad, 'parent frozen source changed')
    return dict(identical_AST_definitions=list(names), parent_frozen_sources_checked=len(parent_lock['sources']),
                new_frozen_sources_checked=2, source_mismatches=[])


def compare_means(result, previous):
    require(previous['queries'] == previous['parents'] == 64, 'previous matched HC64 scope')
    for baseline in ('ERefStrong', 'GeometryOPD'):
        for name in METRICS:
            require(abs(result['means'][baseline][name] - previous['means'][baseline][name]) < 1e-12,
                    'matched saved baseline mean changed')
    return dict(
        saved_ERefStrong={m: result['means']['ERefStrong'][m] for m in METRICS},
        saved_GeometryOPD={m: result['means']['GeometryOPD'][m] for m in METRICS},
        previous_PCA4_radius4={a: {m: previous['means'][a][m] for m in METRICS} for a in ARMS},
        full_content_calibrated={a: {m: result['means'][a][m] for m in METRICS} for a in ARMS},
        differences_pp={a: {reference: {m: 100 * (result['means'][a][m] - values[m]) for m in METRICS}
                            for reference, values in (
                                ('ERefStrong', result['means']['ERefStrong']),
                                ('GeometryOPD', result['means']['GeometryOPD']),
                                ('previous_same_arm_PCA4_radius4', previous['means'][a]))} for a in ARMS})


def audit(trial, result_path, output):
    begin = time.perf_counter()
    lock, seal, score_receipt, result = ready(trial, result_path)
    selected = lock['selected']; sigma = float(selected['sigma']); tau = float(selected['tau'])
    samples = int(selected['samples']); budget = int(selected['forward_budget'])
    previous_path = ROOT / 'results/state_opd_hc64/hc64.json'
    previous = read(previous_path)
    parent_trial = ROOT / 'outputs/state_opd_hc128'
    parent_lock = read(parent_trial / 'lock.json')
    parent_scope = read(parent_trial / 'user_scope64.json')
    require(lock['parent_fingerprint'] == parent_lock['fingerprint'] == previous['proof']['fingerprint'],
            'same parent inference identity')
    require(lock['ids'] == parent_scope['ids'], 'same fixed HC64 order')
    source_checks = reward_ast_checks(parent_lock)
    comparison = compare_means(result, previous)
    rows = {r['query']: r['values'] for r in result['queries_anonymous']}
    oldrows = {r['query']: r['values'] for r in previous['queries_anonymous']}
    require(set(rows) == set(oldrows) == {f'H{i + 1:03d}' for i in range(64)}, 'anonymous query alignment')
    items = {r['id']: r for r in parent_lock['inputs']}
    evidence_root = ROOT / 'outputs/layout_reward_dev'
    values = {a: collections.defaultdict(list) for a in ARMS}
    per_query = {}; errors = collections.defaultdict(float); checks = collections.Counter()
    initial_updates = collections.defaultdict(list); common_update_prefix = []
    constants = dict(guard_weight=.1, guard_deadzone=.25, guard_mean_weight=.5, guard_max_weight=.5,
                     mask_mass_epsilon=1e-6, accept_epsilon=1e-8)
    expected_identity = np.eye(256, dtype=np.float64)

    for index, key in enumerate(lock['ids']):
        public = f'H{index + 1:03d}'
        entry = seal['files'][key]; prediction_path = checked_path(trial, entry['path'])
        receipt_path = prediction_path.with_suffix('.json'); receipt = read(receipt_path)
        trace_path = checked_path(trial, receipt['trace_path'])
        require(receipt['GT_read'] is False and receipt['fingerprint'] == lock['fingerprint'], 'prediction receipt identity')
        require(sha(prediction_path) == entry['sha256'] == receipt['sha256'], 'prediction hash')
        require(sha(receipt_path) == entry['receipt_sha256'], 'receipt hash')
        require(sha(trace_path) == receipt['trace_sha256'], 'trace hash')
        item = items[key]
        for name in ('reference_sha256', 'cache_sha256', 'native_mask_sha256'):
            require(receipt[name] == item[name], 'same existing Source/expert/reference evidence')
        reference_path = checked_path(evidence_root, item['reference_path'])
        require(sha(reference_path) == item['reference_sha256'], 'reference hash')
        trace = torch.load(trace_path, map_location='cpu', weights_only=False)
        require(tuple(trace) == ARMS, 'trace arm order')
        per_query[public] = {}
        with np.load(prediction_path, allow_pickle=False) as prediction, np.load(reference_path, allow_pickle=False) as reference:
            for name in ('frame_ids', 'positions8', 'Strong_interval'):
                require(np.array_equal(prediction[name], reference[name]), 'physical nodes/positions/Strong unchanged')
            initial_boxes = reference['E_Ref']
            count = len(prediction['frame_ids']); observed = np.zeros(count, dtype=bool)
            observed[prediction['positions8']] = True
            require(count == 65 and initial_boxes.shape == (count, 4) and observed.sum() == 8, 'complete physical trajectory')
            initial = trace[ARMS[0]]['path'][0]
            for arm in ARMS:
                saved = trace[arm]; path = saved['path']; rounds = saved['rounds']
                cost = saved['costs']; initial_checks = saved['initial_checks']; aggregate = values[arm]
                require(np.array_equal(arr(saved['basis']), expected_identity), 'P must be exact I256')
                require(saved['basis_info']['identity_full_content'] is True and saved['basis_info']['rank'] == 256,
                        'identity basis provenance')
                require(hashlib.sha256(arr(saved['basis']).tobytes()).hexdigest() == saved['basis_info']['basis_sha256'], 'basis hash')
                require(initial_checks['config'] == selected, 'all arms use locked selected settings')
                require(initial_checks['state_space'] == dict(dimensions=[8, 256], radius=None, map='identity'), 'identity state metadata')
                require(initial_checks['constants']['content_radius'] is None, 'no content radius limit')
                for name, expected in constants.items():
                    require(initial_checks['constants'][name] == expected, 'common protected utility constant')
                require(initial_checks['original_native_Source_bit_exact'] and initial_checks['original_E_bit_exact'] and
                        initial_checks['network_weights_frozen'] and initial_checks['initial_z_zero'], 'frozen original initialization')
                require(arm != 'DirectFullRef' or initial_checks['full_reference_forward_bit_exact'], 'FullRef forward compatibility')
                require(initial_checks['valid_mask_anchors'] == 8, 'fixed native-mask anchor coverage')
                require(np.array_equal(arr(path[0]['boxes']), initial_boxes), 'same E initialization')
                require(path[0]['U'] == initial['U'] and path[0]['mask'] == initial['mask'], 'same initial evidence utility')
                require(not np.count_nonzero(arr(path[0]['z'])) and not np.count_nonzero(arr(path[0]['eta'])), 'zero state initialization')
                require(np.array_equal(arr(saved['final']), prediction[arm]) and
                        np.array_equal(arr(path[-1]['boxes']), prediction[arm]), 'final trace/prediction agreement')
                require(len(path) == len(rounds) + 1 and len(rounds) <= int(selected['opd_rounds'] if arm in STOCHASTIC else selected['direct_steps']), 'path/update count')
                require(1 <= cost['full_trajectory_evals'] <= budget and cost['allowance'] == budget, 'shared actual forward allowance')
                expected_evals = 1 + sum((samples if arm in STOCHASTIC else 1) + len(r['backtracking']) for r in rounds)
                require(expected_evals == cost['full_trajectory_evals'] == cost['reward_evals'], 'dynamic forward accounting')
                require(cost['native_offset_decoder_calls'] == 2 * expected_evals, 'two native offsets per trajectory')
                require(cost['updates'] == len(rounds) and cost['accepted_updates'] == sum(r['accepted'] for r in rounds), 'acceptance accounting')
                expected_backwards = samples * len(rounds) if arm == 'DirectMC' else len(rounds) if arm != 'StateOPD' else 0
                require(cost['decoder_backwards'] == expected_backwards and
                        cost['state_only_backwards'] == (len(rounds) if arm == 'StateOPD' else 0), 'backward accounting')
                require(cost['setup_trajectory_evals'] == (2 if arm == 'DirectFullRef' else 1) and
                        cost['setup_native_offset_decoder_calls'] == 2 * cost['setup_trajectory_evals'], 'separate setup accounting')
                boxes = np.stack([arr(p['boxes']) for p in path]); z = np.stack([arr(p['z']) for p in path])
                eta = np.stack([arr(p['eta']) for p in path]); utility = np.array([p['U'] for p in path])
                require(z.shape == eta.shape == (len(path), 8, 256) and np.array_equal(z, eta), 'eta=z in full content space')
                require(np.isfinite(z).all() and np.isfinite(boxes).all() and np.all(boxes[..., 2:] > 0), 'finite states and trajectories')
                require(np.all(np.diff(utility) >= -1e-14), 'accepted path utility monotonicity')
                distance = chart_distance(boxes, initial_boxes)
                guard = .5 * np.square(np.maximum(0, distance - .25)).mean(-1) + .5 * np.square(np.maximum(0, distance - .25)).max(-1)
                guard_error = float(np.max(np.abs(guard - np.array([p['guard'] for p in path]))))
                utility_error = float(np.max(np.abs(utility - (np.array([p['mask'] for p in path]) - .1 * guard))))
                errors['guard_max_abs_error'] = max(errors['guard_max_abs_error'], guard_error)
                errors['utility_max_abs_error'] = max(errors['utility_max_abs_error'], utility_error)
                require(guard_error < 1e-12 and utility_error < 1e-12, 'independent shared protected utility')
                final_z, final_eta = arr(saved['final_z']), arr(saved['final_eta'])
                require(np.array_equal(final_z, z[-1]) and np.array_equal(final_eta, eta[-1]), 'final state/path agreement')
                steps = np.diff(z.reshape(len(path), -1).astype(np.float64), axis=0)
                step_norms = np.linalg.norm(steps, axis=-1); path_length = float(step_norms.sum())
                consecutive = [cosine(a, b) for a, b in zip(steps[:-1], steps[1:])]
                consecutive = [v for v in consecutive if v is not None]
                record = dict(initial_U=float(utility[0]), final_U=float(utility[-1]), delta_U=float(utility[-1] - utility[0]),
                              final_mask=float(path[-1]['mask']), final_guard=float(path[-1]['guard']),
                              maximum_guard_on_saved_path=float(guard.max()), z_L2=float(np.linalg.norm(final_z)),
                              eta_L2=float(np.linalg.norm(final_eta)), eta_anchor_L2_mean=float(np.linalg.norm(final_eta, axis=-1).mean()),
                              eta_anchor_L2_max=float(np.linalg.norm(final_eta, axis=-1).max()),
                              accepted_updates=cost['accepted_updates'], updates=cost['updates'],
                              trajectory_evals=cost['full_trajectory_evals'], adaptation_seconds=cost['seconds'],
                              state_path_length_L2=path_length,
                              state_net_over_path_length=float(np.linalg.norm(final_z) / path_length) if path_length else 0.,
                              successive_nonzero_update_cos_mean=float(np.mean(consecutive)) if consecutive else float('nan'),
                              successive_nonzero_update_cos_median=float(np.median(consecutive)) if consecutive else float('nan'),
                              stop_reason=saved['stop_reason'], **motion(prediction[arm], initial_boxes, observed))
                for baseline in ('ERefStrong', 'GeometryOPD', 'previous_same_arm_PCA4_radius4'):
                    base_values = oldrows[public][arm] if baseline.startswith('previous_') else rows[public][baseline]
                    for metric in ('v', 'observed_actual_sIoU', 'unobserved_actual_sIoU'):
                        record[f'delta_{metric}_pp_vs_{baseline}'] = 100 * (rows[public][arm][metric] - base_values[metric])
                per_query[public][arm] = record
                for name, value in record.items():
                    if isinstance(value, (int, float)):
                        aggregate[name].append(value)
                aggregate['path_guard_nonzero'].append(int((guard > 0).sum()))
                aggregate['path_points'].append(len(path)); aggregate['path_standardized_d_max'].append(float(distance.max()))
                aggregate['stopping'].append(saved['stop_reason'])
                for round_index, round_data in enumerate(rounds):
                    require(round_data['after']['U'] >= round_data['before']['U'] - 1e-14, 'round utility monotonicity')
                    if round_data['accepted']:
                        require(round_data['after']['U'] > round_data['before']['U'] + 1e-8, 'fixed acceptance epsilon')
                    else:
                        require(np.array_equal(z[round_index + 1], z[round_index]) and
                                np.array_equal(boxes[round_index + 1], boxes[round_index]), 'rejected proposal state/output rollback')
                    actual_step = float(np.linalg.norm(z[round_index + 1] - z[round_index]))
                    aggregate['actual_state_step_L2'].append(actual_step)
                    if not round_index:
                        aggregate['first_actual_state_step_L2'].append(actual_step)
                    aggregate['grad_norm'].append(round_data['gradient_norm'])
                    aggregate['Adam_delta_norm'].append(round_data['Adam_delta_norm'])
                    aggregate['all_backtrack_guard'].extend(p['guard'] for p in round_data['backtracking'])
                    if 'weights' in round_data:
                        weights = arr(round_data['weights']); candidate_u = arr(round_data['candidate_U'])
                        require(candidate_u.shape == (samples,) and arr(round_data['candidate_z']).shape == (samples, 8, 256), 'real joint content candidate shape')
                        expected_weights = np.exp((candidate_u - candidate_u.max()) / tau); expected_weights /= expected_weights.sum()
                        require(np.max(np.abs(weights - expected_weights)) < 2e-12, 'locked-temperature softmax')
                        candidate_utility_error = float(np.max(np.abs(candidate_u - (arr(round_data['candidate_mask']) - .1 * arr(round_data['candidate_guard'])))))
                        require(candidate_utility_error < 1e-12, 'candidate protected utility')
                        errors['candidate_utility_max_abs_error'] = max(errors['candidate_utility_max_abs_error'], candidate_utility_error)
                        aggregate['ESS'].append(float(1 / np.square(weights).sum()))
                        aggregate['weight_max'].append(float(weights.max()))
                        aggregate['weight_TV_to_uniform'].append(float(np.abs(weights - 1 / samples).sum() / 2))
                        aggregate['candidate_U_std'].append(float(candidate_u.std()))
                        aggregate['candidate_U_range'].append(float(candidate_u.max() - candidate_u.min()))
                        aggregate['candidate_U_std_over_tau'].append(float(candidate_u.std() / tau))
                        aggregate['candidate_guards_nonzero'].append(int((arr(round_data['candidate_guard']) > 0).sum()))
                        aggregate['candidate_guard_max'].append(float(arr(round_data['candidate_guard']).max()))
                checks['arm_query_checks_passed'] += 1
            require(np.array_equal(arr(trace['StateOPD']['rounds'][0]['candidate_z']),
                                   arr(trace['DirectMC']['rounds'][0]['candidate_z'])), 'same initial stored candidate stream')
            for a, b in zip(trace['StateOPD']['rounds'], trace['DirectMC']['rounds']):
                require(a['rng_seed'] == b['rng_seed'], 'shared per-round noise stream')
            initial_candidate_error = float(np.max(np.abs(arr(trace['StateOPD']['rounds'][0]['candidate_U']) -
                                                        arr(trace['DirectMC']['rounds'][0]['candidate_U']))))
            errors['initial_OPD_MC_candidate_utility_max_abs_difference'] = max(
                errors['initial_OPD_MC_candidate_utility_max_abs_difference'], initial_candidate_error)
            first_steps = {a: arr(trace[a]['path'][1]['z']).astype(np.float64) - arr(trace[a]['path'][0]['z']) for a in ARMS}
            for a, b in (('StateOPD', 'Direct'), ('StateOPD', 'DirectMC'), ('StateOPD', 'DirectFullRef'), ('Direct', 'DirectFullRef')):
                value = cosine(first_steps[a], first_steps[b])
                if value is not None:
                    initial_updates[a + '-' + b].append(value)
            if len(trace['StateOPD']['rounds']) >= 40 and len(trace['Direct']['rounds']) >= 40:
                opd40, direct40 = trace['StateOPD']['path'][40], trace['Direct']['path'][40]
                common_update_prefix.append(dict(query=public,
                    Direct_minus_StateOPD_U=float(direct40['U'] - opd40['U']),
                    StateOPD_state_L2=float(np.linalg.norm(arr(opd40['z']))),
                    Direct_state_L2=float(np.linalg.norm(arr(direct40['z'])))))
            for baseline in ('ERefStrong', 'GeometryOPD'):
                for metric in ('v', 't', 'observed_actual_sIoU', 'unobserved_actual_sIoU'):
                    require(abs(rows[public][baseline][metric] - oldrows[public][baseline][metric]) < 1e-12,
                            'matched saved per-query baseline')
            for arm in ARMS:
                require(rows[public][arm]['t'] == rows[public]['ERefStrong']['t'], 'unchanged temporal interval metric')
        checks['queries_passed'] += 1
        del trace

    aggregates = {}
    for arm, data in values.items():
        aggregate = {k: stats(v) for k, v in data.items() if k != 'stopping'}
        aggregate['stop_reasons'] = dict(collections.Counter(data['stopping']))
        aggregate['reward_up_queries'] = sum(v > 1e-12 for v in data['delta_U'])
        aggregate['reward_up_v_down_queries'] = sum(per_query[q][arm]['delta_U'] > 1e-12 and
                                                  per_query[q][arm]['delta_v_pp_vs_ERefStrong'] < -1e-9 for q in per_query)
        aggregate['reward_up_obs_up_unobs_down_queries'] = sum(per_query[q][arm]['delta_U'] > 1e-12 and
            per_query[q][arm]['delta_observed_actual_sIoU_pp_vs_ERefStrong'] > 1e-9 and
            per_query[q][arm]['delta_unobserved_actual_sIoU_pp_vs_ERefStrong'] < -1e-9 for q in per_query)
        aggregate['final_guard_active_queries'] = sum(v > 0 for v in data['final_guard'])
        aggregate['any_path_guard_active_queries'] = sum(v > 0 for v in data['maximum_guard_on_saved_path'])
        aggregate['path_guard_nonzero_total'] = sum(data['path_guard_nonzero'])
        aggregate['candidate_guard_nonzero_total'] = sum(data['candidate_guards_nonzero'])
        aggregate['query_mean_IoU_vs_E_below099'] = sum(v < .99 for v in data['same_node_IoU_vs_E_mean'])
        aggregate['query_mean_IoU_vs_E_below095'] = sum(v < .95 for v in data['same_node_IoU_vs_E_mean'])
        aggregate['ESS_ge_31_over32_rounds'] = sum(v >= samples * 31 / 32 for v in data['ESS'])
        aggregate['ESS_below_half_samples_rounds'] = sum(v < samples / 2 for v in data['ESS'])
        aggregate['reward_task_correlations'] = {metric: correlations(data['delta_U'], data[f'delta_{metric}_pp_vs_ERefStrong'])
                                                for metric in ('v', 'observed_actual_sIoU', 'unobserved_actual_sIoU')}
        aggregate['largest_local_movement_three'] = [dict(query=q, **per_query[q][arm]) for q in
            sorted(per_query, key=lambda q: per_query[q][arm]['standardized_chart_d_max'], reverse=True)[:3]]
        aggregates[arm] = aggregate
    worst = {a: {baseline: [dict(query=q, **per_query[q][a]) for q in
        sorted(per_query, key=lambda q: per_query[q][a][f'delta_v_pp_vs_{baseline}'])[:3]]
        for baseline in ('ERefStrong', 'GeometryOPD', 'previous_same_arm_PCA4_radius4')} for a in ARMS}
    direct_gaps = [dict(query=q,
        Direct_minus_StateOPD_v_pp=100 * (rows[q]['Direct']['v'] - rows[q]['StateOPD']['v']),
        Direct_minus_StateOPD_unobserved_s_pp=100 * (rows[q]['Direct']['unobserved_actual_sIoU'] - rows[q]['StateOPD']['unobserved_actual_sIoU']),
        Direct_minus_StateOPD_U=per_query[q]['Direct']['delta_U'] - per_query[q]['StateOPD']['delta_U'],
        arms={a: per_query[q][a] for a in ARMS}) for q in per_query]
    comparisons = dict(
        same_initial_actual_Adam_update_cos={k: dict(**stats(v), negative_count=sum(x < 0 for x in v),
            excluded_zero_update_queries=64 - len(v)) for k, v in initial_updates.items()},
        same_initial_update_limit='Both start from identical zero state/empty Adam. These are actual first updates after line search, not raw-gradient cosines.',
        Direct_and_StateOPD_at40_saved_updates=dict(queries=len(common_update_prefix),
            Direct_minus_StateOPD_U=stats([r['Direct_minus_StateOPD_U'] for r in common_update_prefix]),
            Direct_higher_U_queries=sum(r['Direct_minus_StateOPD_U'] > 1e-12 for r in common_update_prefix),
            StateOPD_state_L2=stats([r['StateOPD_state_L2'] for r in common_update_prefix]),
            Direct_state_L2=stats([r['Direct_state_L2'] for r in common_update_prefix]),
            limit='Non-GT subset with both paths reaching40 updates; trajectories differ, so this is not a matched-state gradient comparison.'),
        Direct_minus_StateOPD_U_gap_vs_v_gap=correlations([r['Direct_minus_StateOPD_U'] for r in direct_gaps],
                                                        [r['Direct_minus_StateOPD_v_pp'] for r in direct_gaps]),
        largest_Direct_v_advantage_three=sorted(direct_gaps, key=lambda r: r['Direct_minus_StateOPD_v_pp'], reverse=True)[:3],
        largest_StateOPD_v_advantage_three=sorted(direct_gaps, key=lambda r: r['Direct_minus_StateOPD_v_pp'])[:3])
    selected_pairs = {key: value for key, value in result['paired'].items() if key in
        {a + '-ERefStrong' for a in ARMS} | {'StateOPD-Direct', 'StateOPD-GeometryOPD', 'Direct-GeometryOPD'}}
    out = dict(status='all64_saved_full_content_trace_validity_pass', queries=64,
               fingerprint=lock['fingerprint'], selected=selected,
               GT_annotations_read=False, new_GT_scoring=False, already_scored_metrics_read=True,
               model_calls=0, GPU_calls=0, new_candidates_generated=0, CUDA_initialized=torch.cuda.is_initialized(),
               checks=dict(checks), errors=dict(errors), source_checks=source_checks,
               arms=aggregates, comparison_means=comparison, update_comparisons=comparisons,
               saved_task_comparisons=selected_pairs, worst_three_by_saved_vIoU_delta=worst,
               inputs=dict(seal_sha256=sha(trial / 'sealed.json'), lock_sha256=sha(trial / 'lock.json'),
                           score_receipt_sha256=sha(trial / 'score_receipt.json'), scored_result_sha256=sha(result_path),
                           previous_PCA4_scored_result_sha256=sha(previous_path), script_sha256=sha(__file__)),
               limits=['All comparisons use the same exposed HC64 query order and saved Strong intervals; no independent confirmation is claimed.',
                       'This experiment jointly changed state capacity, radius, sigma and tau relative to the previous 32-dimensional trial; their individual effects are not isolated here.',
                       'Real trajectory evaluation is checked from frozen code, shape, constructor checks and exact counters, without independent model replay.',
                       'Path guard/U is recomputed from saved boxes and mask scores. Native mask rewards are not independently rescored.',
                       'Candidate guard values are saved feedback; full candidate trajectories were not retained.',
                       'Movement uses all65 physical input nodes. Its observed/unobserved split is not filtered by GT support.',
                       'Utility improvements, ESS and state/output movement describe optimization; they do not establish GT correctness.',
                       'Reward-task correlations use only saved per-query utility changes and already scored final GT metric changes; they are not candidate-level reward reliability measurements.',
                       'Direct/FullRef and OPD/MC share the forward allowance, but actual consumed counts and stopping can differ.',
                       'FullRef forward identity is the bit-exact saved constructor check; its later derivative-driven trajectory can differ.'],
               seconds=time.perf_counter() - begin)
    require(not out['CUDA_initialized'], 'CPU audit unexpectedly initialized CUDA')
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(output.name + '.tmp')
    temp.write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n')
    os.replace(temp, output)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trial', type=Path, default=ROOT / 'outputs/state_tuning_hc64')
    parser.add_argument('--result', type=Path, default=ROOT / 'results/state_tuning_hc64/hc64.json')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--run-after-score', action='store_true')
    args = parser.parse_args()
    if not args.run_after_score:
        parser.error('--run-after-score is required; preparation/import never audits active prefixes')
    trial, result = args.trial.resolve(), args.result.resolve()
    require(trial.parent == (ROOT / 'outputs').resolve(), 'trial must be an explicit project outputs child')
    output = args.output.resolve() if args.output else trial / 'monitor/final_validity.json'
    require(output.resolve().is_relative_to(trial), 'audit output must stay inside its own trial')
    out = audit(trial, result, output)
    print(json.dumps(dict(status=out['status'], queries=out['queries'], checks=out['checks'],
                          errors=out['errors'], seconds=out['seconds'], output=str(output)), ensure_ascii=False))


if __name__ == '__main__':
    main()
