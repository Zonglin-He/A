"""Audit a complete sealed HC64 trajectory trial using saved scores and traces.

No annotations, RGB, models, decoders, candidate sampling, or reward evaluation.
The optional local-linear comparison concerns the frozen proxy utility only.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import re

import numpy as np

ROOT = Path(os.environ.get('STVG_PROJECT_ROOT', Path(__file__).resolve().parents[1])).expanduser().resolve()
FINGERPRINT = '8ba4676615af13fa4428559c1d9651dc8a0570ee1cbbb994fcf6720f5335ee19'
ARMS = ['TrajectoryOPD', 'Direct', 'DirectMulti', 'DirectMC']
FIELDS = ['v', 'observed_actual_sIoU', 'unobserved_actual_sIoU']


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def within(parent, relative):
    path = (Path(parent) / relative).resolve()
    if not path.is_relative_to(Path(parent).resolve()):
        raise ValueError('Artifact path escapes its declared root')
    return path


def stats(values):
    values = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if not len(values):
        return dict(valid_queries=0)
    assert np.isfinite(values).all()
    return dict(valid_queries=len(values), mean=float(values.mean()), median=float(np.median(values)),
                minimum=float(values.min()), maximum=float(values.max()),
                p10=float(np.quantile(values, .1)), p90=float(np.quantile(values, .9)))


def verify_saved_scores(public):
    rows, names = public['queries_anonymous'], public['arm_order']
    assert public['queries'] == public['parents'] == len(rows) == 64
    assert [r['query'] for r in rows] == ['H%03d' % i for i in range(1, 65)]
    mapping = {'m_vIoU': 'v', 'vIoU@0.3': 'vIoU_at_03', 'vIoU@0.5': 'vIoU_at_05',
               'm_tIoU': 't', 'm_sIoU': 's', 'm_sIoU_fixed_actual': 'fixed_actual_sIoU',
               'm_sIoU_observed_actual': 'observed_actual_sIoU',
               'm_sIoU_unobserved_actual': 'unobserved_actual_sIoU',
               'vIoU_ge_03': 'vIoU_ge_03', 'vIoU_ge_05': 'vIoU_ge_05',
               'tIoU_ge_05': 'tIoU_ge_05', 'tIoU_eq_0': 'tIoU_eq_0',
               'boundary_MAE_seconds': 'boundary_MAE_seconds'}
    mean_error = paired_error = ci_error = 0.
    draws = np.random.default_rng(20261021).integers(0, 64, (10000, 64))
    checks = 0
    for name in names:
        for key, field in mapping.items():
            values = [r['values'][name][field] for r in rows if r['values'][name][field] is not None]
            mean_error = max(mean_error, abs(sum(values) / len(values) - public['means'][name][key]))
        for field, count in public['valid_query_counts'][name].items():
            assert count == 64 == sum(r['values'][name][field] is not None for r in rows)
    for first, second in itertools.permutations(names, 2):
        for field, saved in public['paired'][first + '-' + second].items():
            difference = np.asarray([100 * (r['values'][first][field] - r['values'][second][field]) for r in rows])
            assert saved['valid_parents'] == 64
            paired_error = max(paired_error, abs(float(difference.mean()) - saved['mean_pp']))
            ci_error = max(ci_error, float(np.max(np.abs(np.quantile(difference[draws].mean(1), [.025, .975]) - saved['CI95_pp']))))
            counts = dict(improved=int((difference > 1e-8).sum()), declined=int((difference < -1e-8).sum()),
                          unchanged=int((np.abs(difference) <= 1e-8).sum()), declined_over5pp=int((difference < -5).sum()))
            for key, value in counts.items():
                assert saved[key] == value
                checks += 1
            for key, value in [('positive_mean_contribution_pp', float(difference[difference > 0].sum() / 64)),
                               ('negative_mean_contribution_pp', float(difference[difference < 0].sum() / 64))]:
                paired_error = max(paired_error, abs(value - saved[key]))
    assert max(mean_error, paired_error, ci_error) < 1e-12
    return dict(anonymous_complete_signed_queries=64, mean_max_abs_error=mean_error,
                paired_delta_contribution_max_abs_error_pp=paired_error,
                bootstrap_CI_max_abs_error_pp=ci_error, signed_count_checks=checks)


def analyze():
    import torch
    torch.set_num_threads(1)
    trial = ROOT / 'outputs/trajectory_opd_hc64'
    output = ROOT / 'results/trajectory_opd_hc64'
    public_path = output / 'hc64.json'
    seal_path, lock_path = trial / 'sealed.json', trial / 'lock.json'
    seal, lock = read(seal_path), read(lock_path)
    public, score_receipt = read(public_path), read(trial / 'score_receipt.json')
    assert read(trial / 'run.json')['status'] == 'completed_sealed'
    assert seal['GT_read'] is False and seal['queries'] == len(seal['files']) == 64
    assert seal['fingerprint'] == lock['fingerprint'] == public['proof']['fingerprint'] == FINGERPRINT
    assert seal['lock_sha256'] == public['proof']['lock_sha256'] == sha(lock_path)
    assert public['proof']['seal_sha256'] == sha(seal_path)
    assert score_receipt['proof'] == public['proof']
    assert score_receipt['metrics_sha256'] == sha(trial / 'metrics.json')
    assert public['proof']['all64_validated_before_labels'] is True
    assert public['proof']['official_dense_checks'] == 64 * 7 == 448
    score_checks = verify_saved_scores(public)
    private_rows = read(trial / 'metrics.json')['rows']
    assert [r['id'] for r in private_rows] == lock['ids']
    assert [r['values'] for r in private_rows] == [r['values'] for r in public['queries_anonymous']]
    prior = read(ROOT / 'results/state_tuning_hc64/hc64.json')
    for row, old_row in zip(public['queries_anonymous'], prior['queries_anonymous']):
        assert row['query'] == old_row['query']
        for name in ['ERefStrong', 'GeometryOPD']:
            assert row['values'][name] == old_row['values'][name]
    old_root = ROOT / 'outputs/layout_reward_dev'
    old_seal = read(old_root / 'scope_hc2/sealed.json')
    receipt_costs = {name: [] for name in ARMS}
    trace_summaries = {name: [] for name in ARMS}
    confidences, proxy_rows = [], []
    proxy_positive_count = proxy_nonpositive_linear_count = proxy_candidate_count = 0
    all_weights, all_sources = {}, {**lock['sources'], **lock['frozen_sources']}
    for path, expected in all_sources.items():
        assert sha(ROOT / path) == expected
    for index, key in enumerate(lock['ids']):
        entry = seal['files'][key]
        path = within(trial, entry['path'])
        receipt_path = path.with_suffix('.json')
        receipt = read(receipt_path)
        assert sha(path) == entry['sha256'] == receipt['sha256']
        assert sha(receipt_path) == entry['receipt_sha256']
        assert receipt['GT_read'] is False and receipt['fingerprint'] == FINGERPRINT
        trace_path, evidence_path = within(trial, receipt['trace_path']), within(trial, receipt['evidence_path'])
        assert sha(trace_path) == receipt['trace_sha256'] and sha(evidence_path) == receipt['evidence_sha256']
        reference = within(old_root, old_seal['files'][key]['path'])
        assert receipt['reference_sha256'] == old_seal['files'][key]['sha256'] == sha(reference)
        traces = torch.load(trace_path, map_location='cpu', weights_only=False)
        evidence = torch.load(evidence_path, map_location='cpu', weights_only=False)
        assert list(traces) == ARMS and evidence['GT_read'] is False
        W = evidence['W'].float()
        W_sha = hashlib.sha256(W.numpy().tobytes()).hexdigest()
        observed, event, confidence = evidence['observed'].bool(), evidence['event'].bool(), evidence['confidence'].double()
        valid_unobserved = event & ~observed
        confidences.append(dict(query='H%03d' % (index + 1), evidence_seconds=receipt['evidence_seconds'],
            unobserved_nodes=int((~observed).sum()), event_unobserved_nodes=int(valid_unobserved.sum()),
            positive_confidence_event_unobserved_nodes=int(((confidence > 0) & valid_unobserved).sum()),
            confidence_all_unobserved_mean=float(confidence[~observed].mean()),
            confidence_event_unobserved_mean=float(confidence[valid_unobserved].mean()) if bool(valid_unobserved.any()) else None,
            dense_transport_nonanchor_row_mass_mean=float(W[~observed].sum(-1).mean())))
        assert W.shape == (len(event), 8) and torch.isfinite(W).all() and bool((W >= 0).all())
        torch.testing.assert_close(W.sum(-1).double()[~observed], (confidence * event)[~observed], atol=1e-7, rtol=1e-6)
        for name in ['DirectMulti', 'DirectMC']:
            assert torch.equal(traces[name]['initial_states'], traces['TrajectoryOPD']['initial_states'])
            assert traces[name]['initial_values'] == traces['TrajectoryOPD']['initial_values']
        assert torch.equal(traces['Direct']['initial_states'], traces['TrajectoryOPD']['initial_states'][:1])
        with np.load(path, allow_pickle=False) as prediction, np.load(reference, allow_pickle=False) as old_prediction:
            for field in ['frame_ids', 'positions8', 'Strong_interval']:
                assert np.array_equal(prediction[field], old_prediction[field])
            assert torch.equal(evidence['frame_ids'].cpu(), torch.as_tensor(prediction['frame_ids']))
            expected_event = (evidence['frame_ids'] >= prediction['Strong_interval'][0]) & (evidence['frame_ids'] < prediction['Strong_interval'][1])
            assert torch.equal(event, expected_event)
            assert np.array_equal(prediction['InitialModes'], traces['TrajectoryOPD']['initial_final'].numpy())
            for name in ARMS:
                arm, costs = traces[name], receipt['costs'][name]
                checks, rounds = arm['initial_checks'], arm['rounds']
                for flag in ['original_native_Source_bit_exact', 'original_E_bit_exact',
                             'all_network_parameters_frozen', 'all_network_parameter_grads_none',
                             'direct_content_at_all_physical_nodes']:
                    assert checks[flag] is True
                assert checks['W_sha256'] == W_sha and arm['evidence'] == receipt['evidence_summary'] == evidence['metadata']
                assert checks['full_reference'] == (name != 'TrajectoryOPD')
                if name != 'TrajectoryOPD':
                    assert checks['full_reference_forward_bit_exact'] is True
                assert np.array_equal(prediction[name], arm['final'].numpy())
                final_values = rounds[-1]['after']
                selected = max(range(len(arm['final_states'])), key=lambda c: final_values[c]['U'])
                assert selected == arm['best_component'] and torch.equal(arm['final_z'], arm['final_states'][selected])
                initial_best = max(range(len(arm['initial_values'])), key=lambda c: arm['initial_values'][c]['U'])
                ninitial = len(arm['initial_states'])
                sampled = name in ['TrajectoryOPD', 'DirectMC']
                proposed = sum(sum(r['counts']) for r in rounds) if sampled else len(rounds)
                validation_calls = sum(len(v['attempts']) for r in rounds for v in r['backtracking'])
                evaluations = ninitial + proposed + validation_calls
                accepts = sum(len(r['accepted_components']) for r in rounds)
                assert costs == arm['costs'] and costs['full_trajectory_evals'] == evaluations <= 1321
                assert costs['native_offset_decoder_calls'] == 2 * evaluations
                assert costs['updates'] == len(rounds) and costs['accepted_updates'] == accepts
                assert costs['decoder_backwards'] == (0 if name == 'TrajectoryOPD' else proposed)
                assert costs['setup_full_trajectory_evals'] == (1 if name == 'TrajectoryOPD' else 2)
                assert costs['setup_native_offset_decoder_calls'] == 2 * costs['setup_full_trajectory_evals']
                for record in rounds:
                    for backtrack in record['backtracking']:
                        component = backtrack['component']
                        if backtrack['accepted']:
                            assert record['after'][component]['U'] > record['before'][component]['U'] + 1e-8
                        else:
                            assert record['after'][component] == record['before'][component]
                values = final_values[selected]
                summary = dict(query='H%03d' % (index + 1), selected_component=selected,
                    initial_best_component=initial_best, selected_component_changed=selected != initial_best,
                    knot_state_L2=float(arm['final_z'].norm()), dense_content_L2=float(arm['final_eta'].norm()),
                    U=values['U'], proxy_U_gain_over_initial_readout=values['U'] - arm['initial_values'][initial_best]['U'])
                for field in ['obs', 'unobs', 'link', 'guard', 'd_mean', 'd_max']:
                    summary[field] = values[field]
                trace_summaries[name].append(summary)
                receipt_costs[name].append(costs)
        # Same initial mode zero and the restored full derivative of exactly U.
        opd, direct = traces['TrajectoryOPD'], traces['Direct']
        assert torch.equal(opd['initial_states'][0], direct['initial_states'][0])
        assert opd['initial_values'][0] == direct['initial_values'][0]
        first = opd['rounds'][0]
        assert direct['rounds'][0]['gradient_evaluation'] == direct['initial_values'][0]
        g = -direct['rounds'][0]['initial_loss_gradient_state'].double()
        delta = first['candidate_states'][0].double() - opd['initial_states'][0].double()
        actual = first['candidate_U'][0].double() - opd['initial_values'][0]['U']
        linear = (delta * g).sum((1, 2))
        assert all(bool(torch.isfinite(value).all()) for value in [g, delta, actual, linear])
        positive = actual > 1e-8
        contradiction = positive & (linear <= 0)
        count, positives, contradictions = len(actual), int(positive.sum()), int(contradiction.sum())
        proxy_candidate_count += count
        proxy_positive_count += positives
        proxy_nonpositive_linear_count += contradictions
        proxy_rows.append(dict(query='H%03d' % (index + 1), candidates=count, positive_actual_U=positives,
            positive_actual_U_nonpositive_initial_linear=contradictions,
            fraction_of_all_candidates=contradictions / count,
            fraction_of_positive_actual_U=contradictions / positives if positives else None))
        del traces, evidence, W, g, delta, actual, linear
    assert sha(ROOT / 'outputs/formal/state.json') == lock['formal_state_sha256']
    assert sha(ROOT / 'configs/default.json') == lock['registered_config_sha256']
    assert sha(seal_path) == public['proof']['seal_sha256'] and sha(lock_path) == public['proof']['lock_sha256']
    assert not torch.cuda.is_initialized()
    cost_error = 0.
    for name, saved_fields in public['costs']['arms'].items():
        for field, saved in saved_fields.items():
            values = np.asarray([r[field] for r in receipt_costs[name]], dtype=float)
            assert len(values) == saved['valid_queries'] == 64 and np.isfinite(values).all()
            computed = dict(mean=float(values.mean()), median=float(np.median(values)),
                            p90=float(np.quantile(values, .9)), maximum=float(values.max()))
            if 'total' in saved:
                computed['total'] = float(values.sum())
            cost_error = max(cost_error, *(abs(value - saved[field_name]) for field_name, value in computed.items()))
    assert cost_error < 1e-9
    positive_fractions = [r['fraction_of_positive_actual_U'] for r in proxy_rows if r['fraction_of_positive_actual_U'] is not None]
    positive_fraction_ci = None
    if positive_fractions:
        intervals = np.random.default_rng(20261021).integers(0, len(positive_fractions), (10000, len(positive_fractions)))
        fraction_array = np.asarray(positive_fractions)
        positive_fraction_ci = np.quantile(fraction_array[intervals].mean(1), [.025, .975]).tolist()
    proxy = dict(scope='Initial OPD component-zero actual candidates versus the same initial full-reference Direct gradient',
        actual_gain_threshold=1e-8, linear_nonpositive_threshold=0.,
        candidate_count=proxy_candidate_count, positive_actual_U=proxy_positive_count,
        positive_actual_U_nonpositive_initial_linear=proxy_nonpositive_linear_count,
        pooled_fraction_of_all_candidates=proxy_nonpositive_linear_count / proxy_candidate_count,
        pooled_fraction_of_positive_actual_U=proxy_nonpositive_linear_count / proxy_positive_count if proxy_positive_count else None,
        query_macro_fraction_of_all_candidates=stats([r['fraction_of_all_candidates'] for r in proxy_rows]),
        query_macro_fraction_of_positive_actual_U=stats(positive_fractions),
        query_macro_positive_fraction_CI95=positive_fraction_ci,
        interpretation='Finite nonlocal change of the frozen proxy utility versus its pointwise linear approximation; not GT correctness, causal proof of OPD advantage, or a method-selection rule',
        queries_anonymous=proxy_rows)
    trace_aggregates = {}
    for name, rows in trace_summaries.items():
        trace_aggregates[name] = {field: stats([r[field] for r in rows]) for field in rows[0] if field != 'query'}
        trace_aggregates[name]['selected_component_histogram'] = [sum(r['selected_component'] == c for r in rows) for c in range(3)]
    evidence_aggregates = {field: stats([r[field] for r in confidences]) for field in confidences[0] if field != 'query'}
    focus = {name: {field: public['paired']['TrajectoryOPD-' + name][field] for field in FIELDS}
             for name in ['ERefStrong', 'GeometryOPD', 'Direct', 'DirectMulti', 'DirectMC', 'InitialModes']}
    result = dict(status='all64_saved_trajectory_trace_and_score_audit_pass', queries=64,
        fingerprint=FINGERPRINT, all_positive_negative_unchanged_queries_retained=True,
        score_checks=score_checks, comparisons=focus, costs=public['costs'],
        evidence_confidence=evidence_aggregates, final_mean_displacement_and_rewards=trace_aggregates,
        initial_component_zero_nonlocal_proxy=proxy, anonymous_trace_summaries=trace_summaries,
        anonymous_evidence_summaries=confidences,
        checks=dict(all64_prediction_receipt_trace_evidence_and_reference_hashes=True,
                    same_saved_Strong_intervals=True, old64_baselines_exact=True,
                    all64_same_evidence_W_initial_modes_and_readout=True,
                    all64_zero_Source_E_and_fullref_forward_checks=True,
                    all64_cost_allowances_and_component_acceptance_counts=True,
                    saved_cost_summary_max_abs_error=cost_error,
                    source_code_hashes_unchanged=True, registered_method_and_formal_state_unchanged=True,
                    GT_annotation_reads=0, model_calls=0, decoder_calls=0, reward_evaluations=0,
                    new_candidates_generated=0, CUDA_initialized=False),
        provenance=dict(script_sha256=sha(__file__), source_scores_sha256=sha(public_path),
                        seal_sha256=sha(seal_path), lock_sha256=sha(lock_path),
                        score_receipt_sha256=sha(trial / 'score_receipt.json')))
    serialized = json.dumps(result, indent=2, allow_nan=False) + '\n'
    assert not re.search(r'/home/|/root/|/mnt/|\b(?:RC|BV)\d+\b|Bearer\s|hf_[A-Za-z0-9]{12,}|gh[pousr]_[A-Za-z0-9]{12,}', serialized)
    destination = output / 'analysis.json'
    temporary = destination.with_name(destination.name + '.tmp')
    temporary.write_text(serialized)
    temporary.replace(destination)
    print(json.dumps(dict(status=result['status'], score_checks=score_checks,
                         comparisons=focus, evidence_confidence=evidence_aggregates,
                         nonlocal_proxy={k: v for k, v in proxy.items() if k != 'queries_anonymous'}), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-after-score', action='store_true')
    args = parser.parse_args()
    if not args.run_after_score:
        parser.error('Explicit --run-after-score requires the complete sealed and scored HC64 trial')
    analyze()


if __name__ == '__main__':
    main()
