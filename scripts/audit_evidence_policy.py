"""Reproduce the saved HC64 evidence-policy mechanism audit without scoring.

This is the portable copy of the CPU here-document that produced the existing
completed_mechanism.json receipt. It reads sealed predictions, traces, hashes
and already-scored public metrics. It never reads labels, constructs a bank,
calls Q, generates paths/candidates, or loads a model. Private sealed artifacts
must already exist locally; they are not included in the public export.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import json
import os
from pathlib import Path
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import numpy as np
import torch


ROOT = Path(os.environ.get('STVG_PROJECT_ROOT', Path(__file__).resolve().parents[1])).expanduser().resolve()


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve(root, value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def stats(values):
    a = np.asarray(values, dtype=float)
    return dict(count=len(a), mean=float(a.mean()) if len(a) else None,
        median=float(np.median(a)) if len(a) else None,
        p10=float(np.quantile(a, .1)) if len(a) else None,
        p90=float(np.quantile(a, .9)) if len(a) else None,
        minimum=float(a.min()) if len(a) else None,
        maximum=float(a.max()) if len(a) else None)


def audit(root, trial, report_json, report_md):
    """Only arithmetic on existing signed metrics and saved tensor histories."""
    torch.set_num_threads(1)
    p = read(report_json)
    sr = read(trial / 'score_receipt.json')
    lock = read(trial / 'lock.json')
    seal = read(trial / 'sealed.json')
    assert p['queries'] == p['parents'] == 64 and len(p['queries_anonymous']) == 64
    assert sr['public_json_sha256'] == sha(report_json)
    assert sr['public_md_sha256'] == sha(report_md)
    assert sr['proof'] == p['proof'] and sr['proof']['official_dense_checks'] == 768
    assert sr['proof']['official_dense_max_absolute_error'] <= 1.2e-15
    assert sr['proof']['reused_baseline_metric_max_absolute_error'] == 0
    assert sr['proof']['seal_sha256'] == sha(trial / 'sealed.json')
    assert sr['proof']['lock_sha256'] == sha(trial / 'lock.json')
    assert set(seal['files']) == set(lock['ids']) and len(seal['files']) == 64
    assert seal['queries'] == 64
    for file, digest in lock['sources'].items():
        assert sha(root / file) == digest
    assert sha(root / 'outputs/formal/state.json') == lock['formal_state_sha256']
    assert sha(root / 'configs/default.json') == lock['registered_config_sha256']

    rows, new = p['queries_anonymous'], p['new_arms']
    metricmap = dict(m_vIoU='v', m_tIoU='t', m_sIoU='s',
        m_sIoU_observed_actual='observed_actual_sIoU',
        m_sIoU_unobserved_actual='unobserved_actual_sIoU')
    for name in p['arm_order']:
        for outfield, field in metricmap.items():
            if outfield not in p['means'][name]:
                continue
            values = np.array([r['values'][name][field] for r in rows
                               if r['values'][name][field] is not None])
            assert len(values) == 64
            assert abs(float(values.mean()) - p['means'][name][outfield]) < 2e-15

    # Recompute the existing paired summaries from public signed scores only.
    # This does not evaluate any prediction against labels.
    rng = np.random.default_rng(20261021)
    draws = rng.integers(0, 64, (10000, 64))
    pairchecks = 0
    for key, fields in p['paired'].items():
        first, second = key.split('-', 1)
        for field, summary in fields.items():
            d = 100 * np.array([r['values'][first][field] - r['values'][second][field]
                                for r in rows])
            assert summary['valid_parents'] == 64
            assert abs(d.mean() - summary['mean_pp']) < 1e-13
            assert summary['improved'] == int((d > 1e-8).sum())
            assert summary['declined'] == int((d < -1e-8).sum())
            assert summary['unchanged'] == int((abs(d) <= 1e-8).sum())
            assert summary['declined_over5pp'] == int((d < -5).sum())
            ci = np.quantile(d[draws].mean(1), [.025, .975])
            assert np.max(abs(ci - summary['CI95_pp'])) < 1e-13
            pairchecks += 1
    assert len(p['signed_changes']) == 64 * 31
    for signed in p['signed_changes']:
        row = rows[int(signed['query'][1:]) - 1]
        d = 100 * (row['values'][signed['method']]['v'] - row['values'][signed['baseline']]['v'])
        assert abs(d - signed['vIoU_delta_pp']) < 1e-13
        assert signed['severe_vIoU_drop_over5pp'] == (d < -5)

    acc = {name: dict(outer=Counter(), firstouterrollback=0, firstinnerrollback=0,
        full30=0, updates=[], accepted=[], weights=[], L2=[], reacquisitions=0,
        reacq_nonzero_weights=0, reacq_changed_hard_action_history=0,
        changed_history_queries=0, allzero_final=0, margin_rows=[], margin_querymeans=[],
        count_eligible=0, count_positive=0, count_newpair_positive=0, count_bridge=0,
        count_legacyalone_positive=0, observed_replaced=0, unobserved_replaced=0,
        active_unobserved_replaced=0, rejected=0) for name in new}
    for item in lock['inputs']:
        key = item['id']
        receipt = read(trial / 'receipts' / f'{key}.json')
        entry = seal['files'][key]
        assert sha(trial / entry['path']) == entry['sha256'] == receipt['sha256']
        assert sha(trial / entry['receipt_path']) == entry['receipt_sha256']
        assert sha(trial / receipt['trace_path']) == receipt['trace_sha256']
        trace = torch.load(trial / receipt['trace_path'], map_location='cpu', weights_only=False)
        for name in new:
            out, a = trace[name], acc[name]
            cost, saved, rounds = out['costs'], out['trace'], out['rounds']
            outer = cost['outer_iterations']
            phi, path = out['finaltheta'].numpy(), saved['final_rollout']
            assert cost == receipt['costs'][name] and 1 <= len(rounds) <= 30
            a['outer'][outer] += 1
            a['updates'].append(len(rounds))
            a['accepted'].append(cost['accepted_updates'])
            a['weights'].append(phi.copy())
            a['L2'].append(float(np.linalg.norm(phi)))
            a['allzero_final'] += int(not phi.any())
            a['firstouterrollback'] += int(outer == 1 and rounds[-1]['rolled_back'])
            a['firstinnerrollback'] += int(rounds[0]['rolled_back'])
            a['full30'] += int(len(rounds) == 30 and all(r['accepted'] for r in rounds))
            a['rejected'] += cost['rejected_updates']
            assert rounds[0]['before'].eq(0).all()
            if name.endswith('_OPD'):
                changed = False
                for outerindex in range(1, outer):
                    a['reacquisitions'] += 2
                    first = next(r for r in rounds if r['outer'] == outerindex)
                    a['reacq_nonzero_weights'] += int(bool(first['before'].ne(0).any()))
                    now = saved['outer_acquisitions'][outerindex]
                    old = saved['outer_acquisitions'][outerindex - 1]
                    differs = any(not torch.equal(x['actions'], y['actions']) for x, y in zip(now, old))
                    a['reacq_changed_hard_action_history'] += int(differs)
                    changed |= differs
                a['changed_history_queries'] += int(changed)

            active, observed = path['active'].numpy(), path['observed'].numpy()
            valid, logp = path['effective_valid'].numpy(), path['student_log_probs'].numpy()
            features, prior = path['features'].numpy(), path['prior_log_probs'].numpy()
            actions = path['actions'].numpy()
            assert not actions[~active].any()
            replaced = actions != 0
            a['observed_replaced'] += int((replaced & observed).sum())
            a['unobserved_replaced'] += int((replaced & ~observed).sum())
            a['active_unobserved_replaced'] += int((replaced & active & ~observed).sum())
            eligible = active & ~observed & valid[:, 1:].any(-1)
            nodes = np.flatnonzero(eligible)
            alternatives = np.where(valid[:, 1:], logp[:, 1:], -np.inf).argmax(-1) + 1
            alternative = alternatives[nodes]
            contribution = (features[nodes, alternative] - features[nodes, 0]) * phi[None]
            barrier = prior[nodes, 0] - prior[nodes, alternative]
            margin = logp[nodes, alternative] - logp[nodes, 0]
            legacy, newpair = contribution[:, :6].sum(-1), contribution[:, 6:].sum(-1)
            assert np.max(abs(contribution.sum(-1) - barrier - margin), initial=0) < 3e-15
            positive = margin > 0
            a['count_eligible'] += len(nodes)
            a['count_positive'] += int(positive.sum())
            a['count_newpair_positive'] += int((newpair > 0).sum())
            a['count_bridge'] += int((positive & (legacy - barrier <= 0)).sum())
            a['count_legacyalone_positive'] += int((positive & (legacy - barrier > 0)).sum())
            block = np.column_stack((contribution, legacy, newpair, barrier, margin, positive))
            a['margin_rows'].append(block)
            if len(nodes):
                a['margin_querymeans'].append(block[:, :12].mean(0))
            assert a['unobserved_replaced'] == a['active_unobserved_replaced']
            del out, saved, path
        del trace
        gc.collect()

    summary = {}
    for name, a in acc.items():
        weights = np.stack(a['weights'])
        node, macro = np.concatenate(a['margin_rows']), np.stack(a['margin_querymeans'])
        replacements = p['optimization'][name]['output_replacements']
        assert replacements['observed']['replaced_nodes'] == a['observed_replaced']
        assert replacements['unobserved']['replaced_nodes'] == a['unobserved_replaced']
        reported = p['optimization'][name]['stages']['final']['active_unobserved']['positive_margin_nodes']
        assert reported == a['count_positive'] == a['active_unobserved_replaced']
        assert abs(np.mean(a['updates']) - p['costs']['arms'][name]['updates']['mean']) < 1e-13
        summary[name] = dict(
            outer_iterations_distribution={str(k): v for k, v in sorted(a['outer'].items())},
            first_outer_rollback_queries=a['firstouterrollback'],
            first_inner_rollback_queries=a['firstinnerrollback'],
            complete30accepted_queries=a['full30'], updates=stats(a['updates']),
            accepted_updates=stats(a['accepted']), final_weights_L2=stats(a['L2']),
            zero_final_weight_queries=a['allzero_final'],
            actual_reacquired_history_paths_after_first_outer=a['reacquisitions'],
            reacquisition_outer_entries_with_nonzero_phi=a['reacq_nonzero_weights'],
            reacquisition_outer_entries_with_changed_actions=a['reacq_changed_hard_action_history'],
            queries_with_changed_reacquisition_actions=a['changed_history_queries'],
            weights=[dict(index=i, feature=p['config']['policy_features'][i], **stats(weights[:, i]),
                positive_queries=int((weights[:, i] > 0).sum()),
                negative_queries=int((weights[:, i] < 0).sum())) for i in range(8)],
            observed_replaced_nodes=a['observed_replaced'],
            unobserved_replaced_nodes=a['unobserved_replaced'],
            active_unobserved_eligible_nodes=a['count_eligible'],
            positive_margin_nodes=a['count_positive'],
            new_pair_positive_contribution_nodes=a['count_newpair_positive'],
            selected_alternative_bridge_nodes=a['count_bridge'],
            selected_alternative_legacy_alone_positive_nodes=a['count_legacyalone_positive'],
            margin_component_query_macro_mean=macro.mean(0).tolist(),
            margin_component_pooled_mean=node[:, :12].mean(0).tolist())
    focus = {}
    for first, second in [('A_OPD', 'ERefStrong'), ('B_OPD', 'ERefStrong'),
            ('A_OPD', 'A_Direct'), ('B_OPD', 'B_Direct'), ('A_OPD', 'A_FixedKD'),
            ('B_OPD', 'B_FixedKD'), ('B_OPD', 'A_OPD')]:
        key = first + '-' + second
        focus[key] = {f: p['paired'][key][f]
                      for f in ('v', 'observed_actual_sIoU', 'unobserved_actual_sIoU')}
    assert not torch.cuda.is_initialized()
    return dict(status='pass_saved_results_and_trace_mechanism_audit_no_new_scoring',
        queries=64, GT_read=False, new_Q_calls=0, new_paths=0, new_candidates=0, new_models=0,
        score_proof=sr['proof'], public_json_sha256=sr['public_json_sha256'],
        public_md_sha256=sr['public_md_sha256'], pair_field_checks=pairchecks,
        signed_rows_verified=len(p['signed_changes']), mechanism=summary, focus_results=focus,
        margin_component_order=[*p['config']['policy_features'], 'legacy_six_sum',
            'new_two_sum', 'prior_barrier', 'signed_margin'],
        interpretation_limits=[
            'Same-history logit decomposition, not a counterfactual rollout or causal feature ablation.',
            'A new outer acquisition also changes its fixed noise pair, so changed actions cannot be attributed only to phi.',
            'H4 reference uses the frozen instantaneous reference, not continuation under learned student weights.',
            'No original labels or new IoU evaluations were read or computed.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default=str(ROOT))
    parser.add_argument('--trial', default='outputs/evidence_policy_hc64')
    parser.add_argument('--report-json', default='results/evidence_policy_hc64/report.json')
    parser.add_argument('--report-md', default='results/evidence_policy_hc64/report.md')
    parser.add_argument('--output', default='results/evidence_policy_hc64/mechanism.json')
    parser.add_argument('--run', action='store_true', help='Explicitly run the saved-artifact CPU audit')
    args = parser.parse_args()
    if not args.run:
        parser.print_help()
        return
    root = Path(args.root).expanduser().resolve()
    trial, report_json, report_md, output = [resolve(root, value) for value in
        (args.trial, args.report_json, args.report_md, args.output)]
    begin = time.perf_counter()
    result = audit(root, trial, report_json, report_md)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(status=result['status'], queries=result['queries'],
        pair_field_checks=result['pair_field_checks'], output_sha256=sha(output),
        seconds=time.perf_counter() - begin)))


if __name__ == '__main__':
    main()
