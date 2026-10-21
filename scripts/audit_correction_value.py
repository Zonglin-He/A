"""Saved-state propagation audit; no rollout, Q, model or label evaluation.

Requires the private sealed correction-value traces and the already-scored
public report. Only differences of saved memories/features/probabilities/actions
and existing forced-node IoUs are summarized. No scientific branch is created.
"""
from __future__ import annotations
import argparse
from collections import Counter
import gc
import hashlib
import json
import os
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import numpy as np
import torch

ROOT = Path(os.environ.get('STVG_PROJECT_ROOT', Path(__file__).resolve().parents[1])).expanduser().resolve()


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(values):
    x = np.asarray(values, dtype=float)
    assert np.isfinite(x).all()
    return dict(count=len(x), mean=float(x.mean()) if len(x) else None,
        median=float(np.median(x)) if len(x) else None,
        p10=float(np.quantile(x, .1)) if len(x) else None,
        p90=float(np.quantile(x, .9)) if len(x) else None,
        maximum=float(x.max()) if len(x) else None,
        nonzero=int((x != 0).sum()), above1e_12=int((x > 1e-12).sum()),
        above1e_6=int((x > 1e-6).sum()))


def _new_group():
    return {k: [] for k in ('memory_before_L2', 'valid_logp_max_abs', 'probability_L1',
        'relative_logit_change_max_abs', 'memory_term_relative_change_max_abs',
        'previous_feature_term_relative_change_max_abs', 'previous_link_term_relative_change_max_abs',
        'base_sampled_winner_margin', 'actual_action_changed', 'noise_free_argmax_changed')}


def audit(root, trial, report_path):
    torch.set_num_threads(1)
    lock, seal, sr, public = [read(p) for p in
        (trial / 'lock.json', trial / 'sealed.json', trial / 'score_receipt.json', report_path)]
    assert seal['queries'] == public['queries'] == 64 and len(seal['files']) == 64
    assert sha(report_path) == sr['public_json_sha256']
    assert sha(trial / 'sealed.json') == sr['proof']['seal_sha256']
    assert sha(trial / 'lock.json') == sr['proof']['lock_sha256']
    assert sr['proof'] == public['proof'] and sr['proof']['GT_used_to_select_interventions'] is False
    for path, digest in lock['sources'].items():
        assert sha(root / path) == digest
    assert sha(root / 'outputs/formal/state.json') == lock['formal_state_sha256']
    assert sha(root / 'configs/default.json') == lock['registered_config_sha256']
    groups = ('t_plus_1', 't_plus_3', 't_plus_1_active_unobserved', 't_plus_3_active_unobserved',
              'all_active_unobserved_tail', 'far_active_unobserved')
    acc = {branch: dict(forced_after_memory_L2=[], forced_box_changed=[],
            groups={g: _new_group() for g in groups}, per_intervention=[])
           for branch in ('Teacher', 'Random')}
    immediate = {group: {pair: [] for pair in ('Teacher-Student', 'Random-Student', 'Teacher-Random')}
                 for group in ('all', 'two_actions', 'three_actions', 'policy_disagreement', 'noise_only_disagreement')}
    instant_cases = []
    counts = Counter()
    for index, item in enumerate(lock['inputs']):
        key = item['id']
        receipt = read(trial / 'receipts' / f'{key}.json')
        entry = seal['files'][key]
        assert sha(trial / entry['path']) == entry['sha256'] == receipt['prediction_sha256']
        assert sha(trial / entry['receipt_path']) == entry['receipt_sha256']
        assert sha(trial / receipt['trace_path']) == receipt['trace_sha256']
        saved = torch.load(trial / receipt['trace_path'], map_location='cpu', weights_only=False)
        points = public['queries_anonymous'][index]['points']
        assert len(saved['branches']) == receipt['interventions'] == len(points)
        assert saved['GT_read'] is False and saved['parameter_updates'] == 0
        base = saved['base']
        counts['queries'] += 1
        counts['queries_with_interventions'] += int(bool(points))
        counts['interventions'] += len(points)
        for intervention, (branch, scored) in enumerate(zip(saved['branches'], points)):
            point, paths = branch['point'], branch['paths']
            node = point['node']
            T = len(base['actions'])
            active, observed = base['active'].numpy(), base['observed'].numpy()
            assert scored['intervention'] == intervention
            assert scored['time_bin'] == point['time_bin']
            assert scored['legal_action_count'] == len(point['valid_actions'])
            assert scored['disagreement_type'] == point['disagreement_type']
            assert point['teacher_action'] != point['student_action']
            for name, path in paths.items():
                assert torch.equal(path['phi'], base['phi']) and torch.equal(path['noise'], base['noise'])
                for field in ('features', 'student_log_probs', 'effective_valid', 'memory_before',
                        'memory_after', 'previous_action_probs', 'previous_features', 'actions', 'boxes', 'omega'):
                    assert torch.equal(path[field][:node], base[field][:node])
                    if name == 'Student':
                        assert torch.equal(path[field], base[field])
                for field in ('memory_before', 'previous_action_probs', 'previous_features',
                              'features', 'student_log_probs'):
                    assert torch.equal(path[field][node], base[field][node])
                assert not bool(path['actions'][~path['active']].any())
                assert torch.equal(path['memory_before'][~path['active']],path['memory_after'][~path['active']])
                counts['verified_branches'] += 1
            if point['teacher_action'] == point['random_action']:
                counts['Teacher_equals_Random_interventions'] += 1
                for field in ('features', 'student_log_probs', 'memory_before', 'memory_after', 'actions', 'boxes'):
                    assert torch.equal(paths['Teacher'][field], paths['Random'][field])

            source = paths['Student']
            nodes = np.arange(T)
            primary = (nodes > node) & active & ~observed
            masks = dict(t_plus_1=nodes == node + 1, t_plus_3=nodes == node + 3,
                t_plus_1_active_unobserved=(nodes == node + 1) & active & ~observed,
                t_plus_3_active_unobserved=(nodes == node + 3) & active & ~observed,
                all_active_unobserved_tail=primary,
                far_active_unobserved=(nodes > node + 3) & active & ~observed)
            valid = source['effective_valid'].numpy()
            lp_source = source['student_log_probs'].numpy()
            safe_source = np.where(valid, lp_source, 0.)
            phi = source['phi'].numpy()
            sample_scores = np.where(valid, lp_source + source['noise'].numpy(), -np.inf)
            ordered = np.sort(sample_scores, axis=1)
            winner_margin = ordered[:, -1] - ordered[:, -2]
            # E-only margins are unbounded, not empirical finite margins.
            winner_margin[~np.isfinite(winner_margin)] = 0.
            current_stats = {}
            for name in ('Teacher', 'Random'):
                path, a = paths[name], acc[name]
                mem_l2 = np.linalg.norm((path['memory_before'] - source['memory_before']).numpy(), axis=-1)
                force_l2 = float(torch.linalg.vector_norm(path['memory_after'][node] - source['memory_after'][node]))
                a['forced_after_memory_L2'].append(force_l2)
                box_changed = not torch.equal(path['boxes'][node], source['boxes'][node])
                a['forced_box_changed'].append(int(box_changed))
                lp = path['student_log_probs'].numpy()
                safe = np.where(valid, lp, 0.)
                logpdiff = np.max(np.abs(safe - safe_source), axis=-1)
                pdiff = np.abs(np.exp(lp) - np.exp(lp_source)).sum(-1)
                df = (path['features'] - source['features']).numpy()
                assert np.array_equal(df[:, :, 3:], np.zeros_like(df[:, :, 3:]))
                dlogits = (df * phi).sum(-1)
                relative = dlogits - dlogits[:, :1]
                relative_saved = (safe - safe_source) - (safe - safe_source)[:, :1]
                assert np.max(np.abs((relative - relative_saved)[valid]), initial=0) < 5e-15
                relative = np.where(valid, relative, 0.)
                dparts = [df[:, :, i] * phi[i] for i in range(3)]
                dparts = [np.where(valid, part - part[:, :1], 0.) for part in dparts]
                changes = (path['actions'] != source['actions']).numpy()
                greedy_changes = np.argmax(lp, axis=-1) != np.argmax(lp_source, axis=-1)
                metrics = dict(memory_before_L2=mem_l2, valid_logp_max_abs=logpdiff,
                    probability_L1=pdiff, relative_logit_change_max_abs=np.abs(relative).max(-1),
                    memory_term_relative_change_max_abs=np.abs(dparts[0]).max(-1),
                    previous_feature_term_relative_change_max_abs=np.abs(dparts[1]).max(-1),
                    previous_link_term_relative_change_max_abs=np.abs(dparts[2]).max(-1),
                    base_sampled_winner_margin=winner_margin,
                    actual_action_changed=changes.astype(float),noise_free_argmax_changed=greedy_changes.astype(float))
                for group, mask in masks.items():
                    for field, values in metrics.items():
                        a['groups'][group][field].extend(values[mask].tolist())
                current_stats[name] = dict(forced_memory_changed=force_l2 > 1e-12,
                    tail_nodes=int(primary.sum()), tail_changed_actions=int(changes[primary].sum()),
                    far_changed_actions=int(changes[masks['far_active_unobserved']].sum()),
                    tail_memory_changed_nodes=int((mem_l2[primary] > 1e-12).sum()),
                    tail_probabilities_changed_nodes=int((logpdiff[primary] > 1e-12).sum()))
                a['per_intervention'].append(current_stats[name])
                expected = scored['action_changes']['active_unobserved_tail'][name+'-Student']
                assert expected == int(changes[primary].sum())
            ious = scored['forced_node_IoU']
            if ious['Student'] is not None:
                assert all(ious[name] is not None for name in ('Student', 'Teacher', 'Random'))
                strata = ('all', 'two_actions' if len(point['valid_actions']) == 2 else 'three_actions',
                          point['disagreement_type'])
                for pair in immediate['all']:
                    first, second = pair.split('-')
                    delta = 100 * (ious[first] - ious[second])
                    for stratum in strata:
                        immediate[stratum][pair].append(delta)
                instant_cases.append(dict(query=f'H{index+1:03d}',intervention=intervention,
                    Teacher_minus_Student_pp=100*(ious['Teacher']-ious['Student']),
                    Teacher_tail_changed_actions=current_stats['Teacher']['tail_changed_actions'],
                    Teacher_memory_changed=current_stats['Teacher']['forced_memory_changed']))
        del saved, base
        gc.collect()
    assert counts['queries'] == 64 and counts['interventions'] == public['execution']['selected_interventions'] == 237
    assert counts['queries_with_interventions'] == public['execution']['queries_with_interventions']
    assert counts['Teacher_equals_Random_interventions'] == public['execution']['Teacher_equals_Random_interventions']
    result = {}
    for name, a in acc.items():
        per = a['per_intervention']
        result[name] = dict(forced_after_memory_L2=stats(a['forced_after_memory_L2']),
            forced_boxes_changed=sum(a['forced_box_changed']),
            interventions_with_later_memory_change=sum(r['tail_memory_changed_nodes'] > 0 for r in per),
            interventions_with_later_probability_change=sum(r['tail_probabilities_changed_nodes'] > 0 for r in per),
            interventions_with_later_action_change=sum(r['tail_changed_actions'] > 0 for r in per),
            groups={group: {field: stats(values) for field, values in fields.items()}
                    for group, fields in a['groups'].items()})
        actual = result[name]['groups']['all_active_unobserved_tail']['actual_action_changed']
        assert actual['count'] == public['actual_action_changes']['active_unobserved_tail']['node_evaluations']
        assert actual['nonzero'] == public['actual_action_changes']['active_unobserved_tail']['comparisons'][name+'-Student']['changed_actions']
    immediate_summary = {}
    for group, values in immediate.items():
        immediate_summary[group] = {}
        for pair, deltas in values.items():
            d = np.asarray(deltas, dtype=float)
            immediate_summary[group][pair] = dict(valid_forced_GT_points=len(d),
                mean_delta_pp=float(d.mean()) if len(d) else None,
                median_delta_pp=float(np.median(d)) if len(d) else None,
                improved=int((d > 1e-8).sum()), declined=int((d < -1e-8).sum()),
                unchanged=int((abs(d) <= 1e-8).sum()),
                improved_over5pp=int((d > 5).sum()), declined_over5pp=int((d < -5).sum()))
    teacher_improved = [r for r in instant_cases if r['Teacher_minus_Student_pp'] > 1e-8]
    assert not torch.cuda.is_initialized()
    return dict(status='saved_correction_value_state_flow_audit_complete', queries=64,
        GT_read=False, existing_scored_IoUs_read=True, new_paths=0, new_Q_calls=0,
        new_models=0, new_GT_scores=0, counts=dict(counts),
        report_sha256=sha(report_path), score_receipt_sha256=sha(trial / 'score_receipt.json'),
        seal_sha256=sha(trial / 'sealed.json'), source_hashes_verified=len(lock['sources']),
        formal_and_registered_config_unchanged=True, branches=result,
        forced_node_IoU_existing_scores=immediate_summary,
        locally_improved_Teacher_points=len(teacher_improved),
        locally_improved_Teacher_points_without_tail_action_change=sum(r['Teacher_tail_changed_actions']==0 for r in teacher_improved),
        locally_improved_Teacher_points_with_forced_memory_change=sum(r['Teacher_memory_changed'] for r in teacher_improved),
        primary_tail_comparisons=public['aggregate']['all']['groups']['active_unobserved_tail']['paired'],
        decision=public['decision'],
        metric_units='State/probability magnitudes are pooled intervention-node evaluations; forced IoU statistics use only existing finite forced-point scores.',
        limits=['Saved-state contrast is not a new rollout, a causal module ablation, or a new GT evaluation.',
            'Overlapping tails and multiple interventions repeat nodes; node counts are not independent query samples.',
            'Probability differences do not by themselves show a different chosen action or a better box.',
            'Zero E-only winner margins represent excluded infinite margins, not a tie.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default=str(ROOT))
    parser.add_argument('--trial', default='outputs/correction_value_hc64')
    parser.add_argument('--report', default='results/correction_value_hc64/report.json')
    parser.add_argument('--output', default='outputs/correction_value_hc64/audits/state_flow.json')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if not args.run:
        parser.print_help()
        return
    root = Path(args.root).expanduser().resolve()
    paths = [Path(value).expanduser() for value in (args.trial, args.report, args.output)]
    trial, report, output = [p.resolve() if p.is_absolute() else root / p for p in paths]
    result = audit(root, trial, report)
    output.parent.mkdir(exist_ok=True, parents=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(status=result['status'], output_sha256=sha(output), counts=result['counts'],
        Teacher_forced_memory=result['branches']['Teacher']['forced_after_memory_L2'],
        Teacher_tail=result['branches']['Teacher']['groups']['all_active_unobserved_tail'],
        Teacher_far=result['branches']['Teacher']['groups']['far_active_unobserved'],
        forced_node_IoU=result['forced_node_IoU_existing_scores']['all']), indent=2))


if __name__ == '__main__':
    main()
