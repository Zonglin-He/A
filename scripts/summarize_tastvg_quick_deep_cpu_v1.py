"""Reconstruct posthoc diagnostic counts from anonymous saved scalar logs only."""
import hashlib
import json
import math
import sys
from pathlib import Path
import numpy as np


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(directory, quick_directory):
    directory, quick_directory = Path(directory), Path(quick_directory)
    steps = read(directory / 'DEEP_STEPS.json')
    references = read(directory / 'REFERENCE_ROWS.json')
    temporal = read(directory / 'TEMPORAL_ROWS.json')
    original = read(quick_directory / 'SPATIAL_STEP_ROWS.json')
    summary = read(directory / 'SUMMARY.json')
    index = {(r['condition'], r['order'], r['arrival']): r for r in references}
    findings, checks = {}, 0
    for group in ('corruption', 'clean'):
        keep = lambda r: (r['condition'] != 'clean') == (group == 'corruption')
        seq = [r for r in steps if keep(r) and 'reward_range' in r]
        first = [r for r in seq if r['step'] == 0]
        refs = [r for r in references if keep(r)]
        s = summary[group]
        assert len(refs) == s['scheduled']
        assert len(seq) == s['updated_steps']
        assert len(first) == s['first_updated_steps']
        checks += 3
        for key in ('reward_range', 'top_gap', 'q_top'):
            np.testing.assert_allclose(np.quantile([r[key] for r in first], [0, .25, .5, .75, 1]), s['first_step'][key]['values'], rtol=0, atol=1e-12)
            checks += 5
        for r in seq:
            assert math.isclose(r['post_v'] - r['pre_v'], r['delta_update'], abs_tol=1e-12)
            checks += 1
        strata = {}
        predicates = dict(
            no_scored_GT_reference=lambda r: r['inside_GT_frames'] == 0,
            scored_reference_IoU_lt_05=lambda r: r['reference_GT_IoU'] is not None and r['reference_GT_IoU'] < .5,
            scored_reference_IoU_gt_07=lambda r: r['reference_GT_IoU'] is not None and r['reference_GT_IoU'] > .7,
        )
        for name, predicate in predicates.items():
            rows = [r for r in first if predicate(index[r['condition'], r['order'], r['arrival']])]
            strata[name] = dict(cells=len(rows), harmed=sum(r['delta_update'] < -1e-10 for r in rows),
                               mean_current_update_delta_pp=100 * np.mean([r['delta_update'] for r in rows]) if rows else None)
        os = [r for r in original if keep(r) and r['step'] == 0 and r['rewards'] is not None]
        good = [r for r in os if r['delta_update'] < -1e-10 and r['teacher_top_best_v'] > r['pre_v'] + 1e-10]
        unique = [r for r in good if len(r['teacher_top_indices']) == 1]
        all_good = [r for r in good if min(r['candidate_v'][i] for i in r['teacher_top_indices']) > r['pre_v'] + 1e-10]
        assert len(good) == s['first_step']['useful_teacher_top_but_GT_harm']
        pair = s['teacher_pair_counts']
        assert all(sum(r[k] for r in first) == pair[k] for k in pair)
        checks += 4
        flat = [r for r in seq if r['flat_rewards']]
        assert len(flat) == s['flat_steps']['count']
        assert sum(r['global_gradient_norm'] > 0 for r in flat) == s['flat_steps']['nonzero_gradient']
        checks += 2
        times = [r for r in temporal if keep(r)]
        for r in times:
            assert int(np.argmax(r['candidate_scores'])) == r['selected']
            assert math.isclose(r['post_v'], r['all_candidates_v'][r['selected']], abs_tol=1e-12)
            checks += 2
        findings[group] = dict(reference_strata=strata,
            unique_teacher_winner_better_but_update_harm=len(unique),
            all_top_ties_better_but_update_harm=len(all_good),
            any_top_member_better_but_update_harm=len(good),
            flat_subset_of_any_top_better=sum(r['flat_rewards'] for r in good),
            decisive_pair_accuracy=pair['teacher_pairs_correct'] / (pair['teacher_pairs_correct'] + pair['teacher_pairs_wrong']),
            flat_step_mean_current_update_delta_pp=100 * np.mean([r['delta_update'] for r in flat]) if flat else None,
            temporal_tIoU_gt_05_but_no_joint_vIoU_gt_03=sum(max(r['all_candidates_t']) > .5 and r['oracle_v'] <= .3 for r in times),
            max_step_parameter_L2=max(r['step_norm'] for r in seq),
            unique_teacher_good_update_bad_cases=sorted(unique, key=lambda r: r['delta_update'])[:4])
    output = directory / 'SUPPLEMENTARY.json'
    if output.exists():
        assert read(output) == findings, 'Saved supplementary counts differ'
    else:
        output.write_text(json.dumps(findings, indent=2, allow_nan=False) + '\n')
    audit = dict(status='pass', scalar_checks=checks, scope='Anonymous scalar reconstruction; does not independently re-read GT or execute a model',
                 input_sha256={n: sha(directory / n) for n in ('DEEP_STEPS.json', 'REFERENCE_ROWS.json', 'TEMPORAL_ROWS.json', 'SUMMARY.json')},
                 original_step_sha256=sha(quick_directory / 'SPATIAL_STEP_ROWS.json'), supplementary_sha256=sha(output))
    readback = directory / 'SCALAR_READBACK.json'
    if readback.exists():
        assert read(readback) == audit, 'Saved scalar audit differs'
    else:
        readback.write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps(audit))


if __name__ == '__main__':
    run(*sys.argv[1:])
