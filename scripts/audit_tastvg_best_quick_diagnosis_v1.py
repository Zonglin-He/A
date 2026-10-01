"""Independent CPU scalar readback of the sealed quick pipeline diagnosis.

Accepts either the private scored directory or its anonymous public export.
No model, media, annotation or GPU is accessed.
"""
import collections
import hashlib
import json
import math
import sys
import time
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def changes(rows, before, after):
    delta = [r[after] - r[before] for r in rows]
    result = dict(
        cells=len(rows),
        degraded=sum(d < -1e-10 for d in delta),
        improved=sum(d > 1e-10 for d in delta),
        unchanged=sum(abs(d) <= 1e-10 for d in delta),
        gross_loss_pp=-100 * math.fsum(min(d, 0) for d in delta) / len(delta),
        gross_gain_pp=100 * math.fsum(max(d, 0) for d in delta) / len(delta),
    )
    for threshold in (.3, .5):
        n = sum(r[before] > threshold for r in rows)
        damaged = sum(r[before] > threshold >= r[after] for r in rows)
        result[str(threshold)] = dict(
            correct_before=n,
            correct_to_wrong=damaged,
            conditional_damage_rate=damaged / n if n else None,
            wrong_before=len(rows) - n,
            wrong_to_correct=sum(r[before] <= threshold < r[after] for r in rows),
        )
    return result


def run(directory):
    directory = Path(directory)
    rows = read(directory / 'ROWS.json')
    steps = read(directory / 'SPATIAL_STEP_ROWS.json')
    diagnosis = read(directory / 'PIPELINE_DIAGNOSIS.json')
    cohort = read(directory / 'COHORT.json')
    assert len(rows) == cohort['total']
    keys = lambda r: (r['parent'], r['order'], r['condition'])
    index = {keys(r): r for r in rows}
    assert len(index) == len(rows)
    by_arrival = collections.defaultdict(list)
    for step in steps:
        assert keys(step) in index
        by_arrival[keys(step)].append(step)
    checks = 0

    def equal(actual, expected):
        nonlocal checks
        if isinstance(expected, dict):
            assert set(actual) == set(expected), (set(actual), set(expected))
            for name, value in expected.items():
                equal(actual[name], value)
        elif isinstance(expected, float):
            assert abs(actual - expected) < 1e-10, (actual, expected)
            checks += 1
        else:
            assert actual == expected, (actual, expected)
            checks += 1

    for key, arrival in index.items():
        trace = by_arrival.get(key, [])
        assert bool(trace) == arrival['expert_scheduled']
        assert len(trace) <= cohort['params']['steps']
        for j, step in enumerate(trace):
            assert step['step'] == j
            if j:
                equal(step['pre_v'], trace[j - 1]['post_v'])
            equal(step['delta_update'], step['post_v'] - step['pre_v'])
        if trace:
            equal(trace[0]['pre_v'], arrival['slow_v'])
            equal(trace[-1]['post_v'], arrival['post_fixed_time_v'])
        equal(math.fsum(s['delta_update'] for s in trace), arrival['delta_post_fixed_time'])
        equal(arrival['delta_total'], arrival['delta_inherited_boxes'] + arrival['delta_inherited_interval'] + arrival['delta_temporal_rerank'])

    for name in ('full', 'outside_tuning_sources', 'tuning_sources'):
        predicate = lambda r: name == 'full' or r['tuning_source_exposed'] == (name == 'tuning_sources')
        for group in ('clean', 'corruption'):
            rr = [r for r in rows if predicate(r) and (r['condition'] != 'clean') == (group == 'corruption')]
            ss = [r for r in steps if predicate(r) and (r['condition'] != 'clean') == (group == 'corruption')]
            result = {}
            for component, before, after in (
                ('inherited_boxes', 'frozen_v', 'boxes_only_v'),
                ('inherited_interval', 'boxes_only_v', 'slow_v'),
                ('temporal_rerank', 'slow_v', 'final_v'),
                ('total', 'frozen_v', 'final_v'),
                ('post_update_fixed_time', 'slow_v', 'post_fixed_time_v'),
            ):
                result[component] = changes(rr, before, after)
                if component in ('inherited_boxes', 'temporal_rerank'):
                    result[component]['damaged_source_count'] = len({r['source_id'] for r in rr if r[after] < r[before] - 1e-12})
            ex = [r for r in rr if r['expert_scheduled']]
            result['temporal'] = {}
            for threshold in (.3, .5):
                result['temporal'][str(threshold)] = dict(
                    scheduled=len(ex),
                    correct_candidate_exists=sum(r['temporal_best_v'] > threshold for r in ex),
                    missed_correct_candidate=sum(r['temporal_best_v'] > threshold >= r['final_v'] for r in ex),
                    no_correct_candidate=sum(r['temporal_best_v'] <= threshold for r in ex),
                    correct_destroyed=sum(r['slow_v'] > threshold >= r['final_v'] for r in ex),
                    correct_rescued=sum(r['slow_v'] <= threshold < r['final_v'] for r in ex),
                    good_teacher_bad_contributor=sum(r['slow_v'] > threshold >= r['final_v'] and r['best_teacher_t'] > .5 >= r['winning_teacher_t'] for r in ex),
                )
            result['spatial_by_step'] = {}
            for j in sorted({r['step'] for r in ss}):
                stage = [r for r in ss if r['step'] == j]
                thresholds = {}
                for threshold in (.3, .5):
                    thresholds[str(threshold)] = dict(
                        support_correct=sum(r['oracle_v'] > threshold for r in stage),
                        expert_top_has_no_correct=sum(r['oracle_v'] > threshold >= r['teacher_top_best_v'] and bool(r['teacher_top_indices']) for r in stage),
                        empty_evidence_with_correct_support=sum(r['oracle_v'] > threshold and r['valid_frames'] == 0 for r in stage),
                        useful_top_but_update_wrong=sum(r['teacher_top_best_v'] > threshold >= r['post_v'] for r in stage),
                    )
                result['spatial_by_step'][str(j)] = dict(
                    cells=len(stage),
                    empty_evidence=sum(r['valid_frames'] == 0 for r in stage),
                    flat_rewards=sum(r['flat_rewards'] for r in stage),
                    loss_down_but_GT_harm=sum(r['loss_decreased'] and r['delta_update'] < -1e-12 for r in stage),
                    transitions=changes(stage, 'pre_v', 'post_v'),
                    thresholds=thresholds,
                )
            equal(diagnosis[name][group], result)
    report = dict(status='pass', cells=len(rows), spatial_steps=len(steps), scalar_checks=checks,
                  net_fixed_time_trace_checked=True, all_diagnostic_tables_reconstructed=True,
                  model_or_GT_read=False, GPU_used=False,
                  inputs={n: digest(directory / n) for n in ('COHORT.json', 'ROWS.json', 'SPATIAL_STEP_ROWS.json', 'PIPELINE_DIAGNOSIS.json')},
                  script_sha256=digest(Path(__file__)), time=time.time())
    (directory / 'DIAGNOSIS_ROOT_READBACK.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    run(sys.argv[1])
