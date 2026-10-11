"""Additive CPU table readback correction; pinned GPU runtime remains intact."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import collections
import json
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts import finalize_stvg_opd_design_ablation_v1 as original
from scripts.stvg_opd_design_ablation_common_v1 import BASE, PUB, OLDPUB, read, write, sha, verify

REV = BASE / 'CPU_projection_revision002'
SPECS = [
    ('Full method', 'P3', 'on_policy', 'After_v'),
    ('w/o Action-Reward Correspondence', 'P2', 'shuffled_feedback', 'After_v'),
    ('w/o On-Policy Refresh', 'P2', 'frozen_rollout', 'After_v'),
    ('w/o Query Residual', 'P3', 'LN_only', 'After_v'),
    ('w/o LayerNorm Adaptation', 'P3', 'query_only', 'After_v'),
    ('w/o Slow-State Consolidation', 'P3', 'joint_alpha0', 'After_v'),
    ('Frozen source', 'P3', 'on_policy', 'Frozen_v')]
COLUMNS = [('hc2', 'cross_clean'), ('vidstg', 'cross_clean'),
           ('hc2', 'same_5percent'), ('vidstg', 'same_5percent')]
PASTED = {
    'Full method': [23.502, 18.337, 33.024, 25.769],
    'w/o Action-Reward Correspondence': [19.665, 9.517, 26.704, 15.547],
    'w/o On-Policy Refresh': [22.378, 14.477, 29.667, 23.190],
    'w/o Query Residual': [23.672, 17.873, 33.015, 25.349],
    'w/o LayerNorm Adaptation': [22.320, 15.015, 29.082, 24.331],
    'w/o Slow-State Consolidation': [23.264, 18.079, 32.807, 25.289],
    'Frozen source': [22.156, 14.466, 29.093, 23.594]}


def existing_table():
    result = {}; checks = []; corrections = []
    for label, phase, arm, field in SPECS:
        cells = []
        for index, (dataset, setting) in enumerate(COLUMNS):
            path = OLDPUB / f'{phase}_{dataset}_{setting}' / 'ROWS.jsonl.gz'
            rows = [r for r in original.rows_from(path) if r['arm'] == arm]
            groups = collections.defaultdict(list); identifiers = set()
            for row in rows:
                key = (row['query_ordinal'], row['condition'], row['order'])
                assert key not in identifiers
                identifiers.add(key); groups[row['source_id']].append(float(row[field]))
            assert len(groups) == 128 and len({len(v) for v in groups.values()}) == 1
            assert len(rows) == (256 if setting == 'cross_clean' else 640)
            independent = sum(sum(v) / len(v) for v in groups.values()) / len(groups)
            query_macro = sum(float(r[field]) for r in rows) / len(rows)
            stat = original.statistics(rows, [field])
            actual = stat['metrics'][field]['mean']
            assert abs(actual - independent) < 2e-15
            assert abs(actual - query_macro) < 2e-15
            mean_percent = actual * 100
            rendered = f'{mean_percent:.3f}'
            pasted = f'{PASTED[label][index]:.3f}'
            checks.append(dict(configuration=label, dataset=dataset, setting=setting,
                rows=len(rows), parents=128, rows_sha256=sha(path),
                independent_parent_mean_equal=True, balanced_query_mean_equal=True))
            if rendered != pasted:
                corrections.append(dict(configuration=label, dataset=dataset,
                    setting=setting, pasted_percent=pasted, actual_rendered_percent=rendered,
                    actual_unrounded_percent=mean_percent,
                    scope='display transcription/rounding only; original predictions and scores unchanged'))
            cells.append(dict(mean_percent=mean_percent, statistics=stat, rows_sha256=sha(path)))
        result[label] = cells
    assert len(checks) == 28
    assert [(r['configuration'], r['dataset'], r['setting'], r['actual_rendered_percent']) for r in corrections] == [
        ('w/o Action-Reward Correspondence', 'vidstg', 'cross_clean', '9.516'),
        ('Frozen source', 'hc2', 'same_5percent', '29.094')]
    receipt = dict(status='pass', scope='all original sealed component table cells independently reread',
        cells=28, corrections=corrections, checks=checks,
        old_scores_rewritten=False, old_predictions_rewritten=False,
        new_GT_read=False, new_model_calls=0, CPU_only=True, time=time.time())
    target = REV / 'ORIGINAL_TABLE_READBACK.json'
    if target.exists():
        previous = read(target)
        for key in ['status', 'cells', 'corrections', 'checks']:
            assert previous[key] == receipt[key]
    else:
        write(target, receipt)
    return result


def prepare():
    verify(); existing_table()
    path = REV / 'REVISION_RUNTIME.json'
    record = dict(status='additive_CPU_projection_revision_locked',
        original_runtime_sha256=sha(BASE / 'RUNTIME_LOCK.json'),
        original_finalize_sha256=sha(ROOT / 'scripts/finalize_stvg_opd_design_ablation_v1.py'),
        pins={'scripts/finalize_stvg_opd_design_ablation_cpu002.py':sha(Path(__file__))},
        original_table_readback_sha256=sha(REV / 'ORIGINAL_TABLE_READBACK.json'),
        capture_receipt_sha256=sha(BASE / 'recovery/CPU_existing_table_projection_001/CAPTURE_RECEIPT.json'),
        scope='CPU projection only; no GPU runtime, science, thresholds, bootstrap or old-row modification',
        CPU_only=True, new_GT_read=False, time=time.time())
    if path.exists():
        previous = read(path)
        for key in ['original_runtime_sha256', 'original_finalize_sha256', 'pins', 'original_table_readback_sha256', 'capture_receipt_sha256']:
            assert previous[key] == record[key]
    else:
        write(path, record)
    return record


def run():
    prepare(); revision = read(REV / 'REVISION_RUNTIME.json')
    original.existing_table = existing_table
    original.run()
    table = read(PUB / 'COMPONENT_TABLE.json')
    table['historical_display_corrections'] = read(REV / 'ORIGINAL_TABLE_READBACK.json')['corrections']
    table['CPU_revision_runtime_sha256'] = sha(REV / 'REVISION_RUNTIME.json')
    write(PUB / 'COMPONENT_TABLE.json', table)
    correction = ['','Historical table readback corrected two display values: shuffled-feedback Cross VidSTG 9.517 → 9.516, and frozen-source Corrupt HC2 29.093 → 29.094. All original predictions, per-query rows and underlying scores remain unchanged.']
    for name in ['REPORT.md', 'COMPONENT_TABLE.md']:
        with (PUB / name).open('a') as f: f.write('\n'.join(correction)+'\n')
    complete = read(BASE / 'CPU_COMPLETION.json')
    complete['CPU_revision_runtime_sha256'] = sha(REV / 'REVISION_RUNTIME.json')
    complete['original_table_readback_sha256'] = sha(REV / 'ORIGINAL_TABLE_READBACK.json')
    complete['output_files'] = {str(p.relative_to(ROOT)):sha(p) for p in PUB.rglob('*') if p.is_file()}
    write(BASE / 'CPU_COMPLETION.json', complete)
    write(REV / 'COMPLETION.json', dict(status='CPU_projection_and_plots_complete_pending_actual_root',
        CPU_only=True, actual_new_model_calls=0,
        CPU_completion_sha256=sha(BASE / 'CPU_COMPLETION.json'),
        revision_runtime_sha256=sha(REV / 'REVISION_RUNTIME.json'),
        original_GPU_runtime_unchanged=True, time=time.time()))


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'prepare':
        record = prepare(); print('CPU_PROJECTION_REVISION002_LOCKED', sha(REV / 'REVISION_RUNTIME.json'), flush=True)
    else:
        run()
