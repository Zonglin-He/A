#!/usr/bin/env python3
"""Recompute every metric and summarize only completed, aligned comparisons."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts.run_temporal_anchor_spatial_v1 import write, sha
from vg_tta.metrics import compute_stvg_metrics, interval_from_logits


def read(path):
    return json.loads(Path(path).read_text())


def summarize(records, reference):
    assert [r['sample_index'] for r in records] == [r['sample_index'] for r in reference]
    sources = sorted({r['source'] for r in records})
    matrix = np.array([[np.mean([r[m] for r in records if r['source'] == source])
                        for m in ['vIoU_corrected', 'sIoU', 'tIoU', 'vIoU_legacy']]
                       for source in sources]) * 100
    base = np.array([np.mean([r['vIoU_corrected'] for r in reference if r['source'] == source])
                     for source in sources]) * 100
    delta = matrix[:, 0] - base
    rng = np.random.default_rng(20260907)
    sampled = rng.integers(len(sources), size=(10000, len(sources)))
    ci = np.quantile(delta[sampled].mean(1), [.025, .975]).tolist()
    return {'queries': len(records), 'sources': len(sources), 'vIoU': float(matrix[:, 0].mean()),
            'sIoU': float(matrix[:, 1].mean()), 'tIoU': float(matrix[:, 2].mean()),
            'vIoU_legacy': float(matrix[:, 3].mean()), 'delta_vIoU_vs_frozen_pp': float(delta.mean()),
            'delta_ci95_pp': ci, 'source_median_delta_pp': float(np.median(delta)),
            'win_neutral_loss_sources': [int((delta > .1).sum()), int((np.abs(delta) <= .1).sum()), int((delta < -.1).sum())],
            'mean_runtime_sec_caveated': float(np.mean([r['runtime_sec'] for r in records]))}


def analyze(out):
    lock = read(out / 'lock.json')
    cells = {}
    audited = 0
    for name, cell in lock['cells'].items():
        spec = lock['datasets'][cell['target']]
        entries = []
        method_rows = defaultdict(list)
        for i in spec['indices']:
            path = out / name / 'episodes' / f'{i:06d}.json'
            if not path.exists():
                continue
            episode = read(path)
            assert episode['lock_sha256'] == sha(out / 'lock.json')
            assert episode['gt_used'] is False and episode['audits']['reset_exact']
            for record in episode['records']:
                assert record['gt_used'] is False
                assert record['source'] == spec['sources'][str(i)]
                p = record['metric_replay']
                metrics = compute_stvg_metrics(torch.tensor(p['pred_boxes']),
                    [{'boxes': torch.tensor(b).reshape(-1, 4)} for b in p['target_boxes']],
                    interval_from_logits(torch.tensor(p['pred_sted'], dtype=torch.float64)), tuple(p['gt_indices']),
                    frame_ids=p['frame_ids'], gt_frame_interval=p['gt_frame_interval'])
                for key in ['vIoU_corrected', 'vIoU_legacy', 'sIoU', 'tIoU']:
                    assert abs(metrics[key] - record[key]) < 1e-7, (name, i, key)
                method_rows[record['method']].append(record)
                audited += 1
            entries.append(episode)
        if not entries:
            cells[name] = {'status': 'not_started', 'planned_queries': len(spec['indices'])}
            continue
        grid = [r['sample_index'] for r in method_rows['frozen']]
        for rows in method_rows.values():
            assert [r['sample_index'] for r in rows] == grid
        complete = len(entries) == len(spec['indices'])
        cells[name] = {'status': 'complete' if complete else 'partial', 'planned_queries': len(spec['indices']),
            'completed_queries': len(entries), 'methods': {m: summarize(rs, method_rows['frozen']) for m, rs in method_rows.items()},
            'adaptation_coverage': {method: sum(e['audits'][method]['optimizer_steps'] > 0 for e in entries)
                                    for method in ['tent', 'memo', 'sar']},
            'nonzero_parameter_changes': {method: sum(e['audits'][method]['adapted_parameter_delta_l2'] > 0 for e in entries)
                                          for method in ['tent', 'memo', 'sar']},
            'input_frames': [min(e['sampled_input_frames'] for e in entries), max(e['sampled_input_frames'] for e in entries)],
            'peak_vram_gb': max(r['peak_vram_gb'] for r in method_rows['frozen']),
            'paired_fusion_deltas_pp': {m: summarize(method_rows[m + '_fusion'], method_rows['frozen_fusion'])['delta_vIoU_vs_frozen_pp']
                for m in ['tent', 'memo', 'sar']},
            'ours_vs_frozen_fusion': summarize(method_rows['ours'], method_rows['frozen_fusion']),
            'ours_vs_direct_fullspan_fusion': summarize(method_rows['ours'], method_rows['direct_fullspan_fusion']),
            'episode_sha256': {str(e['index']): sha(out / name / 'episodes' / f'{e["index"]:06d}.json') for e in entries}}
    result = {'experiment': str(out), 'lock_sha256': sha(out / 'lock.json'), 'smoke': lock['smoke'],
              'completed_cells': sum(c['status'] == 'complete' for c in cells.values()),
              'total_cells': len(cells), 'metric_records_recomputed': audited, 'cells': cells,
              'interpretation': 'Fixed configurations; adapted classification baselines, not optimized official STVG methods. Prior dataset exposure not certified absent; no confirmation claim.',
              'uncertainty': '10000 source-level paired bootstrap; source-macro percentages; selection seed fixed; one augmentation seed; confidence intervals exploratory, no multiple-comparison correction',
              'pending': lock['settings']['pending']}
    recoveries = {}
    for target in ['hc', 'vid']:
        clean_name = f'{target}_to_{target}_clean'
        if clean_name not in cells:
            continue
        clean = cells[clean_name]
        if clean['status'] != 'complete':
            continue
        for condition in ['blur3', 'low_light3', 'subsample2']:
            name = f'{target}_to_{target}_{condition}'
            if name not in cells:
                continue
            row = cells[name]
            if row['status'] != 'complete':
                continue
            frozen_loss = clean['methods']['frozen']['vIoU'] - row['methods']['frozen']['vIoU']
            fusion_loss = clean['methods']['frozen_fusion']['vIoU'] - row['methods']['frozen_fusion']['vIoU']
            recoveries[name] = {
                'frozen_degradation_pp': frozen_loss,
                'fusion_frozen_degradation_pp': fusion_loss,
                'method_recovery_ratio': {m: ((x['vIoU'] - row['methods']['frozen']['vIoU']) / frozen_loss
                                               if frozen_loss > .1 else None)
                                          for m, x in row['methods'].items()},
                'ours_parameter_only_recovery_given_fusion': (
                    (row['methods']['ours']['vIoU'] - row['methods']['frozen_fusion']['vIoU']) / fusion_loss
                    if fusion_loss > .1 else None),
                'null_reason': 'ratio undefined/uninformative when clean-corrupt gap is <= 0.1 pp'}
    result['controlled_shift_recovery'] = recoveries
    result['analyzer_sha256'] = sha(Path(__file__))
    write(out / 'analysis.json', result)
    print(json.dumps({k: result[k] for k in ['completed_cells', 'total_cells', 'metric_records_recomputed']}))
    for name, row in cells.items():
        if 'methods' in row:
            print(name, row['status'], {m: round(s['vIoU'], 3) for m, s in row['methods'].items()})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, default=ROOT / 'artifacts/baseline_expansion_v1')
    analyze(p.parse_args().out.resolve())
