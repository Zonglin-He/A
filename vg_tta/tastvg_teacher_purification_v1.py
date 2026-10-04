"""CPU-only, prelocked raw-proposal medoid. No model or adaptation imports."""
import math
import os
import numpy as np


def validate(proposals, confidence):
    p = np.asarray(proposals, dtype=np.float64)
    c = np.asarray(confidence, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 2 or not len(p) or c.shape != (len(p),):
        raise ValueError('Nonempty aligned raw proposal support required')
    if not np.isfinite(p).all() or not np.isfinite(c).all() or not (p[:, 1] > p[:, 0]).all():
        raise ValueError('Invalid interval or confidence')
    return p, c


def overlap_matrix(proposals):
    p = np.asarray(proposals, dtype=np.float64)
    overlap = np.maximum(0., np.minimum(p[:, None, 1], p[None, :, 1]) - np.maximum(p[:, None, 0], p[None, :, 0]))
    lengths = p[:, 1] - p[:, 0]
    return overlap / (lengths[:, None] + lengths[None, :] - overlap)


def consensus(proposals):
    """Scores depend only on proposals, preserving every raw row and duplicate."""
    p, _ = validate(proposals, np.zeros(len(proposals)))
    matrix = overlap_matrix(p)
    np.fill_diagonal(matrix, 0.)
    # fsum avoids permutation-dependent reduction errors at exact score ties.
    scores = np.array([math.fsum(row) / (len(p) - 1) for row in matrix]) if len(p) > 1 else np.array([0.])
    return int(np.argmax(scores)), scores


def select(proposals, confidence):
    p, c = validate(proposals, confidence)
    medoid, scores = consensus(p)
    return dict(Confidence_index=int(np.argmax(c)), Consensus_index=medoid,
                consensus_scores=scores.tolist(), confidence_scores=c.tolist(), proposal_count=len(p),
                unique_intervals=len(set(map(tuple, p.tolist()))),
                singleton_agreement_undefined=len(p) == 1, GT_used=False)


def truth_iou(proposals, span):
    p, _ = validate(proposals, np.zeros(len(proposals)))
    s, e = map(float, span)
    if not np.isfinite([s, e]).all() or e <= s:
        raise ValueError('Invalid evaluation span')
    overlap = np.maximum(0., np.minimum(p[:, 1], e) - np.maximum(p[:, 0], s))
    return overlap / (p[:, 1] - p[:, 0] + e - s - overlap)


def read_guard(event, args):
    if event != 'open' or not args or not isinstance(args[0], (str, bytes)):
        return
    path = os.fsdecode(args[0]); mode = args[1] if len(args) > 1 else None
    reading = mode is None or (isinstance(mode, str) and ('r' in mode or '+' in mode)) or (isinstance(mode, int) and mode & os.O_ACCMODE != os.O_WRONLY)
    denied = ['GT_LABELS', 'SOURCE_GT', 'GT_EXPOSURE', 'GT_SUBSET', 'test_annotations', 'valv2_proc',
              'vidstd-test-anno', '/ORACLE_SELECTION', '/oracle_runs/', 'TEACHER_SELECTION_SCORED',
              '/ROWS.json', '/SUMMARY.json', '/CASES.json', '/target_features/', '/HEAD.pt',
              '/checkpoints/', '.safetensors', '/SOURCE_VALIDATION_ROWS', '/SOURCE_SELECTION.json']
    if reading and any(x in path for x in denied):
        raise PermissionError('Unlabeled teacher selection forbids GT/scored results/model inputs: ' + path)


ARMS = ['Confidence', 'Consensus', 'Oracle']
FIELDS = [a + '_t' for a in ARMS] + [a + '_success_gt_05' for a in ARMS] + [a + '_disjoint' for a in ARMS] + [
    'Consensus_minus_Confidence_t', 'Oracle_minus_Confidence_t', 'Oracle_minus_Consensus_t',
    'Consensus_minus_Confidence_success', 'Consensus_minus_Confidence_disjoint',
    'Consensus_gross_gain', 'Consensus_gross_loss', 'raw_mean_t', 'raw_fraction_gt_05',
    'selected_Consensus_agreement', 'selected_Confidence_agreement', 'proposal_count', 'unique_intervals']


def add_metrics(row):
    values = row['proposal_GT_tIoU']
    for arm in ARMS:
        t = float(values[row[arm + '_index']])
        row.update({arm + '_t': t, arm + '_success_gt_05': float(t > .5), arm + '_disjoint': float(t == 0.)})
    d = row['Consensus_t'] - row['Confidence_t']
    row.update(Consensus_minus_Confidence_t=d,
               Oracle_minus_Confidence_t=row['Oracle_t']-row['Confidence_t'],
               Oracle_minus_Consensus_t=row['Oracle_t']-row['Consensus_t'],
               Consensus_minus_Confidence_success=row['Consensus_success_gt_05']-row['Confidence_success_gt_05'],
               Consensus_minus_Confidence_disjoint=row['Consensus_disjoint']-row['Confidence_disjoint'],
               Consensus_gross_gain=max(d, 0.), Consensus_gross_loss=max(-d, 0.),
               raw_mean_t=float(np.mean(values)), raw_fraction_gt_05=float(np.mean(np.asarray(values) > .5)),
               selected_Consensus_agreement=row['consensus_scores'][row['Consensus_index']],
               selected_Confidence_agreement=row['consensus_scores'][row['Confidence_index']])
    return row


def paired_summary(rows, fields=FIELDS):
    grouped = {}
    for row in rows:
        grouped.setdefault(row['source_id'], []).append([row[f] for f in fields])
    ids = sorted(grouped)
    matrix = np.array([np.mean(grouped[s], axis=0) for s in ids])
    rng = np.random.default_rng(20261004)
    draws = np.concatenate([matrix[rng.integers(0, len(ids), size=(100, len(ids)))].mean(axis=1) for _ in range(100)])
    low, high = np.percentile(draws, [2.5, 97.5], axis=0)
    means = matrix.mean(axis=0)
    metrics = {f: dict(mean=float(means[i]), ci95=[float(low[i]), float(high[i])],
                      source_values={str(s): float(matrix[j, i]) for j, s in enumerate(ids)},
                      cell_macro=float(np.mean([r[f] for r in rows]))) for i, f in enumerate(fields)}
    d = np.array([metrics['Consensus_minus_Confidence_t']['source_values'][str(s)] for s in ids])
    return dict(cells=len(rows), sources=len(ids), metrics=metrics,
                positive_sources=int(sum(d > 1e-12)), negative_sources=int(sum(d < -1e-12)),
                unchanged_sources=int(sum(abs(d) <= 1e-12)),
                leave_one_source_out_delta={str(s): float(np.delete(d, j).mean()) for j, s in enumerate(ids)},
                counts={
                    'choices_changed': sum(r['Consensus_index'] != r['Confidence_index'] for r in rows),
                    'teacher_improved': sum(r['Consensus_minus_Confidence_t'] > 1e-12 for r in rows),
                    'teacher_worsened': sum(r['Consensus_minus_Confidence_t'] < -1e-12 for r in rows),
                    'confidence_success_destroyed': sum(r['Confidence_t'] > .5 and r['Consensus_t'] <= .5 for r in rows),
                    'confidence_failure_rescued': sum(r['Confidence_t'] <= .5 and r['Consensus_t'] > .5 for r in rows),
                    'new_disjoint_event': sum(r['Confidence_t'] > 0. and r['Consensus_t'] == 0. for r in rows),
                    'disjoint_event_rescued': sum(r['Confidence_t'] == 0. and r['Consensus_t'] > 0. for r in rows),
                    **{a + '_success_gt_05': sum(r[a+'_t'] > .5 for r in rows) for a in ARMS},
                    **{a + '_disjoint': sum(r[a+'_t'] == 0. for r in rows) for a in ARMS}})


def summaries(rows):
    result = {}
    for ds in ['vidstg', 'hc2']:
        result[ds] = {}
        for split in ['search', 'confirm']:
            cells = [r for r in rows if r['dataset'] == ds and r['split'] == split]
            result[ds][split] = {}
            for name, subset in [('corrupt', [r for r in cells if r['condition'] != 'clean']),
                                 ('clean', [r for r in cells if r['condition'] == 'clean']), ('all', cells)]:
                result[ds][split][name] = paired_summary(subset)
                result[ds][split][name]['orders'] = {order: paired_summary([r for r in subset if r['order'] == order]) for order in ['order1', 'order2']}
    return result
