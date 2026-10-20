"""Query-average paired reporting with parent-cluster uncertainty.

Legacy evaluation.py and saved results intentionally keep their old estimand.
No inference code is imported here.
"""
import json
from pathlib import Path
import numpy as np
from .evaluation import read_rows, sha
from .metrics import DenseTube, box_iou
from .metrics_base import xyxy, DenseMetric
from .metrics_hc import HC2DenseMetric


ARMS = {
    'Source': ('Source', 'Native'),
    'Spatial': ('Live', 'Native'),
    'Temporal': ('Source', 'Strong'),
    'WithoutOPD': ('E_Ref', 'Strong'),
    'Full': ('Live', 'Strong'),
}


def aggregate(rows, bootstrap=10000, seed=20261020):
    """Resample whole parents; query means use cluster sums / cluster counts."""
    if not rows:
        raise ValueError('Cannot score an empty evaluation')
    parents = sorted({r['parent'] for r in rows})
    groups = [[r for r in rows if r['parent'] == p] for p in parents]
    arms = list(rows[0]['values'])
    fields = list(rows[0]['values'][arms[0]])
    means, parent_means, matrices = {}, {}, {}
    for arm in arms:
        means[arm], parent_means[arm], matrices[arm] = {}, {}, {}
        for field in fields:
            values = [[r['values'][arm][field] for r in g if r['values'][arm][field] is not None] for g in groups]
            sums = np.array([sum(v) for v in values], dtype=float)
            counts = np.array([len(v) for v in values])
            matrix = np.divide(sums, counts, out=np.full(len(groups), np.nan), where=counts > 0)
            means[arm][field] = float(sums.sum()/counts.sum()) if counts.sum() else None
            parent_means[arm][field] = float(np.nanmean(matrix)) if counts.sum() else None
            matrices[arm][field] = (sums, counts, matrix)
    rng = np.random.default_rng(seed)
    # Batched draws avoid a large bootstrap x parents x queries allocation.
    paired = {}
    comparisons = [('Full', 'Source'), ('Full', 'WithoutOPD'), ('Full', 'Spatial'),
                   ('Full', 'Temporal'), ('WithoutOPD', 'Temporal')]
    for a, b in comparisons:
        if a not in arms or b not in arms:
            continue
        aa, ca, pa = matrices[a]['m_vIoU']
        bb, cb, pb = matrices[b]['m_vIoU']
        if not np.array_equal(ca, cb):
            raise ValueError('Unpaired query populations')
        draws, parent_draws = [], []
        for begin in range(0, bootstrap, 256):
            indices = rng.integers(0, len(groups), (min(256, bootstrap-begin), len(groups)))
            draws.extend(100*(aa-bb)[indices].sum(1)/ca[indices].sum(1))
            parent_draws.extend(100*(pa-pb)[indices].mean(1))
        delta = np.array([100*(r['values'][a]['m_vIoU']-r['values'][b]['m_vIoU']) for r in rows])
        paired[a+'-'+b] = dict(
            query_mean_pp=float(100*(aa.sum()-bb.sum())/ca.sum()),
            query_parent_cluster_CI95_pp=np.quantile(draws, [.025, .975]).tolist(),
            equal_parent_mean_pp=float(100*np.mean(pa-pb)),
            equal_parent_CI95_pp=np.quantile(parent_draws, [.025, .975]).tolist(),
            positive_queries=int((delta > 1e-10).sum()), negative_queries=int((delta < -1e-10).sum()),
            unchanged_queries=int((abs(delta) <= 1e-10).sum()), below_minus5_queries=int((delta < -5).sum()))
    clusters = [dict(parent=p, queries=len(g), values={a: {k: matrices[a][k][2][i].item()
                if np.isfinite(matrices[a][k][2][i]) else None for k in fields} for a in arms})
                for i, (p, g) in enumerate(zip(parents, groups))]
    denominators = {a: {k: sum(r['values'][a][k] is not None for r in rows) for k in fields} for a in arms}
    return dict(queries=len(rows), parents=len(parents), query_means=means, query_denominators=denominators,
                equal_parent_means=parent_means, paired=paired, clusters=clusters,
                bootstrap=dict(unit='parent', draws=bootstrap, seed=seed,
                               query_estimand='sum sampled cluster query totals / sum sampled cluster counts'))


def score(manifest, labels, predictions, output, arms=('Source', 'Full')):
    records = read_rows(manifest)
    root = Path(predictions).resolve()
    seal = json.loads((root/'sealed.json').read_text())
    ids = [r['id'] for r in records]
    if len(set(ids)) != len(ids) or seal['manifest_sha256'] != sha(manifest) or seal['queries'] != len(records) or set(ids) != set(seal['files']):
        raise ValueError('Prediction seal/manifest mismatch')
    for entry in seal['files'].values():
        path = (root/entry['path']).resolve()
        if not path.is_relative_to(root) or sha(path) != entry['sha256']:
            raise ValueError('Prediction hash mismatch')
        if 'receipt_path' in entry:
            receipt = (root/entry['receipt_path']).resolve()
            if not receipt.is_relative_to(root) or sha(receipt) != entry['receipt_sha256']:
                raise ValueError('Prediction cost/identity receipt hash mismatch')
    # Open labels only after validating the whole prediction population.
    gt_rows = read_rows(labels)
    if len({r['id'] for r in gt_rows}) != len(gt_rows) or set(ids) != {r['id'] for r in gt_rows}:
        raise ValueError('Label population differs from manifest')
    truth = {r['id']: r for r in gt_rows}
    # Compile the immutable upstream metric functions once per dataset. Each
    # call still creates its own evaluator state/GT maps; arithmetic is identical.
    evaluators = {ds: HC2DenseMetric() if ds == 'hc2' else DenseMetric()
                  for ds in {r['dataset'] for r in records}}
    rows = []
    for record in records:
        key, ds, meta = record['id'], record['dataset'], record['metadata']
        g = truth[key]
        if g['dataset'] != ds or not meta.get('fps', 0) > 0:
            raise ValueError('Dataset mismatch or missing physical FPS')
        gt = {int(f): b for f, b in g['boxes'].items()}
        span = g['interval']
        values = {}
        with np.load(root/seal['files'][key]['path'], allow_pickle=False) as z:
            fids = z['frame_ids'].tolist()
            if fids != meta['frame_ids']:
                raise ValueError('Predicted physical frame grid changed')
            positions = z['positions8'].tolist()
            if len(set(positions)) != 8 or not all(0 <= p < len(fids) for p in positions):
                raise ValueError('Invalid observation grid')
            observed = {fids[p] for p in positions}
            row = dict(input=meta, frame_ids=fids)
            for arm in arms:
                box_key, interval_key = ARMS[arm]
                boxes = z[box_key]
                interval = z[interval_key+'_interval'].tolist()
                if boxes.shape != (len(fids), 4) or not np.isfinite(boxes).all() or not (boxes[:, 2:] > 0).all():
                    raise ValueError('Invalid predicted tube')
                pix = xyxy(boxes, meta['width'], meta['height'])
                if ds == 'hc2':
                    pix = np.maximum(pix, 0)
                upstream = evaluators[ds](pix, fids, interval, gt, span)
                v = dict(v=upstream['m_vIoU'], t=upstream['m_tIoU'], s=upstream['sIoU_dense_GT'])
                dense = DenseTube(boxes, row, gt, span, clip=ds == 'hc2').score(interval)
                if max(abs(v[k]-dense[k]) for k in ('v', 't', 's')) >= 2e-10:
                    raise ValueError('Official/dense metric cross-check failed')
                nodes = {f: float(box_iou(pix[i], gt[f])) for i, f in enumerate(fids) if f in gt}
                obs = [value for f, value in nodes.items() if f in observed]
                unobs = [value for f, value in nodes.items() if f not in observed]
                values[arm] = dict(m_vIoU=v['v'], **{'vIoU@0.3': float(upstream['vIoU@0.3']), 'vIoU@0.5': float(upstream['vIoU@0.5'])},
                    m_tIoU=v['t'], m_sIoU=v['s'], tIoU_ge_0_5=float(v['t'] >= .5), tIoU_zero=float(v['t'] == 0),
                    boundary_MAE_seconds=(abs(interval[0]-span[0])+abs(interval[1]-span[1]))/(2*meta['fps']),
                    observed_actual_sIoU=float(np.mean(obs)) if obs else None,
                    unobserved_actual_sIoU=float(np.mean(unobs)) if unobs else None,
                    observed_GT_nodes=len(obs), unobserved_GT_nodes=len(unobs))
        rows.append(dict(id=key, dataset=ds, parent=g['parent'], values=values))
    datasets = {ds: aggregate([r for r in rows if r['dataset'] == ds]) for ds in sorted({r['dataset'] for r in rows})}
    result = dict(status='scored_sealed_predictions', version=2, main_estimand='equal queries',
                  legacy_results_modified=False, manifest_sha256=sha(manifest), labels_sha256=sha(labels),
                  undefined_spatial_subset_policy='null when no GT nodes; denominator reported',
                  boundary_unit='seconds using predeclared physical FPS', datasets=datasets, rows=rows)
    dest = Path(output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result
