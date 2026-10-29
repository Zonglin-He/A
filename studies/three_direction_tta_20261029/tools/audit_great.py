"""Independent CPU audit of sealed GREAT predictions and declared metric masks."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata


def measure(pred, target):
    p, t = np.asarray(pred).ravel(), np.asarray(target).ravel()
    if p.shape != t.shape or not np.isfinite(p).all() or not np.isfinite(t).all():
        raise ValueError('Invalid prediction or target')
    if p.min() < -1e-7 or p.max() > 1 + 1e-7:
        raise ValueError('Expected probability-valued prediction')
    positive = t >= .5
    npos, nneg = int(positive.sum()), int((~positive).sum())
    # Mann-Whitney rank formulation, independent of sklearn roc_auc_score.
    auc = ((rankdata(p)[positive].sum() - npos * (npos + 1) / 2) / (npos * nneg)
           if npos and nneg else None)
    intersections, unions = [], []
    for threshold in np.linspace(0., 1., 20):
        selected = p >= threshold
        intersections.append(np.count_nonzero(selected & positive))
        unions.append(np.count_nonzero(selected | positive))
    aiou = float(np.mean(np.asarray(intersections) / np.asarray(unions))) if npos else None
    sim = float(np.minimum(p / (p.sum() + 1e-12), t / (t.sum() + 1e-12)).sum())
    return {'auc': None if auc is None else float(auc), 'aiou': aiou,
            'sim': sim, 'mae': float(np.mean(np.abs(p - t)))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run', type=Path)
    ap.add_argument('targets', type=Path)
    ap.add_argument('output', type=Path)
    args = ap.parse_args()
    seal = json.loads((args.run / 'SEALED.json').read_text())
    for file, expected in seal.items():
        assert hashlib.file_digest((args.run / file).open('rb'), 'sha256').hexdigest() == expected
    targets = json.loads(args.targets.read_text())
    first = json.loads((args.run / 'SCORED.json').read_text())
    target_lookup = {r['id']: (i, r) for i, r in enumerate(targets)}
    errors, rows = [], []
    for row in first['rows']:
        i, truth = target_lookup[row['id']]
        file = args.run / f"{i:04d}_{row['objective']}_k{row['step']}.npz"
        with np.load(file, allow_pickle=False) as data:
            metrics = measure(data[row['arm']], np.load(truth['target'], allow_pickle=False))
        for key, value in metrics.items():
            expected = row[key]
            if value is None:
                assert expected is None or not np.isfinite(expected), (row['id'], key)
            else:
                error = abs(value - expected)
                if not np.isfinite(error) or error > 1e-6:
                    raise RuntimeError((row['id'], key, value, expected))
                errors.append(error)
        rows.append({k: row[k] for k in ['id', 'objective', 'step', 'arm']} | metrics)
    result = {'seal_verified': True, 'scoring_agrees': True,
              'maximum_absolute_error': max(errors), 'n_parents': len(targets), 'rows': rows}
    with args.output.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps({k:v for k,v in result.items() if k != 'rows'}, indent=2))


if __name__ == '__main__':
    main()
