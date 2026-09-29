"""Independent parent aggregation of one sealed DESTA phase; no new GT or GPU."""
import argparse
from collections import defaultdict
from functools import cmp_to_key
import hashlib
import json
from pathlib import Path
import statistics
import time


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(directory, phase):
    import numpy as np
    start = time.monotonic()
    output = directory / 'phase_readback' / phase / 'ROOT_SUMMARY.json'
    if output.exists():
        raise FileExistsError('Preserve previous readback; do not overwrite.')
    completed = directory / 'scores' / phase / 'COMPLETE.json'
    for path, value in read(completed)['pins'].items():
        assert digest(Path(path)) == value
    source = directory / 'scores' / phase / 'REPORT.json'
    report = read(source)
    cfg = read(directory / 'CONFIG.json')
    metrics = ('tIoU', 'sIoU', 'vIoU')
    parents, summary, comparisons, tails, retained = {}, {}, {}, {}, {}
    error = 0.

    def close(actual, expected):
        nonlocal error
        error = max(error, abs(float(actual)-float(expected)))
        assert abs(float(actual)-float(expected)) < 1e-8, (actual, expected)

    for arm, rows in report['rows'].items():
        grouped = defaultdict(list)
        assert len({row['key'] for row in rows}) == len(rows)
        for row in rows:
            grouped[row['source']].append(row['metrics'])
        parents[arm] = {m: {p: statistics.fmean(r[m] for r in rs)
                           for p, rs in grouped.items()} for m in metrics}
        summary[arm] = {m: 100*statistics.fmean(parents[arm][m].values()) for m in metrics}
        for metric in metrics:
            close(summary[arm][metric], report['summary_percent'][arm][metric])
    for arm, rows in report['rows'].items():
        if arm == 'B1':
            continue
        comparisons[arm], tails[arm], retained[arm] = {}, {}, {}
        for metric in metrics:
            assert parents[arm][metric].keys() == parents['B1'][metric].keys()
            values = np.array([100*(parents[arm][metric][p]-parents['B1'][metric][p])
                               for p in sorted(parents['B1'][metric])])
            indices = np.random.default_rng(20260927).integers(0, len(values), (10000, len(values)))
            ci = np.quantile(values[indices].mean(1), [.025, .975])
            expected = report['comparisons'][arm][metric]
            close(values.mean(), expected['mean_delta_pp'])
            for actual, target in zip(ci, expected['bootstrap_ci95_pp']):
                close(actual, target)
            comparisons[arm][metric] = dict(delta_pp=float(values.mean()), ci95_pp=ci.tolist())
            base = {r['key']: r['metrics'][metric] for r in report['rows']['B1']}
            qd = [(r['metrics'][metric]-base[r['key']])*100 for r in rows]
            tails[arm][metric] = dict(harm_gt5pp=sum(d < -5 for d in qd),
                positive=sum(d > 0 for d in qd), negative=sum(d < 0 for d in qd),
                zero=sum(d == 0 for d in qd), worst_pp=min(qd), best_pp=max(qd))
            retained[arm][metric] = dict(eligible=sum(v > .5 for v in base.values()),
                retained=sum(base[r['key']] > .5 and r['metrics'][metric] > .5 for r in rows))
            assert tails[arm][metric] == report['query_tails'][arm][metric]
            assert retained[arm][metric] == report['B1_good_retention'][arm][metric]
    keys = list(report['ranked'])

    def compare(a, b):
        delta = summary[a]['vIoU']-summary[b]['vIoU']
        if abs(delta) > cfg['tie_tolerance_pp']:
            return -1 if delta > 0 else 1
        va, vb = tails[a]['vIoU']['harm_gt5pp'], tails[b]['vIoU']['harm_gt5pp']
        return (-1 if va < vb else 1) if va != vb else (-1 if a < b else int(a > b))

    ranked = sorted(keys, key=cmp_to_key(compare))
    assert ranked == report['ranked']
    assert ranked[:6 if phase == 'dev16' else 1] == report['selected']
    factors = {}
    if phase == 'dev16':
        for factor in ('steps', 'radius', 'temporal_weight'):
            levels = sorted({v[factor] for v in cfg['grid'].values()})
            marginals = {str(level): statistics.fmean(summary[k]['vIoU'] for k in keys
                            if cfg['grid'][k][factor] == level) for level in levels}
            expected = report['factorial_dev16'][factor]
            for level, value in marginals.items():
                close(value, expected['marginal_parent_macro_percent'][level]['vIoU'])
            deltas = []
            for parent in sorted(parents['B1']['vIoU']):
                lo = statistics.fmean(parents[k]['vIoU'][parent] for k in keys if cfg['grid'][k][factor] == levels[0])
                hi = statistics.fmean(parents[k]['vIoU'][parent] for k in keys if cfg['grid'][k][factor] == levels[-1])
                deltas.append(100*(hi-lo))
            values = np.asarray(deltas)
            idx = np.random.default_rng(20260927).integers(0, len(values), (10000, len(values)))
            factors[factor] = dict(marginal_v_percent=marginals,
                span_pp=max(marginals.values())-min(marginals.values()),
                high_minus_low_pp=float(values.mean()),
                high_minus_low_ci95_pp=np.quantile(values[idx].mean(1), [.025, .975]).tolist())
            close(factors[factor]['span_pp'], expected['marginal_span_pp']['vIoU'])
            close(values.mean(), expected['highest_minus_lowest_v_pp'])
    result = dict(status='passed', phase=phase, queries=len(report['rows']['B1']),
        parents=len(parents['B1']['vIoU']), configurations=len(keys),
        above_B1_v=sum(summary[k]['vIoU'] > summary['B1']['vIoU'] for k in keys),
        summary_percent=summary, comparisons=comparisons, tails=tails, B1_good_retention=retained,
        ranked=ranked, selected=report['selected'], factorial=factors,
        max_abs_summary_error=error, CPU_seconds=time.monotonic()-start, new_GPU_seconds=0,
        new_GT_read=False, source_report_sha256=digest(source),
        scope='Exposed development selection; parent bootstrap CIs are descriptive and unadjusted. Not independent fresh evaluation.')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps({k: result[k] for k in ('status', 'phase', 'above_B1_v', 'max_abs_summary_error', 'factorial')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--phase', choices=['dev16', 'dev64', 'ablations'], required=True)
    args = parser.parse_args()
    check(args.directory.resolve(), args.phase)
