"""Root arithmetic audit of each domain, condition and paired-parent contrast."""
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.desta3d_v2_p0 import read, sha
from scripts.score_desta3d_v2_aux_recovery import save_once


def main():
    out = ROOT / 'artifacts/desta3d_v2/tta_v2/target8_B1_output_anchor_v3'
    score = out / 'independent_readback_v1'
    report = read(score / 'REPORT.json')
    assert read(score / 'COMPLETE.json')['report_sha'] == sha(score / 'REPORT.json')
    records = read(score / 'QUERY_METRICS.json')
    lookup = {(r['key'], r['condition']): r for r in records}
    assert len(records) == len(lookup) == 24
    max_error = 0.; comparisons = 0
    for domain, conditions in report['groups'].items():
        keys = sorted({r['key'] for r in records if domain == 'all' or r['domain'] == domain})
        for condition, summary in conditions.items():
            cc = ['noise_medium', 'defocus_extreme'] if condition == 'corruption' else [condition]
            assert summary['parents'] == len(keys)
            samples = [lookup[(k, c)] for k in keys for c in cc]
            values = {}
            for arm, measures in summary['absolute'].items():
                values[arm] = {}
                for metric in measures:
                    values[arm][metric] = {k: np.mean([lookup[(k, c)]['arms'][arm][metric] for c in cc])
                        for k in keys if all(lookup[(k, c)]['arms'][arm][metric] is not None for c in cc)}
                assert summary['format_ok'][arm] == sum(r['arms'][arm]['format_ok'] for r in samples)
                assert summary['native_good_retained'][arm] == sum(r['native_good_retained'][arm] for r in samples)
            assert summary['native_good_frames'] == sum(r['native_good_frames'] for r in samples)
            for contrast, measures in summary['contrasts'].items():
                a, b = contrast.split('-')
                for metric, row in measures.items():
                    parent_keys = sorted(set(values[a][metric]) & set(values[b][metric]))
                    x = np.array([values[a][metric][k]-values[b][metric][k] for k in parent_keys])
                    if not len(x): assert row is None; continue
                    rng = np.random.default_rng(20260927)
                    idx = rng.integers(0, len(x), size=(10000, len(x)))
                    ci = np.percentile(x[idx].sum(1)/len(x), [2.5, 97.5])
                    errors = [abs(float(x.sum()/len(x))-row['mean']), *np.abs(ci-row['ci95'])]
                    max_error = max(max_error, *errors)
                    assert row['n'] == len(x)
                    assert row['positive'] == int((x > 1e-12).sum())
                    assert row['negative'] == int((x < -1e-12).sum())
                    assert row['severe_loss_gt5pp'] == int((x < -.05).sum())
                    comparisons += 1
    assert max_error < 1e-12
    result = {'status': 'passed', 'time': time.time(), 'report_sha': sha(score / 'REPORT.json'),
              'contrast_metric_checks': comparisons, 'max_mean_CI_abs_error': max_error,
              'all_domain_condition_support_retention_tails_checked': True,
              'geometry_scope': 'scorer separately compared 504 scalar/P3 metrics from 168 raw predictions',
              'GT_reopened_by_this_summary_audit': False, 'GPU_started': False}
    save_once(out / 'ROOT_SUMMARY_CROSSCHECK.json', result)
    print(result)


if __name__ == '__main__': main()
