"""Standalone independent all32 anonymous timing arithmetic; no GT or model."""
import hashlib
import json
import math
import sys
from pathlib import Path
REL = 'results/stvg_opd_supplement/2026-10-13/cost_complete'
KEYS = ['decode_preprocess_seconds', 'native_student_prefix_seconds', 'fresh_expert_seconds',
        'adaptation_seconds', 'final_readout_seconds', 'total_seconds']


def read(path):
    return json.loads(path.read_text())


def quantile(values, probability):
    x = sorted(float(v) for v in values)
    at = (len(x)-1)*probability
    lo, hi = math.floor(at), math.ceil(at)
    return x[lo]+(x[hi]-x[lo])*(at-lo)


def run(root):
    out = Path(root)/REL
    summary = read(out/'END_TO_END_SUMMARY.json')
    count = checks = 0
    for ds in ['hc2', 'vidstg']:
        z = read(out/ds/'QUERY_TIMINGS.json')
        rr = z['rows']
        assert len(rr) == 16 and [r['anonymous_profile_id'] for r in rr] == list(range(16))
        saved = summary['datasets'][ds]
        for r in rr:
            assert 'query_ordinal' not in r and not r.get('GT_read', False)
            assert all(math.isfinite(r[k]) and r[k] >= 0 for k in KEYS)
            assert abs(math.fsum(r[k] for k in KEYS[:-1])+r['dispatcher_seconds']-r['total_seconds']) < 1e-7
            checks += 1
        for key in KEYS:
            for name, probability in [('median', .5), ('p95', .95)]:
                assert abs(quantile([r[key] for r in rr], probability)-saved['timings'][key][name]) < 2e-11
                checks += 1
        assert saved['queries'] == 16
        assert saved['CUDA_peak_allocated_max'] == max(r['CUDA_peak_allocated'] for r in rr)
        assert saved['CUDA_peak_reserved_max'] == max(r['CUDA_peak_reserved'] for r in rr)
        assert saved['actual_new_DINO_calls'] == sum(r['actual_new_DINO_calls'] for r in rr)
        counts = read(out/'ALL_PATH_COUNTS.json')['datasets'][ds]
        assert counts['zero_DINO_paths'] == sum(r['actual_new_DINO_calls'] == 0 for r in rr)
        assert counts['zero_admitted_paths'] == sum(r['admitted_references'] == 0 for r in rr)
        assert counts['zero_backward_paths'] == sum(r['backward_steps'] == 0 for r in rr)
        count += len(rr)
    assert count == summary['query_count'] == 32
    assert summary['final_readout_counted_once'] and not summary['GT_read']
    return dict(status='pass', scope='all32 anonymous actual timed rows/five synchronized stage sum/independent literal linear median andp95/all memory-call/no-reference/no-update counts',
        complete_timing_rows=32, scalar_comparisons=checks, CPU_only=True, GT_read=False,
        raw_GT_read=False, new_model_calls=0,
        summary_sha256=hashlib.sha256((out/'END_TO_END_SUMMARY.json').read_bytes()).hexdigest())


if __name__ == '__main__':
    print(json.dumps(run(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1])))
