"""Independent integer-grid mathematical qualification, no dataset/GT access."""
from pathlib import Path
import hashlib, json, sys, time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_stvg_posterior_decision_tree_v1 import direct, greedy_check
from vg_tta.stvg_posterior_decision_tree_v1 import expected_iou_fast, greedy4
import numpy as np

def main():
    rng = np.random.default_rng(7182026)
    start = time.perf_counter()
    checks = region_checks = cases = 0
    largest_error = coverage_error = 0.
    grids = [[0, 1, 2, 3], [0, 1, 17, 21, 70], [-29, -28, -3, 0, 2, 40],
             [0, 3, 9, 10, 11, 48, 49, 130],
             np.cumsum(rng.integers(1, 60, 13)).tolist()]
    for ids in grids:
        n = len(ids)
        a, b = np.triu_indices(n, 1)
        distributions = []
        for selected in [0, len(a)//2, len(a)-1]:
            p = np.zeros(len(a)); p[selected] = 1.; distributions.append(p)
        distributions += [np.ones(len(a))/len(a), rng.random(len(a)),
                          rng.random(len(a)) * (rng.random(len(a)) > .75)]
        for p in distributions:
            if not p.sum(): p[0] = 1.
            p = p / p.sum()
            q = np.zeros((n, n)); q[a, b] = p
            x, y, answer = direct(ids, q)
            fast = expected_iou_fast(ids, q)[x, y]
            error = float(np.max(np.abs(fast - answer)))
            assert error < 1e-12
            largest_error = max(largest_error, error)
            # Verify the endpoint partition itself, including equal boundaries and
            # physically disjoint intervals on adjacent/irregular integer frames.
            for s, e in zip(a, b):
                intersects = (np.minimum(np.asarray(ids)[e]+1, np.asarray(ids)[b]+1)
                              > np.maximum(np.asarray(ids)[s], np.asarray(ids)[a]))
                regions = ((a >= s) & (b <= e)).astype(int)
                regions += ((a < s) & (b > e)).astype(int)
                regions += ((a < s) & (b <= e) & (np.asarray(ids)[b]+1 > ids[s])).astype(int)
                regions += ((a >= s) & (a <= e) & (b > e)).astype(int)
                assert np.array_equal(regions, intersects.astype(int))
                region_checks += len(a)
            layout, trace = greedy4(ids, q)
            independently_selected, err = greedy_check(ids, q, trace)
            assert layout == independently_selected
            coverage_error = max(coverage_error, err)
            checks += len(answer); cases += 1
    result = dict(status='pass', GT_read=False, dataset_files_read=0, new_models=0,
                  cases=cases, irregular_and_adjacent_integer_grids=len(grids),
                  exhaustive_utility_checks=checks, endpoint_partition_checks=region_checks,
                  deterministic_ties_and_point_mass=True, greedy_direct_objective=True,
                  max_utility_error=largest_error, max_coverage_error=coverage_error,
                  CPU_wall_seconds=time.perf_counter()-start,
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    target = ROOT/'artifacts/stvg_posterior_decision_tree_v1/ROOT_CPU_QUALIFICATION.json'
    assert not target.exists()
    target.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result))

if __name__ == '__main__':
    main()
