"""Independent checks on exact coverage increments, geometry, and all paired task counts."""
import csv
import gzip
from collections import Counter, defaultdict
from statistics import mean

import numpy as np

from collect import OUT, digest, read, save
from endpoint_score import verify_inputs
from joint_coverage import ARMS, assignment, bearings, estimate_shape, nominate_joint, nominate_single
from predict_all import DECISIONS, SEALED
from roles_joint import direction_tensor


def soft_union(k, ids):
    uncovered = np.ones(k.shape[1])
    for j in ids:
        uncovered *= 1 - k[j]
    return 1 - uncovered


def verify_math():
    rng = np.random.default_rng(5753)
    max_error = 0.
    for i in range(60):
        ns, nr = rng.integers(5, 11, size=2)
        def kernel(n):
            x = rng.uniform(0.03, 0.93, (n,n))
            x = (x + x.T)/2
            np.fill_diagonal(x, 1)
            return x
        ks, kr = kernel(ns), kernel(nr)
        p = rng.random((ns,nr)); p /= p.sum()
        s, r = nominate_joint(ks, kr, p)
        for step in range(4):
            prev_s, prev_r = list(s[:step]), list(r[:step])
            left = [soft_union(ks, prev_s + [j]) for j in range(ns)]
            right = [soft_union(kr, prev_r + [j]) for j in range(nr)]
            best = max(a @ p @ b for j, a in enumerate(left) if j not in prev_s
                       for k, b in enumerate(right) if k not in prev_r)
            got = left[s[step]] @ p @ right[r[step]]
            np.testing.assert_allclose(got, best, rtol=0, atol=1.e-13)
            max_error = max(max_error, abs(got-best))
        su, ru = soft_union(ks, s), soft_union(kr, r)
        mass = assignment(ks[s]) @ p @ assignment(kr[r]).T
        np.testing.assert_allclose(mass.sum(), su @ p @ ru)
        a, b = p.sum(1), p.sum(0)
        ss, rr = nominate_joint(ks, kr, np.outer(a,b))
        np.testing.assert_array_equal(ss, nominate_single(ks,a))
        np.testing.assert_array_equal(rr, nominate_single(kr,b))
        # Count two independently drawn returns directly, including ties and missing support.
        h, g = rng.integers(0,6,(3,7)), rng.integers(0,5,(4,7))
        h[1] = 0
        actual = np.empty((3,4))
        for j in range(3):
            l = np.repeat(np.arange(7),h[j])
            for k in range(4):
                v = np.repeat(np.arange(7),g[k])
                actual[j,k] = ((l[:,None]<v).mean() + 0.5*(l[:,None]==v).mean()) if len(l) and len(v) else .5
        np.testing.assert_allclose(bearings(h,g,1),actual,atol=1.e-15)
        np.testing.assert_allclose(bearings(h,g,-1),1-actual,atol=1.e-15)
        boxes = rng.normal(size=(9,7))
        boxes[:,3:6] = np.exp(boxes[:,3:6])
        w = rng.random(9); w /= w.sum()
        output = estimate_shape(boxes,w,boxes[0])
        def geom(box):
            return np.column_stack((box[:,:3], np.log(box[:,5]), direction_tensor(box)))
        np.testing.assert_allclose(geom(output[None])[0],w @ geom(boxes),atol=2.e-10)
        other = boxes.copy()
        other[:,[3,4]] = boxes[:,[4,3]]
        other[:,6] += np.pi/2
        np.testing.assert_allclose(geom(estimate_shape(other,w,other[0])[None]),geom(output[None]),atol=2.e-10)
    return max_error


def verify_rows():
    with gzip.open(OUT/'all_rows.csv.gz','rt') as f:
        groups = defaultdict(dict)
        for row in csv.DictReader(f):
            key = (row['group'], row['source'], row['target'], row['case'])
            assert row['arm'] not in groups[key]
            groups[key][row['arm']] = row
    assert len(groups) == 501 and all(set(d) == set(ARMS) for d in groups.values())
    blocks = defaultdict(list)
    for (g,s,t,_), arms in groups.items():
        for group in ([g, 'natural_all'] if g.startswith('natural') else [g]):
            blocks[group, s+'_to_'+t].append(arms)
            blocks[group, 'macro'].append(arms)
    def indicators(r, t):
        return float(r['iou']) > t
    n = 0
    with (OUT/'paired.csv').open() as f:
        for row in csv.DictReader(f):
            data = blocks[row['group'], row['direction']]
            arm, control = row['arm'], row['control']
            assert len(data) == int(row['n'])
            for level in (25,50):
                t = level/100
                transitions = []
                for a in data:
                    left,right = a[arm],a[control]
                    transitions.append((left['source'],left['target'],int(indicators(left,t))-int(indicators(right,t))))
                assert sum(x[2] == 1 for x in transitions) == int(row['rescue'+str(level)])
                assert sum(x[2] == -1 for x in transitions) == int(row['harm'+str(level)])
                by_direction = defaultdict(list)
                for s,r,m in transitions:
                    by_direction[s,r].append(m)
                np.testing.assert_allclose(100*mean(mean(v) for v in by_direction.values()),
                                           float(row['delta'+str(level)]), atol=1.e-10)
            n += 1
    applied, fallbacks = 0, 0
    seal = read(SEALED)
    for name in seal['files']:
        with np.load(DECISIONS/name,allow_pickle=False) as dec:
            box, raw, ref = dec['boxes'], dec['raw'], dec['refs']
            np.testing.assert_array_equal(raw[-3:],np.repeat(raw[-1:],3,axis=0))
            np.testing.assert_array_equal(ref[-3:],np.repeat(ref[-1:],3,axis=0))
            np.testing.assert_array_equal(box[-3],raw[-1])
            if dec['applied']:
                applied += 1
                assert dec['subjects'].shape == (len(ARMS),4,7)
                assert dec['references'].shape == (len(ARMS),4,7)
                for k, arm in enumerate(ARMS[1:],1):
                    assert np.isfinite(dec['coverage'][k]).all()
                    if arm!='whole_pool_map':
                        assert np.isfinite(dec['subjects'][k]).all() and np.isfinite(dec['references'][k]).all()
            else:
                fallbacks += 1
                np.testing.assert_array_equal(box,np.repeat(box[:1],len(box),axis=0))
    assert applied == seal['applied'] == 172 and fallbacks == 329
    return n, applied, fallbacks


def verify_ordinary():
    expected = defaultdict(Counter)
    seen = set()
    with gzip.open(OUT/'ordinary_component_rows.csv.gz','rt') as f:
        for row in csv.DictReader(f):
            d, case, t = row['direction'], row['case'], row['cut']
            assert (d,case,t) not in seen
            seen.add((d,case,t))
            for key in ((d,t),('all_records',t)):
                counts = expected[key]
                counts[row['state']] += 1
                counts['n'] += 1
                counts['hard_bound'] += int(row['hard_bound'])
                counts['single_stream_can_win'] += int(row['one_stream_can_win'] or 0)
                if row['state']=='linearly_separable':
                    assert float(row['best_normalized_margin']) > 1.e-9
                    counts['bound_and_separable'] += int(row['hard_bound'])
                elif row['best_normalized_margin']:
                    value = float(row['best_normalized_margin'])
                    assert value <= 1.e-9
                    counts['positive_below_tolerance'] += int(value > 0)
    assert len(seen)==2*18620
    with (OUT/'ordinary_component_upper_bounds.csv').open() as f:
        for row in csv.DictReader(f):
            key = row['direction'], row['cut']
            for k,v in row.items():
                if k not in ('direction','cut'):
                    assert int(v)==expected[key][k]
    return len(seen)


if __name__ == '__main__':
    verify_inputs()
    max_error = verify_math()
    n, applied, fallbacks = verify_rows()
    receipt = dict(records=501, arms=len(ARMS), posthoc_scoring=True, applied=applied, fallbacks=fallbacks,
                   paired_aggregate_rows_independently_checked=n, random_coverage_geometry_problems=60,
                   exhaustive_greedy_decisions=240, max_greedy_objective_error=max_error,
                   ordinary_direction_threshold_records=verify_ordinary(),
                   private_prediction_seal_sha256=digest(SEALED), FINAL_access=False)
    receipt['public_results'] = {p.name:digest(p) for p in sorted(OUT.iterdir())
                                 if p.is_file() and p.name != 'audit.json'}
    save(OUT/'audit.json',receipt)
