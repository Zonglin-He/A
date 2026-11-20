"""Can the three existing score streams distinguish passing boxes even with oracle weights?"""
import csv
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent/'3eed_v1_subject_margin_attribution_20261120'
FILES = PARENT/'run_PRIVATE'
OUT = HERE/'results'
EPS = 1.e-9


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_file(path):
    return json.loads(path.read_bytes())


def max_margin(delta):
    inequalities = np.column_stack((-delta, np.ones(len(delta))))
    result = linprog([0,0,0,-1], A_ub=inequalities, b_ub=np.zeros(len(delta)),
                     A_eq=[[1,1,1,0]], b_eq=[1], bounds=[(0,1)]*3+[(None,None)],
                     options=dict(primal_feasibility_tolerance=1.e-10, dual_feasibility_tolerance=1.e-10))
    assert result.success
    achieved = (delta @ result.x[:3]).min()
    np.testing.assert_allclose(achieved,result.x[3],rtol=0,atol=2.e-10)
    return float(achieved)


def clipped_feasible(delta, threshold):
    # Independent half-plane clipping. (x,y,1-x-y) are nonnegative on the triangle.
    polygon = [np.array(p,float) for p in ((0,0),(1,0),(0,1))]
    for da, du, dr in delta:
        p = np.array([da-dr,du-dr])
        values = [v @ p + dr - threshold for v in polygon]
        keep = []
        for k, endpoint in enumerate(polygon):
            start, first, last = polygon[k-1], values[k-1], values[k]
            if (first >= 0) != (last >= 0):
                keep.append(start + first/(first-last)*(endpoint-start))
            if last >= 0:
                keep.append(endpoint)
        polygon = keep
        if not polygon:
            return False
    return True


def classify(scores, passing):
    good, bad = scores[passing], scores[~passing]
    all_dominated = True
    margins = []
    primary = False
    for g in good:
        diff = g - bad
        dominated = (diff <= 0).all(1).any()
        all_dominated &= dominated
        margin = max_margin(diff)
        assert (margin > EPS) == clipped_feasible(diff, EPS)
        if dominated:
            assert margin < EPS
        primary |= bool((diff.min(0) > EPS).any())
        margins.append(margin)
    best = max(margins)
    category = ('linearly_separable' if best > EPS else
                'all_passing_dominated' if all_dominated else 'undominated_but_not_separable')
    return category, best, int(primary)


def run():
    spec = json_file(HERE/'ordinary_diagnostic.json')
    seal_path = FILES/'SEALED.json'
    assert sha(seal_path) == spec['private_parent_seal']
    parent = json_file(seal_path)
    assert parent['n'] == 18620 and not parent['target_labels_used'] and not parent['FINAL_access']
    with (PARENT/'results/ordinary_attribution.csv').open() as f:
        parent_counts = {(r['direction'], float(r['threshold'])):r for r in csv.DictReader(f)}
    totals, rows = defaultdict(Counter), []
    for direction, signature in parent['directions'].items():
        sealed = FILES/direction/'SEALED.json'
        assert sha(sealed) == signature
        native = json_file(sealed)
        scores = FILES/f'scored_{direction}.npz'
        audit = json_file(scores.with_suffix('.json'))
        assert audit['prediction_seal'] == sha(seal_path) and sha(scores) == audit['file_sha256']
        with np.load(scores,allow_pickle=False) as f:
            ious = f['pool_iou']
        seen = set()
        for part, pin in sorted(native['files'].items()):
            path = sealed.parent/part
            assert sha(path) == pin
            with np.load(path,allow_pickle=False) as data:
                components = np.stack([data[key] for key in ('prior','unary','rel')],axis=-1).astype(float)
                np.testing.assert_allclose(components.sum(-1),data['final'],rtol=0,atol=2.e-6)
                ordinals, slots, winner, active = data['ordinals'], data['slots'], data['choices'][:,0], data['active']
            for i, ix in enumerate(ordinals):
                assert int(ix) not in seen
                seen.add(int(ix))
                case = hashlib.sha256(f'v1-bound-audit-20261120|ordinary|{direction}|{ix}'.encode()).hexdigest()[:16]
                for cut in spec['thresholds']:
                    passing = ious[ix, slots[i]] > cut
                    cap = 2.5 if active[i] else 1.5
                    signs = components[i,:,0]
                    bounded = bool(passing.any() and signs.max()-signs[passing].max() >= cap)
                    margin, one = '', ''
                    if passing[winner[i]]:
                        state = 'v1_hit'
                    elif not passing.any():
                        state = 'no_good_in_four'
                    else:
                        state, margin, one = classify(components[i],passing)
                    rows.append(dict(direction=direction,case=case,cut=cut,reference=int(active[i]),hard_bound=int(bounded),
                                     state=state,best_normalized_margin=margin,one_stream_can_win=one))
                    for key in [(direction,cut),('all_records',cut)]:
                        total = totals[key]
                        total['n'] += 1
                        total[state] += 1
                        total['hard_bound'] += int(bounded)
                        total['bound_and_separable'] += int(bounded and state=='linearly_separable')
                        total['single_stream_can_win'] += int(one or 0)
                        total['positive_below_tolerance'] += int(margin!='' and 0 < margin <= EPS)
        assert seen == set(range(native['n']))
    summary = []
    keys = ('n','v1_hit','no_good_in_four','all_passing_dominated','undominated_but_not_separable',
            'linearly_separable','hard_bound','bound_and_separable','single_stream_can_win','positive_below_tolerance')
    for (direction, cut), x in sorted(totals.items()):
        old = parent_counts[direction, cut]
        assert x['n']==int(old['n']) and x['v1_hit']==int(old['hit']) and x['hard_bound']==int(old['blocked'])
        assert (x['all_passing_dominated'] + x['undominated_but_not_separable'] + x['linearly_separable']
                == int(old['recoverable_four_misses']))
        summary.append(dict(direction=direction,cut=cut,**{name:x[name] for name in keys}))
    with (OUT/'ordinary_component_upper_bounds.csv').open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=summary[0]); writer.writeheader(); writer.writerows(summary)
    with gzip.open(OUT/'ordinary_component_rows.csv.gz','wt',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)


if __name__ == '__main__':
    run()
