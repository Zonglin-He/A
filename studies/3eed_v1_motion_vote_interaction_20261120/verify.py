"""Independently check the factorial statistic, source fit and sealed target outcomes."""
import argparse
from types import SimpleNamespace

from motion import (ARMS, HERE, RESULTS, WORK, atomic, channels, digest, engine,
                    initialize, np, observations, read)
from source_motion import cases, catalog, canonical, observation, native_boxes
from verify_transfer import audit_panel, loaded, read_rows


def overlaps_one(box, others):
    side = np.maximum(0., box[3:6])
    sizes = np.maximum(0., others[:, 3:6])
    intersection = np.maximum(0., np.minimum(box[:3]+side/2, others[:, :3]+sizes/2)
                              - np.maximum(box[:3]-side/2, others[:, :3]-sizes/2)).prod(axis=1)
    union = side.prod() + sizes.prod(axis=1) - intersection
    return np.divide(intersection, union, out=np.zeros_like(union), where=union > 0)


def conditional_weights(anchors, boxes, category_weights):
    boxes = np.asarray(boxes, dtype=float)
    densities = []
    for j, box in enumerate(boxes):
        overlap = overlaps_one(box, boxes)
        overlap[j] = 1.
        densities.append(overlap @ overlap)
    return np.stack([overlaps_one(b, boxes) * category_weights / densities for b in anchors])


def verify_physical():
    model = engine()
    for _, row, pairs in cases('fit'):
        objects, _ = catalog(row)
        view = observations(model, dict(observation(row), text=canonical(objects[pairs[0][0]]['caption'])),
                            parity=True)
        if view.neutral is None:
            continue
        anchors = native_boxes(view.ev).astype(float)
        vals, mask = channels(anchors, view)
        if mask.sum() < 2:
            continue
        f, c = view.full, view.neutral
        pf = np.maximum(f.probability.astype(float), 1e-300)
        pc = np.maximum(c.probability.astype(float), 1e-300)
        ratio = pf/pc
        expected = []
        normalizers = []
        for boxes in (f.boxes, c.boxes):
            w = conditional_weights(anchors, boxes, pc)
            normalizers.append(w.sum(axis=1))
            conditional = w[mask]/w[mask].sum(axis=1, keepdims=True)
            expected.append(np.log(conditional @ np.stack((ratio, ratio[view.permutation]), axis=1)))
        np.testing.assert_array_equal(mask, np.minimum(*normalizers) > 0)
        manual = np.column_stack((expected[0]-expected[1], np.log(normalizers[1][mask])))
        manual -= manual.mean(axis=0)
        physical_error = float(np.max(np.abs(manual-vals[mask])))
        assert physical_error < 1e-9
        errors = []
        for boxes, weights in ((f.boxes, pc), (c.boxes, pf)):
            alternate = SimpleNamespace(boxes=boxes, probability=weights)
            result, supported = channels(anchors, SimpleNamespace(full=f, neutral=alternate,
                                                                  permutation=view.permutation))
            errors.append(float(np.max(np.abs(result[supported, 0]))))
        assert max(errors) < 1e-10
        return dict(full_self_replay=True, compared_anchors=int(mask.sum()),
                    conditional_expectation_error=physical_error, null_effect_errors=errors)
    raise AssertionError("Source roster has no jointly supported physical comparison")


def verify_source():
    model = read(WORK/'MODEL_LOCK.json')
    assert model['source'] == digest(WORK/'fit.npz') and model['motion'] == digest(HERE/'motion.py')
    z = loaded(WORK/'fit.npz')
    data = []
    for base, values, available, (i, j) in zip(z['j'], z['val'], z['ok'], z['slots']):
        if min(i, j) < 0:
            continue
        diff = base.astype(float)[:, i]-base.astype(float)[:, j]
        for t in np.flatnonzero(available[:, i] & available[:, j]):
            data.append((values[t, i]-values[t, j], diff.mean()))
    x, y = map(np.asarray, zip(*data))
    coeff = np.array([np.linalg.lstsq(column[:, None], y, rcond=None)[0][0] for column in x.T])
    np.testing.assert_allclose(coeff, model['unclipped'], atol=1e-12, rtol=0)
    np.testing.assert_allclose(np.clip(coeff, 0, 1), model['coefficients'], atol=1e-12, rtol=0)
    rows = read_rows(RESULTS/'source_all.csv.gz')
    for s in read_rows(RESULTS/'source_summary.csv'):
        arm, split = s['arm'], s['split']
        group = {r['pair']: r for r in rows if r['arm'] == arm and r['split'] == split}
        assert len(group) == int(s['n'])
        bases = {r['pair']: r for r in rows if r['arm'] == 'v1' and r['split'] == split}
        assert group.keys() == bases.keys()
        for metric, column in (('available', 'covered'), ('both', 'both'), ('both_four', 'both_four'),
                               ('measured_directions', 'measurable_directions')):
            assert sum(int(r[metric]) for r in group.values()) == int(s[column])
        actual = np.array([[float(r[k]) for k in ('native_a','native_b')] for r in group.values()])
        frozen = np.array([[float(bases[p][k]) for k in ('native_a','native_b')] for p in group])
        for threshold in (25, 50):
            np.testing.assert_allclose(100*(actual > threshold/100).mean(), float(s[f'native_Acc{threshold}']))
        assert ((actual > .5) & (frozen <= .5)).sum() == int(s['native_rescue50'])
        assert ((actual <= .5) & (frozen > .5)).sum() == int(s['native_harm50'])
    return dict(fit_directions=len(y), source_output_rows=len(rows), source_least_squares=coeff.tolist())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('source', 'complete'))
    stage = parser.parse_args().stage
    initialize()
    if stage == 'source':
        result = verify_source()
        result.update(verify_physical())
        atomic(RESULTS/'source_verification.json', result)
    else:
        result = audit_panel(RESULTS, WORK)
        result.update(verify_source())
        result['physical_verification'] = digest(RESULTS/'source_verification.json')
        predict = loaded(WORK/'predictions.npz')
        if read(WORK/'MODEL_LOCK.json')['coefficients'][0] == 0.:
            np.testing.assert_array_equal(predict['picks'][:, ARMS.index('source_interaction')],
                                          predict['picks'][:, ARMS.index('v1')])
        result['FINAL_access'] = False
        atomic(RESULTS/'verification.json', result)
    print(result)
