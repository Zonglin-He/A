"""Natural same-frame, same-class source pairs, with fixed physical v1 four-slot banks."""
import itertools
from time import perf_counter

from paired import (HERE, WORK, SOURCE, affine, anon, archive_npz, atomic, capture, digest,
                    engine, fixed_prior, initialize, many_iou, np, read, torch)
from features import observation
from roles import inventory
from scope import canonical, source_rows
from crossplatform.round7.camera_projection import raw_to_model_points


def clean(text):
    return ' '.join(text.split()).casefold()


def catalog(row):
    objects = read(row['metapath'])['ground_info']
    pairs = []
    for a, b in itertools.combinations(range(len(objects)), 2):
        x, y = objects[a], objects[b]
        if (x['class'].lower() == y['class'].lower() and clean(x['caption']) != clean(y['caption'])
                and np.linalg.norm(np.subtract(x['bbox_3d'][:3], y['bbox_3d'][:3])) >= .12):
            pairs.append((a, b))
    return objects, pairs


def source_identity(row, objects, boxes):
    annotated, _ = inventory(row)
    mat = np.stack([many_iou(boxes, target) for target in annotated], axis=-1)
    centers = raw_to_model_points(np.array([v['bbox_3d'][:3] for v in objects], dtype=float),
                                  'waymo', row['suffix'])
    dist = np.linalg.norm(centers[:, None] - annotated[None, :, :3], axis=-1)
    inv = dist.argmin(1)
    assert np.all(dist.min(1) < .121)
    return mat, inv


@torch.inference_mode()
def sample(model, row, objects, pairs):
    frame = observation(row)
    assert canonical(objects[row['object_index']]['caption']) == frame['text']
    bank, ev = model._read_observation(frame)
    anchors = ev['subject_boxes'][0].cpu().numpy().copy()
    truth, inv = source_identity(row, objects, anchors)
    assignment = (truth > .25)
    proper = assignment & (assignment.sum(1) == 1)[:, None]
    measurements = []
    for a, b in pairs:
        identity = inv[[a, b]]
        assert identity[0] != identity[1]
        hits = proper[:, identity]
        good = hits.any(0)
        selected = np.where(hits, truth[:, identity], -1).argmax(0)
        selected[~good] = -1
        measurements.append(dict(pair=(a, b), slots=selected, valid=good.all(),
                                 ious=truth[:, identity].T, slot_hits=hits.T,
                                 ambiguous=(assignment[:, identity] & ~proper[:, identity]).any(0)))
    enc = {}
    prior_error = 0.
    raw_first = int(row['object_index'])
    for idx in sorted(set(i for pair in pairs for i in pair)):
        if idx == raw_first:
            scene, current = bank, ev
        else:
            other = dict(frame, text=canonical(objects[idx]['caption']))
            scene, current = model._read_observation(other)
            np.testing.assert_array_equal(bank.sampled_points, scene.sampled_points)
        prior, supported = fixed_prior(anchors, scene)
        if idx == raw_first:
            # Zero-sized raw proposals may not overlap any physical box.
            original = current['base_logit'][0].cpu().numpy()
            if supported.all():
                prior_error = float(np.max(np.abs(prior - original)))
                np.testing.assert_allclose(prior, original, atol=2e-6, rtol=0)
        else:
            current = dict(current, subject_boxes=ev['subject_boxes'],
                           base_logit=torch.as_tensor(prior[None], device=ev['base_logit'].device))
        hidden, scores = capture(model._decoder, current)
        enc[idx] = (hidden, scores, supported)
    records = []
    for m in measurements:
        a, b = m['pair']
        records.append(dict(frame_key=anon('scene', row['point_sha256']),
                            rec_key=anon('rec', row['recording']),
                            key=anon('pair', row['point_sha256'], a, b),
                            primary=raw_first in (a, b), h=np.stack([enc[i][0] for i in (a,b)]),
                            j=np.stack([enc[i][1] for i in (a,b)]), slots=m['slots'],
                            ious=m['ious'], hits=m['slot_hits'], ambiguous=m['ambiguous'],
                            support=np.stack([enc[i][2] for i in (a,b)])))
    return records, prior_error


def main():
    initialize()
    start = perf_counter()
    model = engine()
    splits = {s: source_rows('waymo', s) for s in ('fit', 'calibration')}
    for key in ('recording', 'point_sha256'):
        assert not ({r[key] for r in splits['fit']} & {r[key] for r in splits['calibration']})
    archive_npz(WORK/'initial.npz', head=affine(model).numpy())
    seals = {}
    for split, rows in splits.items():
        records, counter, err = [], 0, 0.
        for row in rows:
            objects, pairs = catalog(row)
            if not pairs:
                continue
            part, error = sample(model, row, objects, pairs)
            err = max(err, error)
            records.extend(part)
            counter += 1
            if counter % 40 == 0:
                print(split, counter, 'paired frames;', len(records), 'pairs', round(perf_counter()-start,1), flush=True)
        assert len(records) == {'fit':213, 'calibration':44}[split]
        assert len({r['key'] for r in records}) == len(records)
        path = WORK/f'{split}.npz'
        arrays = {k:np.stack([r[k] for r in records]) for k in records[0]}
        archive_npz(path, **arrays)
        n_valid = int((arrays['slots'] >= 0).all(1).sum())
        seals[split] = dict(sha256=digest(path), n_manifest=len(rows), frames=counter,
                            recordings=len(set(arrays['rec_key'])), pairs=len(records),
                            eligible=n_valid, maximum_native_prior_error=err)
        print(split, seals[split], flush=True)
    atomic(WORK/'SOURCE_SEAL.json', dict(splits=seals, initial=digest(WORK/'initial.npz'),
           protocol=digest(HERE/'PROTOCOL.json'), capture_code={n:digest(HERE/n) for n in ('paired.py','source_pairs.py')},
           elapsed_seconds=perf_counter()-start, target_labels_read=False, FINAL_access=False))


if __name__ == '__main__':
    main()
