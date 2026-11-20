"""Read ordinary frozen target activations, with no model selection or target labels."""
from time import perf_counter
from common import (HERE, PREV, WORK, affine, archive_npz, atomic, capture, digest,
                    engine, legacy, np, pin, read, setup, torch)
from scope183 import check_input, frame, ordinary
from crossplatform.candidate_frontend import sensor_frame_box


@torch.inference_mode()
def main():
    setup()
    out = WORK / 'target_features'
    out.mkdir(exist_ok=True)
    history = PREV / 'run_PRIVATE/target'
    previous = read(history / 'SEALED.json')
    pin(out / 'LOCK.json', dict(code=digest(HERE/'target_features.py'), common=digest(HERE/'common.py'),
        previous=digest(history/'SEALED.json'), protocol=digest(HERE/'PROTOCOL.json'),
        inputs=digest(legacy.ORDINARY/'ordinary_PRIVATE/SEALED.json'), target_labels_read=False))
    expected = {t: ordinary()[t] for t in ('drone','quad')}
    assert [len(x) for x in expected.values()] == [2792, 2736]
    model = engine()
    start = perf_counter()
    all_files = {}
    for target, rows in expected.items():
        folder = out / target
        folder.mkdir(exist_ok=True)
        files = {}
        for filename, saved_pin in sorted(previous['files'][target].items()):
            filepath = folder / filename
            receipt = filepath.with_suffix('.json')
            if receipt.exists():
                stamp = read(receipt)
                assert digest(filepath) == stamp['sha256']
            else:
                oldfile = history/target/filename
                assert digest(oldfile) == saved_pin['sha256']
                with np.load(oldfile, allow_pickle=False) as z:
                    seq, boxes, baseline = z['ordinals'], z['four'], z['scores'][:,0]
                hidden, scores = [], []
                for j, index in enumerate(seq):
                    row = rows[int(index)]
                    check_input(row)
                    scene, ev = model._read_observation(frame(row))
                    features, logits = capture(model._decoder, ev)
                    physical = np.stack([sensor_frame_box(box,target,scene.view_index)
                                         for box in ev['subject_boxes'][0].cpu().numpy()])
                    np.testing.assert_array_equal(physical, boxes[j])
                    np.testing.assert_allclose(logits, baseline[j], atol=2e-6, rtol=0)
                    assert logits.argmax() == baseline[j].argmax()
                    hidden.append(features)
                    scores.append(logits)
                archive_npz(filepath, h=np.stack(hidden), j=np.stack(scores), ordinals=seq)
                stamp = dict(sha256=digest(filepath), previous=saved_pin['sha256'], n=len(seq))
                atomic(receipt, stamp)
            files[filename] = stamp
            print(target, filename, round(perf_counter()-start,1), flush=True)
        all_files[target] = files
    atomic(out/'SEALED.json', dict(files=all_files, lock=digest(out/'LOCK.json'),
                                  target_labels_read=False, final_access=False))


if __name__ == '__main__':
    main()
