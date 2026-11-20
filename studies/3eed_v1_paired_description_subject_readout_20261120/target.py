"""Single natural target expression; share the complete frozen v1 forward among the three arms."""
from time import perf_counter
from paired import (ARMS, HERE, ORDINARY, WORK, affine, archive_npz, atomic, capture,
                    digest, direct, engine, initialize, logit, np, read, torch)
from scope183 import check_input, frame, ordinary
from crossplatform.candidate_frontend import sensor_frame_box

OUT = WORK/'target'
BLOCK = 128


def execution():
    headlock = read(WORK/'HEADS_LOCK.json')
    assert digest(WORK/'heads.npz') == headlock['heads']
    assert digest(HERE/'PROTOCOL.json') == headlock['protocol']
    assert digest(HERE/'fit.py') == headlock['training_code']
    assert headlock['parameters'] == 97 and headlock['arms'] == list(ARMS)
    source_seal = read(WORK/'SOURCE_SEAL.json')
    assert source_seal['capture_code']['paired.py'] == digest(HERE/'paired.py')
    return dict(head_lock=digest(WORK/'HEADS_LOCK.json'), code=digest(HERE/'target.py'),
                protocol=digest(HERE/'PROTOCOL.json'), inputs=digest(ORDINARY/'ordinary_PRIVATE/SEALED.json'),
                historical_predictions=digest(ORDINARY/'predictions_PRIVATE/SEALED.json'), block_size=BLOCK,
                arms=list(ARMS), target_labels_read=False, FINAL_access=False)


@torch.inference_mode()
def main():
    cfg = initialize()
    runs = execution()
    OUT.mkdir(exist_ok=True)
    if (OUT/'LOCK.json').exists():
        assert read(OUT/'LOCK.json') == runs
    else:
        atomic(OUT/'LOCK.json', runs)
    roster = ordinary()
    historical = read(ORDINARY/'predictions_PRIVATE/SEALED.json')
    model = engine()
    heads = torch.from_numpy(np.load(WORK/'heads.npz')['heads'])
    w0 = affine(model)
    np.testing.assert_array_equal(w0, heads[0])
    start = perf_counter()
    complete = {}
    for target, count in cfg['ordinary_counts'].items():
        direction = 'waymo_to_'+target
        directory = OUT/target
        directory.mkdir(exist_ok=True)
        pins = {}
        assert len(roster[target]) == count
        for left in range(0,count,BLOCK):
            right = min(count,left+BLOCK)
            filename = f'{left:05d}.npz'
            path = directory/filename
            receipt = path.with_suffix('.json')
            if receipt.exists():
                pin = read(receipt)
                assert digest(path)==pin['sha256'] and pin['first']==left and pin['stop']==right
            else:
                original = ORDINARY/'predictions_PRIVATE'/direction/filename
                assert digest(original)==historical['directions'][direction]['files'][filename]['sha256']
                with np.load(original, allow_pickle=False) as saved:
                    old_boxes, old_slots = saved['four'], saved['choices'][:,0]
                choices, boxes, scores = [], [], []
                maximum = 0.
                for i, row in enumerate(roster[target][left:right]):
                    check_input(row)
                    scene, ev = model._read_observation(frame(row))
                    hidden, base = capture(model._decoder,ev)
                    h,j = torch.from_numpy(hidden), torch.from_numpy(base)
                    logits = np.stack([logit(h,w0,j,head).numpy() for head in heads])
                    np.testing.assert_array_equal(logits[0], base)
                    if i == 0:
                        device = ev['subject_boxes'].device
                        for stored, head in zip(logits, heads):
                            inside = direct(model._decoder,ev,head.to(device))
                            maximum = max(maximum,float(np.abs(inside-stored).max()))
                            np.testing.assert_allclose(inside,stored,rtol=0,atol=3e-6)
                            assert inside.argmax() == stored.argmax()
                    physical = np.stack([sensor_frame_box(box,target,scene.view_index)
                                         for box in ev['subject_boxes'][0].cpu().numpy()])
                    np.testing.assert_array_equal(physical,old_boxes[i])
                    picks = logits.argmax(-1)
                    assert picks[0] == old_slots[i]
                    choices.append(picks)
                    scores.append(logits)
                    boxes.append(physical)
                archive_npz(path, ordinals=np.arange(left,right), choices=np.array(choices,dtype=np.int8),
                            four=np.stack(boxes), scores=np.stack(scores))
                pin = dict(first=left,stop=right,sha256=digest(path),frame_parity=right-left,
                           direct_tolerance_max=maximum)
                atomic(receipt,pin)
            pins[filename] = pin
            print(direction, right, count, round(perf_counter()-start,1),flush=True)
        complete[target] = pins
    atomic(OUT/'SEALED.json',dict(files=complete, n=sum(cfg['ordinary_counts'].values()),
           lock=digest(OUT/'LOCK.json'), target_labels_read=False, FINAL_access=False,
           elapsed_seconds=perf_counter()-start))


if __name__ == '__main__':
    main()
