"""Paired target-frozen prediction. Truth is consumed only by score.py."""
from pathlib import Path
import gc
import time

import numpy as np
import torch

from common import (ARMS, HERE, ORDER, PLATFORMS, PRIVATE, R165, ROOT, SPEC, array_sha, json_atomic,
                    npz_atomic, ordinary_rows, points_for, read, sha, verify_inputs)
from crossplatform.ground_single_frame import orient_box
from decoder import Decoder
from evidence import check_seed_coordinates, encoder, extract

NAMES = ('r158_pca', *ARMS)
WORK = ROOT/'crossplatform/round165/ordinary'
OUTPUT = PRIVATE/'predictions'
BATCH = 16


def lock_method():
    verify_inputs()
    models = {}
    for source in PLATFORMS:
        d = PRIVATE/source
        seal = read(d/'MODEL_SEAL.json')
        assert seal['inputs'] == sha(PRIVATE/'INPUT_LOCK.json')
        assert seal['fixed_and_joint_weights_bit_equal']
        for arm in ARMS:
            file = d/f'{arm}_TRAINING.json'
            assert sha(file) == seal['arms'][arm]
            stats = read(file)
            assert stats['inputs']['protocol'] == sha(SPEC)
            assert stats['inputs']['decoder'] == sha(HERE/'decoder.py')
            assert stats['inputs']['trainer'] == sha(HERE/'train.py')
            ckpt = d/f'{arm}.pth'
            assert sha(ckpt) == stats['checkpoint']
            models[f'{source}/{arm}'] = stats['checkpoint']
    pinned = tuple(HERE/f for f in ('common.py', 'evidence.py', 'decoder.py', 'predict.py',
                                   'score.py', 'independent_audit.py', 'diagnostics.py', 'protocol.json'))
    contract = {'checkpoints': models, 'code': {p.name: sha(p) for p in pinned},
                'input_lock': sha(PRIVATE/'INPUT_LOCK.json'), 'target_parameter_updates': 0,
                'predicted_arms': NAMES, 'protected_population_opened': False}
    p = PRIVATE/'EVALUATION_LOCK.json'
    if p.exists():
        assert read(p) == dict(contract, predicted_arms=list(NAMES))
    else:
        json_atomic(p, contract)
    return sha(p)


def models_for(source):
    result = {}
    for arm in ARMS:
        d = Decoder(arm).eval().requires_grad_(False)
        state = torch.load(PRIVATE/source/f'{arm}.pth', weights_only=True, map_location='cpu')
        d.load_state_dict(state, strict=True)
        result[arm] = d.cuda()
    return result


def from_block(block, start, stop):
    token_width = int(np.diff(block['offsets'][start:stop+1]).max())
    t = np.zeros((stop-start, token_width, 64), dtype=np.float32)
    for j, (a, b) in enumerate(zip(block['offsets'][start:stop], block['offsets'][start+1:stop+1], strict=True)):
        t[j, :b-a] = block['tokens'][a:b]
    chunk = slice(start, stop)
    return block['raw'][chunk], block['q'][chunk], t


@torch.inference_mode()
def predict_batch(source, target, native_pass, decoders, block, start, stop, coord):
    indices = block['ordinals'][start:stop]
    rows = [ordinary_rows(target)[int(j)] for j in indices]
    pairs = [points_for(row) for row in rows]
    for j, (points, ids) in enumerate(pairs, start):
        assert array_sha(points) == block['points_sha'][j].tobytes()
        assert array_sha(ids) == block['indices_sha'][j].tobytes()
    sampled = np.stack([a for a, _ in pairs])
    archived = from_block(block, start, stop)
    t0 = time.perf_counter()
    feat = extract(native_pass, rows, sampled, archived=archived, coordinate_audit=coord)
    assert np.array_equal(feat['valid'].sum(-1).cpu().numpy(), np.diff(block['offsets'][start:stop+1]))
    torch.cuda.synchronize()
    seconds = [time.perf_counter()-t0]
    orig = list(block['arms']).index(NAMES[0])
    boxes, references, weights, options = [block['boxes'][start:stop, orig]], [], [], []
    supports, entropies, continuous_slots = [], [], []
    for arm in ARMS:
        t0 = time.perf_counter()
        out = decoders[arm](feat)
        b = out['boxes'].cpu().numpy()
        continuous_slots.append(out['passes'][-1][0][:, 0].cpu().numpy())
        k = out['choices'].cpu().numpy()
        assert np.isfinite(b).all() and (b[..., 3:6] > 0).all()
        native_index = np.full(len(b), -1, np.int16)
        if arm == 'fixed_proposals':
            native_index = out['proposal_logits'].argmax(-1).cpu().numpy().astype(np.int16)
            raw = archived[0]
            fixed = [orient_box(sample, raw[j, w]) for j, (sample, w) in
                     enumerate(zip(sampled, native_index, strict=True))]
            b[:, 0] = np.stack(fixed)
            np.testing.assert_array_equal(b[:, 0, :3], raw[np.arange(len(raw)), native_index, :3])
            np.testing.assert_array_equal(b[:, 0, 3:6],
                                          raw[np.arange(len(raw)), native_index, 3:6].clip(min=1e-6))
        p = out['support']
        torch.testing.assert_close(p.sum(-1), torch.ones_like(p[..., 0]), atol=1e-6, rtol=0.)
        supports.append((p*feat['visible'][:, None]).sum(-1).cpu().numpy())
        entropies.append(-(p*p.clamp_min(1e-9).log()).sum(-1).cpu().numpy())
        weights.append(p.half().cpu().numpy())
        boxes.append(b[:, 0])
        references.append(b[:, 1])
        options.append(np.c_[k, native_index].astype(np.int16))
        seconds.append(time.perf_counter()-t0)
    return dict(boxes=np.stack(boxes, axis=1), refs=np.stack(references, axis=1),
                support=np.stack(weights, axis=1), seeds=feat['xyz'].cpu().numpy(),
                visibility=np.stack(supports, axis=1), entropy=np.stack(entropies, axis=1),
                choices=np.stack(options, axis=1), ordinals=indices,
                subject_slots=np.stack(continuous_slots, axis=1),
                seed_visible=feat['visible'].sum(-1).cpu().numpy(),
                is_relation=feat['has_reference'].cpu().numpy(),
                left_right=feat['left_right'].cpu().numpy()), seconds


def direction(source, target, raw_model, decoders, locked, batch=BATCH):
    name = f'{source}_to_{target}'
    out = OUTPUT/name
    out.mkdir(parents=True, exist_ok=True, mode=0o700)
    archived = WORK/name
    seal = read(archived/'SEAL.json')
    assert seal['source'] == source and seal['target'] == target
    assert not seal['protected_payload_access'] and not seal['oracle_target_class_read']
    assert seal['n'] == len(ordinary_rows(target))
    input_receipt = read(archived/'BBF_CUDA_RECEIPT.json')
    assert sha(archived/'SEAL.json') == input_receipt['direction_forward_seal']
    assert seal['n'] == input_receipt['n']
    files, seconds, done, start_time = {}, np.zeros(5), 0, time.perf_counter()
    for fname, digest in sorted(seal['files'].items()):
        f = archived/fname
        assert sha(f) == digest
        p = out/fname
        if p.exists():
            with np.load(p, allow_pickle=False) as z:
                assert str(z['contract']) == locked and str(z['source_block']) == digest
                assert np.array_equal(z['ordinals'], np.arange(done, done+len(z['ordinals'])))
                done += len(z['ordinals'])
                seconds += z['seconds']
        else:
            with np.load(f, allow_pickle=False) as content:
                block = dict(content)
            n = len(block['ordinals'])
            assert np.array_equal(block['ordinals'], np.arange(done, done+n))
            chunks, times = [], np.zeros(5)
            for lo in range(0, n, batch):
                pred, t = predict_batch(source, target, raw_model, decoders, block, lo,
                                        min(n, lo+batch), coord=done == 0 and lo == 0)
                chunks.append(pred)
                times += t
            merged = {key: np.concatenate([c[key] for c in chunks]) for key in chunks[0]}
            npz_atomic(p, **merged, arms=np.asarray(NAMES), contract=locked, source_block=digest, seconds=times)
            seconds += times
            done += n
        files[fname] = sha(p)
        print('TARGET_FROZEN', name, done, seal['n'], round(time.perf_counter()-start_time, 1), flush=True)
    assert done == seal['n']
    value = dict(source=source, target=target, n=done, contract=locked, files=files,
                 seconds=seconds.tolist(), archived_source=sha(archived/'SEAL.json'),
                 target_box_labels_used=False, target_weights_updated=False)
    json_atomic(out/'SEAL.json', value)
    return value


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True, mode=0o700)
    locked = lock_method()
    proofs = {}
    for source in PLATFORMS:
        native_pass = encoder(source)
        decoders = models_for(source)
        for target in PLATFORMS:
            if target == source:
                continue
            path = OUTPUT/f'{source}_to_{target}'/'SEAL.json'
            if not path.exists():
                direction(source, target, native_pass, decoders, locked)
            proof = read(path)
            assert proof['contract'] == locked
            assert proof['target'] == target and proof['source'] == source
            assert all(sha(path.parent/p) == h for p, h in proof['files'].items())
            proofs[f'{source}_to_{target}'] = (sha(path), proof['n'])
        del native_pass, decoders
        gc.collect()
        torch.cuda.empty_cache()
    n = sum(count for _, count in proofs.values())
    assert len(proofs) == len(ORDER) and n == 18620
    json_atomic(OUTPUT/'PREDICTION_SEAL.json', dict(n=n, contract=locked, directions=proofs,
                                                 target_truth_used=False, FINAL_access=False))
    print('TARGET_ALL_PREDICTIONS_SEALED', n, flush=True)


if __name__ == '__main__':
    main()
