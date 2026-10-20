"""Real-checkpoint, real-input interface qualification, without GT scoring."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--hc2-request', required=True)
    p.add_argument('--vidstg-request', required=True)
    p.add_argument('--output', default='results/tubedetr/qualification.json')
    args = p.parse_args()
    import numpy as np
    import torch
    from stvg.runtime import setup, lease, sha
    from stvg.tube_decode import decode, asset_identity
    from stvg.inputs import validate_input, validate_metadata
    from stvg.tensors import state_hash
    from stvg.tubedetr import TubePredictor, load_source, native_forward, TubeReplay, endpoint_posterior
    from stvg.teacher_service import PersistentRefer
    from stvg.positions import observation_positions
    from stvg.tube_runner import implementation_hash, runtime_identity, write, save_prediction
    setup()
    lock = lease()
    start = time.perf_counter()
    attempt = str(time.time_ns())
    results = {}
    teacher = None
    try:
        teacher = PersistentRefer(ROOT/'logs/qualification_teacher.log')
        for ds, path in [('hc2', args.hc2_request), ('vidstg', args.vidstg_request)]:
            request = json.loads(Path(path).read_text())
            if set(request) != {'dataset', 'key', 'metadata'} or request['dataset'] != ds:
                raise ValueError('Qualification requires an input-only request for each target')
            validate_metadata(request['metadata'])
            frames, fids = decode(request['metadata'], ds)
            validate_input(frames, fids, request['metadata'])
            predictor = TubePredictor(ds, teacher=teacher)
            model = load_source(predictor.config)
            predictor.model = model
            before = state_hash(model.state_dict())
            native = native_forward(model, frames, fids, request['metadata']['caption'])
            positions = observation_positions(fids, native['extent'])
            rp = TubeReplay(model, native, native['source'].double().numpy(), positions)
            boxes = rp.values()['boxes']
            gradient = torch.autograd.grad(boxes[positions].sum(), rp.state)[0]
            if not torch.isfinite(gradient).all() or not gradient.abs().sum() > 0:
                raise RuntimeError('The actual pretrained decoder has no finite state gradient')
            with torch.no_grad():
                rp.state.add_(.01*gradient/gradient.abs().max())
                changed = rp.values()['native']
                if torch.equal(changed, native['source'].to(changed)):
                    raise RuntimeError('Content perturbation has no output effect')
                rp.state.zero_()
                if not torch.equal(rp.values()['native'].cpu(), native['source']):
                    raise RuntimeError('Native output changed after state reset')
            q = endpoint_posterior(native['logits'])
            if not np.isclose(q.sum(), 1) or np.tril(q).any() or len(native['layers']) != 6:
                raise RuntimeError('Invalid native temporal interface')
            del rp, boxes, changed, gradient
            torch.cuda.empty_cache()
            output = predictor.predict(frames, fids, request['metadata'], request['key'])
            # Fresh re-encoding after adaptation catches accidental shared network state.
            repeated = native_forward(model, frames, fids, request['metadata']['caption'])
            if not torch.equal(native['source'], repeated['source']) or not torch.equal(native['logits'], repeated['logits']):
                raise RuntimeError('Query adaptation altered a subsequent native prediction')
            if state_hash(model.state_dict()) != before:
                raise RuntimeError('Frozen network state changed')
            dest = ROOT/f'outputs/qualification/{attempt}/{ds}.npz'
            save_prediction(dest, output, dict(request_sha256=sha(path), scope='interface qualification; no GT scoring'))
            results[ds] = dict(checkpoint_sha256=model.load_audit['checkpoint_sha256'],
                load=model.load_audit, request_sha256=sha(path),
                physical_grid_sha256=hashlib.sha256(np.array(fids, dtype=np.int64).tobytes()).hexdigest(),
                real_frames=len(fids), zero_replay_bitexact=True, reset_bitexact=True,
                full_reencode_after_adaptation_bitexact=True, network_state_hash_unchanged=True,
                actual_pretrained_content_gradient_finite_nonzero=True, temporal_layers=6,
                official_native_temporal_postprocess=True, same_eight_frame_evidence=True,
                full_locked_updates=output['cost']['actual_updates'], cost=output['cost'], GT_read=False)
            del model, predictor, native, repeated, frames, output
            gc.collect()
            torch.cuda.empty_cache()
        write(args.output, dict(status='pass', scope='two real-input interface checks, not accuracy evidence',
            implementation_sha256=implementation_hash(), datasets=results, decoder_asset=asset_identity(), runtime_asset=runtime_identity(),
            elapsed_seconds=time.perf_counter()-start, labels_read=False,
            development_exposure='register both input identities as engineering exposure before any later task scoring'))
        print(json.dumps(dict(status='pass', output=args.output, datasets=list(results))), flush=True)
    finally:
        if teacher:
            teacher.close()
        lock.close()


if __name__ == '__main__':
    main()
