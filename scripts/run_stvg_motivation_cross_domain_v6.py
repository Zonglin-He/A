"""One frozen source checkpoint on one target roster, under exclusive GPU lease."""
import argparse
import gc
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.stvg_motivation_cross_domain_common_v6 import *


def make_adapter(model, source, checkpoint):
    if model == 'tubedetr':
        from vg_tta.native_support_decode_fig1_v1 import TubeAdapter
        return TubeAdapter(checkpoint)
    import torch
    from scripts.run_tastvg_evidence_vulnerability_v2 import install_clean_loader
    install_clean_loader()
    from scripts.run_spatial_regression_alignment_v1 import model_load
    from scripts.run_spatial_ssl_gpu_v1 import config
    from vg_tta.native_support_decode_fig1_v1 import TAAdapter
    cohort = 'hcstvg1_test' if source == 'vidstg' else 'vidstg_test'
    cfg = config(cohort)
    assert (ROOT/cfg.checkpoint).resolve() == checkpoint.resolve()
    assert cfg.checkpoint_sha256 == CHECKPOINTS['tastvg'][source][1]
    adapter = TAAdapter.__new__(TAAdapter)
    adapter.model = model_load(cohort).eval().requires_grad_(False)
    adapter.torch = torch
    assert not adapter.model.taev_loader_provenance['annotation_files_opened']
    return adapter


def native_only(z):
    assert z['native_parity'] and len(z['boxes']) == len(z['intervals']) == 6
    return dict(boxes=z['boxes'][0], box_frame_ids=z['box_frame_ids'],
        interval=z['intervals'][0], spatial_valid=z['spatial_valid'][0],
        temporal_valid=z['temporal_valid'][0], format_valid=z['format_valid'],
        native_parity=True, native_final_layer_only=True, preprocess=z['preprocess'])


def state_hash(model):
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        h.update(name.encode()); h.update(str(value.dtype).encode()); h.update(str(tuple(value.shape)).encode())
        h.update(value.reshape(-1).view(__import__('torch').uint8).numpy().tobytes())
    return h.hexdigest()


def run(model, direction, phase):
    import numpy as np
    import torch
    assert model in MODELS and direction in DIRECTIONS and phase in ('qualification', 'formal')
    verify()
    assert priority_release() is not None, 'Current OPD P1 retains GPU priority'
    sys.addaudithook(guard)
    source, target = direction_datasets(direction)
    checkpoint = verify_checkpoint(model, source)
    folder = BASE / direction / model
    if phase == 'formal':
        q = read(BASE / 'QUALIFICATION_BARRIER.json')
        assert q['status'] == 'pass' and q['qualification_cells'] == 8 and not q['GT_read']
        for rel, h in q['files'].items():
            assert sha(BASE/rel) == h
    from scripts.run_final_simplification_v1 import lease
    start = time.time(); done = 0
    inherited = os.environ.get('STVG_FIGURE_GPU_LEASE_FD')
    if inherited is None:
        gpu = lease()
    else:
        gpu = os.fdopen(int(inherited), 'a')
        expected = (ROOT/'artifacts/spatial_tta_research_v2/gpu.lock').stat()
        actual = os.fstat(gpu.fileno())
        assert (actual.st_dev, actual.st_ino) == (expected.st_dev, expected.st_ino)
    try:
        torch.set_num_threads(4)
        torch.manual_seed(SEED); np.random.seed(SEED)
        torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
        torch.cuda.reset_peak_memory_stats()
        adapter = make_adapter(model, source, checkpoint)
        assert not any(p.requires_grad for p in adapter.model.parameters())
        before = state_hash(adapter.model)
        versions = [p._version for p in adapter.model.parameters()]
        roster = read(BASE / direction / 'ROSTER.json')
        subjects = read(BASE / direction / 'SUBJECTS.json')
        rows = roster['rows'][:2] if phase == 'qualification' else roster['rows']
        from vg_tta.exact_frame_decode_audit_v2 import decode as decode_vid
        from vg_tta.tastvg_paper48_hc2_decode_v1 import decode as decode_hc
        passed = []
        for row in rows:
            verify()
            root = folder / ('qualification_predictions' if phase == 'qualification' else 'predictions')
            output = root / f"{row['ordinal']:05}.json"
            assert not output.exists() and not output.with_suffix('.receipt.json').exists(), 'Do not overwrite or silently replay a saved output'
            tick = time.time()
            frames, ids = (decode_hc if target == 'hc2' else decode_vid)(row['input'])
            assert ids == row['frame_ids']
            pixel = hashlib.sha256(frames.tobytes()).hexdigest()
            cp = BASE / direction / 'common_pixels' / f"{row['ordinal']:05}.json"
            binding = dict(pixel_sha256=pixel, frame_ids=ids, shape=list(frames.shape), input_binding_sha256=digest(row['input']))
            if cp.exists():
                assert read(cp) == binding
            else:
                write(cp, binding)
            subject = subjects[str(row['ordinal'])]
            z = native_only(adapter.predict(frames, row, subject, smoke=phase == 'qualification'))
            if phase == 'qualification':
                other = native_only(adapter.predict(frames, row, subject, smoke=True))
                assert digest(z) == digest(other), 'Native qualification must repeat exactly'
                passed.append(dict(ordinal=row['ordinal'], pixel_sha256=pixel,
                    native_output_sha256=digest(z), repeated_native_bitwise=True,
                    GT_read=False))
            assert all(p.grad is None for p in adapter.model.parameters())
            assert versions == [p._version for p in adapter.model.parameters()]
            z.update(model=model, direction=direction, source_dataset=source, target_dataset=target,
                ordinal=row['ordinal'], key_sha256=digest(row['key']), parent_sha256=digest(row['source']),
                frame_ids=ids, pixel_sha256=pixel, input_binding_sha256=digest(row['input']),
                checkpoint_sha256=CHECKPOINTS[model][source][1], parameter_updates=0, GT_read=False,
                qualification=phase == 'qualification', seconds=time.time()-tick,
                runtime_lock_sha256=sha(BASE/'RUNTIME_LOCK.json'))
            write(output, z)
            write(output.with_suffix('.receipt.json'), dict(sha256=sha(output), bytes=output.stat().st_size,
                runtime_lock_sha256=sha(BASE/'RUNTIME_LOCK.json'), GT_read=False,
                qualification=phase == 'qualification', time=time.time()))
            done += 1
            status(folder/'STATUS.json', dict(status='running', phase=phase, direction=direction,
                model=model, source_dataset=source, target_dataset=target, done=done, total=len(rows),
                pid=os.getpid(), GT_read=False, parameter_updates=0, seconds=time.time()-start, time=time.time()))
            print('CROSS_DOMAIN_NATIVE', phase, direction, model, done, len(rows), flush=True)
            del frames, z
            gc.collect(); torch.cuda.empty_cache()
        after = state_hash(adapter.model)
        assert before == after and versions == [p._version for p in adapter.model.parameters()]
        unchanged = dict(parameter_and_buffer_sha256_before=before,
            parameter_and_buffer_sha256_after=after, parameter_versions_unchanged=True,
            parameter_updates=0, optimizer_created=False, frozen_model=True)
        if phase == 'qualification':
            write(folder/'QUALIFICATION.json', dict(status='pass', actual_GPU_qualification_cells=2,
                repeated_native_outputs=True, native_parity=True, cells=passed, source_dataset=source,
                target_dataset=target, source_checkpoint_sha256=CHECKPOINTS[model][source][1],
                accepted_formal_predictions=0, annotations_denied=True, GT_read=False,
                **unchanged, seconds=time.time()-start, peak_allocated=torch.cuda.max_memory_allocated(), time=time.time()))
        else:
            files = seal_cell(model, direction)
            write(folder/'PREDICTION_BARRIER.json', dict(status='sealed', cells=done,
                source_dataset=source, target_dataset=target, files=files, GT_read=False,
                **unchanged, seconds=time.time()-start, peak_allocated=torch.cuda.max_memory_allocated(), time=time.time()))
        status(folder/'STATUS.json', dict(status='qualification_pass' if phase == 'qualification' else 'sealed',
            phase=phase, direction=direction, model=model, done=done, total=len(rows),
            GT_read=False, parameter_updates=0, seconds=time.time()-start, time=time.time()))
    except BaseException:
        status(folder/'STATUS.json', dict(status='failed_preserved', phase=phase, direction=direction,
            model=model, done=done, traceback=traceback.format_exc(), GT_read=False, time=time.time()))
        raise
    finally:
        gpu.close()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('phase', choices=('qualification', 'formal'))
    ap.add_argument('direction', choices=DIRECTIONS)
    ap.add_argument('model', choices=MODELS)
    args = ap.parse_args()
    run(args.model, args.direction, args.phase)
