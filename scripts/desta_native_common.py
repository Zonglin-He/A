"""Paths and write-once bookkeeping for the user-authorized DESTA grid."""
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
from vg_tta.desta3d_v3_oracle_io import OUT, read, write, sha, check_pins, tensor_sha, total_prior

D = OUT / 'dual_expert_native_v1'
A0 = OUT / 'a0_fast_screen_v1'
A3 = OUT / 'a03_shared_basis_v1'
A5 = OUT / 'a05_unlabeled_signal_v1'
GAP = OUT / 'oracle_mixer_gap_v1'
DINO = ROOT / '.cache/grounding_dino_http_v1/models--IDEA-Research--grounding-dino-tiny/snapshots/a2bb814dd30d776dcf7e30523b00659f4f141c71'
SAM = ROOT / 'checkpoints/STVG-R1-tracking/sam2.1_hiera_large.pt'


def load(path):
    import torch
    return torch.load(path, map_location='cpu', weights_only=False)


def stage(name, dependencies):
    dest = OUT / ('desta_native_' + name)
    assert not dest.exists(), 'Use a new run version; preserve failures'
    cfg = read(D / 'CONFIG.json')
    write(dest / 'CONFIG.json', {**cfg, 'phase_seconds': 7200})
    paths = [D / 'CONFIG.json', D / 'DEV64.json', D / 'DEV16.json', D / 'QC.pt', *dependencies]
    write(dest / 'LOCK.json', {'pins': {str(p): sha(p) for p in paths}})
    write(dest / 'REGISTRATION.json', {'time': time.time(), 'name': name, 'no_GT_in_worker': True})
    return dest


def complete(directory, **extra):
    write(directory / 'COMPLETE.json', {'files': {p.name: sha(p) for p in directory.iterdir() if p.is_file()}, **extra})


def verified(directory):
    receipt = directory / 'COMPLETE.json'
    if not receipt.exists():
        if directory.exists():
            raise RuntimeError(f'Partial retained, explicit recovery required: {directory}')
        return False
    check_pins({str(directory / p): h for p, h in read(receipt)['files'].items()})
    return True


def prepare():
    import numpy as np
    import torch
    from dataclasses import asdict
    from desta3d.native_adaptation import grid_configs
    assert not D.exists()
    rows = read(A0 / 'DEV64.json')
    small = read(A5 / 'INPUTS.json')
    assert len(rows) == 64 and len(small) == 16
    assert len({r['source'] for r in rows}) == len({r['source'] for r in small}) == 16
    mapping = {r['key']: i for i, r in enumerate(rows)}
    indices = [mapping[r['key']] for r in small]
    q = load(A0 / 'BASIS.pt').double().numpy()
    b = np.load(A3 / 'TRAIN_BASIS.npz')['basis'][:, :16]
    qc = (q @ b).astype(np.float32)
    assert np.max(abs(qc.astype(float).T @ qc.astype(float) - np.eye(16))) < 2e-6
    cfg = dict(seed=20260928, grid={k: asdict(v) for k, v in grid_configs().items()},
               dev16_indices=indices, keep_top=6, tie_tolerance_pp=1e-6,
               minimum_free_bytes=8*2**30, maximum_new_bytes=25*2**30,
               cap=None, prior_seconds=total_prior(), fresh_read=False, target_read=False,
               source_GT_in_worker=False, native_support='fixed B1 trace across all updates',
               temporal_expert='UniversalVTG pretrained best + PE-Core-L14-336',
               temporal_sampling='uniform physical 2fps slots, nearest existing PTD observations only',
               spatial_expert='GroundingDINO tiny full-caption top-scoring anchor + SAM2.1 large bidirectional tracking',
               dino_box_threshold=.25, dino_text_threshold=.30, mask_threshold=0.,
               spatial_selection='global maximum detector score; ties earliest frame then detector order',
               update='full F normalize, weight, project R16, normalized step, actual-Gram radius projection',
               scoring='offline exposed-source GT after complete per-stage seals; parent macro vIoU',
               selection='Dev16 top6; Dev64 best1; tie fewer >5pp vIoU harms then lexical config id',
               ablations='best Dev64 configuration T-only and S-only; B1/no-update identity; saved GT-R16 diagnostic')
    import shutil
    assert shutil.disk_usage(ROOT).free > cfg['minimum_free_bytes'] + cfg['maximum_new_bytes']
    write(D / 'CONFIG.json', cfg); write(D / 'DEV64.json', rows); write(D / 'DEV16.json', small)
    torch.save(torch.from_numpy(qc), D / 'QC.pt')
    paths = [A0 / 'DEV64.json', A5 / 'INPUTS.json', A0 / 'BASIS.pt', A3 / 'TRAIN_BASIS.npz',
             D / 'CONFIG.json', D / 'DEV64.json', D / 'DEV16.json', D / 'QC.pt',
             ROOT / 'protocols/desta_dual_expert_native_v1.md', ROOT / 'desta3d/native_adaptation.py']
    write(D / 'REGISTRATION.json', {'time': time.time(), 'status': 'registered_before_GPU',
          'pins': {str(p): sha(p) for p in paths}, 'old_NO_GO_preserved': True,
          'user_authorized_reopening': 'two independent specialists + native pseudo-gradient R16 iterative adaptation'})
    write(D / 'EXPERT_ASSETS.json', {'pins': {str(p): sha(p) for p in [
        ROOT/'checkpoints/universalvtg/models/best.pth', ROOT/'checkpoints/universalvtg/opt.yaml',
        Path('/home/wwww/.cache/huggingface/hub/models--facebook--PE-Core-L14-336/snapshots/bafb0f76541d399057e980a25947f67acec76575/PE-Core-L14-336.pt'),
        DINO/'model.safetensors', SAM]}})
    print('Registered 27 -> 6 -> 1 DESTA grid', flush=True)


if __name__ == '__main__':
    prepare()
