"""Read-only source query sensitivity: two pre-registered identical-grid pairs.

No optimizer, target data, new source fit, best selection, or negative queries.
All common/B0/B1/B2 fixed states are measured, not selected by this diagnostic.
"""
from __future__ import annotations
import argparse
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from scripts.desta3d_v2_aux_backflow_run import model_setup, fields_for, adapter_sha256, Desta3DAdapterV2
from scripts.desta3d_v2_source_fit import prediction_record
from scripts.score_desta3d_v2_reference_audit import sha256_file as sha, read_json as read
from vg_tta.optimizer_checkpoint import cpu_clone
from vg_tta.desta3d_v2_ptd import decode_two_pass

BASE = ROOT / 'artifacts/desta3d_v2/aux_backflow_v1'
FIT = BASE / 'fit_recovery_v2'
OUT = BASE / 'event_query_audit_v1'
INPUTS = ROOT / 'artifacts/desta3d_v2/source_fit/INPUTS.json'
PAIRS = BASE / 'label_audit/SAME_VIDEO_EVENT_PAIRS.json'
ARMS = ('common', 'B0', 'B1', 'B2')


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(payload, f, indent=2, allow_nan=False); f.write('\n')


def tensor_sha(t):
    t = t.detach().contiguous().cpu()
    h = hashlib.sha256()
    h.update(str((str(t.dtype), tuple(t.shape))).encode())
    h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def register():
    assert not OUT.exists()
    assert read(FIT / 'ROOT_SCORE_CROSSCHECK.json')['status'] == 'passed'
    pairs = [p for p in read(PAIRS)['pairs'] if p['physical_time_grid_exact_match']]
    assert len(pairs) == 2 and len({p['source'] for p in pairs}) == 2
    rows_by_key = {r['key']: r for r in read(INPUTS) if r['split'] == 'train'}
    rows = [rows_by_key[p[q]['key']] for p in pairs for q in ('query_a', 'query_b')]
    assert len(rows) == len({r['key'] for r in rows}) == 4
    for a, b in zip(rows[::2], rows[1::2]):
        for field in ('frame_ids', 'video_sha256', 'fps'):
            assert a['input'][field] == b['input'][field]
    config = {'source_queries': 4, 'parents': 2, 'pairs': pairs, 'arms': ARMS,
        'selection': 'all exact-grid pairs from previously registered eight metadata pairs, no outcome-based selection',
        'scope': 'source-train diagnostic only; queries are not contrastive negatives or necessarily semantically exclusive',
        'weights': 'same immutable common A and all three fixed B155 states',
        'measure': 'event response and text pool differences at identical video grid, plus native two-pass PTD endpoint/tube',
        'updates': 0, 'source_labels': 'only source-train labels, after all16 predictions and fields sealed',
        'target_GT_read': False, 'phase_seconds': 600, 'cumulative_cap_seconds': None,
        'remaining_scope': 'not heldout validation, not source-fit selection or TTA efficacy'}
    write(OUT/'CONFIG.json', config); write(OUT/'INPUTS.json', rows)
    pins = dict(read(FIT/'LOCK.json')['pins'])
    for path in (Path(__file__), PAIRS, FIT/'FINAL_B.pt', FIT/'COMMON_A.pt', OUT/'CONFIG.json', OUT/'INPUTS.json'):
        pins[str(path)] = sha(path)
    write(OUT/'LOCK.json', {'pins': pins})
    write(OUT/'REGISTRATION.json', {'time': time.time(), 'config_sha': sha(OUT/'CONFIG.json'),
        'lock_sha': sha(OUT/'LOCK.json'), 'target_GT_read': False, 'new_GPU_started': False})


def run():
    for path, digest in read(OUT/'LOCK.json')['pins'].items():
        assert sha(Path(path)) == digest, path
    lease = (ROOT/'artifacts/spatial_tta_research_v2/gpu.lock').open('a')
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    receipt = ROOT/'artifacts/desta3d_v2/receipts/event_query_audit001.json'
    assert not receipt.exists() and not (OUT/'STARTED.json').exists()
    began = time.monotonic(); status = 'running'; error = None
    def guard():
        assert shutil.disk_usage(ROOT).free > 8 * 2**30, '8GiB reserve'
        assert time.monotonic() - began < 600, 'bounded query diagnostic'
    try:
        write(OUT/'STARTED.json', {'time': time.time(), 'pid': os.getpid(), 'phase_seconds': 600,
            'prior_cumulative_seconds': 33776.531520033, 'cumulative_cap_seconds': None})
        guard(); torch.set_num_threads(4); torch.manual_seed(20260927); torch.cuda.manual_seed_all(20260927)
        torch.cuda.reset_peak_memory_stats()
        weights = torch.load(FIT/'COMMON_A.pt', map_location='cpu', weights_only=False)
        weights.update(torch.load(FIT/'FINAL_B.pt', map_location='cpu', weights_only=False))
        processor, backbone = model_setup(); backbone.eval()
        versions = {n: p._version for n, p in backbone.named_parameters()}
        model = Desta3DAdapterV2(hidden_dim=128, architecture='dual3d', p1_enabled=False).cuda()
        model.set_train_stage('frozen'); model.eval()
        rows = read(OUT/'INPUTS.json'); hashes = {}; grid_rows = []
        for i, row in enumerate(rows):
            guard()
            prompt, prep, fields = fields_for(backbone, processor, row)
            info = {'key': row['key'], 'source': row['source'], 'preprocess': prep,
                'visual_grid_shape': list(fields['visual_grid'].shape), 'visual_grid_sha': tensor_sha(fields['visual_grid']),
                'query_tokens_shape': list(fields['query_tokens'].shape), 'query_tokens_sha': tensor_sha(fields['query_tokens']),
                'frame_times_sha': tensor_sha(fields['frame_times'])}
            grid_rows.append(info)
            if i % 2:
                other = grid_rows[i-1]
                assert info['preprocess'] == other['preprocess']
                assert info['visual_grid_sha'] == other['visual_grid_sha']
                assert info['frame_times_sha'] == other['frame_times_sha']
            for arm in ARMS:
                guard(); model.load_state_dict(weights[arm]); expected = adapter_sha256(model); hashes[arm] = expected
                with torch.inference_mode():
                    field = model(fields['visual_grid'], fields['query_tokens'], query_mask=fields['query_mask'], frame_times=fields['frame_times'])
                    result = decode_two_pass(backbone, processor, prompt, model, fields)
                assert adapter_sha256(model) == expected
                small = {k: cpu_clone(field[k]) for k in ('event_logits', 'referent_logits', 'z_event', 'z_spatial',
                                                        'alpha_event', 'alpha_spatial')}
                small['delta_relative_norm'] = {b: float(field['delta_'+b].float().norm() / fields['visual_grid'].float().norm().clamp_min(1e-12)) for b in ('event','spatial')}
                record = prediction_record(result, row, prep, expected)
                assert torch.equal(record['event_logits'].reshape(-1), small['event_logits'].reshape(-1))
                path = OUT/'predictions'/arm/f'{i:03}.pt'; path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({'prediction': record, 'evidence': small}, path)
                del result, field, small, record
            del prompt, fields; gc.collect()
        assert all(p._version == versions[n] and p.grad is None and not p.requires_grad for n,p in backbone.named_parameters())
        write(OUT/'PHYSICAL_GRID_AUDIT.json', {'rows': grid_rows, 'pairs': 2,
            'actual_pixels_visual_grid_physical_times_equal_within_pairs': True,
            'backbone_unchanged_and_frozen': True, 'optimizer_steps': 0})
        paths = sorted((OUT/'predictions').rglob('*.pt')); assert len(paths) == 16
        write(OUT/'PREDICTIONS_SEAL.json', {'time': time.time(), 'pins': {str(p): sha(p) for p in paths},
            'adapter_hashes': hashes, 'source_train_labels_read': False, 'target_GT_read': False})
        write(OUT/'COMPLETE.json', {'time': time.time(), 'status': 'sealed_readonly_source_predictions',
            'seal_sha': sha(OUT/'PREDICTIONS_SEAL.json'), 'source_queries': 4, 'parents': 2, 'optimizer_steps': 0})
        status = 'predictions_complete'
    except BaseException:
        status = 'failed'; error = traceback.format_exc(); raise
    finally:
        write(receipt, {'allocation': 'event_query_audit001', 'seconds': time.monotonic()-began,
            'status': status, 'failure': error, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'cumulative_cap_seconds': None, 'optimizer_steps': 0})


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('action', choices=['register','run'])
    args = ap.parse_args(); register() if args.action == 'register' else run()
