"""Locked 8-parent v2 episodic TTA, with no GT access in registration or inference."""
from __future__ import annotations
import argparse
import fcntl
import gc
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from unittest.mock import patch
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.desta3d_v2_p0 import read, sha, adapter_sha256
from scripts.score_desta3d_v2_aux_recovery import save_once
from scripts.desta3d_tta_run_v1 import scan_nested_gpu_receipts, tensor_sha256, make_mild_photometric_view
from vg_tta.optimizer_checkpoint import cpu_clone
from vg_tta.desta3d_v2_tta_pilot import adapt, configure
from vg_tta.desta3d_v2_prediction_contract import validate_prediction

PARENT = ROOT / 'artifacts/desta3d_v2'
BASE = PARENT / 'tta_v2'
OLD_OUT = BASE / 'target8_B1_v1'
OUT = BASE / 'target8_B1_v2'
SOURCE = BASE / 'source_moments_B1_v1'
RECEIPT = PARENT / 'receipts/tta8_B1_v2.json'
ARMS = ['Frozen', 'sourcefit_noTTA', 'convolution_view', 'calibration_view',
        'convolution_alignment', 'calibration_alignment']
CONDITIONS = ['clean', 'noise_medium', 'defocus_extreme']


def observe_time(pg, processor, nframes, callback):
    """Observe actual PTD endpoint logits without changing probe/cache execution."""
    token_ids = pg.build_ptd_token_ids(processor.tokenizer, max_time_tokens=nframes)
    original_probe = pg._run_cached_ptd_probe
    captures = []
    def probe(*args, **kwargs):
        query = torch.as_tensor(kwargs['query_token_ids']).flatten().tolist()
        if query != [token_ids['ref_end']]:
            return original_probe(*args, **kwargs)
        original_lm = pg._run_language_model
        def lm(*a, **k):
            output, logits = original_lm(*a, **k)
            assert logits.shape[:2] == (1, 6)
            indices = torch.tensor(token_ids['ordered_time_tokens'], device=logits.device)
            captures.append(logits[0, 1:3].index_select(-1, indices).detach().float().cpu())
            return output, logits
        with patch.object(pg, '_run_language_model', lm):
            return original_probe(*args, **kwargs)
    with patch.object(pg, '_run_cached_ptd_probe', probe):
        result = callback()
    return result, {'endpoint_logits': captures[0] if captures else None,
                    'probe_count': len(captures), 'time_token_ids': token_ids['ordered_time_tokens'],
                    'scope': 'first temporal probe; event pass for dual decode; free generated reference'}


def save_pt(p, x):
    assert not p.exists()
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_suffix('.tmp')
    torch.save(cpu_clone(x), temp)
    os.replace(temp, p)


def register():
    assert not OUT.exists()
    a = read(SOURCE / 'ROOT_FULL_READBACK.json')
    assert a['status'] == 'passed' and a['queries'] == 618 and a['parents'] == 95
    assert a['moments_sha'] == sha(SOURCE / 'MOMENTS.json')
    old = ROOT / 'artifacts/desta3d_v1/tta_run/target64_v1/INPUTS.json'
    historical = read(old)
    rows = []
    for cohort in ['hcstvg1_test', 'vidstg_test']:
        pool = sorted([r for r in historical if r['cohort'] == cohort], key=lambda r: r['key'])
        assert len(pool) == 32
        rows.extend(pool[:4])
    assert len(rows) == len({r['source'] for r in rows}) == len({r['input']['video_sha256'] for r in rows}) == 8
    training = read(SOURCE / 'INPUTS.json')
    assert not ({r['source'] for r in rows} & {r['source'] for r in training})
    assert not ({r['input']['video_sha256'] for r in rows} & {r['input']['video_sha256'] for r in training})
    source_config = read(SOURCE / 'CONFIG.json')
    from vg_tta.desta3d_v2 import Desta3DAdapterV2
    model = Desta3DAdapterV2(hidden_dim=128, architecture='dual3d', p1_enabled=False)
    counts = {name: configure(model, name)[1] for name in ['convolution', 'calibration']}
    cfg = {'checkpoint': source_config['checkpoint'], 'queries': 8, 'parents': 8,
        'conditions': CONDITIONS, 'arms': ARMS, 'predictions': 144,
        'selection': 'lexical first 4 keys within each original historical HC32/Vid32 cohort; metadata only',
        'historically_exposed_development_sources': True, 'source_training': 'Vid only; HC is cross-dataset transfer',
        'seed': 20260927, 'optimizer': {'name': 'AdamW', 'lr': 1e-5, 'weight_decay': 0., 'steps': 3, 'clip': 1.},
        'fixed_steps_rationale': 'small matched interface test; three steps permit observing post-initial parameter anchor, no best-state selection',
        'updated_parameters': counts, 'gates': 'frozen, because all declared losses are pre-gate',
        'weights': {'latent': 1., 'referent': 1., 'event': 1., 'parameter_anchor': 1., 'joint': 0.,
                    'alignment_view': 0., 'alignment_alignment': .01},
        'teacher': 'fixed sourcefit on the same observed corrupted input (clean only in explicit clean audit)',
        'student_view': 'same observed pixels; brightness 1.05, contrast .95; identical physical support',
        'decoder': 'event-only then fresh spatial KV with event reference/time; official cached-v3 schedule',
        'diagnostic': 'free output intervals/coordinate distributions plus sourcefit fixed-prefix before/after coordinate KL and cast injection',
        'selection_by_loss_or_target_GT': False, 'target_GT_read': False,
        'primary_readout': 'parent paired TTA-sourcefit vIoU; noise/blur average within parent; clean separate',
        'full_readout': 'also TTA-Frozen, s/t, domain breakdown, format failures, native-good retention and >5pp parent tails',
        'metric': 'original P3 sampled-support convention, including valid time when spatial format fails',
        'phase_seconds': 3600, 'free_disk_bytes': 8*2**30, 'cumulative_cap_seconds': None,
        'resume': 'no automatic restart; preserve partial files and receipt on failure',
        'decision': '8-source development screen; no effect-based sample removal, no automatic promotion or expansion to 64'}
    save_once(OUT / 'CONFIG.json', cfg)
    save_once(OUT / 'INPUTS.json', rows)
    dependencies = [Path(__file__), ROOT / 'vg_tta/desta3d_v2_tta_pilot.py',
        ROOT / 'vg_tta/desta3d_v2_prediction_contract.py', ROOT / 'tests/test_desta3d_v2_prediction_contract.py',
        ROOT / 'scripts/desta3d_v2_tta8.py', OLD_OUT / 'LOCK.json', OLD_OUT / 'FAILURE.json',
        ROOT / 'tests/test_desta3d_v2_tta_pilot.py', ROOT / 'protocols/desta3d_v2_tta8_v1.md',
        ROOT / 'vg_tta/desta3d_v2.py', ROOT / 'vg_tta/desta3d_v2_ptd.py',
        ROOT / 'vg_tta/desta3d_v2_shared_reference.py', ROOT / 'vg_tta/desta3d_v2_shared_reference_cached.py',
        ROOT / 'vg_tta/desta3d_v2_tta_objective.py', ROOT / 'vg_tta/optimizer_checkpoint.py',
        ROOT / 'scripts/ptd_spatial_adapter_ab_v1.py', ROOT / 'scripts/desta3d_tta_run_v1.py',
        ROOT / 'scripts/desta3d_v2_source_fit.py', ROOT / 'scripts/desta3d_v2_reference_audit_cached_v3.py',
        ROOT / 'scripts/corruption_route_retest_v1.py', ROOT / 'vg_tta/exact_frame_decode_audit_v2.py',
        ROOT / 'external/ParallelTubeDecoding/src/model/ptd_generation.py',
        ROOT / 'methods/CURRENT_METHOD.json', PARENT / 'source_fit/LOCK.json',
        SOURCE / 'MOMENTS.json', SOURCE / 'COMPLETE.json', SOURCE / 'ROOT_FULL_READBACK.json',
        Path(cfg['checkpoint']['checkpoint']), old, OUT / 'CONFIG.json', OUT / 'INPUTS.json']
    save_once(OUT / 'LOCK.json', {'pins': {str(p): sha(p) for p in dependencies}})
    save_once(OUT / 'REGISTRATION.json', {'status': 'registered_before_target_inference', 'time': time.time(),
        'GT_read': False, 'GPU_used_by_registration': False, 'scope': 'one 2x2 update-interface/alignment comparison',
        'source_gate': 'selected B1 source-development point mean >= Frozen; CI crosses zero; no stable-gain claim'})
    assert cfg == read(OLD_OUT / 'CONFIG.json') and rows == read(OLD_OUT / 'INPUTS.json')
    reused = {}
    for complete in sorted((OLD_OUT / 'episodes').rglob('EPISODE_COMPLETE.json')):
        assert read(complete)['reset_exact']
        directory = complete.parent
        files = [directory / (arm + '.pt') for arm in ARMS] + [directory / 'INPUT_IDENTITY.json', complete]
        for path in files:
            if path.suffix == '.pt':
                value = torch.load(path, map_location='cpu', weights_only=False)
                validate_prediction(value, 32)
            dest = OUT / path.relative_to(OLD_OUT)
            dest.parent.mkdir(parents=True, exist_ok=True)
            os.link(path, dest)
            assert sha(path) == sha(dest)
            reused[str(dest)] = {'sha256': sha(path), 'source': str(path)}
    assert len(reused) == 40
    save_once(OUT / 'REUSE_SEAL.json', {'status': 'five_complete_episodes_reused_byte_identically',
        'episodes': 5, 'predictions': 30, 'files': reused, 'GT_read': False,
        'old_failed_native_payload': 'not saved before old assertion; input identity preserved; failed episode will be rerun and raw native saved before validation'})
    print('REGISTERED', OUT, counts, flush=True)


def frozen_record(z, row, pre):
    return {'key': row['key'], 'source': row['source'], 'frame_ids': row['input']['frame_ids'],
        'positions': z.get('positions', []), 'boxes_cxcywh': z.get('boxes', torch.empty(0, 4)),
        'geometry_valid': z.get('geometry_valid', torch.empty(0, dtype=torch.bool)),
        'interval': z['interval'], 'format_ok': z['format_ok'], 'completion': z['completion'],
        'preprocess': pre, 'video_sha': row['input']['video_sha256'], 'GT_read': False,
        'coordinate_logits': z.get('logits')}


def run():
    assert not (OUT / 'STARTED.json').exists() and not RECEIPT.exists()
    cfg = read(OUT / 'CONFIG.json')
    for p, h in read(OUT / 'LOCK.json')['pins'].items(): assert sha(Path(p)) == h, p
    for p, h in read(PARENT / 'source_fit/LOCK.json')['pins'].items(): assert sha(Path(p)) == h, p
    reuse = read(OUT / 'REUSE_SEAL.json')
    for p, record in reuse['files'].items():
        assert sha(Path(p)) == record['sha256'] == sha(Path(record['source']))
    lease = (ROOT / 'artifacts/spatial_tta_research_v2/gpu.lock').open('a')
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    start = time.monotonic(); status = 'running'; failure = None; completed = 0
    prior = sum(scan_nested_gpu_receipts(ROOT / 'artifacts' / n)[0] for n in ['desta3d_v1', 'desta3d_v2'])
    try:
        save_once(OUT / 'STARTED.json', {'time': time.time(), 'pid': os.getpid(), 'prior_seconds': prior})
        def guard():
            assert shutil.disk_usage(ROOT).free >= cfg['free_disk_bytes'], 'disk reserve'
            assert time.monotonic()-start < cfg['phase_seconds'], 'engineering phase limit; partials preserved'
        guard(); torch.set_num_threads(4); torch.manual_seed(cfg['seed']); torch.cuda.manual_seed_all(cfg['seed'])
        torch.cuda.reset_peak_memory_stats()
        from scripts.ptd_spatial_adapter_ab_v1 import processor_load, model_load, frames_for, inputs_for, infer
        from scripts.desta3d_v2_source_fit import prediction_record
        from scripts.desta3d_v2_reference_audit_cached_v3 import details
        from vg_tta.desta3d_v2 import Desta3DAdapterV2
        from vg_tta.desta3d_v2_ptd import capture_stock_fields
        from vg_tta.desta3d_v2_shared_reference_cached import decode_shared_reference_two_pass, _spatial_decode_official_cached
        processor = processor_load(); model = model_load()
        import model.ptd_generation as pg
        model.eval().requires_grad_(False)
        adapter = Desta3DAdapterV2(hidden_dim=128, architecture='dual3d', p1_enabled=False).cuda().eval()
        initial = torch.load(cfg['checkpoint']['checkpoint'], map_location='cpu', weights_only=False)['adapter']
        adapter.load_state_dict(initial); adapter.set_train_stage('frozen')
        initial_hash = adapter_sha256(adapter)
        assert initial_hash == cfg['checkpoint']['adapter_sha256']
        moments = read(SOURCE / 'MOMENTS.json')
        raw_paths = []
        for condition in CONDITIONS:
            for index, row in enumerate(read(OUT / 'INPUTS.json')):
                guard()
                episode = OUT / 'episodes' / condition / f'{index:02}'
                if (episode / 'EPISODE_COMPLETE.json').exists():
                    assert read(episode / 'EPISODE_COMPLETE.json')['reset_exact']
                    raw_paths.extend([episode / (arm + '.pt') for arm in ARMS])
                    raw_paths.extend([episode / 'INPUT_IDENTITY.json', episode / 'EPISODE_COMPLETE.json'])
                    completed += 1
                    print('TTA8_REUSE_EPISODE', completed, 24, condition, row['key'], flush=True)
                    continue
                adapter.load_state_dict(initial); adapter.zero_grad(set_to_none=True); adapter.set_train_stage('frozen')
                assert adapter_sha256(adapter) == initial_hash
                frames, ids = frames_for(row, condition)
                assert ids == row['input']['frame_ids']
                prompt, pre = inputs_for(row, processor, frames)
                mild = make_mild_photometric_view(frames)
                view_prompt, view_pre = inputs_for(row, processor, mild)
                fields = capture_stock_fields(model, processor, prompt, row['input']['caption'], ids, row['input']['fps'])
                view = capture_stock_fields(model, processor, view_prompt, row['input']['caption'], ids, row['input']['fps'])
                assert pre['grid'] == view_pre['grid']
                assert torch.equal(fields['frame_times'], view['frame_times'])
                assert fields['visual_grid'].shape == view['visual_grid'].shape
                identity = {'key': row['key'], 'source': row['source'], 'cohort': row['cohort'], 'condition': condition,
                    'frame_ids': ids, 'preprocess': pre, 'view_preprocess': view_pre,
                    'visual_grid_sha': tensor_sha256(fields['visual_grid']), 'query_tokens_sha': tensor_sha256(fields['query_tokens']),
                    'view_visual_grid_sha': tensor_sha256(view['visual_grid']), 'view_query_tokens_sha': tensor_sha256(view['query_tokens']),
                    'frame_times_sha': tensor_sha256(fields['frame_times']), 'sourcefit_adapter_sha': initial_hash,
                    'source_moments_sha': sha(SOURCE / 'MOMENTS.json'), 'GT_read': False}
                p = episode / 'INPUT_IDENTITY.json'; save_once(p, identity); raw_paths.append(p)
                native, native_time = observe_time(pg, processor, len(ids), lambda: infer(model, processor, prompt))
                frozen = frozen_record(native, row, pre)
                frozen['time_distribution'] = native_time
                p = episode / 'Frozen.pt'; save_pt(p, frozen); raw_paths.append(p)
                validate_prediction(frozen, len(ids))
                baseline, baseline_time = observe_time(pg, processor, len(ids),
                    lambda: decode_shared_reference_two_pass(model, processor, prompt, adapter, fields))
                no_update = prediction_record(baseline, row, pre, initial_hash)
                no_update['readout'] = details(baseline)
                no_update['time_distribution'] = baseline_time
                validate_prediction(no_update, len(ids))
                p = episode / 'sourcefit_noTTA.pt'; save_pt(p, no_update); raw_paths.append(p)
                shared = baseline['event'].get('shared_reference_time')
                token_ids = pg.build_ptd_token_ids(processor.tokenizer, max_time_tokens=len(ids))
                baseline_logits = (baseline.get('spatial') or {}).get('logits')
                for arm in ARMS[2:]:
                    guard()
                    adapter.load_state_dict(initial); adapter.zero_grad(set_to_none=True)
                    assert adapter_sha256(adapter) == initial_hash
                    torch.manual_seed(cfg['seed']); torch.cuda.manual_seed_all(cfg['seed'])
                    interface, objective = arm.split('_')
                    update = adapt(adapter, fields, view, interface=interface,
                        alignment=.01 if objective == 'alignment' else 0., moments=moments,
                        steps=cfg['optimizer']['steps'], lr=cfg['optimizer']['lr'])
                    after_hash = adapter_sha256(adapter)
                    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
                    result, updated_time = observe_time(pg, processor, len(ids),
                        lambda: decode_shared_reference_two_pass(model, processor, prompt, adapter, fields))
                    pred = prediction_record(result, row, pre, after_hash)
                    pred['time_distribution'] = updated_time
                    pred['readout'] = details(result); pred['update'] = update
                    pred['sourcefit_adapter_sha'] = initial_hash
                    validate_prediction(pred, len(ids))
                    control = {'defined': False, 'reason': 'sourcefit event/spatial format failure retained'}
                    if shared is not None and baseline_logits is not None:
                        fixed, injection = _spatial_decode_official_cached(
                            model, processor, prompt, fields, adapter, shared, token_ids)
                        changed = fixed.get('logits')
                        if changed is not None:
                            assert changed.shape == baseline_logits.shape
                            log_p, log_q = baseline_logits.float().log_softmax(-1), changed.float().log_softmax(-1)
                            kl = float((log_p.exp() * (log_p-log_q)).sum(-1).mean())
                            control = {'defined': True, 'fixed_sourcefit_reference_time': True,
                                'coordinate_logits': changed, 'KL_sourcefit_to_updated': kl,
                                'changed_elements': int((changed != baseline_logits).sum()),
                                'injection': {k: injection[k] for k in ['relative_injection_norm', 'changed_elements']}}
                    pred['fixed_prefix_control'] = control
                    p = episode / (arm + '.pt'); save_pt(p, pred); raw_paths.append(p)
                    del pred, result, update
                adapter.load_state_dict(initial); adapter.zero_grad(set_to_none=True); adapter.set_train_stage('frozen')
                assert adapter_sha256(adapter) == initial_hash
                save_once(episode / 'EPISODE_COMPLETE.json', {'status': 'complete', 'reset_exact': True,
                    'sourcefit_adapter_sha': initial_hash, 'arms': ARMS, 'GT_read': False})
                raw_paths.append(episode / 'EPISODE_COMPLETE.json')
                completed += 1
                print('TTA8_EPISODE', completed, 24, condition, row['key'], flush=True)
                del frames, mild, prompt, view_prompt, fields, view, baseline, no_update, native, frozen
                gc.collect(); torch.cuda.empty_cache()
        assert completed == 24
        save_once(OUT / 'ALL_PREDICTIONS_SEAL.json', {'status': 'all_144_predictions_sealed',
            'queries': 8, 'parents': 8, 'episodes': 24, 'predictions': 144,
            'pins': {str(p): sha(p) for p in raw_paths}, 'GT_read': False})
        save_once(OUT / 'COMPLETE.json', {'status': 'completed_predictions_unscored', 'GT_read': False,
            'predictions': 144, 'episodes': 24, 'optimizer_steps': 288,
            'sourcefit_reset_exact': True, 'reused_episodes': 5, 'new_optimizer_steps': 228, 'seal_sha': sha(OUT / 'ALL_PREDICTIONS_SEAL.json')})
        status = 'completed'
    except BaseException:
        status = 'failed'; failure = traceback.format_exc()
        save_once(OUT / 'FAILURE.json', {'time': time.time(), 'failure': failure, 'completed_episodes': completed})
        raise
    finally:
        seconds = time.monotonic()-start
        save_once(RECEIPT, {'status': status, 'seconds': seconds, 'prior_seconds': prior,
            'cumulative_seconds': prior+seconds, 'cumulative_cap_seconds': None, 'failure': failure,
            'completed_episodes': completed,
            'peak_bytes': torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else None})
        lease.close()


def launch():
    assert not (OUT / 'STARTED.json').exists() and not RECEIPT.exists()
    start = time.monotonic()
    child = subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()), 'run'], cwd=ROOT)
    wall = time.monotonic()-start
    worker = read(RECEIPT)['seconds'] if RECEIPT.exists() else 0.
    save_once(PARENT / 'receipts/tta8_B1_wrapper_v2.json', {'status': 'completed' if child.returncode == 0 else 'failed',
        'seconds': max(0., wall-worker), 'child_wall_seconds': wall, 'worker_seconds': worker,
        'returncode': child.returncode, 'scope': 'incremental subprocess overhead excluding worker receipt'})
    raise SystemExit(child.returncode)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['register', 'run', 'launch'])
    {'register': register, 'run': run, 'launch': launch}[p.parse_args().action]()
