"""Fixed two-state source diagnostic, preserving actual injection and joint_loss."""
import argparse, fcntl, gc, os, shutil, subprocess, sys, time, traceback
from pathlib import Path
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.desta3d_v2_p0 import read, sha, adapter_sha256
from scripts.score_desta3d_v2_aux_recovery import save_once
from scripts.desta3d_tta_run_v1 import scan_nested_gpu_receipts, tensor_sha256
from vg_tta.desta3d_v2_cast_probe import TokenEvidence, cast_difference, serialize_with_guard
BASE = ROOT/'artifacts/desta3d_v2'
OLD = BASE/'tta_v2/source_task_control_v2'
MODE = BASE/'tta_v2/source_task_ce_mode_probe_v1'
OUT = BASE/'tta_v2/source_cast_probe_v1'
RECEIPT = BASE/'receipts/source_cast_probe_v1.json'


def register():
    assert not (OUT/'REGISTRATION.json').exists()
    preflight = read(OUT/'CPU_PREFLIGHT.json'); assert preflight['status'] == 'passed'
    for p, h in read(OLD/'LOCK.json')['pins'].items(): assert sha(Path(p)) == h, p
    cfg = {'checkpoint': read(OLD/'CONFIG.json')['checkpoint'], 'source_key': 'vidstg_source_query:28199',
           'source': '5624461612', 'states': ['B1', 'saved_supervised3'], 'optimizer_steps': 0,
           'seed': 20260927, 'phase_seconds': 600, 'storage_cap_bytes': 90_000_000,
           'disk_reserve_bytes': 8*2**30, 'cumulative_cap_seconds': None,
           'source_GT': 'locked first training record only, task diagnostics', 'target_data': False,
           'injection': 'original branch_injection FP32 updated_tokens then BF16 cast',
           'CE': 'original joint_loss, nonmutating lm_head evidence hook, no_grad',
           'parameter_scope': 'saved exact 66816 FiLM/LN difference; no new update',
           'deterministic': {'torch_algorithms': True, 'cudnn': True, 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}}
    assert shutil.disk_usage(ROOT).free - cfg['storage_cap_bytes'] > cfg['disk_reserve_bytes']
    save_once(OUT/'CONFIG.json', cfg)
    save_once(OUT/'INPUT.json', read(OLD/'INPUTS.json')[0])
    save_once(OUT/'SOURCE_RECORD.json', read(OLD/'SOURCE_RECORDS.json')[0])
    paths = [Path(__file__), ROOT/'vg_tta/desta3d_v2_cast_probe.py',
             ROOT/'tests/test_desta3d_v2_cast_probe.py', ROOT/'protocols/desta3d_v2_source_cast_probe_v1.md',
             ROOT/'external/ParallelTubeDecoding/scripts/finetune_video.sh',
             ROOT/'external/ParallelTubeDecoding/src/train/monkey_patch_forward.py',
             ROOT/'methods/CURRENT_METHOD.json', OLD/'LOCK.json', MODE/'ROOT_MODE_READBACK.json',
             MODE/'REPORT.json', OUT/'CPU_PREFLIGHT.json', OUT/'CONFIG.json', OUT/'INPUT.json', OUT/'SOURCE_RECORD.json']
    paths += [OLD/'episodes/00'/p for p in ['INPUT_IDENTITY.json', 'no_update.pt', 'supervised.pt',
              'supervised/FINAL_CALIBRATION.pt', 'supervised/step1.pt', 'supervised/step2.pt', 'supervised/step3.pt']]
    pins = dict(read(OLD/'LOCK.json')['pins']); pins.update({str(p): sha(p) for p in paths})
    save_once(OUT/'LOCK.json', {'pins': pins})
    save_once(OUT/'REGISTRATION.json', {'time': time.time(), 'status': 'registered_before_GPU',
              'source_queries': 1, 'source_parents': 1, 'optimizer_steps': 0,
              'no_target_data': True, 'free_disk_at_registration': shutil.disk_usage(ROOT).free})
    print('REGISTERED', OUT, flush=True)


def run():
    assert not (OUT/'STARTED.json').exists()
    cfg = read(OUT/'CONFIG.json')
    for p, h in read(OUT/'LOCK.json')['pins'].items(): assert sha(Path(p)) == h, p
    lease = (ROOT/'artifacts/spatial_tta_research_v2/gpu.lock').open('a')
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    start = time.monotonic(); status = 'failed'
    prior = sum(scan_nested_gpu_receipts(ROOT/'artifacts'/n)[0] for n in ['desta3d_v1','desta3d_v2'])
    try:
        save_once(OUT/'STARTED.json', {'time': time.time(), 'pid': os.getpid(), 'prior_seconds': prior})
        def guard():
            assert time.monotonic()-start < cfg['phase_seconds'], 'engineering phase'
            assert shutil.disk_usage(ROOT).free >= cfg['disk_reserve_bytes'], 'disk reserve'
        guard()
        torch.set_num_threads(4); torch.use_deterministic_algorithms(True); torch.backends.cudnn.deterministic = True
        torch.manual_seed(cfg['seed']); torch.cuda.manual_seed_all(cfg['seed'])
        from scripts.ptd_spatial_adapter_ab_v1 import processor_load, model_load, frames_for, inputs_for
        from scripts.desta3d_source_fit_v1 import _training_inputs
        from scripts.ptd_8b_teacher_feasibility_v1 import joint_loss
        from vg_tta.desta3d_v2 import Desta3DAdapterV2
        from vg_tta.desta3d_v2_ptd import capture_stock_fields, branch_injection
        from vg_tta.desta3d_v2_source import split_source_loss_masks
        from vg_tta.desta3d_v2_tta_pilot import configure
        from vg_tta.desta3d_v2_source_task_control import active
        pr = processor_load(); model = model_load().eval().requires_grad_(False)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        initial = torch.load(cfg['checkpoint']['checkpoint'], map_location='cpu', weights_only=False)['adapter']
        final = torch.load(OLD/'episodes/00/supervised/FINAL_CALIBRATION.pt', map_location='cpu', weights_only=False)
        adapter = Desta3DAdapterV2(hidden_dim=128, architecture='dual3d', p1_enabled=False).cuda().eval()
        adapter.load_state_dict(initial); configure(adapter, 'calibration')
        assert adapter_sha256(adapter) == cfg['checkpoint']['adapter_sha256']
        names, _ = active(adapter); assert set(names) == set(final)
        row = read(OUT/'INPUT.json'); record = read(OUT/'SOURCE_RECORD.json')
        assert row['key'] == record['key'] == cfg['source_key'] and row['source'] == cfg['source']
        frames, ids = frames_for(row, 'clean'); prompt, pre = inputs_for(row, pr, frames)
        fields = capture_stock_fields(model, pr, prompt, row['input']['caption'], ids, row['input']['fps'])
        data, _, dpre = _training_inputs(pr, model, row, record)
        ident = read(OLD/'episodes/00/INPUT_IDENTITY.json')
        assert dpre == pre == ident['preprocess'] and ids == ident['frame_ids']
        for k in ['visual_grid','query_tokens','frame_times']: assert tensor_sha256(fields[k]) == ident[k+'_sha']
        assert tensor_sha256(data['input_ids']) == ident['teacher_forced_input_sha']
        masks = split_source_loss_masks(data, pr.tokenizer)
        support = {k: v.detach().cpu() for k, v in data.items() if isinstance(v, torch.Tensor) and k != 'pixel_values_videos'}
        support_sha = {k: tensor_sha256(v) for k,v in support.items()}
        mode = read(MODE/'REPORT.json'); states = {}; endpoints = {}
        for state_name, state in [('B1', initial), ('saved_supervised3', {**initial, **final})]:
            adapter.load_state_dict(state); configure(adapter, 'calibration'); h = adapter_sha256(adapter)
            expected_pred = OLD/'episodes/00'/('no_update.pt' if state_name == 'B1' else 'supervised.pt')
            assert h == torch.load(expected_pred, map_location='cpu', weights_only=False)['adapter_sha']
            model.train(); model.model.visual.eval(); adapter.eval(); states[state_name] = {'adapter_sha': h, 'branches': {}}
            for branch in ['event', 'spatial']:
                guard(); d = dict(data); d['labels'] = data['labels'].clone()
                d['labels'][~masks[branch].to(d['labels'].device)] = -100
                ev = TokenEvidence(d); handle = model.lm_head.register_forward_hook(ev.hook)
                try:
                    with torch.no_grad(), branch_injection(model, adapter, data, fields, branch) as cap:
                        ce, stats = joint_loss(model, d)
                        before_cast = cap['fields']['updated_tokens_'+branch].detach().float().cpu().clone()
                        after_cast = cap['updated_tokens'].clone()
                        assert before_cast.numel() == 32*8*14*2560
                        assert after_cast.dtype == torch.bfloat16
                        assert torch.equal(before_cast.reshape_as(after_cast).bfloat16(), after_cast)
                        info = {'ce': float(ce), 'tokens': int(masks[branch].sum()),
                                'relative_injection_norm': cap['relative_injection_norm'],
                                'cast_changed_elements': cap['changed_elements'], **stats}
                        # This hook observes the actual forward; compare exact old loss/injection metadata.
                        assert info == mode['states'][state_name]['cases'][0]['info'][branch], (state_name, branch, info)
                    evidence = ev.finish()
                    states[state_name]['branches'][branch] = {'info': info, 'tokens': evidence,
                        'precast_shape': list(before_cast.shape), 'postcast_shape': list(after_cast.shape),
                        'precast_sha': tensor_sha256(before_cast), 'postcast_sha': tensor_sha256(after_cast),
                        'support_sha': support_sha}
                    endpoints[state_name, branch] = (before_cast, after_cast)
                    print('ENDPOINT', state_name, branch, info, flush=True)
                finally:
                    handle.remove()
                del cap, ce, d; gc.collect()
            assert adapter_sha256(adapter) == h
        model.eval(); adapter.load_state_dict(initial); adapter.set_train_stage('frozen')
        assert adapter_sha256(adapter) == cfg['checkpoint']['adapter_sha256']
        assert all(p.grad is None for p in adapter.parameters())
        assert all(not p.requires_grad and p.grad is None for p in model.parameters())
        differences = {}
        for branch in ['event','spatial']:
            differences[branch] = cast_difference(*endpoints['B1',branch], *endpoints['saved_supervised3',branch])
        step1 = torch.load(OLD/'episodes/00/supervised/step1.pt',map_location='cpu',weights_only=False)
        assert step1['ordered_names'] == names
        delta = torch.cat([(final[n] - initial[n]).reshape(-1) for n in names]).double()
        gt = (step1['raw']['task_event'] + step1['raw']['task_spatial']).double()
        linear = {'ordered_names': names, 'parameter_count': delta.numel(),
                  'actual_three_step_parameter_delta_L2': float(delta.norm()),
                  'initial_task_gradient_L2': float(gt.norm()), 'initial_task_dot_actual_three_step_delta': float(gt.dot(delta))}
        raw = {'states': states, 'differences': differences, 'input_identity': ident,
               'task_support': support, 'task_support_sha': support_sha, 'linearization': linear}
        guard(); used = sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
        blob = serialize_with_guard(raw, used_bytes=used, free_bytes=shutil.disk_usage(ROOT).free,
                                   cap_bytes=cfg['storage_cap_bytes'], reserve_bytes=cfg['disk_reserve_bytes'])
        save_once(OUT/'PRE_WRITE_STORAGE.json', {'actual_serialized_raw_bytes': len(blob), 'used_bytes': used,
                      'free_before': shutil.disk_usage(ROOT).free, 'cap_bytes': cfg['storage_cap_bytes'],
                      'metadata_buffer_bytes': 1_000_000, 'support_truncated': False})
        path = OUT/'RAW_ENDPOINTS.pt'; assert not path.exists()
        with path.with_suffix('.tmp').open('xb') as f: f.write(blob); f.flush(); os.fsync(f.fileno())
        os.replace(path.with_suffix('.tmp'),path)
        report = {'source_key': row['key'], 'source': row['source'], 'optimizer_steps': 0,
                  'states': {k: {'adapter_sha': v['adapter_sha'], 'branches': {b:x['info'] for b,x in v['branches'].items()}} for k,v in states.items()},
                  'cast': {k:v['stats'] for k,v in differences.items()}, 'linearization': linear,
                  'exact_old_CE_injection_replay': True, 'exact_B1_restore': True, 'no_parameter_grad': True,
                  'target_data': False, 'source_GT_diagnostic': True, 'free_disk_after_raw': shutil.disk_usage(ROOT).free,
                  'raw_bytes': path.stat().st_size, 'does_not_establish_cast_causality_or_task_improvement': True}
        save_once(OUT/'REPORT.json', report); guard()
        assert sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()) < cfg['storage_cap_bytes']
        save_once(OUT/'COMPLETE.json', {'time': time.time(), 'status': 'completed',
                  'pins': {str(p):sha(p) for p in OUT.iterdir() if p.is_file() and p.name != 'RUN001.log'}})
        status = 'completed'
    except BaseException:
        save_once(OUT/'FAILURE.json', {'time': time.time(), 'failure': traceback.format_exc()}); raise
    finally:
        seconds = time.monotonic()-start
        save_once(RECEIPT, {'status': status, 'seconds': seconds, 'prior_seconds': prior,
                   'cumulative_seconds': prior+seconds, 'cap': None, 'optimizer_steps': 0})
        lease.close()


def launch():
    assert not RECEIPT.exists()
    start = time.monotonic()
    child = subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()), 'run'], cwd=ROOT)
    wall = time.monotonic()-start
    worker = read(RECEIPT)['seconds'] if RECEIPT.exists() else 0.
    save_once(BASE/'receipts/source_cast_probe_wrapper_v1.json', {'status': 'completed' if child.returncode == 0 else 'failed',
          'seconds': max(0.,wall-worker), 'child_wall_seconds': wall, 'worker_seconds': worker,
          'scope': 'nonoverlapping subprocess import/check/finalization overhead', 'cap': None})
    raise SystemExit(child.returncode)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['register','run','launch'])
    {'register':register, 'run':run, 'launch':launch}[p.parse_args().action]()
