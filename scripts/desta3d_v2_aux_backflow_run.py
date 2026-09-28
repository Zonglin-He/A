"""Source-only, isolated auxiliary-gradient experiment; never mutates source_fit.

Panel updates are discarded. Fit uses one new common evidence epoch followed by
one fixed B epoch for B0/B1/B2, with complete-window atomic checkpoints.
"""
from __future__ import annotations
import argparse, copy, fcntl, gc, json, math, os, random, shutil, sys, time, traceback
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from scripts.desta3d_v2_p0 import read, write, sha, adapter_sha256
from scripts.desta3d_tta_run_v1 import cpu_copy, scan_nested_gpu_receipts
from vg_tta.desta3d_v2 import Desta3DAdapterV2
from vg_tta.desta3d_v2_training import source_evidence_losses, make_source_optimizer, set_source_lr

BASE = ROOT / 'artifacts/desta3d_v2/aux_backflow_v1'
OLD = ROOT / 'artifacts/desta3d_v2/source_fit'
V1 = ROOT / 'artifacts/desta3d_v1'
SEED = 20260927
MODES = ('B0', 'B1', 'B2')


def atomic_pt(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp.pt')
    torch.save(cpu_copy(value), temp)
    os.replace(temp, path)


def empty_buffers(adapter):
    return {n: None for n, _ in adapter.named_parameters()}


def get_gradients(adapter):
    return {n: None if p.grad is None else p.grad.detach().float().cpu().clone()
            for n, p in adapter.named_parameters()}


def add_buffers(a, b):
    assert a.keys() == b.keys()
    for n, value in b.items():
        if value is not None:
            a[n] = value.clone() if a[n] is None else a[n] + value
    return a


def group_names(adapter):
    names = {id(p): n for n, p in adapter.named_parameters()}
    return {g: [names[id(p)] for p in params]
            for g, params in adapter.parameter_groups().items() if params}


def vector(buffers, names, params):
    return torch.cat([(torch.zeros_like(params[n], device='cpu', dtype=torch.float32)
                       if buffers[n] is None else buffers[n]).reshape(-1) for n in names])


def gradient_summary(adapter, buffers):
    params = dict(adapter.named_parameters())
    groups = group_names(adapter)
    groups['all'] = list(params)
    result = {}
    for group, names in groups.items():
        vs = {name: vector(b, names, params) for name, b in buffers.items()}
        norms = {n: float(v.norm()) for n, v in vs.items()}
        pairs = {}
        for a, b in [('event', 'spatial'), ('task', 'aux')]:
            denom = norms[a] * norms[b]
            pairs[a+'__'+b] = float(vs[a].dot(vs[b]))/denom if denom else None
        ratio = norms['aux']/norms['task'] if norms['task'] else None
        result[group] = {'numel': sum(params[n].numel() for n in names), 'norms': norms,
                         'cosines': pairs, 'weighted_aux_task_ratio': ratio,
                         'raw_SGD_task_factor': None if ratio is None else
                         1 + ratio * (pairs['task__aux'] or 0.)}
    return result


def query_gradients(model, processor, adapter, row, record, fields, data, divisor):
    """Separate task branches and weighted auxiliary gradients, same objective."""
    from vg_tta.desta3d_v2_ptd import branch_injection
    from vg_tta.desta3d_v2_source import split_source_loss_masks
    from scripts.ptd_8b_teacher_feasibility_v1 import joint_loss
    adapter.train()
    parts = {k: empty_buffers(adapter) for k in ('event', 'spatial', 'task', 'aux')}
    details = {'key': row['key'], 'source': row['source'], 'ce_available': data is not None}
    if data is not None:
        masks = split_source_loss_masks(data, processor.tokenizer)
        model.train(); model.model.visual.eval()
        for branch in ('event', 'spatial'):
            adapter.zero_grad(set_to_none=True)
            d = dict(data); d['labels'] = data['labels'].clone()
            d['labels'][~masks[branch].to(d['labels'].device)] = -100
            with branch_injection(model, adapter, data, fields, branch) as capture:
                ce, stats = joint_loss(model, d)
                loss = ce / divisor
                assert torch.isfinite(loss)
                loss.backward()
                details[branch] = {'ce': float(ce.detach()), 'tokens': int(masks[branch].sum()),
                                   'actual_injection_ratio': capture['relative_injection_norm'], **stats}
            parts[branch] = get_gradients(adapter)
            del loss, ce, d, capture
    parts['task'] = add_buffers(add_buffers(empty_buffers(adapter), parts['event']), parts['spatial'])
    adapter.zero_grad(set_to_none=True)
    out = adapter(fields['visual_grid'], fields['query_tokens'], query_mask=fields['query_mask'],
                  frame_times=fields['frame_times'])
    aux = source_evidence_losses(out, record)
    loss = .1 * (aux['ref'] + aux['event']) / divisor
    assert torch.isfinite(loss)
    loss.backward()
    parts['aux'] = get_gradients(adapter)
    details['aux'] = {k: float(v.detach()) for k, v in aux.items()}
    for branch in ('spatial', 'event'):
        details['delta_'+branch+'_norm'] = float(out['delta_'+branch].detach().float().norm())
    adapter.zero_grad(set_to_none=True)
    assert all(p.grad is None and not p.requires_grad for p in model.parameters())
    assert all(v is None or torch.isfinite(v).all() for b in parts.values() for v in b.values())
    return parts, details


def take_step(adapter, optimizer, task, aux, mode):
    from vg_tta.desta3d_v2_aux_backflow import combine_window_gradient_buffers
    combined, diagnostics = combine_window_gradient_buffers(adapter, task, aux, mode=mode)
    old = {n: p.detach().float().cpu().clone() for n, p in adapter.named_parameters()}
    for n, p in adapter.named_parameters():
        p.grad = None if combined[n] is None else combined[n].to(device=p.device, dtype=p.dtype)
    norm = float(torch.nn.utils.clip_grad_norm_([p for p in adapter.parameters() if p.requires_grad], 1.))
    assert math.isfinite(norm)
    optimizer.step(); optimizer.zero_grad(set_to_none=True)
    params = dict(adapter.named_parameters()); groups = group_names(adapter); groups['all'] = list(params)
    actual = {}
    for group, names in groups.items():
        delta = torch.cat([(params[n].detach().float().cpu()-old[n]).reshape(-1) for n in names])
        gt = vector(task, names, params)
        denom = float(gt.norm()*delta.norm())
        actual[group] = {'delta_norm': float(delta.norm()), 'task_dot_delta': float(gt.dot(delta)),
                         'task_delta_cosine': float(gt.dot(delta))/denom if denom else None,
                         'descent_first_order_task_proxy': float(gt.dot(delta)) < 0}
    assert all(torch.isfinite(p).all() for p in adapter.parameters())
    return {'combination': diagnostics, 'pre_clip_norm': norm, 'clip_triggered': norm > 1,
            'actual_AdamW_delta': actual, 'scope': 'local first-order proxy, not measured loss or tube utility'}


def equivalence_check(model, processor, adapter, row, record, fields, data, parts, rng):
    """Real B0 decomposition versus the pinned original combined-loss backward."""
    from scripts.desta3d_v2_source_fit import train_query
    from vg_tta.desta3d_v2_aux_backflow import combine_window_gradient_buffers
    torch.set_rng_state(rng[0]); torch.cuda.set_rng_state_all(rng[1])
    adapter.zero_grad(set_to_none=True)
    train_query(model, processor, adapter, row, record, fields, data, 'repaired', 1, 4)
    native = get_gradients(adapter)
    combined, _ = combine_window_gradient_buffers(adapter, parts['task'], parts['aux'], mode='B0')
    params = dict(adapter.named_parameters()); groups = group_names(adapter); groups['all'] = list(params)
    checks = {}
    for name, names in groups.items():
        a, b = vector(native, names, params), vector(combined, names, params)
        relative = float((a-b).norm()/a.norm().clamp_min(1e-12))
        checks[name] = {'relative_l2': relative, 'max_abs': float((a-b).abs().max()),
                        'passed': bool(torch.allclose(a, b, rtol=.002, atol=2e-6))}
    adapter.zero_grad(set_to_none=True)
    assert all(c['passed'] for c in checks.values()), checks
    return checks


def register(panel_path):
    assert not (BASE/'CONFIG.json').exists()
    config = {'seed': SEED, 'modes': list(MODES), 'rho': .25, 'eps': 1e-12,
              'coefficient_scope': 'one complete accumulation window, shared_stem 19968 parameters',
              'head_gradients_scaled': False, 'task_gradients_scaled': False,
              'panel_manifest': str(Path(panel_path).resolve()), 'source_only': True,
              'A_epochs': 1, 'B_epochs': 1, 'accumulation': 4, 'B_schedule_horizon_steps': 775,
              'state_selection': 'fixed final B; no source-validation selection',
              'source_data': '618 train queries/95 Vid parents;198 source-val queries/31 parents',
              'common_A': 'new evidence warmup; missing old E0 is not reconstructed by assertion',
              'target_GT_read': False, 'cumulative_cap_seconds': None, 'minimum_free_bytes': 8*2**30}
    write(BASE/'CONFIG.json', config)
    paths = list(read(OLD/'LOCK.json')['pins'])
    paths += [str(Path(__file__)), str(ROOT/'vg_tta/desta3d_v2_aux_backflow.py'),
              str(BASE/'CONFIG.json'), config['panel_manifest'],
              str(ROOT/'protocols/desta3d_v2_aux_backflow_v1.md')]
    write(BASE/'LOCK.json', {'time': time.time(), 'pins': {p: sha(p) for p in dict.fromkeys(paths)}})


def snapshot():
    """Called only after original worker releases the GPU lease and writes receipts."""
    lease = (ROOT/'artifacts/spatial_tta_research_v2/gpu.lock').open('a')
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert (BASE/'LOCK.json').exists() and not (BASE/'PANEL_START.pt').exists()
    assert (OLD.parent/'receipts/source_fit003.json').exists()
    before = sha(OLD/'LATEST.pt')
    state = torch.load(OLD/'LATEST.pt', map_location='cpu', weights_only=False)
    assert state['epoch'] >= 1 and state['cursor'] % 4 == 0
    payload = {'adapter': state['adapters']['dual_repaired'], 'optimizer': state['optimizers']['dual_repaired'],
               'epoch': state['epoch'], 'cursor': state['cursor'], 'stage_steps': state['stage_steps']['dual_repaired'],
               'original_checkpoint_sha': before, 'cpu_rng': state['cpu_rng'], 'cuda_rng': state['cuda_rng']}
    atomic_pt(BASE/'PANEL_START.pt', payload)
    assert sha(OLD/'LATEST.pt') == before
    write(BASE/'PANEL_START.json', {k: payload[k] for k in ('epoch','cursor','stage_steps','original_checkpoint_sha')} |
          {'snapshot_sha': sha(BASE/'PANEL_START.pt'), 'original_checkpoint_unchanged': True})


def model_setup():
    from scripts.ptd_spatial_adapter_ab_v1 import processor_load, model_load
    processor = processor_load(); model = model_load()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    return processor, model


def fields_for(model, processor, row):
    from scripts.ptd_spatial_adapter_ab_v1 import frames_for, inputs_for
    from vg_tta.desta3d_v2_ptd import capture_stock_fields
    model.eval()
    frames, ids = frames_for(row, 'clean'); prompt, preprocess = inputs_for(row, processor, frames)
    fields = capture_stock_fields(model, processor, prompt, row['input']['caption'], ids, row['input']['fps'])
    return prompt, preprocess, fields


def run_panel(config, guard):
    from scripts.desta3d_source_fit_v1 import _training_inputs
    manifest = read(config['panel_manifest'])
    # The CPU manifest has one ordered source metadata row per selected query.
    keys = manifest['keys']
    rows_by_key = {r['key']: r for r in read(OLD/'INPUTS.json') if r['split'] == 'train'}
    rows = [rows_by_key[k] for k in keys]
    assert len(rows) == len({r['source'] for r in rows}) == 16
    records = {r['key']: r for r in read(V1/'source_fit/SOURCE_TRAIN_RECORDS.json')}
    state = torch.load(BASE/'PANEL_START.pt', map_location='cpu', weights_only=False)
    processor, model = model_setup()
    adapter = Desta3DAdapterV2(hidden_dim=128, architecture='dual3d', p1_enabled=False).cuda()
    optimizer = make_source_optimizer(adapter, 'repaired', 'B')
    all_results = []
    for wi in range(4):
        guard()
        outpath = BASE/'panel'/f'WINDOW_{wi}.json'
        if outpath.exists():
            all_results.append(read(outpath)); continue
        adapter.load_state_dict(state['adapter']); optimizer.load_state_dict(copy.deepcopy(state['optimizer']))
        start_hash = adapter_sha256(adapter)
        set_source_lr(optimizer, state['stage_steps'], 775, 'repaired')
        parts = {k: empty_buffers(adapter) for k in ('event','spatial','task','aux')}
        query_details = []
        for index, row in enumerate(rows[4*wi:4*wi+4]):
            guard(); torch.manual_seed(SEED+wi*4+index); torch.cuda.manual_seed_all(SEED+wi*4+index)
            prompt, preprocess, fields = fields_for(model, processor, row)
            data, _, _ = _training_inputs(processor, model, row, records[row['key']])
            rng = (torch.get_rng_state(), torch.cuda.get_rng_state_all())
            one, details = query_gradients(model, processor, adapter, row, records[row['key']], fields, data, 4)
            details['gradient_groups'] = gradient_summary(adapter, one)
            if wi == 0 and index == 0:
                eq = equivalence_check(model, processor, adapter, row, records[row['key']], fields, data, one, rng)
                write(BASE/'panel/REAL_B0_EQUIVALENCE.json', {'checks': eq, 'key': row['key'], 'passed': True})
            for k in parts: add_buffers(parts[k], one[k])
            details['preprocess'] = preprocess; query_details.append(details)
            del prompt, fields, data, one
        assert adapter_sha256(adapter) == start_hash
        updates = {}
        stem_names = group_names(adapter)['shared_stem']
        params = dict(adapter.named_parameters())
        raw_stem = {k: vector(b, stem_names, params) for k,b in parts.items()}
        for mode in MODES:
            adapter.load_state_dict(state['adapter']); optimizer.load_state_dict(copy.deepcopy(state['optimizer']))
            set_source_lr(optimizer, state['stage_steps'], 775, 'repaired')
            updates[mode] = take_step(adapter, optimizer, parts['task'], parts['aux'], mode)
            raw_stem['actual_delta_'+mode] = torch.cat([
                (params[n].detach().float().cpu()-state['adapter'][n].float()).reshape(-1)
                for n in stem_names])
        adapter.load_state_dict(state['adapter']); optimizer.load_state_dict(copy.deepcopy(state['optimizer']))
        assert adapter_sha256(adapter) == start_hash
        raw_path = BASE/'panel'/f'WINDOW_{wi}_STEM.pt'
        atomic_pt(raw_path, {'vectors': raw_stem, 'parameter_names': stem_names,
                            'numel': 19968, 'scope': 'shared_stem raw audit vectors only'})
        result = {'window': wi, 'queries': query_details, 'snapshot_sha': sha(BASE/'PANEL_START.pt'),
                  'gradients': gradient_summary(adapter, parts), 'updates': updates,
                  'raw_stem_path': str(raw_path), 'raw_stem_sha': sha(raw_path),
                  'restored_exactly': True, 'validation_GT_read': False, 'target_GT_read': False}
        write(outpath, result); all_results.append(result)
        print('PANEL_WINDOW', wi, 'COMPLETE', flush=True)
        del parts; gc.collect()
    write(BASE/'panel/COMPLETE.json', {'windows': 4, 'queries': 16, 'parents': 16,
          'diagnostic_optimizer_steps': 12, 'persisted_updates': 0,
          'source_only': True, 'snapshot_sha': sha(BASE/'PANEL_START.pt'),
          'windows_sha': {str(BASE/'panel'/f'WINDOW_{i}.json'): sha(BASE/'panel'/f'WINDOW_{i}.json') for i in range(4)}})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=('register', 'snapshot', 'panel'))
    ap.add_argument('--panel-manifest'); ap.add_argument('--allocation', default='panel001')
    ap.add_argument('--phase-seconds', type=float, default=1800)
    args = ap.parse_args()
    if args.action == 'register': return register(args.panel_manifest)
    if args.action == 'snapshot': return snapshot()
    config = read(BASE/'CONFIG.json')
    for p, expected in read(BASE/'LOCK.json')['pins'].items(): assert sha(p) == expected, p
    receipt = OLD.parent/'receipts'/f'aux_backflow_{args.allocation}.json'
    assert not receipt.exists()
    lease = (ROOT/'artifacts/spatial_tta_research_v2/gpu.lock').open('a')
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    began = time.monotonic(); status = 'running'; failure = None
    def guard():
        if shutil.disk_usage(ROOT).free < config['minimum_free_bytes']: raise RuntimeError('8GiB disk reserve')
        if time.monotonic()-began > args.phase_seconds-30: raise TimeoutError('safe finite phase review')
    try:
        guard(); torch.set_num_threads(4); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
        torch.cuda.reset_peak_memory_stats()
        write(BASE/'starts'/f'{args.allocation}.json', {'pid': os.getpid(), 'time': time.time(),
             'action': args.action, 'phase_seconds': args.phase_seconds,
             'prior_seconds': sum(scan_nested_gpu_receipts(p)[0] for p in (V1, OLD.parent))})
        run_panel(config, guard)
        status = 'complete'
    except BaseException:
        failure = traceback.format_exc(); status = 'failed'; raise
    finally:
        write(receipt, {'stage': 'v2_aux_backflow_'+args.action, 'allocation': args.allocation,
             'status': status, 'seconds': time.monotonic()-began, 'failure': failure,
             'peak_allocated_bytes': torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0,
             'cumulative_cap_seconds': None})


if __name__ == '__main__': main()
