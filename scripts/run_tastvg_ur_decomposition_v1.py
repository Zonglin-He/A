"""One shared probe set, two exact gradients, three matched writes, frozen H."""
import os
import sys
import time
import traceback
import functools
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
from scripts.tastvg_ur_decomposition_common_v1 import *


def run(mode):
    import torch
    import numpy as np
    from scripts.run_tastvg_evidence_vulnerability_v2 import install_clean_loader
    from scripts.run_tastvg_evidence_vulnerability_v1 import device_tree
    from scripts.run_tastvg_paper48_p5_online_v1 import compact_prediction
    from scripts.run_final_simplification_v1 import lease as gpu_lease
    from vg_tta.tastvg_spatial_online_opd_s1_v1 import SpatialActor
    from vg_tta.tastvg_selected_rollout_v1 import central_with_candidates, rollout_states, target_loss
    from vg_tta.tastvg_native_spatial_rollout_s05_v1 import predict
    from vg_tta.tastvg_spatial_critic_s06_v1 import rewards
    from vg_tta.tastvg_reference_selection_v1 import student_frames
    from vg_tta.tastvg_ur_write_decomposition_v1 import sgd, difference, gradient_geometry
    from methods.decota_final_simplified_v1.tensors import state_hash
    assert mode in ['smoke', 'full']
    verify(); budget()
    cohort = read(BASE / 'COHORT.json'); config = read(BASE / 'CONFIG.json'); cfg = config['params']
    if mode == 'full':
        assert read(BASE / 'SMOKE_ROOT_ACCEPTANCE.json')['status'] == 'pass'
    install_clean_loader(); sys.addaudithook(guard)
    actor = lease = None
    tick = time.monotonic(); counts = dict(suffix_replays=0, backward_calls=0, new_writes=0,
                                          capture_cache_reads=0, expert_cache_reads=0)
    try:
        lease = gpu_lease(); torch.set_num_threads(4); torch.manual_seed(20260929); np.random.seed(20260929)
        torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
        from scripts.run_spatial_regression_alignment_v1 import model_load
        model = model_load('vidstg_test').eval().requires_grad_(False)
        mh = state_hash(model.state_dict()); assert mh == config['checkpoint_state_sha256']
        actor = SpatialActor(model)
        initial = actor.state(); probe_states, spec, _ = rollout_states(initial, cfg['rho'], cfg['direction_count'])
        deltas = [{name: value-initial[name] for name, value in state.items()} for state in probe_states]
        coeff = [model.cfg.SOLVER.BBOX_COEF, model.cfg.SOLVER.GIOU_COEF]
        capture_barrier = read(POOL / 'hc2/CAPTURE_BARRIER.json')
        @functools.lru_cache(maxsize=32)
        def cached(parent, condition):
            path = POOL / 'hc2/capture' / condition / f'{parent:05}.json'
            assert sha(path) == capture_barrier['files'][str(path.relative_to(POOL / 'hc2'))]
            record = read(path); cache = POOL / 'hc2' / record['cache']
            assert sha(cache) == record['sha256']
            counts['capture_cache_reads'] += 1
            return load(cache), record
        def data_at(parent, condition):
            data, record = cached(parent, condition)
            return device_tree(data, 'cuda'), record
        def native(data, state):
            counts['suffix_replays'] += 1
            _, _, pred = predict(model, data, device_tree(state, 'cuda'))
            assert torch.isfinite(pred['boxes']).all()
            return compact_prediction(pred)
        def write_path(row):
            return BASE / 'writes' / row['condition'] / row['order'] / f'{row["donor_arrival"]:05}.pt'
        def make_write(row):
            path = write_path(row)
            if path.with_suffix('.json').exists():
                receipt(path); value = load(path); assert value['row'] == row
                return value
            budget(); old = load(ROOT / row['donor_payload']); step = old['update_steps'][0]
            state = device_tree(old['pre_state'], 'cuda'); actor.restore(state)
            assert state_hash(actor.state()) == row['common_pre_sha'] == step['pre_state_sha256']
            data, cr = data_at(row['donor_parent'], row['condition'])
            assert cr['pixel_sha256'] == row['pixel_sha256']
            with torch.no_grad():
                _, _, pre, _, temporal_candidates = central_with_candidates(actor, data)
                counts['suffix_replays'] += 1
                route = student_frames(temporal_candidates, data['frame_ids'])
            assert route == old['routing'] and route['positions'] == row['routed_positions']
            assert torch.equal(pre['boxes'], step['prediction']['boxes'])
            assert pre['indices'] == step['prediction']['indices']
            probes = []
            for delta, saved in zip(deltas, step['candidates']):
                pred = native(data, {name: state[name]+delta[name] for name in state})
                assert torch.equal(pred['boxes'], saved['prediction']['boxes'])
                assert pred['indices'] == saved['prediction']['indices']
                probes.append(pred)
            assert len(probes) == 9
            evidence = {}
            for branch, relative in [('U', row['uniform_cache']), ('R', row['routed_cache'])]:
                evidence[branch] = load(ROOT / relative); counts['expert_cache_reads'] += 1
            assert evidence['R']['positions'] == route['positions']
            for branch in evidence:
                assert len(evidence[branch]['positions']) == 5
                assert evidence[branch]['boxes'].shape == tuple(probes[0]['boxes'].shape)
            reward = {branch: rewards([p['boxes'].numpy() for p in probes],
                      evidence[branch]['boxes'], evidence[branch]['valid']) for branch in evidence}
            if reward['R'] is None:
                assert step['rewards'] is None and step['update'] is None
            else:
                assert reward['R'].tolist() == step['rewards']
            gradients, metadata = {}, {}
            available = [branch for branch in ['U', 'R'] if reward[branch] is not None]
            if available:
                _, bg, pg = actor.values(data); counts['suffix_replays'] += 1
                assert torch.equal(bg.detach().cpu(), pre['boxes'])
                target = torch.stack([p['boxes'] for p in probes]).cuda().detach()
                for j, branch in enumerate(available):
                    rt = torch.tensor(reward[branch], device='cuda', dtype=torch.float64)
                    loss, pi, qi, distances, _, _ = target_loss(bg, target, rt, coeff,
                        cfg['teacher_temperature'], cfg['student_temperature'], 'rank', 1.)
                    grads = torch.autograd.grad(loss, [p for _, p in actor.named],
                                               retain_graph=j < len(available)-1)
                    counts['backward_calls'] += 1
                    gradients[branch] = {n: g.detach().cpu() for (n, _), g in zip(actor.named, grads)}
                    metadata[branch] = dict(loss=float(loss.detach()), p=pi.detach().cpu(),
                        q=qi.detach().cpu(), distances=distances.detach().cpu(), coefficients=coeff,
                        rewards=reward[branch].tolist(), valid_frames=int(evidence[branch]['valid'].sum()),
                        positions=list(evidence[branch]['positions']), available=True,
                        flat_rewards=bool(float(np.ptp(reward[branch])) == 0.))
            for branch in ['U', 'R']:
                if branch not in gradients:
                    gradients[branch] = {name: torch.zeros_like(value).cpu() for name, value in state.items()}
                    metadata[branch] = dict(loss=None, p=None, q=None, distances=None, coefficients=coeff,
                        rewards=None, valid_frames=0, positions=list(evidence[branch]['positions']),
                        available=False, flat_rewards=False)
            gradients['Specific'] = difference(gradients['R'], gradients['U'])
            common = {name: value.cpu() for name, value in state.items()}
            states = {'pre': common, **{branch: sgd(common, gradients[branch], cfg['lr'])
                       for branch in ['U', 'R', 'Specific']}}
            assert state_hash(states['R']) == step['post_state_sha256']
            if step['update']:
                u = step['update']; assert metadata['R']['loss'] == u['loss_before']
                for field in ['p', 'q', 'distances']:
                    assert torch.equal(metadata['R'][field], u[field]), field
                for name in gradients['R']:
                    assert torch.equal(gradients['R'][name], u['gradients'][name]), name
            donor_predictions = {'pre': compact_prediction(pre)}
            predictions_by_state = {state_hash(common): donor_predictions['pre']}
            for branch in ['U', 'R', 'Specific']:
                sh = state_hash(states[branch])
                if sh not in predictions_by_state:
                    predictions_by_state[sh] = native(data, states[branch])
                donor_predictions[branch] = predictions_by_state[sh]
            assert torch.equal(donor_predictions['R']['boxes'], step['post_prediction']['boxes'])
            assert donor_predictions['R']['indices'] == step['post_prediction']['indices']
            actor.restore(actor.initial)
            assert state_hash(model.state_dict()) == mh
            value = dict(row=row, states=states, gradients=gradients, metadata=metadata,
                probes=probes, donor_predictions=donor_predictions, route=route,
                geometry=gradient_geometry(gradients['U'], gradients['R']),
                state_hashes={b: state_hash(s) for b, s in states.items()},
                exact_R_first_step_parity=True, no_GT=True, GT_read=False,
                common_state_sha256=row['common_pre_sha'], pixel_sha256=cr['pixel_sha256'],
                capture_sha256=cr['sha256'], probe_spec=spec, time=time.time())
            commit(path, value); counts['new_writes'] += 1
            return value
        selected = cohort['rows'][:2] if mode == 'smoke' else cohort['rows']
        for done, row in enumerate(selected, 1):
            value = make_write(row)
            if mode == 'full':
                for target in row['targets']:
                    path = BASE / 'predictions' / row['condition'] / row['order'] / f'{row["donor_arrival"]:05}' / f'{target["arrival"]:05}.pt'
                    if path.with_suffix('.json').exists():
                        receipt(path); continue
                    budget(); data, cr = data_at(target['parent'], row['condition'])
                    if target['parent'] == row['donor_parent']:
                        out = value['donor_predictions']
                    else:
                        out = {}; aliases = {}
                        for branch in ['pre', 'U', 'R', 'Specific']:
                            sh = value['state_hashes'][branch]
                            if sh not in aliases:
                                aliases[sh] = native(data, value['states'][branch])
                            out[branch] = aliases[sh]
                    commit(path, dict(row_key={k: row[k] for k in ['condition', 'order', 'donor_arrival', 'donor_parent', 'source_id']},
                        target=target, predictions=out, interval=out['pre']['indices'],
                        states_sha256=value['state_hashes'], write_payload_sha256=sha(write_path(row)),
                        capture_sha256=cr['sha256'], pixel_sha256=cr['pixel_sha256'],
                        GT_read=False, new_experts=0, new_backbone_forwards=0, target_updates=0))
                    del data
            status(BASE / 'STATUS.json', dict(status='running_'+mode, done=done, total=len(selected),
                    condition=row['condition'], order=row['order'], donor_arrival=row['donor_arrival'],
                    worker_pid=os.getpid(), seconds=time.monotonic()-tick, **counts, time=time.time()))
            print('UR', mode, done, len(selected), row['condition'], row['order'], row['donor_arrival'], flush=True)
        actor.close(); actor = None
        assert state_hash(model.state_dict()) == mh
        verify()
        resources = dict(**counts, worker_wall_seconds=time.monotonic()-tick,
            wall_includes_loading_IO=True, peak_allocated_vram_bytes=torch.cuda.max_memory_allocated(),
            new_experts=0, new_backbone_forwards=0, checkpoint_restored=True, GT_read=False)
        write(BASE / (mode.upper()+'_RESOURCES.json'), resources)
        if mode == 'smoke':
            write(BASE / 'SMOKE.json', dict(status='pass', donors=2, R_first_step_bitwise=True,
                  same_probes=True, route_bitwise=True, GT_read=False, time=time.time()))
            status(BASE / 'STATUS.json', dict(status='smoke_pending_root', done=2, total=96, time=time.time()))
        else:
            receipts = list((BASE / 'writes').rglob('*.json')) + list((BASE / 'predictions').rglob('*.json'))
            assert len(receipts) == 96+1392
            for path in receipts:
                receipt(path.with_suffix('.pt'))
            write(BASE / 'GLOBAL_PREDICTION_BARRIER.json', dict(status='sealed', donors=96,
                targets=1392, logical_role_cells=384, broader_cells=1296,
                files={str(path.relative_to(BASE)): sha(path) for path in receipts},
                cohort_sha256=sha(BASE / 'COHORT.json'), GT_read=False, time=time.time()))
            status(BASE / 'STATUS.json', dict(status='sealed_pending_root_score', done=96, total=96, time=time.time()))
    finally:
        if actor:
            actor.close()
        if lease:
            lease.close()


if __name__ == '__main__':
    try:
        run(sys.argv[1])
    except BaseException as error:
        write(BASE / f'FAILURE_{time.time_ns()}.json', dict(error=repr(error), traceback=traceback.format_exc(), time=time.time()))
        status(BASE / 'STATUS.json', dict(status='failed', error=repr(error), worker_pid=os.getpid(), time=time.time()))
        raise
