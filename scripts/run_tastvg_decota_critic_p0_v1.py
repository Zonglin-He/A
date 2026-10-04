"""One P0: original Direct and differentiable critic, source reset per input."""
import sys, os, time, gc
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.tastvg_decota_critic_common_v1 import *


def prepare():
    import torch
    from scripts import tastvg_decota_c1_common_v1 as c1
    if (BASE / 'RUNTIME_LOCK.json').exists():
        return verify()
    assert read(OLD / 'FINAL_COMPLETION.json')['status'] == 'completed_verified_publication'
    original = c1.verify_seal()
    own = ['protocols/tastvg_decota_critic_p0_v1.md',
        'vg_tta/tastvg_decota_critic_p0_v1.py', 'scripts/tastvg_decota_critic_common_v1.py',
        'scripts/run_tastvg_decota_critic_p0_v1.py', 'scripts/continue_tastvg_decota_critic_p0_v1.py',
        'scripts/score_audit_tastvg_decota_critic_p0_v1.py', 'scripts/report_tastvg_decota_critic_p0_v1.py',
        'tests/test_tastvg_decota_critic_p0_v1.py']
    pins = {f: sha(ROOT / f) for f in own}
    inherited_pins = dict(c1.verify()['pins'])
    for revision in sorted((OLD/'revisions').glob('*.json')):
        inherited_pins.update(read(revision)['pin_overrides'])
    pins.update({f: h for f, h in inherited_pins.items() if f.endswith('.py')})
    inputs = {str(f.relative_to(ROOT)): sha(f) for f in
        [OLD/'RUNTIME_LOCK.json', OLD/'GLOBAL_PREDICTION_BARRIER.json']}
    inputs.update({str(f.relative_to(ROOT)): sha(f) for f in
        [OLD/d/'PLAN.json' for d in DATASETS]})
    support_counts = {}; unique = 0
    for ds in DATASETS:
        p = read(OLD / ds / 'PLAN.json'); assert len(p['rows']) == 48
        write(BASE / ds / 'PLAN.json', p)
        inputs[str((BASE/ds/'PLAN.json').relative_to(ROOT))] = sha(BASE/ds/'PLAN.json')
        stats = dict(inputs=0, observed_frames=0, proposal_count=0, singleton_fallback_frames=0,
            admitted_frames=0, critic_frames=0, skipped_direct_inputs=0, skipped_critic_inputs=0)
        for r in p['rows']:
            for cond in p['conditions']:
                ef = OLD / ds / 'evidence' / cond / f"{r['ordinal']:05}.pt"
                assert sha(ef) == original['files'][str(ef.relative_to(OLD))]
                ex = c1.checked(ef); assert ex['GT_read'] is False
                _, rc = c1.cache(ds, r['pool_parent'], cond, content=False)
                assert ex['pixel_sha256'] == rc['pixel_sha256']
                cf = POOL/ds/read(POOL/ds/'capture'/cond/f"{r['pool_parent']:05}.json")['cache']
                for f in [ef, ef.with_suffix('.json'), cf,
                          POOL/ds/'capture'/cond/f"{r['pool_parent']:05}.json"]:
                    inputs[str(f.relative_to(ROOT))] = sha(f)
                stats['inputs'] += 1; unique += 1
                stats['skipped_direct_inputs'] += not bool(ex['expert']['anchors']['single4'])
                nframes = 0
                for (_, pos), x in ex['expert']['observations'].items():
                    q = x['probe']; b = torch.as_tensor(q['boxes']); s = torch.as_tensor(q['target_scores'])
                    assert b.shape == (len(s), 4) and len(s) <= 3
                    assert torch.isfinite(b).all() and torch.isfinite(s).all()
                    stats['observed_frames'] += 1; stats['proposal_count'] += len(s)
                    stats['admitted_frames'] += bool(q['accepted'])
                    stats['singleton_fallback_frames'] += not x['receipt']['context_active']
                    nframes += bool(len(s) and (b[:, 2:] > 0).all(1).any())
                stats['critic_frames'] += nframes
                stats['skipped_critic_inputs'] += nframes == 0
        support_counts[ds] = stats
    assert unique == 576
    protected = {f: sha(ROOT/f) for f in ['methods/CURRENT_METHOD.json',
        'methods/CURRENT_WORKING_METHOD.json', 'methods/C1_FINAL_RESEARCH_CONFIG.json',
        'methods/C1_TEMPORAL_RESEARCH_STATUS.json']}
    write(BASE/'RUNTIME_LOCK.json', dict(version='tastvg_decota_critic_p0_v1', time=time.time(),
        pins=pins, inputs=inputs, protected_registries=protected,
        unique_inputs=576, logical_arrivals=1152, arms=ARMS, episodic=True,
        new_DINO_forwards=0, formal_backbone_forwards=0, smoke_backbone_inputs=2,
        proposal_temperature=1., reward_temperature=1., spatial_lr=.03,
        spatial_steps=10, spatial_parameters=1792, LN_writeback_fraction=0,
        temporal_updates=0, GT_prediction=False, historical_exposure=True,
        supports=support_counts, old_seal_sha256=sha(OLD/'GLOBAL_PREDICTION_BARRIER.json')))
    status(BASE/'STATUS.json', dict(status='locked_pending_smoke', predictions=0, GT_read=False))
    archive('名单/缓存支持/两臂/温度1/源状态重置锁定，零预测，待无GT完整正控')
    print('PREPARED', support_counts, flush=True)


def start_gpu():
    import numpy as np, torch
    from scripts.run_tastvg_evidence_vulnerability_v2 import install_clean_loader
    install_clean_loader(); sys.addaudithook(guard)
    torch.set_num_threads(4); torch.manual_seed(20260920); np.random.seed(20260920)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
    from scripts.run_final_simplification_v1 import lease
    return lease()


def compare_fit(a, b):
    import torch
    from methods.decota_final_simplified_v1.tensors import state_hash
    assert a['selected_step'] == b['selected_step'] and len(a['path']) == len(b['path'])
    for x, y in zip(a['path'], b['path']):
        assert x['loss'] == y['loss'] and torch.equal(x['boxes'], y['boxes'])
        assert state_hash(x['state']) == state_hash(y['state'])
        if 'update' in x:
            assert torch.equal(x['update']['gradient'], y['update']['gradient'])
    return len(a['path'])


def smoke():
    import torch
    from scripts.run_tastvg_decota_c1_same_domain_v1 import model_for, data_input, decode_for
    from scripts.run_spatial_ssl_gpu_v1 import frozen_forward
    from scripts import tastvg_decota_c1_common_v1 as c1
    from vg_tta.tastvg_decota_c1_same_domain_v1 import NormalizedSpatialReplay
    from vg_tta.c1_luna_tricks_v1 import fit
    from vg_tta.tastvg_decota_critic_p0_v1 import fit_critic
    from methods.decota_final_simplified_v1.tensors import detached, state_hash
    from methods.decota_final_simplified_v1.backbone import query_subject, full_prediction
    verify(); handle = start_gpu(); results = []; tick = time.time()
    try:
        for ds in DATASETS:
            model = model_for(ds); row = read(BASE/ds/'PLAN.json')['rows'][0]
            data, rc = data_input(ds, row, 'clean'); s = NormalizedSpatialReplay(model, data)
            frames, ids = decode_for(ds)(row['input']); assert ids == row['frame_ids']
            batch, records, raw = frozen_forward(model, frames, row)
            assert records == data['records'] and torch.equal(raw.zero['boxes'], s.zero['boxes'])
            assert all(torch.equal(a, b) for a, b in zip(raw.zero['logits'], s.zero['logits']))
            ex = c1.checked(OLD/ds/'evidence/clean/00000.pt')['expert']
            counts = {}
            for arm in ARMS:
                if arm == 'direct':
                    z = fit(s, s.initial, ex['anchors']['single4'], ids, row['key'], 'Scale06')
                    positive = fit(raw, raw.initial, ex['anchors']['single4'], ids, row['key'], 'Scale06')
                else:
                    z = fit_critic(s, s.initial, ex, ids, row['key'])
                    positive = fit_critic(raw, raw.initial, ex, ids, row['key'])
                counts[arm] = compare_fit(z, positive)
                with query_subject(model, batch, row['parses']['subject']):
                    full_prediction(model, batch, ids, records, detached(z['state'], 'cuda'),
                        dict(boxes=z['final'].cuda(), logits=data['prediction']['logits']))
                assert state_hash(s.state()) == state_hash(s.initial)
            results.append(dict(dataset=ds, source_id=0, native_bitwise=True,
                original_direct_all_steps_bitwise=True, critic_full_replay_all_steps_bitwise=True,
                full_reinsertion_native_time_bitwise=True, paths=counts, new_DINO=0))
            del model, data, s, raw, frames, batch, z, positive, ex
            gc.collect(); torch.cuda.empty_cache()
        write(BASE/'SMOKE_ROOT_ACCEPTANCE.json', dict(status='pass', GT_read=False,
            records=results, seconds=time.time()-tick, full_backbone_inputs=2))
        status(BASE/'STATUS.json', dict(status='smoke_pass_pending_formal_predictions', GT_read=False))
        print('SMOKE_PASS', results, flush=True)
    finally:
        handle.close()


def predict(ds):
    import torch
    from scripts.run_tastvg_decota_c1_same_domain_v1 import model_for, data_input
    from scripts import tastvg_decota_c1_common_v1 as c1
    from vg_tta.tastvg_decota_c1_same_domain_v1 import NormalizedSpatialReplay
    from vg_tta.c1_luna_tricks_v1 import fit
    from vg_tta.tastvg_decota_critic_p0_v1 import fit_critic
    from methods.decota_final_simplified_v1.tensors import detached, state_hash
    verify(); assert read(BASE/'SMOKE_ROOT_ACCEPTANCE.json')['status'] == 'pass'
    handle = start_gpu(); model = model_for(ds); mh = state_hash(model.state_dict())
    p = read(BASE/ds/'PLAN.json'); tick = time.time(); done = backwards = 0
    try:
        for row in p['rows']:
            for cond in p['conditions']:
                budget(); f = BASE/ds/'predictions'/cond/f"{row['ordinal']:05}.pt"
                assert not f.exists(), 'No silent replay/overwrite'
                data, rc = data_input(ds, row, cond); s = NormalizedSpatialReplay(model, data)
                ex = c1.checked(OLD/ds/'evidence'/cond/f"{row['ordinal']:05}.pt")
                assert ex['pixel_sha256'] == rc['pixel_sha256']
                origin = detached(s.initial, 'cpu'); fits = {}; seconds = {}
                for arm in ARMS:
                    assert state_hash(s.state()) == state_hash(s.initial)
                    torch.cuda.synchronize(); at = time.perf_counter()
                    if arm == 'direct':
                        z = fit(s, s.initial, ex['expert']['anchors']['single4'],
                                row['frame_ids'], row['key'], 'Scale06')
                    else:
                        z = fit_critic(s, s.initial, ex['expert'], row['frame_ids'], row['key'])
                    torch.cuda.synchronize(); seconds[arm] = time.perf_counter()-at
                    assert state_hash(s.state()) == state_hash(s.initial)
                    fits[arm] = z; backwards += z['gradient_calls']
                value = dict(dataset=ds, parent=row['ordinal'], key=row['key'],
                    condition=cond, frame_ids=row['frame_ids'], source_state=origin,
                    native=detached(data['prediction'], 'cpu'), fits=fits,
                    anchors=ex['expert']['anchors']['single4'], fit_seconds=seconds,
                    evidence_path=str((OLD/ds/'evidence'/cond/f"{row['ordinal']:05}.pt").relative_to(ROOT)),
                    evidence_sha256=sha(OLD/ds/'evidence'/cond/f"{row['ordinal']:05}.pt"),
                    pixel_sha256=rc['pixel_sha256'], episodic_reset=True,
                    LN_writeback_fraction=0, temporal_updates=0, GT_read=False)
                commit(f, detached(value, 'cpu')); done += 1
                status(BASE/ds/'PREDICTION_STATUS.json', dict(status='running', unique_done=done,
                    unique_total=288, backward_calls=backwards, pid=os.getpid(), GT_read=False,
                    seconds=time.time()-tick))
                print('PREDICT', ds, done, 288, round(time.time()-tick, 1), flush=True)
                del s, data, fits, value, z, ex
                if done % 12 == 0:
                    gc.collect()
        assert done == 288 and state_hash(model.state_dict()) == mh
        files = {str(f.relative_to(BASE)):sha(f) for f in (BASE/ds/'predictions').rglob('*.pt')}
        write(BASE/ds/'PREDICTION_BARRIER.json', dict(status='sealed', files=files,
            unique_inputs=done, logical_arrivals=576, GT_read=False, source_unchanged=True,
            seconds=time.time()-tick, backwards=backwards,
            peak_memory_bytes=torch.cuda.max_memory_allocated()))
        status(BASE/ds/'PREDICTION_STATUS.json', dict(status='completed', unique_done=done,
            backward_calls=backwards, seconds=time.time()-tick))
    finally:
        handle.close()


def seal():
    verify(); files = {}
    for ds in DATASETS:
        b = read(BASE/ds/'PREDICTION_BARRIER.json')
        assert b['status'] == 'sealed' and b['unique_inputs'] == 288 and not b['GT_read']
        files.update(b['files'])
    assert len(files) == 576
    for f, h in files.items():
        assert sha(BASE/f) == h
    write(BASE/'GLOBAL_PREDICTION_BARRIER.json', dict(status='sealed', files=files,
        unique_inputs=576, logical_arrivals=1152, arms=ARMS, time=time.time(), GT_read=False))
    status(BASE/'STATUS.json', dict(status='predictions_sealed_pending_CPU', GT_read=False))
    print('SEALED', len(files), flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['prepare','smoke','predict','seal'])
    parser.add_argument('dataset', nargs='?', choices=DATASETS); args = parser.parse_args()
    if args.stage == 'prepare': prepare()
    elif args.stage == 'smoke': smoke()
    elif args.stage == 'seal': seal()
    else: predict(args.dataset)
