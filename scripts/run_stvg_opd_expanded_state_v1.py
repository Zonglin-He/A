"""Expanded non-development state streams with original finite GPU fit and qualification."""
import collections
import gc
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.stvg_opd_expanded_state_common_v1 import *


def captured(model, expert, stage, row, condition, cache, old_payload):
    import torch
    from scripts.run_decota_paper_main_v1 import frames_for, pack_expert, unpack_expert
    from scripts.run_spatial_ssl_gpu_v1 import frozen_forward
    from scripts.stvg_opd_paper_inputs_v1 import observation
    from scripts.stvg_opd_legacy_input_bridge003 import normalized
    from methods.decota_final_simplified_v1.objectives import prediction
    from methods.decota_final_simplified_v1.observations import observations, uniform_positions
    tick = time.perf_counter()
    frames, ids, reuse = frames_for(stage['dataset'], row, cache)
    shifted, pixel, spec = observation(stage['dataset'], row, condition, frames)
    decode_seconds = time.perf_counter() - tick
    torch.cuda.synchronize(); tick = time.perf_counter()
    batch, records, base = frozen_forward(model, shifted, row)
    native = prediction(base.zero['logits'], base.zero['boxes'], records, ids)
    torch.cuda.synchronize(); forward_seconds = time.perf_counter() - tick
    oldinput = OLD / old_payload['input']['path']
    assert sha(oldinput) == old_payload['input']['sha256']
    inp = normalized(load(oldinput))
    assert inp['pixel_sha256'] == pixel and inp['source_model_state_sha256'] == model._fixed_full_state_hash
    assert inp['frame_ids'] == ids and inp['indices'] == native['indices']
    assert inp['interval'] == old_payload['interval'] == native['physical_interval']
    assert inp['corruption_spec'] == spec and torch.equal(inp['native_boxes'], native['boxes'].cpu())
    original_ex = unpack_expert(inp['expert'])
    input_receipts = {a:dict(path=str(oldinput.relative_to(ROOT)),sha256=sha(oldinput),observation_scope='native_event') for a in ARMS}
    cost = dict(decode_corruption_seconds=decode_seconds,STVG_frozen_forward_seconds=forward_seconds,
                actual_new_DINO_calls=0,decoded_frame_reuse=reuse,shared_capture_once_for_three_arms=True)
    del batch,shifted,frames
    return base,native,{a:original_ex for a in ARMS},input_receipts,cost


def validate_lifecycle(z, previous, prevhash, source=None):
    import torch
    fit = z['fit']; arm = z['arm']
    assert z['previous_payload_sha256'] == prevhash and z['Adam_reset'] is True
    assert fit['config'] == z['config'] and fit['selected_step'] == z['config']['steps']
    assert fit['active_parameters'] == 1792 and not fit['GT_used'] and not z['GT_read']
    if previous is None:
        assert torch.count_nonzero(fit['initial'][QUERY]) == 0
        if source is not None:
            assert all(torch.equal(fit['initial'][n], source[n].cpu()) for n in source)
    else:
        expected = initial_state(previous, previous, arm)
        assert all(torch.equal(fit['initial'][n], expected[n]) for n in expected)
    expected = committed(fit['initial'], fit['state'], z['config']['writeback'], arm)
    assert all(torch.equal(expected[n], z['committed'][n]) for n in expected)
    assert z['query_reset'] is True and z['Native_WHEN_fixed'] is True


def run_gpu(name, qualify):
    verify(); activate_old()
    assert os.environ.get('PYTORCH_CUDA_ALLOC_CONF') == ALLOCATOR
    import torch
    from scripts.run_decota_paper_main_v1 import gpu, model_for, read_row
    from methods.decota_final_simplified_v1.observations import SpatialExpert
    from methods.decota_final_simplified_v1.config import EXPERT_SNAPSHOT
    from methods.decota_final_simplified_v1.tensors import state_hash
    from vg_tta.stvg_opd_p4_joint_precision001 import Installation
    from vg_tta.decota_spatial_opd_tunable_v1 import fit as original_import_unused
    from scripts.run_stvg_opd_p1_softmax_recovery_002 import equal
    from scripts.stvg_opd_paper_common_v1 import budget
    stage = stages()[name]
    dest = BASE / ('qualification' if qualify else 'stages') / name
    if (dest / 'PREDICTION_BARRIER.json').exists(): return
    if not qualify:
        assert read(BASE / 'ROOT_QUALIFICATION_READBACK.json')['status'] == 'pass'
    orders = {o: seq[:2] if qualify else seq for o, seq in stage['orders'].items()
              if True}
    total = sum(map(len, orders.values())) * len(stage['conditions']) * len(ARMS)
    lease = gpu(); model = model_for(stage['source']); mh = state_hash(model.state_dict())
    expert = SpatialExpert(ROOT / EXPERT_SNAPSHOT); eh = state_hash(expert.model.state_dict())
    recorder = Installation(qualification=qualify)
    cache = collections.OrderedDict(); logical = {}; files = {}; inputs = {}; controls = []
    start = time.time(); count = dino_calls = 0
    try:
        for condition in stage['conditions']:
            for order, seq in orders.items():
                previous = {a: None for a in ARMS}; prevhash = {a: None for a in ARMS}
                for at, q in enumerate(seq):
                    budget(); row = read_row(stage['dataset'], q)
                    oldpath, oldbinding = old_record(stage, condition, order, at)
                    oldz = load(oldpath)
                    assert oldz['query_ordinal'] == q and oldz['config'] == stage['variant_configs']['full']
                    paths = {a: dest / condition / order / a / f'{at:05}.pt' for a in ARMS}
                    needs = any(not paths[a].exists() for a in ARMS)
                    base = None
                    if needs:
                        base, native, experts, input_receipts, shared = captured(model, expert, stage, row, condition, cache, oldz)
                        dino_calls += shared['actual_new_DINO_calls']
                    for arm in ARMS:
                        f = paths[arm]; cfg = stage['variant_configs'][arm]
                        if f.exists():
                            rc = read(f.with_suffix('.json')); assert sha(f) == rc['sha256'] and not rc['GT_read']
                            z = load(f); assert z['config'] == cfg and z['query_ordinal'] == q and z['arm'] == arm
                        else:
                            initial = initial_state(base.initial, previous[arm], arm); ex = experts[arm]
                            if qualify and not (dest / 'SOURCE_INITIAL.pt').exists(): save(dest / 'SOURCE_INITIAL.pt', clone_state(base.initial))
                            from vg_tta.decota_spatial_opd_tunable_v1 import fit
                            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); tick = time.perf_counter()
                            result = fit(base, initial, ex, row['frame_ids'], row['key'], 'on_policy', config=cfg)
                            torch.cuda.synchronize(); seconds = time.perf_counter() - tick
                            vjp = list(recorder.last_vjp)
                            try:
                                tick = time.perf_counter(); mathcheck = audit(result, ex); cpu_seconds = time.perf_counter() - tick
                            except BaseException:
                                failed = BASE / 'failures' / str(time.time_ns()); failed.mkdir(parents=True)
                                save(failed / 'COMPLETE_UNACCEPTED_FIT.pt', dict(fit=result, input=input_receipts[arm],
                                    stage=name, condition=condition, order=order, arrival=at, query_ordinal=q, arm=arm, GT_read=False))
                                raise
                            z = dict(stage=name, dataset=stage['dataset'], source=stage['source'], condition=condition,
                                order=order, arrival=at, query_ordinal=q, arm=arm, config=cfg, fit=result,
                                committed=committed(initial, result['state'], cfg['writeback'], arm),
                                previous_payload_sha256=prevhash[arm], input=input_receipts[arm],
                                source_input_binding_path=str(oldpath.relative_to(ROOT)), source_input_binding_sha256=sha(oldpath),
                                interval=native['physical_interval'], query_reset=True, Adam_reset=True,
                                Native_WHEN_fixed=True, math_audit=mathcheck, GT_read=False,
                                runtime_sha256=sha(BASE / 'RUNTIME_LOCK.json'),
                                compute=dict(**shared, capture_new_DINO_owner=False,
                                    fit_GPU_seconds=seconds, CPU_math_seconds=cpu_seconds,
                                    CUDA_peak_allocated=torch.cuda.max_memory_allocated(),
                                    CUDA_peak_reserved=torch.cuda.max_memory_reserved(), nominal_observation_budget=4))
                            validate_lifecycle(z, previous[arm], prevhash[arm], base.initial)
                            if qualify:
                                # Same declared intervention with the already qualified numerical recorder.
                                # This is a real second full fit, not a synthetic arithmetic replay.
                                raw = fit(base, initial, ex, row['frame_ids'], row['key'], 'on_policy', config=cfg)
                                torch.cuda.synchronize(); counts = dict(tensors=0, tensor_coordinates=0, scalar_values=0)
                                equal(scientific(result), scientific(raw), counts)
                                assert audit(raw, ex) == mathcheck
                                control = dest / 'repeat_controls' / condition / order / arm / f'{at:05}.pt'
                                save(control, raw)
                                legacy_first = None  # New order has its own source/reset/state trajectory.
                                assert state_hash(model.state_dict()) == mh and state_hash(expert.model.state_dict()) == eh
                                control_record = dict(stage=name, condition=condition, order=order, arrival=at, arm=arm,
                                    query_ordinal=q, first_full_control_bitwise=legacy_first,
                                    complete_repeat_bitwise=counts, repeat_path=str(control.relative_to(ROOT)), repeat_sha256=sha(control),
                                    native_head_VJP=vjp, actual_source_model_state_sha256=mh, actual_expert_model_state_sha256=eh,
                                    source_and_expert_process_hashes_unchanged=True, actual_GPU_fits=2,
                                    formal_predictions_accepted=0, GT_read=False)
                                write(f.with_suffix('.control.json'), control_record)
                            elif at < 2:
                                qf = BASE / 'qualification' / name / condition / order / arm / f.name
                                qualified = load(qf); counts = dict(tensors=0, tensor_coordinates=0, scalar_values=0)
                                assert qualified['input']['sha256'] == z['input']['sha256'] and qualified['config'] == cfg
                                equal(scientific(result), scientific(qualified['fit']), counts)
                                write(BASE / 'first_formal' / name / condition / order / arm / f'{at:05}.json',
                                    dict(status='pass', scope='qualified complete fit must match before formal acceptance',
                                        stage=name, condition=condition, order=order, arrival=at, arm=arm, query_ordinal=q,
                                        qualified_fit_sha256=sha(qf), complete_fit_bitwise=counts, GT_read=False, time=time.time()))
                            save(f, z); write(f.with_suffix('.json'), dict(sha256=sha(f), bytes=f.stat().st_size,
                                runtime_sha256=sha(BASE / 'RUNTIME_LOCK.json'), GT_read=False, time=time.time()))
                        validate_lifecycle(z, previous[arm], prevhash[arm], None if base is None else base.initial)
                        if qualify:
                            control_record = read(f.with_suffix('.control.json'))
                            assert control_record['query_ordinal'] == q and control_record['arm'] == arm
                            controls.append(control_record)
                        previous[arm] = z['committed']; prevhash[arm] = sha(f)
                        inputs[z['input']['path']] = z['input']['sha256']; files[str(f.relative_to(ROOT))] = sha(f)
                        logical[f'{condition}/{order}/{arm}/{at:05}'] = dict(path=str(f.relative_to(ROOT)), sha256=sha(f),
                            query_ordinal=q, input=z['input'])
                        count += 1; del z
                    if base is not None: base.restore(base.initial)
                    del base, oldz
                    gc.collect(); torch.cuda.empty_cache()
                    state = dict(status='running', stage=name, qualification=qualify, done=count, total=total,
                        pid=os.getpid(), condition=condition, order=order, arrival=at, actual_new_DINO_calls=dino_calls,
                        GT_read=False, elapsed_seconds=time.time()-start, time=time.time())
                    status(dest / 'STATUS.json', state); status(BASE / 'PROGRESS.json', state)
                    print('EXPANDED_STATE_PROGRESS', name, condition, order, at, count, total, round(time.time()-start, 1), flush=True)
        assert state_hash(model.state_dict()) == mh and state_hash(expert.model.state_dict()) == eh
        assert count == len(logical) == total
        if qualify:
            assert len(controls) == total
            write(dest / 'BITWISE_CONTROLS.json', dict(status='pass', qualification_arrivals=total,
                actual_GPU_fits=total*2, actual_new_DINO_calls=dino_calls, records=controls,
                formal_predictions_accepted=0, GT_read=False, time=time.time()))
        write(dest / 'PREDICTION_BARRIER.json', dict(status='sealed', stage=name, qualification=qualify,
            new_fit_arrivals=total, files=files, inputs=inputs, logical_records=logical, orders=orders,
            variant_configs=stage['variant_configs'], source_model_state_sha256=mh, expert_model_state_sha256=eh,
            actual_new_DINO_calls=dino_calls, source_and_expert_process_hashes_unchanged=True,
            runtime_sha256=sha(BASE / 'RUNTIME_LOCK.json'), GT_read=False, time=time.time()))
        status(dest / 'STATUS.json', dict(status='sealed', done=count, total=total, GT_read=False, time=time.time()))
    finally:
        recorder.uninstall(); lease.close()


def root_qualification():
    verify()
    import torch
    torch.set_num_threads(2)
    from scripts.run_decota_paper_main_v1 import unpack_expert
    from methods.decota_final_simplified_v1.observations import uniform_positions
    from scripts.run_stvg_opd_p1_softmax_recovery_002 import equal
    from scripts.stvg_opd_legacy_input_bridge003 import normalized
    count = rounds = dino = pair_coords = 0; records = []
    for name, stage in stages().items():
        folder = BASE / 'qualification' / name
        b = read(folder / 'PREDICTION_BARRIER.json'); controls = read(folder / 'BITWISE_CONTROLS.json')
        assert b['status'] == 'sealed' and controls['status'] == 'pass' and not b['GT_read']
        dino += controls['actual_new_DINO_calls']
        for c in controls['records']:
            f = folder / c['condition'] / c['order'] / c['arm'] / f"{c['arrival']:05}.pt"
            assert sha(f) == b['files'][str(f.relative_to(ROOT))]
            z = load(f); inp = normalized(load(ROOT / z['input']['path'])); expert = unpack_expert(inp['expert'])
            assert sha(ROOT / z['input']['path']) == z['input']['sha256']
            assert z['interval'] == inp['interval'] and audit(z['fit'], expert) == z['math_audit']
            prev = None if z['arrival'] == 0 else load(f.with_name('00000.pt'))
            validate_lifecycle(z, None if prev is None else prev['committed'], None if prev is None else sha(f.with_name('00000.pt')))
            if prev is None:
                source = load(folder / 'SOURCE_INITIAL.pt')
                assert all(torch.equal(z['fit']['initial'][n], source[n]) for n in source)
            extent = observation_extent(inp['frame_ids'], inp['indices'], z['arm'])
            assert expert['positions4'] == uniform_positions(inp['frame_ids'], extent, 4)
            rawfile = ROOT / c['repeat_path']; assert sha(rawfile) == c['repeat_sha256']
            compared = dict(tensors=0, tensor_coordinates=0, scalar_values=0)
            equal(scientific(z['fit']), scientific(load(rawfile)), compared)
            assert compared == c['complete_repeat_bitwise']
            count += 1; rounds += len(z['fit']['rounds']); pair_coords += compared['tensor_coordinates']
            records.append(dict(stage=name, condition=z['condition'], arm=z['arm'], arrival=z['arrival'],
                query_ordinal=z['query_ordinal'], qualified_fit_sha256=sha(f), input_sha256=z['input']['sha256']))
    assert count == 24
    write(BASE / 'ROOT_QUALIFICATION_READBACK.json', dict(status='pass', scope='all actual complete qualified fits, independent math/state/input and repeats',
        qualification_arrivals=count, actual_GPU_fits=count*2, complete_fit_rounds=rounds,
        state_coordinates=count*1792, bitwise_pair_coordinates=pair_coords, actual_new_DINO_calls=dino,
        formal_predictions_accepted=0, GT_read=False, records=records,
        independent_full_decoder_Jacobian=False, runtime_sha256=sha(BASE / 'RUNTIME_LOCK.json'), time=time.time()))


if __name__ == '__main__':
    try:
        job = sys.argv[1]
        if job == 'root_qualification': root_qualification()
        else:
            assert job in ['qualification', 'GPU']
            run_gpu(sys.argv[2], job == 'qualification')
    except BaseException:
        d = BASE / 'failures' / str(time.time_ns()); d.mkdir(parents=True, exist_ok=True)
        (d / 'traceback.txt').write_text(traceback.format_exc())
        status(BASE / 'STATUS.json', dict(status='failed_preserved_pending_actual_root', evidence=str(d.relative_to(ROOT)),
            GT_read=False, pid=os.getpid(), time=time.time()))
        raise
