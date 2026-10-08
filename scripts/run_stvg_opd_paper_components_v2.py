"""Independent later-stage online streams, with exact whole-stream aliases.

All new variants run from source; aliases point to an identical previously sealed
complete stream and retain its original cost. They never splice a Full state into
query-only, LN-only, alpha-zero or Direct. Qualification never uses aliases.
"""
import collections
import gc
import os
import sys
import time
import traceback
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.stvg_opd_paper_later_common_v1 import *


def run(stage_name, qualify=False):
    verify_later(); bridge()
    import torch
    from scripts.run_decota_paper_main_v1 import gpu, model_for, read_row
    from methods.decota_final_simplified_v1.observations import SpatialExpert
    from methods.decota_final_simplified_v1.config import EXPERT_SNAPSHOT
    from methods.decota_final_simplified_v1.tensors import state_hash
    from vg_tta.spatial_online_state_v1 import arrival
    from vg_tta.decota_spatial_opd_tunable_v1 import mean_coordinates, action_boxes
    from vg_tta.stvg_opd_paper_ablations_v1 import fit_variant
    from vg_tta.stvg_opd_paper_component_audit_v1 import audit
    from scripts.stvg_opd_paper_inputs_v1 import capture
    stage = stage_definition(stage_name)
    phase = stage_name.split('_')[0]
    ds, arms = stage['dataset'], stage['arms']
    dest = BASE / ('component_qualification' if qualify else 'stages') / stage_name
    if (dest / 'PREDICTION_BARRIER.json').exists():
        return
    assert read(BASE / 'P1_STAGE_AUTHORIZATION.json')['status'] == 'completed_before_later_stages'
    assert read(BASE / f'{phase}_STAGE_AUTHORIZATION.json')['status'] == 'actual_previous_stage_root_closed'
    if not qualify:
        assert read(BASE / f'{phase}_QUALIFICATION.json')['status'] == 'pass'
    # All declared conditions are qualified. A clean-only smoke is insufficient
    # evidence for the physical corruption decoder/admission path.
    orders = {o: seq[:2] if qualify else seq for o, seq in stage['orders'].items()
              if not qualify or o == next(iter(stage['orders']))}
    expected = sum(map(len, orders.values())) * len(stage['conditions']) * len(arms)
    aliases = {} if qualify else {a: build_alias(stage_name, stage, a) for a in arms}
    aliases = {a: v for a, v in aliases.items() if v is not None}
    new_arms = [a for a in arms if a not in aliases]
    files, inputs, logical, qualifications = {}, {}, {}, []
    count = new_count = 0
    for arm, alias in aliases.items():
        for key, record in alias['records'].items():
            logical[key] = record
            files[record['path']] = record['sha256']
            inputs[record['input']['path']] = record['input']['sha256']
            count += 1
    model = expert = lease = None
    start = time.time()
    try:
        if new_arms:
            lease = gpu(); model = model_for(stage['source'])
            modelhash = state_hash(model.state_dict())
            expert = SpatialExpert(ROOT / EXPERT_SNAPSHOT)
            experthash = state_hash(expert.model.state_dict())
            cache = collections.OrderedDict()
            for condition in stage['conditions']:
                for order, seq in orders.items():
                    previous = {a: None for a in new_arms}
                    prevhash = {a: None for a in new_arms}
                    for at, q in enumerate(seq):
                        budget()
                        row = read_row(ds, q)
                        paths = {a: dest / condition / order / a / f'{at:05}.pt' for a in new_arms}
                        needs_capture = any(not f.exists() for f in paths.values())
                        base = native = ex = None
                        if needs_capture:
                            torch.cuda.synchronize(); tick = time.perf_counter()
                            base, native, ex, inputrc = capture(model, expert, stage, row, condition, cache)
                            torch.cuda.synchronize(); capture_seconds = time.perf_counter() - tick
                            assert torch.equal(base.values()['boxes'], base.zero['boxes'])
                            roundtrip = float((action_boxes(mean_coordinates(base.zero['boxes'])) - base.zero['boxes']).abs().max())
                            assert roundtrip < 2e-7
                            captured = load(BASE / inputrc['path'])
                            for alias_arm in aliases:
                                key = record_key(condition, order, alias_arm, at)
                                input_equivalent(captured, load(BASE / logical[key]['input']['path']))
                            del captured
                        for arm in new_arms:
                            cfg, f = stage['variant_configs'][arm], paths[arm]
                            key = record_key(condition, order, arm, at)
                            if f.exists():
                                rc = read(f.with_suffix('.json')); h = sha(f); z = load(f)
                                assert rc['sha256'] == h and rc['bytes'] == f.stat().st_size and not rc['GT_read']
                                assert z['query_ordinal'] == q and z['config'] == cfg and z['arm'] == arm
                                assert z['previous_payload_sha256'] == prevhash[arm] and not z['GT_read']
                            else:
                                initial = arrival(base.initial, previous[arm], 'O-split')
                                assert torch.count_nonzero(initial['spatial.query_residual']) == 0
                                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                                tick = time.perf_counter()
                                result = fit_variant(base, initial, ex, row['frame_ids'], row['key'], arm, cfg)
                                torch.cuda.synchronize(); seconds = time.perf_counter() - tick
                                active = {'query_only': 256, 'LN_only': 1536}.get(arm, 1792)
                                assert result['selected_step'] == cfg['steps'] and result['active_parameters'] == active
                                tick = time.perf_counter(); mathcheck = audit(result, ex)
                                cpu_seconds = time.perf_counter() - tick
                                nextstate = committed(initial, result['state'], cfg['writeback'])
                                if arm == 'query_only' or arm == 'joint_alpha0':
                                    assert all(torch.equal(nextstate[n], base.initial[n].cpu())
                                               for n in nextstate if n != 'spatial.query_residual')
                                z = dict(dataset=ds, source=stage['source'], stage=stage_name, arm=arm,
                                    condition=condition, order=order, arrival=at, query_ordinal=q, parent=q,
                                    fit=result, config=cfg, committed=nextstate,
                                    previous_payload_sha256=prevhash[arm], input=inputrc,
                                    interval=native['physical_interval'], math_audit=mathcheck,
                                    query_reset=True, Adam_reset=True, Native_WHEN_fixed=True,
                                    chart_roundtrip_max_error=roundtrip, GT_read=False,
                                    component_runtime_sha256=sha(BASE / 'COMPONENT_RUNTIME_LOCK_revision001.json'),
                                    runtime_lock_sha256=sha(BASE / 'RUNTIME_LOCK.json'),
                                    compute=dict(shared_capture_seconds=capture_seconds, fit_GPU_seconds=seconds,
                                        CPU_math_seconds=cpu_seconds, new_DINO_calls=inputrc['new_DINO_calls'],
                                        DINO_observation_budget=stage['observation_budget'], input_capture_components=inputrc,
                                        CUDA_peak_allocated=torch.cuda.max_memory_allocated(),
                                        CUDA_peak_reserved=torch.cuda.max_memory_reserved(), backward_steps=result['gradient_calls']))
                                save(f, z); h = sha(f)
                                write(f.with_suffix('.json'), dict(sha256=h, bytes=f.stat().st_size, GT_read=False,
                                    runtime_lock_sha256=sha(BASE / 'RUNTIME_LOCK.json'), time=time.time()))
                            previous[arm] = z['committed']; prevhash[arm] = h
                            inputs[z['input']['path']] = z['input']['sha256']; files[str(f.relative_to(BASE))] = h
                            logical[key] = dict(path=str(f.relative_to(BASE)), sha256=h, query_ordinal=q,
                                reused_complete_identical_stream=False, input=z['input'])
                            count += 1; new_count += 1
                            if qualify:
                                qualifications.append(dict(dataset=ds, condition=condition, arm=arm,
                                    query_ordinal=q, active_parameters=z['fit']['active_parameters'],
                                    actual_backward_rounds=z['fit']['gradient_calls'], math_audit=z['math_audit'],
                                    no_information=z['fit']['empty'], last_step=z['fit']['selected_step'],
                                    compute=z['compute']))
                            del z
                        if base is not None:
                            base.restore(base.initial)
                        del base, native, ex
                        gc.collect(); torch.cuda.empty_cache()
                        status(dest / 'STATUS.json', dict(status='running', pid=os.getpid(), stage=stage_name,
                            qualification=qualify, done=count, total=expected, new_fit_arrivals=new_count,
                            reused_arrivals=sum(len(x['records']) for x in aliases.values()),
                            condition=condition, order=order, GT_read=False, time=time.time()))
                        print('OPD_COMPONENT_PROGRESS', stage_name, condition, order, count, expected,
                              round(time.time() - start, 1), flush=True)
            assert state_hash(model.state_dict()) == modelhash
            assert state_hash(expert.model.state_dict()) == experthash
        assert count == expected and len(logical) == expected
        write(dest / 'PREDICTION_BARRIER.json', dict(status='sealed', stage=stage_name, qualification=qualify,
            adapted_arrivals=count, actual_fit_arrivals=new_count, reused_arrivals=count-new_count,
            Frozen_logical_arrivals=sum(map(len, orders.values())) * len(stage['conditions']),
            logical_records=logical, files=files, inputs=inputs, orders=orders,
            variant_configs=stage['variant_configs'], alias_barriers={a: {k: v for k, v in x.items() if k != 'records'}
                for a, x in aliases.items()}, GT_read=False, source_checkpoint_unchanged=True,
            runtime_lock_sha256=sha(BASE / 'RUNTIME_LOCK.json'),
            component_runtime_sha256=sha(BASE / 'COMPONENT_RUNTIME_LOCK_revision001.json'), time=time.time()))
        if qualify:
            assert len(qualifications) == expected
            write(dest / 'QUALIFICATION.json', dict(status='pass', actual_fits=expected, records=qualifications,
                GT_read=False, parameter_math_independent=True, independent_decoder_Jacobian=False, time=time.time()))
        status(dest / 'STATUS.json', dict(status='sealed_pending_phase_global_barrier',
            done=count, total=expected, new_fit_arrivals=new_count, GT_read=False, time=time.time()))
    finally:
        if lease is not None:
            lease.close()


if __name__ == '__main__':
    stage_name = sys.argv[1]
    try:
        run(stage_name, len(sys.argv) > 2 and sys.argv[2] == 'qualification')
    except BaseException:
        directory = BASE / 'component_failures' / str(time.time_ns())
        directory.mkdir(parents=True, exist_ok=True)
        (directory / 'traceback.txt').write_text(traceback.format_exc())
        status(BASE / 'COMPONENT_STAGE.json', dict(status='failed_preserved', stage=stage_name,
            evidence=str(directory), pid=os.getpid(), GT_read=False, time=time.time()))
        raise
