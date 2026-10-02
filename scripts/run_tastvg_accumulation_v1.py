"""Exact cached suffix replays at Source/All/Last; no GT or optimization."""
import os
import sys
import time
import gc
import traceback
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
from scripts.tastvg_accumulation_common_v1 import *


def run():
    import torch
    import numpy as np
    from scripts.run_tastvg_evidence_vulnerability_v2 import install_clean_loader
    from scripts.run_tastvg_evidence_vulnerability_v1 import device_tree
    from scripts.run_final_simplification_v1 import lease
    from scripts.run_tastvg_paper48_p5_online_v1 import compact_prediction
    from vg_tta.tastvg_native_spatial_rollout_s05_v1 import predict, central_state
    from vg_tta.tastvg_saved_write_accumulation_v1 import delta, at_origin
    from methods.decota_final_simplified_v1.tensors import state_hash
    lock = verify(); cohort = read(BASE / 'COHORT.json')
    install_clean_loader(); sys.addaudithook(guard)
    handle = lease()
    torch.set_num_threads(4); torch.manual_seed(20260929); np.random.seed(20260929)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    from scripts.run_spatial_regression_alignment_v1 import model_load
    tick = time.monotonic()
    model = model_load('vidstg_test').eval().requires_grad_(False)
    checkpoint_sha = state_hash(model.state_dict())
    assert checkpoint_sha == lock['checkpoint_state_sha256']
    native_origin = device_tree(central_state(model), 'cpu')
    assert state_hash(native_origin) == read(BASE / 'PREPARATION_AUDIT.json')['origin_sha']
    done = replays = parity_all = parity_source = 0
    grouped = {}
    for target in cohort['rows']:
        grouped.setdefault((target['arm'], target['condition'], target['order']), []).append(target)
    for (arm, condition, order), targets in grouped.items():
        history = OLD / 'hc2' / arm / 'online' / condition / order
        first = load(history / '00000.pt')
        origin = first['pre_state']
        writes = {arrival: delta(load(history / f'{arrival:05}.pt')['pre_state'],
                                 load(history / f'{arrival:05}.pt')['post_state'])
                  for arrival in range(0, 32, 4)}
        for target in targets:
            file = BASE / 'predictions' / arm / condition / order / f'{target["arrival"]:05}.pt'
            if file.with_suffix('.json').exists():
                receipt(file); done += 1; continue
            budget()
            old = load(ROOT / target['old_payload'])
            all_state = at_origin(origin, [writes[j] for j in sorted(writes) if j < target['arrival']])
            last_state = at_origin(origin, [writes[target['latest_write_arrival']]])
            assert state_hash(all_state) == target['all_sha'] == old['pre_sha']
            assert state_hash(last_state) == target['last_sha']
            assert all(torch.equal(all_state[key], old['pre_state'][key]) for key in origin)
            cr = read(ROOT / target['capture_receipt'])
            cache = POOL / 'hc2' / cr['cache']
            assert sha(cache) == target['capture_sha256'] == cr['sha256']
            data = device_tree(load(cache), 'cuda')
            predictions, reused = {}, {}
            for name, state in [('source', origin), ('all', all_state), ('last', last_state)]:
                fingerprint = state_hash(state)
                if fingerprint not in reused:
                    _, _, output = predict(model, data, device_tree(state, 'cuda'))
                    reused[fingerprint] = compact_prediction(output)
                    replays += 1
                predictions[name] = reused[fingerprint]
            for name, saved in [('all', old['slow']), ('source', data['prediction'])]:
                assert torch.equal(predictions[name]['boxes'].cpu(), saved['boxes'].cpu()), (arm, condition, order, target['arrival'], name, 'boxes')
                assert predictions[name]['indices'] == saved['indices'], (arm, condition, order, target['arrival'], name, 'interval')
            parity_all += 1; parity_source += 1
            assert state_hash(model.state_dict()) == checkpoint_sha
            assert not any(parameter.grad is not None for parameter in model.parameters())
            commit(file, dict(target=target, predictions=predictions,
                              source_interval=predictions['source']['indices'],
                              suffix_replays=len(reused), aliases={name: [other for other in predictions if
                                      {'source':target['source_sha'], 'all':target['all_sha'], 'last':target['last_sha']}[other] ==
                                      {'source':target['source_sha'], 'all':target['all_sha'], 'last':target['last_sha']}[name]]
                                      for name in predictions},
                              all_state=all_state, last_state=last_state,
                              GT_read=False, parameter_updates=0, new_experts=0,
                              new_backbone_forwards=0, checkpoint_restored=True))
            done += 1
            status(BASE / 'STATUS.json', dict(status='running_cached_suffix', done=done,
                                            total=576, worker_pid=os.getpid(),
                                            suffix_replays=replays, GT_read=False, time=time.time()))
            if done % 12 == 0:
                print('ACCUMULATION', done, 576, arm, condition, order, flush=True)
            del data, predictions, reused
            gc.collect(); torch.cuda.empty_cache()
    assert done == 576
    verify()
    files = {str(path.relative_to(BASE)): sha(path) for path in (BASE / 'predictions').rglob('*.json')}
    assert len(files) == 576
    write(BASE / 'GLOBAL_PREDICTION_BARRIER.json', dict(status='sealed', targets=576,
                                                    source_all_last_predictions=1728,
                                                    files=files, cohort_sha256=sha(BASE / 'COHORT.json'),
                                                    GT_read=False, time=time.time()))
    write(BASE / 'RESOURCES.json', dict(worker_wall_seconds=time.monotonic()-tick,
                                      wall_includes_loading_IO=True, suffix_replays=replays,
                                      bitwise_all_parity=parity_all, bitwise_source_parity=parity_source,
                                      new_backbone_forwards=0, new_experts=0,
                                      parameter_updates=0, backward_calls=0,
                                      peak_vram_bytes=torch.cuda.max_memory_allocated()))
    status(BASE / 'STATUS.json', dict(status='sealed_pending_root_score', done=576,
                                    total=576, GT_read=False, time=time.time()))
    handle.close()


if __name__ == '__main__':
    try:
        run()
    except BaseException as error:
        write(BASE / f'FAILURE_{time.time_ns()}.json', dict(error=repr(error),
             traceback=traceback.format_exc(), time=time.time()))
        status(BASE / 'STATUS.json', dict(status='failed', error=repr(error), time=time.time()))
        raise
