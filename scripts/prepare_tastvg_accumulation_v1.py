"""Lock every historical state chain and every future target without GT."""
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.tastvg_accumulation_common_v1 import *


def run():
    import torch
    from methods.decota_final_simplified_v1.tensors import state_hash
    from vg_tta.tastvg_saved_write_accumulation_v1 import delta, at_origin, geometry
    assert not (BASE / 'COHORT.json').exists(), 'New isolated namespace required'
    assert not BASE.exists() or {p.name for p in BASE.iterdir()} <= {'recovery'}, 'Only a preserved pre-inference failure may exist'
    torch.set_num_threads(2)
    assert read(OLD / 'FINAL_COMPLETION.json')['status'] == 'completed'
    plan = read(OLD / 'hc2/PLAN.json')
    capture = read(POOL / 'hc2/CAPTURE_BARRIER.json')
    parents = sorted(plan['splits']['search']['orders']['order1'])
    assert len(parents) == 32
    source_ids = {source: i for i, source in enumerate(sorted({plan['rows'][j]['source'] for j in parents}))}
    inputs = {}
    def bind(path):
        inputs[str(path.relative_to(ROOT))] = sha(path)
    for path in [OLD / 'FINAL_COMPLETION.json', OLD / 'hc2/PLAN.json',
                 POOL / 'hc2/CAPTURE_BARRIER.json', ROOT / 'methods/CURRENT_METHOD.json']:
        bind(path)
    targets, streams = [], []
    origin_sha = None
    chains = 0
    for arm in ['A', 'R']:
        for condition in plan['conditions']:
            for order, sequence in plan['splits']['search']['orders'].items():
                assert sorted(sequence) == parents
                writes, write_records = [], []
                previous, origin = None, None
                for arrival, parent in enumerate(sequence):
                    path = OLD / 'hc2' / arm / 'online' / condition / order / f'{arrival:05}.pt'
                    bind(path); bind(path.with_suffix('.json'))
                    payload = load(path)
                    assert read(path.with_suffix('.json'))['sha256'] == sha(path)
                    assert payload['parent'] == parent and payload['arrival'] == arrival
                    assert payload['order'] == order and payload['condition'] == condition and payload['arm'] == arm
                    assert payload['GT_read'] is False and payload['expert_scheduled'] == (arrival % 4 == 0)
                    pre, post = payload['pre_state'], payload['post_state']
                    assert state_hash(pre) == payload['pre_sha'] and state_hash(post) == payload['post_sha']
                    if origin is None:
                        origin = pre
                        assert sum(v.numel() for v in origin.values()) == 1792
                        if origin_sha is None:
                            origin_sha = state_hash(origin)
                        assert state_hash(origin) == origin_sha
                    else:
                        assert previous == payload['pre_sha']
                    reconstructed = at_origin(origin, writes)
                    assert state_hash(reconstructed) == payload['pre_sha']
                    assert all(torch.equal(reconstructed[k], pre[k]) for k in origin)
                    chains += 1
                    if arrival % 4 == 0:
                        saved_delta = delta(pre, post)
                        assert bool(any(torch.count_nonzero(v) for v in saved_delta.values())) == bool(payload['updated'])
                        writes.append(saved_delta)
                        write_records.append(dict(arrival=arrival, parent=parent,
                                                  source_id=source_ids[plan['rows'][parent]['source']],
                                                  updated=payload['updated'], pre_sha=payload['pre_sha'], post_sha=payload['post_sha']))
                    else:
                        assert not payload['updated'] and payload['pre_sha'] == payload['post_sha']
                        assert payload['final_indices'] == payload['slow']['indices']
                        last = at_origin(origin, [writes[-1]])
                        receipt_path = POOL / 'hc2/capture' / condition / f'{parent:05}.json'
                        capture_receipt = read(receipt_path)
                        assert sha(receipt_path) == capture['files'][str(receipt_path.relative_to(POOL / 'hc2'))]
                        assert capture_receipt['pixel_sha256'] == payload['pixel_sha256']
                        cache_path = POOL / 'hc2' / capture_receipt['cache']
                        assert sha(cache_path) == capture_receipt['sha256']
                        bind(receipt_path); bind(cache_path)
                        targets.append(dict(arm=arm, condition=condition, order=order,
                                            arrival=arrival, parent=parent,
                                            source_id=source_ids[plan['rows'][parent]['source']],
                                            prior_writes=len(writes), latest_write_arrival=write_records[-1]['arrival'],
                                            latest_write_updated=write_records[-1]['updated'],
                                            lag=arrival-write_records[-1]['arrival'],
                                            all_sha=state_hash(reconstructed), last_sha=state_hash(last), source_sha=origin_sha,
                                            old_payload=str(path.relative_to(ROOT)), old_payload_sha256=sha(path),
                                            capture_receipt=str(receipt_path.relative_to(ROOT)),
                                            capture_sha256=capture_receipt['sha256'], pixel_sha256=payload['pixel_sha256']))
                    previous = payload['post_sha']
                assert len(writes) == 8
                streams.append(dict(arm=arm, condition=condition, order=order,
                                    writes=write_records, **geometry(writes)))
    assert len(targets) == 576 and chains == 768 and len(streams) == 24
    cohort = dict(dataset='hc2', historical_sources=32, target_sources=30,
                  target_cells_per_arm=288, corruption_cells_per_arm=240,
                  clean_cells_per_arm=48, conditions=plan['conditions'],
                  orders=plan['splits']['search']['orders'], params=plan['params'],
                  historical_exposure=True, GT_read=False, rows=targets,
                  numerical_definition='FP64 difference/sum then one cast to saved FP32',
                  current_method_sha256=sha(ROOT / 'methods/CURRENT_METHOD.json'))
    write(BASE / 'COHORT.json', cohort)
    write(BASE / 'WRITE_GEOMETRY.json', streams)
    pins = ['protocols/tastvg_accumulation_audit_v1.md',
            'scripts/tastvg_accumulation_common_v1.py',
            'scripts/prepare_tastvg_accumulation_v1.py',
            'scripts/run_tastvg_accumulation_v1.py',
            'vg_tta/tastvg_saved_write_accumulation_v1.py',
            'vg_tta/tastvg_native_spatial_rollout_s05_v1.py',
            'vg_tta/tastvg_causal_round2_v1.py',
            'vg_tta/tastvg_evidence_capture_v1.py',
            'scripts/decota_matrix_common_v1.py',
            'scripts/run_tastvg_evidence_vulnerability_v1.py',
            'scripts/run_tastvg_evidence_vulnerability_v2.py',
            'scripts/run_tastvg_paper48_p5_online_v1.py',
            'scripts/run_spatial_regression_alignment_v1.py',
            'methods/decota_final_simplified_v1/backbone.py',
            'methods/decota_final_simplified_v1/tensors.py']
    bind(BASE / 'COHORT.json'); bind(BASE / 'WRITE_GEOMETRY.json')
    write(BASE / 'RUNTIME_LOCK.json', dict(pins={path: sha(ROOT / path) for path in pins},
                                         inputs=inputs, GT_read=False, time=time.time(),
                                         checkpoint_state_sha256=capture['checkpoint_state_sha256']))
    write(BASE / 'PREPARATION_AUDIT.json', dict(status='pass', state_chains=chains,
                                             exact_prefix_reconstructions=chains,
                                             targets=len(targets), streams=len(streams),
                                             origin_sha=origin_sha, GT_read=False, time=time.time()))
    status(BASE / 'STATUS.json', dict(status='ready', done=0, total=576, GT_read=False, time=time.time()))
    archive('CPU名单与768条状态链及576个target已锁定，准备正式缓存后缀重放')
    print('READY', len(targets), chains, flush=True)


if __name__ == '__main__':
    run()
