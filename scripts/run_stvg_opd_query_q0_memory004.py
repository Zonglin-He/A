"""Qualified frozen-storage repair of the fixed correct-Native Q0 science."""
import collections
import gc
import inspect
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.stvg_opd_query_q0_native_scope003 import BASE, context, bind_geometry, bind_runner
context()
import scripts.run_stvg_opd_query_q0_v1 as original
from scripts.decota_matrix_common_v1 import read, write, status, sha, load, save

REC = BASE / 'frozen_memory_revision004'
ROOT_SOURCE = inspect.getsource(original.root)


def setup():
    bind_geometry()
    bind_runner(original)
    previous = original.verify
    def verify():
        value = previous()
        for path, digest in read(REC / 'REVISION_RUNTIME.json')['pins'].items():
            assert sha(ROOT / path) == digest, path
        return value
    original.verify = verify
    import scripts.run_spatial_ssl_gpu_v1 as capture_module
    from vg_tta.stvg_opd_frozen_mask_cache012 import frozen_forward
    capture_module.frozen_forward = frozen_forward
    source = inspect.getsource(original.gpu_lease)
    source = source.replace("BASE/'PRIORITY_PAUSE_RECEIPT.json'", "REC/'PRIORITY_PAUSE_RECEIPT.json'")
    namespace = dict(original.__dict__, REC=REC)
    exec(compile(source, __file__, 'exec'), namespace)
    original.gpu_lease = namespace['gpu_lease']
    return verify()


def qualification():
    setup()
    import torch
    import numpy as np
    from scripts.stvg_opd_design_ablation_common_v1 import activate_old
    activate_old()
    from scripts.run_decota_paper_main_v1 import model_for
    from methods.decota_final_simplified_v1.tensors import state_hash
    from scripts.run_stvg_opd_p1_softmax_recovery_002 import equal
    from vg_tta.stvg_opd_query_q0_v1 import episode
    design = read(BASE / 'DESIGN_LOCK.json')
    indices = read(REC / 'DESIGN_LOCK.json')['qualification_indices']
    lease = original.gpu_lease()
    files, count, model, dataset = {}, 0, None, None
    try:
        for index in indices:
            record = design['records'][index]
            if dataset != record['dataset']:
                if model is not None:
                    assert state_hash(model.state_dict()) == model_hash
                    del model
                    gc.collect(); torch.cuda.empty_cache()
                torch.manual_seed(20260920); np.random.seed(20260920)
                model = model_for(record['source'])
                model_hash = state_hash(model.state_dict())
                dataset = record['dataset']
            assert sha(ROOT / record['original_Full']['path']) == record['original_Full']['sha256']
            old = load(ROOT / record['original_Full']['path'])
            for repeat in (0, 1):
                base, row, expert = original.capture(model, record, old, collections.OrderedDict())
                result = episode(base, old, expert, dataset, row['key'], design['pooled_output_RMS_budget'])
                compared = dict(tensors=0, tensor_coordinates=0, scalar_values=0)
                if record['qualify']:
                    prior = load(BASE / 'qualification' / f'{index:05}.pt')
                    equal(result, prior['episode'], compared)
                if repeat:
                    first = load(REC / 'qualification' / f'UNACCEPTED_{index:05}.pt')
                    equal(result, first['episode'], compared)
                    path = REC / 'qualification' / f'REPEAT_{index:05}.pt'
                    save(path, result)
                    first['repeat'] = dict(sha256=sha(path), complete_bitwise=compared)
                    first['complete_qualification_episodes'] = 2
                    save(REC / 'qualification' / f'{index:05}.pt', first)
                    write((REC / 'qualification' / f'{index:05}.pt').with_suffix('.json'),
                          dict(status='pass', sha256=sha(REC / 'qualification' / f'{index:05}.pt'),
                               GT_read=False, time=time.time()))
                    files[str((REC / 'qualification' / f'{index:05}.pt').relative_to(ROOT))] = sha(REC / 'qualification' / f'{index:05}.pt')
                else:
                    path = REC / 'qualification' / f'UNACCEPTED_{index:05}.pt'
                    save(path, dict(**record, episode=result, runtime_sha256=sha(BASE / 'RUNTIME.json'),
                         execution_revision_runtime_sha256=sha(REC / 'REVISION_RUNTIME.json'),
                         source_model_sha256=model_hash, interval=old['interval'],
                         GT_read=False, complete_comparison=compared))
                count += 1
                status(REC / 'STATUS.json', dict(status='running_actual_complete_Q0_memory_qualification',
                       dataset=dataset, index=index, repeat=repeat, done=count, total=10,
                       pid=os.getpid(), GT_read=False, time=time.time()))
                print('Q0_MEMORY004_QUALIFICATION', count, 10, index, repeat, flush=True)
                base.restore(base.initial)
                del base, row, expert, result
                gc.collect(); torch.cuda.empty_cache()
            assert state_hash(model.state_dict()) == model_hash
            del old
        assert count == 10
        write(REC / 'QUALIFICATION_BARRIER.json', dict(status='pass', files=files, cells=5,
              actual_complete_GPU_episodes=10, formal_predictions_accepted=0,
              GT_read=False, new_DINO_calls=0, optimizer_steps=0, time=time.time()))
    finally:
        if model is not None:
            del model
        gc.collect(); torch.cuda.empty_cache(); lease.close()


def qualification_root():
    setup()
    source = ROOT_SOURCE
    old = "before=ep['before'].numpy().astype(float);boxes=c['boxes'].numpy().astype(float)"
    new = "before=ep['before'].numpy().astype(float)[ep['native_positions']];boxes=c['boxes'].numpy().astype(float)[ep['native_positions']]"
    assert source.count(old) == 1
    source = source.replace(old, new)
    source = source.replace("t['boxes'].numpy().astype(float)-before", "t['boxes'].numpy().astype(float)[ep['native_positions']]-before")
    source = source.replace('(4 if qualification else 64)', '(5 if qualification else 64)')
    namespace = dict(original.__dict__, BASE=REC)
    exec(compile(source, __file__, 'exec'), namespace)
    namespace['root'](True)
    capture = read(REC / 'CAPTURE_RECEIPT.json')
    for record in capture['accepted_prefix']:
        path = ROOT / record['path']
        assert sha(path) == record['sha256'] and sha(path.with_suffix('.json')) == record['receipt_sha256']
    from scripts.run_stvg_opd_p1_softmax_recovery_002 import equal
    counts = dict(tensors=0, tensor_coordinates=0, scalar_values=0)
    for index in read(REC / 'DESIGN_LOCK.json')['qualification_indices']:
        if index in (0, 1, 32, 33):
            current = load(REC / 'qualification' / f'{index:05}.pt')
            old = load(BASE / 'qualification' / f'{index:05}.pt')
            equal(current['episode'], old['episode'], counts)
    write(REC / 'ROOT_MEMORY_QUALIFICATION_READBACK.json', dict(status='pass',
          scope='five fixed correct-Native queries, ten complete independent GPU captures/episodes; four original complete episodes exact and full repeat checks',
          actual_complete_GPU_episodes=10, original_qualified_episode_pairs=4,
          complete_repeat_pairs=5, complete_comparison=counts,
          original_formal_prefix_rehashed=len(capture['accepted_prefix']),
          candidate_math_root_sha256=sha(REC / 'ROOT_QUALIFICATION.json'),
          failed_complete_episode_serialization=0, dead_failed_memory_bitwise_claim=False,
          formal_predictions_accepted=0, new_DINO_calls=0, optimizer_steps=0,
          GT_read=False, CPU_only=True, future_OOM_safety_claim=False, time=time.time()))


def formal():
    setup()
    assert read(REC / 'ROOT_MEMORY_QUALIFICATION_READBACK.json')['status'] == 'pass'
    source = inspect.getsource(original.run)
    needle = '        for r in records:\n'
    assert source.count(needle) == 1
    source = source.replace(needle, '''        for r in records:
            existing=dest/f"{r['index']:05}.pt"
            if existing.with_suffix('.json').exists():
                receipt=read(existing.with_suffix('.json'))
                assert sha(existing)==receipt['sha256'] and not receipt['GT_read']
                prior=load(existing)
                assert prior['index']==r['index'] and prior['runtime_sha256']==sha(BASE/'RUNTIME.json')
                assert sha(ROOT/r['input']['path'])==r['input']['sha256']
                assert sha(ROOT/r['original_Full']['path'])==r['original_Full']['sha256']
                files[str(existing.relative_to(ROOT))]=sha(existing)
                status(BASE/'GPU_STATUS.json',dict(status='rechecking_preserved_Q0_prefix_then_formal',done=len(files),total=len(records),pid=os.getpid(),GT_read=False,time=time.time()))
                continue
''')
    needle = "            p=dest/f\"{r['index']:05}.pt\";save(dest/f\"UNACCEPTED_{r['index']:05}.pt\",evidence)\n"
    assert source.count(needle) == 1
    source = source.replace(needle, '''            evidence['execution_revision_runtime_sha256']=sha(REC/'REVISION_RUNTIME.json')
            if r['index']==45:
                repaired=load(REC/'qualification'/f"{r['index']:05}.pt")
                repaired_counts=dict(tensors=0,tensor_coordinates=0,scalar_values=0)
                equal(z,repaired['episode'],repaired_counts)
                evidence['memory_qualified_formal']=repaired_counts
''' + needle)
    namespace = dict(original.__dict__, REC=REC)
    exec(compile(source, __file__, 'exec'), namespace)
    namespace['run'](False)


if __name__ == '__main__':
    job = sys.argv[1]
    try:
        {'qualification': qualification, 'qualification_root': qualification_root,
         'formal': formal}[job]()
    except BaseException:
        path = REC / 'failures' / str(time.time_ns())
        path.mkdir(parents=True, exist_ok=True)
        (path / 'traceback.txt').write_text(traceback.format_exc())
        status(BASE / 'STATUS.json', dict(status='failed_preserved_pending_actual_root',
               stage=job, revision='memory004', evidence=str(path.relative_to(ROOT)),
               time=time.time()))
        raise
