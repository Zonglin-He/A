"""Two original queries, four bounded native passes; never read labels."""
import fcntl
import gc
import json
import time
from pathlib import Path
import numpy as np
import torch
from stvg.runtime import ROOT, setup, sha
from stvg.tube_runner import write, digest
from stvg.tube_decode import decode
from stvg.tensors import state_hash
from stvg.tubedetr import load_source
from tools.reuse_cache.catalog import Catalog
from tools.reuse_cache.store import ArrayStore
from .adapter import capture, finish

PUBLIC = ROOT / 'results/simplepath_tubedetr'
OUT = ROOT / 'outputs/simplepath_tubedetr'


def main():
    setup()  # Annotation guard precedes file/model/input validation.
    plan_path = PUBLIC / 'qualification_plan.json'
    p = json.loads(plan_path.read_text())
    for name, expected in {**p['code_sha256'], **p['official_code_sha256']}.items():
        assert sha(ROOT / name) == expected, name
    if (PUBLIC / 'qualification.json').exists() or (OUT / 'failure.json').exists():
        raise ValueError('Qualification already terminated; no automatic retry')
    OUT.mkdir(parents=True, exist_ok=True)
    catalog = Catalog(ROOT / 'runtime/reuse_cache/catalog.sqlite')
    cache = ArrayStore(ROOT / 'runtime/reuse_cache/arrays')
    checks = []
    with (ROOT / 'runtime/gpu.lock').open('a') as lock:
        print('Waiting for shared gpu.lock', flush=True)
        fcntl.flock(lock, fcntl.LOCK_EX)
        for ds in p['dataset_order']:
            d = p['datasets'][ds]
            assert sha(ROOT / d['manifest']) == d['manifest_sha256']
            row = json.loads((ROOT / d['manifest']).read_text().splitlines()[0])
            frames, ids = decode(row['metadata'], ds)
            model = load_source(d['configuration'])
            before = state_hash(model.state_dict())
            tick = time.perf_counter()
            cold = capture(model, frames, ids, row['metadata']['caption'])
            prefix = {k: np.asarray(v) for k, v in cold['visual_prefix'].items()}
            context = dict(backbone='tubedetr', checkpoint_sha256=model.load_audit['checkpoint_sha256'],
                           implementation_sha256=digest(p['code_sha256']),
                           input_sha256=digest(row), sampling_sha256=digest(ids),
                           representation='native_input_proj_slow_fast_masks_positions',
                           scope='query', caption_sha256=digest(row['metadata']['caption']), seed=20260920)
            stored = cache.put(context, prefix, required=True)
            warm = capture(model, frames, ids, row['metadata']['caption'], prefix=cache.get(context))
            for key in ['Source', 'Native_interval', 'features', 'valid', 'native_logits6', 'native_prior']:
                assert np.array_equal(np.asarray(cold[key]), np.asarray(warm[key])), ('cold/cache', ds, key)
            teacher, teacher_trace = catalog.teacher('tubedetr', frames[cold['positions8']],
                                        row['metadata']['caption'], p['teacher_fingerprint'])
            full_cold, full_warm = finish(cold, teacher, d['configuration']), finish(warm, teacher, d['configuration'])
            for key in ['Source', 'Spatial', 'Temporal', 'Full', 'Selected_interval', 'posterior', 'pooled_pretext']:
                assert np.array_equal(full_cold[key], full_warm[key]), ('coupled cold/cache', ds, key)
            old_root = ROOT / 'outputs/tubedetr_boxfixed_development' / ds / 'full'
            seal = json.loads((old_root / 'sealed.json').read_text())
            e = seal['files'][row['id']]
            assert sha(old_root / e['path']) == e['sha256']
            with np.load(old_root / e['path'], allow_pickle=False) as old:
                for key in ['Source', 'Native_interval', 'frame_ids', 'positions8']:
                    assert np.array_equal(old[key], np.asarray(cold[key])), ('old Source', ds, key)
            assert before == state_hash(model.state_dict())
            assert all(not x.requires_grad and x.grad is None for x in model.parameters())
            target = OUT / 'qualification' / ds
            target.mkdir(parents=True, exist_ok=True)
            payload = target / 'output.npz'
            np.savez_compressed(payload, **{k: v for k, v in full_cold.items() if isinstance(v, np.ndarray)})
            checks.append(dict(dataset=ds, previous_query_used=True, native_passes=2,
                cold_cache_bitexact=True, Source_native_interval_positions_equal_old_full=True,
                coupled_Full_bitexact=True, all_model_state_unchanged=True,
                actual_pretext_shape=list(prefix['fast_grid'].shape),
                spatial_grid=list(cold['features'].shape),
                prefix_inputs_have_no_caption_or_text=True,
                prefix_cache=context, prefix_key=stored['key'],
                new_teacher_calls=0, exact_teacher_cache=teacher_trace['cache_hit'],
                output_sha256=sha(payload), finite=True, GT_read=False,
                seconds=time.perf_counter() - tick))
            print(ds, 'real cold/cache qualification pass', flush=True)
            del model, frames, cold, warm, teacher, full_cold, full_warm
            gc.collect()
            torch.cuda.empty_cache()
    write(PUBLIC / 'qualification.json', dict(status='pass', plan_sha256=sha(plan_path),
          checks=checks, native_passes=4, new_teacher_calls=0, GT_reads=0,
          task_performance_evaluated=False, not_new_full_results=True))
    write(OUT / 'status.json', dict(status='qualification_pass_pending_fixed128_implementation',
                                   plan_sha256=sha(plan_path)))
    catalog.db.close()


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        OUT.mkdir(parents=True, exist_ok=True)
        write(OUT / 'failure.json', dict(status='stopped', stage='qualification',
              error=type(error).__name__ + ': ' + str(error), automatic_retry=False))
        raise
