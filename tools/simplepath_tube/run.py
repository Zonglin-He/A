"""Fixed SimplePath/Native2D: both original128, then Vid10303 and HC1902.

Prediction workers are label-guarded. Complete seals precede every independent
CPU score. All old evidence stays immutable; no search or performance gate.
"""
import argparse
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
import numpy as np
import torch
from stvg.runtime import ROOT, sha, setup
from stvg.tube_runner import write, digest
from tools.reuse_cache.store import ArrayStore
from tools.reuse_cache.catalog import Catalog

PUBLIC = ROOT / 'results/simplepath_tubedetr'
OUT = ROOT / 'outputs/simplepath_tubedetr'
PLAN = PUBLIC / 'run_plan.json'
ARMS = ['Source', 'Spatial', 'NativeMBR', 'Temporal', 'Full']


def checked():
    p = json.loads(PLAN.read_text())
    for name, h in p['code_sha256'].items():
        assert sha(ROOT / name) == h, name
    assert sha(PUBLIC / 'qualification.json') == p['qualification_sha256']
    assert json.loads((PUBLIC / 'qualification.json').read_text())['status'] == 'pass'
    return p


def prepare():
    if PLAN.exists():
        raise ValueError('Run plan already frozen; never replace')
    qpath = PUBLIC / 'qualification.json'
    q = json.loads(qpath.read_text()); assert q['status'] == 'pass'
    qp = PUBLIC / 'qualification_plan.json'
    p = json.loads(qp.read_text()); assert sha(qp) == q['plan_sha256']
    for name, h in {**p['code_sha256'], **p['official_code_sha256']}.items():
        assert sha(ROOT / name) == h, name
    p.update(status='frozen_fixed_method_before_new128_scores_and_fulls',
             qualification_plan_sha256=sha(qp), qualification_sha256=sha(qpath),
             both128_before_any_full=True, new_expert_calls=0,
             scoring='independent CPU only after whole-domain complete seal',
             comparison_arms=ARMS, full_order=['vidstg', 'hc2'],
             no_performance_based_retuning_or_query_selection=True,
             cross_query_prefix='same exact RGB/physical grid/checkpoint; structurally pre-text; cold/cache qualified; every native Source compared to old sealed reference',
             cache_priority='all small native outputs and video prefixes required; optional query grids until48GiB; overall64GiB and25GiB free floor',
             new_phase='both fixed128 diagnostics then complete populations, retain all signed outcomes',
             scientific_scope='fixed-method exposed development and complete populations; no unexposed-test claim')
    p['code_sha256'].update({str(f.relative_to(ROOT)): sha(f)
                            for f in sorted((ROOT / 'tools/simplepath_tube').rglob('*.py'))})
    for name in ['tools/llava_st/evidence.py', 'tools/tubedetr_opd_direct_diagnosis.py']:
        p['code_sha256'][name] = sha(ROOT / name)
    p['code_sha256'].update(p['official_code_sha256'])
    write(PLAN, p)
    write(PUBLIC / 'fixed_method_lock.json', dict(status='locked_before_any_new_score',
          plan_sha256=sha(PLAN), fixed=p['fixed'], datasets={ds: {
          'initialization_gamma': p['datasets'][ds]['configuration']['initialization_gamma']}
          for ds in p['dataset_order']}, no_search=True))
    print('run plan frozen', sha(PLAN), flush=True)


def rows_for(p, ds, full):
    d = p['datasets'][ds]
    key = 'full_manifest' if full else 'manifest'
    assert sha(ROOT / d[key]) == d[key + '_sha256']
    rows = [json.loads(line) for line in (ROOT / d[key]).read_text().splitlines()]
    assert len(rows) == (d['full_queries'] if full else 128)
    assert len({r['id'] for r in rows}) == len(rows)
    return rows


def gate_both128(p):
    gate = json.loads((PUBLIC / 'both128_gate.json').read_text())
    assert gate['plan_sha256'] == sha(PLAN)
    for name, h in gate['files'].items():
        assert sha(ROOT / name) == h, name


def numeric_output(value):
    return {k: v for k, v in value.items() if isinstance(v, np.ndarray)}


def gpu(ds, full):
    setup()
    p = checked(); rows = rows_for(p, ds, full)
    if full:
        gate_both128(p)
    d = p['datasets'][ds]; cfg = d['configuration']
    target = OUT / ds / ('full' if full else 'dev128')
    target.mkdir(parents=True, exist_ok=True)
    from stvg.tubedetr import load_source
    from stvg.tube_decode import decode
    from stvg.tensors import state_hash
    from .adapter import capture, finish
    cache = ArrayStore(ROOT / 'runtime/reuse_cache/arrays')
    catalog = Catalog(ROOT / 'runtime/reuse_cache/catalog.sqlite')
    oldroot = ROOT / 'outputs/tubedetr_boxfixed_development' / ds / 'full'
    oldseal = json.loads((oldroot / 'sealed.json').read_text())
    qualification = json.loads((PUBLIC / 'qualification.json').read_text())
    first = next(x for x in qualification['checks'] if x['dataset'] == ds)
    entries = {}; prefix_hits = 0; native_calls = 0
    with (ROOT / 'runtime/gpu.lock').open('a') as lock:
        print('Waiting for shared gpu.lock', flush=True); fcntl.flock(lock, fcntl.LOCK_EX)
        model = load_source(cfg); before = state_hash(model.state_dict())
        for i, row in enumerate(rows):
            q = row['id']; path = target / (q + '.npz'); receipt = path.with_suffix('.json')
            if path.exists() and receipt.exists():
                meta = json.loads(receipt.read_text())
                assert meta['plan_sha256'] == sha(PLAN) and meta['request_sha256'] == digest(row)
                assert sha(path) == meta['payload_sha256']
            elif path.exists() or receipt.exists():
                raise ValueError('Uncommitted prediction retained; stop for audit')
            else:
                tick = time.perf_counter()
                dev = OUT / ds / 'dev128'
                dev_entry = None
                if full:
                    devseal = json.loads((dev / 'sealed.json').read_text())
                    dev_entry = devseal['files'].get(q)
                if dev_entry is not None:
                    assert sha(dev / dev_entry['path']) == dev_entry['sha256']
                    oldmeta = json.loads((dev / dev_entry['receipt_path']).read_text())
                    assert oldmeta['request_sha256'] == digest(row)
                    shutil.copy2(dev / dev_entry['path'], path)
                    meta = dict(oldmeta, reused_fixed128=True, new_native_calls=0, new_teacher_calls=0)
                else:
                    if shutil.disk_usage(ROOT).free < (25 << 30) + (64 << 20):
                        raise RuntimeError('25GiB free-space floor reached; preserve existing evidence')
                    frames, ids = decode(row['metadata'], ds)
                    rgb = hashlib.sha256(frames.tobytes()).hexdigest()
                    ctx = dict(backbone='tubedetr', scope='video', checkpoint_sha256=cfg['source_checkpoint_sha256'],
                         implementation_sha256=digest(json.loads((PUBLIC / 'qualification_plan.json').read_text())['code_sha256']),
                         input_sha256=rgb, sampling_sha256=digest(ids),
                         representation='native_pretext_input_proj_slow_fast_masks_positions_fp32_res224_stride2',
                         query_independence_proof_sha256=p['qualification_sha256'], torch_version=torch.__version__)
                    try:
                        prefix = cache.get(ctx)
                    except FileNotFoundError:
                        prefix = None
                    if prefix is None and digest(row) == first['prefix_cache']['input_sha256']:
                        prefix = cache.get(first['prefix_cache'])
                    cap = capture(model, frames, ids, row['metadata']['caption'], prefix=prefix)
                    native_calls += 1; prefix_hits += prefix is not None
                    prefix_arrays = {k: np.asarray(v) for k, v in cap['visual_prefix'].items()}
                    prefix_receipt = cache.put(ctx, prefix_arrays, required=True)
                    teacher, teacher_trace = catalog.teacher('tubedetr', frames[cap['positions8']],
                                         row['metadata']['caption'], p['teacher_fingerprint'])
                    ref = oldseal['files'][q]; assert sha(oldroot / ref['path']) == ref['sha256']
                    with np.load(oldroot / ref['path'], allow_pickle=False) as old:
                        for key in ['Source', 'Native_interval', 'frame_ids', 'positions8']:
                            assert np.array_equal(old[key], np.asarray(cap[key])), ('old Source drift', ds, q, key)
                    value = finish(cap, teacher, cfg)
                    arrays = numeric_output(value)
                    assert all(np.isfinite(v).all() for v in arrays.values())
                    small_ctx = dict(ctx, scope='query', input_sha256=digest(row),
                                     caption_sha256=digest(row['metadata']['caption']), seed=20260920,
                                     representation='all_original_native_outputs_logits6_and_q0')
                    small = {k: np.asarray(cap[k]) for k in ['Source', 'Native_interval', 'frame_ids', 'positions8', 'native_logits6', 'native_prior']}
                    native_cache = cache.put(small_ctx, small, required=True)
                    used = sum(f.stat().st_size for f in cache.root.glob('*.npz'))
                    dense_cache = {'status': 'not_cached_priority_limit'}
                    if not full or used < (48 << 30):
                        dense_ctx = dict(small_ctx, representation='native_postfusion_spatial_features_valid')
                        dense_cache = cache.put(dense_ctx, {'features': np.asarray(cap['features']),
                                                'valid': np.asarray(cap['valid'])}, required=not full)
                    tmp = path.with_suffix('.pending')
                    with tmp.open('xb') as stream:
                        np.savez_compressed(stream, **arrays)
                    tmp.replace(path)
                    meta = dict(reused_fixed128=False, new_native_calls=1,
                           visual_prefix_hit=prefix is not None, prefix_cache=prefix_receipt,
                           native_cache=native_cache, dense_cache=dense_cache,
                           teacher_cache_key=teacher_trace['cache_key'], new_teacher_calls=0,
                           Source_native_positions_bitexact_old=True, model_checkpoint_sha256=cfg['source_checkpoint_sha256'],
                           elapsed_seconds=time.perf_counter()-tick)
                    del frames, cap, teacher, value
                meta.update(plan_sha256=sha(PLAN), request_sha256=digest(row),
                            payload_sha256=sha(path), labels_read=False)
                write(receipt, meta)
            entries[q] = dict(path=path.name, sha256=sha(path), receipt_path=receipt.name,
                              receipt_sha256=sha(receipt))
            write(OUT / 'worker_progress.json', dict(dataset=ds, stage='full' if full else 'dev128',
                  complete=i+1, total=len(rows), labels_read=False,
                  visual_prefix_hits_this_worker=prefix_hits, native_calls_this_worker=native_calls))
            print(ds, 'full' if full else 'dev128', i+1, '/', len(rows), flush=True)
        assert before == state_hash(model.state_dict())
        write(target / 'sealed.json', dict(status='complete', queries=len(rows),
              dataset=ds, plan_sha256=sha(PLAN), files=entries, labels_read=False,
              model_state_unchanged=True, new_teacher_calls=0, native_calls=native_calls,
              visual_prefix_hits=prefix_hits, fixed_method=True))
        del model; gc.collect(); torch.cuda.empty_cache()
    catalog.db.close()


def score(ds, full):
    p = checked(); rows = rows_for(p, ds, full)
    target = OUT / ds / ('full' if full else 'dev128')
    s = json.loads((target / 'sealed.json').read_text())
    assert s['status'] == 'complete' and s['queries'] == len(rows) and s['plan_sha256'] == sha(PLAN)
    assert set(s['files']) == {r['id'] for r in rows}
    for e in s['files'].values():
        assert sha(target / e['path']) == e['sha256'] and sha(target / e['receipt_path']) == e['receipt_sha256']
    # No labels are hashed or parsed before the complete population verification.
    d = p['datasets'][ds]; label = 'full_labels' if full else 'labels'
    assert sha(ROOT / d[label]) == d[label + '_sha256']
    truth = {r['id']: r for r in map(json.loads, (ROOT / d[label]).read_text().splitlines())}
    from stvg.metrics import DenseTube, official
    from tools.tubedetr_opd_direct_diagnosis import cluster_difference
    details = []
    for row in rows:
        q = row['id']; gt = truth[q]; values = {}; observed = {}; unobserved = {}
        with np.load(target / s['files'][q]['path'], allow_pickle=False) as a:
            ids = a['frame_ids']; obs = set(ids[a['positions8']].tolist())
            meta = dict(frame_ids=ids.tolist(), input=row['metadata'])
            boxes = {int(k): v for k, v in gt['boxes'].items()}
            for arm in ARMS:
                b = a['Source' if arm == 'NativeMBR' else arm]
                interval = a['Native_interval'] if arm in ['Source','Spatial'] else a['Native_MBR_interval'] if arm == 'NativeMBR' else a['Selected_interval']
                m = DenseTube(b, meta, boxes, gt['interval'], clip=ds == 'hc2')
                values[arm] = m.score(interval)
                if len(details) < 2:
                    ref = official(b, meta, boxes, gt['interval'], interval, ds)
                    assert all(abs(values[arm][k] - ref[k]) < 1e-10 for k in ['t','s','v'])
                unseen = np.array([int(t) not in obs for t in m.fids])
                unobserved[arm] = float(m.iou[unseen].mean()) if unseen.any() else None
                observed[arm] = float(m.iou[~unseen].mean()) if (~unseen).any() else None
        details.append(dict(id=q, parent=gt['parent'], values=values, observed=observed, unobserved=unobserved))
    mean = lambda xs: float(np.mean([x for x in xs if x is not None])) if any(x is not None for x in xs) else None
    pp = lambda xs: None if mean(xs) is None else 100*mean(xs)
    unobserved_delta = [None if x['unobserved']['Full'] is None or x['unobserved']['Source'] is None else x['unobserved']['Full']-x['unobserved']['Source'] for x in details]
    result = dict(status='complete', dataset=ds, queries=len(rows), parents=len({r['parent'] for r in details}),
        means={arm:{k:100*mean([x['values'][arm][k] for x in details]) for k in ['v','s','t']} for arm in ARMS},
        unobserved_sIoU={arm:pp([x['unobserved'][arm] for x in details]) for arm in ARMS},
        observed_sIoU={arm:pp([x['observed'][arm] for x in details]) for arm in ARMS},
        observed_denominator={arm:sum(x['observed'][arm] is not None for x in details) for arm in ARMS},
        paired={k:cluster_difference(unobserved_delta if k == 'unobserved' else [x['values']['Full'][k]-x['values']['Source'][k] for x in details], [x['parent'] for x in details]) for k in ['t','v','s','unobserved']},
        temporal_feedback_vs_native_MBR_pp=100*mean([x['values']['Full']['t']-x['values']['NativeMBR']['t'] for x in details]),
        severe_time_or_v_union=sum(x['values']['Full']['t'] < x['values']['Source']['t']-.05 or x['values']['Full']['v'] < x['values']['Source']['v']-.05 for x in details),
        plan_sha256=sha(PLAN), prediction_seal_sha256=sha(target/'sealed.json'),
        frozen_no_search=True, development_and_previous_full_exposure=True,
        cache_cost_is_not_cold_deployment_cost=True)
    write(target/'details_private.json', details); write(target/'metrics.json', result)
    name = 'full_metrics.json' if full else 'development_metrics.json'
    write(PUBLIC/ds/name, result)
    if full:
        write(PUBLIC/ds/'final_seal.json', dict(status='complete', plan_sha256=sha(PLAN),
             prediction_seal_sha256=sha(target/'sealed.json'), files={str(f.relative_to(ROOT)):sha(f)
             for f in [PLAN, PUBLIC/'qualification.json', PUBLIC/'fixed_method_lock.json', PUBLIC/ds/name]},
             publication_pending=True))


def child(stage, ds, full=False):
    cmd = [sys.executable, '-B', '-u', '-m', 'tools.simplepath_tube.run', stage, '--dataset', ds]
    if full: cmd.append('--full')
    name = 'full' if full else 'dev128'
    log = ROOT/'logs'/f'simplepath_tubedetr_{ds}_{name}_{stage}.log'
    write(OUT/'status.json', dict(status=stage, dataset=ds, full=full, plan_sha256=sha(PLAN)))
    with log.open('ab') as stream:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                  env={**os.environ, 'CUDA_VISIBLE_DEVICES':''} if stage == 'score' else os.environ.copy())
        reg = ROOT/'runtime/tubedetr_active_jobs.json'; jobs = json.loads(reg.read_text())
        jobs['jobs']['simplepath_tubedetr_controller'] = dict(pid=os.getpid(),command='python -B -u -m tools.simplepath_tube.run run')
        jobs['jobs']['simplepath_tubedetr_worker'] = dict(pid=proc.pid, command=shlex.join(cmd), stage=stage,
              dataset=ds, full=full, log=str(log.relative_to(ROOT)))
        write(reg, jobs)
        if proc.wait(): raise RuntimeError('New fixed-method worker failed: ' + str(log))


def run():
    p = checked()
    if (OUT/'failure.json').exists(): raise ValueError('Prior failure; no automatic retry')
    for ds in p['dataset_order']:
        child('gpu', ds); child('score', ds)
    files = [OUT/ds/'dev128/sealed.json' for ds in p['dataset_order']] + [PUBLIC/ds/'development_metrics.json' for ds in p['dataset_order']]
    write(PUBLIC/'both128_gate.json', dict(status='complete_both_fixed128_before_full',
          plan_sha256=sha(PLAN), files={str(f.relative_to(ROOT)):sha(f) for f in files}, no_selection_or_retuning=True))
    for ds in p['full_order']:
        child('gpu', ds, True); child('score', ds, True)
    files = [PLAN, PUBLIC/'qualification.json', PUBLIC/'both128_gate.json', *[PUBLIC/ds/'final_seal.json' for ds in p['dataset_order']]]
    write(PUBLIC/'final_seal.json', dict(status='complete', files={str(f.relative_to(ROOT)):sha(f) for f in files}, publication_pending=True))
    write(OUT/'status.json', dict(status='complete_both_fixed128_and_full', plan_sha256=sha(PLAN)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['prepare','run','gpu','score'])
    parser.add_argument('--dataset', choices=['vidstg','hc2']); parser.add_argument('--full', action='store_true')
    args = parser.parse_args()
    try:
        if args.stage in ['prepare','run']: globals()[args.stage]()
        else: globals()[args.stage](args.dataset, args.full)
    except Exception as error:
        OUT.mkdir(parents=True, exist_ok=True)
        write(OUT/'failure.json',dict(status='stopped',stage=args.stage,dataset=args.dataset,
              error=type(error).__name__+': '+str(error),automatic_retry=False))
        raise
