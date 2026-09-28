"""Finite VidSTG native baselines and same-source safety; no implicit queue/GPU start.

Reuses unchanged tested ports. Native uses HC2 weights; safety uses Vid weights
and their locked vid_to_hc1 preset on VidSTG inputs (no target retuning).
"""
import argparse
import collections
import gc
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.decota_matrix_common_v1 import read, write, status, sha, save, load

OUT = ROOT / 'artifacts/decota_paper_execution_20260917/vid_last'
PARENT = ROOT / 'artifacts/decota_final_simplification_v1'
METHODS = ['Tent', 'MEMO', 'SAR']
COHORT = 'vidstg_test'
CODE = ['scripts/run_vid_last_paper_20260917.py',
        'tests/test_vid_last_paper_20260917.py',
        'protocols/decota_vid_last_20260917.md',
        'scripts/decota_matrix_common_v1.py',
        'scripts/run_final_simplification_v1.py',
        'vg_tta/native_baselines_paper_v1.py',
        'vg_tta/native_probability_interface_v1.py',
        'vg_tta/tastvg_baseline_expansion.py',
        'vg_tta/paper_continuation_v1.py',
        'vg_tta/exact_frame_decode_audit_v2.py',
        'vg_tta/unanchored_dense_shift_data_v1.py',
        'scripts/analyze_spatial10_components_v1.py',
        'scripts/score_decota_final_freeze_v1.py']
INPUT_FIELDS = {'caption', 'index', 'width', 'height', 'fps', 'source',
                'original_video_id', 'video_path', 'video_sha256', 'frame_ids',
                'frame_count', 'duration', 'start_frame', 'end_frame', 'kind'}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def validate_roster(rows):
    require(len({r['key'] for r in rows}) == len(rows), 'Duplicate query')
    require(len({r['ordinal'] for r in rows}) == len(rows), 'Duplicate ordinal')
    for row in rows:
        q = row['input']; ids = q['frame_ids']
        require(set(q) <= INPUT_FIELDS, 'Non-input annotation field')
        require(q['kind'] == 'vidstg' and q['fps'] > 0, 'Wrong target/fps')
        require(ids == sorted(set(ids)) and len(ids) >= 4, 'Invalid frame grid')
        require(ids[0] >= q['start_frame'] and ids[-1] < q['end_frame'], 'Clip/grid mismatch')
        require(row['key'] == f'{COHORT}:{row["ordinal"]:06d}', 'Key mismatch')


def prepare():
    from methods.decota_final_simplified_v1.release import verify_release
    from methods.decota_final_simplified_v1.config import MethodConfig
    from vg_tta.native_baselines_paper_v1 import DEFAULTS
    verify_release()
    if (OUT / 'LOCK.json').exists():
        return verify()
    parent = read(PARENT / 'LOCK.json')
    full = parent['full_rows'][COHORT]
    # Original first eight development sources; no metric/result selection.
    selected = [r for r in parent['rows'][COHORT] if r['f44_role'] == 'development'][:8]
    by_key = {r['key']: r for r in full}
    dev = [by_key[r['key']] for r in selected]
    validate_roster(full); validate_roster(dev)
    require(len(full) == 10303 and len({r['input']['source'] for r in full}) == 732,
            'Full target roster changed')
    require(len(dev) == 8 and len({r['input']['source'] for r in dev}) == 8, 'Dev roster changed')
    parent_receipts = {}; parent_lock_sha256 = sha(PARENT / 'LOCK.json')
    for row in full:
        path = PARENT / 'full' / COHORT / f'{row["ordinal"]:06d}.pt'
        recpath = path.with_suffix('.json'); rec = read(recpath)
        require(rec['key'] == row['key'] and rec['path'] == str(path), 'Parent receipt identity')
        require(rec['lock_sha256'] == parent_lock_sha256, 'Parent receipt lock')
        parent_receipts[row['key']] = dict(path=str(path), receipt_path=str(recpath),
            receipt_sha256=sha(recpath), output_sha256=rec['sha256'])
    pins = [PARENT / 'LOCK.json', PARENT / 'FINAL_CONFIG.json',
            ROOT / 'methods/CURRENT_METHOD.json', ROOT / 'methods/CURRENT_WORKING_METHOD.json',
            ROOT / 'methods/decota_final_simplified_v1/RELEASE.json',
            ROOT / 'protocols/decota_paper_execution_20260917_amendment1.md']
    p = dict(created=time.time(), target=COHORT, dev=dev, full=full, methods=METHODS,
             config=DEFAULTS, native_direction='hc2_to_vid', safety_direction='vid_to_hc1',
             safety_actual_direction='vidstg_source_to_vidstg_test',
             source_configs={s: MethodConfig.for_direction(d).to_dict() for s, d in
                             [('native', 'hc2_to_vid'), ('safety', 'vid_to_hc1')]},
             pins={str(f): sha(f) for f in pins}, code_pins={f: sha(ROOT/f) for f in CODE},
             parent_receipts=parent_receipts, labels=parent['labels'],
             labels_sha256=parent['labels_sha256'], GT_online=False,
             labels_used_only_in_scoring=True, historical_exposure=True,
             retuning=False, no_recurring_task=True, finite_hours=48,
             safety_Frozen='fresh same-Vid-source live forward, exact replay parity every query',
             native_Frozen='exact HC2-source sealed-parent parity every method/query')
    write(OUT / 'LOCK.json', p)
    print('PREPARED; no GPU started:', len(full), 'queries / 732 sources; dev 8', flush=True)
    return p


def verify():
    from methods.decota_final_simplified_v1.release import verify_release
    verify_release(); p = read(OUT / 'LOCK.json')
    for name, digest in {**p['pins'], **p['code_pins']}.items():
        require(sha(ROOT/name) == digest, 'Locked dependency changed: ' + name)
    return p


def write_same(path, value):
    """Resume a partly written CPU score without replacing any existing data."""
    path = Path(path)
    if path.exists():
        require(read(path) == value, 'Partial score differs; preserved for review: '+str(path))
    else:
        write(path, value)


def parity(actual, expected):
    import torch
    for name in ['indices', 'physical_interval', 'raw_offset_indices', 'raw_physical_intervals']:
        require(actual[name] == expected[name], 'Frozen interval parity: ' + name)
    require(torch.equal(actual['boxes'], expected['boxes']), 'Frozen box parity')
    require(len(actual['logits']) == len(expected['logits']) == 2, 'Frozen offset count')
    require(all(torch.equal(a, b) for a, b in zip(actual['logits'], expected['logits'])),
            'Frozen logit parity')


def checked_parent(p, row):
    ref = p['parent_receipts'][row['key']]
    require(sha(ref['receipt_path']) == ref['receipt_sha256'], 'Parent receipt changed')
    require(sha(ref['path']) == ref['output_sha256'], 'Parent output changed')
    x = load(ref['path'])
    require(x['key'] == row['key'] and x['input'] == row['input'], 'Parent input mismatch')
    return x, ref


def receipt(path):
    path = Path(path); rp = path.with_suffix('.json')
    if rp.exists():
        r = read(rp)
        require(r['path'] == str(path) and r['lock_sha256'] == sha(OUT/'LOCK.json'), 'Receipt identity')
        require(sha(path) == r['sha256'], 'Receipt content mismatch')
        return r
    require(not path.exists() and not path.with_suffix('.tmp').exists(),
            'Unreceipted output preserved; manual review required')
    return None


def barrier(p, stage, phase):
    folder = OUT/stage/phase; b = read(folder/'BARRIER.json')
    rows = p[phase]
    require(b['complete'] and b['functional_validation_passed'], 'Incomplete functional barrier')
    require(b['lock_sha256'] == sha(OUT/'LOCK.json'), 'Barrier lock mismatch')
    require([r['key'] for r in b['receipts']] == [r['key'] for r in rows], 'Barrier roster mismatch')
    for row, r in zip(rows, b['receipts']):
        path = folder/f'{row["ordinal"]:06d}.pt'
        require(receipt(path) == r, 'Barrier receipt mismatch')
    return b


def model_load(direction):
    import numpy as np
    import torch
    from methods.decota_final_simplified_v1.config import MethodConfig
    from methods.decota_final_simplified_v1._tastvg_load import load_model_on_device
    from methods.decota_final_simplified_v1.observations import QuerySubjectParser
    cfg = MethodConfig.for_direction(direction); ckpt = ROOT/cfg.checkpoint
    require(sha(ckpt) == cfg.checkpoint_sha256, 'Checkpoint mismatch')
    torch.set_num_threads(4); torch.manual_seed(20260910); np.random.seed(20260910)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    runtime = ROOT/'artifacts/tastvg_runtime'/('hc-stvg2' if cfg.source_dataset == 'hcstvg2' else 'vidstg')
    model, _, _ = load_model_on_device(cfg.source_dataset, runtime, checkpoint=ckpt,
                                    device='cuda', source_dataset=cfg.source_dataset)
    return model.eval().requires_grad_(False), QuerySubjectParser(ROOT/'.cache/stanza')


def native_episode(p, phase, at, model, parser, frames, ids, row):
    import torch
    from vg_tta.native_baselines_paper_v1 import episode
    parent, ref = checked_parent(p, row); frozen = parent['predictions']['Frozen']
    x = dict(methods={}, controls={}, parent_ref=ref)
    for method in METHODS:
        torch.cuda.synchronize(); start = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
        z = episode(model, parser, frames, ids, row['input'], method, p['config'][method])
        torch.cuda.synchronize()
        z.update(seconds_with_parity_forward=time.perf_counter()-start,
                 peak_memory_allocated=torch.cuda.max_memory_allocated())
        parity(z['Frozen'], frozen)
        audit = z['audit']
        require(audit['source_restored'] and audit['expert_calls'] == 0, 'Native reset/expert failure')
        require(torch.isfinite(z['prediction']['boxes']).all(), 'Nonfinite native boxes')
        if method in ('Tent', 'MEMO'):
            require(audit['backwards'] > 0 and max(audit['gradient_norms']) > 0, 'Missing native gradient')
        if not audit['parameter_changed']:
            parity(z['prediction'], frozen)
        x['methods'][method] = z
        if phase == 'dev' and at == 0:
            for name, override in [('steps0', dict(steps=0)), ('lr0', dict(lr=0.))]:
                ctrl = episode(model, parser, frames, ids, row['input'], method,
                               {**p['config'][method], **override})
                parity(ctrl['Frozen'], frozen); parity(ctrl['prediction'], frozen)
                require(ctrl['audit']['source_restored'] and not ctrl['audit']['parameter_changed'],
                        'Native no-op/reset failure')
                x['controls'][method+'_'+name] = ctrl['audit']
    return x


def safety_episode(predictor, frames, ids, row):
    import torch
    from methods.decota_final_simplified_v1.backbone import make_batch
    from vg_tta.native_baselines_paper_v1 import live_output
    from vg_tta.paper_continuation_v1 import generalization_prediction
    # Source-specific live Frozen, never HC2 cross-source parent predictions.
    batch = make_batch(frames, ids, row['input'], predictor.model)
    subject = predictor.parser(row['input']['caption'])['subject']
    with torch.no_grad():
        frozen = live_output(predictor.model, batch, ids, subject)[1]
    x = generalization_prediction(predictor, frames, ids, row['input'])
    parity(x['predictions']['Frozen'], frozen)
    require(x['source_restored'] and x['GT_online'] is False, 'Safety restore/input failure')
    require(x['spatial']['source_restored'] and x['temporal']['source_restored'], 'Safety fitter restore failure')
    require(all(torch.isfinite(z['boxes']).all() for z in x['predictions'].values()), 'Safety nonfinite boxes')
    x.update(fresh_same_source_Frozen_parity=True, extra_Frozen_forward=True,
             source_dataset='vidstg', target_dataset=COHORT,
             preset_direction='vid_to_hc1', actual_direction='vidstg_source_to_vidstg_test')
    return x


def run(stage, phase, limit=0):
    import torch
    from scripts.run_final_simplification_v1 import lease
    from vg_tta.exact_frame_decode_audit_v2 import decode
    p = verify(); folder = OUT/stage/phase; rows = p[phase]
    if phase == 'full':
        barrier(p, stage, 'dev')
    if (folder/'BARRIER.json').exists():
        barrier(p, stage, phase); print('Already complete', stage, phase, flush=True); return
    gpu = lease()
    try:
        active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                         '--format=csv,noheader'], text=True).strip()
        require(not active, 'GPU not exclusive: '+active)
        if stage == 'native':
            model, parser = model_load(p['native_direction']); predictor = None
        else:
            from methods.decota_final_simplified_v1 import DeCoTAPredictor
            predictor = DeCoTAPredictor.from_pretrained(p['safety_direction'])
        receipts = []; pixels = collections.OrderedDict(); media = {}; done = 0; started = time.time()
        for at, row in enumerate(rows):
            path = folder/f'{row["ordinal"]:06d}.pt'; r = receipt(path)
            if r:
                require(r['key'] == row['key'], 'Resume query mismatch'); receipts.append(r); continue
            require(time.time()-started < p['finite_hours']*3600, 'Finite stage cap reached')
            require(shutil.disk_usage(ROOT).free >= 40*2**30, 'Disk guard')
            q = row['input']
            if row.get('input_unavailable'):
                x = dict(status='known_input_unavailable', reason=row['input_unavailable'])
            else:
                if q['video_path'] not in media:
                    media[q['video_path']] = sha(q['video_path'])
                require(media[q['video_path']] == q['video_sha256'], 'Media changed')
                pk = (q['video_sha256'], tuple(q['frame_ids'])); tick = time.perf_counter()
                if pk in pixels:
                    frames, ids = pixels.pop(pk)
                else:
                    frames, ids = decode(q)
                pixels[pk] = (frames, ids)
                while len(pixels) > 2:
                    pixels.popitem(last=False)
                require(ids == q['frame_ids'], 'Decoder changed full locked frame grid')
                decode_seconds = time.perf_counter()-tick
                x = (native_episode(p, phase, at, model, parser, frames, ids, row) if stage == 'native'
                     else safety_episode(predictor, frames, ids, row))
                x.update(status='available', decode_seconds=decode_seconds)
            x.update(key=row['key'], input=q, source=q['source'], stage=stage, phase=phase,
                     GT_online=False, lock_sha256=sha(OUT/'LOCK.json'))
            save(path, x)
            r = dict(key=row['key'], path=str(path), sha256=sha(path), lock_sha256=x['lock_sha256'],
                     completed=time.time(), status=x['status'])
            write(path.with_suffix('.json'), r); receipts.append(r); done += 1
            status(folder/'STATUS.json', dict(status='running', done=len(receipts), total=len(rows),
                   key=row['key'], pid=os.getpid(), updated=time.time()))
            print(stage, phase, len(receipts), '/', len(rows), row['key'], flush=True)
            del x; gc.collect(); torch.cuda.empty_cache()
            if limit and done >= limit:
                break
        if len(receipts) == len(rows):
            write(folder/'BARRIER.json', dict(complete=True, receipts=receipts, queries=len(rows),
                  sources=len({r['input']['source'] for r in rows}), functional_validation_passed=True,
                  selection_by_GT=False, lock_sha256=sha(OUT/'LOCK.json')))
            status(folder/'STATUS.json', dict(status='prediction_complete', done=len(rows),
                   total=len(rows), updated=time.time()))
    finally:
        gpu.close()


def score(stage, phase):
    from scripts.analyze_spatial10_components_v1 import checked_score
    from scripts.score_decota_final_freeze_v1 import aggregate
    p = verify(); b = barrier(p, stage, phase); dest = OUT/stage/phase/'analysis'
    if (dest/'COMPLETION.json').exists():
        for f, digest in read(dest/'COMPLETION.json')['files'].items():
            require(sha(f) == digest, 'Scored output changed')
        return
    # This is the only target-label read in this module.
    require(sha(p['labels']) == p['labels_sha256'], 'Scorer labels changed')
    labels = read(p['labels']); scored = []; missing = []; audits = []
    for row, r in zip(p[phase], b['receipts']):
        x = load(r['path'])
        require(x['input'] == row['input'] and x['key'] == row['key'], 'Scoring input mismatch')
        if x['status'] != 'available':
            missing.append(dict(key=x['key'], reason=x['reason'])); continue
        if stage == 'native':
            parent, _ = checked_parent(p, row)
            predictions = {m: x['methods'][m]['prediction'] for m in METHODS}
            predictions.update(Frozen=parent['predictions']['Frozen'], DeCoTA=parent['predictions']['Full_DeCoTA'])
            audit = {m: x['methods'][m]['audit'] for m in METHODS}
        else:
            predictions = x['predictions']
            audit = dict(source_restored=x['source_restored'], backwards=x['backwards'],
                         actual_DINO_calls=x['actual_DINO_calls'],
                         spatial_parameter_changed=x['spatial']['parameter_changed'],
                         temporal_parameter_changed=x['temporal']['shrunk_parameter_changed'],
                         fresh_same_source_Frozen_parity=x['fresh_same_source_Frozen_parity'])
        vals = {m: checked_score(z['boxes'], labels[x['key']], row['input']['frame_ids'], z['indices'])[0]
                for m, z in predictions.items()}
        scored.append(dict(key=x['key'], source=x['source'], arms=vals))
        audits.append(dict(key=x['key'], audit=audit))
    names = ['Frozen', *METHODS, 'DeCoTA'] if stage == 'native' else ['Frozen', 'Full_DeCoTA']
    metrics = ['vIoU_corrected', 'sIoU', 'tIoU']; sources = [r['source'] for r in scored]
    arms = {a: {m: aggregate([r['arms'][a][m] for r in scored], sources) for m in metrics} for a in names}
    pairs = {a+' - Frozen': {m: aggregate([None if r['arms'][a][m] is None or r['arms']['Frozen'][m] is None
              else r['arms'][a][m]-r['arms']['Frozen'][m] for r in scored], sources) for m in metrics}
             for a in names if a != 'Frozen'}
    write_same(dest/'ALL_QUERY_RESULTS.json', dict(rows=scored, missing=missing))
    write_same(dest/'ALL_SOURCE_RESULTS.json', dict(nominal=len(p[phase]), queries=len(scored),
          sources=len(set(sources)), stage=stage, phase=phase, arms=arms, paired=pairs,
          source_config=p['source_configs'][stage], historical_exposure=True, no_target_retuning=True))
    write_same(dest/'AUDIT.json', dict(rows=audits, missing=missing, GT_online=False,
          source_matched_Frozen=True, historical_exposure=True))
    write(dest/'COMPLETION.json', dict(status='scored_complete', lock_sha256=sha(OUT/'LOCK.json'),
          files={str(f): sha(f) for f in dest.iterdir() if f.is_file()}))
    print('SCORED', stage, phase, len(scored), len(set(sources)), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action', choices=['prepare', 'native-run', 'native-score', 'safety-run', 'safety-score'])
    ap.add_argument('--phase', choices=['dev', 'full'], default='dev')
    ap.add_argument('--limit', type=int, default=0, help='Optional bounded debugging; never creates a partial barrier')
    a = ap.parse_args()
    require(a.limit >= 0, 'Negative limit')
    if a.action == 'prepare':
        prepare(); return
    stage, action = a.action.split('-'); folder = OUT/stage/a.phase
    process = dict(pid=os.getpid(), started=time.time(), action=a.action, phase=a.phase, status='running')
    status(folder/'PROCESS.json', process)
    try:
        (run(stage, a.phase, a.limit) if action == 'run' else score(stage, a.phase))
    except BaseException as exc:
        process.update(status='failed', error=repr(exc), finished=time.time())
        status(folder/'PROCESS.json', process)
        status(folder/'STATUS.json', dict(status='failed', action=a.action, error=repr(exc), updated=time.time()))
        raise
    process.update(status='complete' if not a.limit else 'bounded_return', finished=time.time())
    status(folder/'PROCESS.json', process)


if __name__ == '__main__':
    main()
