"""Run source calibration and target prediction as separate, sealed stages."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
import gc
import multiprocessing
import os
from pathlib import Path
import sys
import time

import numpy as np

from evidence162 import EvidenceGrounder
from manifest162 import (DEV, EXTRA, REC, HERE, PLATFORMS, ROOT, WORK, atomic_json,
                         check_frozen, read, sensor_row, sha, target_rows)
from solver162 import CALIBRATION, CONTROLS, MAIN, Options, simple_boxes, solve
from support162 import SceneSupport

sys.path.insert(0, str(ROOT / 'research/round6/source_validation_scoring'))
from batch_yaw_iou import many_iou

PROTOCOL = HERE / 'experiment/PROTOCOL.json'
SETTINGS = HERE / 'experiment/SOURCE_PARAMETERS.json'


def array_file(path, **values):
    temp = path.with_suffix('.tmp')
    with temp.open('wb') as stream:
        np.savez_compressed(stream, **values)
    os.replace(temp, path)


def prior_from_validation(records):
    sizes = np.array([r['val_GT'][3:6] for r in records])
    if not np.isfinite(sizes).all() or (sizes <= 0).any():
        raise ValueError('Invalid weak source size prior')
    logs = np.log(sizes)
    mid = np.median(logs, axis=0)
    return mid, np.maximum(1.4826 * np.median(np.abs(logs - mid), axis=0), .65)


def calibrate_case(path, ev, prior, sigma):
    scene = SceneSupport(ev)
    boxes = [solve(scene, opt, prior, sigma).box for opt in CALIBRATION.values()]
    simple = simple_boxes(scene, prior, sigma)
    array_file(path, names=np.array([*CALIBRATION, *simple, 'r158']), boxes=np.stack([*boxes, *simple.values(), ev.old_pca]))


def all_pca(ev):
    """Equivalent to applying the frozen R155 orientation to all 256 old boxes."""
    from scipy.spatial import cKDTree
    p = ev.points.astype(float)
    tree = cKDTree(p)
    raw = ev.boxes
    eligible = (raw[:, 3:6] > 0).all(1) & (raw[:, 3] / np.maximum(raw[:, 4], 1e-6) >= 1.5)
    out = np.zeros(256, dtype=np.float32)
    for k in np.flatnonzero(eligible):
        b = raw[k]
        around = np.asarray(tree.query_ball_point(b[:3], b[3:6].max() / 2 + 1e-5, p=np.inf), dtype=int)
        # Exact R155 float32 crop and repeated point treatment.
        sample = ev.points[around]
        crop = np.unique(sample[np.all(np.abs(sample - b[:3]) <= b[3:6] / 2 + 1e-7, axis=1)], axis=0)
        if len(crop) < 8:
            continue
        values, v = np.linalg.eigh(np.cov(crop[:, :2].astype(float), rowvar=False))
        if values[-1] > 1e-12:
            angle = np.arctan2(v[1, -1], v[0, -1])
            out[k] = (angle + np.pi / 2) % np.pi - np.pi / 2
    if out[ev.old_winner] != ev.old_pca[6]:
        raise ValueError('Batched old-pool PCA differs from the frozen winner PCA')
    return out


def control_objects(scene, results, prior, sigma):
    main = results['lcib']
    simple = simple_boxes(scene, prior, sigma, main.old_support)
    answers = {**simple, **{name: result.box for name, result in results.items()}}
    # A one-fit endpoint already produced by the joint solve is a genuine
    # one-cycle result. The independent one-fit control replays it only on the panel.
    return answers


def pack_case(path, ev, options, prior, sigma, panel, off, wrong):
    scene = SceneSupport(ev)
    names = tuple(CONTROLS) if panel else ('lcib', 'serial')
    results = {name: solve(scene, replace(options, **CONTROLS[name]), prior, sigma) for name in names}
    boxes = control_objects(scene, results, prior, sigma)
    if panel:
        results['system_no_rgb'] = solve(SceneSupport(off), options, prior, sigma)
        boxes['system_no_rgb'] = results['system_no_rgb'].box
        boxes['system_no_rgb_old'] = off.old_pca
        if wrong is not None:
            results['wrong_frame_rgb'] = solve(SceneSupport(wrong), options, prior, sigma)
            boxes['wrong_frame_rgb'] = results['wrong_frame_rgb'].box
        else:
            boxes['wrong_frame_rgb'] = results['lcib'].box
    result = results['lcib']
    values = dict(arms=np.array(tuple(boxes)), boxes=np.stack(list(boxes.values())), all_pca=all_pca(ev),
                  old_pca=ev.old_pca, old_winner=ev.old_winner, generated=result.hypotheses,
                  origins=result.origins, selected_origin=result.result_origin, fallback=result.fallback,
                  visibility=np.mean(ev.visible), roles=len(ev.affinity), known_edges=len(ev.edges),
                  swapped_seeds=int(np.count_nonzero(scene.swap != np.arange(scene.nseed))),
                  enough_wrong_frame=wrong is not None, role_unknown=int(np.sum(result.role_queries == -2)),
                  forward_seconds=ev.forward_seconds, retained=len(result.origins),
                  fit_calls=result.fit_calls, sweeps=result.sweeps, solve_seconds=result.elapsed,
                  serial_seconds=results['serial'].elapsed, serial_sweeps=results['serial'].sweeps,
                  serial_fit_calls=results['serial'].fit_calls,
                  trace=np.asarray(result.trace), trace_boxes=result.trace_boxes)
    if panel:
        for key in asdict(ev):
            # Retain the frozen evidence on the dedicated local diagnostic panel.
            values['ev_'+key] = getattr(ev, key)
        for name, r in results.items():
            values[name+'_support'] = r.role_support
            values[name+'_initial_support'] = r.initial_support
            values[name+'_old_support'] = r.old_support
            values[name+'_role_boxes'] = r.role_boxes
            values[name+'_role_queries'] = r.role_queries
            values[name+'_trace'] = np.asarray(r.trace)
            values[name+'_trace_boxes'] = r.trace_boxes
            values[name+'_pool'] = r.hypotheses
            values[name+'_origins'] = r.origins
            values[name+'_fit_work'] = np.array([r.sweeps, r.fit_calls, r.elapsed])
    assert np.isfinite(values['boxes']).all() and (values['boxes'][:, 3:6] > 0).all()
    array_file(path, **values)


def matching_wrong_frames(rows, panel):
    """Use another ordinary recording frame with the same camera, platform and scene."""
    sets = {}
    for i, e in enumerate(rows):
        r = e['r']
        key = (e['target'], Path(r['metapath']).resolve().parent.parent, int(r['suffix'] or 0))
        sets.setdefault(key, {})[e['image_sha']] = i
    chosen = {}
    for i in panel:
        e = rows[i]
        r = e['r']
        key = (e['target'], Path(r['metapath']).resolve().parent.parent, int(r['suffix'] or 0))
        others = [j for h, j in sets[key].items() if h != e['image_sha'] and rows[j]['point_sha'] != e['point_sha']]
        if others:
            # Adjacent within the same recording order; geometry stays at the original frame.
            chosen[i] = rows[min(others, key=lambda j: abs(j - i))]['r']['imagepath']
    return chosen


def pool():
    return ProcessPoolExecutor(max_workers=6, mp_context=multiprocessing.get_context('spawn'))


def submit(executor, pending, fn, *arguments):
    pending.append(executor.submit(fn, *arguments))
    if len(pending) >= 18:
        pending.pop(0).result()


def sync(pending):
    for future in pending:
        future.result()
    pending.clear()


def seal(name, folder, expected, **extra):
    files = {p.name: sha(p) for p in sorted(folder.glob('*.npz'))}
    if set(files) != {f'{i:05d}.npz' for i in expected}:
        raise ValueError('Sealed scope differs from the declared keys')
    path = WORK/(name+'_FILES_PRIVATE.json')
    atomic_json(path, files)
    atomic_json(HERE/(name+'_SEAL.json'), dict(n=len(files), file_index_sha256=sha(path),
                protocol_sha256=sha(PROTOCOL), FINAL_access=False, **extra))


def calibration():
    import torch

    if SETTINGS.exists():
        raise FileExistsError(SETTINGS)
    samples = read(WORK/'SOURCE_VAL_PRIVATE.json')
    folder = WORK/'calibration_predictions'
    folder.mkdir(exist_ok=True)
    selected = {}
    start = time.perf_counter()
    with pool() as executor:
        for source in PLATFORMS:
            indices = [i for i, r in enumerate(samples) if r['source'] == source]
            prior, sigma = prior_from_validation([samples[i] for i in indices])
            model = EvidenceGrounder(source)
            pending = []
            for i in indices:
                path = folder / f'{i:05d}.npz'
                if not path.exists():
                    ev = model.evidence_frame(sensor_row(samples[i]))
                    submit(executor, pending, calibrate_case, path, ev, prior, sigma)
            sync(pending)
            selected[source] = dict(log_dimensions=prior.tolist(), log_scale=sigma.tolist())
            print('Source validation outputs ready', source, len(indices), flush=True)
            del model
            gc.collect()
            torch.cuda.empty_cache()
    seal('CALIBRATION', folder, range(len(samples)), source_labels_for_size_prior=True,
         elapsed=time.perf_counter()-start, target_labels_for_selection=False)
    for source in PLATFORMS:
        values = []
        for i, row in enumerate(samples):
            if source != row['source']:
                continue
            with np.load(folder/f'{i:05d}.npz', allow_pickle=False) as pred:
                names = pred['names'].tolist()
                values.append(many_iou(pred['boxes'], np.array(row['val_GT'])))
        iou = np.stack(values)
        score = 50 * ((iou > .25).mean(0) + (iou > .50).mean(0))
        ci = int(np.argmax(score[:len(CALIBRATION)]))
        geo = [names.index(n) for n in ('pca_obb', 'uniform_fit', 'uniform_fixed')]
        strongest = geo[int(np.argmax(score[geo]))]
        selected[source].update(configuration=names[ci], options=asdict(CALIBRATION[names[ci]]),
            simple_control=names[strongest], calibration_count=len(iou),
            metrics={n:dict(Acc25=100 * float((iou[:, j] > .25).mean()),
                            Acc50=100 * float((iou[:, j] > .5).mean()), mIoU=100 * float(iou[:, j].mean()))
                     for j, n in enumerate(names)})
    atomic_json(SETTINGS, dict(sources=selected, source_only=True, code_sha256=sha(Path(__file__)),
                protocol_sha256=sha(PROTOCOL), calibration_prediction_seal_sha256=sha(HERE/'CALIBRATION_SEAL.json'),
                FINAL_access=False))
    print('Source parameters locked', {s:x['configuration'] for s,x in selected.items()}, flush=True)


def predict(scope):
    import torch

    fixed = read(SETTINGS)
    if fixed['protocol_sha256'] != sha(PROTOCOL):
        raise ValueError('Protocol changed after source calibration')
    registry = target_rows()
    subset = [i for i, e in enumerate(registry)
              if (e['cohort'] == DEV if scope == 'DEV' else e['cohort'] in (EXTRA, REC))]
    count = 1798 if scope == 'DEV' else 9014
    assert len(subset) == count
    panel = set(read(WORK/'AUDIT_PANEL_PRIVATE.json')) if scope == 'DEV' else set()
    wrong = matching_wrong_frames(registry, panel)
    folder = WORK/scope
    folder.mkdir(exist_ok=True)
    start = time.perf_counter()
    gpu_peak = 0.
    counts = Counter()
    with pool() as executor:
        for source in PLATFORMS:
            model = EvidenceGrounder(source)
            settings = fixed['sources'][source]
            options = Options(**settings['options'])
            prior, sigma = (np.array(settings[p]) for p in ('log_dimensions','log_scale'))
            pending = []
            indices = [i for i in subset if registry[i]['source'] == source]
            for n, i in enumerate(indices, 1):
                path = folder / f'{i:05d}.npz'
                if not path.exists():
                    row = sensor_row(registry[i]['r'])
                    ev = model.evidence_frame(row)
                    check_frozen(i, ev)
                    off = model.evidence_frame(row, rgb_mode='off') if i in panel else None
                    other = model.evidence_frame(row, rgb_mode='other_frame', other_image=wrong[i]) if i in wrong else None
                    submit(executor, pending, pack_case, path, ev, options, prior, sigma, i in panel, off, other)
                if n % 128 == 0:
                    print(scope, source, n, len(indices), round(time.perf_counter()-start, 1), flush=True)
            sync(pending)
            counts[source] = len(indices)
            gpu_peak = max(gpu_peak, torch.cuda.max_memory_allocated() / 2 ** 20)
            del model
            gc.collect()
            torch.cuda.empty_cache()
    seal(scope, folder, subset, source_counts=counts, presealed_target_annotations_unused=True,
         all_points_and_frozen_R158_outputs_checked=True, target_parameter_updates=0,
         source_parameters_sha256=sha(SETTINGS), predictor_sha256=sha(Path(__file__)),
         forward_count_one_frame_each=count + len(panel) + len(wrong),
         neural_forwards_per_standard_observation=dict(native=1, route_bootstrap=1, prepared_full=1),
         elapsed=time.perf_counter()-start, peak_MiB=gpu_peak)
    print('Sealed', scope, len(subset), flush=True)


if __name__ == '__main__':
    mode, = sys.argv[1:]
    if mode == 'calibration':
        calibration()
    else:
        if mode not in ('DEV', 'EVAL'):
            raise ValueError(mode)
        predict(mode)
