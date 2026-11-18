"""Replay raw sensor inputs independently against the R158 sealed ordinary predictions."""
import gc
import json
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'crossplatform/round7')]
from camera_projection import WAYMO_ANGLES, raw_to_model_points, read_camera_allowlist
from crossplatform.ground_single_frame import SingleFrameGrounder, WAYMO_VIEWS, file_sha256, orient_box


# Fail the raw entry if it attempts to consume prior predictions, saved role maps or evaluation splits.
guarded = False


def check_open(event, args):
    if not guarded or event != 'open' or not isinstance(args[0], (str, bytes)):
        return
    p = Path(args[0].decode() if isinstance(args[0], bytes) else args[0])
    if (any(part.startswith('FINAL') or part in {'roles_PRIVATE', 'samples_PRIVATE', 'records_PRIVATE',
             'evidence_PRIVATE', 'prefix_PRIVATE', 'features_PRIVATE', 'shared_physical_geometry_PRIVATE'} for part in p.parts)
            or p.name in {'EXTRA_PRIVATE.npz', 'EXTRA_REGISTRY_PRIVATE.json', 'INPUT_REGISTRY_PRIVATE.json',
                          'SELECTED_SOURCE_ONLY.json'}
            or p == annotation_metadata):
        raise RuntimeError('The raw entry attempted to consume a research or annotation artifact')


@contextmanager
def deployment():
    global guarded
    guarded = True
    try:
        yield
    finally:
        guarded = False


def read(path):
    return json.loads(path.read_text())


def convert_camera_value(value):
    # The project allowlist uses streaming JSON Decimal numbers.
    return float(value)


def main():
    global annotation_metadata
    annotation_metadata = None
    sys.addaudithook(check_open)
    import torch
    extra = ROOT / 'crossplatform/round158'
    manifest = extra / 'EXTRA_REGISTRY_PRIVATE.json'
    records = read(manifest)
    seal = read(extra / 'EXTRA_SEAL.json')
    assert file_sha256(manifest) == seal['registry']
    assert file_sha256(extra / 'EXTRA_PRIVATE.npz') == seal['sha256']
    with np.load(extra / 'EXTRA_PRIVATE.npz', allow_pickle=False) as p:
        indices, boxes = p['indices'], p['boxes']
    selected = {}
    for i, rec in enumerate(records):
        key = (rec['source'], rec['target'], int(rec['row']['suffix'] or 0) if rec['target'] == 'waymo' else 0)
        selected.setdefault(key, i)
    assert len(selected) == 14
    cases = []
    check = read(HERE / 'experiment/PROTOCOL.json')['acceptance_tolerance']
    start = time.monotonic()
    extra_checks = {}
    with tempfile.TemporaryDirectory() as tmp:
        calibration = Path(tmp) / 'calibration.json'
        for src in ('waymo', 'drone', 'quad'):
            cwd = Path.cwd()
            with deployment():
                model = SingleFrameGrounder(src)
            assert Path.cwd() == cwd
            assert not any(p.requires_grad for m in (model.base, model.matcher) for p in m.parameters())
            for (s, target, view), i in selected.items():
                if s != src:
                    continue
                r = records[i]['row']
                annotation_metadata = Path(r['metapath'])
                fields = read_camera_allowlist(annotation_metadata)
                calibration.write_text(json.dumps(fields, default=convert_camera_value))
                args = dict(target=target, points=r['pointpath'], rgb=r['imagepath'], calibration=calibration,
                            text=r['caption'], seed=r['seed'], waymo_view=WAYMO_VIEWS[view])
                tic = time.monotonic()
                with deployment():
                    output = model.score_frame(**args)
                    sensor_box = output.box()
                old, new = int(output.probability.argmax()), output.winner()
                assert [old, new] == indices[i].tolist(), (s, target, view, 'winner')
                both = np.stack([orient_box(output.sampled_points, output.boxes[q]) for q in (old, new)])
                box_error = float(abs(both-boxes[i]).max())
                path = ROOT / 'crossplatform/round97' / f'{s}_to_{target}' / 'samples_PRIVATE' / (r['id']+'.npz')
                with np.load(path, allow_pickle=False) as prior:
                    p = prior['scores'][prior['arm_names'].tolist().index('prepared_frozen')]
                    prob_error = float(np.max(abs(p - output.probability)))
                    all_box_error = float(np.max(abs(prior['source_boxes'] - output.boxes)))
                xyz = raw_to_model_points(sensor_box[None,:3], target, view)[0]
                angle = np.deg2rad(WAYMO_ANGLES[view]) if target=='waymo' else 0.
                yaw_error = abs((sensor_box[6]+angle-both[1,6]+np.pi/2) % np.pi-np.pi/2)
                inverse_error = float(max(abs(xyz-both[1,:3]).max(), yaw_error, abs(sensor_box[3:6]-both[1,3:6]).max()))
                assert max(box_error, all_box_error, inverse_error) < check['box_max_abs_m_or_rad']
                assert prob_error < check['probability_max_abs'], (s,target,view,prob_error)
                cases.append(dict(source=s, target=target, camera=WAYMO_VIEWS[view] if target=='waymo' else 'target',
                                  both_winners_match=True, prob_error=prob_error, box_error=box_error,
                                  all_box_error=all_box_error, coordinate_error=inverse_error, seconds=time.monotonic()-tic))
                if not extra_checks:
                    default_args = dict(args, text='the vehicle alongside the roadway')
                    default_args.pop('seed')
                    with deployment():
                        novel = model.predict(**default_args)
                        explicit_zero = model.predict(**dict(default_args, seed=0))
                        repeated = model.predict(**args)
                    assert novel.shape == (7,) and np.isfinite(novel).all() and np.array_equal(novel,explicit_zero)
                    assert np.array_equal(repeated, sensor_box)
                    extra_checks = dict(novel_expression=True, repeated_prediction_identical=True, default_seed_reproducible=True)
                print('raw',s,target,view,'PASS',round(prob_error,9), flush=True)
            del model
            gc.collect()
            torch.cuda.empty_cache()
    # Record the first-party runtime closure separately from third-party packages and local weights.
    components = set()
    for module in tuple(sys.modules.values()):
        file = getattr(module, '__file__', None)
        if not file:
            continue
        path = Path(file).resolve()
        if path.suffix != '.py' or not path.is_relative_to(ROOT):
            continue
        rel = path.relative_to(ROOT)
        if (set(rel.parts) & {'upstream','runtime','stanza','env','site-packages'}
                or rel.parts[:2] == ('research','round159') or rel.parts[0] not in {'research','crossplatform'}):
            continue
        components.add(rel.as_posix())
    (HERE / 'FIRST_PARTY_RUNTIME.json').write_text(json.dumps(sorted(components),indent=2)+'\n')
    report = dict(status='PASS', cases=cases, n=len(cases), directions=6, new_source_training_runs=0,
                  parameter_updates=0, protect_read_guard=True, new_target_scoring=False, FINAL_access=False,
                  ordinary_prediction_parent=seal['sha256'], seconds=time.monotonic()-start, **extra_checks)
    (HERE / 'RAW_REPLAY.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print('PASS raw entry', len(cases), flush=True)


if __name__ == '__main__':
    main()
