"""R51 raw-safe CPU preparation. No import-time sample or cache access.

Only registered raw RGB/XYZ/camera, original source boxes/scores and sealed
pure-text graphs may be opened. External vision outputs and mask-derived
geometry are deliberately absent. Hybrid collectors run in another process.
The root owns execution authorization; this file does not create grants.
"""
from pathlib import Path
import argparse
import collections
import datetime
import hashlib
import json
import sys
import time
import traceback
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from instances51 import build_instances, array_sha256, DEFAULT_CONFIG
from attributes51 import measure_attributes
sys.path.insert(0, str(ROOT / 'crossplatform/round7'))
from camera_projection import CameraSpec, read_camera_allowlist

DEV_MANIFEST = 'research/round24/input/ALL336_PRIVATE.json'
FULL_MANIFEST = 'research/round8/experiment/full5912_gpu_inputs.json'
SOURCE_PROFILES = {
    'R24_DEV': ('crossplatform/round24/capture336_attempt001', 'samples_PRIVATE', 'boxes', 'scores'),
    'R8_FULL': ('crossplatform/round8/full5912_attempt001', '', 'source_boxes_raw6', 'source_text_scores256'),
}
PURE_TEXT_ROOTS = (
    'crossplatform/round50/full5912_attempt001/text/cases_PRIVATE',
    'crossplatform/round23/text272_attempt001/cases_PRIVATE',
    'crossplatform/round24/textM64_attempt001/cases_PRIVATE',
    'crossplatform/round51/text/cases_PRIVATE',
)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _object(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError('Duplicate JSON key rejected')
        out[key] = value
    return out


def read(path):
    return json.loads(Path(path).read_text(), object_pairs_hook=_object,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def path_in_workspace(value):
    p = Path(value)
    p = (ROOT / p).resolve() if not p.is_absolute() else p.resolve()
    if not p.is_relative_to(ROOT):
        raise ValueError('Program/registration/cache path outside workspace')
    if any(x in p.as_posix() for x in ('/round46/final/', 'FINAL384', '/round46/final384')):
        raise ValueError('Closed FINAL-specific artifacts forbidden')
    return p


def _key(row):
    return tuple(row[k] for k in ('imagepath', 'pointpath', 'metapath', 'caption'))


class ReadLedger:
    def __init__(self, out):
        self.path = out / 'preparation_IO.json'
        self.rows = []

    def bind(self, path, role, expected=None, data_id=None, selected_fields=None):
        p = Path(path).resolve()
        # The ledger is private because it contains raw input/cache paths.
        actual = sha(p)
        item = {'role': role, 'path_PRIVATE': str(p), 'sha256': actual,
                'data_id_PRIVATE': data_id,
                'selected_fields': selected_fields, 'bytes_hashed': p.stat().st_size,
                'status': 'HASH_BOUND_BEFORE_PARSE'}
        self.rows.append(item)
        dump(self.path, {'schema': 'R51_raw_safe_preparation_IO_v1', 'files': self.rows,
                         'metadata_file_hash_is_not_annotation_parsing': True})
        if expected is not None and actual != expected:
            raise ValueError('Input hash drift: ' + role)
        return actual


def context(args):
    pp = path_in_workspace(args.plan)
    ap = path_in_workspace(args.authorization)
    auth = read(ap)
    if auth.get('state') != 'ROOT_AUTHORIZED_R51_RAW_SAFE_PREPARATION':
        raise ValueError('Explicit R51 raw-safe preparation grant required')
    if auth.get('plan_sha256') != sha(pp):
        raise ValueError('Plan authorization hash mismatch')
    plan = read(pp)
    if plan.get('state') != 'ROOT_LOCKED_R51_RAW_SAFE_PREPARATION':
        raise ValueError('Locked root plan required')
    if plan.get('n') not in (336, 5912) or plan.get('no_GT_or_external_vision_outputs') is not True:
        raise ValueError('Scope mismatch')
    expected_manifest = DEV_MANIFEST if plan['n'] == 336 else FULL_MANIFEST
    if plan['manifest'] != expected_manifest:
        raise ValueError('Only registered old DEV336/general full5912 are allowed')
    if plan['source_profile'] not in SOURCE_PROFILES:
        raise ValueError('Unknown source profile')
    if plan['n'] == 5912 and plan['source_profile'] != 'R8_FULL':
        raise ValueError('Full cohort must preserve original R8 source')
    profile = SOURCE_PROFILES[plan['source_profile']]
    if plan['source_root'] != profile[0]:
        raise ValueError('Source cache may not be substituted')
    required_programs = [__file__, ROOT / 'crossplatform/round51/instances51.py',
                         ROOT / 'crossplatform/round51/attributes51.py',
                         ROOT / 'crossplatform/round7/camera_projection.py']
    for p in required_programs:
        rel = Path(p).resolve().relative_to(ROOT).as_posix()
        if plan['programs_sha256'].get(rel) != sha(p):
            raise ValueError('Required program is not hash locked: ' + rel)
    for rel, digest in plan['programs_sha256'].items():
        if sha(path_in_workspace(rel)) != digest:
            raise ValueError('Program drift')
    # Input map is restricted to registrations, source seals, pure text seals or
    # explicit pure-text/source-hash indexes. It must not be a cache discovery list.
    allowed_inputs = {plan['manifest'], plan['source_root'] + '/SEALED.json'}
    if plan['graph_mode'] == 'full_text_join':
        if plan['full_text_manifest'] != FULL_MANIFEST or plan['full_text_root'] != 'crossplatform/round50/full5912_attempt001/text':
            raise ValueError('Only registered pure-text full graph join allowed')
        allowed_inputs.update([plan['full_text_manifest'], plan['full_text_root'] + '/SEALED.json'])
    elif plan['graph_mode'] == 'explicit_index':
        allowed_inputs.add(plan['graph_index'])
    else:
        raise ValueError('No implicit graph/cache lookup')
    if plan.get('raw_input_hash_index') is not None:
        allowed_inputs.add(plan['raw_input_hash_index'])
    if set(plan['inputs_sha256']) != allowed_inputs:
        raise ValueError('Unexpected input discovery/caches forbidden')
    for rel, digest in plan['inputs_sha256'].items():
        if sha(path_in_workspace(rel)) != digest:
            raise ValueError('Registered input drift')
    if plan['source_seal_sha256'] != plan['inputs_sha256'][plan['source_root'] + '/SEALED.json']:
        raise ValueError('Source seal binding mismatch')
    rows = read(path_in_workspace(plan['manifest']))
    if not isinstance(rows, list) or len(rows) != plan['n'] or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Cohort size/identity mismatch')
    if any(r['platform'] not in ('drone', 'quad') for r in rows) or len({_key(r) for r in rows}) != len(rows):
        raise ValueError('Unexpected platform/duplicate exact input')
    if args.limit is not None:
        if args.limit != 2 or not plan.get('allow_engineering_limit') or auth.get('engineering_limit') != 2:
            raise ValueError('Engineering --limit requires separate explicit root scope')
        # One per platform, using only immutable input hash; not method/GT result.
        chosen = []
        for platform in ('drone', 'quad'):
            rr = [r for r in rows if r['platform'] == platform]
            chosen.append(min(rr, key=lambda r: hashlib.sha256(json.dumps(_key(r), ensure_ascii=False).encode()).hexdigest()))
        rows = chosen
    out = path_in_workspace(plan['output_root'])
    if not out.is_relative_to(ROOT / 'crossplatform/round51'):
        raise ValueError('R51 output root required')
    if out.exists():
        raise FileExistsError('Attempt exists; no overwrite, retry or resume')
    return plan, rows, out, pp, ap


def graph_routes(plan, rows, ledger):
    if plan['graph_mode'] == 'full_text_join':
        manifest = path_in_workspace(plan['full_text_manifest'])
        ledger.bind(manifest, 'REGISTERED_PURE_TEXT_JOIN_INPUT', plan['inputs_sha256'][plan['full_text_manifest']])
        full_rows = read(manifest)
        index = {_key(r): r['id'] for r in full_rows}
        if len(index) != 5912:
            raise ValueError('Full text join must be exact three paths plus caption')
        folder = path_in_workspace(plan['full_text_root'])
        sp = folder / 'SEALED.json'
        ledger.bind(sp, 'PURE_TEXT_GRAPH_SEAL', plan['inputs_sha256'][plan['full_text_root'] + '/SEALED.json'])
        seal = read(sp)
        result = {}
        for row in rows:
            full_id = index[_key(row)]
            rel = 'cases_PRIVATE/' + full_id + '.json'
            result[row['id']] = {'path': folder / rel, 'sha256': seal['files'][rel],
                                 'caption_sha256': hashlib.sha256(row['caption'].encode()).hexdigest()}
        return result
    p = path_in_workspace(plan['graph_index'])
    ledger.bind(p, 'EXPLICIT_PURE_TEXT_GRAPH_INDEX', plan['inputs_sha256'][plan['graph_index']])
    index = read(p)
    if not isinstance(index, dict) or set(index) != {r['id'] for r in read(path_in_workspace(plan['manifest']))}:
        raise ValueError('Explicit graph index must exactly cover the registered cohort')
    result = {}
    for row in rows:
        record = index[row['id']]
        if set(record) != {'path', 'sha256', 'caption_sha256'}:
            raise ValueError('Only pure text path/hash/caption binding fields allowed')
        p = path_in_workspace(record['path'])
        if not any(p.is_relative_to(ROOT / root) for root in PURE_TEXT_ROOTS):
            raise ValueError('Graph path is not a permitted pure-text graph directory')
        if record['caption_sha256'] != hashlib.sha256(row['caption'].encode()).hexdigest():
            raise ValueError('Explicit text graph caption mismatch')
        result[row['id']] = dict(record, path=p)
    return result


def load_source(row, plan, source_seal, ledger):
    folder, sub, box_key, score_key = SOURCE_PROFILES[plan['source_profile']]
    relative = ((sub + '/') if sub else '') + row['id'] + '.npz'
    path = ROOT / folder / relative
    digest = ledger.bind(path, 'ORIGINAL_GROUNDER_SOURCE_ONLY_ARRAYS',
                         source_seal['files'][relative], row['id'], [box_key, score_key])
    # np.load is lazy; unrelated ZIP entries (including old external outputs in
    # the R8 container) are not materialized or inspected.
    with np.load(path, allow_pickle=False) as z:
        boxes = z[box_key].copy()
        scores = z[score_key].copy()
    if boxes.shape != (256, 6) or scores.shape != (256,) or not np.isfinite(boxes).all() or not np.isfinite(scores).all():
        raise ValueError('Original 256 source boxes/scores required')
    return boxes, scores, digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', required=True)
    p.add_argument('--authorization', required=True)
    p.add_argument('--limit', type=int)
    args = p.parse_args()
    plan, rows, out, plan_path, auth_path = context(args)
    out.mkdir(parents=True, exist_ok=False)
    for sub in ('arrays_PRIVATE', 'cases_PRIVATE', 'attributes_PRIVATE'):
        (out / sub).mkdir()
    ledger = ReadLedger(out)
    started = time.perf_counter()
    done = []
    counts = collections.Counter()
    counts['requested_cohort_n'] = plan['n']
    counts['processed_n'] = 0
    source_seal = read(ROOT / plan['source_root'] / 'SEALED.json')
    try:
        ledger.bind(ROOT / plan['manifest'], 'REGISTERED_COHORT', plan['inputs_sha256'][plan['manifest']])
        ledger.bind(ROOT / plan['source_root'] / 'SEALED.json', 'ORIGINAL_GROUNDER_SOURCE_SEAL', plan['source_seal_sha256'])
        routes = graph_routes(plan, rows, ledger)
        raw_hashes = None
        if plan.get('raw_input_hash_index') is not None:
            path = path_in_workspace(plan['raw_input_hash_index'])
            ledger.bind(path, 'ORIGINAL_RAW_INPUT_HASH_INDEX_ONLY', plan['inputs_sha256'][plan['raw_input_hash_index']])
            raw_hashes = read(path)
            allowed = {'image_sha256', 'point_sha256', 'metadata_sha256'}
            if not isinstance(raw_hashes, dict) or any(set(raw_hashes[r['id']]) != allowed for r in rows):
                raise ValueError('Raw hash index must contain only original input hashes')
        for number, row in enumerate(rows):
            tick = time.perf_counter()
            data_id = row['id']
            # No row fields are interpreted as instructions; file paths come
            # from the hash-locked cohort only, never from caption/graph strings.
            bindings = {}
            for field, role, digest_key in (
                    ('imagepath', 'ORIGINAL_RGB', 'image_sha256'),
                    ('pointpath', 'ORIGINAL_POINTCLOUD', 'point_sha256'),
                    ('metapath', 'ORIGINAL_METADATA_CAMERA_WHITELIST_ONLY', 'metadata_sha256')):
                expected = raw_hashes[data_id][digest_key] if raw_hashes is not None else None
                bindings[digest_key] = ledger.bind(row[field], role, expected, data_id,
                    ['image_intrinsic', 'image_extrinsic', 'camera_distortion'] if field == 'metapath' else None)
            with Image.open(row['imagepath']) as im:
                rgb = np.asarray(im.convert('RGB')).copy()
            pcpath = Path(row['pointpath'])
            if pcpath.suffix == '.npy':
                raw = np.load(pcpath, allow_pickle=False)
            elif pcpath.suffix == '.bin':
                raw = np.fromfile(pcpath, np.float32).reshape(-1, 4)
            else:
                raise ValueError('Only registered original .npy / float32 Nx4 .bin point format allowed')
            if raw.ndim != 2 or raw.shape[1] < 3 or not len(raw):
                raise ValueError('Original Nx3+ point format required')
            raw = np.ascontiguousarray(raw[:, :3], np.float32)
            fields = read_camera_allowlist(row['metapath'])
            camera = CameraSpec.from_camera_fields(row['platform'], fields,
                waymo_view_suffix=0, missing_distortion_policy='assume_zero_declared')
            boxes, scores, source_sha = load_source(row, plan, source_seal, ledger)
            graph_record = routes[data_id]
            ledger.bind(graph_record['path'], 'PURE_TEXT_GRAPH', graph_record['sha256'], data_id)
            graph = read(graph_record['path'])
            receipt, arrays = build_instances(raw_points=raw, source_boxes=boxes,
                image_size_wh=(rgb.shape[1], rgb.shape[0]), camera_spec=camera)
            if arrays['source_boxes'].dtype != boxes.dtype or arrays['source_boxes'].shape != boxes.shape or arrays['source_boxes'].tobytes() != boxes.tobytes():
                raise AssertionError('Source box bytes changed')
            arrays['source_scores'] = scores.copy()
            attrs = measure_attributes(rgb=rgb, raw_points=raw, instance_arrays=arrays, text_graph=graph)
            ap = out / 'arrays_PRIVATE' / (data_id + '.npz')
            np.savez_compressed(ap, **arrays)
            attr_path = out / 'attributes_PRIVATE' / (data_id + '.json')
            dump(attr_path, attrs)
            receipt.update(id_PRIVATE=data_id, platform=row['platform'],
                caption_sha256=hashlib.sha256(row['caption'].encode()).hexdigest(),
                raw_bindings=bindings, historical_raw_hash_bridge=raw_hashes is not None,
                source_cache_sha256=source_sha, source_profile=plan['source_profile'],
                source_score_copy_not_used_in_geometry=True,
                pure_text_graph_sha256=graph_record['sha256'],
                arrays_sha256=sha(ap), attributes_sha256=sha(attr_path),
                seconds=time.perf_counter() - tick)
            dump(out / 'cases_PRIVATE' / (data_id + '.json'), receipt)
            counts['processed_n'] += 1
            counts['queries'] += len(boxes)
            counts['instances'] += receipt['n_instances']
            counts['strict_mapped_queries'] += receipt['n_mapped_queries']
            counts['unknown_queries'] += len(boxes) - receipt['n_mapped_queries']
            counts['visible_points'] += receipt['n_visible_points']
            counts['visible_unassigned'] += receipt['n_visible_unassigned']
            counts['visible_ambiguous'] += receipt['n_visible_ambiguous']
            counts['instances_color_unknown'] += sum(x['color_status'].startswith('UNKNOWN') for x in attrs['instances'])
            done.append(data_id)
            dump(out / 'PROGRESS.json', {'completed_n': len(done), 'requested_n': len(rows),
                                        'seconds': time.perf_counter() - started})
            if number == 0 or (number + 1) % 16 == 0:
                print('R51_RAW_SAFE_PREPARE', number + 1, len(rows), flush=True)
        dump(out / 'COUNTS.json', dict(counts))
        dump(out / 'COSTS.json', {'seconds': time.perf_counter() - started,
            'new_grounder_forwards': 0, 'new_detector_forwards': 0,
            'new_VLM_forwards': 0, 'new_DINO_forwards': 0, 'optimizer_steps': 0,
            'graph_parser_forwards': 0, 'CPU_point_instance_frames': len(done)})
        dump(out / 'SEALED.json', {'state': 'R51_RAW_SAFE_INSTANCE_ATTRIBUTE_PREPARATION_COMPLETE_NO_GT',
            'n': len(done), 'registered_n': plan['n'], 'engineering_limit': args.limit,
            'plan_sha256': sha(plan_path), 'authorization_sha256': sha(auth_path),
            'files': {path.relative_to(out).as_posix(): sha(path)
                      for path in sorted(out.rglob('*')) if path.is_file()},
            'nfiles': sum(path.is_file() for path in out.rglob('*')),
            'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'original_source_boxes_unchanged_bytes': True,
            'external_vision_outputs_or_mask_derived_cache_read': False,
            'pure_text_graph_only_reused': True, 'task_GT_access': False,
            'main_association': 'conditional_known_exclusive_support',
            'subprob_unknown_mass_diagnostic_only': True})
    except BaseException as error:
        dump(out / 'FAILED.json', {'state': 'TECHNICAL_FAILURE_PRESERVED_NO_AUTOMATIC_RETRY',
            'error_type': type(error).__name__, 'error': str(error), 'completed_n': len(done),
            'seconds': time.perf_counter() - started, 'traceback_PRIVATE': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
