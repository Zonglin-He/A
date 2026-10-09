"""Private native cross-domain diagnostic. No model or label I/O on import."""
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/stvg_motivation_cross_domain_v6'
P1 = ROOT / 'artifacts/stvg_opd_paper_hc2_revision_v2'
PAPER = ROOT / 'artifacts/decota_paper_experiments_v1'
PYTHON = ROOT / '.conda/tubedetr/bin/python'
MODELS = ('tastvg', 'tubedetr')
DIRECTIONS = ('vidstg_to_hc2', 'hc2_to_vidstg')
SEED = 20261009
THRESHOLD = .5
CHECKPOINTS = {
    'tastvg': {
        'vidstg': ('checkpoints/TASTVG_VidSTG.pth', '5ab12c86363ef0ce0ee006c00fd11c6b659c3a9b2cb01a4f2c613efe22a2aa83'),
        'hc2': ('checkpoints/TASTVG_HCSTVG2.pth', '47d8f15841cd57e7bbf5a10e8bf23b1054d23b753e0becbd38a07f3dd60d5036'),
    },
    'tubedetr': {
        'vidstg': ('checkpoints/tubedetr_vidstg_res224_stride2.pth', '2802c66049b2e7986a826b4bbd291eb8521446cfba6fee03b56729438bf44213'),
        'hc2': ('checkpoints/tubedetr_hcstvg2_res224_stride2.pth', 'c646a8b7131bb28806c350e16479cd8bb31d4425ccb15129b26a6e4d38dd4222'),
    },
}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def status(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.' + str(os.getpid()) + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def direction_datasets(direction):
    assert direction in DIRECTIONS
    return ('vidstg', 'hc2') if direction == DIRECTIONS[0] else ('hc2', 'vidstg')


def guard(event, args):
    if event == 'open' and args and isinstance(args[0], (str, bytes)):
        name = os.fsdecode(args[0])
        if any(x in name for x in ('/annotations/', '/annos/', 'GT_LABEL',
                'labels_diagnostic', 'GT_SUBSET', 'test_annotations.json',
                'valv2_proc.json', 'val_v2.json', 'SCALAR_ROWS.json',
                '/ROWS.json', '/SUMMARY.json', 'QUADRANT_STATISTICS.json')):
            raise PermissionError('Native figure worker cannot open labels/scores: ' + name)
        # A separate diagnostic must never inspect an active OPD payload.
        if str(P1) in name and ('/predictions/' in name or name.endswith(('.pt', '.npz'))):
            raise PermissionError('Native figure worker cannot inspect OPD predictions: ' + name)


def verify():
    lock = read(BASE / 'RUNTIME_LOCK.json')
    for rel, value in {**lock['code'], **lock['metadata']}.items():
        assert sha(ROOT / rel) == value, rel
    assert sha(BASE / 'DESIGN_LOCK.json') == lock['design_sha256']
    assert shutil.disk_usage(ROOT).free > 8 * 2**30
    return lock


def verify_checkpoint(model, source):
    relative, expected = CHECKPOINTS[model][source]
    assert sha(ROOT / relative) == expected, relative
    return ROOT / relative


def priority_release():
    """Metadata only; never inspect OPD predictions or scores."""
    gp = P1/'P1_GPU_COMPLETION.json'
    gb = P1/'P1_PREDICTION_BARRIER.json'
    if not gp.exists() or not gb.exists():
        return None
    g, complete = read(gb), read(gp)
    assert complete['status'] == 'all_revised_P1_OPD_predictions_sealed'
    assert g['status'] == 'sealed' and g['adapted_arrivals'] == 41355 and not g['GT_read']
    assert g['all_deployment_arms_both_directions']
    assert set(g['barriers']) == {'stages/P1_hc2/PREDICTION_BARRIER.json', 'stages/P1_vidstg/PREDICTION_BARRIER.json'}
    for rel, h in g['barriers'].items():
        assert sha(P1/rel) == h
    return dict(P1_global_barrier_sha256=sha(gb), P1_GPU_completion_sha256=sha(gp),
        scope='GPU-release metadata only; no P1 prediction/scoring access', GT_read=False)


def seal_cell(model, direction):
    root = BASE / direction / model
    rows = read(BASE / direction / 'ROSTER.json')['rows']
    files = {}
    rh = sha(BASE / 'RUNTIME_LOCK.json')
    for row in rows:
        p = root / 'predictions' / f"{row['ordinal']:05}.json"
        rc = read(p.with_suffix('.receipt.json'))
        assert rc['sha256'] == sha(p) and rc['bytes'] == p.stat().st_size
        assert rc['runtime_lock_sha256'] == rh and not rc['GT_read']
        assert rc['qualification'] is False
        z = read(p)
        assert (z['model'], z['direction'], z['ordinal']) == (model, direction, row['ordinal'])
        assert z['frame_ids'] == z['box_frame_ids'] == row['frame_ids']
        assert z['input_binding_sha256'] == digest(row['input'])
        assert z['parameter_updates'] == 0 and not z['GT_read'] and z['native_parity']
        source, target = direction_datasets(direction)
        assert (z['source_dataset'], z['target_dataset']) == (source, target)
        assert z['checkpoint_sha256'] == CHECKPOINTS[model][source][1]
        assert z['runtime_lock_sha256'] == rh and z['qualification'] is False
        assert z['native_final_layer_only'] is True
        assert z['key_sha256'] == digest(row['key']) and z['parent_sha256'] == digest(row['source'])
        pixel = read(BASE / direction / 'common_pixels' / f"{row['ordinal']:05}.json")
        assert z['pixel_sha256'] == pixel['pixel_sha256'] and z['frame_ids'] == pixel['frame_ids']
        files[str(p.relative_to(BASE))] = dict(sha256=sha(p), bytes=p.stat().st_size,
            receipt_sha256=sha(p.with_suffix('.receipt.json')))
    assert len(files) == 128
    return files


def global_seal():
    verify()
    assert not (BASE / 'GLOBAL_PREDICTION_BARRIER.json').exists()
    records = {}
    for direction in DIRECTIONS:
        for model in MODELS:
            p = BASE / direction / model / 'PREDICTION_BARRIER.json'
            b = read(p)
            assert b['status'] == 'sealed' and b['cells'] == 128 and not b['GT_read']
            assert b['files'] == seal_cell(model, direction)
            records[str(p.relative_to(BASE))] = sha(p)
    write(BASE / 'GLOBAL_PREDICTION_BARRIER.json', dict(status='sealed', cells=512,
        backbones=list(MODELS), directions=list(DIRECTIONS), all_native_cells_sealed=True,
        barriers=records, runtime_lock_sha256=sha(BASE / 'RUNTIME_LOCK.json'),
        formal_GT_read=False, GT_read=False, time=time.time()))


def check_global_seal():
    verify()
    g = read(BASE / 'GLOBAL_PREDICTION_BARRIER.json')
    assert g['status'] == 'sealed' and g['cells'] == 512 and not g['GT_read']
    assert g['runtime_lock_sha256'] == sha(BASE / 'RUNTIME_LOCK.json')
    assert set(g['barriers']) == {f'{d}/{m}/PREDICTION_BARRIER.json' for d in DIRECTIONS for m in MODELS}
    for direction in DIRECTIONS:
        for model in MODELS:
            rel = f'{direction}/{model}/PREDICTION_BARRIER.json'
            b = read(BASE / rel)
            assert sha(BASE / rel) == g['barriers'][rel]
            assert b['files'] == seal_cell(model, direction)
            assert all(read((BASE / p).with_suffix('.receipt.json'))['time'] <= b['time'] <= g['time'] for p in b['files'])
    return g
