"""Register ordinary R160 observations and source-only validation inputs."""
from collections import defaultdict, deque
from functools import lru_cache
from pathlib import Path
import csv
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
WORK = ROOT / 'crossplatform/round162'
sys.path.insert(0, str(ROOT / 'research/round160'))
from common160 import DEV, EXTRA, REC, ORDER, PLATFORMS, rows, canonical, sha, array_sha, prediction_file, read


def atomic_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def sensor_row(row, text=None):
    permitted = ('platform', 'suffix', 'seed', 'pointpath', 'imagepath', 'metapath')
    return {**{name: row[name] for name in permitted},
            'caption': canonical(row['caption'] if text is None else text)}


@lru_cache(maxsize=1)
def target_rows():
    out = rows()
    assert len(out) == 10812
    return out


def check_frozen(index, ev):
    """Full candidate parity precedes any use of new geometry or target scoring."""
    import numpy as np
    with np.load(prediction_file(index), allow_pickle=False) as old:
        np.testing.assert_array_equal(ev.boxes, old['raw'])
        np.testing.assert_array_equal(ev.old_pca, old['prepared'][1])
        assert (ev.probability.argmax(), ev.old_winner) == tuple(old['winners'])
        assert array_sha(ev.points) == bytes(old['sampled']).hex()
        assert array_sha(ev.indices) == bytes(old['indices']).hex()


def calibration_inputs():
    """Draw fixed source VAL frames without intersecting the registered target content."""
    import ijson

    observed = target_rows()
    points = {e['point_sha'] for e in observed}
    images = {e['image_sha'] for e in observed}
    paths = {Path(e['r']['metapath']).resolve() for e in observed}
    out = []
    counts = {}
    for source in PLATFORMS:
        path = ROOT / f'research/round96/experiment/OFFICIAL_{source.upper()}_VAL_ORDER_PRIVATE.json'
        roster = read(path)
        selected = []
        by_recording = defaultdict(list)
        sort_key = lambda r: hashlib.sha256(f"lcib-source-val|{r['anno_index']}".encode()).digest()
        for row in sorted(roster, key=sort_key):
            if Path(row['original_meta_path']).resolve() not in paths and row['text_token_length'] <= 252:
                by_recording[row['scene']].append(row)
        assert len(by_recording)
        seen = set()
        groups = [deque(g) for _, g in sorted(by_recording.items())]
        while len(selected) < 64 and any(groups):
            for group in groups:
                if not group:
                    continue
                row = group.popleft()
                meta = Path(row['original_meta_path']).resolve()
                scan = meta.with_name('lidar.npy' if source == 'waymo' else 'lidar.bin')
                image = meta.with_name('image.jpg')
                if meta in seen or sha(scan) in points or sha(image) in images:
                    continue
                seen.add(meta)
                view = int(row['frame'].rsplit('_', 1)[1]) if source == 'waymo' else 0
                with meta.open('rb') as stream:
                    phrases = list(ijson.items(stream, 'ground_info.item.caption'))
                phrase = phrases[row['object_index']]
                if hashlib.sha256(phrase.encode()).hexdigest() != row['caption_sha256']:
                    raise ValueError('Source calibration expression locator changed')
                selected.append(dict(source=source, platform=source, suffix=view, seed=0,
                                     pointpath=str(scan), imagepath=str(image), metapath=str(meta),
                                     caption=canonical(phrase), val_GT=row['source_GT7_float32'],
                                     object_index=row['object_index'], recording=row['scene']))
                if len(selected) == 64:
                    break
        if len(selected) != 64:
            raise RuntimeError(f'Only {len(selected)} qualified calibration inputs for {source}')
        out.extend(selected)
        counts[source] = dict(rows=len(selected), scene_count=len({r['recording'] for r in selected}),
                              source_val_order_sha256=sha(path))
    atomic_json(WORK / 'SOURCE_VAL_PRIVATE.json', out)
    atomic_json(HERE / 'CALIBRATION_INPUTS.json', dict(source_own=counts, source_network_updates=0,
                excludes_R160_camera_and_point_content=True, FINAL_access=False,
                input_sha256=sha(WORK/'SOURCE_VAL_PRIVATE.json')))


def audit_panel():
    """Preselect successes and failures from the old measurement for mechanism audits only."""
    with (ROOT / 'research/round160/results/all_records.csv').open() as stream:
        old = {(r['cohort'], int(r['ordinal'])): float(r['r158_full']) for r in csv.DictReader(stream)}
    sample = defaultdict(list)
    for i, entry in enumerate(target_rows()):
        if entry['cohort'] == DEV:
            key = entry['cohort'], entry['ordinal']
            sample[entry['source'], entry['target'], old[key] > .25].append(i)
    panel = []
    for direction in ORDER:
        for good in (False, True):
            ids = sample[(*direction, good)]
            ids = sorted(ids, key=lambda i: hashlib.sha256(f'lcib-panel|{i}'.encode()).digest())
            chosen = []
            seen = set()
            for i in ids:
                digest = target_rows()[i]['point_sha']
                if digest not in seen:
                    chosen.append(i)
                    seen.add(digest)
                if len(chosen) == 16:
                    break
            if not chosen:
                raise ValueError('Missing old-error or old-success audit cases')
            panel.extend(chosen)
    atomic_json(WORK / 'AUDIT_PANEL_PRIVATE.json', sorted(panel))
    atomic_json(HERE / 'PANEL_LOCK.json', dict(rows=len(panel), old_model_stratification=True,
            distinct_point_content_per_direction_and_stratum=True, target_selection_used_for_mechanism_audit_only=True,
            fixed_model_and_rules_selection_uses_only_source_VAL=True, FINAL_access=False,
            input_sha256=sha(WORK / 'AUDIT_PANEL_PRIVATE.json')))


def smoke():
    import gc
    import torch
    from evidence162 import EvidenceGrounder
    facts = []
    cohort = target_rows()
    for source in PLATFORMS:
        model = EvidenceGrounder(source)
        for target in PLATFORMS:
            if source == target:
                continue
            index, e = next((i, ent) for i, ent in enumerate(cohort)
                            if ent['source'] == source and ent['target'] == target)
            row = sensor_row(e['r'])
            y = model.evidence_frame(row, verify=True)
            check_frozen(index, y)
            off = model.evidence_frame(row, rgb_mode='off', verify=True)
            assert not off.p2i_learned.any() and not off.i2p_learned.any()
            facts.append(dict(source=source, target=target, exact_seed_count=len(y.seeds),
                              roles=len(y.affinity), valid_RGB_point_rows=int(y.visible.sum()),
                              no_RGB_early_and_late_verified=True, forward_seconds=y.forward_seconds))
            print(facts[-1], flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    atomic_json(HERE/'EVIDENCE_SMOKE.json', dict(R160_box_sample_seed_and_winner_parity=len(facts),
                directions=facts, local_attention_output_rebuilt_exactly=True, target_GT_access=False))


if __name__ == '__main__':
    if sys.argv[1:] == ['smoke']:
        smoke()
    else:
        calibration_inputs()
        audit_panel()
