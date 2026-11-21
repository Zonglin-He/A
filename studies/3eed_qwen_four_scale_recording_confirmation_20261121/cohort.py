"""Seal new observations using only eligible rosters and original expressions."""
from collections import Counter, defaultdict, deque
from functools import lru_cache
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
QWEN = HERE.parent / '3eed_frozen_qwen_metric_grounding_20261120'
PREVIOUS = HERE.parent / '3eed_qwen_four_scale_region_grounding_20261121'
ORDINARY = HERE.parent / '3eed_v2_ordinary_val_subject_ranking_20261119'
TRAIN = HERE.parent / '3eed_v1_supervised_instance_grounding_20261119'
PRIVATE = HERE / 'private'
OUT = HERE / 'results'
PROTOCOL = HERE / 'PROTOCOL.json'
LOCK = HERE / 'METHOD_LOCK.json'

sys.path[:0] = [str(p) for p in (PREVIOUS, QWEN, ORDINARY, TRAIN, ROOT)]
from paths import PLATFORMS, digest, key, read, write
from scope183 import DATA, VIEWS, canonical, check_input, ordinary


def checked_lock():
    lock = read(LOCK)
    for relative, checksum in lock['files'].items():
        assert digest(ROOT / relative) == checksum, relative
    assert lock['configuration'] == 'C_E_Qsize'
    return digest(LOCK)


def read_panel():
    seal = read(OUT / 'panel_seal.json')
    path = PRIVATE / 'panel.json'
    assert (digest(path), digest(PROTOCOL), checked_lock()) == (
        seal['panel'], seal['protocol'], seal['method_lock'])
    panel = read(path)
    assert len(panel) == seal['frame_count']
    return panel


def identities(target, path):
    frame = Path(path).parent
    timestamp = frame.name.rsplit('_', 1)[0] if target == 'waymo' else frame.name
    return frame.parent.name, (target, frame.parent.name, timestamp)


@lru_cache(maxsize=None)
def captions(path):
    import ijson

    with Path(path).open('rb') as stream:
        return tuple(ijson.items(stream, 'ground_info.item.caption'))


def register():
    from scope import evaluation_rows

    lock_sha = checked_lock()
    cfg = read(PROTOCOL)
    old_path = QWEN / 'private/panel.json'
    assert digest(old_path) == read(QWEN / 'results/panel_seal.json')['panel']
    development = read(old_path)
    used_frames, used_points, used_images = set(), set(), set()
    dev_recordings = defaultdict(set)
    for row in development:
        target = row['target']
        recording, frame = identities(target, row['calibration'])
        dev_recordings[target].add(recording)
        used_frames.add(frame)
        used_points.add(row['point_sha'])
        used_images.add(row['rgb_sha'])

    validation = ordinary()
    train = evaluation_rows()
    counts = Counter((r['platform'] for r in train))
    assert counts == {t: 512 for t in PLATFORMS}
    inventory, roster = {}, []
    hash_file = lru_cache(maxsize=None)(digest)
    seed = cfg['cohort_seed']

    for panel, quotas in cfg['panels'].items():
        for target, n in quotas.items():
            val_recordings = {r['recording'] for r in validation[target]}
            if panel == 'validation':
                source = list(enumerate(validation[target]))
            else:
                source = [(None, r) for r in train if r['platform'] == target]
                official_train = set((DATA / 'splits' / f'{target}_train.txt').read_text().splitlines())
            groups = defaultdict(lambda: defaultdict(list))
            for ordinal, row in source:
                recording, frame = identities(target, row['metapath'])
                if panel == 'validation':
                    if target == 'waymo' and recording in dev_recordings[target]:
                        continue
                    assert recording == row['recording']
                else:
                    assert recording in official_train and recording not in val_recordings | dev_recordings[target]
                    assert Path(row['metapath']).parent.parent.parent == DATA / target
                if frame not in used_frames:
                    groups[recording][frame].append((ordinal, row))
            queues = {r: deque(sorted(group, key=lambda f: key(seed, 'frame', f)))
                      for r, group in groups.items()}
            order = sorted(queues, key=lambda r: key(seed, 'recording', target, r))
            members = []
            while len(members) < n:
                prior = len(members)
                for rec in order:
                    while queues[rec]:
                        frame = queues[rec].popleft()
                        if frame in used_frames:
                            continue
                        options = sorted(groups[rec][frame], key=lambda p: key(
                            seed, 'utterance', p[1]['metapath'].split('/')[-2],
                            p[1].get('object_index', p[1].get('id'))))
                        for ordinal, row in options:
                            point = hash_file(row['pointpath'])
                            image = hash_file(row['imagepath'])
                            if point in used_points or image in used_images:
                                continue
                            original = captions(row['metapath'])
                            if panel == 'validation':
                                check_input(row)
                                slot = row['object_index']
                                sampling_seed = row['seed']
                            else:
                                assert point == row['point_sha256'] and image == row['image_sha256']
                                hits = [i for i, text in enumerate(original) if canonical(text) == canonical(row['caption'])]
                                assert len(hits) == 1, 'The sealed TRAIN expression must match exactly one original slot'
                                slot = hits[0]
                                identity = f"{target}|{rec}/{Path(row['metapath']).parent.name}/meta_info.json|{slot}"
                                sampling_seed = int(key('R97point', identity)[:16], 16)
                            text = original[slot]
                            assert canonical(text) == canonical(row['caption'])
                            suffix = row['suffix'] if target == 'waymo' else 0
                            used_frames.add(frame)
                            used_points.add(point)
                            used_images.add(image)
                            members.append(dict(uid=key(seed, 'case', frame), target=target, panel=panel,
                                separation='new_recording' if rec not in dev_recordings[target] else 'same_recording',
                                recording=key(seed, 'recording-id', target, rec), ordinal=ordinal, slot=slot,
                                rgb=row['imagepath'], points=row['pointpath'], calibration=row['metapath'],
                                suffix=suffix, waymo_view=VIEWS[suffix], text=text,
                                seed=sampling_seed, point_sha=point, rgb_sha=image))
                            break
                        else:
                            continue
                        break
                    if len(members) == n:
                        break
                if len(members) == prior:
                    raise RuntimeError(f'Insufficient unique observations in {panel}/{target}')
            roster.extend(members)
            inventory[f'{panel}/{target}'] = dict(frames=n, available_recordings=len(groups),
                validation_recordings=len(val_recordings), unused_validation_recordings=len(val_recordings-dev_recordings[target]),
                recordings=len({r['recording'] for r in members}),
                development_recordings=len(dev_recordings[target]),
                recordings_shared_with_development=len({r['recording'] for r in members if r['separation']=='same_recording'}),
                frames_by_separation=dict(Counter(r['separation'] for r in members)))
    assert len({r['uid'] for r in roster}) == len(roster)
    assert len(used_points) == len(used_images) == len(development) + len(roster)
    write(PRIVATE / 'panel.json', roster)
    write(OUT / 'panel_seal.json', dict(protocol=digest(PROTOCOL), method_lock=lock_sha,
          panel=digest(PRIVATE / 'panel.json'), frame_count=len(roster), coverage=inventory,
          excluded_development_panel=digest(old_path), duplicate_points=0, duplicate_images=0,
          selection_uses_box_labels=False, FINAL_access=False))
    print(inventory, flush=True)


if __name__ == '__main__':
    register()
