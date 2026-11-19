"""Reconstitute the ordinary R96/R97 Validation input roster from its filtered view."""

import ast
from collections import Counter
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
VIEW = ROOT / 'crossplatform/round96/data_view'
DATA = ROOT / 'crossplatform/dataset_full/3eed_dataset'
UPSTREAM_LOADER = ROOT / 'crossplatform/round3/p0/upstream/src/joint_det_dataset.py'
TRAIN = HERE.parent / '3eed_v2_fixed_slate_subject_ranking_20261119'
INPUT = HERE / 'ordinary_PRIVATE'
PLATFORMS = ('waymo', 'drone', 'quad')
DIRECTIONS = tuple((s, t) for s in PLATFORMS for t in PLATFORMS if s != t)
ARMS = ('v1', 'frozen_joint', 'frozen_subject', 'frozen_mismatch',
        'rank_subject', 'rank_joint', 'rank_mismatch')
VIEWS = ('F', 'FL', 'FR', 'SL', 'SR')
VALID_VAL = {'waymo': 3782, 'drone': 2792, 'quad': 2736}
EXCLUDED_FRAMES = {'waymo': 0, 'drone': 192, 'quad': 192}
PREFIX = 'studies/3eed_v2_ordinary_val_subject_ranking_20261119'


def read(path):
    return json.loads(Path(path).read_bytes())


def digest(path):
    h = sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def captions_for(frame):
    """Retain only original expression and category for historical loader eligibility."""
    import ijson

    with (frame / 'meta_info.json').open('rb') as stream:
        rows = []
        current = None
        for path, event, value in ijson.parse(stream):
            if path == 'ground_info.item' and event == 'start_map':
                current = {}
            elif path == 'ground_info.item' and event == 'end_map':
                assert set(current) == {'caption', 'class'}
                rows.append((current['caption'], current['class']))
                current = None
            elif path in ('ground_info.item.caption', 'ground_info.item.class') and event == 'string':
                current[path.rsplit('.', 1)[1]] = value
        return rows


def match_pattern():
    """Extract the two literal synonym maps from the pinned released loader."""
    dictionaries = {}
    for statement in ast.parse(UPSTREAM_LOADER.read_text()).body:
        if not isinstance(statement, ast.Assign):
            continue
        names = {node.id for node in statement.targets if isinstance(node, ast.Name)}
        for name in names & {'WAYMO_SYNONYMS', 'M3ED_SYNONYMS'}:
            dictionaries[name] = ast.literal_eval(statement.value)
    assert len(dictionaries) == 2
    return dictionaries


def view_frames():
    """Compare names only with the raw split before opening any ordinary data."""
    assert VIEW.is_dir() and (VIEW / 'splits').resolve() == DATA / 'splits'
    frames, differences, split_sha = {}, {}, {}
    for target in PLATFORMS:
        path = DATA / 'splits' / f'{target}_val.txt'
        split_sha[target] = digest(path)
        split_list = tuple(line.rstrip() for line in path.read_text().splitlines() if line.rstrip())
        assert len(split_list) == len(set(split_list))
        admitted, missing = [], 0
        for name in split_list:
            assert Path(name).name == name
            raw, filtered = DATA / target / name, VIEW / target / name
            if not raw.exists():
                assert not filtered.exists()
                continue
            assert filtered.is_dir() and not filtered.is_symlink()
            raw_names = {x.name for x in raw.iterdir() if x.is_dir()}
            accepted = [f for f in filtered.iterdir() if f.is_symlink() and f.is_dir()]
            names = {f.name for f in accepted}
            assert names <= raw_names
            assert not any(f.is_dir() and not f.is_symlink() for f in filtered.iterdir())
            for f in accepted:
                assert f.resolve() == raw / f.name
            missing += len(raw_names - names)
            admitted.extend(accepted)
        assert missing == EXCLUDED_FRAMES[target]
        frames[target], differences[target] = admitted, missing
    return frames, differences, split_sha


def canonical(text):
    return ' '.join(text.replace(',', ' ,').split())


def candidate_stamp(path):
    s = Path(path).stat()
    assert s.st_size > 0
    return [s.st_size, s.st_mtime_ns]


def register():
    ordinary, excluded, split_sha = view_frames()
    synonyms = match_pattern()
    INPUT.mkdir(mode=0o700)
    rows = {}
    counts, row_hashes = {}, {}
    files = {}
    for target in PLATFORMS:
        mapping = synonyms['WAYMO_SYNONYMS' if target == 'waymo' else 'M3ED_SYNONYMS']
        positions, order = [], []
        rejected, raw_annos = 0, 0
        for frame in ordinary[target]:
            actual = frame.resolve()
            point = actual / ('lidar.npy' if target == 'waymo' else 'lidar.bin')
            image, meta = actual / 'image.jpg', actual / 'meta_info.json'
            stamps = [candidate_stamp(p) for p in (point, image, meta)]
            relative_meta = (frame / 'meta_info.json').relative_to(VIEW).as_posix()
            suffix = int(frame.name.rsplit('_', 1)[-1]) if target == 'waymo' else 0
            assert 0 <= suffix < len(VIEWS) and meta.parents[2] == DATA / target
            for index, (raw, cat) in enumerate(captions_for(frame)):
                raw_annos += 1
                caption, label = canonical(raw), cat.lower()
                if not any(re.search(r'\b' + re.escape(w) + r'\b', f' {caption} ', re.I)
                           for w in mapping.get(label, [label])):
                    rejected += 1
                    continue
                identity = f'{target}|{relative_meta}|{index}'
                seed = int(sha256(('R97point|' + identity).encode()).hexdigest()[:16], 16)
                positions.append(dict(platform=target, recording=frame.parent.name, suffix=suffix,
                                      pointpath=str(point), metapath=str(meta), imagepath=str(image),
                                      caption=caption, object_index=index, seed=seed, stats=stamps))
                order.append((str(VIEW / relative_meta), index, sha256(raw.encode()).hexdigest(), label))
        assert len(positions) == VALID_VAL[target] and raw_annos - rejected == len(positions)
        assert rejected == (12 if target == 'waymo' else 0)
        p = INPUT / f'{target}.json'
        atomic(p, positions)
        rows[target] = positions
        row_hashes[target] = sha256(json.dumps(order, separators=(',', ':')).encode()).hexdigest()
        counts[target] = dict(raw_ordinary_annotations=raw_annos, eligible=len(positions),
                             platform_category_span_excluded=rejected, frames=len(ordinary[target]),
                             recordings=len({x['recording'] for x in positions}))
        files[target] = digest(p)
    assert sum(map(len, rows.values())) * 2 == 18620
    seal = dict(expected_direction_records=18620, unique_target_records=9310, counts=counts, files=files,
                val_split_sha256=split_sha, loader_order_reconstruction_sha256=row_hashes,
                historical_upstream_loader_sha256=digest(UPSTREAM_LOADER), excluded_frame_name_counts=excluded,
                sampling='R97point SHA256 of platform, filtered-view meta relative path, annotation slot',
                variant='R96 pre-filtered data_view, original 3EED annotation eligibility rule',
                accessed_FINAL_payload=False, accessed_FINAL_manifest=False)
    atomic(INPUT / 'SEALED.json', seal)
    print(json.dumps({k: seal[k] for k in ('counts', 'excluded_frame_name_counts',
                                          'expected_direction_records')}, ensure_ascii=False), flush=True)


@lru_cache(maxsize=1)
def ordinary():
    config = read(HERE / 'PROTOCOL.json')
    seal = read(INPUT / 'SEALED.json')
    assert digest(INPUT / 'SEALED.json') == config['input_roster_seal_sha256']
    assert seal['expected_direction_records'] == 18620 and not seal['accessed_FINAL_payload']
    assert not seal['accessed_FINAL_manifest']
    for p, h in seal['val_split_sha256'].items():
        assert digest(DATA / 'splits' / f'{p}_val.txt') == h
    result = {}
    for target, n in VALID_VAL.items():
        path = INPUT / f'{target}.json'
        assert digest(path) == seal['files'][target]
        data = read(path)
        assert len(data) == n and all(r['platform'] == target for r in data)
        result[target] = data
    return result


def frame(row):
    return dict(target=row['platform'], points=row['pointpath'], rgb=row['imagepath'],
                calibration=row['metapath'], text=row['caption'], seed=row['seed'],
                waymo_view=VIEWS[row['suffix']])


def check_input(row):
    path = Path(row['metapath'])
    link = VIEW / row['platform'] / row['recording'] / path.parent.name
    assert link.is_symlink() and link.resolve() == path.parent
    assert all(candidate_stamp(row[key]) == stat for key, stat in
               zip(('pointpath', 'imagepath', 'metapath'), row['stats']))
    assert path.parents[2] == DATA / row['platform']


if __name__ == '__main__':
    register()
