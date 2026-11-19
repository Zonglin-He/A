"""Register disjoint ordinary source-head fit and selection observations."""

from collections import Counter, defaultdict, deque
from hashlib import sha256
import json
from itertools import combinations
import os
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / 'crossplatform/dataset_full/3eed_dataset'
PRIVATE = HERE / 'inputs_PRIVATE'
PLATFORMS = ('waymo', 'drone', 'quad')
PINS = ('crossplatform/ground.py', 'crossplatform/candidate_frontend.py',
        'crossplatform/v1_evidence.py', 'crossplatform/v1_decoder.py', 'crossplatform/CURRENT_METHOD.json',
        'crossplatform/single_frame_sources.json')


def read(path):
    return json.loads(Path(path).read_bytes())


def file_sha(path):
    checksum = sha256()
    with Path(path).open('rb') as stream:
        for value in iter(lambda: stream.read(1 << 20), b''):
            checksum.update(value)
    return checksum.hexdigest()


def save(path, value):
    draft = path.with_suffix(path.suffix + '.tmp')
    draft.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(draft, path)


def stable(*values):
    return sha256('|'.join(str(x) for x in values).encode()).hexdigest()


def protocol():
    return read(HERE / 'protocol.json')


def evaluation_rows():
    cfg = protocol()['evaluation']
    path = ROOT / cfg['manifest']
    evidence = read(ROOT / 'research/round159/RECORDING_INPUT_SEAL.json')
    assert file_sha(path) == cfg['sha256'] == evidence['input_sha256']
    assert not evidence['FINAL_access'] and not evidence['target_labels_used_for_selection']
    rows = read(path)
    assert len(rows) * 2 == cfg['records']
    assert dict(Counter(row['platform'] for row in rows)) == {p: cfg['per_target'] for p in PLATFORMS}
    return rows


def physical_key(platform, path):
    p = Path(path).parent.resolve()
    return platform, p.parent.name, p.name.split('_')[0] if platform == 'waymo' else p.name


def canonical(caption):
    return ' '.join(caption.replace(',', ' ,').split())


def possible_relation(caption):
    pattern = r'\b(?:left|right)(?:\s+side)?\s+of\s+(?!(?:the\s+)?(?:image|frame|picture|viewer|view|screen|center|middle)\b)'
    return bool(re.search(pattern, caption, re.IGNORECASE))


def metadata_candidates(platform, rec, barred, ordinal_seed):
    candidates = []
    for meta_path in sorted((DATA / platform / rec).glob('*/meta_info.json')):
        key = physical_key(platform, meta_path)
        if key in barred:
            continue
        if platform == 'waymo':
            fields = meta_path.parent.name.rsplit('_', 1)
            if len(fields) != 2 or fields[1] not in {'0','1','2','3','4'}:
                continue
            suffix = int(fields[1])
        else:
            suffix = 0
        parent = meta_path.parent
        point = parent / ('lidar.npy' if platform == 'waymo' else 'lidar.bin')
        image = parent / 'image.jpg'
        if not point.is_file() or not image.is_file():
            continue
        meta = read(meta_path)
        for i, item in enumerate(meta['ground_info']):
            text = canonical(item['caption'])
            if not text or len(text.split()) > 145 or not item.get('bbox_3d'):
                continue
            relative = meta_path.relative_to(DATA).as_posix()
            rank = stable(ordinal_seed, platform, relative, i)
            row = dict(platform=platform, recording=rec, suffix=suffix, caption=text,
                       object_index=i, pointpath=str(point), imagepath=str(image),
                       metapath=str(meta_path), natural_relation_candidate=possible_relation(text))
            candidates.append((rank, key, row))
    candidates.sort(key=lambda row: row[0])
    return candidates


def balanced(catalog, n, point_exclude, used_frames, prefer_relation):
    by_recording = {name: deque(row for row in catalog[name] if
                               not prefer_relation or row[2]['natural_relation_candidate'])
                    for name in sorted(catalog, key=lambda value: stable('order', value))}
    rows = []
    while by_recording and len(rows) < n:
        for name in tuple(by_recording):
            queue = by_recording[name]
            if not queue:
                by_recording.pop(name)
                continue
            while queue:
                _, key, row = queue.popleft()
                if key in used_frames:
                    continue
                point_sha = row.get('point_sha256') or file_sha(row['pointpath'])
                if point_sha in point_exclude:
                    continue
                used_frames.add(key)
                point_exclude.add(point_sha)
                rows.append(dict(row, point_sha256=point_sha))
                break
            if len(rows) == n:
                break
    return rows


def holdout_recordings(catalog, config):
    order = sorted(catalog, key=lambda name: stable(config['seed'], 'head-selection', name))
    frame_counts = {key: len({row[1] for row in catalog[key]}) for key in order}
    opt = config['source']
    if len(order) <= 8:
        # Small platforms have highly unequal recording sizes. Choose from
        # counts alone to preserve a recording split and fill the two budgets.
        choices = (group for n in range(1, len(order)) for group in combinations(order, n))
        total = sum(frame_counts.values())

        def coverage(keys):
            held = sum(frame_counts[k] for k in keys)
            return (min(total-held, opt['natural_fit']), min(held, opt['natural_calibration']),
                    -len(keys))

        return set(max(choices, key=coverage))
    holdout = set()
    available = 0
    for rec in order[:-1]:
        holdout.add(rec)
        available += frame_counts[rec]
        if available >= 2 * opt['natural_calibration'] and len(holdout) >= len(order) // 5:
            break
    return holdout


def unique_recording_catalog(catalog, forbidden):
    """Resolve duplicated point contents before allocating small-platform recordings."""
    chosen = {}
    used_hashes = set(forbidden)
    for name in sorted(catalog, key=lambda k: (-len(catalog[k]), stable(k))):
        accepted = []
        used_frames = set()
        for rank, key, row in catalog[name]:
            if key in used_frames:
                continue
            used_frames.add(key)
            digest = file_sha(row['pointpath'])
            if digest in used_hashes:
                continue
            used_hashes.add(digest)
            accepted.append((rank, key, dict(row, point_sha256=digest)))
        if accepted:
            chosen[name] = accepted
    return chosen


def build():
    config = protocol()
    target = evaluation_rows()
    blocked = {physical_key(r['platform'], r['metapath']) for r in target}
    point_exclude = {r['point_sha256'] for r in target}
    used = set()
    counts, checks, splits = {}, {}, {}
    PRIVATE.mkdir(mode=0o700)
    for platform in PLATFORMS:
        split_file = DATA / 'splits' / f'{platform}_train.txt'
        official = split_file.read_text().splitlines()
        by_recording = {}
        for rec in official:
            path = DATA / platform / rec
            assert path.resolve() == path.absolute() and path.parent == DATA / platform
            rows = metadata_candidates(platform, rec, blocked, config['seed'])
            if rows:
                by_recording[rec] = rows
        if platform != 'waymo':
            by_recording = unique_recording_catalog(by_recording, point_exclude)
            print(platform, 'content-distinct inventory counts by recording',
                  sorted(len(rows) for rows in by_recording.values()), flush=True)
        assert len(by_recording) >= 2
        holdout = holdout_recordings(by_recording, config)
        assert holdout and len(holdout) < len(by_recording)
        counts[platform] = {}
        for subset in ('fit','calibration'):
            target_n = config['source']['natural_' + subset]
            catalog = {k:v for k,v in by_recording.items() if (k in holdout) == (subset=='calibration')}
            rows = balanced(catalog, target_n // 2, point_exclude, used, True)
            rows += balanced(catalog, target_n-len(rows), point_exclude, used, False)
            if not rows:
                raise RuntimeError(f'{platform} {subset} has no disjoint observations')
            assert all(Path(row['metapath']).parent.parent.name in catalog for row in rows)
            path = PRIVATE / f'{platform}_{subset}.json'
            save(path, rows)
            checks[path.name] = file_sha(path)
            counts[platform][subset] = dict(expressions=len(rows),
                recordings=len({r['recording'] for r in rows}),
                likely_natural_relation=sum(r['natural_relation_candidate'] for r in rows))
        splits[platform] = file_sha(split_file)
        print(platform,counts[platform],flush=True)
    save(PRIVATE / 'SEAL.json', dict(
        files=checks, counts=counts, split_hashes=splits,
        protocol_sha256=file_sha(HERE/'protocol.json'), evaluation_sha256=config['evaluation']['sha256'],
        excluded_same_frame=True, excluded_same_point_content=True, head_holdout_by_recording=True,
        FINAL_access=False))


def source_rows(platform, subset):
    seal = read(PRIVATE/'SEAL.json')
    p = PRIVATE/f'{platform}_{subset}.json'
    assert seal['protocol_sha256']==file_sha(HERE/'protocol.json') and not seal['FINAL_access']
    assert seal['files'][p.name] == file_sha(p)
    assert seal['split_hashes'][platform] == file_sha(DATA/'splits'/f'{platform}_train.txt')
    return read(p)


if __name__ == '__main__':
    build()
