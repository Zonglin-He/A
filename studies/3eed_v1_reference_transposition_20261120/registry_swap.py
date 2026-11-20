"""Observation-only roster and fixed references to the audited v1 study."""
from collections import defaultdict
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PARENT = HERE.parent / '3eed_v1_joint_endpoint_coverage_20261120'
ORD = HERE.parent / '3eed_v2_ordinary_val_subject_ranking_20261119'
sys.path[:0] = [str(PARENT), str(ORD)]
from collect import SOURCES, digest, read, save
from capture_roles import PANELS
from predict import frame
from scope183 import ordinary

WORK = HERE / 'private'
OBS = WORK / 'observed'
PRED = WORK / 'predictions'
OUT = HERE / 'results'
SPEC = HERE / 'PROTOCOL.json'
MARKER = re.compile(r'\b(?:left|right)(?:\s+side)?\s+of\b', re.I)


def token(value):
    return sha256(('lr-transplant-v1|' + str(value)).encode()).hexdigest()


@lru_cache(maxsize=1)
def records():
    old = []
    used_paths = set()
    for panel, path in PANELS.items():
        for index, entry in enumerate(read(path)):
            row = entry['row']
            used_paths.add(Path(row['pointpath']).resolve())
            for src in SOURCES:
                if panel == 'source_proxy' or src != row['platform']:
                    old.append(dict(panel=panel, case=index, source=src, frame=frame(row)))
    assert len(old) == 501
    # Identical physical scans can have different paths and multiple annotations or camera views.
    past = {digest(path) for path in used_paths}
    seen = set(past)
    clouds = {}
    def signature(path):
        p = Path(path).resolve()
        if p not in clouds:
            clouds[p] = digest(p)
        return clouds[p]
    new = []
    for target, rows in ordinary().items():
        for panel in ('ordinary_uniform', 'ordinary_relation'):
            buckets = defaultdict(list)
            for ix, row in enumerate(rows):
                if panel == 'ordinary_relation' and not MARKER.search(row['caption']):
                    continue
                buckets[row['recording']].append(ix)
            for rec, ids in buckets.items():
                ids.sort(key=lambda i: token((panel, target, rec, i)), reverse=True)
            recs = sorted(buckets, key=lambda rec: token((panel, target, rec)))
            picked = []
            while len(picked) < 48:
                before = len(picked)
                for rec in recs:
                    ids = buckets[rec]
                    while ids:
                        ix = ids.pop()
                        key = signature(rows[ix]['pointpath'])
                        if key not in seen:
                            seen.add(key)
                            picked.append(ix)
                            break
                    if len(picked) == 48:
                        break
                if len(picked) == before:
                    break
            for ix in picked:
                for src in SOURCES:
                    if src != target:
                        new.append(dict(panel=panel, case=ix, source=src, frame=frame(rows[ix])))
    return old + new


def key(rec):
    return f"{rec['panel']}_{rec['frame']['target']}_{rec['case']:05d}_{rec['source']}.npz"


def parent_key(rec):
    return f"{rec['panel']}_{rec['case']:03d}_{rec['source']}.npz"


def manifest():
    data = []
    for rec in records():
        item = dict(rec)
        item['frame'] = {k: str(v) if isinstance(v, Path) else v for k, v in rec['frame'].items()}
        data.append(item)
    WORK.mkdir(exist_ok=True, mode=0o700)
    save(WORK/'roster.json', data)
    return data


if __name__ == '__main__':
    from collections import Counter
    data = manifest()
    print(dict(Counter((r['panel'],r['frame']['target']) for r in data)), len(data))
