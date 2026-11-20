from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / '3eed_v1_object_local_shape_20261120'
sys.path.insert(0, str(PREVIOUS))
import local_paths as prior
from local_paths import CLASSES, ROOT, SOURCES, archive_npz, catalog, digest, read, row_id, save

SPEC = HERE / 'PROTOCOL.json'
WORK = HERE / 'private'
OUT = HERE / 'results'
ARMS = tuple(read(SPEC)['controls'])


def roster():
    cfg = read(SPEC)
    assert digest(ROOT/'crossplatform/CURRENT_METHOD.json') == cfg['frozen_v1']
    assert digest(ROOT/'crossplatform/single_frame_sources.json') == cfg['source_registry']
    doc = catalog()
    rows = [r for r in doc['rows'] if r['scope'] == 'noun']
    assert len(rows) == 128 and all(r['active'] for r in rows)
    return rows, {r['frame']: doc['frames'][r['frame']] for r in rows}


def signatures():
    return {path.name: digest(path) for path in
            [SPEC, HERE/'token_paths.py', HERE/'capture_token.py', HERE/'decide_token.py',
             prior.WORK/'PLANS.json', prior.previous.WORK/'CATALOG.json']}
