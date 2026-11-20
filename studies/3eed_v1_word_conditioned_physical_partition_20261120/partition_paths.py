from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
OLDER = HERE.parent/'3eed_v1_object_local_shape_20261120'
sys.path.insert(0,str(OLDER))
import local_paths as archive
from local_paths import ROOT, SOURCES, CLASSES, digest, read, save, archive_npz, row_id, catalog
SPEC = HERE/'PROTOCOL.json'
CFG = read(SPEC)
ARMS = tuple(CFG['arms'])
WORK = HERE/'private'
OUT = HERE/'results'
GEOMS = ('measured','local','scene_pca','envelope')


def signed():
    deps = {'registry':ROOT/'crossplatform/CURRENT_METHOD.json',
            'checkpoints':ROOT/'crossplatform/single_frame_sources.json',
            'catalog':archive.previous.WORK/'CATALOG.json'}
    assert all(digest(p)==CFG[k] for k,p in deps.items())
    return {f:digest(HERE/f) for f in ('PROTOCOL.json','partition_paths.py','merge_forest.cpp','physical_partition.py','predict_partition.py')}
