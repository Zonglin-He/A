"""Binding of the frozen source and the label-free target rosters."""
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
LAST=HERE.parent/'3eed_v1_noun_vote_transport_20261120'
sys.path.insert(0,str(LAST))
import vote_paths as previous
from vote_paths import ROOT, SOURCES, CLASSES, old, digest, read, save, row_id, archive_npz, catalog
SPEC=HERE/'PROTOCOL.json';CFG=read(SPEC);ARMS=tuple(CFG['arms'])
WORK=HERE/'private';OUT=HERE/'results'
def base():
    paths=(ROOT/'crossplatform/CURRENT_METHOD.json',ROOT/'crossplatform/single_frame_sources.json',
           previous.WORK/'SEALED.json',previous.WORK/'CATALOG.json')
    pins=[digest(f) for f in paths]
    assert pins==CFG['inputs']
    return dict(source=pins,protocol=digest(SPEC),paths=digest(Path(__file__)))
