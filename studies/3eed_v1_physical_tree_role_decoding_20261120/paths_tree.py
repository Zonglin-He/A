"""Fixed research inputs and publication-safe outputs for single-frame physical role inference."""
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
OLD = HERE.parent / '3eed_v1_local_noun_shape_20261120'
sys.path.insert(0, str(OLD))
import paths_shape as parent
from paths_shape import ROOT, CLASSES, SOURCES, digest, read, save, archive_npz, name
from predict_shape import observations, FIELDS
from symmetric_shape import SP
WORK = HERE / 'private'
OBJECTS = WORK / 'objects'
PRED = WORK / 'predictions'
OUT = HERE / 'results'
SPEC = HERE / 'PROTOCOL.json'
CFG = read(SPEC)
ARMS = tuple(CFG['controls'])
def base_pins():
    pins={key:digest(path) for key,path in [('registry',ROOT/'crossplatform/CURRENT_METHOD.json'),
        ('sources',ROOT/'crossplatform/single_frame_sources.json'),('noun_observations',parent.OBS/'SEALED.json'),
        ('previous_predictions',SP/'SEALED.json'),('previous_fields',FIELDS/'SEALED.json')]}
    assert all(CFG[k]==v for k,v in pins.items())
    return pins
