"""Read only the sealed ordinary roster and the original, unlabeled predictions."""
from pathlib import Path
import sys

from PIL import Image
import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / '3eed_qwen_four_scale_recording_confirmation_20261121'
QWEN = HERE.parent / '3eed_frozen_qwen_metric_grounding_20261120'
ROOT = HERE.parents[1]
PRIVATE, OUT, SPEC = HERE / 'private', HERE / 'results', HERE / 'PROTOCOL.json'
sys.path[:0] = [str(QWEN), str(ROOT)]
from paths import read, write, digest

def inputs():
    seal = read(BASE/'results/prediction_seal.json')
    assert not seal['task_labels_read'] and not seal['FINAL_access']
    for name, k in [('predictions','predictions'), ('panel','panel')]:
        assert digest(BASE/'private'/f'{name}.json') == seal[k]
    assert digest(BASE/'METHOD_LOCK.json') == seal['method_lock']
    assert digest(BASE/'PROTOCOL.json') == seal['protocol']
    assert digest(BASE/'predict.py') == seal['code']
    panel = read(BASE/'private/panel.json')
    frozen = read(BASE/'private/predictions.json')
    assert len(panel) == len(frozen) == len({r['point_sha'] for r in panel}) == 416
    assert {r['uid'] for r in panel} == set(frozen)
    return panel, frozen

def crop_photo(row, entry):
    assert digest(row['rgb']) == row['rgb_sha']
    with Image.open(row['rgb']) as src:
        size = np.array(src.size)
        window = np.array(entry['pixel']) * np.tile(size,2)/1000
        pad = (window[2:]-window[:2])/8
        lo = np.maximum(0, np.floor(window[:2]-pad)).astype(int)
        hi = np.minimum(size, np.ceil(window[2:]+pad)).astype(int)
        crop = src.convert('RGB').crop(tuple([*lo,*hi]))
    return crop, [int(x) for x in [*lo,*hi]]
