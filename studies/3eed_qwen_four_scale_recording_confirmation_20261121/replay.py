"""Run the registered v3 on both incoming frozen sources per frame."""
import os
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import torch

from cohort import OUT, PLATFORMS, PRIVATE, PROTOCOL, ROOT, canonical, checked_lock, digest, read, read_panel, write
from crossplatform.ground_regions import Grounder, selection, semantic
from inputs import v3_pins


@torch.inference_mode()
def run():
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    panel = read_panel()
    config_pins = v3_pins()
    stamp = dict(panel=digest(PRIVATE/'panel.json'), method_lock=checked_lock(),
                 code=digest(Path(__file__)), protocol=digest(PROTOCOL), pins=config_pins)
    files, counts = {}, {}
    start = perf_counter()
    for source in PLATFORMS:
        expected = {f['uid'] for f in panel if source != f['target']}
        path = PRIVATE / f'v3_{source}.json'
        if path.exists():
            previous = read(path)
            assert previous['stamp'] == stamp and set(previous['rows']) == expected
            predictions = previous['rows']
        else:
            engine = Grounder(source)
            predictions = {}
            for row in panel:
                uid = row['uid']
                if uid not in expected:
                    continue
                inputs = {k: row[k] for k in ('target','points','rgb','calibration','seed','waymo_view')}
                with patch.object(semantic, 'observe', wraps=semantic.observe) as entered:
                    with patch.object(selection, 'select', wraps=selection.select) as selection_calls:
                        box = engine.predict(**inputs, text=canonical(row['text']))
                assert np.isfinite(box).all() and min(box[3:6]) > 0.
                assert entered.call_count <= 1 and selection_calls.call_count <= entered.call_count
                predictions[uid] = dict(box=box.tolist(), region_branch=bool(entered.call_count))
                if len(predictions) % 32 == 0:
                    print(source, len(predictions), '/', len(expected), flush=True)
            assert set(predictions) == expected
            write(path, dict(stamp=stamp, rows=predictions))
            del engine
            torch.cuda.empty_cache()
        counts[source] = dict(rows=len(predictions), region_entries=sum(r['region_branch'] for r in predictions.values()))
        files[source] = digest(path)
    assert sum(x['rows'] for x in counts.values()) == 2 * len(panel)
    assert v3_pins() == config_pins and checked_lock() == stamp['method_lock']
    decoder = read(ROOT/'crossplatform/CURRENT_METHOD.json')['decoders']
    write(OUT/'v3_seal.json', dict(stamp=stamp, files=files, counts=counts, decoder_registry=decoder,
         caption_format='registered canonical', seconds=perf_counter()-start, FINAL_access=False,
         source_training=0, target_updates=0, task_labels_read=False))
    print('Registered v3 predictions sealed', counts, flush=True)


if __name__ == '__main__':
    run()
