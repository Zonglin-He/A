"""Replay the exact official v3 entry and capture its active branch."""
import os
from time import perf_counter
from unittest.mock import patch

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from inputs import OUT, PLATFORMS, PRIVATE, PROTOCOL, QWEN, digest, load_panel, read, v3_pins, write
from crossplatform.ground_regions import Grounder, selection, semantic


@torch.inference_mode()
def run():
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    OUT.mkdir(exist_ok=True)
    PRIVATE.mkdir(exist_ok=True, mode=0o700)
    panel = load_panel()
    pins = v3_pins()
    stamp = dict(protocol=digest(PROTOCOL), panel=digest(QWEN / "private/panel.json"), code=digest(__file__),
                 inputs=digest(__file__.replace("replay_v3.py", "inputs.py")), pins=pins)
    files, counts = {}, {}
    began = perf_counter()
    for source in PLATFORMS:
        dest = PRIVATE / ("v3_" + source + ".json")
        if dest.exists():
            done = read(dest)
            assert done["stamp"] == stamp and len(done["rows"]) == 64
            prediction = done["rows"]
        else:
            engine = Grounder(source)
            prediction = {}
            for row in panel:
                if row["target"] == source:
                    continue
                fields = {k: row[k] for k in ("target", "points", "rgb", "calibration", "text", "seed", "waymo_view")}
                selected = []
                original = selection.select

                def register(*args, **kwargs):
                    decision = original(*args, **kwargs)
                    selected.append(decision[0])
                    return decision

                with (patch.object(semantic, "observe", wraps=semantic.observe) as noun,
                      patch.object(selection, "select", side_effect=register)):
                    box = engine.predict(**fields)
                assert np.isfinite(box).all() and min(box[3:6]) > 0.
                assert noun.call_count in (0, 1) and len(selected) <= noun.call_count
                prediction[row["uid"]] = dict(box=box.tolist(), entered=bool(noun.call_count),
                    selected=bool(selected and selected[0] >= 0))
                if len(prediction) % 16 == 0:
                    print(source, len(prediction), "/ 64", flush=True)
            assert len(prediction) == 64
            write(dest, dict(stamp=stamp, rows=prediction))
            del engine
            torch.cuda.empty_cache()
        files[source] = digest(dest)
        counts[source] = dict(rows=len(prediction), entered=sum(r["entered"] for r in prediction.values()),
                              selected=sum(r["selected"] for r in prediction.values()))
    assert v3_pins() == pins and sum(x["rows"] for x in counts.values()) == 192
    write(OUT / "v3_seal.json", dict(stamp=stamp, files=files, counts=counts,
          seconds=perf_counter()-began, model=torch.cuda.get_device_name(), target_updates=0, FINAL_access=False))


if __name__ == "__main__":
    run()
