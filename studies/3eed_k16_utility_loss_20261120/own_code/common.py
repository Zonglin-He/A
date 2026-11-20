"""Shared, fixed evidence inputs and experiment identifiers."""
import csv
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
R183 = HERE.parent / "3eed_v2_ordinary_val_subject_ranking_20261119"
R182 = HERE.parent / "3eed_v2_fixed_slate_subject_ranking_20261119"
SINGLE = HERE.parent / "3eed_v1_locked_single_vote_ordinary_validation_20261120"
PRIVATE, OUT = HERE / "run_PRIVATE", HERE / "results"
sys.path[:0] = [str(R183), str(R182), str(ROOT)]
from scope183 import DIRECTIONS, PLATFORMS, atomic, digest, read

ARMS = tuple(read(HERE / "PROTOCOL.json")["arms"])
OFFSETS = (0, 1000, 2000)
LEVELS = (.25, .5)
METRICS = ("Acc25", "Acc50", "mIoU")


def register():
    inputs = {str(p.relative_to(ROOT)):digest(p) for p in (
        HERE/"PROTOCOL.json", R182/"fit182.py", R182/"ranker182.py", R182/"PROTOCOL.json",
        R182/"source_PRIVATE/SEALED.json", R182/"target_PRIVATE/SEALED.json",
        R182/"rankers_PRIVATE/FITTED.json", R183/"ordinary_PRIVATE/SEALED.json",
        R183/"predictions_PRIVATE/SEALED.json", R183/"predict183.py", ROOT/"crossplatform/CURRENT_METHOD.json")}
    cfg = read(R183/"PROTOCOL.json")
    assert inputs[str((R183/"ordinary_PRIVATE/SEALED.json").relative_to(ROOT))] == cfg["input_roster_seal_sha256"]
    assert inputs[str((R182/"rankers_PRIVATE/FITTED.json").relative_to(ROOT))] == cfg["r182_fit_receipt_sha256"]
    record = dict(inputs=inputs, primary="Acc25", offsets=OFFSETS, target_task_labels_read=False, FINAL_access=False)
    path = PRIVATE/"LOCK.json"
    if path.exists():
        assert read(path) == dict(record, offsets=list(OFFSETS))
    else:
        atomic(path, record)
    return inputs


def source_data(platform, split):
    import numpy as np
    base = R182 / "source_PRIVATE"
    receipt = read(base / "SEALED.json")
    path = base / f"{platform}_{split}.npz"
    assert receipt["source_only"] and not receipt["target_labels_read"] and not receipt["FINAL_access"]
    assert digest(path) == receipt["files"][path.name]["sha256"]
    with np.load(path, allow_pickle=False) as data:
        return {k:data[k] for k in ("features","scores","ious","incumbent","hard")}


def write_csv(path, rows):
    tmp = path.with_suffix(".tmp.csv")
    with tmp.open("w", newline="") as output:
        w = csv.DictWriter(output, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


if __name__ == "__main__":
    register()
