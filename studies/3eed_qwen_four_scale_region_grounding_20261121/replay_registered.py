"""Repeat B with the original 3EED evaluation caption normalization."""
import csv
from unittest.mock import patch

import numpy as np

from inputs import OUT, PLATFORMS, PRIVATE, QWEN, digest, load_panel, ordinary, read, write
from scope183 import canonical
from score import aggregate, canonical_box, score_boxes, truth_for_slot

import replay_v3


def locked_text_panel():
    panel = load_panel()
    roster = ordinary()
    for frame in panel:
        spelling = roster[frame["target"]][frame["ordinal"]]["caption"]
        assert canonical(frame["text"]) == spelling
        frame["text"] = spelling
    return panel


def run():
    folder = PRIVATE / "normalized_replay"
    folder.mkdir(exist_ok=True, mode=0o700)
    original = load_panel()
    normalized = locked_text_panel()
    altered = sum(x["text"] != y["text"] for x,y in zip(original, normalized))
    assert all(x["uid"] == y["uid"] for x,y in zip(original, normalized))
    with (patch.object(replay_v3, "load_panel", side_effect=locked_text_panel),
          patch.object(replay_v3, "PRIVATE", folder), patch.object(replay_v3, "OUT", folder)):
        replay_v3.run()
    run_seal = read(folder / "v3_seal.json")
    for source, sha in run_seal["files"].items():
        assert digest(folder / ("v3_" + source + ".json")) == sha
    sources = {s: read(folder / ("v3_" + s + ".json"))["rows"] for s in PLATFORMS}
    with (OUT / "source_pairs.csv").open(newline="") as f:
        comparison = {(r["source"], r["uid"]): r for r in csv.DictReader(f)}
    scores = []
    for row in normalized:
        uid = row["uid"]
        gt = canonical_box(truth_for_slot(row, read(row["calibration"])["ground_info"]))
        included = [s for s in PLATFORMS if s != row["target"]]
        candidates = [sources[s][uid]["box"] for s in included]
        iou, gap = score_boxes(np.asarray(candidates), gt)
        assert gap < 1e-8
        for s, value in zip(included, iou):
            peer = comparison[s, uid]
            info = sources[s][uid]
            assert int(info["entered"]) == int(peer["entered"]) and int(info["selected"]) == int(peer["selected"])
            scores.append(dict(uid=uid, source=s, target=row["target"], recording=row["recording"],
                  region_path=peer["entered"], B_raw_caption=float(peer["B_v3"]),
                  B_canonical_caption=float(value), B_archived=float(peer["B_archived_v3"])))
    changes_raw = [abs(r["B_raw_caption"]-r["B_archived"]) for r in scores]
    changes_norm = [abs(r["B_canonical_caption"]-r["B_archived"]) for r in scores]
    fp = OUT / "v3_text_replay_rows.csv"
    with fp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(scores[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(scores)
    summary = {}
    for target in (*PLATFORMS, "macro"):
        group = [r for r in scores if target in (r["target"], "macro")]
        summary[target] = {arm: aggregate([r[arm] for r in group])
                           for arm in ("B_raw_caption", "B_canonical_caption", "B_archived")}
    result = dict(
        ordinary_caption_normalization_changed=altered, raw_diff_count=sum(x > 1e-8 for x in changes_raw),
        canonical_diff_count=sum(x > 1e-8 for x in changes_norm), raw_max_iou_diff=max(changes_raw),
        canonical_max_iou_diff=max(changes_norm), scores=summary, prediction_seal=digest(folder / "v3_seal.json"),
        code=digest(__file__), qwen_original_predictions_not_regenerated=True, FINAL_access=False)
    write(OUT / "v3_text_replay_audit.json", result)
    print({k:v for k,v in result.items() if k not in ("scores", "code", "prediction_seal")}, flush=True)


if __name__ == "__main__":
    run()
