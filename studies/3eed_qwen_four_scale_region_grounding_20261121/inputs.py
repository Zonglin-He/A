"""Verify the frozen Qwen panel and match its members to the ordinary roster."""
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PRIVATE = HERE / "private"
OUT = HERE / "results"
PROTOCOL = HERE / "PROTOCOL.json"
QWEN = HERE.parent / "3eed_frozen_qwen_metric_grounding_20261120"
V3 = HERE.parent / "3eed_four_scale_direct_subject_20261120"
PLATFORMS = ("waymo", "drone", "quad")

sys.path[:0] = [str(QWEN), str(ROOT), str(HERE.parent / "3eed_v2_ordinary_val_subject_ranking_20261119")]
from paths import digest, read, write
from scope183 import VIEWS, check_input, ordinary


def load_panel():
    q = QWEN / "private/panel.json"
    assert digest(q) == read(QWEN / "results/panel_seal.json")["panel"]
    panel = read(q)
    assert len(panel) == len({a["point_sha"] for a in panel}) == 96
    ordinary_rows = ordinary()
    for platform in PLATFORMS:
        assert sum(a["target"] == platform for a in panel) == 32
    for a in panel:
        ro = ordinary_rows[a["target"]][a["ordinal"]]
        check_input(ro)
        for old, new in (("rgb", "imagepath"), ("points", "pointpath"), ("calibration", "metapath"),
                         ("slot", "object_index"), ("suffix", "suffix")):
            assert a[old] == ro[new]
        assert digest(a["points"]) == a["point_sha"] and digest(a["rgb"]) == a["rgb_sha"]
        a["seed"] = ro["seed"]
        a["waymo_view"] = VIEWS[a["suffix"]]
    return panel


def qwen_predictions():
    folder = QWEN / "private"
    old = folder / "predictions.json"
    baseline = read(QWEN / "results/geometry_seal.json")
    model = QWEN / "results/model_seal.json"
    assert digest(old) == baseline["predictions"] and digest(model) == baseline["model"]
    hashes = read(folder / "reply_hashes.json")
    seal = read(model)
    assert seal["pins"]["panel"] == digest(folder / "panel.json")
    assert seal["ordered_reply_hash"] == hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    assert len(hashes) == 192
    predictions = read(old)
    assert len(predictions) == 96
    for uid, entry in predictions.items():
        for mode in ("2d", "3d"):
            key = uid + "_" + mode
            f = folder / "replies" / (key + ".json")
            assert digest(f) == hashes[key]
            value = entry["pixel"] if mode == "2d" else entry["camera_boxes"].get("raw")
            assert read(f)["value"] == value
    return predictions


def case_id(row):
    return hashlib.sha256(("shared-role-scene/1|" + str((row["target"], row["ordinal"]))).encode()).hexdigest()[:22]


def v3_pins():
    registry = ROOT / "crossplatform/CURRENT_METHOD.json"
    registered = read(registry)
    assert registered["version"] == "3" and registered["variant"] == "four_scale_direct_subject"
    reference = read(V3 / "private/SEAL.json")["code"]
    paths = dict(
        ground_regions=ROOT / "crossplatform/ground_regions.py",
        source_assets=ROOT / "crossplatform/single_frame_sources.json",
        **{name: ROOT / "crossplatform/physical_regions" / (name + ".py")
           for name in ("regions", "selection", "semantic", "tree")},
    )
    for name, path in paths.items():
        suffix = ".json" if name == "source_assets" else ".py"
        key = "single_frame_sources.json" if name == "source_assets" else name + suffix
        assert digest(path) == reference[key]
    return {k: digest(p) for k, p in paths.items()} | {"registry": digest(registry)}
