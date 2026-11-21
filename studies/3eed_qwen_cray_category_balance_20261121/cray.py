"""Frozen dual-pose readout with an applicability check on Qwen's target labels."""
import json
from pathlib import Path
import re
import sys


ROAD_VEHICLE = frozenset({
    "automobile", "bus", "car", "coach", "coupe", "hatchback", "jeep", "lorry",
    "minibus", "minivan", "pickup", "sedan", "suv", "taxi", "truck", "van",
    "vehicle", "wagon",
})
CONFLICT = frozenset({
    "bicycle", "bicyclist", "bike", "boy", "child", "cyclist", "dog", "girl",
    "guy", "human", "man", "motorbike", "motorcycle", "pedestrian", "people",
    "person", "rider", "scooter", "woman",
})
FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.I)


def vehicle_reply(reply):
    """Treat the Qwen JSON label as untrusted, optional prediction data."""
    if reply["status"] != "ok":
        return False
    try:
        value = json.loads(FENCE.sub("", reply["raw"].strip()))
    except (TypeError, ValueError):
        return False
    if isinstance(value, list) and len(value) == 1:
        value = value[0]
    if not isinstance(value, dict) or not isinstance(value.get("label"), str):
        return False
    tokens = set(re.findall(r"[a-z]+", value["label"].lower()))
    return bool(tokens & ROAD_VEHICLE) and not tokens & CONFLICT


def applicable(replies, geometry):
    return (
        geometry["selection_rays"] >= 5
        and vehicle_reply(replies["2d"])
        and vehicle_reply(replies["3d"])
    )


def select(replies, geometry):
    use_ray = applicable(replies, geometry)
    route = "C_ray" if use_ray else "H"
    return {"box": geometry["boxes"][route], "route": route, "applicable": use_ray}


def predict(frame, replies):
    """Use the historical frozen decoder on a normal geometry input and two replies."""
    original = str(Path(__file__).resolve().parents[1] / "3eed_qwen_full_ordinary_validation_20261121")
    sys.path.insert(0, original)
    try:
        from geometry_full import predict as predict_original
    finally:
        sys.path.remove(original)
    return select(replies, predict_original(frame, replies))
