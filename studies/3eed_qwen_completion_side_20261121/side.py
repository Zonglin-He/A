"""Frozen-box geometry, signed decision labels and matched prompt construction."""
from pathlib import Path
import json
import re
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
SPAN = HERE.parent / "3eed_visible_amodal_span_20261121"
sys.path.insert(0, str(SPAN))
from common import BASE, QWEN, crop_photo, digest, inputs, read, write

OUT = HERE / "results"
PRIVATE = HERE / "private"
SPEC = HERE / "PROTOCOL.json"
ARMS = ("A", "B", "C", "D")
ETA = {"positive": 1, "negative": -1, "balanced": 0}
TOL = 1e-6


def geometry(entry):
    box = np.asarray(entry["boxes"]["H"])
    pts = np.asarray(entry["support_raw"])
    yaw = box[6]
    u = np.array([np.cos(yaw), np.sin(yaw), 0.])
    v = np.array([-u[1], u[0], 0.])
    basis = np.stack((u, v, [0., 0., 1.]))
    # Centering first avoids unnecessary cancellation in distant sensor frames.
    projected = (pts - box[:3]) @ basis.T
    lo, hi = projected.min(0), projected.max(0)
    span = hi - lo
    prior = np.array(entry["boxes"]["A"][3:6])
    ordered_xy = np.sort(prior[:2])
    prior[:2] = ordered_xy[::-1] if span[0] >= span[1] else ordered_xy
    np.testing.assert_allclose(np.maximum(span[:2], prior[:2]), box[3:5], atol=TOL, rtol=0.)
    assert box[5] + TOL >= max(span[2], prior[2])
    slack = np.maximum(0., box[3:5] - span[:2])
    k = int(np.argmax(slack))
    midpoint = (lo[k] + hi[k]) / 2
    step = slack[k] / 2
    offsets = {}
    for label, sign in ETA.items():
        delta = float(midpoint + sign * step)
        offsets[label] = delta if step * 2 > TOL and abs(delta) > TOL else 0.
    # Parent far-face completion must coincide with one feasible state.
    assert min(abs(x) for x in offsets.values()) < TOL
    for delta in offsets.values():
        assert delta - box[3+k]/2 <= lo[k]+TOL and delta + box[3+k]/2 >= hi[k]-TOL
    order = [k, 1-k, 2]
    return dict(k=k, axis=basis[k].tolist(), slack_m=float(slack[k]),
                offsets=offsets, measured=span[order].tolist(), completed=box[3:6][order].tolist(),
                prior=prior[order].tolist())


def legend(row, entry, g):
    from geometry import camera, rigid_xyz, sensor_pixels
    spec, _, transform = camera(row)
    direction = np.array(g["axis"])
    center = np.asarray(entry["support_raw"]).mean(0)
    xyz = rigid_xyz(np.stack([center-.02*direction, center+.02*direction]), transform)
    uv = sensor_pixels(xyz, spec)
    if uv is None or not np.isfinite(uv).all():
        return None
    dx, dy = uv[1]-uv[0]
    compass = ("right","down-right","down","down-left","left","up-left","up","up-right")
    near_radial = np.linalg.norm(np.cross(transform[:3,:3] @ direction,
                          xyz.mean(0) / np.linalg.norm(xyz.mean(0)))) < .05
    if near_radial:
        screen = "almost along the viewing ray, with very little screen displacement"
    else:
        index = int(np.floor(np.arctan2(dy, dx)/(np.pi/4) + .5)) % 8
        screen = "toward the " + compass[index] + " of the image"
    dz = float((transform[:3,:3] @ direction)[2])
    depth = ("farther from the camera" if dz > .25 else
             "nearer to the camera" if dz < -.25 else "at roughly the same camera depth")
    return f"The positive direction runs {screen}, {depth}. Negative is the opposite."


def hint(values, source):
    a,b,h = values
    return (f"{source} in metres (tested axis, other horizontal axis, height): "
            f"{a:.2f}, {b:.2f}, {h:.2f}. Ratios (tested/other, height/other): "
            f"{a/max(b,1.e-6):.2f}, {h/max(b,1.e-6):.2f}.")


def prompts(description, direction, g):
    context = ("The image is a crop around the referred object. Target description (quoted data): "
               + json.dumps(description, ensure_ascii=False) +
               "\nA laser observes only part of the object. Along ONE horizontal 3D axis, "
               "decide where the unobserved length should be added beyond the observed span. "
               + direction + "\n")
    question = ("\nUse the object's visible pose and exposed surfaces. Choose positive or negative "
                "when missing length should predominantly extend that way. Choose balanced when "
                "the evidence supports similar extension on both ends. If the visible object and "
                "the supplied evidence do not identify the missing side or a balanced split, choose "
                "unknown. Output only one JSON object with key side and one of the exact values "
                "positive, negative, balanced, unknown.")
    sizes = "Estimated complete object size"
    evidence = {"A": "No numeric extent hint.", "B": hint(g["measured"], "Measured LiDAR span"),
                "C": hint(g["completed"], sizes), "D": hint(g["prior"], sizes)}
    return {arm:context+evidence[arm]+question for arm in ARMS}


def parse(answer):
    clean = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", answer.strip(), flags=re.I)
    try:
        obj = json.loads(clean)
    except ValueError:
        return "invalid"
    if (type(obj) is dict and set(obj) == {"side"} and isinstance(obj["side"], str)
            and obj["side"] in (*ETA, "unknown")):
        return obj["side"]
    return "invalid"


def proposed_box(parent, axis, delta):
    if parent is None or not delta:
        return parent
    box = np.array(parent)
    box[:3] += delta * np.asarray(axis)
    return box.tolist()
