"""Evaluate v3 noun/relationship evidence and three fixed-node completions."""
import os
from time import perf_counter

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from inputs import HERE, OUT, PLATFORMS, PRIVATE, PROTOCOL, QWEN, digest, load_panel, ordinary, qwen_predictions, read, v3_pins, write
from crossplatform.ground_regions import Grounder, language, phrase_bank, regions, selection, semantic, sensor_frame_box
from crossplatform.physical_regions.tree import SideEvidence, disjoint_pairs, overlap_sum
from crossplatform.round7.camera_projection import WAYMO_ANGLES
from geometry import camera, pixel_rays, to_sensor
from scope183 import canonical


PLAN = HERE / "FOLLOWUP.json"
ARMS = ("D_noun", "D_joint", "E_v3_box", "E_qwen_size", "E_qwen_pose")


def noun_index(phrase):
    name = language.nominal(phrase)
    if name == "othervehicle":
        name = "other vehicle"
    return language.CLASSES.index(name) if name in language.CLASSES[:-1] else None


def subject_text(text, graph):
    target = next((e for e in graph["entities"] if e["role"] == "T"), None)
    if target is None:
        return "", None
    lo, hi = target["span"]
    phrase = text[lo:hi].strip(" ,.")
    noun = noun_index(phrase)
    if noun is None:
        noun = noun_index(target["head"])
    return phrase, noun


def occupancy(parent, stop):
    survives = np.ones(len(parent))
    for i in reversed(range(len(parent))):
        p = parent[i]
        if p >= 0:
            assert p > i
            survives[i] = survives[p]*(1-stop[p])
    return survives*stop


def chosen(utility, fallback):
    return int(np.argmax(utility)) if utility.max(initial=0.) > 0. else fallback


def role_utility(data, weight, words, side, depth, immediate):
    parent, stop, size, view, p = (data[k] for k in ("parent", "stop", "size", "view", "p"))
    if depth or immediate:
        joint, _ = selection.shared(parent, disjoint_pairs(parent), stop, p, words, view, side,
                                    depth, immediate, "physical")
        mass = joint.sum(1)
        return np.where(mass>0., overlap_sum(parent, size, weight*mass), 0.)
    mass = SideEvidence(parent, stop, view, shared_cut=False).marginal(p[:,words[0]], p[:,words[1]], side)
    return weight*mass


def aligned_completion(xyz, terrain, yaw, size_prior):
    """Measure the same points at fixed yaw, then preserve their facing slabs."""
    axes = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])
    q = xyz[:, :2] @ axes.T
    lo, hi = q.min(0), q.max(0)
    measured = hi-lo
    original_center = (lo+hi)/2
    measured_center = original_center @ axes
    low, top = float(xyz[:,2].min()), float(xyz[:,2].max())
    if np.isfinite(terrain).all():
        low = min(low, float(measured_center@terrain[:2]+terrain[2]))
    measured = np.r_[measured,top-low]
    extent = np.maximum(measured, size_prior)
    advance = np.where(np.abs(original_center)>measured[:2]/2, np.sign(original_center), 0.)
    center = (original_center + advance*(extent[:2]-measured[:2])/2)@axes
    box = np.r_[center, low+extent[2]/2, extent, yaw]
    assert np.isfinite(box).all() and min(box[3:6]) > 0.
    return box


def methods_for_frame(engine, row, obs, qwen):
    spec, size, transform = camera(row)
    c = obs["variants"].get("expanded")
    base = obs["baseline"] if c is None else c["camera_box"]
    old = None if base is None else to_sensor(base, transform)[0].tolist()
    result = dict(c_box=old, outputs={arm:old for arm in ARMS}, c_applicable=False, noun_applied=False,
                  relation_applied=False, noun_changed=False, relation_changed=False, original_shape=False)
    if c is None or c["status"] != "used_region":
        return result

    frame = {k: row[k] for k in ("target","points","rgb","calibration","text","seed","waymo_view")}
    text = frame["text"].strip()
    captured = {}
    with engine._lock:
        scene, _ = engine._read_observation(frame, capture=captured, defer_decoder_evidence=True)
        phrase, noun = subject_text(text, captured["graph"])
        parsed = language.interpret(text, captured)
        words = None if parsed is None else tuple(noun_index(x) for x in parsed[:2])
        if words is not None and None not in words:
            phrase, noun = parsed[0], words[0]
        model = engine._frontend.base
        if not hasattr(engine, "_noun_banks"):
            engine._noun_banks = semantic.prompts(model), semantic.prepare_nouns(model)
        scores, dims = semantic.observe(model, scene.sampled_points, *engine._noun_banks)
        seg = regions.build(scene.sampled_points, scores, dims["xyz"], frame, native_logits=True)
        data = regions.build_inventory(seg, dims)
        candidates = {entry["node"]:entry for entry in c["candidates"]}
        assert len(data["parent"]) == c["regions"] and c["chosen"] in candidates
        for i, entry in candidates.items():
            membership = data["assign"][entry["scale"]] == i
            assert data["size"][i] == entry["sampled_size"] == int(membership.sum())
        visual = np.zeros(len(data["parent"]))
        for i, entry in candidates.items():
            visual[i] = entry["visual_weight"]

        c_node = c["chosen"]
        unary, joint = c_node, c_node
        if noun is not None and phrase:
            proposal = engine._frontend.score_frame(target=row["target"], points=row["points"],
                text="The "+language.role_text(phrase)+".", seed=row["seed"], waymo_view=row["waymo_view"])
            bank = {"t_"+k:v for k,v in phrase_bank(proposal, scene, frame, normalize=False).items()}
            weights = selection.overlap_prior(data, bank, "t", regions.Returns(data["xyz"]))
            prior = occupancy(data["parent"], data["stop"])
            utils = visual*weights*prior*data["p"][:,noun]
            unary = chosen(utils, c_node)
            result["noun_applied"] = bool(utils.max(initial=0.)>0.)
            result["noun_changed"] = unary != c_node
            joint = unary
            if words is not None and None not in words:
                value = visual*role_utility(data, weights, words, *parsed[2:])
                joint = chosen(value, unary)
                result["relation_applied"] = bool(value.max(initial=0.)>0.)
                result["relation_changed"] = joint != unary

    raw = np.asarray(qwen["camera_boxes"]["raw"], float)
    rectangle = np.asarray(qwen["pixel"],float)*np.tile(size,2)/1000.
    guide = pixel_rays((rectangle[:2]+rectangle[2:])/2, spec)[0]
    qwen_upright, _ = to_sensor(raw, transform)
    for name, node in (("D_noun",unary), ("D_joint",joint)):
        box = raw.copy()
        box[:3] = guide*candidates[node]["depth"]
        result["outputs"][name] = to_sensor(box, transform)[0].tolist()
        if node == c_node:
            np.testing.assert_allclose(result["outputs"][name],old,rtol=0.,atol=1e-12)
    result["c_applicable"] = True
    result["original_shape"] = noun is not None
    anchor = data["boxes"][joint]
    result["outputs"]["E_v3_box"] = result["outputs"]["D_joint"]
    if noun is not None:
        box = regions.complete(anchor, data["dimensions"][joint,noun])
        result["outputs"]["E_v3_box"] = sensor_frame_box(box,row["target"],row["suffix"]).tolist()
    shape = regions.complete(anchor,qwen_upright[3:6])
    result["outputs"]["E_qwen_size"] = sensor_frame_box(shape,row["target"],row["suffix"]).tolist()
    yaw = qwen_upright[6]
    if row["target"] == "waymo":
        yaw += np.deg2rad(WAYMO_ANGLES[row["suffix"]])
    level = candidates[joint]["scale"]
    xyz = data["xyz"][data["assign"][level] == joint].astype(float)
    posed = aligned_completion(xyz, data["plane"], yaw, qwen_upright[3:6])
    result["outputs"]["E_qwen_pose"] = sensor_frame_box(posed,row["target"],row["suffix"]).tolist()
    # The complete rule must retain the rotated rectangle of a measured cluster.
    yaw_check = result["outputs"]["E_qwen_pose"][6]-qwen_upright[6]
    assert abs(np.sin(yaw_check)) < 1e-12
    return result


@torch.inference_mode()
def run():
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    pins = v3_pins()
    data = PRIVATE / "region_predictions.json"
    first = read(OUT / "region_seal.json")
    assert digest(data) == first["predictions"] and first["protocol"] == digest(PROTOCOL)
    region = read(data)
    qwen = qwen_predictions()
    panel = load_panel()
    rows = ordinary()
    output = PRIVATE / "followup"
    output.mkdir(mode=0o700, exist_ok=True)
    hashes = {}
    stamp = dict(protocol=digest(PLAN), code=digest(__file__), pins=pins, C=digest(OUT/"region_seal.json"),
                 panel=digest(QWEN/"private/panel.json"))
    started = perf_counter()
    for source in PLATFORMS:
        file = output / (source + ".json")
        if file.exists():
            assert read(file)["stamp"] == stamp and len(read(file)["rows"]) == 64
        else:
            engine = Grounder(source)
            table = {}
            for frame in panel:
                if frame["target"] == source:
                    continue
                text = rows[frame["target"]][frame["ordinal"]]["caption"]
                assert canonical(frame["text"]) == text
                uid = frame["uid"]
                table[uid] = methods_for_frame(engine,frame | {"text": text},region[uid],qwen[uid])
                if len(table)%16 == 0:
                    print(source,len(table),"/64",flush=True)
            assert len(table) == 64
            write(file,dict(rows=table,stamp=stamp))
            del engine
            torch.cuda.empty_cache()
        hashes[source] = digest(file)
    assert v3_pins() == pins
    write(OUT / "followup_prediction_seal.json",dict(files=hashes, stamp=stamp, directions=192,
          elapsed_seconds=perf_counter()-started, target_updates=0, qwen_generations=0, FINAL_access=False))


if __name__ == "__main__":
    run()
