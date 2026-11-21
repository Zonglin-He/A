"""Check no-noun completion and matched-support completion using frozen C choices."""
import argparse
import sys
from time import perf_counter

import numpy as np

from inputs import HERE, OUT, PLATFORMS, PRIVATE, ROOT, digest, load_panel, qwen_predictions, read, write
from geometry import camera, cell_points, raw_returns, rigid_xyz, to_sensor
from followup import aligned_completion
from region_depth import model_sample, partition, held_cells, project_model_points
from crossplatform.candidate_frontend import sensor_frame_box
from crossplatform.physical_regions import regions
from crossplatform.round7.camera_projection import WAYMO_ANGLES, model_to_camera_matrix

sys.path.insert(0, str(ROOT / "crossplatform/round7"))


PLAN = HERE / "CONTROL_PLAN.json"
ARMS = ("C_E_Qsize", "C_E_Qpose", "Fit_window_E", "Fit_region_E", "Fit_mismatch_E")


def run():
    began = perf_counter()
    region_path = PRIVATE / "region_predictions.json"
    assert digest(region_path) == read(OUT / "region_seal.json")["predictions"]
    previous = read(region_path)
    qwen = qwen_predictions()
    output = {}
    for frame in load_panel():
        uid, target, view = (frame[k] for k in ("uid","target","suffix"))
        result = previous[uid]
        original = qwen[uid]
        baseline = original["sensor_boxes"].get("ray_depth")
        boxes = {arm:baseline for arm in ARMS}
        output[uid] = boxes
        if baseline is None or not result["variants"]:
            continue
        specification, size, transform = camera(frame)
        expanded, fitted = (result["variants"][mode] for mode in ("expanded", "fit_tree"))
        upright, _ = to_sensor(result["baseline"], transform)
        qdims = upright[3:6]
        qyaw = upright[6]
        qyaw += np.deg2rad(WAYMO_ANGLES[view]) if target == "waymo" else 0.
        # Default C-alone fallbacks are preserved exactly.
        c = to_sensor(expanded["camera_box"], transform)[0].tolist()
        boxes.update(C_E_Qsize=c, C_E_Qpose=c)
        sample = model_sample(frame)
        if expanded["status"] == "used_region":
            _, levels, born, _ = partition(sample)
            chosen = expanded["chosen"]
            support = sample[levels[born[chosen]] == chosen]
            plane = regions.terrain(sample)
            nominal = regions.measured_box(support, plane)
            size_box = regions.complete(nominal, qdims)
            pose_box = aligned_completion(support, plane, qyaw, qdims)
            boxes["C_E_Qsize"] = sensor_frame_box(size_box,target,view).tolist()
            boxes["C_E_Qpose"] = sensor_frame_box(pose_box,target,view).tolist()
        if fitted["status"] == "used_region":
            # For this comparison even terrain uses fitting-side points only.
            project = project_model_points(sample, specification, size)
            visible = project["in_image"]
            hold = np.zeros(len(sample), bool)
            hold[visible] = held_cells((project["uv"][visible] // 4).astype(np.int64))
            plane = regions.terrain(sample[~hold])
            uv, cam = raw_returns(frame, specification, size)
            rectangle = np.asarray(original["pixel"], float)*np.tile(size,2)/1000.
            fit, _ = cell_points(uv, cam, rectangle)
            points = rigid_xyz(fit, np.linalg.inv(model_to_camera_matrix(specification)))
            actual = np.asarray(fitted["roi_ids"],int)
            wrong = np.asarray(fitted["mismatch_ids"],int)
            assert len(actual) == len(wrong) >= 5
            for arm, cloud in (("Fit_window_E",points), ("Fit_region_E",points[actual]), ("Fit_mismatch_E",points[wrong])):
                measured = regions.measured_box(cloud, plane)
                boxes[arm] = sensor_frame_box(regions.complete(measured, qdims),target,view).tolist()
    path = PRIVATE / "control_boxes.json"
    write(path, output)
    assert len(output)==96
    write(OUT / "controls_prediction_seal.json", dict(boxes=digest(path), plan=digest(PLAN),
        region=digest(OUT/"region_seal.json"), code=digest(__file__), elapsed_seconds=perf_counter()-began,
        frames=96, source_weights_used=False, qwen_generations=0,_FINAL_access=False))


def score():
    from score import aggregate, canonical_box, delta, recorder_ci, score_boxes, truth_for_slot
    from score_region import save, table

    seal = read(OUT / "controls_prediction_seal.json")
    path = PRIVATE / "control_boxes.json"
    assert digest(path) == seal["boxes"] and digest(PLAN) == seal["plan"] and digest(__file__) == seal["code"]
    values = read(path)
    base = {r["uid"]:r for r in table(OUT/"frames.csv")}
    out = []
    for f in load_panel():
        uid = f["uid"]
        pred = values[uid]
        gt = canonical_box(truth_for_slot(f, read(f["calibration"])["ground_info"]))
        arms = [k for k in ARMS if pred[k] is not None]
        score, residual = (score_boxes(np.stack([canonical_box(pred[a]) for a in arms]),gt) if arms
                           else ([],0.))
        assert residual<1e-8
        original = base[uid]
        scores = {a:0. for a in ARMS}
        scores.update({k:float(x) for k,x in zip(arms,score)})
        out.append({k:f[k] for k in ("uid","target","recording")} | {"A_median":float(original["A_median"]),
                   "C_expanded":float(original["C_expanded"]),
                   "fit_available":int(original["fit_tree_status"]=="used_region"), **scores})
    comparisons = (("C_E_Qsize","A_median"),("C_E_Qpose","A_median"),("C_E_Qsize","C_expanded"),
                   ("C_E_Qpose","C_expanded"),("C_E_Qpose","C_E_Qsize"),
                   ("Fit_region_E","Fit_window_E"),("Fit_region_E","Fit_mismatch_E"),
                   ("Fit_window_E","A_median"),("Fit_region_E","A_median"))
    summary, paired = [], []
    for scope, subset in (("all",out), ("fit_available",[r for r in out if r["fit_available"]])):
        for t in (*PLATFORMS,"macro"):
            rows = [r for r in subset if t in (r["target"],"macro")]
            for a in ("A_median","C_expanded",*ARMS):
                summary.append(dict(scope=scope,target=t,arm=a,n=len(rows),**aggregate([r[a] for r in rows])))
            for a,b in comparisons:
                paired.append(dict(scope=scope,target=t,arm=a,reference=b,n=len(rows),**delta(rows,a,b)))
    save(OUT/"controls_rows.csv",out)
    save(OUT/"controls_metrics.csv",summary)
    save(OUT/"controls_paired.csv",paired)
    write(OUT/"controls_score_receipt.json",dict(frames=len(out), sealed=digest(OUT/"controls_prediction_seal.json"),
          intervals={f"{a}_vs_{b}":recorder_ci(out,a,b) for a,b in comparisons}, FINAL_access=False))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("predict","score"))
    {"predict":run,"score":score}[parser.parse_args().stage]()
