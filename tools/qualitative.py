"""Private offline illustrations from fully sealed and scored formal jobs.

Never run inference or score a new prediction. Every requested cohort is
verified in full before this command reads labels or decodes RGB. Images and
the selection receipt remain under ignored outputs/formal/qualitative/.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_formal import JOBS, canonical, digest, validate_worker

CATEGORIES = ("spatial", "temporal", "joint", "failure")
EPS = 1e-10
RULE = {
    "spatial": "delta_v>0, delta_fixed_actual_sIoU>0, delta_tIoU<=0; rank spatial delta",
    "temporal": "delta_v>0, delta_tIoU>0, delta_fixed_actual_sIoU<=0; rank temporal delta",
    "joint": "delta_v>0 and both spatial/temporal deltas>0; rank delta_v",
    "failure": "delta_v<0; rank delta_v",
    "selection": "ascending relevant delta, stable cohort ordinal tie-break; upper median (n//2); exclude already selected parents in spatial/temporal/joint/failure order",
    "frames": "six evenly indexed actual input feature frames across the full predeclared support; no dense/interpolated frame is selected",
}


def records(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def safe_path(root, relative):
    logical = Path(relative)
    if logical.is_absolute() or ".." in logical.parts:
        raise ValueError("Unsafe formal job path")
    return root / logical


def verified_job(root, backbone, dataset):
    """Validate complete cohort/run/seal/score/receipts before any GT read."""
    root = Path(root).resolve()
    folder = root / "outputs/formal"
    plan = json.loads((folder / "plan.json").read_text())
    state = json.loads((folder / "state.json").read_text())
    if hashlib.sha256(canonical(plan)).hexdigest() != state.get("plan_sha256"):
        raise ValueError("Formal plan/state binding differs")
    key = backbone + "-" + dataset
    matched = [j for j in plan["jobs"] if j["id"] == key]
    if len(matched) != 1 or state.get("jobs", {}).get(key, {}).get("status") != "scored":
        raise ValueError("Qualitative output requires this complete formal job to be scored")
    job = matched[0]
    if job.get("backbone") != backbone or job.get("dataset") != dataset:
        raise ValueError("Formal job identity differs")
    manifest = safe_path(root, job["manifest"])
    output = safe_path(root, job["output"])
    labels = safe_path(root, job["labels"])
    if digest(manifest) != job["manifest_sha256"]:
        raise ValueError("Formal manifest changed")
    progress = state["jobs"][key]
    for name, field in [("sealed.json", "seal_sha256"), ("metrics.json", "metrics_sha256")]:
        if digest(output / name) != progress.get(field):
            raise ValueError("Scored formal artifact changed")
    # This verifies every NPZ and every per-query exact-input receipt, including
    # queries that the illustration selector will not use. It does not read GT.
    validate_worker(root, job, require_seal=True)
    seal = json.loads((output / "sealed.json").read_text())
    metrics = json.loads((output / "metrics.json").read_text())
    cohort = records(manifest)
    if (metrics.get("status") != "scored_sealed_predictions"
            or metrics.get("provenance", {}).get("manifest_sha256") != job["manifest_sha256"]):
        raise ValueError("Scoring is not bound to the complete prediction cohort")
    rows = metrics["rows"]
    expected = {r["id"] for r in cohort}
    if (len(rows) != len(cohort) or len(expected) != job["queries"]
            or len({r["id"] for r in rows}) != len(rows)
            or {r["id"] for r in rows} != expected or set(seal["files"]) != expected
            or any(r["dataset"] != dataset for r in rows)
            or metrics["datasets"][dataset]["queries"] != len(rows)):
        raise ValueError("Qualitative selection is not the full scored cohort")
    return dict(id=key, backbone=backbone, dataset=dataset, output=output,
                manifest=manifest, labels=labels, cohort=cohort, rows=rows, seal=seal,
                manifest_sha256=job["manifest_sha256"],
                seal_sha256=progress["seal_sha256"], metrics_sha256=progress["metrics_sha256"])


def select_examples(rows):
    """Predeclared median signed examples, never per-video stop/parameter selection."""
    candidates = {c: [] for c in CATEGORIES}
    for ordinal, row in enumerate(rows):
        source, ours = row["values"]["Source"], row["values"]["Full"]
        fields = ("v", "t", "fixed_actual_sIoU")
        if any(not isinstance(values.get(k), (int, float)) or isinstance(values.get(k), bool)
               or not math.isfinite(values[k]) for values in (source, ours) for k in fields):
            continue
        dv, dt, ds = (ours[k] - source[k] for k in fields)
        entry = dict(id=row["id"], parent=str(row["parent"]), ordinal=ordinal,
                     query=f"Query{ordinal+1:06d}", delta_v=dv, delta_t=dt, delta_s=ds)
        category = None
        if dv > EPS:
            if ds > EPS and dt > EPS:
                category = "joint"
            elif ds > EPS and dt <= EPS:
                category = "spatial"
            elif dt > EPS and ds <= EPS:
                category = "temporal"
        elif dv < -EPS:
            category = "failure"
        if category:
            candidates[category].append(entry)
    selected, used = {}, set()
    for category in CATEGORIES:
        eligible = [x for x in candidates[category] if x["parent"] not in used]
        field = "delta_s" if category == "spatial" else "delta_t" if category == "temporal" else "delta_v"
        eligible.sort(key=lambda x: (x[field], x["ordinal"]))
        if not eligible:
            selected[category] = dict(status="missing", reason="no_eligible_example_with_an_unused_parent",
                                      signed_candidates=len(candidates[category]))
            continue
        chosen = dict(eligible[len(eligible) // 2])
        used.add(chosen["parent"])
        chosen.update(status="selected", signed_candidates=len(candidates[category]),
                      eligible_after_parent_exclusion=len(eligible))
        selected[category] = chosen
    return selected


def six_actual_positions(ids):
    if len(ids) < 6 or list(ids) != sorted(set(ids)):
        raise ValueError("Six distinct actual physical frames are required")
    positions = np.rint(np.linspace(0, len(ids) - 1, 6)).astype(int).tolist()
    if len(set(positions)) != 6:
        raise ValueError("Actual frame selection degenerated")
    return positions


def load_selected_truth(path, selected, dataset):
    """Only called after every requested job has passed complete verification."""
    wanted = {x["id"] for x in selected.values() if x["status"] == "selected"}
    found = {}
    for row in records(path):
        if row["id"] in wanted:
            if row["id"] in found or row["dataset"] != dataset:
                raise ValueError("Qualitative labels repeat IDs or differ from the fixed dataset")
            found[row["id"]] = row
    if set(found) != wanted:
        raise ValueError("Selected scored query lacks its offline annotation")
    return found


def pixel_boxes(normalized, width, height, clip):
    box = np.asarray(normalized, float)
    result = np.column_stack((box[:, :2] - box[:, 2:] / 2, box[:, :2] + box[:, 2:] / 2))
    result *= [width, height, width, height]
    return np.maximum(result, 0) if clip else result


def render_example(job, choice, record, truth, destination):
    """Display only actual feature nodes, using the original dataset decoder."""
    from PIL import Image, ImageDraw, ImageFont, ImageEnhance
    from stvg.decode import decode
    meta = record["metadata"]
    ids = meta["frame_ids"]
    positions = six_actual_positions(ids)
    shown = [ids[p] for p in positions]
    # Same HC official FFmpeg output grid / Vid zero-based select semantics.
    frames, decoded_ids = decode(dict(meta, frame_ids=shown), job["dataset"])
    if decoded_ids != shown:
        raise ValueError("Offline RGB physical frame mapping changed")
    prediction = job["output"] / job["seal"]["files"][choice["id"]]["path"]
    with np.load(prediction, allow_pickle=False) as saved:
        if saved["frame_ids"].tolist() != ids:
            raise ValueError("Saved prediction physical nodes differ")
        source = pixel_boxes(saved["Source"], meta["width"], meta["height"], job["dataset"] == "hc2")
        ours = pixel_boxes(saved["Live"], meta["width"], meta["height"], job["dataset"] == "hc2")
        source_interval = saved["Native_interval"].tolist()
        ours_interval = saved["Strong_interval"].tolist()
        observed = {ids[p] for p in saved["positions8"].tolist()}
    target = {int(k): np.asarray(v, float) for k, v in truth["boxes"].items()}
    title = f"{job['backbone']} / {job['dataset']} / {destination.stem} / {choice['query']}"
    label_width, cell_width, top = 220, 300, 100
    image_height = min(320, round(cell_width * meta["height"] / meta["width"]))
    scale = min(cell_width / meta["width"], image_height / meta["height"])
    image_width = round(meta["width"] * scale)
    image_height = round(meta["height"] * scale)
    row_height = image_height + 64
    canvas = Image.new("RGB", (label_width + 6 * cell_width, top + 3 * row_height + 42), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 15)
        small = ImageFont.truetype("DejaVuSans.ttf", 12)
    except OSError:
        font = small = ImageFont.load_default()
    draw.text((12, 10), title, fill="black", font=font)
    draw.text((12, 35), f"Delta vIoU {100*choice['delta_v']:+.3f} pp | Delta tIoU {100*choice['delta_t']:+.3f} pp | Delta actual-node sIoU {100*choice['delta_s']:+.3f} pp", fill="black", font=font)
    draw.text((12, 60), "Offline after sealing/scoring. Six actual physical feature frames; no interpolated frames or boxes shown. No private caption displayed.", fill="black", font=small)
    rows = [("Source", source_interval, source, "#c66b00"),
            ("Ours", ours_interval, ours, "#0059c7"),
            ("GT", truth["interval"], None, "#07833e")]
    for ri, (name, interval, boxes, color) in enumerate(rows):
        y = top + ri * row_height
        draw.text((12, y + 12), name, fill=color, font=font)
        draw.text((12, y + 38), f"physical interval\n[{interval[0]}, {interval[1]})", fill="black", font=font)
        for ci, (frame, fid, pos) in enumerate(zip(frames, shown, positions)):
            x = label_width + ci * cell_width
            active = interval[0] <= fid < interval[1]
            im = Image.fromarray(frame).resize((image_width, image_height), Image.Resampling.BILINEAR)
            if not active:
                im = ImageEnhance.Brightness(im).enhance(.35)
            canvas.paste(im, (x, y))
            if active:
                box = target.get(fid) if name == "GT" else boxes[pos]
                if box is not None:
                    box = np.asarray(box) * scale + [x, y, x, y]
                    draw.rectangle(box.tolist(), outline=color, width=3)
                else:
                    draw.text((x + 4, y + 4), "GT box unavailable", fill="#ffbb55", font=font)
            else:
                draw.text((x + 4, y + 4), "OUTSIDE interval: no output box", fill="white", font=small)
            draw.text((x, y + image_height + 5), f"physical frame {fid} | actual node", fill="black", font=small)
            draw.text((x, y + image_height + 24), "expert observed" if fid in observed else "expert unobserved", fill="black", font=small)
    draw.text((12, top + 3 * row_height + 10), "Rows share identical RGB/time nodes. Blank prediction boxes outside the predicted interval do not count as emitted trajectory. Private visualization; do not upload.", fill="black", font=small)
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)
    return dict(physical_frames=shown, actual_feature_nodes=True, interpolation=False,
                Source_interval=source_interval, Ours_interval=ours_interval, GT_interval=truth["interval"])


def generate(root=ROOT, backbone=None, dataset=None):
    root = Path(root).resolve()
    requested = [(b, d) for b, d in JOBS if (backbone is None or b == backbone) and (dataset is None or d == dataset)]
    # All requested complete seals are checked before opening any label/RGB.
    jobs = [verified_job(root, b, d) for b, d in requested]
    summary = dict(status="private_offline_qualitative", rules=RULE, jobs=[])
    for job in jobs:
        chosen = select_examples(job["rows"])
        folder = root / "outputs/formal/qualitative" / job["backbone"] / job["dataset"]
        if folder.exists() and any(folder.iterdir()):
            raise FileExistsError("Private qualitative output already exists; do not overwrite a saved selection")
        truth = load_selected_truth(job["labels"], chosen, job["dataset"])
        by_id = {r["id"]: r for r in job["cohort"]}
        entry = dict(backbone=job["backbone"], dataset=job["dataset"],
                     full_scored_queries=len(job["rows"]), manifest_sha256=job["manifest_sha256"],
                     seal_sha256=job["seal_sha256"], metrics_sha256=job["metrics_sha256"],
                     labels_sha256_at_visualization=digest(job["labels"]), examples={})
        for category in CATEGORIES:
            choice = chosen[category]
            if choice["status"] != "selected":
                entry["examples"][category] = choice
                continue
            destination = folder / (category + ".png")
            info = render_example(job, choice, by_id[choice["id"]], truth[choice["id"]], destination)
            # The private image index still excludes actual IDs, captions and
            # source paths; ordinal aliases map through the locked cohort.
            entry["examples"][category] = dict(
                status="saved", query=choice["query"], file=destination.name,
                signed_candidates=choice["signed_candidates"],
                eligible_after_parent_exclusion=choice["eligible_after_parent_exclusion"],
                delta_vIoU_pp=100*choice["delta_v"], delta_tIoU_pp=100*choice["delta_t"],
                delta_actual_sIoU_pp=100*choice["delta_s"], **info)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "selection.json").write_text(json.dumps(dict(rules=RULE, **entry), indent=2) + "\n")
        summary["jobs"].append(entry)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backbone", choices=["tastvg", "cgstvg"])
    p.add_argument("--dataset", choices=["hc2", "vidstg"])
    args = p.parse_args()
    result = generate(ROOT, args.backbone, args.dataset)
    print(json.dumps(dict(status=result["status"], completed_jobs=len(result["jobs"]),
                          private_output="outputs/formal/qualitative; do not upload real media or queries")))


if __name__ == "__main__":
    main()
