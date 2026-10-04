"""Independent CPU audit of P1 private inputs and portable anonymous results.

The audit has its own interval, argmax, patch-intersection and bootstrap
calculations. It does not import the P1 scorer, construct models, or decode
media. Root GT access is permitted only after checking the complete prediction
barrier and the recorded scoring chronology.
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import collections
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.decota_matrix_common_v1 import load, read, sha, write

BASE = ROOT / "artifacts/tastvg_spatial_conditioned_temporal_p1_v1"
PUB = ROOT / "results/tastvg_spatial_conditioned_temporal_p1/2026-10-04"
P0 = ROOT / "artifacts/tastvg_spatial_guided_temporal_p0_v1"
POOL = ROOT / "artifacts/tastvg_extended_sensitivity_v3"
VIEW = ROOT / "artifacts/tastvg_current_correction_views_v1"
ARMS = ["Full", "Soft", "A_ROI"]
CONTRASTS = [("Soft", "Full"), ("Soft", "A_ROI"), ("A_ROI", "Full")]


def t_iou(interval, truth):
    a, b = map(float, interval)
    x, y = map(float, truth)
    assert np.isfinite([a, b, x, y]).all() and b > a and y > x
    intersection = max(0., min(b, y) - max(a, x))
    union = (b - a) + (y - x) - intersection
    return intersection / union


def first_argmax(proposals, confidence):
    p, c = np.asarray(proposals, float), np.asarray(confidence, float)
    assert p.ndim == 2 and p.shape[1] == 2 and c.ndim == 1 and len(p) == len(c) > 0
    assert np.isfinite(p).all() and np.isfinite(c).all() and np.all(p[:, 1] > p[:, 0])
    best = max(range(len(c)), key=lambda j: c[j])
    return dict(index=best, interval=p[best].tolist(), confidence=float(c[best]), proposal_count=len(p))


def independent_mask(box, width, height):
    """Independent cell-by-cell area intersections after PE square squash."""
    assert width > 0 and height > 0 and np.isfinite([width, height]).all()
    output = np.zeros((24, 24), dtype=np.float64)
    b = np.asarray(box, float)
    if b.shape != (4,) or not np.isfinite(b).all() or b[2] <= b[0] or b[3] <= b[1]:
        return output
    x0, y0, x1, y1 = np.clip(b * ([336. / width, 336. / height] * 2), 0, 336)
    if x1 <= x0 or y1 <= y0:
        return output
    for row in range(24):
        dy = max(0., min(14. * (row + 1), y1) - max(14. * row, y0))
        for col in range(24):
            dx = max(0., min(14. * (col + 1), x1) - max(14. * col, x0))
            output[row, col] = min(1., max(0., dx * dy / 196.))
    return output


def _summary(rows, fields):
    grouped = collections.defaultdict(list)
    for row in rows:
        grouped[row["source_id"]].append([row[f] for f in fields])
    if not rows:
        return dict(cells=0, sources=0, metrics={})
    ids = sorted(grouped)
    values = np.stack([np.asarray(grouped[s], dtype=float).mean(axis=0) for s in ids])
    random = np.random.default_rng(20261004)
    draws = []
    for _ in range(100):
        sampled = random.integers(len(ids), size=(100, len(ids)))
        draws.append(values[sampled].mean(axis=1))
    bounds = np.percentile(np.concatenate(draws), [2.5, 97.5], axis=0)
    return dict(cells=len(rows), sources=len(ids), bootstrap_draws=10000, seed=20261004,
                metrics={f: dict(mean=float(values[:, j].mean()), ci95=bounds[:, j].tolist(),
                                 cell_macro=float(np.mean([r[f] for r in rows])),
                                 source_values={str(s): float(v) for s, v in zip(ids, values[:, j])},
                                 source_positive=int(np.count_nonzero(values[:, j] > 1e-12)),
                                 source_negative=int(np.count_nonzero(values[:, j] < -1e-12)))
                         for j, f in enumerate(fields)})


def _tails(rows, candidate, baseline):
    delta = np.asarray([r[candidate + "_t"] - r[baseline + "_t"] for r in rows])
    return dict(cells=len(rows), gain=int((delta > 1e-12).sum()), harm=int((delta < -1e-12).sum()),
                zero=int((np.abs(delta) <= 1e-12).sum()), severe_harm_gt5pp=int((delta < -.05).sum()),
                severe_harm_gt20pp=int((delta < -.20).sum()),
                new_disjoint=sum(r[baseline + "_t"] > 0 and r[candidate + "_t"] == 0 for r in rows),
                success_destroyed=sum(r[baseline + "_t"] > .5 and r[candidate + "_t"] <= .5 for r in rows),
                success_rescued=sum(r[baseline + "_t"] <= .5 and r[candidate + "_t"] > .5 for r in rows))


class Checks:
    def __init__(self):
        self.count = collections.Counter()
        self.maximum = 0.

    def close(self, actual, expected, tolerance=1e-11):
        a, b = np.asarray(actual), np.asarray(expected)
        assert a.shape == b.shape, (a.shape, b.shape)
        assert np.isfinite(a).all() and np.isfinite(b).all()
        error = float(np.max(np.abs(a - b))) if a.size else 0.
        assert error <= tolerance, (error, tolerance)
        self.maximum = max(self.maximum, error)
        self.count["numeric_scalars"] += a.size

    def nested(self, a, b):
        if isinstance(b, dict):
            assert set(a) == set(b)
            for k in b:
                self.nested(a[k], b[k])
        elif isinstance(b, (list, tuple)):
            assert len(a) == len(b)
            for x, y in zip(a, b):
                self.nested(x, y)
        elif isinstance(b, (int, float)) and not isinstance(b, bool):
            self.close(a, b)
        else:
            assert a == b


def public(directory=PUB):
    directory = Path(directory)
    tick, checks = time.monotonic(), Checks()
    cfg = read(directory / "CONFIG.json")
    rows, pred = read(directory / "ROWS.json"), read(directory / "PREDICTIONS.json")
    saved, completion = read(directory / "SUMMARY.json"), read(directory / "SCORE_COMPLETION.json")
    barrier = read(directory / "GLOBAL_PREDICTION_BARRIER.json")
    smoke, acceptance = read(directory / "SMOKE.json"), read(directory / "SMOKE_ROOT_ACCEPTANCE.json")
    assert cfg["arms"] == ARMS and cfg["alpha"] == .5
    assert cfg["bootstrap_draws"] == 10000 and cfg["bootstrap_seed"] == 20261004
    assert cfg["parameter_updates"] == cfg["backward_calls"] == cfg["backbone_calls"] == cfg["spatial_expert_calls"] == 0
    assert cfg["spatial_A_fixed"] and not cfg["DTA_started"] and not cfg["production_promoted"]
    assert cfg["historical_exposure"] and cfg["fresh_test"] is False
    assert barrier["status"] == "sealed" and barrier["GT_read"] is False and barrier["temporal_GT_scoring_started"] is False
    assert barrier["cells"] == 288 and barrier["arm_predictions"] == 864
    assert barrier["Soft_receipt_count"] == barrier["A_ROI_cached_receipt_count"] == 288
    assert len(barrier["sealed_Soft_receipt_sha256"]) == len(barrier["sealed_A_ROI_receipt_sha256"]) == 288
    assert acceptance["status"] == smoke["status"] == "pass" and smoke["GT_read"] is False
    assert smoke["time"] <= acceptance["time"] < barrier["time"] < completion["temporal_GT_scoring_start_time"] <= completion["time"]
    assert completion["private_global_seal_sha256"] == barrier["private_barrier_sha256"]
    assert completion["status"] == "completed" and completion["cells"] == 288 and completion["arm_predictions"] == 864
    assert len(rows) == len(pred) == len({r["cell_key"] for r in rows}) == 288
    assert sum(r["condition"] != "clean" for r in rows) == 240
    expected_files = {"CONFIG.json", "CODE_BINDINGS.json", "PREDICTIONS.json", "ROWS.json", "SUMMARY.json", "DECISION.json",
                      "ROWS.csv", "GLOBAL_PREDICTION_BARRIER.json", "RESOURCES.json", "SMOKE.json", "SMOKE_ROOT_ACCEPTANCE.json"}
    assert set(completion["hashes"]) == expected_files
    for filename, digest in completion["hashes"].items():
        assert sha(directory / filename) == digest
        checks.count["public_seal_hashes"] += 1
    fields = list(saved["vidstg"]["search"]["corrupt"]["metrics"])
    for row, prediction in zip(rows, pred):
        assert {f: row[f] for f in prediction} == prediction
        assert row["all_teacher_predictions_GT_used"] is False and row["temporal_GT_scored_after_global_seal"]
        assert row["cached_A_ROI"] and row["Full_raw_proposals_unchanged"] and not row["fixed_hypothesis_reranking"]
        assert set(row["selections"]) == set(ARMS)
        for arm in ARMS:
            assert 0 <= row[arm + "_t"] <= row[arm + "_oracle_t"] + 1e-12 <= 1 + 1e-12
            checks.close(row[arm + "_success"], float(row[arm + "_t"] > .5))
            checks.close(row[arm + "_disjoint"], float(row[arm + "_t"] == 0))
            selection = row["selections"][arm]
            assert 0 <= selection["index"] < selection["proposal_count"]
            assert np.isfinite(selection["confidence"])
            assert 0 <= selection["interval"][0] < selection["interval"][1] <= 1.000001
        for a, b in CONTRASTS:
            checks.close(row[a + "_minus_" + b + "_t"], row[a + "_t"] - row[b + "_t"])
            checks.close(row[a + "_minus_" + b + "_oracle_t"], row[a + "_oracle_t"] - row[b + "_oracle_t"])
        assert 0 <= row["Soft_mask_mean"] <= 1 and .5 <= row["Soft_gate_mean"] <= 1
        assert 0 <= row["Soft_global_pool_fallback_fraction"] <= 1 and 0 <= row["A_full_fallback_fraction"] <= 1
        checks.close(row["Soft_gate_mean"], .5 + .5 * row["Soft_mask_mean"] + .5 * row["Soft_global_pool_fallback_fraction"])
    assert set(saved) == {"vidstg", "hc2"}
    for dataset in saved:
        assert set(saved[dataset]) == {"search", "confirm"}
        for split in saved[dataset]:
            rr = [r for r in rows if r["dataset"] == dataset and r["split"] == split]
            assert set(saved[dataset][split]) == {"corrupt", "clean", "all"}
            for panel, z in saved[dataset][split].items():
                q = [r for r in rr if panel == "all" or (panel == "clean") == (r["condition"] == "clean")]
                calculated = _summary(q, fields)
                for f in ["cells", "sources", "bootstrap_draws", "seed", "metrics"]:
                    checks.nested(z[f], calculated[f])
                assert z["sources"] == cfg["expert_sources"][dataset][split]
                for a, b in CONTRASTS:
                    contrast = a + "_minus_" + b
                    checks.nested(z["tails"][contrast], _tails(q, a, b))
                    checks.count["tail_counts"] += len(z["tails"][contrast])
                    values = calculated["metrics"][contrast + "_t"]["source_values"]
                    positive = sorted([(s, v) for s, v in values.items() if v > 1e-12], key=lambda sv: (-sv[1], int(sv[0])))
                    negative = sorted([(s, v) for s, v in values.items() if v < -1e-12], key=lambda sv: (sv[1], int(sv[0])))
                    n, total, gross = len(values), sum(values.values()), sum(v for _, v in positive)
                    context = z["source_concentration"][contrast]
                    checks.close([context["sources"], context["positive"], context["negative"], context["zero"]],
                                 [n, len(positive), len(negative), n - len(positive) - len(negative)])
                    checks.nested(context["largest_two_positive_sources"], positive[:2])
                    checks.nested(context["worst_two_negative_sources"], negative[:2])
                    checks.close(context["gross_positive_sum"], gross)
                    fraction = sum(v for _, v in positive[:2]) / gross if gross else None
                    checks.nested(context["largest_two_fraction_of_gross_positive"], fraction)
                    checks.nested(context["leave_one_source_out_means"], {s: (total - v) / (n - 1) for s, v in values.items()} if n > 1 else {})
                    assert context["primary_population_unchanged"] and context["post_score_descriptive"]
                    assert context["concentration_denominator"] == "Sum of positive source-mean differences, not net gain"
                    checks.count["source_concentration_panels"] += 1
                for order in ["order1", "order2"]:
                    calculated = _summary([r for r in q if r["order"] == order], fields)
                    checks.nested(z["orders"][order], calculated)
    decision = read(directory / "DECISION.json")
    primary = {ds + "/" + sp: saved[ds][sp]["corrupt"]["metrics"]["Soft_minus_Full_t"] for ds in saved for sp in saved[ds]}
    secondary = {ds + "/" + sp: saved[ds][sp]["corrupt"]["metrics"]["Soft_minus_A_ROI_t"] for ds in saved for sp in saved[ds]}
    passed = {p: z["ci95"][0] > 0 for p, z in primary.items()}
    checks.nested(decision["primary"], primary)
    checks.nested(decision["secondary"], secondary)
    assert decision["Soft_panel_pass"] == passed
    assert decision["status"] == ("GO_TEACHER_EVIDENCE_ONLY" if all(passed.values()) else "NO_GO_CURRENT_SOFT_PRIOR")
    assert not decision["DTA_started"] and not decision["production_promoted"] and not decision["universal_impossibility_claim"]
    with (directory / "ROWS.csv").open(newline="") as handle:
        table = list(csv.DictReader(handle))
    assert len(table) == 288
    for row, record in zip(rows, table):
        for field in ["cell_key", "dataset", "split", "condition", "order", "source_id"]:
            assert str(row[field]) == record[field]
        for field in fields:
            checks.close(row[field], float(record[field]))
    return dict(status="pass", checks=dict(checks.count), max_absolute_error=checks.maximum,
                CPU_wall_seconds=time.monotonic() - tick, time=time.time(),
                scope="Independent anonymous arithmetic, source macro and 10000 paired-source bootstrap, tails, source influence, order controls, CSV and pre-score chronology; no private inputs")


def root():
    tick, checks = time.monotonic(), Checks()
    lock, barrier = read(BASE / "RUNTIME_LOCK.json"), read(BASE / "GLOBAL_PREDICTION_BARRIER.json")
    completion = read(PUB / "SCORE_COMPLETION.json")
    assert barrier["status"] == "sealed" and not barrier["GT_read"] and not barrier["temporal_GT_scoring_started"]
    assert barrier["cells"] == 288 and barrier["arm_predictions"] == 864
    assert read(BASE / "SMOKE_ROOT_ACCEPTANCE.json")["time"] < barrier["time"] < completion["temporal_GT_scoring_start_time"]
    assert sha(BASE / "GLOBAL_PREDICTION_BARRIER.json") == completion["private_global_seal_sha256"]
    assert sha(BASE / "RUNTIME_LOCK.json") == barrier["runtime_lock_sha256"]
    assert sha(BASE / "FULL_INPUT.json") == barrier["full_input_sha256"]
    code = dict(lock["code"])
    for revision in sorted((BASE / "revisions").glob("*.json")):
        code.update(read(revision)["pin_overrides"])
    for path, digest in {**code, **lock["assets"], **lock["inputs"]}.items():
        path = Path(path)
        assert sha(path if path.is_absolute() else ROOT / path) == digest
        checks.count["immutable_input_pins"] += 1
    cells = read(BASE / "COHORT.json")["cells"]
    cellkey = lambda c: "/".join(str(c[f]) for f in ["dataset", "split", "condition", "order", "arrival"])
    filename = lambda c: f'{c["dataset"]}_{c["split"]}_{c["condition"]}_{c["order"]}_{c["arrival"]:05}.json'
    assert len(cells) == len({cellkey(c) for c in cells}) == 288
    assert set(barrier["receipts"]) == {"Soft/" + filename(c) for c in cells}
    assert set(barrier["A_ROI_receipts"]) == {str((P0 / "A_ROI" / filename(c)).relative_to(ROOT)) for c in cells}
    for path, digest in barrier["receipts"].items():
        assert sha(BASE / path) == digest
        checks.count["pre_score_receipts"] += 1
    for path, digest in barrier["A_ROI_receipts"].items():
        assert sha(ROOT / path) == digest
        checks.count["pre_score_cached_receipts"] += 1
    full_input, atlas = read(BASE / "FULL_INPUT.json"), read(BASE / "A_BOXES_INPUT.json")
    plans = {ds: read(VIEW / ds / "PLAN.json") for ds in ["vidstg", "hc2"]}
    scored = {r["cell_key"]: r for r in read(PUB / "ROWS.json")}
    assert set(scored) == set(atlas) == set(full_input) == {cellkey(c) for c in cells}
    prepared = []
    for cell in cells:
        k = cellkey(cell)
        plan = plans[cell["dataset"]]["rows"][cell["parent"]]
        ids, inp = plan["frame_ids"], plan["input"]
        old = load(ROOT / cell["old_payload"])
        assert (old["pre_sha"], old["post_sha"], old["pixel_sha256"]) == (cell["pre_sha"], cell["post_sha"], cell["pixel_sha256"])
        box = old["slow"]["boxes"].numpy().astype(float)
        xyxy = np.column_stack(((box[:, 0] - box[:, 2] / 2) * inp["width"],
                                (box[:, 1] - box[:, 3] / 2) * inp["height"],
                                (box[:, 0] + box[:, 2] / 2) * inp["width"],
                                (box[:, 1] + box[:, 3] / 2) * inp["height"]))
        checks.close(atlas[k], xyxy, 0)
        assert len(xyxy) == len(ids)
        full = full_input[k]
        assert sha(ROOT / full["full_cache"]) == full["full_cache_sha256"]
        value = load(ROOT / full["full_cache"])
        assert value["GT_read"] is False
        duration = (ids[-1] - ids[0] + 1) / inp["fps"]
        slots = ids[0] + np.arange(max(1, int(duration * 2))) * (inp["fps"] / 2)
        picked = np.abs(np.asarray(ids)[None, :] - slots[:, None]).argmin(axis=1).tolist()
        assert picked == value["picked_observations"] and duration == value["duration"]
        selection = first_argmax(value["proposals"], value["proposal_confidence"])
        checks.nested(selection, full["Full"])
        supports, selections = {"Full": value["proposals"]}, {"Full": selection}
        for arm, base in [("Soft", BASE), ("A_ROI", P0)]:
            receipt = read(base / arm / filename(cell))
            assert receipt["cell_key"] == k and receipt["arm"] == arm
            assert sha(base / receipt["cache"]) == receipt["cache_sha256"]
            cached = load(base / receipt["cache"])
            selected = first_argmax(cached["proposals"], cached["confidence"])
            checks.nested(selected, receipt["selection"])
            checks.nested(selected, cached["selection"])
            assert receipt["original_pixel_sha256"] == cell["pixel_sha256"]
            assert receipt["A_state_pre_sha256"] == cell["pre_sha"] and receipt["A_state_post_sha256"] == cell["post_sha"]
            assert receipt["original_frame_count"] == len(ids) and receipt["feature_frame_count"] == len(picked)
            assert cached["duration"] == duration and cached["picked"] == picked
            assert cached["feature_input_sha256"] == receipt["feature_input_sha256"]
            assert cached["video_features"].shape == value["video_features"].shape
            assert np.isfinite(cached["video_features"].numpy()).all()
            if arm == "Soft":
                assert receipt["GT_read"] is False and cached["GT_read"] is False
                unique = np.unique(picked)
                masks = cached["masks"]
                assert isinstance(masks, np.ndarray) and masks.dtype == np.float64 and masks.flags.c_contiguous
                assert masks.shape == (len(unique), 24, 24)
                assert hashlib.sha256(masks.tobytes()).hexdigest() == receipt["mask_sha256"]
                for index, observed in zip(unique, masks):
                    checks.close(observed, independent_mask(xyxy[index], inp["width"], inp["height"]), 1e-12)
                    checks.count["independent_patch_masks"] += 1
                metadata = dict(pixel_sha256=cell["pixel_sha256"], caption=inp["caption"], ids=ids, fps=inp["fps"],
                                duration=duration, picked=picked, boxes=xyxy[unique].tolist(), alpha=.5,
                                pool_impl_sha256=sha(ROOT / "vg_tta/tastvg_spatial_conditioned_temporal_p1_v1.py"),
                                transform="stock_PE336_FP16_batch16_squash", pool="CLS1_patch.5+.5fractional_mask_original_projection")
                digest = hashlib.sha256(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                assert digest == receipt["feature_input_sha256"] and Path(receipt["cache"]).name == digest + ".pt"
                empty = masks.sum((1, 2)) == 0
                checks.close(scored[k]["Soft_mask_mean"], masks.mean())
                checks.close(scored[k]["Soft_gate_mean"], np.where(empty[:, None, None], 1., .5 + .5 * masks).mean())
                checks.close(scored[k]["Soft_global_pool_fallback_fraction"], empty.mean())
                assert cached["stats"] == receipt["stats"]
                checks.close(cached["stats"]["mask_mean"], masks.mean())
                checks.close(cached["stats"]["mask_empty_frames"], empty.sum())
                for public_field, private_field in [("Soft_feature_relative_L2", "feature_relative_L2"),
                                                    ("Soft_feature_cosine", "feature_cosine"),
                                                    ("Soft_global_feature_mean_norm", "global_feature_mean_norm"),
                                                    ("Soft_feature_mean_norm", "soft_feature_mean_norm")]:
                    checks.close(scored[k][public_field], cached["stats"][private_field], 0)
            else:
                assert not cached["GT_spatial_used"] and not cached["GT_temporal_span_used"]
                checks.close(scored[k]["A_full_fallback_fraction"], receipt["invalid_box_full_fallbacks"] / len(ids))
            supports[arm], selections[arm] = cached["proposals"], selected
        prepared.append((cell, ids, supports, selections))
        checks.count["fixed_A_state_chains"] += 1
    # GT pins/labels are opened only after all physical predictions and masks pass.
    labels = {}
    for dataset in ["vidstg", "hc2"]:
        for split in ["search", "confirm"]:
            path = POOL / dataset / f"GT_LABELS_{split}.json"
            assert sha(path) == lock["GT_inputs"][str(path.relative_to(ROOT))]
            checks.count["private_temporal_GT_pins"] += 1
            labels[(dataset, split)] = read(path)
    for cell, ids, supports, selected in prepared:
        row = scored[cellkey(cell)]
        truth = labels[(cell["dataset"], cell["split"])][str(cell["parent"])]["span"]
        origin, length = ids[0], ids[-1] - ids[0] + 1
        for arm in ARMS:
            choice = selected[arm]
            assert row["selections"][arm]["index"] == choice["index"]
            checks.close(row["selections"][arm]["interval"], [(v - origin) / length for v in choice["interval"]], 0)
            checks.close(row["selections"][arm]["confidence"], choice["confidence"], 0)
            assert row["selections"][arm]["proposal_count"] == choice["proposal_count"]
            values = [t_iou(p, truth) for p in supports[arm]]
            checks.close(row[arm + "_t"], values[choice["index"]])
            checks.close(row[arm + "_oracle_t"], max(values))
            checks.count["independent_raw_proposal_tIoU"] += len(values)
    portable = public(PUB)
    write(PUB / "PUBLIC_AUDIT.json", portable)
    output = dict(status="pass", checks=dict(checks.count), max_absolute_error=checks.maximum,
                  public_audit=portable, time=time.time(), CPU_wall_seconds=time.monotonic() - tick,
                  no_media_decode=True, no_model_forward=True,
                  CUDA_initialized=sys.modules["torch"].cuda.is_initialized(),
                  spatial_states_and_prior_inputs_unchanged=True,
                  scope="All private pins, pre-score receipts, A conversions, stock-squash fractional geometry, mask/cache input hashes, original time grids, first confidence argmax and independent temporal IoU; no GT box scoring")
    write(PUB / "ROOT_AUDIT.json", output)
    write(BASE / "ROOT_AUDIT.json", output)
    print(output["status"], output["checks"], flush=True)
    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["root", "public"], default="public", nargs="?")
    parser.add_argument("directory", type=Path, nargs="?", default=PUB)
    args = parser.parse_args()
    result = root() if args.mode == "root" else public(args.directory)
    print(json.dumps(result, indent=2))
