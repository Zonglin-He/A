"""CPU teacher-only scoring after the complete P1 prediction barrier.

This module never constructs a model, decodes media, or changes spatial A.
Physical predictions are private; the public rows contain normalized selected
intervals and anonymous scalar metrics only. GT access starts in ``score`` after
all input pins, cached arms and new Soft receipts have been verified.
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import csv
import hashlib
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.decota_matrix_common_v1 import load, read, sha, status, write
from scripts.audit_tastvg_dta_oracle_r1_v1 import summary
from scripts.spatial_guided_temporal_import_v1 import core

BASE = ROOT / "artifacts/tastvg_spatial_conditioned_temporal_p1_v1"
PUB = ROOT / "results/tastvg_spatial_conditioned_temporal_p1/2026-10-04"
P0 = ROOT / "artifacts/tastvg_spatial_guided_temporal_p0_v1"
POOL = ROOT / "artifacts/tastvg_extended_sensitivity_v3"
VIEW = ROOT / "artifacts/tastvg_current_correction_views_v1"
ARMS = ["Full", "Soft", "A_ROI"]
CONTRASTS = [("Soft", "Full"), ("Soft", "A_ROI"), ("A_ROI", "Full")]
FIELDS = (
    [a + "_t" for a in ARMS]
    + [a + "_minus_" + b + "_t" for a, b in CONTRASTS]
    + [a + "_success" for a in ARMS]
    + [a + "_disjoint" for a in ARMS]
    + [a + "_oracle_t" for a in ARMS]
    + [a + "_minus_" + b + "_oracle_t" for a, b in CONTRASTS]
    + ["Soft_mask_mean", "Soft_gate_mean", "Soft_global_pool_fallback_fraction", "A_full_fallback_fraction",
       "Soft_feature_relative_L2", "Soft_feature_cosine", "Soft_global_feature_mean_norm", "Soft_feature_mean_norm"]
)


def key(cell):
    return "/".join(str(cell[f]) for f in ["dataset", "split", "condition", "order", "arrival"])


def name(cell):
    return f'{cell["dataset"]}_{cell["split"]}_{cell["condition"]}_{cell["order"]}_{cell["arrival"]:05}'


def array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def mask_digest(value):
    """The worker contract is C-contiguous float64 bytes on unique observations."""
    masks = array(value)
    assert masks.dtype == np.float64 and masks.flags.c_contiguous
    return hashlib.sha256(masks.tobytes()).hexdigest()


def _path(path, relative_to=ROOT):
    path = Path(path)
    return path if path.is_absolute() else relative_to / path


def verify_seal():
    """Fail closed before opening any temporal labels or scored P0 rows."""
    barrier = read(BASE / "GLOBAL_PREDICTION_BARRIER.json")
    assert barrier["status"] == "sealed"
    assert barrier["GT_read"] is False and barrier["temporal_GT_scoring_started"] is False
    assert barrier["cells"] == 288 and barrier["arm_predictions"] == 864
    lock = read(BASE / "RUNTIME_LOCK.json")
    code = dict(lock["code"])
    for revision in sorted((BASE / "revisions").glob("*.json")):
        code.update(read(revision)["pin_overrides"])
    for path, digest in {**code, **lock["assets"], **lock["inputs"]}.items():
        assert sha(_path(path)) == digest, path
    assert sha(BASE / "RUNTIME_LOCK.json") == barrier["runtime_lock_sha256"]
    assert sha(BASE / "FULL_INPUT.json") == barrier["full_input_sha256"]
    cells = read(BASE / "COHORT.json")["cells"]
    assert len(cells) == len({key(c) for c in cells}) == 288
    assert sum(c["condition"] != "clean" for c in cells) == 240
    expected = {"Soft/" + name(c) + ".json" for c in cells}
    assert set(barrier["receipts"]) == expected
    expected_old = {str((P0 / "A_ROI" / (name(c) + ".json")).relative_to(ROOT)) for c in cells}
    assert set(barrier["A_ROI_receipts"]) == expected_old
    for path, digest in barrier["receipts"].items():
        assert sha(BASE / path) == digest, path
    for path, digest in barrier["A_ROI_receipts"].items():
        assert sha(ROOT / path) == digest, path
    assert read(BASE / "SMOKE.json")["status"] == "pass"
    assert read(BASE / "SMOKE_ROOT_ACCEPTANCE.json")["status"] == "pass"
    return barrier, lock, cells


def tails(rows, candidate, baseline):
    delta = [r[candidate + "_t"] - r[baseline + "_t"] for r in rows]
    return dict(
        cells=len(rows),
        gain=sum(d > 1e-12 for d in delta),
        harm=sum(d < -1e-12 for d in delta),
        zero=sum(abs(d) <= 1e-12 for d in delta),
        severe_harm_gt5pp=sum(d < -.05 for d in delta),
        severe_harm_gt20pp=sum(d < -.20 for d in delta),
        new_disjoint=sum(r[baseline + "_t"] > 0 and r[candidate + "_t"] == 0 for r in rows),
        success_destroyed=sum(r[baseline + "_t"] > .5 and r[candidate + "_t"] <= .5 for r in rows),
        success_rescued=sum(r[baseline + "_t"] <= .5 and r[candidate + "_t"] > .5 for r in rows),
    )


def concentration(values):
    """Descriptive source influence; never changes the primary population."""
    positive = sorted([(s, v) for s, v in values.items() if v > 1e-12], key=lambda sv: (-sv[1], int(sv[0])))
    negative = sorted([(s, v) for s, v in values.items() if v < -1e-12], key=lambda sv: (sv[1], int(sv[0])))
    gross = float(sum(v for _, v in positive))
    total = float(sum(values.values()))
    n = len(values)
    return dict(
        sources=n,
        positive=len(positive), negative=len(negative), zero=n - len(positive) - len(negative),
        largest_two_positive_sources=positive[:2], worst_two_negative_sources=negative[:2],
        gross_positive_sum=gross,
        largest_two_fraction_of_gross_positive=sum(v for _, v in positive[:2]) / gross if gross else None,
        concentration_denominator="Sum of positive source-mean differences, not net gain",
        leave_one_source_out_means={s: (total - v) / (n - 1) for s, v in values.items()} if n > 1 else {},
        primary_population_unchanged=True, post_score_descriptive=True,
    )


def summarize(rows):
    output = {}
    for dataset in ["vidstg", "hc2"]:
        output[dataset] = {}
        for split in ["search", "confirm"]:
            rr = [r for r in rows if r["dataset"] == dataset and r["split"] == split]
            output[dataset][split] = {}
            for panel in ["corrupt", "clean", "all"]:
                q = [r for r in rr if panel == "all" or (panel == "clean") == (r["condition"] == "clean")]
                z = summary(q, FIELDS)
                z["tails"] = {a + "_minus_" + b: tails(q, a, b) for a, b in CONTRASTS}
                z["source_concentration"] = {
                    a + "_minus_" + b: concentration(z["metrics"][a + "_minus_" + b + "_t"]["source_values"])
                    for a, b in CONTRASTS
                }
                z["orders"] = {o: summary([r for r in q if r["order"] == o], FIELDS) for o in ["order1", "order2"]}
                output[dataset][split][panel] = z
    return output


def _public_barrier(barrier):
    # The full private receipt identity/path map is retained in BASE only.
    fields = ["status", "cells", "arm_predictions", "GT_read", "temporal_GT_scoring_started", "time",
              "runtime_lock_sha256", "full_input_sha256"]
    out = {f: barrier[f] for f in fields}
    out.update(
        private_barrier_sha256=sha(BASE / "GLOBAL_PREDICTION_BARRIER.json"),
        Soft_receipt_count=len(barrier["receipts"]),
        A_ROI_cached_receipt_count=len(barrier["A_ROI_receipts"]),
        sealed_Soft_receipt_sha256=sorted(barrier["receipts"].values()),
        sealed_A_ROI_receipt_sha256=sorted(barrier["A_ROI_receipts"].values()),
        publication_note="Private pre-score predictions and cache receipts were sealed earlier; anonymous export is generated after scoring",
    )
    return out


def score():
    tick = time.monotonic()
    assert not (PUB / "SCORE_COMPLETION.json").exists(), "P1 scoring already completed; preserve that result"
    barrier, lock, cells = verify_seal()
    full_input = read(BASE / "FULL_INPUT.json")
    plans = {ds: read(VIEW / ds / "PLAN.json") for ds in ["vidstg", "hc2"]}
    prepared = []
    for cell in cells:
        k = key(cell)
        full = full_input[k]
        assert sha(ROOT / full["full_cache"]) == full["full_cache_sha256"]
        cached_full = load(ROOT / full["full_cache"])
        assert core.top(cached_full["proposals"], cached_full["proposal_confidence"]) == full["Full"]
        selected = {"Full": full["Full"]}
        supports = {"Full": cached_full["proposals"]}
        receipts = {}
        masks = None
        for arm, base in [("Soft", BASE), ("A_ROI", P0)]:
            receipt = read(base / arm / (name(cell) + ".json"))
            assert receipt["cell_key"] == k and receipt["arm"] == arm
            assert sha(base / receipt["cache"]) == receipt["cache_sha256"]
            value = load(base / receipt["cache"])
            choice = core.top(value["proposals"], value["confidence"])
            assert choice == receipt["selection"] == value["selection"]
            assert value["duration"] == cached_full["duration"] and value["picked"] == cached_full["picked_observations"]
            assert value["feature_input_sha256"] == receipt["feature_input_sha256"]
            assert receipt["original_pixel_sha256"] == cell["pixel_sha256"]
            assert receipt["A_state_pre_sha256"] == cell["pre_sha"] and receipt["A_state_post_sha256"] == cell["post_sha"]
            assert not receipt.get("GT_spatial_used", False) and not receipt.get("GT_temporal_span_used", False)
            if arm == "Soft":
                assert value["GT_read"] is False and receipt["GT_read"] is False
                masks = array(value["masks"])
                assert masks.shape == (len(np.unique(value["picked"])), 24, 24)
                assert masks.dtype == np.float64 and np.isfinite(masks).all() and masks.min() >= 0 and masks.max() <= 1
                assert mask_digest(masks) == receipt["mask_sha256"]
                assert value["stats"] == receipt["stats"]
            selected[arm] = choice
            supports[arm] = value["proposals"]
            receipts[arm] = receipt
        prepared.append((cell, selected, supports, receipts, masks))

    # No temporal GT file or scored P0 row is opened above this point.
    scoring_start = time.time()
    assert barrier["time"] < scoring_start
    labels = {}
    for dataset in ["vidstg", "hc2"]:
        for split in ["search", "confirm"]:
            path = POOL / dataset / f"GT_LABELS_{split}.json"
            assert sha(path) == lock["GT_inputs"][str(path.relative_to(ROOT))]
            labels[(dataset, split)] = read(path)

    rows, predictions, physical = [], [], []
    for cell, selected, supports, receipts, masks in prepared:
        k = key(cell)
        plan = plans[cell["dataset"]]["rows"][cell["parent"]]
        ids = plan["frame_ids"]
        origin, length = ids[0], ids[-1] - ids[0] + 1
        span = labels[(cell["dataset"], cell["split"])][str(cell["parent"])]["span"]
        normalized = {a: dict(v, interval=[(x - origin) / length for x in v["interval"]]) for a, v in selected.items()}
        prediction = dict(
            cell_key=k, dataset=cell["dataset"], split=cell["split"], condition=cell["condition"], order=cell["order"],
            source_id=cell["parent"], arrival=cell["arrival"], selections=normalized,
            interval_coordinates="clip-normalized, retained full observation envelope; no temporal crop",
            A_state_pre_sha256=cell["pre_sha"], A_state_post_sha256=cell["post_sha"],
            original_pixel_sha256=cell["pixel_sha256"], Full_input_sha256=full_input[k]["full_input_sha256"],
            feature_input_sha256={a: receipts[a]["feature_input_sha256"] for a in receipts},
            Soft_mask_sha256=receipts["Soft"]["mask_sha256"], all_teacher_predictions_GT_used=False,
            cached_A_ROI=True, Full_raw_proposals_unchanged=True, fixed_hypothesis_reranking=False,
        )
        metric = {}
        for arm in ARMS:
            t = core.iou(selected[arm]["interval"], span)
            metric.update({arm + "_t": t, arm + "_success": float(t > .5), arm + "_disjoint": float(t == 0),
                           arm + "_oracle_t": max(core.iou(p, span) for p in supports[arm])})
        for a, b in CONTRASTS:
            metric[a + "_minus_" + b + "_t"] = metric[a + "_t"] - metric[b + "_t"]
            metric[a + "_minus_" + b + "_oracle_t"] = metric[a + "_oracle_t"] - metric[b + "_oracle_t"]
        mask_mean = float(np.asarray(masks, dtype=np.float64).mean())
        empty = np.asarray(masks).sum((1, 2)) == 0
        gate_mean = float(np.where(empty[:, None, None], 1., .5 + .5 * masks).mean())
        stats = receipts["Soft"]["stats"]
        row = dict(prediction, **metric, Soft_mask_mean=mask_mean, Soft_gate_mean=gate_mean,
                   Soft_global_pool_fallback_fraction=float(empty.mean()),
                   A_full_fallback_fraction=receipts["A_ROI"]["invalid_box_full_fallbacks"] / len(ids),
                   Soft_feature_relative_L2=stats["feature_relative_L2"], Soft_feature_cosine=stats["feature_cosine"],
                   Soft_global_feature_mean_norm=stats["global_feature_mean_norm"], Soft_feature_mean_norm=stats["soft_feature_mean_norm"],
                   temporal_GT_scored_after_global_seal=True)
        rows.append(row)
        predictions.append(prediction)
        physical.append(dict(cell_key=k, selections=selected))

    write(BASE / "SEALED_PHYSICAL_SELECTION_READBACK.json", physical)
    write(PUB / "PREDICTIONS.json", predictions)
    write(PUB / "ROWS.json", rows)
    output = summarize(rows)
    write(PUB / "SUMMARY.json", output)
    primary = {ds + "/" + sp: output[ds][sp]["corrupt"]["metrics"]["Soft_minus_Full_t"]
               for ds in output for sp in output[ds]}
    secondary = {ds + "/" + sp: output[ds][sp]["corrupt"]["metrics"]["Soft_minus_A_ROI_t"]
                 for ds in output for sp in output[ds]}
    passed = {panel: value["ci95"][0] > 0 for panel, value in primary.items()}
    decision = dict(
        status="GO_TEACHER_EVIDENCE_ONLY" if all(passed.values()) else "NO_GO_CURRENT_SOFT_PRIOR",
        Soft_panel_pass=passed, primary_contrast="Soft_minus_Full_t", secondary_contrast="Soft_minus_A_ROI_t",
        primary=primary, secondary=secondary, decision_rule="All four corrupt primary paired-source lower 95% CI > 0",
        DTA_started=False, production_promoted=False, universal_impossibility_claim=False,
        interpretation="Frozen teacher confidence-top1 interval tIoU only; not final STVG vIoU or adaptation gain",
        support_scope="Each arm has its own raw proposal support; no fixed-hypothesis ranking claim",
    )
    write(PUB / "DECISION.json", decision)
    write(PUB / "GLOBAL_PREDICTION_BARRIER.json", _public_barrier(barrier))
    for filename in ["SMOKE.json", "SMOKE_ROOT_ACCEPTANCE.json"]:
        write(PUB / filename, read(BASE / filename))
    soft_barrier = barrier
    resources = dict(
        Full_cached_unique_inputs=len({full_input[key(c)]["full_cache_sha256"] for c in cells}),
        A_ROI_cached_unique_inputs=len({r[3]["A_ROI"]["cache_sha256"] for r in prepared}),
        A_ROI_new_calls=0,
        Soft={f: soft_barrier[f] for f in ["status", "cells", "new_calls", "GPU_worker_wall_seconds", "backward_calls", "parameter_updates", "time"]},
        smoke=read(BASE / "SMOKE.json"), CPU_score_wall_seconds=time.monotonic() - tick,
        Soft_main_new_calls=barrier["new_calls"],
        Soft_smoke_format_calls=read(BASE / "SMOKE.json")["new_Soft_format_calls"],
        Soft_total_new_calls_including_smoke=barrier["new_calls"] + read(BASE / "SMOKE.json")["new_Soft_format_calls"],
        Full_smoke_control_calls=read(BASE / "SMOKE.json")["new_Full_control_calls"],
        total_GPU_worker_wall_seconds=barrier["GPU_worker_wall_seconds"] + read(BASE / "SMOKE.json")["GPU_worker_wall_seconds"],
        budget_max_new_Soft_requests=288, backbone_calls=0, spatial_expert_calls=0, backward_calls=0, parameter_updates=0,
        GPU_time_scope="Worker wall includes initialization, decoding, CPU masking and I/O; not pure kernel latency",
    )
    write(PUB / "RESOURCES.json", resources)
    csv_fields = ["cell_key", "dataset", "split", "condition", "order", "source_id"] + FIELDS
    with (PUB / "ROWS.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows({f: row[f] for f in csv_fields} for row in rows)
    files = ["CONFIG.json", "CODE_BINDINGS.json", "PREDICTIONS.json", "ROWS.json", "SUMMARY.json", "DECISION.json",
             "ROWS.csv", "GLOBAL_PREDICTION_BARRIER.json", "RESOURCES.json", "SMOKE.json", "SMOKE_ROOT_ACCEPTANCE.json"]
    completion = dict(status="completed", cells=288, arm_predictions=864,
                      private_global_seal_sha256=sha(BASE / "GLOBAL_PREDICTION_BARRIER.json"),
                      hashes={f: sha(PUB / f) for f in files}, temporal_GT_scoring_start_time=scoring_start,
                      time=time.time(), CPU_wall_seconds=time.monotonic() - tick)
    write(PUB / "SCORE_COMPLETION.json", completion)
    status(BASE / "STATUS.json", dict(status="completed_pending_root_audit_publication", time=time.time(), cells=288, arm_predictions=864))
    print(decision["status"], {p: 100 * z["mean"] for p, z in primary.items()}, flush=True)
    return decision


if __name__ == "__main__":
    score()
