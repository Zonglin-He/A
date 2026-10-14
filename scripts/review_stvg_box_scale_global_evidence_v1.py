"""Post-seal, source-bound review of the fixed STVG-R1 box-scale grid.

Importing this module does not load labels, media, a model, or GPU state.  The
only label read occurs in ``main`` after the complete eight-case grid and all
160 frozen teacher responses, runtime pins, and byte inventories pass.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/stvg_box_scale_global_evidence_v1"
OLD = ROOT / "artifacts/stvg_temporal_caption_query_spatial_opd_v1"
DESIGN = BASE / "DESIGN_LOCK.json"
RUNTIME = BASE / "RUNTIME.json"
INPUT_BARRIER = BASE / "GRID_INPUT_BARRIER.json"
JOB_LOCK = BASE / "TEACHER_JOB_LOCK.json"
TEACHER_BARRIER = BASE / "TEACHER_GLOBAL_BARRIER.json"
WEIGHT_CUSTODY = BASE / "ACTUAL_TEACHER_WEIGHT_CUSTODY.json"
LABEL_CONTRACT = BASE / "ACTUAL_LABEL_TOKEN_CONTRACT.json"
RESOURCE_LIMITS = BASE / "ACTUAL_LIVE_RESOURCE_LIMITS.json"
LAUNCH_RECEIPT = BASE / "TEACHER_LAUNCH.json"
WORKER_EXIT = BASE / "TEACHER_WORKER_EXIT.json"
COST_SCOPE = BASE / "COST_SCOPE_CORRECTION.json"
PROTOCOL = ROOT / "protocols/stvg_box_scale_global_evidence_v1.md"
EXPECTED_CASES = 8
EXPECTED_WINDOWS = 76
EXPECTED_CALLS = 160
EXPECTED_LABELS = ("support", "contradict", "insufficient")
EXPECTED_LETTERS = ("A", "B", "C")


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _canonical_sha(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _safe_relpath(value: Any) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("evidence path must be a ROOT-relative POSIX string")
    path = Path(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError("absolute or traversing evidence path")
    lowered = "/" + value.lower().strip("/") + "/"
    forbidden = ("/annotations/", "/annotation/", "/ground_truth/", "/groundtruth/",
                 "/gt_labels/", "/labels/", "/truths/")
    if any(part in lowered for part in forbidden):
        raise ValueError("GT/annotation paths are not allowed in preflight inventories")
    return value


def _verify_path_digest(relpath: str, digest: str) -> Path:
    relpath = _safe_relpath(relpath)
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError(f"malformed SHA256 receipt for {relpath}")
    path = ROOT / relpath
    if not path.is_file() or _sha(path) != digest:
        raise RuntimeError(f"missing file or SHA256 mismatch: {relpath}")
    return path


def _verify_runtime() -> tuple[dict[str, Any], str]:
    runtime = _json(RUNTIME)
    _require(runtime.get("status") == "locked" and runtime.get("GT_read") is False,
             "RUNTIME is not a locked pre-GT record")
    pins = runtime.get("pins")
    protected = runtime.get("protected")
    _require(isinstance(pins, dict) and pins and isinstance(protected, dict) and protected,
             "RUNTIME has no complete code/evidence pins or protected-method pin")
    for relpath, digest in pins.items():
        _verify_path_digest(relpath, digest)
    for relpath, digest in protected.items():
        _verify_path_digest(relpath, digest)
    _require(runtime.get("cases") == EXPECTED_CASES
             and runtime.get("anchor_count") == EXPECTED_CASES
             and runtime.get("grid_windows") == EXPECTED_WINDOWS
             and runtime.get("maximum_teacher_requests") == EXPECTED_CALLS
             and runtime.get("new_student_fits") == 0
             and runtime.get("new_model_downloads") == 0,
             "RUNTIME scope differs from the eight-anchor, 76-window, 160-request lock")
    _require(runtime.get("quantized") is False and runtime.get("dtype") == "BF16"
             and runtime.get("attention") == "SDPA"
             and runtime.get("image_side") == 448
             and runtime.get("Teacher_params") == 8292166656
             and runtime.get("Teacher_repository") == "XiaowenZhang/stvg-r1-model-7b",
             "RUNTIME frozen teacher/input execution contract changed")
    return runtime, _sha(RUNTIME)


def _inventory(document: Mapping[str, Any], *, expected_paths: set[str],
               label: str, check_bytes: bool = False) -> dict[str, dict[str, Any]]:
    rows = document.get("files")
    _require(isinstance(rows, list), f"{label} has no file inventory")
    found: dict[str, dict[str, Any]] = {}
    for row in rows:
        _require(isinstance(row, dict), f"{label} contains a malformed file row")
        relpath = _safe_relpath(row.get("path"))
        _require(relpath not in found, f"{label} repeats a file path")
        path = _verify_path_digest(relpath, row.get("sha256"))
        if check_bytes and row.get("bytes") is not None:
            _require(path.stat().st_size == int(row["bytes"]), f"{label} byte count differs: {relpath}")
        found[relpath] = row
    _require(set(found) == expected_paths, f"{label} does not seal the exact locked file set")
    return found


def _safe_softmax(logits: Sequence[float]) -> list[float]:
    top = max(float(x) for x in logits)
    exps = [math.exp(float(x) - top) for x in logits]
    total = math.fsum(exps)
    return [x / total for x in exps]


def _logsumexp_pair(left: float, right: float) -> float:
    top = max(left, right)
    return top + math.log(math.exp(left - top) + math.exp(right - top))


def _validate_result(result: Mapping[str, Any], job: Mapping[str, Any]) -> dict[str, Any]:
    _require(result.get("status") == "actual_categorical_window_evidence"
             and result.get("GT_read") is False and result.get("teacher_frozen") is True,
             "teacher result status/frozen/GT fields are invalid")
    for key in ("case_id", "dataset", "kind", "window_id", "input_file_sha256",
                "input_pixel_sha256", "prompt_sha256", "source_case_sha256"):
        _require(result.get(key) == job.get(key), f"teacher result does not match locked job field {key}")
    logits = result.get("label_logits")
    probs = result.get("probabilities")
    _require(isinstance(logits, list) and len(logits) == 3
             and all(math.isfinite(float(x)) for x in logits), "teacher logits are not finite A/B/C scalars")
    _require(isinstance(probs, list) and len(probs) == 3
             and all(math.isfinite(float(x)) and 0.0 <= float(x) <= 1.0 for x in probs),
             "teacher probabilities are malformed")
    expected_probs = _safe_softmax(logits)
    probability_error = max(abs(float(a) - b) for a, b in zip(probs, expected_probs))
    _require(probability_error <= 2e-7, "stored categorical probabilities differ from logits")
    expected_score = float(logits[0]) - _logsumexp_pair(float(logits[1]), float(logits[2]))
    score_error = abs(float(result.get("referent_support_log_odds")) - expected_score)
    _require(math.isfinite(float(result.get("referent_support_log_odds")))
             and score_error <= 2e-6,
             "referent score is not logit(A)-logsumexp(logit(B),logit(C))")
    argmax = max(range(3), key=lambda index: float(logits[index]))
    _require(result.get("preferred_label") == EXPECTED_LABELS[argmax],
             "preferred label does not follow stable A/B/C argmax order")
    for key in ("input_tokens", "visual_tokens", "output_tokens", "GPU_peak_bytes"):
        value = result.get(key)
        _require(isinstance(value, int) and not isinstance(value, bool) and value >= 0,
                 f"invalid teacher cost field: {key}")
    elapsed = result.get("total_seconds")
    _require(isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool)
             and math.isfinite(float(elapsed)) and float(elapsed) >= 0.0,
             "invalid per-request inference duration")
    _require(result.get("output_tokens") == 0, "forced categorical inference unexpectedly generated output tokens")
    _require(result.get("input_tokens") <= 4096,
             "teacher request exceeded the locked 4,096-token ceiling")
    _require(result.get("visual_tokens") > 0, "teacher request has no visual tokens")
    return {"log_odds": expected_score, "preferred_label": EXPECTED_LABELS[argmax],
            "top_label_unknown": EXPECTED_LABELS[argmax] == "insufficient",
            "input_tokens": int(result["input_tokens"]), "visual_tokens": int(result["visual_tokens"]),
            "output_tokens": 0, "model_forward_plus_cuda_sync_seconds": float(elapsed),
            "GPU_peak_bytes": int(result["GPU_peak_bytes"]),
            "probability_reconstruction_max_abs_error": probability_error,
            "score_reconstruction_abs_error": score_error}


def _rank_descending(values: Sequence[float]) -> list[int]:
    order = sorted(range(len(values)), key=lambda i: (-float(values[i]), i))
    ranks = [0] * len(order)
    for rank, index in enumerate(order, start=1):
        ranks[index] = rank
    return ranks


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: (float(values[i]), i))
    ranks = [0.0] * len(order)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and float(values[order[end]]) == float(values[order[start]]):
            end += 1
        rank = (start + 1 + end) / 2.0
        for j in range(start, end):
            ranks[order[j]] = rank
        start = end
    return ranks


def _spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    rx, ry = _average_ranks(x), _average_ranks(y)
    mx, my = math.fsum(rx) / len(rx), math.fsum(ry) / len(ry)
    dx, dy = [a - mx for a in rx], [b - my for b in ry]
    nx = math.sqrt(math.fsum(a * a for a in dx))
    ny = math.sqrt(math.fsum(b * b for b in dy))
    if nx == 0.0 or ny == 0.0:
        return None
    return math.fsum(a * b for a, b in zip(dx, dy)) / (nx * ny)


def _pixel_xyxy_from_cxcywh(box: Sequence[float], width: int, height: int,
                            *, clip_lower: bool) -> list[float]:
    cx, cy, bw, bh = [float(x) for x in box]
    values = [(cx - bw / 2.0) * width, (cy - bh / 2.0) * height,
              (cx + bw / 2.0) * width, (cy + bh / 2.0) * height]
    if clip_lower:
        values = [max(0.0, value) for value in values]
    return values


def _window_xyxy_pixels(window: Mapping[str, Any], width: int, height: int) -> list[float]:
    x0, y0, w, h = [float(window[key]) for key in ("x0", "y0", "w", "h")]
    return [x0 * width, y0 * height, (x0 + w) * width, (y0 + h) * height]


def _mean_defined(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, Any]:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return {"defined_cases": len(values), "unknown_cases": len(rows) - len(values),
            "mean": (math.fsum(values) / len(values) if values else None)}


def _sign_counts(values: Sequence[float | None], *, tolerance: float = 1e-12) -> dict[str, int]:
    defined = [float(x) for x in values if x is not None]
    return {"positive": sum(x > tolerance for x in defined),
            "negative": sum(x < -tolerance for x in defined),
            "neutral": sum(abs(x) <= tolerance for x in defined),
            "unknown": len(values) - len(defined)}


def _preflight() -> dict[str, Any]:
    from scripts.stvg_temporal_caption_query_v1 import catalog, roster
    from scripts.prepare_stvg_box_scale_grid_v1 import marked_prompt, support_prompt
    from scripts.stvg_box_scale_grid_v1 import full_scale_grid

    design = _json(DESIGN)
    _require(design.get("status") == "locked_before_grid_preparation"
             and design.get("GT_read") is False
             and design.get("teacher_new_model") is False
             and design.get("spatial_optimizer_launched") is False
             and design.get("historically_exposed") is True
             and design.get("fresh_confirmation") is False,
             "design lock status or scope differs")
    _require(design.get("protocol_sha256") == _sha(PROTOCOL), "locked protocol SHA differs")
    runtime, runtime_sha = _verify_runtime()
    pins = runtime["pins"]
    for path, digest in (("protocols/stvg_box_scale_global_evidence_v1.md", _sha(PROTOCOL)),
                         ("artifacts/stvg_box_scale_global_evidence_v1/DESIGN_LOCK.json", _sha(DESIGN)),
                         ("artifacts/stvg_box_scale_global_evidence_v1/GRID_INPUT_BARRIER.json", _sha(INPUT_BARRIER)),
                         ("artifacts/stvg_box_scale_global_evidence_v1/TEACHER_JOB_LOCK.json", _sha(JOB_LOCK)),
                         ("artifacts/stvg_temporal_caption_query_spatial_opd_v1/ROSTER_LOCK.json",
                          _sha(OLD / "ROSTER_LOCK.json"))):
        _require(pins.get(path) == digest, f"RUNTIME is not bound to required pre-inference input: {path}")
    roster_doc = _json(OLD / "ROSTER_LOCK.json")
    records = roster()
    _require(roster_doc.get("status") == "locked" and len(records) == EXPECTED_CASES
             and len(roster_doc.get("records", [])) == EXPECTED_CASES
             and design.get("cohort_sha256") == _sha(OLD / "ROSTER_LOCK.json"),
             "fixed historically exposed source roster is incomplete or changed")
    record_by_id = {row["case_id"]: row for row in records}
    _require(len(record_by_id) == EXPECTED_CASES, "duplicate cases in source roster")
    domain_counts = {name: sum(row.get("dataset") == name for row in records)
                     for name in ("hc2", "vidstg")}
    _require(domain_counts == {"hc2": 4, "vidstg": 4}, "locked cohort is not four cases per dataset")

    input_barrier = _json(INPUT_BARRIER)
    _require(input_barrier.get("status") == "complete" and input_barrier.get("count") == EXPECTED_CASES
             and input_barrier.get("total_grid_windows") == EXPECTED_WINDOWS
             and input_barrier.get("GT_read") is False
             and input_barrier.get("historically_exposed") is True,
             "grid input barrier is incomplete or has crossed the GT boundary")
    job_lock = _json(JOB_LOCK)
    jobs = job_lock.get("records")
    _require(job_lock.get("status") == "locked" and job_lock.get("GT_read") is False
             and job_lock.get("all_grid_candidates_preserved") is True
             and job_lock.get("fixed_one_anchor_per_case") is True
             and isinstance(jobs, list) and job_lock.get("count") == EXPECTED_CALLS
             and len(jobs) == EXPECTED_CALLS,
             "teacher job lock is incomplete or changed")

    grid_records = input_barrier.get("records")
    _require(isinstance(grid_records, list) and len(grid_records) == EXPECTED_CASES,
             "grid input barrier has no complete per-case grid metadata")
    grid_by_id = {row.get("case_id"): row for row in grid_records if isinstance(row, dict)}
    _require(len(grid_by_id) == EXPECTED_CASES and set(grid_by_id) == set(record_by_id),
             "grid input barrier case IDs differ from the locked roster")
    input_by_path = {}
    for row in input_barrier.get("files", []):
        _require(isinstance(row, dict), "grid input barrier has a malformed file row")
        path, digest = _safe_relpath(row.get("path")), row.get("sha256")
        _require(path not in input_by_path, "grid input barrier repeats an input file")
        _verify_path_digest(path, digest)
        input_by_path[path] = digest

    # Source-only recomputation verifies the fixed observation anchor and the
    # complete same-scale, center-independent coarse grid before labels.
    source_cases: dict[str, Any] = {}
    expected_job_keys = set()
    expected_windows_by_case: dict[str, list[dict[str, Any]]] = {}
    for case_id, rec in record_by_id.items():
        case, _catalog = catalog(rec)
        source_sha = case.get("_source_case_sha256", case.get("source_case_sha256"))
        grid_row = grid_by_id[case_id]
        _require(source_sha == rec.get("source_case_sha256")
                 and grid_row.get("source_case_sha256") == source_sha
                 and grid_row.get("dataset") == rec.get("dataset"),
                 f"grid record is not source-bound for anonymous case {case_id}")
        positions = case.get("observation_positions")
        _require(isinstance(positions, (list, tuple)) and len(positions) >= 2,
                 "source case lacks its fixed second observation position")
        anchor = int(positions[1])
        _require(grid_row.get("observation_position") == anchor
                 and 0 <= anchor < len(case["frame_ids"])
                 and int(grid_row.get("physical_frame_id")) == int(case["frame_ids"][anchor]),
                 "grid does not use the prelocked second observation position")
        source_box = case["zero"]["boxes"][anchor].detach().cpu()
        expected_grid = full_scale_grid(source_box)
        _require(grid_row.get("grid") == expected_grid,
                 "locked grid differs from the deterministic source-width/height full-image grid")
        _require(grid_row.get("grid", {}).get("center_used_for_grid") is not True
                 and expected_grid.get("student_baseline", {}).get("center_used_for_grid") is False
                 and expected_grid.get("complete_normalized_image_coverage") is True
                 and expected_grid.get("half_step_or_refinement") is False
                 and expected_grid.get("truncated") is False,
                 "grid is not complete coarse coverage with a separate source baseline")
        _require(list(grid_row.get("source_wh", [])) == [int(case["width"]), int(case["height"])],
                 "grid source dimensions differ from the locked source case")
        _require(grid_row.get("grid", {}).get("window_count") == len(expected_grid["windows"]),
                 "per-case window count differs from the deterministic grid")
        query_sha = hashlib.sha256(case["caption"].encode("utf-8")).hexdigest()
        _require(grid_row.get("query_sha256") == query_sha,
                 "grid query hash does not bind the locked source query")
        raw_pixel = case.get("source_RGB", {}).get(int(case["frame_ids"][anchor]))
        if raw_pixel is not None:
            _require(grid_row.get("source_RGB_sha256") == raw_pixel.get("RGB_sha256"),
                     "anchor raw pixel hash differs from source RGB custody")
        expected_windows_by_case[case_id] = expected_grid["windows"]
        source_cases[case_id] = case
        for window in expected_grid["windows"]:
            for kind in ("masked", "marked"):
                expected_job_keys.add((case_id, kind, int(window["window_id"])))
        expected_job_keys.add((case_id, "original", 0))

    _require(sum(len(x) for x in expected_windows_by_case.values()) == EXPECTED_WINDOWS
             and len(expected_job_keys) == EXPECTED_CALLS,
             "source recomputation does not produce exactly 76 windows and 160 requests")
    jobs_by_key = {}
    input_hashes = {}
    for job in jobs:
        _require(isinstance(job, dict), "teacher job lock contains a malformed job")
        key = (job.get("case_id"), job.get("kind"), job.get("window_id"))
        _require(key not in jobs_by_key, "teacher job lock repeats a candidate request")
        jobs_by_key[key] = job
        case_id, kind, window_id = key
        _require(key in expected_job_keys and job.get("source_case_sha256") == record_by_id[case_id]["source_case_sha256"],
                 "teacher job is outside the fixed complete grid or source binding")
        _require(job.get("dataset") == record_by_id[case_id]["dataset"]
                 and isinstance(job.get("prompt"), str)
                 and hashlib.sha256(job["prompt"].encode("utf-8")).hexdigest() == job.get("prompt_sha256"),
                 "teacher job dataset or prompt hash differs")
        expected_prompt = marked_prompt(source_cases[case_id]["caption"]) if kind == "marked" else support_prompt(source_cases[case_id]["caption"])
        _require(job["prompt"] == expected_prompt,
                 "teacher prompt is not the fixed own-query referent-support prompt")
        path = _safe_relpath(job.get("input_path"))
        _require(input_by_path.get(path) == job.get("input_file_sha256"),
                 "teacher job input is not bound to the complete input barrier")
        _verify_path_digest(path, job.get("input_file_sha256"))
        input_hashes[path] = job["input_file_sha256"]
        meta = job.get("metadata")
        _require(isinstance(meta, dict), "teacher job omits input geometry metadata")
        if kind == "original":
            _require(window_id == 0 and path.endswith(f"/{case_id}/original.png")
                     and meta.get("size") == 448 and meta.get("resampling") == "bicubic",
                     "original input is not the unmarked 448px single-frame baseline")
        else:
            window = next((row for row in expected_windows_by_case[case_id]
                           if int(row["window_id"]) == int(window_id)), None)
            _require(window is not None, "candidate job window ID is not in the complete grid")
            source_width, source_height = int(source_cases[case_id]["width"]), int(source_cases[case_id]["height"])
            expected_pixel_box = [
                max(0, min(source_width, int(round(float(window["x0"]) * source_width)))),
                max(0, min(source_height, int(round(float(window["y0"]) * source_height)))),
                max(0, min(source_width, int(round((float(window["x0"]) + float(window["w"])) * source_width)))),
                max(0, min(source_height, int(round((float(window["y0"]) + float(window["h"])) * source_height)))),
            ]
            _require(path.endswith(f"/{case_id}/{kind}_{int(window_id):03d}.png")
                     and meta.get("window_id") == int(window_id)
                     and meta.get("full_frame_preserved") is True and meta.get("crop") is False
                     and list(meta.get("source_wh", meta.get("letterbox", {}).get("source_wh", []))) == [int(source_cases[case_id]["width"]), int(source_cases[case_id]["height"])]
                     and meta.get("window_xyxy_pixels_half_open") == expected_pixel_box
                     and float(meta.get("window_area_fraction_normalized")) == float(window["w"]) * float(window["h"])
                     and meta.get("output_wh") == [448, 448]
                     and job.get("input_pixel_sha256") == meta.get("pixel_sha256"),
                     "candidate input is not a full-context 448px window view")
            if kind == "masked":
                _require(meta.get("mode") == "gaussian_blur" and meta.get("blur_radius_px") == 8.0
                         and meta.get("replacement_rule") == "blur_full_original_frame_then_replace_window_region",
                         "masked input changed its fixed blur protocol")
            else:
                _require(kind == "marked" and meta.get("visual_label") == "1"
                         and meta.get("marked_window_count") == 1
                         and meta.get("marker") == "single_red_rectangle_and_red_window_id_on_full_frame",
                         "marked input changed the fixed single-window ID-1 protocol")
    _require(set(jobs_by_key) == expected_job_keys and set(input_by_path) == set(input_hashes),
             "teacher jobs and grid input files are not a one-to-one complete mapping")

    # Actual model custody and categorical contract are generated by the
    # frozen runner; neither can substitute for its exact source pins.
    official_receipt_path = ROOT / "artifacts/stvg_target_teacher_distillation_v1/ROOT_OFFICIAL_MODEL_FILE_READBACK.json"
    official_receipt = _json(official_receipt_path)
    weight_custody = _json(WEIGHT_CUSTODY)
    _require(weight_custody.get("status") == "pass" and weight_custody.get("GT_read") is False
             and weight_custody.get("official_revision") == runtime.get("Teacher_revision")
             and weight_custody.get("all_local_files_stream_hash_verified") is True
             and weight_custody.get("new_download") is False
             and weight_custody.get("count") == len(official_receipt.get("files", {}))
             and weight_custody.get("weight_bytes") == official_receipt.get("weight_bytes"),
             "actual local frozen teacher weight custody does not match its pinned official receipt")
    token_contract = _json(LABEL_CONTRACT)
    _require(token_contract.get("status") == "pass" and token_contract.get("GT_read") is False
             and token_contract.get("labels") == list(EXPECTED_LABELS)
             and token_contract.get("answer_letters") == list(EXPECTED_LETTERS)
             and token_contract.get("equal_one_token_answers") is True
             and token_contract.get("not_calibrated_IoU") is True
             and token_contract.get("single_image_not_video_protocol") is True
             and isinstance(token_contract.get("ids"), list) and len(token_contract["ids"]) == 3
             and len(set(token_contract["ids"])) == 3,
             "actual one-token A/B/C contract or single-image scope changed")

    # The launcher's actual cgroup, GPU/host admission and termination scope are
    # post-run receipts.  They are checked as provenance, never used to alter
    # the pre-locked model/input protocol.
    resource = _json(RESOURCE_LIMITS)
    launch = _json(LAUNCH_RECEIPT)
    worker_exit = _json(WORKER_EXIT)
    cost_scope = _json(COST_SCOPE)
    gi = 1024 ** 3
    props = resource.get("properties", {})
    admission = resource.get("admission", {})
    _require(resource.get("status") == "pass" and resource.get("GT_read") is False
             and resource.get("unit") == "stvg-box-scale-teacher-001.service"
             and int(props.get("MemoryHigh", -1)) == 16 * gi
             and int(props.get("MemoryMax", -1)) == 20 * gi
             and int(props.get("MemorySwapMax", -1)) == gi
             and int(admission.get("available", 0)) >= 8 * gi
             and int(admission.get("gpu_free", 0)) >= 22 * gi
             and int(admission.get("disk_free", 0)) >= 30 * gi,
             "actual own-unit memory/swap/resource admission receipt violates the fixed guard")
    _require(launch.get("status") == "actual_launched" and launch.get("GT_read") is False
             and launch.get("runtime_sha256") == runtime_sha
             and worker_exit.get("status") == "actual_worker_exited"
             and worker_exit.get("exit_code") == 0 and worker_exit.get("GT_read") is False,
             "actual teacher launch/exit receipts do not bind a clean run to the locked runtime")
    worker_wall = cost_scope.get("teacher_owned_worker_wall_seconds")
    _require(cost_scope.get("status") == "recorded" and cost_scope.get("GT_read") is False
             and cost_scope.get("teacher_actual_start_receipt_sha256") == _sha(LAUNCH_RECEIPT)
             and cost_scope.get("teacher_actual_exit_receipt_sha256") == _sha(WORKER_EXIT)
             and isinstance(worker_wall, (int, float)) and not isinstance(worker_wall, bool)
             and math.isfinite(float(worker_wall)) and float(worker_wall) >= 0.0
             and cost_scope.get("per_request_total_seconds_actual_scope") ==
                 "model_forward_plus_CUDA_sync_only_after_preprocessing_and_transfer"
             and cost_scope.get("per_request_full_end_to_end_seconds") is None
             and cost_scope.get("grid_preparation_seconds") is None
             and cost_scope.get("not_measured_fields_remain_unknown") is True
             and cost_scope.get("original_teacher_records_changed") is False,
             "teacher cost-scope correction is missing, altered, or not bound to the actual launch/exit")
    for forbidden_receipt in (BASE / "RESOURCE_STOP_RECEIPT.json",
                              BASE / "RESOURCE_SAMPLING_ERROR.json",
                              BASE / "TEACHER_FAILURE_CAPTURE.json"):
        _require(not forbidden_receipt.exists(),
                 f"resource/failure receipt exists: {forbidden_receipt.name}")

    teacher_barrier = _json(TEACHER_BARRIER)
    _require(teacher_barrier.get("status") == "complete"
             and teacher_barrier.get("count") == EXPECTED_CALLS
             and teacher_barrier.get("GT_read") is False
             and teacher_barrier.get("runtime_sha256") == runtime_sha
             and teacher_barrier.get("grid_input_barrier_sha256") == _sha(INPUT_BARRIER)
             and teacher_barrier.get("job_lock_sha256") == _sha(JOB_LOCK),
             "teacher global barrier is incomplete or does not bind the frozen inputs/runtime")
    expected_result_paths = {
        f"{BASE.relative_to(ROOT)}/teacher/{cid}_{kind}_{int(window_id):03d}.json"
        for cid, kind, window_id in expected_job_keys
    }
    result_files = _inventory(teacher_barrier, expected_paths=expected_result_paths,
                              label="teacher result barrier")
    _require(len(result_files) == EXPECTED_CALLS, "teacher barrier does not seal all 160 responses")

    # Every raw categorical result is read and reconciled to its locked request
    # before the first target-label loader is imported or called.
    checked_results: dict[tuple[str, str, int], dict[str, Any]] = {}
    max_probability_error = 0.0
    for key, job in jobs_by_key.items():
        case_id, kind, window_id = key
        path = BASE / "teacher" / f"{case_id}_{kind}_{int(window_id):03d}.json"
        result = _json(path)
        checked = _validate_result(result, job)
        checked_results[key] = checked
        max_probability_error = max(max_probability_error,
                                    float(checked["probability_reconstruction_max_abs_error"]))
    _require(len(checked_results) == EXPECTED_CALLS,
             "teacher response set is incomplete before the GT boundary")

    kind_counts = {kind: sum(key[1] == kind for key in checked_results)
                   for kind in ("original", "masked", "marked")}
    _require(kind_counts == {"original": EXPECTED_CASES,
                             "masked": EXPECTED_WINDOWS, "marked": EXPECTED_WINDOWS},
             "sealed teacher request counts differ by input protocol")
    return {
        "design": design, "runtime": runtime, "runtime_sha256": runtime_sha,
        "input_barrier": input_barrier, "job_lock": job_lock, "jobs_by_key": jobs_by_key,
        "grid_by_id": grid_by_id, "windows_by_case": expected_windows_by_case,
        "records": records, "record_by_id": record_by_id, "source_cases": source_cases,
        "teacher_barrier": teacher_barrier, "teacher_result_files": result_files,
        "checked_results": checked_results, "max_probability_error": max_probability_error,
        "max_score_error": max((x["score_reconstruction_abs_error"] for x in checked_results.values()), default=0.0),
        "kind_counts": kind_counts,
        "token_contract": token_contract, "weight_custody": weight_custody,
        "resource_limits": resource, "launch_receipt": launch, "worker_exit": worker_exit,
        "cost_scope": cost_scope,
        "hashes": {"design": _sha(DESIGN), "protocol": _sha(PROTOCOL),
                   "runtime": runtime_sha, "roster": _sha(OLD / "ROSTER_LOCK.json"),
                   "input_barrier": _sha(INPUT_BARRIER), "job_lock": _sha(JOB_LOCK),
                   "teacher_barrier": _sha(TEACHER_BARRIER),
                   "weight_custody": _sha(WEIGHT_CUSTODY),
                   "label_contract": _sha(LABEL_CONTRACT),
                   "resource_limits": _sha(RESOURCE_LIMITS),
                   "launch_receipt": _sha(LAUNCH_RECEIPT),
                   "worker_exit": _sha(WORKER_EXIT),
                   "cost_scope": _sha(COST_SCOPE),
                   "official_model_receipt": _sha(official_receipt_path),
                   "protected_method": runtime["protected"]["methods/CURRENT_METHOD.json"],
                   "teacher_response_inventory": _canonical_sha({
                       alias: result_files[rel]["sha256"]
                       for alias, rel in enumerate(sorted(result_files))})},
    }


def _score_after_seal(gate: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np
    from scripts.review_stvg_sparse_r1_v2 import _truths_after_seal
    from vg_tta.tastvg_oracle_event5_v1 import box_iou

    truths, truth_bindings = _truths_after_seal(gate["source_cases"])
    roster_records = gate["record_by_id"]
    aliases: dict[str, str] = {}
    counts = {"hc2": 0, "vidstg": 0}
    for rec in gate["records"]:
        dataset = rec["dataset"]
        counts[dataset] += 1
        aliases[rec["case_id"]] = ("HC" if dataset == "hc2" else "V") + f"{counts[dataset]:02d}"

    cases_out = []
    all_candidates = []
    metric_rows_for_agg = []
    cost_by_kind: dict[str, dict[str, Any]] = {}
    all_teacher = list(gate["checked_results"].values())
    for kind in ("original", "masked", "marked"):
        subset = [row for (cid, k, _), row in gate["checked_results"].items() if k == kind]
        cost_by_kind[kind] = {
            "calls": len(subset),
            "input_tokens": sum(x["input_tokens"] for x in subset),
            "visual_tokens": sum(x["visual_tokens"] for x in subset),
            "output_tokens": sum(x["output_tokens"] for x in subset),
            "model_forward_plus_cuda_sync_seconds": math.fsum(
                x["model_forward_plus_cuda_sync_seconds"] for x in subset),
            "max_gpu_peak_bytes": max((x["GPU_peak_bytes"] for x in subset), default=None),
        }

    for rec in gate["records"]:
        cid, dataset = rec["case_id"], rec["dataset"]
        alias = aliases[cid]
        case = gate["source_cases"][cid]
        grid_record = gate["grid_by_id"][cid]
        grid = grid_record["grid"]
        windows = gate["windows_by_case"][cid]
        anchor_fid = int(grid_record["physical_frame_id"])
        width, height = int(case["width"]), int(case["height"])
        truth = truths[cid]["truth"]
        anchor_gt = truth.get(anchor_fid)
        source_iou = None
        if anchor_gt is not None:
            source_box = case["zero"]["boxes"][int(grid_record["observation_position"])].detach().cpu().double().tolist()
            source_pixel = _pixel_xyxy_from_cxcywh(source_box, width, height, clip_lower=(dataset == "hc2"))
            source_iou = float(box_iou(source_pixel, anchor_gt))

        baseline_result = gate["checked_results"][(cid, "original", 0)]
        baseline = baseline_result["log_odds"]
        per_candidate = []
        for window in windows:
            wid = int(window["window_id"])
            mask = gate["checked_results"][(cid, "masked", wid)]
            marked = gate["checked_results"][(cid, "marked", wid)]
            response = baseline - mask["log_odds"]
            iou = None
            if anchor_gt is not None:
                iou = float(box_iou(_window_xyxy_pixels(window, width, height), anchor_gt))
            per_candidate.append({
                "window_id": wid,
                "row_major_row": int(window["row"]),
                "row_major_column": int(window["column"]),
                "masking_response_original_minus_masked_log_odds": response,
                "mask_preferred_label": mask["preferred_label"],
                "mask_top_label_unknown": mask["top_label_unknown"],
                "marked_referent_support_log_odds": marked["log_odds"],
                "marked_preferred_label": marked["preferred_label"],
                "marked_top_label_unknown": marked["top_label_unknown"],
                "anchor_GT_IoU": iou,
            })
        mask_scores = [r["masking_response_original_minus_masked_log_odds"] for r in per_candidate]
        marked_scores = [r["marked_referent_support_log_odds"] for r in per_candidate]
        mask_ranks = _rank_descending(mask_scores)
        marked_ranks = _rank_descending(marked_scores)
        for row, mask_rank, marked_rank in zip(per_candidate, mask_ranks, marked_ranks):
            row["mask_teacher_rank_stable"] = mask_rank
            row["marked_teacher_rank_stable"] = marked_rank
        if anchor_gt is None:
            for row in per_candidate:
                row["oracle_IoU_rank_stable"] = None
        else:
            iou_scores = [float(r["anchor_GT_IoU"]) for r in per_candidate]
            for row, rank in zip(per_candidate, _rank_descending(iou_scores)):
                row["oracle_IoU_rank_stable"] = rank

        top_mask_index = min(range(len(per_candidate)), key=lambda i: (-mask_scores[i], i))
        top_marked_index = min(range(len(per_candidate)), key=lambda i: (-marked_scores[i], i))
        top_mask, top_marked = per_candidate[top_mask_index], per_candidate[top_marked_index]
        max_mask, max_marked = max(mask_scores), max(marked_scores)
        mask_ties = sum(x == max_mask for x in mask_scores)
        marked_ties = sum(x == max_marked for x in marked_scores)
        oracle = None
        oracle_ties = None
        delta_oracle = delta_mask = delta_marked = None
        if anchor_gt is not None:
            oracle = max(float(r["anchor_GT_IoU"]) for r in per_candidate)
            oracle_ties = sum(float(r["anchor_GT_IoU"]) == oracle for r in per_candidate)
            delta_oracle = oracle - float(source_iou)
            delta_mask = float(top_mask["anchor_GT_IoU"]) - float(source_iou)
            delta_marked = float(top_marked["anchor_GT_IoU"]) - float(source_iou)

        rho_mask = (_spearman(mask_scores, [float(r["anchor_GT_IoU"]) for r in per_candidate])
                    if anchor_gt is not None else None)
        rho_marked = (_spearman(marked_scores, [float(r["anchor_GT_IoU"]) for r in per_candidate])
                      if anchor_gt is not None else None)
        mask_sign = _sign_counts([float(x) for x in mask_scores])
        iou_defined = [float(r["anchor_GT_IoU"]) for r in per_candidate if r["anchor_GT_IoU"] is not None]
        iou_variance = (math.fsum((x - math.fsum(iou_defined) / len(iou_defined)) ** 2 for x in iou_defined) /
                        len(iou_defined)) if iou_defined else None
        all_iou_zero = (bool(iou_defined) and all(x == 0.0 for x in iou_defined)) if iou_defined else None
        row = {
            "case": alias, "dataset": dataset,
            "anchor_annotation_status": "available" if anchor_gt is not None else "unknown_anchor_GT_box_absent",
            "anchor_GT_available": anchor_gt is not None,
            "candidate_count": len(per_candidate),
            "source_box_anchor_IoU": source_iou,
            "oracle_grid_IoU": oracle,
            "oracle_candidate_tie_count": oracle_ties,
            "oracle_delta_vs_source_IoU": delta_oracle,
            "masked_teacher_top1_window_id": top_mask["window_id"],
            "masked_teacher_top1_candidate_IoU": top_mask["anchor_GT_IoU"],
            "masked_teacher_delta_vs_source_IoU": delta_mask,
            "masked_teacher_top1_label": top_mask["mask_preferred_label"],
            "masked_teacher_top1_label_unknown": top_mask["mask_top_label_unknown"],
            "masked_teacher_top_score_tie_count": mask_ties,
            "marked_teacher_top1_window_id": top_marked["window_id"],
            "marked_teacher_top1_candidate_IoU": top_marked["anchor_GT_IoU"],
            "marked_teacher_delta_vs_source_IoU": delta_marked,
            "marked_teacher_top1_label": top_marked["marked_preferred_label"],
            "marked_teacher_top1_label_unknown": top_marked["marked_top_label_unknown"],
            "marked_teacher_top_score_tie_count": marked_ties,
            "masked_reward_vs_anchor_IoU_spearman": rho_mask,
            "marked_log_odds_vs_anchor_IoU_spearman": rho_marked,
            "negative_masking_response_count": mask_sign["negative"],
            "zero_masking_response_count": mask_sign["neutral"],
            "positive_masking_response_count": mask_sign["positive"],
            "masking_response_sign_tolerance": 1e-12,
            "coarse_grid_IoU_reward_variance": iou_variance,
            "all_grid_candidate_IoUs_zero": all_iou_zero,
            "local_policy_interpretation": (
                "no positive IoU candidate in this finite coarse grid; diagnostic only, no OPD fit was run"
                if all_iou_zero is True else
                "candidate-IoU variation measured on the finite coarse grid only; no OPD fit was run"
                if anchor_gt is not None else
                "unknown because the fixed anchor has no GT box; no OPD fit was run"),
            "baseline_referent_support_log_odds": baseline,
            "baseline_preferred_label": baseline_result["preferred_label"],
            "baseline_top_label_unknown": baseline_result["top_label_unknown"],
            "candidate_rows": per_candidate,
        }
        cases_out.append(row)
        metric_rows_for_agg.append(row)
        all_candidates.extend({"case": alias, **candidate} for candidate in per_candidate)

    aggregate = {}
    for dataset in ("hc2", "vidstg", "all"):
        group = (cases_out if dataset == "all" else [r for r in cases_out if r["dataset"] == dataset])
        aggregate[dataset] = {
            "cases": len(group),
            "source_box_anchor_IoU": _mean_defined(group, "source_box_anchor_IoU"),
            "oracle_grid_IoU": _mean_defined(group, "oracle_grid_IoU"),
            "oracle_delta_vs_source_IoU": _mean_defined(group, "oracle_delta_vs_source_IoU"),
            "masked_teacher_top1_candidate_IoU": _mean_defined(group, "masked_teacher_top1_candidate_IoU"),
            "masked_teacher_delta_vs_source_IoU": _mean_defined(group, "masked_teacher_delta_vs_source_IoU"),
            "marked_teacher_top1_candidate_IoU": _mean_defined(group, "marked_teacher_top1_candidate_IoU"),
            "marked_teacher_delta_vs_source_IoU": _mean_defined(group, "marked_teacher_delta_vs_source_IoU"),
            "masked_teacher_spearman": _mean_defined(group, "masked_reward_vs_anchor_IoU_spearman"),
            "marked_teacher_spearman": _mean_defined(group, "marked_log_odds_vs_anchor_IoU_spearman"),
            "oracle_delta_signs": _sign_counts([r["oracle_delta_vs_source_IoU"] for r in group]),
            "masked_teacher_delta_signs": _sign_counts([r["masked_teacher_delta_vs_source_IoU"] for r in group]),
            "marked_teacher_delta_signs": _sign_counts([r["marked_teacher_delta_vs_source_IoU"] for r in group]),
            "masked_top1_unknown_labels": sum(r["masked_teacher_top1_label_unknown"] for r in group),
            "marked_top1_unknown_labels": sum(r["marked_teacher_top1_label_unknown"] for r in group),
            "masked_top1_ties": sum(r["masked_teacher_top_score_tie_count"] > 1 for r in group),
            "marked_top1_ties": sum(r["marked_teacher_top_score_tie_count"] > 1 for r in group),
            "anchor_GT_unknown_cases": sum(not r["anchor_GT_available"] for r in group),
            "all_grid_candidate_IoUs_zero_cases": sum(r["all_grid_candidate_IoUs_zero"] is True for r in group),
        }

    total_cost = {
        "actual_teacher_requests": len(all_teacher),
        "by_kind": cost_by_kind,
        "all_requests": {
            "input_tokens": sum(x["input_tokens"] for x in all_teacher),
            "visual_tokens": sum(x["visual_tokens"] for x in all_teacher),
            "output_tokens": sum(x["output_tokens"] for x in all_teacher),
            "model_forward_plus_cuda_sync_seconds": math.fsum(
                x["model_forward_plus_cuda_sync_seconds"] for x in all_teacher),
            "max_gpu_peak_bytes": max((x["GPU_peak_bytes"] for x in all_teacher), default=None),
            "timing_definition": "per-request model forward plus CUDA synchronization only; the runner starts this timer after PNG loading, image processing, and tensor transfer, so this is not end-to-end request latency",
        },
        "teacher_owned_worker_wall_seconds": float(gate["cost_scope"]["teacher_owned_worker_wall_seconds"]),
        "teacher_owned_worker_wall_definition": "measured actual launch-to-exit wall time for the guarded worker, including its validation/model loading and all 160 jobs; excludes earlier grid preparation and external custody work",
        "per_request_full_end_to_end_seconds": None,
        "grid_preparation_seconds": None,
        "preprocessing_and_transfer_request_latency_seconds": None,
        "unmeasured_timing_fields_remain_unknown": True,
        "student_fits": 0,
        "new_model_downloads": 0,
    }
    report = {
        "schema": "stvg-box-scale-global-evidence-postseal-v1",
        "status": "complete_actual_eight_case_sealed_grid_readback",
        "scope": "one source-frame anchor per historically exposed case; full same-scale coarse grid; original-minus-masked and marked-window teacher rankings",
        "GT_read": True,
        "GT_loaded_only_after_complete_8_case_76_window_160_response_preflight": True,
        "historically_exposed": True,
        "fresh_confirmation": False,
        "protocol_boundary": "single still image per request, not the STVG-R1 native video/ID/temporal protocol; no multi-frame tube or tracking claim",
        "teacher_feedback_semantics": {
            "letters": {"A": "referent support", "B": "contradict or absent", "C": "insufficient evidence"},
            "score": "logit(A)-logsumexp(logit(B),logit(C))",
            "masking_response": "unmarked original support score minus full-frame blurred-window support score; positive means hiding that window reduced measured support",
            "marked_score": "marked-window support log odds; fixed visible ID 1 for every candidate",
            "not_claimed": ["calibrated IoU", "ground-truth teacher box", "student adaptation benefit", "video identity or temporal association"],
        },
        "cohort": {"cases": EXPECTED_CASES, "windows": EXPECTED_WINDOWS,
                   "teacher_requests": EXPECTED_CALLS, "one_fixed_anchor_per_case": True,
                   "coverage": "all generated same-width/height coarse windows over the normalized image, no half-step/refinement or GT filtering"},
        "metrics": {
            "anchor_metric": "single-frame continuous half-open xyxy IoU using source-pixel dimensions and HC2 lower-bound clipping consistent with the official dense evaluator",
            "per_candidate_rows": len(all_candidates),
            "unknown_anchor_GT_is_null": True,
            "probability_reconstruction_max_abs_error": gate["max_probability_error"],
            "support_log_odds_reconstruction_max_abs_error": gate["max_score_error"],
            "rank_ties": "top-1 ties use earliest 1-based row-major window; Spearman uses average ranks for exact score ties",
        },
        "aggregate_by_dataset": aggregate,
        "cases": cases_out,
        "all_candidate_rows": all_candidates,
        "actual_cost": total_cost,
        "provenance": {
            "design_sha256": gate["hashes"]["design"],
            "protocol_sha256": gate["hashes"]["protocol"],
            "runtime_sha256": gate["hashes"]["runtime"],
            "roster_sha256": gate["hashes"]["roster"],
            "grid_input_barrier_sha256": gate["hashes"]["input_barrier"],
            "teacher_job_lock_sha256": gate["hashes"]["job_lock"],
            "teacher_global_barrier_sha256": gate["hashes"]["teacher_barrier"],
            "actual_weight_custody_sha256": gate["hashes"]["weight_custody"],
            "actual_label_contract_sha256": gate["hashes"]["label_contract"],
            "actual_resource_limits_sha256": gate["hashes"]["resource_limits"],
            "actual_launch_receipt_sha256": gate["hashes"]["launch_receipt"],
            "actual_worker_exit_receipt_sha256": gate["hashes"]["worker_exit"],
            "cost_scope_correction_sha256": gate["hashes"]["cost_scope"],
            "official_model_receipt_sha256": gate["hashes"]["official_model_receipt"],
            "protected_current_method_sha256": gate["hashes"]["protected_method"],
            "sealed_teacher_result_inventory_root": gate["hashes"]["teacher_response_inventory"],
            "truth_binding_sha256": _canonical_sha(truth_bindings),
            "reader_sha256": _sha(Path(__file__)),
        },
        "limitations": [
            "The cohort is historically exposed and is not fresh confirmation.",
            "Grid-oracle IoU is post-seal geometry headroom only; it did not enter prompts, rankings, or fitting.",
            "A zero-IoU candidate grid means no positive reward exists within this finite grid; it does not establish that a continuous policy cannot fit.",
            "Per-request timing covers model forward plus CUDA synchronization only, after PNG loading, preprocessing, and transfer; full end-to-end request and grid-preparation times are unknown. The separate worker launch-to-exit time includes model loading and all jobs.",
        ],
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    return report


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable readback: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True,
                               indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def main() -> None:
    gate = _preflight()
    report = _score_after_seal(gate)
    report_path = BASE / "ROOT_POSTSEAL_REPORT.json"
    _write_once(report_path, report)
    receipt = {
        "schema": "stvg-box-scale-global-evidence-readback-v1",
        "status": "pass",
        "cases": EXPECTED_CASES,
        "grid_windows": EXPECTED_WINDOWS,
        "teacher_requests": EXPECTED_CALLS,
        "candidate_rows": len(report["all_candidate_rows"]),
        "GT_read": True,
        "GT_after_complete_seal": True,
        "CPU_only_readback": True,
        "student_fits": 0,
        "report_sha256": _sha(report_path),
        "reader_sha256": _sha(Path(__file__)),
        "teacher_global_barrier_sha256": _sha(TEACHER_BARRIER),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_once(BASE / "ROOT_READBACK.json", receipt)
    print(json.dumps({"status": receipt["status"], "cases": EXPECTED_CASES,
                      "grid_windows": EXPECTED_WINDOWS, "requests": EXPECTED_CALLS,
                      "candidate_rows": len(report["all_candidate_rows"])}, sort_keys=True))


if __name__ == "__main__":
    main()
