"""Prepare a locked, GT-free ECLD teacher input manifest.

This stage reads the already sealed new-crop candidate arrays, verifies their
parent byte/tensor bindings, and writes A/B prompts plus the fixed P0 visual
controls.  It does not load a model, run inference, or score annotations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/stvg_evidence_conditioned_v1"
OUT = BASE / "inputs_teacher"
DESIGN = BASE / "DESIGN_LOCK.json"
ROSTER = ROOT / "artifacts/stvg_evidence_constrained_v1/DEVELOPMENT_ROSTER_LOCK.json"
INPUT_PHASE = ROOT / "artifacts/stvg_evidence_constrained_v1/round2_prompt009"
OLD_TOP1_PHASE = ROOT / "artifacts/stvg_evidence_constrained_v1/round2"
# The first preparation attempt is retained under the original unversioned
# names as a failed-binding snapshot.  The corrected, runnable input package
# is written once under V2 names so no previously written artifact is replaced.
JOB_LOCK = OUT / "TEACHER_JOB_LOCK_V2.json"
INPUT_BARRIER = OUT / "INPUT_BARRIER_V2.json"
LEGACY_REFS = OUT / "LEGACY_TOP1_REFERENCES_V2.json"
CONTRACTS = OUT / "CPU_CONTRACTS_V2.json"

import sys
sys.path.insert(0, str(ROOT))
from scripts import stvg_ecld_teacher_protocol_v1 as protocol


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path.relative_to(ROOT)}")
    return value


def write_json_once(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to replace sealed or partial output: {path.relative_to(ROOT)}")
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_bytes(protocol.canonical_json_bytes(value))
    os.replace(temp, path)


def _is_sha(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(ch in "0123456789abcdef" for ch in value))


def _job_digest(job: Mapping[str, Any]) -> str:
    body = dict(job)
    body.pop("job_binding_sha256", None)
    return protocol.sha256_bytes(protocol.canonical_json_bytes(body))


def _lookup_outputs(barrier: Mapping[str, Any], base: Path, *, expected_count: int,
                    label: str) -> dict[str, dict[str, Any]]:
    rows = barrier.get("files")
    if barrier.get("status") != "complete" or barrier.get("GT_read") is not False:
        raise RuntimeError(f"{label} barrier must be complete and GT-free")
    if barrier.get("count") != expected_count or not isinstance(rows, list) or len(rows) != expected_count:
        raise RuntimeError(f"{label} barrier count is unexpected")
    found: dict[str, dict[str, Any]] = {}
    for row in rows:
        rel = row.get("path")
        if not isinstance(rel, str) or not rel.endswith(".json") or Path(rel).is_absolute() or ".." in Path(rel).parts:
            raise RuntimeError(f"{label} barrier contains an unsafe output path")
        path = ROOT / rel
        if (not path.is_file() or path.stat().st_size != row.get("bytes")
                or sha_file(path) != row.get("sha256")):
            raise RuntimeError(f"{label} output changed: {rel}")
        job_id = path.stem
        if job_id in found:
            raise RuntimeError(f"{label} output job id is duplicated")
        found[job_id] = dict(path=rel, sha256=row["sha256"], bytes=row["bytes"])
    return found


def _validate_roster_and_design() -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    design = read_json(DESIGN)
    roster = read_json(ROSTER)
    rel_roster = str(ROSTER.relative_to(ROOT))
    if design.get("status") != "locked" or design.get("schema") != "stvg-evidence-conditioned-study-v1":
        raise RuntimeError("DESIGN_LOCK is not locked to the ECLD study schema")
    if design.get("GT_before_global_predictions") is not False or design.get("GT_oracle_fits") is not False:
        raise RuntimeError("DESIGN_LOCK does not forbid pre-seal GT reads/oracle fitting")
    if design.get("roster_path") != rel_roster or design.get("roster_sha256") != sha_file(ROSTER):
        raise RuntimeError("DESIGN_LOCK does not bind the original development roster")
    if roster.get("status") != "locked" or roster.get("GT_read") is not False:
        raise RuntimeError("development roster is not a locked GT-free roster")
    records = roster.get("records")
    if not isinstance(records, list) or len(records) != 32:
        raise RuntimeError("expected the fixed 32-source development roster")
    aliases = [r.get("probe_alias") for r in records]
    if len(set(aliases)) != 32 or set(aliases) != {*(f"HC{i:02d}" for i in range(1, 17)),
                                                    *(f"V{i:02d}" for i in range(1, 17))}:
        raise RuntimeError("development aliases differ from the locked 16+16 cohort")
    if design.get("teacher_main_call_cap") != 256 or design.get("teacher_control_call_cap") != 64:
        raise RuntimeError("DESIGN_LOCK call caps do not match 256 main plus 64 controls")
    if design.get("teacher_output_token_cap") != protocol.MAX_NEW_TOKENS:
        raise RuntimeError("DESIGN_LOCK output cap differs from protocol")
    if design.get("teacher_protocols_same_output_options") is not True:
        raise RuntimeError("DESIGN_LOCK does not bind matched A/B output options")
    if design.get("control_uses_GT_selection") is not False:
        raise RuntimeError("P0 controls may not be selected using GT")
    if len(design.get("control_aliases", [])) != 8 or len(set(design["control_aliases"])) != 8:
        raise RuntimeError("DESIGN_LOCK must select exactly eight fixed P0 aliases")
    if design.get("control_center_slots") != [0, 2]:
        raise RuntimeError("P0 control center slots differ from the locked design")
    return design, roster, records


def _parent_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any],
                               dict[tuple[str, int], dict[str, Any]], dict[str, dict[str, Any]]]:
    input_barrier = read_json(INPUT_PHASE / "CANDIDATE_INPUT_BARRIER.json")
    input_jobs = read_json(INPUT_PHASE / "TEACHER_JOB_LOCK.json")
    input_receipt = read_json(INPUT_PHASE / "TEACHER_GLOBAL_BARRIER.json")
    old_jobs = read_json(OLD_TOP1_PHASE / "TEACHER_JOB_LOCK.json")
    old_receipt = read_json(OLD_TOP1_PHASE / "TEACHER_GLOBAL_BARRIER.json")
    if input_barrier.get("status") != "complete" or input_barrier.get("GT_read") is not False or input_barrier.get("count") != 32:
        raise RuntimeError("new_crop input phase is not a complete GT-free 32-source barrier")
    if input_jobs.get("status") != "locked" or input_jobs.get("GT_read") is not False:
        raise RuntimeError("new_crop parent job lock is not locked and GT-free")
    input_job_rows = input_jobs.get("records")
    if not isinstance(input_job_rows, list) or len(input_job_rows) != 512:
        raise RuntimeError("new_crop parent job lock must have the exact 512-job scope")
    input_outputs = _lookup_outputs(input_receipt, INPUT_PHASE, expected_count=512,
                                    label="prompt009 historical teacher")
    old_job_rows = old_jobs.get("records")
    if old_jobs.get("status") != "locked" or old_jobs.get("GT_read") is not False or not isinstance(old_job_rows, list) or len(old_job_rows) != 512:
        raise RuntimeError("original Top-1 comparison job lock is not the exact 512-job scope")
    old_outputs = _lookup_outputs(old_receipt, OLD_TOP1_PHASE, expected_count=512,
                                  label="original Top-1 historical teacher")
    crop_rows = [r for r in input_job_rows if r.get("variant") == "new_crop"]
    old_crop_rows = [r for r in old_job_rows if r.get("variant") == "new_crop"]
    if len(crop_rows) != 128 or len(old_crop_rows) != 128:
        raise RuntimeError("each historical new_crop phase must have exactly 128 jobs")
    crop_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for row in crop_rows:
        key = (str(row.get("case_id")), int(row.get("center_slot", -1)))
        if key in crop_by_key or row.get("GT_read") is not False:
            raise RuntimeError("duplicate or non-GT-free new_crop parent job")
        crop_by_key[key] = row
    old_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for row in old_crop_rows:
        key = (str(row.get("case_id")), int(row.get("center_slot", -1)))
        if key in old_by_key or row.get("GT_read") is not False:
            raise RuntimeError("duplicate or non-GT-free original Top-1 new_crop job")
        old_by_key[key] = row
    if set(crop_by_key) != set(old_by_key) or len(crop_by_key) != 128:
        raise RuntimeError("the two historical new_crop phases do not cover the same case/center grid")
    return (input_barrier, input_jobs, input_receipt, old_receipt,
            crop_by_key, {"prompt009": input_outputs, "original_top1": old_outputs,
                          "input_job_rows": {r["job_id"]: r for r in crop_rows},
                          "old_job_rows": {r["job_id"]: r for r in old_crop_rows},
                          "old_by_key": old_by_key})


def _verify_video_file(video: Mapping[str, Any], parent_files: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    rel = video.get("path")
    if not isinstance(rel, str) or Path(rel).is_absolute() or ".." in Path(rel).parts:
        raise RuntimeError("parent input NPY path is unsafe")
    # prompt009 reused the exact original new_crop tensors; their immutable
    # payloads therefore live in the round2 parent directory, while the
    # prompt009 barrier records the reuse and the current job binding.
    if not rel.startswith("artifacts/stvg_evidence_constrained_v1/round2/new_crop/teacher_inputs/"):
        raise RuntimeError("new_crop input file escaped the locked original tensor directory")
    row = parent_files.get(rel)
    if row is None:
        raise RuntimeError("new_crop NPY is not listed in the parent input barrier")
    path = ROOT / rel
    if not path.is_file() or path.stat().st_size != row.get("bytes") or sha_file(path) != row.get("sha256"):
        raise RuntimeError(f"parent NPY bytes changed: {rel}")
    if video.get("sha256") != row["sha256"] or video.get("dtype") != "uint8" or video.get("tensor_layout") != "TCHW":
        raise RuntimeError("parent job NPY binding differs from the parent input barrier")
    if video.get("frame_shape_chw") != [3, 448, 448] or not _is_sha(video.get("tensor_sha256")):
        raise RuntimeError("parent job tensor metadata is incomplete")
    if (isinstance(video.get("fps"), bool) or not isinstance(video.get("fps"), (int, float))
            or not math.isfinite(float(video["fps"])) or float(video["fps"]) <= 0):
        raise RuntimeError("parent input video FPS is invalid")
    # Tensor SHA is recomputed from the existing, locked NPY without saving or
    # rendering another image.  The worker repeats this check before use.
    import numpy as np
    array = np.load(path, allow_pickle=False)
    if array.dtype != np.uint8 or array.ndim != 4 or array.shape[1:] != (3, 448, 448):
        raise RuntimeError(f"parent NPY shape/dtype is not the locked uint8 TCHW 448 array: {rel}")
    array = np.ascontiguousarray(array)
    tensor_sha = protocol.sha256_bytes(array.tobytes())
    if tensor_sha != video["tensor_sha256"]:
        raise RuntimeError(f"parent NPY tensor bytes differ from their sealed tensor hash: {rel}")
    if len(array) not in (4, 5):
        raise RuntimeError("new_crop input clip must have exactly four context or five local frames")
    return dict(path=rel, sha256=row["sha256"], bytes=int(row["bytes"]),
                tensor_sha256=tensor_sha, shape=list(array.shape), dtype=str(array.dtype),
                fps=float(video["fps"]), clip_role=video.get("clip_role"),
                parent_phase="round2_prompt009_reuses_round2_tensor", parent_variant="new_crop")


def _minimal_bindings(row: Mapping[str, Any], frame_count: int) -> list[dict[str, Any]]:
    out = []
    for binding in row.get("candidate_bindings", []):
        stable = binding.get("candidate_id", binding.get("id"))
        if isinstance(stable, bool) or not isinstance(stable, int):
            raise RuntimeError("candidate binding stable ID is not an integer")
        boxes = binding.get("boxes")
        if not isinstance(boxes, list) or len(boxes) != frame_count:
            raise RuntimeError("candidate presence metadata is not aligned with the local five-frame crop")
        out.append(dict(id=stable, present_by_frame=[box is not None for box in boxes]))
    ids = [x["id"] for x in out]
    if not out or ids != list(range(1, len(ids) + 1)):
        raise RuntimeError("stable candidate bindings must be the frozen contiguous 1..N pool")
    return out


def _make_job(*, parent: Mapping[str, Any], alias: str, dataset: str,
              source_case_sha256: str, query_sha256: str, query_text: str,
              control: str, variant: str, old_top1_ref: Mapping[str, Any],
              input_files: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    candidate_bindings = _minimal_bindings(parent, len(parent["videos"][1]["actual_timestamps_seconds"]))
    candidate_map = protocol._normalise_visible_map(parent.get("visible_id_to_candidate", {}))
    stable_ids = [x["id"] for x in candidate_bindings]
    if sorted(candidate_map.values()) != stable_ids or int(parent.get("candidate_count", -1)) != len(stable_ids):
        raise RuntimeError("candidate pool mapping/count differs from the preserved new_crop inputs")
    source_rows = [r for r in parent["candidate_bindings"] if r.get("role") == "Source"]
    if len(source_rows) != 1:
        raise RuntimeError("new_crop candidate set must have exactly one Source candidate")
    source_id = source_rows[0].get("candidate_id", source_rows[0].get("id"))
    if source_id != 1:
        raise RuntimeError("current frozen design binds the Source stable candidate ID to 1")

    old_to_new = None
    visible_map = candidate_map
    if control == "id_permutation":
        visible_map, old_to_new = protocol.cyclic_id_permutation(candidate_map)
    elif control not in {"base", "visual_null"}:
        raise ValueError(f"unknown locked condition {control}")

    videos = []
    for video in parent["videos"]:
        parent_file = input_files[video["path"]]
        videos.append({**parent_file,
                       "actual_timestamps_seconds":[float(x) for x in video["actual_timestamps_seconds"]],
                       "actual_gap_seconds":[float(x) for x in video["actual_gap_seconds"]],
                       "encoder_uniform_spacing_approximation":bool(video["encoder_uniform_spacing_approximation"])})
    job_id = f"{alias}_new_crop_{int(parent['center_slot'])}_{variant}_{control}"
    prompt = protocol.build_prompt(
        query_text=query_text,
        context_times=videos[0]["actual_timestamps_seconds"],
        local_times=videos[1]["actual_timestamps_seconds"],
        visible_id_to_candidate=visible_map,
        source_candidate_id=source_id,
        protocol_variant=variant,
        control=control,
    )
    body = dict(
        schema="stvg-ecld-teacher-job-v1", job_id=job_id,
        case_id=alias, dataset=dataset, center_slot=int(parent["center_slot"]),
        center_physical_frame_id=int(parent["center_physical_frame_id"]),
        protocol_variant=variant, control=control,
        study_phase="P1_main" if control == "base" else "P0_fixed_control",
        candidate_count=len(stable_ids), stable_candidate_ids=stable_ids,
        source_candidate_id=source_id,
        base_visible_id_to_candidate=candidate_map,
        visible_id_to_candidate=visible_map,
        old_to_new_visible_id=old_to_new,
        allowed_output_ids=sorted(map(int, visible_map)),
        candidate_bindings=candidate_bindings,
        videos=videos,
        prompt=prompt,
        prompt_sha256=protocol.sha256_bytes(prompt.encode("utf-8")),
        query_sha256=query_sha256,
        source_case_sha256=source_case_sha256,
        source_request_sha256=parent.get("source_request_sha256"),
        parent_job_id=parent.get("job_id"),
        parent_job_schema_sha256=protocol.canonical_sha256(parent),
        parent_candidate_input_barrier_sha256=None,
        legacy_original_top1=dict(old_top1_ref),
        output_token_cap=protocol.MAX_NEW_TOKENS,
        parsed_output_schema=protocol.OUTPUT_SCHEMA,
        GT_read=False,
    )
    body["job_binding_sha256"] = _job_digest(body)
    return body


def build_manifests(*, design: Mapping[str, Any], roster: Mapping[str, Any],
                    roster_records: Sequence[Mapping[str, Any]], old_jobs_by_key: Mapping[tuple[str, int], Mapping[str, Any]],
                    new_jobs_by_key: Mapping[tuple[str, int], Mapping[str, Any]],
                    response_paths: Mapping[str, Mapping[str, Mapping[str, Any]]],
                    parent_file_rows: Mapping[str, Mapping[str, Any]],
                    parent_input_barrier_sha256: str,
                    parent_job_lock_sha256: str,
                    parent_input_teacher_barrier_sha256: str,
                    original_top1_barrier_sha256: str,
                    design_lock_sha256: str,
                    development_roster_sha256: str,
                    query_by_alias: Mapping[str, tuple[str, str, str]]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    control_aliases = list(design["control_aliases"])
    control_slots = [int(x) for x in design["control_center_slots"]]
    if set(control_aliases) - {str(r["probe_alias"]) for r in roster_records}:
        raise RuntimeError("P0 alias list is outside the development roster")
    record_by_alias = {str(r["probe_alias"]): r for r in roster_records}
    jobs: list[dict[str, Any]] = []
    legacy: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    unique_parent_files: dict[str, dict[str, Any]] = {}
    for rec in roster_records:
        alias = str(rec["probe_alias"])
        dataset = str(rec["dataset"])
        source_case_sha256, query_sha256, query_text = query_by_alias[alias]
        main_count = control_count = 0
        for slot in range(4):
            key = (alias, slot)
            parent = new_jobs_by_key.get(key)
            old_parent = old_jobs_by_key.get(key)
            if parent is None or old_parent is None:
                raise RuntimeError(f"locked new_crop job is missing for {alias} center {slot}")
            if (parent.get("case_id") != alias or parent.get("dataset") != dataset
                    or parent.get("source_case_sha256") != source_case_sha256
                    or parent.get("source_request_sha256") != old_parent.get("source_request_sha256")):
                raise RuntimeError(f"new_crop parent job source binding differs for {alias} center {slot}")
            if old_parent.get("source_case_sha256") != source_case_sha256:
                raise RuntimeError("historical Top-1 reference is not bound to this exact frozen source case")
            original_ref = response_paths["original_top1"].get(str(old_parent["job_id"]))
            prompt009_ref = response_paths["prompt009"].get(str(parent["job_id"]))
            if original_ref is None or prompt009_ref is None:
                raise RuntimeError("historical Top-1 or prompt009 response is missing from its sealed barrier")
            legacy.append(dict(case_id=alias, dataset=dataset, center_slot=slot,
                               original_top1=dict(path=original_ref["path"], sha256=original_ref["sha256"],
                                                  bytes=original_ref["bytes"], phase="round2",
                                                  interpretation="historical_original_top1_non_equivalent_to_A"),
                               prompt009_historical=dict(path=prompt009_ref["path"], sha256=prompt009_ref["sha256"],
                                                        bytes=prompt009_ref["bytes"], phase="round2_prompt009",
                                                        interpretation="historical_prior_prompt_not_A_equivalent"),
                               parent_job_id=parent["job_id"], source_case_sha256=source_case_sha256,
                               source_request_sha256=parent["source_request_sha256"], GT_read=False))
            parent_files: dict[str, dict[str, Any]] = {}
            for video in parent.get("videos", []):
                if not isinstance(video, Mapping):
                    raise RuntimeError("parent video spec is malformed")
                desc = _verify_video_file(video, parent_file_rows)
                previous = unique_parent_files.get(desc["path"])
                if previous is not None and previous != desc:
                    raise RuntimeError("same parent NPY path has conflicting bindings")
                unique_parent_files[desc["path"]] = desc
                parent_files[desc["path"]] = desc
            ref = dict(original_top1=original_ref, prompt009_historical=prompt009_ref)
            for variant in protocol.PROMPT_VARIANTS:
                job = _make_job(parent=parent, alias=alias, dataset=dataset,
                                source_case_sha256=source_case_sha256,
                                query_sha256=query_sha256, query_text=query_text,
                                control="base", variant=variant, old_top1_ref=ref,
                                input_files=parent_files)
                job["parent_candidate_input_barrier_sha256"] = parent_input_barrier_sha256
                job["parent_teacher_job_lock_sha256"] = parent_job_lock_sha256
                job["job_binding_sha256"] = _job_digest(job)
                jobs.append(job)
                main_count += 1
            if alias in control_aliases and slot in control_slots:
                for control in ("id_permutation", "visual_null"):
                    for variant in protocol.PROMPT_VARIANTS:
                        job = _make_job(parent=parent, alias=alias, dataset=dataset,
                                        source_case_sha256=source_case_sha256,
                                        query_sha256=query_sha256, query_text=query_text,
                                        control=control, variant=variant, old_top1_ref=ref,
                                        input_files=parent_files)
                        job["parent_candidate_input_barrier_sha256"] = parent_input_barrier_sha256
                        job["parent_teacher_job_lock_sha256"] = parent_job_lock_sha256
                        job["job_binding_sha256"] = _job_digest(job)
                        jobs.append(job)
                        control_count += 1
        case_rows.append(dict(case_id=alias, dataset=dataset,
                              source_case_sha256=source_case_sha256,
                              query_sha256=query_sha256, center_count=4,
                              main_job_count=main_count,
                              control_job_count=control_count,
                              total_job_count=main_count + control_count,
                              GT_read=False))
    if len(jobs) != 320 or sum(x["study_phase"] == "P1_main" for x in jobs) != 256 or sum(x["study_phase"] == "P0_fixed_control" for x in jobs) != 64:
        raise RuntimeError("job roster differs from the locked 256+64 design")
    ids = [j["job_id"] for j in jobs]
    if len(set(ids)) != len(ids):
        raise RuntimeError("ECLD teacher job IDs are not unique")
    expected_per_alias = {str(r["probe_alias"]): (16 if str(r["probe_alias"]) in control_aliases else 8)
                          for r in roster_records}
    for row in case_rows:
        if row["total_job_count"] != expected_per_alias[row["case_id"]]:
            raise RuntimeError("per-case job count differs from the fixed P0/P1 allocation")

    lock = dict(schema="stvg-ecld-teacher-job-lock-v1", status="locked", count=len(jobs),
                main_job_count=256, control_job_count=64,
                design_lock_sha256=design_lock_sha256,
                development_roster_sha256=development_roster_sha256,
                parent_candidate_input_barrier_sha256=parent_input_barrier_sha256,
                parent_teacher_job_lock_sha256=parent_job_lock_sha256,
                parent_teacher_barrier_sha256=parent_input_teacher_barrier_sha256,
                original_top1_teacher_barrier_sha256=original_top1_barrier_sha256,
                parsed_output_schema=protocol.OUTPUT_SCHEMA, output_token_cap=protocol.MAX_NEW_TOKENS,
                records=jobs, GT_read=False)
    legacy_lock = dict(schema="stvg-ecld-legacy-top1-reference-v1", status="read_only_reference",
                       count=len(legacy), source_phase="artifacts/stvg_evidence_constrained_v1/round2",
                       historical_original_top1=True, equivalent_to_protocol_A=False,
                       records=legacy, GT_read=False)
    return lock, legacy_lock, dict(case_rows=case_rows, parent_files=sorted(unique_parent_files.values(), key=lambda x: x["path"]))


def _synthetic_contracts() -> dict[str, Any]:
    from scripts.stvg_ecld_teacher_protocol_v1 import cpu_contracts
    base = cpu_contracts()
    fake_records = [dict(probe_alias=(f"HC{i:02d}" if i <= 16 else f"V{i-16:02d}"),
                         dataset=("hc2" if i <= 16 else "vidstg")) for i in range(1, 33)]
    main = sum(4 * 2 for _ in fake_records)
    controls = sum(2 * 2 * 2 for x in fake_records if x["probe_alias"] in ["HC01", "HC02", "HC03", "HC04", "V01", "V02", "V03", "V04"])
    assert main == 256 and controls == 64 and main + controls == 320
    assert len({f"{r['probe_alias']}_new_crop_{s}_{v}_{c}" for r in fake_records for s in range(4)
                for v in ("A", "B") for c in (("base",) if r["probe_alias"] not in {"HC01","HC02","HC03","HC04","V01","V02","V03","V04"} or s not in (0,2)
                                                     else ("base","id_permutation","visual_null"))}) == 320
    return dict(schema="stvg-ecld-teacher-preparation-contracts-v1", status="pass",
                check_count=base["check_count"] + 4,
                checks=base["checks"] + ["locked_32_source_cardinality", "256_main_job_count",
                                          "64_fixed_control_job_count", "unique_case_center_prompt_control_job_ids"],
                GT_read=False, CPU_only=True, model_loaded=False, GPU_used=False,
                annotation_scoring=False, synthetic_roster_only=True)


def prepare() -> dict[str, Any]:
    design, roster, records = _validate_roster_and_design()
    input_barrier, input_jobs, input_receipt, old_receipt, crop_by_key, extra = _parent_inputs()
    expected_refs = {
        "prompt009": len([r for r in input_receipt["files"] if r.get("path")]),
        "original_top1": len([r for r in old_receipt["files"] if r.get("path")]),
    }
    if expected_refs != {"prompt009": 512, "original_top1": 512}:
        raise RuntimeError("historical teacher file receipts changed cardinality")
    input_file_rows = {r["path"]: r for r in input_barrier.get("files", []) if isinstance(r, Mapping)}
    if len(input_file_rows) != len(input_barrier.get("files", [])):
        raise RuntimeError("parent candidate input file table has duplicate paths")
    response_paths = extra
    response_paths["prompt009"] = extra["prompt009"]
    response_paths["original_top1"] = extra["original_top1"]

    # Make sure the historical response files we cite belong to the matching
    # job identities and preserve the frozen new_crop input/source bindings.
    for key, parent in crop_by_key.items():
        original = extra["old_by_key"][key]
        if original.get("job_id") not in response_paths["original_top1"]:
            raise RuntimeError("original Top-1 response is absent for a locked center")
        if parent.get("source_case_sha256") != original.get("source_case_sha256"):
            raise RuntimeError("historical Top-1 response source case differs from the new_crop parent")
        for left, right in zip(parent.get("videos", []), original.get("videos", [])):
            for field in ("sha256", "tensor_sha256", "fps"):
                if left.get(field) != right.get(field):
                    raise RuntimeError("original Top-1 and new_crop history do not share the same input video tensor")

    # Query text is transiently read from the already locked no-GT development
    # plan, used to form private prompts, and never copied as a standalone field.
    from scripts.run_stvg_table1_source_v1 import _validate_roster_record
    query_by_alias: dict[str, tuple[str, str, str]] = {}
    for rec in records:
        alias = str(rec["probe_alias"])
        ds, row = _validate_roster_record(rec)
        if ds != rec.get("dataset") or not isinstance(row.get("input"), Mapping):
            raise RuntimeError("transient development row does not match its locked roster alias")
        query_text = row["input"].get("caption")
        if not isinstance(query_text, str) or not query_text.strip():
            raise RuntimeError("locked source row has no query text")
        query_sha = protocol.sha256_bytes(query_text.encode("utf-8"))
        query_by_alias[alias] = (str(rec.get("source_case_sha256", "")), query_sha, query_text)
        # The roster binds plan and subject; source-case SHA is added by the
        # parent new_crop job, so use it as the authoritative binding below.
        if not query_by_alias[alias][0]:
            parent = next(v for (cid, _), v in crop_by_key.items() if cid == alias)
            query_by_alias[alias] = (str(parent["source_case_sha256"]), query_sha, query_text)

    # Parent NPY bytes and tensor content are checked exactly once here, then
    # checked again at each load by the teacher worker.  Keep only references.
    lock, legacy_lock, body = build_manifests(
        design=design, roster=roster, roster_records=records,
        old_jobs_by_key=extra["old_by_key"], new_jobs_by_key=crop_by_key,
        response_paths=response_paths, parent_file_rows=input_file_rows,
        parent_input_barrier_sha256=sha_file(INPUT_PHASE / "CANDIDATE_INPUT_BARRIER.json"),
        parent_job_lock_sha256=sha_file(INPUT_PHASE / "TEACHER_JOB_LOCK.json"),
        parent_input_teacher_barrier_sha256=sha_file(INPUT_PHASE / "TEACHER_GLOBAL_BARRIER.json"),
        original_top1_barrier_sha256=sha_file(OLD_TOP1_PHASE / "TEACHER_GLOBAL_BARRIER.json"),
        design_lock_sha256=sha_file(DESIGN),
        development_roster_sha256=sha_file(ROSTER),
        query_by_alias=query_by_alias)
    job_lock_bytes = protocol.canonical_json_bytes(lock)
    legacy_bytes = protocol.canonical_json_bytes(legacy_lock)
    job_lock_sha = protocol.sha256_bytes(job_lock_bytes)
    legacy_sha = protocol.sha256_bytes(legacy_bytes)
    contracts = _synthetic_contracts()
    protocol_sha = sha_file(ROOT / "scripts/stvg_ecld_teacher_protocol_v1.py")
    prepare_sha = sha_file(Path(__file__))
    write_json_once(CONTRACTS, contracts)
    write_json_once(JOB_LOCK, lock)
    write_json_once(LEGACY_REFS, legacy_lock)
    barrier = dict(
        schema="stvg-ecld-teacher-input-barrier-v1", status="complete", count=32,
        job_count=320, main_job_count=256, control_job_count=64,
        control_aliases=list(design["control_aliases"]), control_center_slots=list(design["control_center_slots"]),
        bindings=dict(
            design_lock_path=str(DESIGN.relative_to(ROOT)), design_lock_sha256=sha_file(DESIGN),
            development_roster_path=str(ROSTER.relative_to(ROOT)), development_roster_sha256=sha_file(ROSTER),
            parent_candidate_input_barrier_path=str((INPUT_PHASE / "CANDIDATE_INPUT_BARRIER.json").relative_to(ROOT)),
            parent_candidate_input_barrier_sha256=sha_file(INPUT_PHASE / "CANDIDATE_INPUT_BARRIER.json"),
            parent_teacher_job_lock_path=str((INPUT_PHASE / "TEACHER_JOB_LOCK.json").relative_to(ROOT)),
            parent_teacher_job_lock_sha256=sha_file(INPUT_PHASE / "TEACHER_JOB_LOCK.json"),
            parent_teacher_global_barrier_path=str((INPUT_PHASE / "TEACHER_GLOBAL_BARRIER.json").relative_to(ROOT)),
            parent_teacher_global_barrier_sha256=sha_file(INPUT_PHASE / "TEACHER_GLOBAL_BARRIER.json"),
            historical_original_top1_barrier_path=str((OLD_TOP1_PHASE / "TEACHER_GLOBAL_BARRIER.json").relative_to(ROOT)),
            historical_original_top1_barrier_sha256=sha_file(OLD_TOP1_PHASE / "TEACHER_GLOBAL_BARRIER.json"),
            job_lock_path=str(JOB_LOCK.relative_to(ROOT)), job_lock_sha256=job_lock_sha,
            legacy_top1_references_path=str(LEGACY_REFS.relative_to(ROOT)), legacy_top1_references_sha256=legacy_sha,
            protocol_path="scripts/stvg_ecld_teacher_protocol_v1.py", protocol_sha256=protocol_sha,
            preparation_path="scripts/prepare_stvg_ecld_teacher_v1.py", preparation_sha256=prepare_sha,
            cpu_contracts_path=str(CONTRACTS.relative_to(ROOT)), cpu_contracts_sha256=sha_file(CONTRACTS),
        ),
        cases=body["case_rows"],
        files=body["parent_files"],
        raw_queries_separate=False,
        query_text_in_private_prompts=True,
        transforms="in-memory only; no derived video NPYs written",
        GT_read=False, model_loaded=False, GPU_used=False, annotation_scoring=False)
    write_json_once(INPUT_BARRIER, barrier)
    return dict(status="complete", case_count=32, job_count=320,
                main_job_count=256, control_job_count=64,
                parent_npy_count=len(body["parent_files"]),
                job_lock_sha256=job_lock_sha, legacy_top1_references_sha256=legacy_sha,
                input_barrier_sha256=sha_file(INPUT_BARRIER), GT_read=False,
                model_loaded=False, GPU_used=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        result = _synthetic_contracts()
        print(json.dumps(result, sort_keys=True))
        if not args.prepare:
            return
    if args.prepare:
        result = prepare()
        print(json.dumps(result, sort_keys=True))
        return
    parser.error("choose --self-test and/or --prepare")


if __name__ == "__main__":
    main()
