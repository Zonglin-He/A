"""Prepare fixed-pool CRC inputs from sealed round2_prompt009 new_crop.

CPU-only: verifies existing locks/NPYs, reads approved query metadata to build
private prompts, and writes immutable manifests. No model, GT, scoring, or video.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/stvg_contrastive_correction_v1"
PARENT = ROOT / "artifacts/stvg_evidence_constrained_v1/round2_prompt009"
OLD = ROOT / "artifacts/stvg_evidence_constrained_v1"
SOURCE_CASES = ROOT / "artifacts/stvg_final_table1_v1/source_cases"
OUT = BASE / "inputs_teacher"
DEV_ROSTER = OLD / "DEVELOPMENT_ROSTER_LOCK.json"
AUTHORITY_ATTACHMENT_SHA256 = "7bbbe456764d64a226a70d7488762fc12f28ff85e17b0df1942d3cbddd2a89ca"

import sys
sys.path.insert(0, str(ROOT))
from scripts import stvg_crc_protocol_v1 as protocol


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


def write_once(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to replace existing artifact: {path.relative_to(ROOT)}")
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_bytes(protocol.canonical_json_bytes(value))
    os.replace(temp, path)


def safe_file(rel: Any) -> Path:
    if not isinstance(rel, str) or Path(rel).is_absolute() or ".." in Path(rel).parts:
        raise RuntimeError("unsafe parent relative path")
    path = (ROOT / rel).resolve(strict=True)
    if not path.is_relative_to(ROOT.resolve()) or not path.is_file():
        raise RuntimeError("parent file escaped root or is not a regular file")
    return path


def _verify_barrier_files(rows: Any, label: str, *, expected_count: int | None = None,
                          required_prefix: str | None = None) -> dict[str, dict[str, Any]]:
    if not isinstance(rows, list) or (expected_count is not None and len(rows) != expected_count):
        raise RuntimeError(f"{label} file list count mismatch")
    found = {}
    for row in rows:
        rel = row.get("path")
        if not isinstance(rel, str) or (required_prefix and not rel.startswith(required_prefix)):
            raise RuntimeError(f"{label} path outside fixed parent scope")
        path = safe_file(rel)
        if path.stat().st_size != row.get("bytes") or sha_file(path) != row.get("sha256"):
            raise RuntimeError(f"{label} bytes/SHA mismatch: {rel}")
        if rel in found:
            raise RuntimeError(f"{label} has duplicate path")
        found[rel] = dict(path=rel, sha256=row["sha256"], bytes=row["bytes"])
    return found


def _verify_parent() -> dict[str, Any]:
    paths = {
        "input": PARENT / "CANDIDATE_INPUT_BARRIER.json",
        "jobs": PARENT / "TEACHER_JOB_LOCK.json",
        "teacher": PARENT / "TEACHER_GLOBAL_BARRIER.json",
        "refs": PARENT / "REFERENCE_LOCK.json",
        "students": PARENT / "STUDENT_GLOBAL_BARRIER.json",
        "dev": DEV_ROSTER,
    }
    objs = {k: read_json(p) for k, p in paths.items()}
    for key, status, count in [
        ("input", "complete", 32), ("jobs", "locked", None),
        ("teacher", "complete", 512), ("refs", "locked", None),
        ("students", "complete", 32), ("dev", "locked", None),
    ]:
        obj = objs[key]
        if obj.get("status") != status or obj.get("GT_read") is not False:
            raise RuntimeError(f"parent {key} is not sealed and GT-free")
        if count is not None and obj.get("count") != count:
            raise RuntimeError(f"parent {key} count mismatch")
    all_jobs = objs["jobs"].get("records")
    if not isinstance(all_jobs, list) or len(all_jobs) != 512:
        raise RuntimeError("prompt009 parent job lock must contain 512 jobs")
    if (not isinstance(objs["refs"].get("records"), list) or len(objs["refs"]["records"]) != 32
            or not isinstance(objs["dev"].get("records"), list) or len(objs["dev"]["records"]) != 32):
        raise RuntimeError("parent reference/development locks must contain the fixed 32 aliases")
    job_by_id = {str(j.get("job_id")): j for j in all_jobs}
    if len(job_by_id) != 512:
        raise RuntimeError("prompt009 job IDs are duplicated")
    newcrop = [j for j in all_jobs if j.get("variant") == "new_crop"]
    if len(newcrop) != 128 or any(j.get("GT_read") is not False for j in newcrop):
        raise RuntimeError("prompt009 must have exactly 128 GT-free new_crop jobs")
    by_key = {(str(j["case_id"]), int(j["center_slot"])): j for j in newcrop}
    if len(by_key) != 128:
        raise RuntimeError("prompt009 new_crop case/center slots are duplicated")
    input_files = _verify_barrier_files(objs["input"].get("files"), "prompt009 input")
    teacher_files = _verify_barrier_files(objs["teacher"].get("files"), "prompt009 teacher",
        expected_count=512, required_prefix=str(PARENT.relative_to(ROOT)) + "/teacher/")
    responses = {Path(p).stem: r for p, r in teacher_files.items()}
    if len(responses) != 512:
        raise RuntimeError("prompt009 response files are not unique")
    hashes = {
        "parent_input_barrier_sha256": sha_file(paths["input"]),
        "parent_job_lock_sha256": sha_file(paths["jobs"]),
        "parent_teacher_barrier_sha256": sha_file(paths["teacher"]),
        "parent_reference_lock_sha256": sha_file(paths["refs"]),
        "parent_student_barrier_sha256": sha_file(paths["students"]),
        "development_roster_sha256": sha_file(paths["dev"]),
    }
    return dict(**objs, paths=paths, hashes=hashes, newcrop=newcrop,
                job_by_key=by_key, parent_files=input_files, responses=responses)


def _box(value: Any) -> list[float] | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 4:
        raise RuntimeError("candidate box must be null or four normalized xyxy coordinates")
    b = [float(x) for x in value]
    if not all(math.isfinite(x) for x in b):
        raise RuntimeError("candidate box is nonfinite")
    if not (0 <= b[0] < b[2] <= 1 and 0 <= b[1] < b[3] <= 1):
        raise RuntimeError("candidate box is outside normalized xyxy coordinates")
    return b


def _verify_video(spec: Mapping[str, Any], parent_files: Mapping[str, Mapping[str, Any]],
                  cache: dict[str, dict[str, Any]]) -> dict[str, Any]:
    import numpy as np
    rel = spec.get("path")
    if not isinstance(rel, str) or not rel.endswith(".npy"):
        raise RuntimeError("locked new_crop input must be an NPY")
    if not rel.startswith("artifacts/stvg_evidence_constrained_v1/round2/new_crop/teacher_inputs/"):
        raise RuntimeError("new_crop NPY escaped the fixed historical input directory")
    file_row = parent_files.get(rel)
    if file_row is None or spec.get("sha256") != file_row["sha256"]:
        raise RuntimeError("job NPY is not bound by parent input barrier")
    if rel not in cache:
        path = safe_file(rel)
        if path.stat().st_size != file_row["bytes"] or sha_file(path) != file_row["sha256"]:
            raise RuntimeError(f"parent NPY bytes/SHA mismatch: {rel}")
        if spec.get("dtype") != "uint8" or spec.get("tensor_layout") != "TCHW" or spec.get("frame_shape_chw") != [3, 448, 448]:
            raise RuntimeError("parent NPY job metadata changed")
        arr = np.load(path, allow_pickle=False)
        if arr.dtype != np.uint8 or arr.ndim != 4 or tuple(arr.shape[1:]) != (3, 448, 448):
            raise RuntimeError("parent NPY tensor shape/dtype changed")
        arr = np.ascontiguousarray(arr)
        if len(arr) not in (4, 5):
            raise RuntimeError("new_crop parent tensor must retain four context or five local frames")
        tensor_sha = protocol.sha256_bytes(arr.tobytes())
        if tensor_sha != spec.get("tensor_sha256"):
            raise RuntimeError("parent NPY tensor SHA mismatch")
        fps = spec.get("fps")
        if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(float(fps)) or float(fps) <= 0:
            raise RuntimeError("parent NPY fps is invalid")
        cache[rel] = dict(path=rel, sha256=file_row["sha256"], bytes=file_row["bytes"],
            tensor_sha256=tensor_sha, shape=list(arr.shape), dtype=str(arr.dtype), fps=float(fps),
            clip_role=spec.get("clip_role"),
            actual_timestamps_seconds=[float(x) for x in spec["actual_timestamps_seconds"]],
            actual_gap_seconds=[float(x) for x in spec["actual_gap_seconds"]],
            encoder_uniform_spacing_approximation=bool(spec["encoder_uniform_spacing_approximation"]))
    result = dict(cache[rel])
    if float(spec["fps"]) != result["fps"] or spec.get("tensor_sha256") != result["tensor_sha256"]:
        raise RuntimeError("same NPY reused with inconsistent timing or tensor binding")
    return result


def _pair_candidates(parent_job: Mapping[str, Any], response: Mapping[str, Any]) -> tuple[int | None, int | None, list[float] | None, float | None, str]:
    parsed = response.get("parsed", {})
    if (response.get("status") != "actual_video_candidate_verification"
            or response.get("GT_read") is not False):
        return None, None, None, None, "parent_top1_invalid_or_unbound"
    if parsed.get("status") != "valid" or response.get("output_truncated_at_cap") is not False:
        return None, None, None, None, "parent_top1_invalid_or_truncated"
    stable = parsed.get("selected_candidate_id")
    visible = parsed.get("visible_id")
    if stable is None:
        return None, visible if isinstance(visible, int) else None, None, None, "parent_top1_abstained"
    if isinstance(stable, bool) or not isinstance(stable, int):
        return None, None, None, None, "selected_id_not_integer"
    candidate_rows = parent_job.get("candidate_bindings", [])
    binding = next((b for b in candidate_rows if b.get("candidate_id", b.get("id")) == stable), None)
    mapping = parent_job.get("visible_id_to_candidate", {})
    stable_from_visible = mapping.get(str(visible), mapping.get(visible)) if isinstance(visible, int) and not isinstance(visible, bool) else None
    if binding is None or stable_from_visible != stable:
        return None, visible if isinstance(visible, int) else None, None, None, "selected_id_not_in_pool"
    if stable == 1:
        return stable, visible, None, float(binding.get("presence", 0)), "parent_top1_selected_source"
    presence = float(binding.get("presence", 0.0))
    boxes = binding.get("boxes")
    if not isinstance(boxes, list) or len(boxes) != 5:
        raise RuntimeError("prompt009 Top-1 box sequence must contain five local positions")
    center_box = _box(boxes[2])
    if _box(binding.get("center_box")) != center_box:
        raise RuntimeError("prompt009 center box differs from the fixed center local frame")
    if presence < 0.6:
        return stable, visible, center_box, presence, "presence_below_0.6"
    if center_box is None:
        return stable, visible, None, presence, "selected_candidate_missing_center_box"
    return stable, visible, center_box, presence, "pairable"


def _replay_parent_parse(parent_job: Mapping[str, Any], response: Mapping[str, Any]) -> None:
    """Replay the prior ID parser without retaining or printing its text."""
    import re
    raw = response.get("raw_output")
    if not isinstance(raw, str):
        raise RuntimeError("prompt009 response has no raw output to replay")
    parsed = response.get("parsed", {})
    fence = chr(96) * 3
    try:
        clean = re.sub(r"^" + re.escape(fence) + r"(?:json)?\s*", "", raw.strip())
        clean = re.sub(r"\s*" + re.escape(fence) + r"$", "", clean)
        value = json.loads(clean)
        visible = value["id"]
        if isinstance(visible, bool) or not isinstance(visible, int) or visible not in parent_job["allowed_output_ids"]:
            raise ValueError("visible id outside fixed parent output enum")
        stable = parent_job["visible_id_to_candidate"].get(str(visible))
        if (parsed.get("status") != "valid" or parsed.get("visible_id") != visible
                or parsed.get("selected_candidate_id") != stable):
            raise RuntimeError("parent parsed selection does not replay from its locked raw response")
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError("parent Top-1 raw response is not valid under the original parser") from exc


def prepare() -> dict[str, Any]:
    outputs = [BASE/"DESIGN_LOCK.json", BASE/"ROSTER_LOCK.json", OUT/"INPUT_BARRIER.json",
               OUT/"TEACHER_JOB_LOCK.json", OUT/"PARENT_TOP1_CUSTODY.json", OUT/"CPU_CONTRACTS.json"]
    if any(p.exists() for p in outputs):
        raise FileExistsError("CRC immutable preparation files already exist")
    p = _verify_parent()
    dev_rows = p["dev"].get("records")
    if not isinstance(dev_rows, list) or len(dev_rows) != 32:
        raise RuntimeError("original development roster must contain 32 rows")
    dev_by_alias = {str(r["probe_alias"]): r for r in dev_rows}
    if len(dev_by_alias) != 32:
        raise RuntimeError("original development aliases are duplicated")
    from scripts.run_stvg_table1_source_v1 import _validate_roster_record
    new_roster = []
    query_by_alias = {}
    source_path_by_alias = {}
    for alias, rec in sorted(dev_by_alias.items()):
        ds, row = _validate_roster_record(rec)
        query = row.get("input", {}).get("caption")
        if ds != rec["dataset"] or not isinstance(query, str) or not query.strip():
            raise RuntimeError("validated source row does not match development roster")
        source_path = SOURCE_CASES / f"{alias}.pt"
        sidecar = source_path.with_suffix(".receipt.json")
        if not source_path.is_file() or not sidecar.is_file():
            raise RuntimeError(f"source-case artifact or receipt is missing for {alias}")
        receipt = read_json(sidecar)
        if (sha_file(source_path) != receipt.get("sha256") or source_path.stat().st_size != receipt.get("bytes")
                or receipt.get("query_key_sha256") != rec["query_key_sha256"] or receipt.get("GT_read") is not False):
            raise RuntimeError(f"source-case receipt mismatch for {alias}")
        matching = [x for x in p["newcrop"] if str(x["case_id"]) == alias]
        if len(matching) != 4:
            raise RuntimeError(f"expected four prompt009 slots for {alias}")
        source_hashes = {x.get("source_case_sha256") for x in matching}
        request_hashes = {x.get("source_request_sha256") for x in matching}
        if len(source_hashes) != 1 or len(request_hashes) != 1 or next(iter(source_hashes)) != receipt.get("source_case_sha256"):
            raise RuntimeError(f"prompt009 and source receipt disagree for {alias}")
        rr = dict(rec)
        rr.update(case_id=alias, source_case_path=str(source_path.relative_to(ROOT)),
                  source_case_file_sha256=receipt["sha256"], source_case_receipt_sha256=sha_file(sidecar),
                  source_case_sha256=next(iter(source_hashes)), source_request_sha256=next(iter(request_hashes)))
        new_roster.append(rr)
        query_by_alias[alias] = query
        source_path_by_alias[alias] = source_path
    roster_obj = dict(schema="stvg-crc-roster-v1", status="locked", count=32,
                      records=new_roster, GT_read=False,
                      parent_development_roster_sha256=p["hashes"]["development_roster_sha256"],
                      previous_development_GT_exposure=p["dev"].get("previous_development_GT_exposure"))
    write_once(BASE/"ROSTER_LOCK.json", roster_obj)
    roster_sha = sha_file(BASE/"ROSTER_LOCK.json")
    spatial_configs = {
      "hc2": {"lr":0.003535533905932738,"sigma":0.0125,"tau":0.2,"steps":5,"writeback":0.0625,"samples":32},
      "vidstg": {"lr":0.0075,"sigma":0.2,"tau":1.0,"steps":5,"writeback":0.125,"samples":32}}
    design = dict(schema="stvg-crc-design-v1", status="locked", study="contrastive referent correction",
      authority_attachment_sha256=AUTHORITY_ATTACHMENT_SHA256,
      roster_path=str((BASE/"ROSTER_LOCK.json").relative_to(ROOT)), roster_sha256=roster_sha,
      parent_scope="round2_prompt009 new_crop Top-1 only; never substitute the earlier round2 Top-1",
      source_population="same historical 32-source development cohort; prior development exposure is retained",
      source_count=32, slots=128, centers_per_source=4, pair_job_count=49,
      pair_admission="valid uncapped prompt009 selection of non-Source candidate in its original pool, presence >=0.6, nonempty center-frame box",
      labels=list(protocol.LABELS), source_candidate_id=1,
      alternative_geometry="copied from locked original Top-1 candidate; teacher label never supplies or changes geometry",
      input_policy="reuse both exact parent NPYs, original nine candidate tiles and physical clock; no rerender, proposal, or candidate regeneration",
      generation=protocol.EXPECTED_GENERATION, spatial_configs=spatial_configs,
      bootstrap_executed=False, GT_read=False, GT_before_global_predictions=False,
      GT_oracle_fits=False, model_loaded=False, GPU_used=False, scoring_executed=False,
      download_executed=False, parent_bindings=dict(p["hashes"]))
    write_once(BASE/"DESIGN_LOCK.json", design)
    design_sha = sha_file(BASE/"DESIGN_LOCK.json")

    files_by_path = {}
    cache = {}
    slot_rows = []
    pair_rows = []
    custody_rows = []
    for j in p["newcrop"]:
        alias, slot = str(j["case_id"]), int(j["center_slot"])
        roster_row = next(r for r in new_roster if r["probe_alias"] == alias)
        if j["dataset"] != roster_row["dataset"] or j["source_case_sha256"] != roster_row["source_case_sha256"] or j["source_request_sha256"] != roster_row["source_request_sha256"]:
            raise RuntimeError("prompt009 job source/request binding differs from roster")
        center = int(j["center_physical_frame_id"])
        frame_ids = [int(x) for x in roster_row["frame_ids"]]
        if center not in frame_ids or len(j.get("videos", [])) != 2:
            raise RuntimeError("center is not on locked source grid or two inputs are missing")
        position = frame_ids.index(center)
        videos = [_verify_video(v, p["parent_files"], cache) for v in j["videos"]]
        for v in videos:
            files_by_path[v["path"]] = v
        if [len(v["actual_timestamps_seconds"]) for v in videos] != [4,5]:
            raise RuntimeError("parent visual inputs no longer match context/local frame counts")
        response_meta = p["responses"].get(j["job_id"])
        if response_meta is None:
            raise RuntimeError("Top-1 response is missing from sealed parent teacher barrier")
        response = read_json(safe_file(response_meta["path"]))
        if response.get("job_id") != j["job_id"] or response.get("prompt_sha256") != j.get("prompt_sha256") or response.get("source_case_sha256") != j.get("source_case_sha256"):
            raise RuntimeError("Top-1 response is not bound to exact parent job")
        _replay_parent_parse(j, response)
        stable, visible, alt_box, presence, reason = _pair_candidates(j, response)
        pair = None
        if reason == "pairable":
            query = query_by_alias[alias]
            prompt = protocol.build_prompt(query, 1, int(stable))
            mapping = {str(k): int(v) for k,v in j["visible_id_to_candidate"].items()}
            inverse = {v:int(k) for k,v in mapping.items()}
            if inverse.get(1) is None or inverse.get(stable) is None:
                raise RuntimeError("Source/Alternative has no visible ID in original contact sheet")
            candidate = next(c for c in j["candidate_bindings"] if c.get("candidate_id",c.get("id")) == stable)
            source_candidate = next(c for c in j["candidate_bindings"] if c.get("candidate_id",c.get("id")) == 1)
            pair = dict(schema="stvg-crc-pair-job-v1", job_id=f"{alias}_crc_pair_{slot}",
                case_id=alias, dataset=j["dataset"], center_slot=slot, center_physical_frame_id=center,
                source_grid_position=position, source_case_sha256=j["source_case_sha256"],
                source_request_sha256=j["source_request_sha256"], query_key_sha256=roster_row["query_key_sha256"],
                query_text_sha256=hashlib.sha256(query.encode("utf-8")).hexdigest(),
                parent_job_id=j["job_id"], parent_job_canonical_sha256=protocol.canonical_sha256(j),
                parent_prompt_sha256=j["prompt_sha256"], parent_response_path=response_meta["path"],
                parent_response_sha256=response_meta["sha256"], parent_candidate_id=int(stable),
                source_candidate_id=1, alternative_candidate_id=int(stable),
                source_visible_id=inverse[1], alternative_visible_id=inverse[stable],
                visible_id_to_candidate=mapping, candidate_count=int(j["candidate_count"]),
                candidate_binding_sha256=protocol.canonical_sha256(candidate),
                alternative_box_xyxy_normalized=alt_box,
                source_box_xyxy_normalized=_box(source_candidate["boxes"][2]),
                physical_frame_ids=[int(x) for x in j["physical_frame_ids"]],
                local_frame_indices=[int(x) for x in j["local_frame_indices"]],
                videos=videos, prompt=prompt, prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                output_token_cap=protocol.MAX_NEW_TOKENS, status="locked_pair", GT_read=False)
            pair["pair_binding_sha256"] = protocol.canonical_sha256(pair)
            pair_rows.append(pair)
        slot_id=f"{alias}_slot_{slot}"
        parent_job_sha=protocol.canonical_sha256(j)
        slotrow=dict(schema="stvg-crc-slot-v1",slot_id=slot_id,case_id=alias,dataset=j["dataset"],
            center_slot=slot,center_physical_frame_id=center,source_grid_position=position,
            source_case_sha256=j["source_case_sha256"],source_request_sha256=j["source_request_sha256"],
            parent_job_id=j["job_id"],parent_job_canonical_sha256=parent_job_sha,
            parent_response_path=response_meta["path"],parent_response_sha256=response_meta["sha256"],
            parent_response_bytes=response_meta["bytes"],parent_parsed_status=response.get("parsed",{}).get("status"),
            parent_output_truncated=response.get("output_truncated_at_cap") is True,
            parent_selected_candidate_id=stable,parent_visible_id=visible,parent_presence=presence,
            parent_candidate_count=int(j["candidate_count"]),pair_status="locked_pair" if pair else "no_pair",
            no_pair_reason=None if pair else reason,pair_job=pair,GT_read=False)
        slot_rows.append(slotrow)
        custody_rows.append(dict(slot_id=slot_id,case_id=alias,dataset=j["dataset"],center_slot=slot,
            center_physical_frame_id=center,source_grid_position=position,
            source_case_sha256=j["source_case_sha256"],source_request_sha256=j["source_request_sha256"],
            parent_job_id=j["job_id"],parent_job_canonical_sha256=parent_job_sha,
            parent_prompt_sha256=j["prompt_sha256"],response_path=response_meta["path"],
            response_sha256=response_meta["sha256"],response_bytes=response_meta["bytes"],
            response_runtime_sha256=response.get("runtime_sha256"),parsed_status=response.get("parsed",{}).get("status"),
            output_truncated_at_cap=response.get("output_truncated_at_cap") is True,
            selected_candidate_id=stable,visible_id=visible,presence=presence,
            center_box_xyxy_normalized=alt_box,pair_status=slotrow["pair_status"],
            no_pair_reason=slotrow["no_pair_reason"],GT_read=False))
    slot_rows.sort(key=lambda x:(x["case_id"],x["center_slot"]))
    custody_rows.sort(key=lambda x:(x["case_id"],x["center_slot"]))
    if len(slot_rows)!=128 or len(pair_rows)!=49 or len(files_by_path)!=160:
        raise RuntimeError(f"unexpected actual scope: slots={len(slot_rows)}, pairs={len(pair_rows)}, NPYs={len(files_by_path)}")

    custody=dict(schema="stvg-crc-parent-top1-custody-v1",status="complete",count=128,pairable_count=49,
        original_scope="round2_prompt009/new_crop",records=custody_rows,
        bindings=dict(**p["hashes"],roster_sha256=roster_sha,design_lock_sha256=design_sha),GT_read=False)
    write_once(OUT/"PARENT_TOP1_CUSTODY.json",custody)
    custody_sha=sha_file(OUT/"PARENT_TOP1_CUSTODY.json")
    npy_files=sorted(files_by_path.values(),key=lambda x:x["path"])
    slot_bindings=[dict(slot_id=r["slot_id"],case_id=r["case_id"],center_slot=r["center_slot"],
        parent_job_id=r["parent_job_id"],parent_job_canonical_sha256=r["parent_job_canonical_sha256"],
        parent_response_path=r["parent_response_path"],parent_response_sha256=r["parent_response_sha256"],
        input_paths=[v["path"] for v in r["pair_job"]["videos"]] if r["pair_job"] else
            [v["path"] for v in p["job_by_key"][(r["case_id"],r["center_slot"])]["videos"]],
        source_case_sha256=r["source_case_sha256"],source_request_sha256=r["source_request_sha256"],
        pair_status=r["pair_status"],pair_job_id=r["pair_job"]["job_id"] if r["pair_job"] else None,GT_read=False)
        for r in slot_rows]
    input_barrier=dict(schema="stvg-crc-input-barrier-v1",status="complete",count=128,pair_job_count=49,
        slot_bindings=slot_bindings,files=npy_files,
        bindings=dict(design_lock_sha256=design_sha,roster_sha256=roster_sha,
            **p["hashes"],parent_top1_custody_sha256=custody_sha),
        GT_read=False,CPU_only=True,model_loaded=False,GPU_used=False)
    write_once(OUT/"INPUT_BARRIER.json",input_barrier)
    input_sha=sha_file(OUT/"INPUT_BARRIER.json")
    jobs=dict(schema="stvg-crc-teacher-job-lock-v1",status="locked",count=128,pair_job_count=49,
        records=slot_rows,pair_jobs=pair_rows,
        bindings=dict(design_lock_sha256=design_sha,roster_sha256=roster_sha,input_barrier_sha256=input_sha,
            parent_top1_custody_sha256=custody_sha,**p["hashes"]),GT_read=False)
    write_once(OUT/"TEACHER_JOB_LOCK.json",jobs)
    job_sha=sha_file(OUT/"TEACHER_JOB_LOCK.json")
    no_pair={}
    for r in slot_rows:
        if r["pair_status"]=="no_pair": no_pair[r["no_pair_reason"]]=no_pair.get(r["no_pair_reason"],0)+1
    c=dict(schema="stvg-crc-preparation-cpu-contracts-v1",status="pass",protocol=protocol.cpu_contracts(),
        source_count=32,slot_count=128,pair_job_count=49,no_pair_count=79,no_pair_reasons=no_pair,
        unique_input_npy_count=len(npy_files),parent_top1_counts={
            "valid_uncapped_source":sum(r["parent_parsed_status"]=="valid" and not r["parent_output_truncated"] and r["parent_selected_candidate_id"]==1 for r in slot_rows),
            "valid_uncapped_non_source":sum(r["parent_parsed_status"]=="valid" and not r["parent_output_truncated"] and r["parent_selected_candidate_id"] not in (None,1) for r in slot_rows),
            "abstain":sum(r["parent_parsed_status"]=="valid" and not r["parent_output_truncated"] and r["parent_selected_candidate_id"] is None for r in slot_rows)},
        GT_read=False,CPU_only=True,model_loaded=False,GPU_used=False,annotations_read=False,
        generation=protocol.EXPECTED_GENERATION,
        bindings=dict(design_lock_sha256=design_sha,roster_sha256=roster_sha,input_barrier_sha256=input_sha,
            job_lock_sha256=job_sha,parent_top1_custody_sha256=custody_sha))
    write_once(OUT/"CPU_CONTRACTS.json",c)
    return dict(status="complete",source_count=32,slot_count=128,pair_job_count=49,
        no_pair_count=79,unique_input_npy_count=len(npy_files),design_lock_sha256=design_sha,
        roster_sha256=roster_sha,input_barrier_sha256=input_sha,job_lock_sha256=job_sha,
        parent_top1_custody_sha256=custody_sha,GT_read=False,model_loaded=False,GPU_used=False)


def main():
    parser=argparse.ArgumentParser()
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare",action="store_true")
    group.add_argument("--self-test",action="store_true")
    args=parser.parse_args()
    result=protocol.cpu_contracts() if args.self_test else prepare()
    print(json.dumps(result,sort_keys=True))


if __name__=="__main__":
    main()
