#!/usr/bin/env python3
"""Post-seal CPU review of the sparse STVG-R1 semantic candidate screen.

The only label-opening entrypoint is ``review_after_seal``.  It first verifies
the complete 48-response P0 barrier, every response byte, the locked roster,
the source-case bindings and both runtime pin sets.  It then freezes all
prediction-only selections before it calls the existing legal target reader.
No model is instantiated and no video/RGB is decoded by this module.

The oracle is computed only as a post-seal diagnostic upper bound.  Rerank and
pseudo-reference choices are frozen from parsed model output before labels are
opened, and the oracle value is never an input to either choice.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Callable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BASE = ROOT / "artifacts/stvg_sparse_stvg_r1_semantic_opd_v1"
SOURCE_BASE = ROOT / "artifacts/stvg_mllm_candidate_feedback_v1"
ROSTER_DEFAULT = BASE / "ROSTER_LOCK.json"
RUNTIME_DEFAULT = BASE / "RUNTIME.json"
SEAL_DEFAULT = BASE / "P0_GLOBAL_READOUT_BARRIER.json"
METRIC_CONTRACT = ROOT / "artifacts/stvg_target_teacher_distillation_v1/evaluation_contracts/CPU_CONTRACTS.json"

SCHEMA = "stvg_sparse_r1_postseal_review_v1"
P0_REQUESTS = 48
P0_CONTROLS = ("base", "id_swap", "wrong_query")
BRANCHES = ("temporal", "spatial")
EXPECTED_PARSER_SCHEMA = "stvg-sparse-r1-semantic-verification-v1"
ABSTAIN_REASONS = ("insufficient_visual_evidence", "semantic_ambiguity",
                   "candidate_set_inadequate", "other")
LABEL_STATES = ("supported", "contradicted", "uncertain")
METRIC_KEYS = ("m_vIoU", "m_tIoU", "m_sIoU_dense_GT")
TIE_EPSILON = 1e-8
FORBIDDEN_KEYS = {
    "ground_truth", "truth", "truth_boxes_xyxy", "gt_span", "gt_boxes",
    "annotations", "annotation", "labels", "label_path", "gt_path",
    "ground_truth_path", "gt_labels", "target_annotation",
}


class ReviewError(RuntimeError):
    pass


def require(ok: Any, message: str) -> None:
    if not ok:
        raise ReviewError(message)


def _pairs_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ReviewError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ReviewError(f"non-finite JSON value: {value}")


def read_json(path: Path) -> Any:
    try:
        with Path(path).open("r", encoding="utf-8") as stream:
            return json.load(stream, object_pairs_hook=_pairs_no_duplicates,
                             parse_constant=_reject_constant)
    except (OSError, json.JSONDecodeError) as exc:
        raise ReviewError(f"cannot read JSON {path}: {exc}") from exc


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_repo_path(relative: str, repo_root: Path = ROOT) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ReviewError("evidence paths must be non-empty project-relative POSIX paths")
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != relative:
        raise ReviewError(f"unsafe evidence path: {relative!r}")
    resolved = (repo_root / path).resolve()
    if not resolved.is_relative_to(repo_root.resolve()):
        raise ReviewError("evidence path escaped the project root")
    return resolved


def _reject_ground_truth_fields(value: Any, where: str) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if str(key).lower() in FORBIDDEN_KEYS:
                raise ReviewError(f"GT/annotation field in label-free artifact: {where}.{key}")
            _reject_ground_truth_fields(nested, f"{where}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _reject_ground_truth_fields(nested, f"{where}[{index}]")


def classify_response(response: Mapping[str, Any]) -> dict[str, Any]:
    """Keep truncation, invalid output, abstention and valid output distinct."""
    parsed = response.get("parsed")
    parser_status = parsed.get("status") if isinstance(parsed, Mapping) else None
    if response.get("output_truncated_at_cap") is True:
        category = "truncated"
    elif (response.get("_structural_wrapper_valid") is True or
          response.get("_minimal_selected_id_valid") is True):
        if isinstance(parsed, Mapping) and parsed.get("status") == "valid" and parsed.get("abstain") is True:
            category = "abstain"
        elif isinstance(parsed, Mapping) and parsed.get("status") == "valid" and parsed.get("abstain") is False:
            category = "valid"
        else:
            category = "invalid_output"
    elif parser_status == "invalid_output":
        category = "invalid_output"
    elif parser_status == "valid":
        if isinstance(parsed.get("abstain"), bool) and parsed["abstain"]:
            category = "abstain"
        elif isinstance(parsed.get("abstain"), bool) and not parsed["abstain"]:
            category = "valid"
        else:
            category = "invalid_output"
    else:
        category = "invalid_output"
    return {"category": category, "parser_status": parser_status,
            "truncated": response.get("output_truncated_at_cap") is True,
            "abstain": bool(parsed.get("abstain")) if isinstance(parsed, Mapping) else None}


def lossless_output_readback(response: Mapping[str, Any], parser: Callable[..., Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Recheck strict bare JSON, with one exact lossless JSON-fence allowance.

    The accepted wrapper is exactly `````json\n<JSON>\n``` ``. No whitespace,
    prose, thought text, repair, or truncation completion is stripped or
    inferred. Original response bytes and the worker's stored strict-parse
    result are preserved. A separate minimal selected-ID readback may expose a
    valid top choice when a complete JSON object contains only a legal prefix
    ranking; it never completes the rank or evaluation list.
    """
    raw = response.get("raw_output")
    metadata = response.get("request_metadata")
    require(isinstance(raw, str) and isinstance(metadata, Mapping) and
            isinstance(metadata.get("catalog"), Mapping),
            "output readback requires the sealed raw text and request catalog")
    strict_value: Mapping[str, Any] | None = None
    strict_error: Exception | None = None
    try:
        strict_value = parser(raw, metadata["catalog"])
    except (ValueError, TypeError, KeyError) as exc:
        strict_error = exc

    inner, wrapper_removed, wrapper_kind = _lossless_unwrap(raw)
    if wrapper_removed:
        # Require one complete JSON object inside the exact fence. The strict
        # parser still decides whether its keys, values, and IDs are valid.
        has_complete_object_edges = inner.strip().startswith("{") and inner.strip().endswith("}")
        structural_value: Mapping[str, Any] | None = None
        structural_error: Exception | None = None
        if has_complete_object_edges:
            try:
                structural_value = parser(inner, metadata["catalog"])
            except (ValueError, TypeError, KeyError) as exc:
                structural_error = exc
        else:
            structural_error = ValueError("fenced content is not one complete JSON object")
    else:
        structural_value = strict_value
        structural_error = strict_error
    minimal_value: dict[str, Any] | None = None
    minimal_error: str | None = None
    if inner is not None:
        try:
            minimal_value = _validate_minimal_selected_id(inner, metadata)
        except (ReviewError, json.JSONDecodeError, TypeError, ValueError) as exc:
            minimal_error = str(exc)

    stored = response.get("parsed")
    require(isinstance(stored, Mapping), "sealed worker parse record is missing")
    strict_bare_valid = strict_value is not None
    if strict_bare_valid:
        require(stored.get("status") == "valid" and
                {k: v for k, v in stored.items() if k != "status"} == dict(strict_value),
                "stored strict-parse result disagrees with parsing the original bare response")
    else:
        require(stored.get("status") == "invalid_output" and strict_error is not None and
                stored.get("error_type") == type(strict_error).__name__ and
                stored.get("error") == str(strict_error),
                "stored invalid parse result disagrees with strict parsing of the original response")

    truncated = response.get("output_truncated_at_cap") is True
    structural_wrapper_valid = structural_value is not None and not truncated
    minimal_selected_id_valid = minimal_value is not None and not truncated
    error = structural_error
    readback = {
        "strict_bare_valid": strict_bare_valid,
        "structural_wrapper_valid": structural_wrapper_valid,
        "minimal_selected_id_valid": minimal_selected_id_valid,
        "wrapper_removed": wrapper_removed,
        "wrapper_kind": wrapper_kind,
        "truncated_at_cap": truncated,
        "status": ("truncated" if truncated else
                   "valid_exact_json_wrapper" if structural_wrapper_valid and wrapper_removed else
                   "valid_bare_json" if structural_wrapper_valid else
                   "valid_minimal_abstention" if minimal_selected_id_valid and
                       minimal_value is not None and minimal_value.get("abstain") is True and
                       not minimal_value.get("ranked_ids") else
                   "valid_minimal_selected_id" if minimal_selected_id_valid else "invalid_structure"),
        "structural_error_type": type(error).__name__ if error is not None else None,
        "structural_error": str(error) if error is not None else None,
        "minimal_error": minimal_error,
        "top_rank_id": (minimal_value.get("ranked_ids", [None])[0]
                         if minimal_value and minimal_value.get("ranked_ids") else None),
        "selected_id": (minimal_value.get("ranked_ids", [None])[0]
                        if minimal_value and minimal_value.get("ranked_ids") and
                        minimal_value.get("abstain") is False else None),
        "abstain": bool(minimal_value.get("abstain")) if minimal_value else None,
        "selected_rank_length": len(minimal_value.get("ranked_ids", [])) if minimal_value else 0,
        "rank_order_complete": bool(minimal_value and len(minimal_value.get("ranked_ids", [])) ==
                                     len(metadata["catalog"].get("candidate_ids", []))),
        "selected_evaluation_six_states_valid": bool(minimal_value and
            minimal_value.get("selected_evaluation_six_states_valid")),
        "selected_evaluation_status": minimal_value.get("selected_evaluation_status")
            if minimal_value else "unknown",
        "selected_evaluation": minimal_value.get("selected_evaluation") if minimal_value else None,
        "selected_evidence_frame_unit_status": minimal_value.get("selected_evidence_frame_unit_status")
            if minimal_value else "unknown",
    }
    view = dict(response)
    view["_structural_wrapper_valid"] = structural_wrapper_valid
    view["_minimal_selected_id_valid"] = minimal_selected_id_valid
    view["_selected_evaluation_six_states_valid"] = bool(
        minimal_value and minimal_value.get("selected_evaluation_six_states_valid"))
    if structural_wrapper_valid and structural_value is not None:
        view["parsed"] = {**dict(structural_value), "status": "valid"}
    elif minimal_selected_id_valid and minimal_value is not None:
        wire_keys = {"schema", "branch", "ranked_ids", "abstain", "abstain_reason", "evaluations"}
        view["parsed"] = {**{k: v for k, v in minimal_value.items() if k in wire_keys},
                           "status": "valid"}
    return readback, view


def _lossless_unwrap(raw: str) -> tuple[str | None, bool, str]:
    """Return bare JSON content or the one exact accepted fence, without repair."""
    prefix, suffix = "```json\n", "\n```"
    if raw.startswith(prefix) and raw.endswith(suffix):
        return raw[len(prefix):-len(suffix)], True, "exact_json_fence"
    stripped = raw.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return raw, False, "bare_json"
    return None, False, "non_bare_text"


def _validate_minimal_selected_id(raw_json: str, metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a complete JSON object sufficient for a top-choice readback.

    This path does not require or synthesize the full ranking/evaluation schema.
    It only accepts IDs already present in the exact request catalog.
    """
    value = json.loads(raw_json, object_pairs_hook=_pairs_no_duplicates,
                       parse_constant=_reject_constant)
    require(isinstance(value, Mapping), "minimal output must be one JSON object")
    required = {"schema", "branch", "ranked_ids", "abstain", "abstain_reason"}
    allowed = required | {"evaluations"}
    require(required.issubset(value) and set(value).issubset(allowed),
            "minimal output has missing or unexpected top-level fields")
    catalog = metadata.get("catalog")
    require(isinstance(catalog, Mapping) and catalog.get("schema") ==
            "stvg-sparse-r1-candidate-catalog-v1", "minimal output request catalog is unpinned")
    require(value.get("schema") == EXPECTED_PARSER_SCHEMA and
            value.get("branch") == catalog.get("branch") in BRANCHES,
            "minimal output schema or branch differs from its request")
    allowed_ids = catalog.get("candidate_ids")
    ranking = value.get("ranked_ids")
    abstain = value.get("abstain")
    reason = value.get("abstain_reason")
    require(isinstance(abstain, bool) and
            ((abstain and reason in ABSTAIN_REASONS) or (not abstain and reason is None)),
            "minimal output abstain state/reason is invalid")
    require(isinstance(allowed_ids, list) and allowed_ids and
            all(isinstance(cid, str) for cid in allowed_ids) and len(allowed_ids) == len(set(allowed_ids)),
            "minimal output catalog IDs are invalid")
    require(isinstance(ranking, list) and len(ranking) <= len(allowed_ids) and
            all(isinstance(cid, str) for cid in ranking) and len(ranking) == len(set(ranking)) and
            set(ranking).issubset(set(allowed_ids)),
            "minimal ranked_ids must be a duplicate-free subset of real catalog IDs")
    require(bool(ranking) or (abstain is True and reason == "candidate_set_inadequate"),
            "empty minimal ranking is only valid for explicit candidate-set inadequacy abstention")
    evaluations = value.get("evaluations", [])
    supplied: list[Mapping[str, Any]] = []
    eval_error = None
    if isinstance(evaluations, list):
        seen: set[str] = set()
        for row in evaluations:
            if not isinstance(row, Mapping):
                eval_error = "evaluation row is not an object"
                break
            cid = row.get("candidate_id")
            if not isinstance(cid, str) or cid not in allowed_ids or cid in seen:
                eval_error = "evaluation row has an unknown or duplicate candidate ID"
                break
            seen.add(cid)
            supplied.append(row)
    else:
        eval_error = "evaluations is not an array"
    selected_rows = [row for row in supplied
                     if ranking and abstain is False and row.get("candidate_id") == ranking[0]]
    selected = dict(selected_rows[0]) if len(selected_rows) == 1 else None
    six_valid = False
    slot_status = "unknown"
    if selected is not None and eval_error is None:
        semantic = selected.get("semantic")
        boundary = selected.get("boundary")
        if isinstance(semantic, Mapping) and isinstance(boundary, Mapping):
            six = [semantic.get("referent_match"), semantic.get("event_relation_match"),
                   boundary.get("start_boundary"), boundary.get("end_boundary"),
                   boundary.get("extent"), selected.get("target_support")]
            six_valid = all(isinstance(label, str) and label in LABEL_STATES for label in six)
        raw_slots = selected.get("evidence_frame_slots")
        if isinstance(raw_slots, list) and all(isinstance(fid, int) and not isinstance(fid, bool)
                                               for fid in raw_slots):
            slot_ids = [int(slot.get("physical_frame_id")) for slot in metadata.get("frame_slots", [])
                        if isinstance(slot, Mapping) and isinstance(slot.get("physical_frame_id"), int)]
            if all(0 <= fid <= 7 for fid in raw_slots):
                slot_status = "valid_slot_indices"
            elif raw_slots and set(raw_slots).issubset(set(slot_ids)):
                slot_status = "physical_frame_ids_not_slot_indices"
            else:
                slot_status = "invalid_slot_indices"
        elif raw_slots is not None:
            slot_status = "invalid_slot_indices"
    selected_status = ("invalid_evaluation_array" if eval_error is not None else
                       "missing_selected_candidate_evaluation" if selected is None else
                       "six_states_valid" if six_valid else "invalid_six_states")
    if not six_valid:
        selected = None
    return {**dict(value), "_eval_error": eval_error,
            "selected_evaluation_six_states_valid": six_valid,
            "selected_evaluation_status": selected_status,
            "selected_evaluation": selected,
            "selected_evidence_frame_unit_status": slot_status}


def _catalog_maps(metadata: Mapping[str, Any]) -> dict[str, Any]:
    catalog = metadata.get("catalog")
    audit = metadata.get("candidate_audit")
    require(isinstance(catalog, Mapping) and catalog.get("schema") == "stvg-sparse-r1-candidate-catalog-v1",
            "request metadata lacks the pinned sparse candidate catalog")
    require(catalog.get("branch") in BRANCHES and isinstance(catalog.get("candidates"), list),
            "candidate catalog branch/list is malformed")
    ids = catalog.get("candidate_ids")
    entries = catalog["candidates"]
    require(isinstance(ids, list) and all(isinstance(x, str) for x in ids) and
            len(ids) == len(set(ids)) and len(ids) == len(entries),
            "candidate IDs must be unique and cover the public candidate catalog")
    require(isinstance(audit, Mapping), "candidate audit mapping is missing")
    source_map = audit.get("source_candidate_id_by_visible_id")
    geometry_map = audit.get("geometry_key_by_visible_id")
    require(isinstance(source_map, Mapping) and isinstance(geometry_map, Mapping) and
            set(source_map) == set(ids) == set(geometry_map),
            "visible IDs must map exactly to source candidate IDs and geometry keys")
    require(all(isinstance(source_map[x], str) and isinstance(geometry_map[x], str)
                for x in ids), "candidate audit maps must contain string identities")
    entries_by_id = {str(item.get("id")): item for item in entries
                     if isinstance(item, Mapping) and isinstance(item.get("id"), str)}
    require(set(entries_by_id) == set(ids), "candidate catalog entry IDs are incomplete or repeated")
    identity_pairs = [(str(geometry_map[cid]), str(source_map[cid]), cid) for cid in ids]
    identity_pairs.sort(key=lambda item: (item[0], item[1]))
    ordinal = {cid: index for index, (_, _, cid) in enumerate(identity_pairs, 1)}
    return {"branch": catalog["branch"], "catalog": catalog, "entries_by_id": entries_by_id,
            "source_id_by_visible_id": dict(source_map), "geometry_key_by_visible_id": dict(geometry_map),
            "geometry_ordinal_by_visible_id": ordinal}


def _validate_parsed_response(response: Mapping[str, Any], metadata: Mapping[str, Any]) -> None:
    parsed = response.get("parsed")
    require(isinstance(parsed, Mapping), "response parsed field is missing")
    if parsed.get("status") != "valid":
        return
    maps = _catalog_maps(metadata)
    parser_status = parsed.get("status")
    result = {key: value for key, value in parsed.items() if key != "status"}
    require(result.get("schema") == EXPECTED_PARSER_SCHEMA and
            result.get("branch") == maps["branch"], "parsed output schema/branch differs from its request")
    ids = list(maps["entries_by_id"])
    ranked = result.get("ranked_ids")
    require(isinstance(ranked, list) and len(ranked) == len(ids) and set(ranked) == set(ids),
            "valid output must rank every catalog ID exactly once")
    evaluations = result.get("evaluations")
    require(isinstance(evaluations, list) and len(evaluations) == len(ids) and
            {x.get("candidate_id") for x in evaluations if isinstance(x, Mapping)} == set(ids),
            "valid output must evaluate every candidate exactly once")
    require(isinstance(result.get("abstain"), bool), "valid output must preserve the abstain boolean")
    require(parser_status == "valid", "unreachable parser status mismatch")


def preflight_global_seal(seal_path: Path = SEAL_DEFAULT, *, repo_root: Path = ROOT,
                          expected_case_ids: Sequence[str] | None = None) -> dict[str, Any]:
    """Verify only the 48 label-free predictions; never opens a label or source case."""
    seal_path = Path(seal_path).resolve()
    require(seal_path.is_file(), "complete P0 response seal is absent")
    seal = read_json(seal_path)
    require(isinstance(seal, Mapping) and seal.get("status") == "complete" and
            seal.get("scope") == "P0_all_sparse_responses_sealed" and
            seal.get("count") == P0_REQUESTS and seal.get("GT_read") is False,
            "the complete 48-response P0 prediction seal is required before any label reader")
    require(isinstance(seal.get("roster_sha256"), str) and len(seal["roster_sha256"]) == 64 and
            isinstance(seal.get("runtime_sha256"), str) and len(seal["runtime_sha256"]) == 64,
            "P0 seal must bind both locked roster and model runtime bytes")
    refs = seal.get("files")
    require(isinstance(refs, list) and len(refs) == P0_REQUESTS,
            "P0 seal must enumerate exactly 48 response JSON files")
    payloads: dict[tuple[str, str, str], dict[str, Any]] = {}
    refs_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    case_ids: set[str] = set()
    for ref in refs:
        require(isinstance(ref, Mapping), "malformed P0 response seal receipt")
        rel = ref.get("path")
        digest = ref.get("sha256")
        require(isinstance(digest, str) and len(digest) == 64,
                "P0 response receipt lacks a SHA-256 digest")
        path = safe_repo_path(rel, repo_root)
        require(path.is_file() and sha256_file(path) == digest,
                "sealed P0 response file is missing or changed")
        data = read_json(path)
        _reject_ground_truth_fields(data, f"response[{rel}]")
        require(isinstance(data, Mapping) and data.get("status") == "accepted_actual_sparse_response" and
                data.get("GT_read") is False and data.get("teacher_frozen") is True and
                data.get("teacher_optimizer_steps") == 0,
                "each P0 response must be an accepted frozen GT-free observation")
        case_id, branch, control = data.get("case_id"), data.get("branch"), data.get("control")
        require(isinstance(case_id, str) and branch in BRANCHES and control in P0_CONTROLS,
                "sealed response case/branch/control identity is invalid")
        key = (case_id, branch, control)
        require(key not in payloads, "duplicate P0 logical request")
        case_ids.add(case_id)
        require(data.get("runtime_sha256") == seal["runtime_sha256"],
                "response runtime differs from the global seal")
        metadata = data.get("request_metadata")
        require(isinstance(metadata, Mapping) and metadata.get("GT_read") is False and
                metadata.get("model_called") is False,
                "request metadata is not label-free or is missing")
        require(isinstance(data.get("prompt_sha256"), str) and len(data["prompt_sha256"]) == 64 and
                isinstance(metadata.get("prompt_sha256"), str) and len(metadata["prompt_sha256"]) == 64 and
                isinstance(data.get("raw_output"), str),
                "raw output/prompt digest receipt is incomplete")
        maps = _catalog_maps(metadata)
        require(maps["branch"] == branch, "candidate catalog branch differs from response branch")
        require(metadata.get("query_control_override") is (control == "wrong_query"),
                "query-control flag differs from the sealed control assignment")
        require(metadata.get("query_control_distribution_shift") is (control == "wrong_query"),
                "wrong-query control must remain marked as a distribution shift")
        _validate_parsed_response(data, metadata)
        payloads[key] = dict(data)
        refs_by_key[key] = dict(ref)
    require(len(case_ids) == 8, "P0 semantic screen must cover exactly eight locked cases")
    if expected_case_ids is not None:
        require(case_ids == set(expected_case_ids), "P0 seal case IDs differ from the locked request cohort")
    expected_keys = {(case_id, branch, control) for case_id in case_ids
                     for branch in BRANCHES for control in P0_CONTROLS}
    require(set(payloads) == expected_keys,
            "P0 response seal must contain every case × branch × base/ID-swap/wrong-query cell exactly once")
    for case_id in case_ids:
        for branch in BRANCHES:
            base = payloads[(case_id, branch, "base")]
            swapped = payloads[(case_id, branch, "id_swap")]
            wrong = payloads[(case_id, branch, "wrong_query")]
            bm = base["request_metadata"]
            sm = swapped["request_metadata"]
            wm = wrong["request_metadata"]
            bmap = _catalog_maps(bm)
            smap = _catalog_maps(sm)
            wmap = _catalog_maps(wm)
            require(set(bmap["geometry_key_by_visible_id"].values()) ==
                    set(smap["geometry_key_by_visible_id"].values()) ==
                    set(wmap["geometry_key_by_visible_id"].values()),
                    "control arms changed candidate geometry")
            require(bm.get("rendered_video_sha256") == wm.get("rendered_video_sha256"),
                    "wrong-query control must preserve the same rendered pixels")
            require(bm.get("frame_slots") == sm.get("frame_slots") and
                    bm.get("candidate_audit",{}).get("color_rgb_by_geometry_key") ==
                    sm.get("candidate_audit",{}).get("color_rgb_by_geometry_key"),
                    "ID-swap must preserve source pixels/frame clock/geometry colors; its displayed IDs change")
            require(bm.get("query_sha256") == sm.get("query_sha256"),
                    "ID-swap control must preserve the query")
        require(bm.get("candidate_catalog_sha256") == wm.get("candidate_catalog_sha256"),
                    "wrong-query control must preserve the candidate catalog")
            # The wrong query is an input distribution shift. It is not assumed
            # to contradict the source clip and must be adjudicated by a person.
    return {"seal": dict(seal), "seal_sha256": sha256_file(seal_path),
            "seal_path": str(seal_path.relative_to(repo_root.resolve())),
            "responses": payloads, "receipts": refs_by_key,
            "case_ids": sorted(case_ids), "GT_read": False, "verified": True}


def _validate_runtime(runtime_path: Path, expected_sha: str, repo_root: Path) -> dict[str, Any]:
    require(runtime_path.is_file() and sha256_file(runtime_path) == expected_sha,
            "P0 runtime bytes differ from the global prediction seal")
    runtime = read_json(runtime_path)
    pins = runtime.get("pins") if isinstance(runtime, Mapping) else None
    require(isinstance(pins, Mapping) and pins, "locked sparse runtime lacks source pins")
    verified: dict[str, str] = {}
    for relative, digest in pins.items():
        require(isinstance(relative, str) and isinstance(digest, str) and len(digest) == 64,
                "malformed sparse runtime source pin")
        path = safe_repo_path(relative, repo_root)
        require(path.is_file() and sha256_file(path) == digest,
                f"runtime source pin changed: {relative}")
        verified[relative] = digest
    require(any(verified.get(name) is not None for name in ("scripts/run_stvg_sparse_r1_v1.py","scripts/run_stvg_sparse_r1_v2.py")) and
            verified.get("scripts/stvg_sparse_r1_prompt_v1.py") is not None,
            "runtime must pin both sparse runner and official prompt/response adapter")
    return {"runtime": runtime, "runtime_sha256": expected_sha, "verified_pins": verified}


def _load_locked_roster(roster_path: Path, expected_sha: str, case_ids: set[str]) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    require(roster_path.is_file() and sha256_file(roster_path) == expected_sha,
            "P0 roster bytes differ from the global prediction seal")
    roster = read_json(roster_path)
    require(isinstance(roster, Mapping) and roster.get("status") == "locked" and
            roster.get("GT_read") is False and isinstance(roster.get("records"), list),
            "P0 cohort roster must be locked and label-free")
    p0_rows = [row for row in roster["records"] if isinstance(row, Mapping) and row.get("P0") is True]
    by_case = {str(row.get("case_id")): dict(row) for row in p0_rows}
    require(len(p0_rows) == 8 and len(by_case) == 8 and set(by_case) == case_ids,
            "locked P0 roster must contain exactly the eight sealed case IDs")
    for case_id, row in by_case.items():
        require(row.get("dataset") in ("hc2", "vidstg") and
                isinstance(row.get("source_case_path"), str) and
                isinstance(row.get("source_case_sha256"), str) and
                len(row["source_case_sha256"]) == 64,
                f"roster source binding is incomplete for {case_id}")
    return dict(roster), by_case


def _to_numpy(value: Any, where: str):
    import numpy as np
    if hasattr(value, "detach"):
        require(str(value.device) == "cpu" and not bool(getattr(value, "requires_grad", False)),
                f"{where} must be a detached CPU tensor")
        value = value.detach().numpy()
    array = np.asarray(value)
    require(np.isfinite(array).all(), f"{where} contains non-finite values")
    return array


def _geometry_hash(value: Any) -> str:
    import numpy as np
    array = np.asarray(value, dtype="<f8", order="C")
    require(np.isfinite(array).all(), "candidate geometry contains non-finite coordinates")
    digest = hashlib.sha256()
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _interval_hash(interval: Sequence[Any]) -> str:
    return sha256_bytes(json.dumps(list(interval), separators=(",", ":"),
                                  allow_nan=False).encode())


def _xyxy_to_cxcywh(boxes: Any):
    import numpy as np
    b = np.asarray(boxes, dtype=np.float64)
    require(b.ndim == 2 and b.shape[1] == 4 and np.isfinite(b).all(),
            "normalized xyxy tube must have shape [frames,4]")
    require(bool(np.all(b[:, 0] < b[:, 2]) and np.all(b[:, 1] < b[:, 3])),
            "normalized xyxy tube contains non-positive boxes")
    return np.stack(((b[:, 0] + b[:, 2]) * .5, (b[:, 1] + b[:, 3]) * .5,
                     b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]), axis=1)


def _physical_interval(value: Any, where: str) -> list[float]:
    require(isinstance(value, (list, tuple)) and len(value) == 2,
            f"{where} must be a two-element half-open physical frame interval")
    require(all(isinstance(x, (int,float)) and not isinstance(x,bool) and math.isfinite(float(x)) for x in value),
            f"{where} endpoints must be finite physical frame coordinates")
    out = [float(x) for x in value]
    require(out[1]>out[0], f"{where} interval is empty")
    return out


def _source_catalog(case: Mapping[str, Any], metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Bind visible catalog IDs to complete source-grid geometries, not 8-box previews."""
    import numpy as np
    maps = _catalog_maps(metadata)
    branch, catalog = maps["branch"], maps["catalog"]
    source_candidates = case.get("candidates")
    require(isinstance(source_candidates, Mapping) and source_candidates.get("no_GT_input") is True and
            source_candidates.get("no_teacher_boxes") is True,
            "candidate source cache is not the expected student-only candidate set")
    frame_ids = _to_numpy(case["frame_ids"], "source.frame_ids").astype(np.int64, copy=False)
    native_xyxy = _to_numpy(case["native_boxes"], "source.native_boxes").astype(np.float64, copy=False)
    require(frame_ids.ndim == 1 and native_xyxy.shape == (len(frame_ids), 4),
            "source physical grid and native tube do not align")
    entries = maps["entries_by_id"]
    source_id_map = maps["source_id_by_visible_id"]
    geometry_map = maps["geometry_key_by_visible_id"]
    candidate_audit = metadata["candidate_audit"]
    full_geometry_map = candidate_audit.get("scientific_geometry_by_visible_id")
    result: dict[str, dict[str, Any]] = {}
    if branch == "spatial":
        raw = source_candidates.get("spatial", {}).get("candidates", [])
        by_source_id = {str(row.get("candidate_id")): row for row in raw if isinstance(row, Mapping)}
        for visible_id in maps["catalog"]["candidate_ids"]:
            source_id = str(source_id_map[visible_id])
            if source_id == "student_native_baseline":
                tube = native_xyxy
            else:
                raw_row = by_source_id.get(source_id)
                require(isinstance(raw_row, Mapping) and raw_row.get("valid") is True,
                        f"spatial catalog candidate has no valid frozen source geometry: {visible_id}")
                tube = _to_numpy(raw_row.get("dense_tube_xyxy"), f"source candidate {visible_id}")
                tube = tube.astype(np.float64, copy=False)
            require(tube.shape == native_xyxy.shape,
                    f"spatial full tube does not match the complete source grid: {visible_id}")
            geom_sha = _geometry_hash(tube)
            require(geom_sha == geometry_map[visible_id],
                    f"spatial candidate geometry SHA differs from its sealed prompt mapping: {visible_id}")
            if isinstance(full_geometry_map, Mapping) and visible_id in full_geometry_map:
                exposed = _to_numpy(full_geometry_map[visible_id], f"prompt full geometry {visible_id}")
                require(exposed.shape == tube.shape and np.array_equal(exposed, tube),
                        f"prompt's full scientific geometry differs from source candidate: {visible_id}")
            item = entries[visible_id]
            slots = item.get("box_frame_ids")
            sampled = _to_numpy(item.get("boxes_by_video_slot_normalized_xyxy"),
                                f"catalog sampled geometry {visible_id}")
            sparse_meta = metadata.get("frame_slots")
            require(isinstance(sparse_meta, list) and len(sparse_meta) == 8 and
                    isinstance(slots, list) and len(slots) == 8 and sampled.shape == (8, 4),
                    "spatial catalog must bind exactly eight sparse physical frame slots")
            expected_ids = [int(slot["physical_frame_id"]) for slot in sparse_meta]
            require([int(x) for x in slots] == expected_ids,
                    "spatial candidate preview frame IDs differ from source request frames")
            positions = {int(fid): pos for pos, fid in enumerate(frame_ids.tolist())}
            require(all(fid in positions for fid in expected_ids),
                    "sparse request contains a physical frame outside the frozen source grid")
            expected_sampled = tube[[positions[fid] for fid in expected_ids]]
            require(np.array_equal(sampled.astype(np.float64), np.asarray([[round(float(x),4) for x in row] for row in expected_sampled])),
                    "candidate display preview differs from its exact source-grid geometry")
            result[visible_id] = {"boxes_xyxy_normalized": tube, "frame_ids": frame_ids,
                                  "interval": _physical_interval(case["Projection_interval"],
                                                                 "source.Projection_interval"),
                                  "geometry_key": geom_sha,
                                  "geometry_ordinal": maps["geometry_ordinal_by_visible_id"][visible_id],
                                  "source_id_internal": source_id}
    else:
        raw = source_candidates.get("temporal", {}).get("candidates", [])
        by_source_id = {str(row.get("candidate_id")): row for row in raw if isinstance(row, Mapping)}
        for visible_id in maps["catalog"]["candidate_ids"]:
            source_id = str(source_id_map[visible_id])
            source_row = by_source_id.get(source_id)
            require(isinstance(source_row, Mapping),
                    f"temporal catalog candidate has no frozen source interval: {visible_id}")
            catalog_interval = _physical_interval(
                entries[visible_id].get("interval_physical_frame_bounds_half_open"),
                f"catalog[{visible_id}].interval_physical_frame_bounds_half_open")
            source_interval = _physical_interval(source_row.get("interval"),
                                                 f"source temporal candidate {visible_id}.interval")
            require(catalog_interval == source_interval,
                    f"temporal candidate interval differs from the frozen source grid: {visible_id}")
            # Prompt code hashes JSON's original numeric representation. Keep
            # its public interval wire for the digest check before converting.
            raw_interval = entries[visible_id].get("interval_physical_frame_bounds_half_open")
            require(_interval_hash(raw_interval) == geometry_map[visible_id],
                    f"temporal interval hash differs from the sealed prompt mapping: {visible_id}")
            result[visible_id] = {"boxes_xyxy_normalized": native_xyxy, "frame_ids": frame_ids,
                                  "interval": catalog_interval,
                                  "geometry_key": geometry_map[visible_id],
                                  "geometry_ordinal": maps["geometry_ordinal_by_visible_id"][visible_id],
                                  "source_id_internal": source_id}
    return {"branch": branch, "catalog_maps": maps, "by_visible_id": result,
            "native_boxes_xyxy_normalized": native_xyxy, "frame_ids": frame_ids}


def _decision_for(response: Mapping[str, Any], metadata: Mapping[str, Any],
                  source_catalog: Mapping[str, Any]) -> dict[str, Any]:
    """Freeze model-only rank/pseudo-reference decisions before any GT reader."""
    status = classify_response(response)
    parsed = response.get("parsed")
    branch = str(response.get("branch"))
    if status["category"] not in ("valid", "abstain") or not isinstance(parsed, Mapping):
        return {"response_status": status, "ranked_geometry_ordinals": None,
                "raw_ranked_candidate": None, "pseudo_reference": None,
                "pseudo_reference_status": "no_information", "source_fallback": True}
    maps = source_catalog["catalog_maps"]
    ranking = parsed.get("ranked_ids")
    if not isinstance(ranking, list) or any(cid not in maps["geometry_ordinal_by_visible_id"] for cid in ranking):
        return {"response_status": {**status, "category": "invalid_output"},
                "ranked_geometry_ordinals": None, "raw_ranked_candidate": None,
                "pseudo_reference": None, "pseudo_reference_status": "no_information",
                "source_fallback": True}
    ranked_ordinals = [int(maps["geometry_ordinal_by_visible_id"][cid]) for cid in ranking]
    raw_top = ranking[0] if ranking else None
    if status["category"] == "abstain" or raw_top is None:
        return {"response_status": status, "ranked_geometry_ordinals": ranked_ordinals,
                "raw_ranked_candidate": raw_top, "pseudo_reference": None,
                "pseudo_reference_status": "abstain_no_information", "source_fallback": True}
    if branch == "temporal":
        return {"response_status": status, "ranked_geometry_ordinals": ranked_ordinals,
                "raw_ranked_candidate": raw_top, "pseudo_reference": raw_top,
                "pseudo_reference_status": "selected_top_rank_no_param_rerank",
                "source_fallback": False}
    evaluations = parsed.get("evaluations")
    by_id = {row.get("candidate_id"): row for row in evaluations if isinstance(row, Mapping)} \
        if isinstance(evaluations, list) else {}
    ev = by_id.get(raw_top, {})
    semantic = ev.get("semantic", {}) if isinstance(ev, Mapping) else {}
    semantic = semantic if isinstance(semantic, Mapping) else {}
    admission = (semantic.get("referent_match") == "supported" and
                 semantic.get("event_relation_match") == "supported" and
                 ev.get("target_support") == "supported" and
                 (response.get("_minimal_selected_id_valid") is not True or
                  response.get("_selected_evaluation_six_states_valid") is True))
    if not admission:
        return {"response_status": status, "ranked_geometry_ordinals": ranked_ordinals,
                "raw_ranked_candidate": raw_top, "pseudo_reference": None,
                "pseudo_reference_status": "no_information_spatial_admission_not_supported",
                "source_fallback": True}
    return {"response_status": status, "ranked_geometry_ordinals": ranked_ordinals,
            "raw_ranked_candidate": raw_top, "pseudo_reference": raw_top,
            "pseudo_reference_status": "selected_top_rank_spatial_support_gate_pass",
            "source_fallback": False}


def remapped_id_swap(base_response: Mapping[str, Any], swap_response: Mapping[str, Any]) -> dict[str, Any]:
    """Compare ID-swap rankings in stable geometry-ordinal space; top1 is separate."""
    base_status = classify_response(base_response)
    swap_status = classify_response(swap_response)
    if (base_status["category"] not in ("valid", "abstain") or
            swap_status["category"] not in ("valid", "abstain")):
        return {"status": "not_comparable_due_to_invalid_or_truncated_output",
                "exact_rank_match": None, "top1_geometry_match": None,
                "tie_aware_rank_agreement": None}
    bmeta, smeta = base_response.get("request_metadata"), swap_response.get("request_metadata")
    bparsed, sparsed = base_response.get("parsed"), swap_response.get("parsed")
    require(isinstance(bmeta, Mapping) and isinstance(smeta, Mapping) and
            isinstance(bparsed, Mapping) and isinstance(sparsed, Mapping),
            "ID-swap comparison lacks parsed response metadata")
    bm, sm = _catalog_maps(bmeta), _catalog_maps(smeta)
    bgeom, sgeom = bm["geometry_ordinal_by_visible_id"], sm["geometry_ordinal_by_visible_id"]
    require(set(bgeom.values()) == set(sgeom.values()),
            "ID swap must preserve the geometry-ordinal set")
    br = [bgeom[x] for x in bparsed.get("ranked_ids", [])]
    sr = [sgeom[x] for x in sparsed.get("ranked_ids", [])]
    bfull = len(br) == len(bgeom) and len(set(br)) == len(bgeom)
    sfull = len(sr) == len(sgeom) and len(set(sr)) == len(sgeom)
    exact = br == sr if bfull and sfull else None
    bchoice = br[0] if br and bparsed.get("abstain") is False else None
    schoice = sr[0] if sr and sparsed.get("abstain") is False else None
    both_abstained = base_status["category"] == "abstain" and swap_status["category"] == "abstain"
    top = (bchoice == schoice) if bchoice is not None and schoice is not None else None
    status = ("both_abstained_no_selected_id" if both_abstained else
              "no_selected_id_comparison_due_to_abstention" if bchoice is None or schoice is None else
              "compared_after_geometry_ordinal_remap" if bfull and sfull else
              "selected_id_only_after_geometry_ordinal_remap")
    return {"status": status,
            "base_geometry_ordinal_rank": br, "id_swap_geometry_ordinal_rank": sr,
            "exact_rank_match": exact, "top1_geometry_match": top,
            "base_has_selected_id": bchoice is not None, "id_swap_has_selected_id": schoice is not None,
            "both_abstained": both_abstained,
            "base_rank_complete": bfull, "id_swap_rank_complete": sfull,
            "tie_aware_rank_agreement": None,
            "tie_aware_rank_reason": "output schema contains ordinal ranks but no confidence scores; exact rank is descriptive and not a pass gate"}


def _metric_functions():
    """Load the same pinned official/independent dense metrics as the old scorer."""
    from scripts import stvg_target_teacher_evaluate_v2 as evaluator
    contract = evaluator.read_json(METRIC_CONTRACT)
    require(contract.get("schema") == "stvg_target_teacher_cpu_contract_v1",
            "existing legal dense-metric contract is unavailable")
    official, independent = evaluator._load_metric_functions(contract)
    return official, independent, contract


def score_tube_xyxy(boxes_xyxy_normalized: Any, frame_ids: Sequence[int], width: int, height: int,
                    interval: Sequence[int], truth_xyxy_pixels: Mapping[int, Sequence[float]],
                    span: Sequence[int], dataset: str, *, official: Any = None,
                    independent: Any = None) -> dict[str, Any]:
    """Score one full source-grid tube using the original official evaluator."""
    import numpy as np
    if official is None or independent is None:
        official, independent, _ = _metric_functions()
    require(dataset in ("hc2", "vidstg") and width > 0 and height > 0,
            "unsupported dataset or image dimensions")
    ids = [int(x) for x in frame_ids]
    require(len(ids) > 1 and all(b > a for a, b in zip(ids, ids[1:])),
            "source-grid physical IDs must be strictly increasing")
    interval_i = _physical_interval(interval, "prediction interval")
    span_i = _physical_interval(span, "ground-truth interval")
    boxes = np.asarray(boxes_xyxy_normalized, dtype=np.float64)
    require(boxes.shape == (len(ids), 4), "tube rows must align with the complete source frame grid")
    cxcywh = _xyxy_to_cxcywh(boxes)
    truth = {int(fid): [float(x) for x in box] for fid, box in truth_xyxy_pixels.items()}
    require(bool(truth) and all(len(box) == 4 and all(math.isfinite(x) for x in box)
                                for box in truth.values()), "official truth support is malformed")
    row = {"frame_ids": ids, "input": {"width": int(width), "height": int(height)}}
    reference = official(cxcywh, row, truth, span_i, interval_i, dataset, gt_space=False)
    manual = independent(cxcywh, row, truth, span_i, interval_i, dataset)[0]
    key_map = {"v": "m_vIoU", "t": "m_tIoU", "s": "m_sIoU_dense_GT"}
    metrics = {key_map[k]: float(reference[k]) for k in ("v", "t", "s")}
    max_error = max(abs(float(reference[k]) - float(manual[k])) for k in ("v", "t", "s"))
    require(max_error <= 2e-9,
            f"official and independent dense metric disagree: maximum absolute error {max_error}")
    require(all(math.isfinite(value) for value in metrics.values()), "non-finite official metric")
    return {"status": "scored", "metrics": metrics,
            "official_independent_max_abs_error": max_error,
            "source_grid_frames": len(ids), "source_grid_bounds": [ids[0], ids[-1]],
            "interval_frame_bounds_half_open": interval_i,
            "gt_support_frame_count": len(truth)}


def _truths_after_seal(source_cases: Mapping[str, Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Call the established paper scorer's target reader; caller must finish every gate first."""
    from scripts.score_stvg_opd_p1_v1 import truths
    by_dataset: dict[str, list[tuple[str, int]]] = {"hc2": [], "vidstg": []}
    for case_id, case in source_cases.items():
        dataset = str(case["dataset"])
        ordinal = case.get("query_ordinal")
        require(dataset in by_dataset and isinstance(ordinal, int) and not isinstance(ordinal, bool),
                f"source case {case_id} lacks an official scorer query ordinal")
        by_dataset[dataset].append((case_id, ordinal))
    target_by_case: dict[str, dict[str, Any]] = {}
    provenance: dict[str, Any] = {}
    for dataset, entries in by_dataset.items():
        if not entries:
            continue
        ordinals = [ordinal for _, ordinal in entries]
        require(len(ordinals) == len(set(ordinals)),
                f"P0 roster repeats an official query ordinal in {dataset}")
        dense, spans, bindings = truths(dataset, {"parents": ordinals})
        require(set(dense) == set(ordinals) and set(spans) == set(ordinals),
                f"official scorer returned an incomplete truth support map for {dataset}")
        for case_id, ordinal in entries:
            target_by_case[case_id] = {"truth": dense[ordinal], "span": spans[ordinal]}
        if isinstance(bindings, Mapping):
            provenance.update({str(k): v for k, v in bindings.items()})
    require(set(target_by_case) == set(source_cases), "official scorer did not return truth for all sealed cases")
    return target_by_case, provenance


def _candidate_pool_scores(source: Mapping[str, Any], response: Mapping[str, Any],
                           truth: Mapping[int, Sequence[float]], span: Sequence[int],
                           official: Any, independent: Any) -> dict[str, Any]:
    maps = source["catalog_maps"]
    boxes_by_id = source["by_visible_id"]
    predictions: dict[str, dict[str, Any]] = {}
    for visible_id, tube in boxes_by_id.items():
        metric = score_tube_xyxy(tube["boxes_xyxy_normalized"], tube["frame_ids"],
            int(response["request_metadata"]["frame_slots"][0]["render_layout"]["original_size"][0]),
            int(response["request_metadata"]["frame_slots"][0]["render_layout"]["original_size"][1]),
            tube["interval"], truth, span, str(response["dataset"]),
            official=official, independent=independent)
        predictions[visible_id] = {"geometry_ordinal": tube["geometry_ordinal"],
            "geometry_sha256": tube["geometry_key"], "metrics": metric["metrics"],
            "metric_audit": {k: v for k, v in metric.items() if k != "metrics"}}
    best_id = min(predictions, key=lambda cid: (
        -predictions[cid]["metrics"]["m_vIoU"], predictions[cid]["geometry_ordinal"]))
    return {"per_candidate": predictions, "oracle_id": best_id,
            "oracle": predictions[best_id], "oracle_diagnostic_only": True,
            "oracle_used_in_method_selection": False}


def _pseudo_metrics(decision: Mapping[str, Any], pool: Mapping[str, Any], native: Mapping[str, Any]) -> dict[str, Any]:
    selected = decision.get("pseudo_reference")
    if selected is None:
        return {"status": str(decision["pseudo_reference_status"]),
                "metrics": dict(native["metrics"]), "selected_geometry_ordinal": None,
                "fallback": "Native", "source_fallback": True}
    candidate = pool["per_candidate"][selected]
    return {"status": str(decision["pseudo_reference_status"]),
            "metrics": dict(candidate["metrics"]),
            "selected_geometry_ordinal": candidate["geometry_ordinal"],
            "fallback": None, "source_fallback": False}


def _wrong_query_evidence(response: Mapping[str, Any], source: Mapping[str, Any],
                          case: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Emit candidate-specific untrusted model evidence for root human review."""
    parsed = response.get("parsed")
    if (not isinstance(parsed, Mapping) or parsed.get("status") != "valid" or
            classify_response(response)["category"] not in ("valid", "abstain")):
        return []
    metadata = response["request_metadata"]
    maps = source["catalog_maps"]
    frame_slots = metadata.get("frame_slots")
    require(isinstance(frame_slots, list) and len(frame_slots) == 8,
            "wrong-query evidence must bind all eight physical input frames")
    evaluations = {row.get("candidate_id"): row for row in parsed.get("evaluations", [])
                   if isinstance(row, Mapping) and isinstance(row.get("candidate_id"), str)}
    evidence: list[dict[str, Any]] = []
    for visible_id in parsed["ranked_ids"]:
        ev = evaluations.get(visible_id)
        if not isinstance(ev, Mapping):
            continue
        raw_slots = ev.get("evidence_frame_slots")
        slot_unit_status = "unknown"
        slots: list[int] = []
        physical: list[int] = []
        if isinstance(raw_slots, list) and all(isinstance(slot, int) and not isinstance(slot, bool)
                                               for slot in raw_slots):
            if all(0 <= slot < len(frame_slots) for slot in raw_slots):
                slot_unit_status = "valid_slot_indices"
                slots = [int(slot) for slot in raw_slots]
                physical = [int(frame_slots[slot]["physical_frame_id"]) for slot in slots]
            else:
                known_physical = {int(frame["physical_frame_id"]) for frame in frame_slots
                                  if isinstance(frame, Mapping) and
                                  isinstance(frame.get("physical_frame_id"), int)}
                if raw_slots and set(raw_slots).issubset(known_physical):
                    slot_unit_status = "physical_frame_ids_not_slot_indices"
                    physical = [int(value) for value in raw_slots]
                else:
                    slot_unit_status = "invalid_slot_indices"
        elif raw_slots is not None:
            slot_unit_status = "invalid_slot_indices"
        geometry = source["by_visible_id"][visible_id]
        boxes = geometry["boxes_xyxy_normalized"]
        frame_ids = list(map(int, geometry["frame_ids"].tolist()))
        pos = {fid: i for i, fid in enumerate(frame_ids)}
        # Invalid physical-vs-slot units never become pixel grounding evidence.
        sample_boxes = ([list(map(float, boxes[pos[fid]])) for fid in physical if fid in pos]
                        if slot_unit_status == "valid_slot_indices" else [])
        rgb_refs: list[dict[str, Any]] = []
        rgb = case.get("sparse_RGB") or case.get("source_RGB") or {}
        for slot in slots:
            fid = int(frame_slots[slot]["physical_frame_id"])
            entry = rgb.get(fid, rgb.get(str(fid))) if isinstance(rgb, Mapping) else None
            if isinstance(entry, Mapping):
                rgb_refs.append({"physical_frame_id": fid, "path": entry.get("path"),
                                 "file_sha256": entry.get("sha256"),
                                 "RGB_sha256": entry.get("RGB_sha256")})
        catalog_entry = maps["entries_by_id"][visible_id]
        evidence.append({"branch": str(response["branch"]), "control": "wrong_query",
            "visible_candidate_id": visible_id,
            "geometry_ordinal": int(geometry["geometry_ordinal"]),
            "geometry_sha256": str(geometry["geometry_key"]),
            "candidate_interval_physical_frame_bounds_half_open": catalog_entry.get(
                "interval_physical_frame_bounds_half_open"),
            "candidate_sampled_xyxy_normalized_by_evidence_slot": sample_boxes,
            "evidence_frame_slots_as_supplied": raw_slots,
            "evidence_frame_slot_unit_status": slot_unit_status,
            "evidence_frame_slots_interpreted_as_slots": slots,
            "evidence_physical_frame_ids": physical,
            "source_rgb_references": rgb_refs,
            "model_evaluation": {"semantic": ev.get("semantic"), "boundary": ev.get("boundary"),
                "target_support": ev.get("target_support"), "evidence_note": ev.get("evidence_note"),
                "text_is_untrusted_model_data": True},
            "query_sha256": metadata.get("query_sha256"),
            "query_control_distribution_shift": True,
            "machine_contradiction_claim": None,
            "human_adjudication_required": True})
    return evidence


def _source_cases_for_responses(by_case: Mapping[str, Mapping[str, Any]],
                                responses: Mapping[tuple[str, str, str], Mapping[str, Any]],
                                runtime: Mapping[str, Any], repo_root: Path = ROOT) -> dict[str, Any]:
    """Load and bind source cases after the complete response seal, before GT."""
    import torch
    require(not torch.cuda.is_initialized(), "review process must remain CPU-only before source loads")
    torch.set_num_threads(2)
    prompt_path = safe_repo_path("scripts/stvg_sparse_r1_prompt_v1.py", repo_root)
    require(runtime["verified_pins"].get("scripts/stvg_sparse_r1_prompt_v1.py") ==
            sha256_file(prompt_path), "sealed runtime prompt source pin differs")
    from scripts import stvg_sparse_r1_prompt_v1 as prompt
    result: dict[str, Any] = {}
    for case_id, row in by_case.items():
        path = safe_repo_path(str(row["source_case_path"]), repo_root)
        require(path.is_file() and sha256_file(path) == row["source_case_sha256"],
                f"cold source case bytes differ from the P0 roster: {case_id}")
        case = prompt.load_source_case(path)
        require(case.get("_source_case_sha256") == row["source_case_sha256"] and
                case.get("case_id") == case_id and case.get("dataset") == row["dataset"] and
                case.get("GT_read") is False and case.get("original_source_only") is True and
                case.get("DINO_read_or_call") is False and
                case.get("DINO_state_initialization") is False and
                case.get("old_adapted_state_loaded") is False and
                case.get("optimizer_steps") == 0 and case.get("backwards") == 0,
                f"cold source case custody/identity is invalid: {case_id}")
        expected_source_runtime = row.get("source_runtime_sha256")
        if expected_source_runtime is not None:
            old_runtime = SOURCE_BASE / "RUNTIME.json"
            require(old_runtime.is_file() and sha256_file(old_runtime) == expected_source_runtime and
                    case.get("runtime_sha256") == expected_source_runtime,
                    f"old source runtime binding differs for {case_id}")
        requests: dict[tuple[str, str], Mapping[str, Any]] = {}
        for branch in BRANCHES:
            for control in P0_CONTROLS:
                response = responses[(case_id, branch, control)]
                requests[(branch, control)] = response["request_metadata"]
        # Base is canonical for the common source catalog. ID swap must preserve
        # geometry; wrong query must preserve catalog/video while changing query.
        catalogs = {branch: _source_catalog(case, requests[(branch, "base")]) for branch in BRANCHES}
        for branch in BRANCHES:
            base_meta = requests[(branch, "base")]
            for control in ("id_swap", "wrong_query"):
                maps = _catalog_maps(requests[(branch, control)])
                base_maps = _catalog_maps(base_meta)
                require(set(maps["geometry_key_by_visible_id"].values()) ==
                        set(base_maps["geometry_key_by_visible_id"].values()),
                        f"{control} changed the P0 geometry set: {case_id}/{branch}")
        result[case_id] = {"case": case, "catalogs": catalogs}
        require(not torch.cuda.is_initialized(), "source cache loading initialized CUDA")
    # The worker hashes the rendered official chat, while request metadata
    # hashes the plain prompt body. Reconstruct both before opening any labels.
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(str(repo_root / 'checkpoints/STVG-R1-7B'), local_files_only=True)
    ordered = list(by_case)
    for (case_id, branch, control), response in responses.items():
        case = result[case_id]['case']; metadata = response['request_metadata']
        peers = [cid for cid in ordered if by_case[cid]['dataset']==by_case[case_id]['dataset']]
        other = peers[(peers.index(case_id)+1)%len(peers)]
        query = result[other]['case']['caption'] if control=='wrong_query' else case['caption']
        require(sha256_bytes(query.encode())==metadata['query_sha256'], 'query-only control binding differs')
        slots = metadata['frame_slots']
        frame_ids = [x['physical_frame_id'] for x in slots]
        times = [x['source_timestamp_seconds'] for x in slots]
        body = prompt._prompt_for(branch, query, metadata['catalog'], frame_ids, times)
        require(sha256_bytes(body.encode())==metadata['prompt_sha256'], 'plain candidate prompt SHA differs')
        messages=[{'role':'user','content':[{'type':'video'},{'type':'text','text':body}]}]
        rendered=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        require(sha256_bytes(rendered.encode())==response['prompt_sha256'], 'rendered official chat SHA differs')
    return result


def _score_review_rows(sealed: Mapping[str, Any], by_case: Mapping[str, Mapping[str, Any]],
                       source_cases: Mapping[str, Any], targets: Mapping[str, Any],
                       official: Any, independent: Any,
                       readbacks: Mapping[tuple[str, str, str], Mapping[str, Any]],
                       frozen_sources: Mapping[tuple[str, str, str], Mapping[str, Any]],
                       frozen_decisions: Mapping[tuple[str, str, str], Mapping[str, Any]],
                       frozen_equivariance: Mapping[tuple[str, str], Mapping[str, Any]],
                       frozen_wrong_query_evidence: Mapping[tuple[str, str, str], Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    status_counts: dict[str, int] = {}
    equivariance: list[dict[str, Any]] = []
    selected_id_equivariance: list[dict[str, Any]] = []
    wrong_evidence: list[dict[str, Any]] = []
    for case_id in sealed["case_ids"]:
        case = source_cases[case_id]["case"]
        target = targets[case_id]
        width, height = int(case["input"]["width"]), int(case["input"]["height"])
        frame_ids = _to_numpy(case["frame_ids"], "case.frame_ids").astype("int64").tolist()
        native_boxes = _to_numpy(case["native_boxes"], "case.native_boxes").astype("float64")
        native_interval = _physical_interval(case["Native_interval"], "Native_interval")
        projection_interval = _physical_interval(case["Projection_interval"], "Projection_interval")
        native_metric = score_tube_xyxy(native_boxes, frame_ids, width, height, native_interval,
            target["truth"], target["span"], str(case["dataset"]), official=official, independent=independent)
        projection_metric = score_tube_xyxy(native_boxes, frame_ids, width, height, projection_interval,
            target["truth"], target["span"], str(case["dataset"]), official=official, independent=independent)
        for branch in BRANCHES:
            response_by_control = {control: sealed["responses"][(case_id, branch, control)]
                                   for control in P0_CONTROLS}
            for control in P0_CONTROLS:
                response = response_by_control[control]
                key = (case_id, branch, control)
                source = frozen_sources[key]
                decision = frozen_decisions[key]
                category = decision["response_status"]["category"]
                status_counts[category] = status_counts.get(category, 0) + 1
                pool = _candidate_pool_scores(source, response, target["truth"], target["span"],
                                               official, independent)
                raw_top = decision.get("raw_ranked_candidate")
                raw_rank = None
                if raw_top is not None and decision["response_status"]["category"] in ("valid", "abstain"):
                    scored = pool["per_candidate"][raw_top]
                    ranking_ids = response.get("parsed", {}).get("ranked_ids", [])
                    abstained = response.get("parsed", {}).get("abstain") is True
                    raw_rank = {"status": ("abstained_top_rank_scored_diagnostic" if abstained
                                           else "selected_top_id_scored"),
                        "geometry_ordinal": scored["geometry_ordinal"],
                        "metrics": dict(scored["metrics"]),
                        "usable_as_pseudo_reference": not decision["source_fallback"],
                        "actual_model_choice": not abstained,
                        "ranked_ids_count": len(ranking_ids),
                        "rank_order_complete": len(ranking_ids) == len(response["request_metadata"]["catalog"]["candidate_ids"]),
                        "full_rank_correlation_computed": False}
                pseudo = _pseudo_metrics(decision, pool, native_metric)
                result = {"case_id": case_id, "dataset": str(case["dataset"]),
                    "query_ordinal": int(case["query_ordinal"]), "branch": branch, "control": control,
                    "response_status": decision["response_status"],
                    "Native": native_metric, "Projection": projection_metric,
                    "candidate_oracle": pool["oracle"],
                    "candidate_oracle_diagnostic_only": True,
                    "candidate_oracle_used_in_selection": False,
                    "candidate_grid_metrics": pool["per_candidate"],
                    "raw_rank_selection": raw_rank,
                    "pseudo_reference": pseudo,
                    "pseudo_reference_decision_frozen_before_GT": True,
                    "output_readback": dict(readbacks[(case_id, branch, control)])}
                if control == "wrong_query":
                    wrong_evidence.extend(frozen_wrong_query_evidence.get(key, []))
                    result["wrong_query_interpretation"] = {
                        "query_control_distribution_shift": True,
                        "contradiction_assumed_from_text": False,
                        "machine_contradiction_claim": None,
                        "human_candidate_specific_visual_review_required": True}
                rows.append(result)
            equivariance_row = frozen_equivariance[(case_id, branch)]
            base_ordinals = equivariance_row.get("base_geometry_ordinal_rank") or []
            swap_ordinals = equivariance_row.get("id_swap_geometry_ordinal_rank") or []
            equivariance.append({"case_id": case_id, "branch": branch, **equivariance_row})
            base_choice_exists = (response_by_control["base"].get("parsed", {}).get("abstain") is False)
            swap_choice_exists = (response_by_control["id_swap"].get("parsed", {}).get("abstain") is False)
            selected_id_equivariance.append({"case_id": case_id, "branch": branch,
                "status": equivariance_row["status"],
                "base_abstained": response_by_control["base"]["parsed"].get("abstain") is True,
                "id_swap_abstained": response_by_control["id_swap"]["parsed"].get("abstain") is True,
                "both_abstained": equivariance_row.get("both_abstained", False),
                "base_selected_geometry_ordinal": base_ordinals[0] if base_ordinals and base_choice_exists else None,
                "id_swap_selected_geometry_ordinal": swap_ordinals[0] if swap_ordinals and swap_choice_exists else None,
                "selected_id_equivariant": equivariance_row.get("top1_geometry_match"),
                "full_rank_order_comparison_available": equivariance_row.get("exact_rank_match") is not None})
    require(len(rows) == P0_REQUESTS, "post-seal review did not score all 48 request cells")
    return {"case_rows": rows, "response_status_counts": status_counts,
            "id_swap_equivariance": equivariance,
            "selected_id_equivariance": selected_id_equivariance,
            "wrong_query_candidate_evidence": wrong_evidence}


def review_after_seal(*, seal_path: Path = SEAL_DEFAULT, roster_path: Path = ROSTER_DEFAULT,
                      runtime_path: Path = RUNTIME_DEFAULT, output_path: Path,
                      repo_root: Path = ROOT) -> dict[str, Any]:
    """Full post-seal CPU review. This is the sole function allowed to call GT."""
    require(Path(seal_path).is_file(), "P0 seal is absent; refusing all GT/annotation access")
    sealed = preflight_global_seal(seal_path, repo_root=repo_root)
    roster, by_case = _load_locked_roster(Path(roster_path), sealed["seal"]["roster_sha256"],
                                         set(sealed["case_ids"]))
    runtime = _validate_runtime(Path(runtime_path), sealed["seal"]["runtime_sha256"], repo_root)
    from scripts import stvg_sparse_r1_prompt_v1 as prompt
    readbacks: dict[tuple[str, str, str], dict[str, Any]] = {}
    response_views: dict[tuple[str, str, str], dict[str, Any]] = {}
    for (case_id, branch, control), response in sealed["responses"].items():
        row = by_case[case_id]
        require(response["dataset"] == row["dataset"] and
                response["source_case_sha256"] == row["source_case_sha256"],
                f"response source/dataset binding differs from locked roster: {case_id}/{branch}/{control}")
        readback, view = lossless_output_readback(response, prompt.parse_output)
        readbacks[(case_id, branch, control)] = readback
        response_views[(case_id, branch, control)] = view
    effective_seal = {**sealed, "responses": response_views}
    # Load exact cold, GT-free source arrays only after all 48 output hashes and
    # roster/runtime bindings pass. Freeze every algorithmic decision now.
    source_cases = _source_cases_for_responses(by_case, response_views, runtime, repo_root)
    frozen_sources: dict[tuple[str, str, str], dict[str, Any]] = {}
    frozen_decisions: dict[tuple[str, str, str], dict[str, Any]] = {}
    frozen_equivariance: dict[tuple[str, str], dict[str, Any]] = {}
    frozen_wrong_query_evidence: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for case_id, entry in source_cases.items():
        case = entry["case"]
        row = by_case[case_id]
        require(case.get("query_ordinal") == row.get("query_ordinal", case.get("query_ordinal")),
                f"source query ordinal differs from roster for {case_id}")
        for branch in BRANCHES:
            for control in P0_CONTROLS:
                response = response_views[(case_id, branch, control)]
                if readbacks[(case_id, branch, control)]["structural_wrapper_valid"]:
                    _validate_parsed_response(response, response["request_metadata"])
                source = _source_catalog(case, response["request_metadata"])
                key = (case_id, branch, control)
                frozen_sources[key] = source
                # Decisions use only the response, never source-to-GT scores.
                frozen_decisions[key] = _decision_for(response, response["request_metadata"], source)
                if control == "wrong_query":
                    frozen_wrong_query_evidence[key] = _wrong_query_evidence(response, source, case)
            frozen_equivariance[(case_id, branch)] = remapped_id_swap(
                response_views[(case_id, branch, "base")],
                response_views[(case_id, branch, "id_swap")])
    # All label-free evidence verification is complete before this sole call
    # to the established original scorer's target loader.
    official, independent, metric_contract = _metric_functions()
    targets, truth_provenance = _truths_after_seal({key: value["case"]
                                                    for key, value in source_cases.items()})
    scored = _score_review_rows(effective_seal, by_case, source_cases, targets,
                                official, independent, readbacks, frozen_sources,
                                frozen_decisions, frozen_equivariance,
                                frozen_wrong_query_evidence)
    original_strict_counts: dict[str, int] = {}
    readback_counts: dict[str, int] = {}
    by_branch_control: dict[str, dict[str, Any]] = {}
    for key, response in sealed["responses"].items():
        _, branch, control = key
        strict_status = str(response.get("parsed", {}).get("status", "missing"))
        original_strict_counts[strict_status] = original_strict_counts.get(strict_status, 0) + 1
        item = readbacks[key]
        for field in ("strict_bare_valid", "structural_wrapper_valid",
                      "minimal_selected_id_valid", "wrapper_removed"):
            if item[field]:
                readback_counts[field] = readback_counts.get(field, 0) + 1
        group_key = f"{branch}/{control}"
        group = by_branch_control.setdefault(group_key, {
            "request_count": 0, "strict_bare_valid_count": 0,
            "structural_wrapper_valid_count": 0, "minimal_selected_id_valid_count": 0,
            "wrapper_removed_count": 0,
            "truncated_count": 0, "abstention_count": 0, "selected_id_available_count": 0})
        group["request_count"] += 1
        group["strict_bare_valid_count"] += int(item["strict_bare_valid"])
        group["structural_wrapper_valid_count"] += int(item["structural_wrapper_valid"])
        group["minimal_selected_id_valid_count"] += int(item["minimal_selected_id_valid"])
        group["wrapper_removed_count"] += int(item["wrapper_removed"])
        group["truncated_count"] += int(item["truncated_at_cap"])
        group["abstention_count"] += int(item.get("abstain") is True)
        group["selected_id_available_count"] += int(item.get("selected_id") is not None)
    report = {"schema": SCHEMA, "status": "postseal_review_complete",
        "cohort": "P0 eight-case sparse semantic screen; historically exposed source pool",
        "case_count": 8, "request_count": P0_REQUESTS,
        "prediction_seal_path": sealed["seal_path"], "prediction_seal_sha256": sealed["seal_sha256"],
        "roster_sha256": sealed["seal"]["roster_sha256"],
        "runtime_sha256": sealed["seal"]["runtime_sha256"],
        "runtime_source_pins_verified": runtime["verified_pins"],
        "official_metric_contract_sha256": sha256_file(METRIC_CONTRACT),
        "official_metric_module_pins": metric_contract["metric_module_pins"],
        "official_truth_provenance": truth_provenance,
        "metrics": scored["case_rows"],
        "response_status_counts": scored["response_status_counts"],
        "original_strict_parser_status_counts": original_strict_counts,
        "lossless_output_readback_counts": {
            "request_count": P0_REQUESTS,
            "original_strict_bare_valid_count": readback_counts.get("strict_bare_valid", 0),
            "structural_wrapper_valid_count": readback_counts.get("structural_wrapper_valid", 0),
            "minimal_selected_id_valid_count": readback_counts.get("minimal_selected_id_valid", 0),
            "abstention_count": sum(1 for response in response_views.values()
                                    if classify_response(response)["category"] == "abstain"),
            "selected_id_available_count": sum(1 for item in readbacks.values()
                                                if item.get("selected_id") is not None),
            "wrapper_removed_count": readback_counts.get("wrapper_removed", 0),
            "truncated_count": sum(int(item["truncated_at_cap"]) for item in readbacks.values()),
            "original_strict_bare_valid_rate": readback_counts.get("strict_bare_valid", 0) / P0_REQUESTS,
            "structural_wrapper_valid_rate": readback_counts.get("structural_wrapper_valid", 0) / P0_REQUESTS,
            "minimal_selected_id_valid_rate": readback_counts.get("minimal_selected_id_valid", 0) / P0_REQUESTS,
            "original_strict_parser_status_counts": original_strict_counts,
            "by_branch_control": by_branch_control,
            "policy": "raw bytes and stored parsed values remain unchanged; only exact JSON fences are removed for one strict reparse; no prose stripping, repair, field completion, or truncated response is structurally eligible",
        },
        "id_swap_equivariance": scored["id_swap_equivariance"],
        "selected_id_equivariance": scored["selected_id_equivariance"],
        "wrong_query_candidate_specific_visual_evidence": scored["wrong_query_candidate_evidence"],
        "wrong_query_policy": "text changes are a distribution-shift control, not an assumed contradiction; candidate evidence is for human review",
        "oracle_policy": "post-seal diagnostic only; never used in rerank or pseudo-reference selection",
        "id_swap_policy": "rank order is remapped to stable geometry ordinals; top-1 is reported separately; no confidence scores exist for tie-aware exact ranking",
        "GT_read": True, "GT_opened_after_complete_prediction_source_runtime_seals": True,
        "historically_exposed": True, "fresh_unseen_confirmation_claim": False,
        "model_calls": 0, "GPU_used": False, "RGB_opened_by_reviewer": False,
        "timestamp_unix": __import__("time").time()}
    output_path = Path(output_path).resolve()
    require(not output_path.exists(), "preserve any prior post-seal review output")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    report["output_path"] = str(output_path.relative_to(repo_root.resolve()))
    report["output_sha256"] = sha256_file(output_path)
    return report


def _synthetic_response(case_id: str, branch: str, control: str, *,
                        visible_ids: Sequence[str], geom_by_id: Mapping[str, str],
                        source_by_id: Mapping[str, str], rank: Sequence[str],
                        status: str = "valid", abstain: bool = False,
                        truncated: bool = False) -> dict[str, Any]:
    catalog_entries = []
    for cid in visible_ids:
        entry: dict[str, Any] = {"id": cid, "interval_seconds": [0.0, 1.0], "color_rgb": [1, 2, 3]}
        if branch == "temporal":
            entry["interval_physical_frame_bounds_half_open"] = [0.0, 4.0]
        else:
            entry["boxes_by_video_slot_normalized_xyxy"] = [[.1, .1, .5, .5]] * 8
            entry["box_frame_ids"] = list(range(8))
        catalog_entries.append(entry)
    catalog = {"schema": "stvg-sparse-r1-candidate-catalog-v1", "branch": branch,
               "candidate_ids": list(visible_ids), "candidates": catalog_entries}
    qhash = sha256_bytes((case_id if control != "wrong_query" else case_id + "/other-query").encode())
    metadata = {"schema": "stvg-sparse-r1-prompt-request-v1", "branch": branch,
        "case_id": case_id, "query_sha256": qhash,
        "query_control_override": control == "wrong_query",
        "query_control_distribution_shift": control == "wrong_query",
        "rendered_video_sha256": "1" * 64,
        "candidate_catalog_sha256": "2" * 64,
        "prompt_sha256": "3" * 64, "GT_read": False, "model_called": False,
        "frame_slots": [{"slot": slot, "physical_frame_id": 100 + slot,
                         "render_layout": {"original_size": [96, 72]}}
                        for slot in range(8)],
        "catalog": catalog, "candidate_audit": {
            "source_candidate_id_by_visible_id": dict(source_by_id),
            "geometry_key_by_visible_id": dict(geom_by_id)}}
    parsed = {"schema": EXPECTED_PARSER_SCHEMA, "branch": branch,
        "ranked_ids": list(rank), "abstain": abstain,
        "abstain_reason": "insufficient_visual_evidence" if abstain else None,
        "evaluations": [{"candidate_id": cid,
            "semantic": {"referent_match": "supported", "event_relation_match": "supported"},
            "boundary": {"start_boundary": "uncertain", "end_boundary": "uncertain", "extent": "supported"},
            "target_support": "supported", "evidence_frame_slots": [0], "evidence_note": "synthetic"}
            for cid in rank], "status": status}
    raw = json.dumps({key: value for key, value in parsed.items() if key != "status"},
                     ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    if status == "invalid_output":
        raw = "not json"
        parsed = {"status": "invalid_output", "error_type": "ValueError",
                  "error": "response must be one bare JSON object without markdown or surrounding prose"}
    return {"status": "accepted_actual_sparse_response", "case_id": case_id,
        "dataset": "hc2", "branch": branch, "control": control,
        "parsed": parsed, "request_metadata": metadata, "raw_output": raw,
        "prompt_sha256": metadata["prompt_sha256"], "source_case_sha256": "4" * 64,
        "runtime_sha256": "5" * 64, "GT_read": False, "teacher_frozen": True,
        "teacher_optimizer_steps": 0, "output_truncated_at_cap": truncated}


def synthetic_contracts() -> dict[str, Any]:
    """Synthetic-only contracts for barrier gates, status separation and math."""
    import numpy as np
    from vg_tta.tastvg_oracle_event5_v1 import official
    from scripts.review_stvg_opd_revised_p1_v2 import dense_metric
    from scripts.stvg_sparse_r1_prompt_v1 import parse_output
    checks: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="stvg-sparse-r1-review-contract-") as tmp:
        root = Path(tmp)
        case_ids = [f"synthetic_{i:02d}" for i in range(8)]
        refs = []
        responses = {}
        for case_id in case_ids:
            for branch in BRANCHES:
                base_ids = ["a", "b", "c"] if branch == "temporal" else ["u", "v", "w"]
                gkeys = ["a" * 64, "b" * 64, "c" * 64]
                sources = ["g1", "g2", "g3"]
                base_map = dict(zip(base_ids, gkeys))
                source_map = dict(zip(base_ids, sources))
                base_rank = [base_ids[0], base_ids[1], base_ids[2]]
                swapped_ids = list(reversed(base_ids))
                swapped_geom = dict(zip(swapped_ids, gkeys))
                swapped_source = dict(zip(swapped_ids, sources))
                swapped_rank = list(swapped_ids)
                wrong_rank = [base_ids[0], base_ids[2], base_ids[1]]
                for control, ids, gm, sm, rank in (
                    ("base", base_ids, base_map, source_map, base_rank),
                    ("id_swap", swapped_ids, swapped_geom, swapped_source, swapped_rank),
                    ("wrong_query", base_ids, base_map, source_map, wrong_rank)):
                    value = _synthetic_response(case_id, branch, control, visible_ids=ids,
                        geom_by_id=gm, source_by_id=sm, rank=rank)
                    path = root / f"{case_id}_{branch}_{control}.json"
                    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")
                    refs.append({"path": path.name, "sha256": sha256_file(path)})
                    responses[(case_id, branch, control)] = value
        seal_path = root / "seal.json"
        seal = {"status": "complete", "scope": "P0_all_sparse_responses_sealed",
            "count": P0_REQUESTS, "files": refs, "roster_sha256": "6" * 64,
            "runtime_sha256": "5" * 64, "GT_read": False}
        seal_path.write_text(json.dumps(seal), encoding="utf-8")
        snapshot = preflight_global_seal(seal_path, repo_root=root, expected_case_ids=case_ids)
        checks["complete_48_response_barrier_passes"] = len(snapshot["responses"]) == 48
        calls: list[str] = []
        broken = dict(seal, count=47)
        bad_path = root / "broken.json"
        bad_path.write_text(json.dumps(broken), encoding="utf-8")
        try:
            preflight_global_seal(bad_path, repo_root=root)
        except ReviewError:
            checks["incomplete_barrier_rejected_before_any_GT_reader"] = calls == []
        else:
            raise AssertionError("incomplete synthetic barrier was accepted")
        base = responses[(case_ids[0], "spatial", "base")]
        swap = responses[(case_ids[0], "spatial", "id_swap")]
        eq = remapped_id_swap(base, swap)
        checks["id_swap_remaps_to_geometry_and_top1_is_separate"] = (
            eq["exact_rank_match"] is True and eq["top1_geometry_match"] is True and
            eq["tie_aware_rank_agreement"] is None)
        no_tie_pass = remapped_id_swap(base,
            {**swap, "parsed": {**swap["parsed"], "ranked_ids": ["w", "u", "v"]}})
        checks["rank_order_can_differ_without_changing_top1"] = (
            no_tie_pass["exact_rank_match"] is False and no_tie_pass["top1_geometry_match"] is True)
        checks["wrong_query_is_not_labeled_contradictory"] = (
            base["request_metadata"]["query_sha256"] !=
            responses[(case_ids[0], "spatial", "wrong_query")]["request_metadata"]["query_sha256"])
        both_abstain_base = _synthetic_response("synthetic", "spatial", "base",
            visible_ids=["u", "v", "w"], geom_by_id={"u": "a" * 64, "v": "b" * 64, "w": "c" * 64},
            source_by_id={"u": "g1", "v": "g2", "w": "g3"}, rank=["u", "v", "w"], abstain=True)
        both_abstain_swap = _synthetic_response("synthetic", "spatial", "id_swap",
            visible_ids=["w", "v", "u"], geom_by_id={"w": "a" * 64, "v": "b" * 64, "u": "c" * 64},
            source_by_id={"w": "g1", "v": "g2", "u": "g3"}, rank=["w", "v", "u"], abstain=True)
        abstain_equivariance = remapped_id_swap(both_abstain_base, both_abstain_swap)
        checks["double_abstention_is_not_reported_as_selected_id_equivariance"] = (
            abstain_equivariance["both_abstained"] is True and
            abstain_equivariance["top1_geometry_match"] is None and
            abstain_equivariance["status"] == "both_abstained_no_selected_id")
        abstain = _synthetic_response("synthetic", "spatial", "base", visible_ids=["a"],
            geom_by_id={"a": "a" * 64}, source_by_id={"a": "baseline"}, rank=["a"], abstain=True)
        invalid = _synthetic_response("synthetic", "spatial", "base", visible_ids=["a"],
            geom_by_id={"a": "a" * 64}, source_by_id={"a": "baseline"}, rank=["a"], status="invalid_output")
        trunc = _synthetic_response("synthetic", "spatial", "base", visible_ids=["a"],
            geom_by_id={"a": "a" * 64}, source_by_id={"a": "baseline"}, rank=["a"], truncated=True)
        checks["valid_abstain_truncated_invalid_are_distinct"] = {
            "valid": classify_response(base)["category"],
            "abstain": classify_response(abstain)["category"],
            "truncated": classify_response(trunc)["category"],
            "invalid": classify_response(invalid)["category"],
        } == {"valid": "valid", "abstain": "abstain", "truncated": "truncated", "invalid": "invalid_output"}
        bare = _synthetic_response("synthetic", "spatial", "base", visible_ids=["a"],
            geom_by_id={"a": "a" * 64}, source_by_id={"a": "baseline"}, rank=["a"])
        bare_readback, bare_view = lossless_output_readback(bare, parse_output)
        checks["bare_json_is_strict_and_structurally_valid"] = (
            bare_readback["strict_bare_valid"] is True and
            bare_readback["structural_wrapper_valid"] is True and
            bare_readback["wrapper_removed"] is False and
            classify_response(bare_view)["category"] == "valid")

        fenced_raw = "```json\n" + bare["raw_output"] + "\n```"
        wrapped = dict(bare, raw_output=fenced_raw)
        try:
            parse_output(fenced_raw, bare["request_metadata"]["catalog"])
        except (ValueError, TypeError, KeyError) as exc:
            wrapped["parsed"] = {"status": "invalid_output", "error_type": type(exc).__name__,
                                 "error": str(exc)}
        else:
            raise AssertionError("strict parser unexpectedly accepted the fenced output")
        wrapped_readback, wrapped_view = lossless_output_readback(wrapped, parse_output)
        checks["exact_json_fence_is_losslessly_reparsed_without_rewriting_original"] = (
            wrapped_readback["strict_bare_valid"] is False and
            wrapped_readback["structural_wrapper_valid"] is True and
            wrapped_readback["wrapper_removed"] is True and
            wrapped["raw_output"] == fenced_raw and
            wrapped["parsed"]["status"] == "invalid_output" and
            classify_response(wrapped_view)["category"] == "valid")

        tailed_raw = fenced_raw + "\nextra"
        tailed = dict(wrapped, raw_output=tailed_raw)
        try:
            parse_output(tailed_raw, bare["request_metadata"]["catalog"])
        except (ValueError, TypeError, KeyError) as exc:
            tailed["parsed"] = {"status": "invalid_output", "error_type": type(exc).__name__,
                                "error": str(exc)}
        else:
            raise AssertionError("strict parser unexpectedly accepted output with trailing prose")
        tailed_readback, _ = lossless_output_readback(tailed, parse_output)
        checks["wrapper_with_trailing_text_is_rejected"] = (
            tailed_readback["structural_wrapper_valid"] is False and
            tailed_readback["wrapper_removed"] is False)

        truncated = dict(bare, output_truncated_at_cap=True)
        truncated_readback, truncated_view = lossless_output_readback(truncated, parse_output)
        checks["truncation_never_becomes_structurally_eligible"] = (
            truncated_readback["strict_bare_valid"] is True and
            truncated_readback["structural_wrapper_valid"] is False and
            classify_response(truncated_view)["category"] == "truncated")

        minimal_payload = {"schema": EXPECTED_PARSER_SCHEMA, "branch": "spatial",
            "ranked_ids": ["u"], "abstain": False, "abstain_reason": None}
        minimal_raw = json.dumps(minimal_payload, separators=(",", ":"))
        partial = dict(_synthetic_response("synthetic", "spatial", "base", visible_ids=["u", "v", "w"],
            geom_by_id={"u": "a" * 64, "v": "b" * 64, "w": "c" * 64},
            source_by_id={"u": "g1", "v": "g2", "w": "g3"}, rank=["u", "v", "w"]))
        partial["raw_output"] = minimal_raw
        try:
            parse_output(minimal_raw, partial["request_metadata"]["catalog"])
        except (ValueError, TypeError, KeyError) as exc:
            partial["parsed"] = {"status": "invalid_output", "error_type": type(exc).__name__,
                                 "error": str(exc)}
        else:
            raise AssertionError("strict full-schema parser unexpectedly accepted a partial rank")
        partial_readback, partial_view = lossless_output_readback(partial, parse_output)
        checks["minimal_selected_id_accepts_only_real_partial_rank_without_completion"] = (
            partial_readback["strict_bare_valid"] is False and
            partial_readback["structural_wrapper_valid"] is False and
            partial_readback["minimal_selected_id_valid"] is True and
            partial_readback["selected_id"] == "u" and
            partial_readback["rank_order_complete"] is False and
            partial_readback["selected_evaluation_six_states_valid"] is False and
            partial_view["parsed"]["ranked_ids"] == ["u"] and
            "evaluations" not in partial_view["parsed"])

        selected_eval = {"candidate_id": "u",
            "semantic": {"referent_match": "supported", "event_relation_match": "uncertain"},
            "boundary": {"start_boundary": "supported", "end_boundary": "uncertain", "extent": "supported"},
            "target_support": "supported", "evidence_frame_slots": [100],
            "evidence_note": "synthetic only"}
        partial_eval_payload = {**minimal_payload, "evaluations": [selected_eval]}
        partial_eval_inner = json.dumps(partial_eval_payload, separators=(",", ":"))
        partial_eval_fenced = "```json\n" + partial_eval_inner + "\n```"
        partial_eval = dict(partial, raw_output=partial_eval_fenced, output_truncated_at_cap=False)
        try:
            parse_output(partial_eval_fenced, partial_eval["request_metadata"]["catalog"])
        except (ValueError, TypeError, KeyError) as exc:
            partial_eval["parsed"] = {"status": "invalid_output", "error_type": type(exc).__name__,
                                       "error": str(exc)}
        else:
            raise AssertionError("strict full-schema parser unexpectedly accepted a partial fenced answer")
        partial_eval_readback, partial_eval_view = lossless_output_readback(partial_eval, parse_output)
        checks["partial_eval_states_are_kept_and_physical_frame_units_are_flagged"] = (
            partial_eval_readback["minimal_selected_id_valid"] is True and
            partial_eval_readback["structural_wrapper_valid"] is False and
            partial_eval_readback["wrapper_removed"] is True and
            partial_eval_readback["selected_evaluation_six_states_valid"] is True and
            partial_eval_readback["selected_evidence_frame_unit_status"] ==
                "physical_frame_ids_not_slot_indices" and
            partial_eval_view["parsed"]["evaluations"] == [selected_eval])

        empty_abstain_payload = {"schema": EXPECTED_PARSER_SCHEMA, "branch": "spatial",
            "ranked_ids": [], "abstain": True, "abstain_reason": "candidate_set_inadequate"}
        empty_abstain_raw = json.dumps(empty_abstain_payload, separators=(",", ":"))
        empty_abstain = dict(partial, raw_output=empty_abstain_raw, output_truncated_at_cap=False)
        try:
            parse_output(empty_abstain_raw, empty_abstain["request_metadata"]["catalog"])
        except (ValueError, TypeError, KeyError) as exc:
            empty_abstain["parsed"] = {"status": "invalid_output", "error_type": type(exc).__name__,
                                        "error": str(exc)}
        else:
            raise AssertionError("strict parser unexpectedly accepted minimal empty abstention")
        empty_readback, empty_view = lossless_output_readback(empty_abstain, parse_output)
        checks["explicit_candidate_inadequacy_empty_abstention_has_no_selected_id"] = (
            empty_readback["minimal_selected_id_valid"] is True and
            empty_readback["selected_id"] is None and
            empty_readback["abstain"] is True and
            classify_response(empty_view)["category"] == "abstain")
        ranked_abstain_payload = {"schema": EXPECTED_PARSER_SCHEMA, "branch": "spatial",
            "ranked_ids": ["u"], "abstain": True, "abstain_reason": "insufficient_visual_evidence"}
        ranked_abstain_raw = json.dumps(ranked_abstain_payload, separators=(",", ":"))
        ranked_abstain = dict(partial, raw_output=ranked_abstain_raw, output_truncated_at_cap=False)
        try:
            parse_output(ranked_abstain_raw, ranked_abstain["request_metadata"]["catalog"])
        except (ValueError, TypeError, KeyError) as exc:
            ranked_abstain["parsed"] = {"status": "invalid_output", "error_type": type(exc).__name__,
                                         "error": str(exc)}
        else:
            raise AssertionError("strict parser unexpectedly accepted minimal ranked abstention")
        ranked_abstain_readback, _ = lossless_output_readback(ranked_abstain, parse_output)
        checks["abstention_keeps_top_rank_diagnostic_without_claiming_a_choice"] = (
            ranked_abstain_readback["minimal_selected_id_valid"] is True and
            ranked_abstain_readback["top_rank_id"] == "u" and
            ranked_abstain_readback["selected_id"] is None and
            ranked_abstain_readback["abstain"] is True)

        duplicate_eval_payload = {**minimal_payload,
            "evaluations": [selected_eval, selected_eval]}
        duplicate_eval_raw = json.dumps(duplicate_eval_payload, separators=(",", ":"))
        duplicate_eval = dict(partial, raw_output=duplicate_eval_raw, output_truncated_at_cap=False)
        try:
            parse_output(duplicate_eval_raw, duplicate_eval["request_metadata"]["catalog"])
        except (ValueError, TypeError, KeyError) as exc:
            duplicate_eval["parsed"] = {"status": "invalid_output", "error_type": type(exc).__name__,
                                         "error": str(exc)}
        else:
            raise AssertionError("strict parser unexpectedly accepted duplicate candidate evaluation")
        duplicate_readback, _ = lossless_output_readback(duplicate_eval, parse_output)
        checks["duplicate_evaluation_makes_selected_semantics_unknown"] = (
            duplicate_readback["minimal_selected_id_valid"] is True and
            duplicate_readback["selected_evaluation_six_states_valid"] is False and
            duplicate_readback["selected_evaluation_status"] == "invalid_evaluation_array" and
            duplicate_readback["selected_evaluation"] is None)

    # Same official evaluator and independent dense scorer as prior STVG audits;
    # synthetic in-memory labels never touch target annotations.
    boxes = np.tile(np.asarray([.1, .1, .5, .5], dtype=np.float64), (4, 1))
    synthetic_truth = {i: [9.6, 7.2, 48.0, 36.0] for i in range(4)}
    metric = score_tube_xyxy(boxes, [0, 1, 2, 3], 96, 72, [0, 4],
                             synthetic_truth, [0, 4], "hc2",
                             official=official, independent=dense_metric)
    checks["official_and_independent_source_grid_metric_agree_on_synthetic"] = (
        metric["status"] == "scored" and metric["official_independent_max_abs_error"] <= 2e-9 and
        all(math.isfinite(metric["metrics"][k]) for k in METRIC_KEYS))
    failed_contracts = [key for key, value in checks.items()
                        if key != "valid_abstain_truncated_invalid_are_distinct" and value is not True]
    require(not failed_contracts,
            "one or more sparse reviewer CPU contracts failed: " + ", ".join(failed_contracts))
    require(checks["valid_abstain_truncated_invalid_are_distinct"],
            "response-state categories collapsed")
    return {"schema": "stvg_sparse_r1_review_cpu_contracts_v1", "status": "pass",
        "synthetic_only": True, "prediction_payloads_opened": False,
        "source_cases_opened": False, "GT_or_annotations_opened": False,
        "RGB_opened": False, "model_calls": 0, "GPU_used": 0,
        "checks": checks}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("contracts", help="synthetic CPU contracts only; no artifact/label reads")
    review = sub.add_parser("review", help="open labels only after complete P0 seal and source/runtime checks")
    review.add_argument("--seal", type=Path, default=SEAL_DEFAULT)
    review.add_argument("--roster", type=Path, default=ROSTER_DEFAULT)
    review.add_argument("--runtime", type=Path, default=RUNTIME_DEFAULT)
    review.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "contracts":
        result = synthetic_contracts()
    else:
        result = review_after_seal(seal_path=args.seal, roster_path=args.roster,
                                   runtime_path=args.runtime, output_path=args.output)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
