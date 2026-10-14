#!/usr/bin/env python3
"""CPU-only readback of Table-1 Spatial-A references under hot-KV IDs.

This reader consumes only sealed no-GT metadata, canonical request frame-map
fields (stream-extracted without materializing query text), and hash-verified
filtered RLE masks for the selected object IDs. It never reads source .pt
files, media pixels, private teacher prose, GT, or model weights.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BASE = ROOT / "artifacts/stvg_table1_spatial_adapter_v1"
HOT = ROOT / "artifacts/stvg_native_hot_teacher_v1/task_equivalence003"
OLD = ROOT / "artifacts/stvg_instance_identity_update_v1"
NATIVE = ROOT / "artifacts/stvg_native_instance_spatial_opd_v1"


class _StreamJSON:
    """Minimal recursive JSON reader that discards unselected scalar strings.

    It keeps only requested key values; non-target strings are consumed and
    discarded. This lets the canonical REQUEST contribute frame-map fields
    without loading its caption/query text into a Python string.
    """

    def __init__(self, path: Path, wanted: set[str]):
        self.f = path.open("r", encoding="utf-8")
        self.wanted = wanted
        self.found: dict[str, Any] = {}

    def close(self):
        self.f.close()

    def _read(self, n: int = 1) -> str:
        return self.f.read(n)

    def _peek(self) -> str:
        pos = self.f.tell()
        char = self._read(1)
        self.f.seek(pos)
        return char

    def _ws(self):
        while self._peek() and self._peek().isspace():
            self._read(1)

    def _string(self, keep: bool) -> str | None:
        if self._read(1) != '"':
            raise ValueError("expected JSON string")
        raw = ['"'] if keep else None
        escaped = False
        while True:
            c = self._read(1)
            if not c:
                raise ValueError("unterminated JSON string")
            if keep:
                raw.append(c)
            if escaped:
                escaped = False
            elif c == "\\":
                escaped = True
            elif c == '"':
                break
        return json.loads("".join(raw)) if keep else None

    def _value(self, keep: bool = False):
        self._ws()
        c = self._peek()
        if c == "{":
            self._read(1)
            obj = {} if keep else None
            self._ws()
            if self._peek() == "}":
                self._read(1)
                return obj
            while True:
                self._ws()
                key = self._string(True)
                self._ws()
                if self._read(1) != ":":
                    raise ValueError("expected colon in JSON object")
                child_keep = keep or key in self.wanted
                value = self._value(child_keep)
                if keep:
                    obj[key] = value
                elif key in self.wanted:
                    if key in self.found and self.found[key] != value:
                        raise ValueError(f"conflicting repeated JSON field {key}")
                    self.found[key] = value
                self._ws()
                sep = self._read(1)
                if sep == "}":
                    return obj
                if sep != ",":
                    raise ValueError("expected comma or object close")
        if c == "[":
            self._read(1)
            arr = [] if keep else None
            self._ws()
            if self._peek() == "]":
                self._read(1)
                return arr
            while True:
                value = self._value(keep)
                if keep:
                    arr.append(value)
                self._ws()
                sep = self._read(1)
                if sep == "]":
                    return arr
                if sep != ",":
                    raise ValueError("expected comma or array close")
        if c == '"':
            return self._string(keep)
        token = [] if keep else None
        while (nxt := self._peek()) and nxt not in ",]} \t\r\n":
            char = self._read(1)
            if keep:
                token.append(char)
        if not token and keep:
            raise ValueError("missing JSON scalar")
        if not keep:
            return None
        raw = "".join(token)
        if raw == "true":
            return True
        if raw == "false":
            return False
        if raw == "null":
            return None
        try:
            return json.loads(raw)
        except Exception as exc:
            raise ValueError("invalid JSON scalar") from exc

    def extract(self) -> dict[str, Any]:
        try:
            self._value(False)
            self._ws()
            if self._read(1):
                raise ValueError("trailing data after JSON document")
            if self.found.keys() < self.wanted:
                raise ValueError(f"missing requested fields: {sorted(self.wanted - self.found.keys())}")
            return {key: self.found[key] for key in sorted(self.wanted)}
        finally:
            self.close()


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _same(a: Any, b: Any) -> bool:
    return json.dumps(a, sort_keys=True, separators=(",", ":"), allow_nan=False) == json.dumps(
        b, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _sha_file_entry(path: Path, expected: dict[str, Any]):
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_bytes = path.stat().st_size
    actual_sha = _sha(path)
    if expected.get("sha256") != actual_sha or (
        expected.get("bytes") is not None and expected.get("bytes") != actual_bytes
    ):
        raise RuntimeError(f"global prediction barrier binding mismatch: {expected.get('path')}")
    return actual_sha


def _close_float(a: Any, b: Any, tol: float = 1e-12) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_close_float(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isfinite(float(a)) and math.isfinite(float(b)) and abs(float(a) - float(b)) <= tol
    return a == b


def _safe_request_projection(path: Path, expected_sha: str) -> dict[str, Any]:
    if _sha(path) != expected_sha:
        raise RuntimeError("canonical REQUEST hash differs from its locked digest")
    # Do not use json.load here: the request also contains private query text.
    return _StreamJSON(path, {"frame_map", "local_frame_count"}).extract()


def run_readback() -> dict[str, Any]:
    from scripts import stvg_table1_spatial_adapter_v1 as adapter

    global_path = HOT / "GLOBAL_BARRIER.json"
    runtime_path = HOT / "RUNTIME.json"
    old_ref_path = OLD / "REFERENCE_LOCK.json"
    old_geo_path = OLD / "TEACHER_GEOMETRY_ONLY_READOUT.json"
    roster_path = NATIVE / "ROSTER_LOCK.json"
    track_barrier_path = NATIVE / "TRACKING_GLOBAL_BARRIER.json"
    global_barrier = _json(global_path)
    runtime = _json(runtime_path)
    old_ref = _json(old_ref_path)
    old_geo = _json(old_geo_path)
    roster = _json(roster_path)
    track_barrier = _json(track_barrier_path)
    if (global_barrier.get("status") != "complete" or global_barrier.get("count") != 16
            or global_barrier.get("GT_read") is not False):
        raise RuntimeError("HOT/task_equivalence003 global barrier is not a complete GT-free 16-call seal")
    if runtime.get("status") != "locked" or runtime.get("GT_read") is not False:
        raise RuntimeError("hot-teacher runtime is not locked and GT-free")
    if any(doc.get("GT_read") is not False for doc in (old_ref, old_geo, roster, track_barrier)):
        raise RuntimeError("one or more prior bindings do not explicitly seal GT_read=false")
    if old_ref.get("count") != 8 or old_geo.get("count") != 8 or roster.get("count") != 8:
        raise RuntimeError("original-eight reference/geometry/roster cardinality mismatch")
    if track_barrier.get("status") != "complete" or track_barrier.get("count") != 8:
        raise RuntimeError("original filtered-RLE tracking barrier is incomplete")

    old_refs = {x["case_id"]: x for x in old_ref["records"]}
    old_geos = {x["case_id"]: x for x in old_geo["records"]}
    roster_rows = {x["case_id"]: x for x in roster["records"]}
    if set(old_refs) != set(old_geos) or set(old_refs) != set(roster_rows):
        raise RuntimeError("original reference, geometry, and source roster do not align")

    barrier_rows = {x["path"]: x for x in global_barrier["rows"]}
    call_rows: dict[str, dict[int, dict[str, Any]]] = {}
    for case_id, rr in roster_rows.items():
        alias = ("HC" + f"{int(case_id.split('_')[-1]) + 1:02d}") if rr["dataset"] == "hc2" else (
            "V" + f"{int(case_id.split('_')[-1]) + 1:02d}"
        )
        call_rows[case_id] = {}
        for repeat in (1, 2):
            rel = f"artifacts/stvg_native_hot_teacher_v1/task_equivalence003/calls/{alias}_{repeat}.json"
            global_row = barrier_rows.get(rel)
            if global_row is None:
                raise RuntimeError(f"global hot barrier does not bind {alias}_{repeat}")
            path = ROOT / rel
            _sha_file_entry(path, global_row)
            receipt = _json(path)
            if receipt.get("GT_read") is not False or receipt.get("raw_query_or_media_embedded") is not False:
                raise RuntimeError(f"{alias}_{repeat} receipt violates no-GT/no-raw-query contract")
            if receipt.get("case_id") != case_id or receipt.get("alias") != alias or receipt.get("repeat_index") != repeat:
                raise RuntimeError(f"{alias}_{repeat} receipt identity mismatch")
            if receipt.get("status") != "complete":
                raise RuntimeError(f"{alias}_{repeat} receipt is not complete")
            call_rows[case_id][repeat] = receipt

    output_cases = []
    input_exact_count = id_exact_count = repeat_id_count = repeat_input_count = repeat_text_count = 0
    anchor_equal_count = weight_changed_count = lock_weight_mismatch_count = 0
    geometry_equal_count = verified_mask_file_count = 0
    id_changed_cases = []
    all_fit_eligibility_equal = True
    for case_id in sorted(old_refs):
        old = old_refs[case_id]
        geo_record = old_geos[case_id]["raw_decoupled"]
        rr = roster_rows[case_id]
        c1, c2 = call_rows[case_id][1], call_rows[case_id][2]
        # The raw selected ID is retained in raw_decoupled even when no
        # presence-supported anchor can be formed (notably vidstg_002).
        # presence_supported.candidate_id is intentionally null in that case.
        old_id = old["raw_decoupled"].get("candidate_id")
        id1 = c1.get("parsed", {}).get("candidate_id")
        id2 = c2.get("parsed", {}).get("candidate_id")
        if id1 != id2:
            raise RuntimeError(f"{case_id}: hot KV repeat selected IDs disagree")
        repeat_id_count += 1
        repeat_input_exact = all(c1.get(key) == c2.get(key) for key in (
            "request_sha256", "query_sha256", "frame_clock_sha256", "source_case_sha256",
            "source_video_sha256", "canonical_video_sha256", "canonical_pixel_sha256",
            "tracking_acceptance_sha256", "marker_video_sha256", "mask_inventory_sha256",
        ))
        if not repeat_input_exact:
            raise RuntimeError(f"{case_id}: two hot calls do not share exact input bindings")
        repeat_input_count += 1
        repeat_text_exact = c2.get("repeat_comparison", {}).get("raw_text_exact_match_previous_repeat")
        if repeat_text_exact is True:
            repeat_text_count += 1
        elif repeat_text_exact is not False:
            raise RuntimeError(f"{case_id}: repeat text comparison is missing or not boolean")

        if c1.get("request_sha256") != rr["request_sha256"] or c1.get("request_sha256") != old["request_sha256"]:
            raise RuntimeError(f"{case_id}: hot call request differs from original locked request")
        if c1.get("source_case_sha256") != rr["source_case_sha256"] or c1.get("source_case_sha256") != old["source_case_sha256"]:
            raise RuntimeError(f"{case_id}: hot call source cache digest differs from original lock")
        if c1.get("canonical_pixel_sha256") != old["canonical_pixel_sha256"]:
            raise RuntimeError(f"{case_id}: hot call pixel binding differs from original reference")
        if c1.get("exact_input_match_old") is True:
            input_exact_count += 1
        else:
            raise RuntimeError(f"{case_id}: hot call does not claim exact original input")
        id_exact = id1 == old_id
        if id_exact:
            id_exact_count += 1
        else:
            id_changed_cases.append(case_id)

        request_path = ROOT / rr["request_path"]
        request = _safe_request_projection(request_path, rr["request_sha256"])
        inventory_path = NATIVE / "TRACKING" / case_id / "MASK_INVENTORY.json"
        accepted_path = inventory_path.parent / "ACCEPTED.json"
        inv_bytes = inventory_path.read_bytes()
        inventory = json.loads(inv_bytes)
        accepted = _json(accepted_path)
        if (accepted.get("GT_read") is not False
                or accepted.get("mask_inventory_sha256") != hashlib.sha256(inv_bytes).hexdigest()
                or c1.get("mask_inventory_sha256") != hashlib.sha256(inv_bytes).hexdigest()):
            raise RuntimeError(f"{case_id}: accepted tracking or hot calls do not bind the original mask inventory")
        accepted_global_row = next((x for x in track_barrier["files"] if x.get("case_id") == case_id), None)
        if accepted_global_row is None or accepted_global_row.get("sha256") != _sha(accepted_path):
            raise RuntimeError(f"{case_id}: tracking barrier does not bind its accepted receipt")

        # Original geometry contains the exact Frozen fallback rows. It is safe
        # to reuse these as F because (a) same-ID frames are overwritten from
        # the selected RLE and (b) on the sole changed-ID case, old geometry is
        # verified to be entirely Frozen below.
        if not id_exact and any(bool(v) for v in geo_record["teacher_mask_present"]):
            raise RuntimeError(f"{case_id}: cannot use prior geometry as Frozen boxes for a changed selected ID")
        frozen_projection = geo_record["boxes_xyxy_normalized"]
        first_rle = ROOT / inventory["files"][0]["path"]
        first_json = json.loads(first_rle.read_bytes())
        any_obj = next(iter(first_json.get("objects", {}).values()), None)
        if not isinstance(any_obj, dict) or not isinstance(any_obj.get("rle"), dict):
            raise RuntimeError(f"{case_id}: filtered RLE inventory has no shape metadata")
        height, width = map(int, any_obj["rle"]["size"])
        source_projection = {
            "case_id": case_id, "dataset": rr["dataset"],
            "frame_ids": geo_record["frame_ids_physical"],
            "width": width, "height": height,
            "native_boxes_xyxy_normalized": frozen_projection,
        }
        result = adapter.build_reference(
            source_projection,
            old["qa_interval_frame_bounds_half_open"],
            request,
            id1,
            inventory,
            root=ROOT,
        )
        verified_mask_file_count += result["rle_inventory_files_sha_verified"]
        old_anchors = old["presence_supported"]["anchors"]
        new_anchors = result["anchors"]
        if len(new_anchors) != len(old_anchors):
            raise RuntimeError(f"{case_id}: selected-ID switch altered the supported-anchor count")
        case_anchor_comparison = []
        for old_anchor, new_anchor in zip(old_anchors, new_anchors):
            geometry_fields = (
                "slot", "position", "physical_frame_id", "local_frame_index",
                "box_xyxy_normalized", "box_cxcywh", "reliability",
                "candidate_id", "coordinate_convention",
            )
            fields_equal = {
                key: _same(old_anchor.get(key), new_anchor.get(key))
                for key in geometry_fields
            }
            if not all(fields_equal.values()):
                raise RuntimeError(f"{case_id}: anchor geometry/evidence changed at slot {old_anchor.get('slot')}")
            anchor_equal_count += 1
            old_lock_weight = float(old_anchor["weight"])
            old_effective_weight = float(old_anchor["reliability"]["presence_fraction"])
            new_weight = float(new_anchor["weight"])
            expected_presence = float(new_anchor["reliability"]["presence_fraction"])
            if not _close_float(new_weight, expected_presence):
                raise RuntimeError(f"{case_id}: new fit weight is not exact presence fraction")
            if not _close_float(old_effective_weight, new_weight):
                weight_changed_count += 1
            if not _close_float(old_lock_weight, new_weight):
                lock_weight_mismatch_count += 1
            case_anchor_comparison.append({
                "slot": int(old_anchor["slot"]),
                "position": int(old_anchor["position"]),
                "physical_frame_id": int(old_anchor["physical_frame_id"]),
                "geometry_evidence_exact": True,
                "old_reference_lock_weight": old_lock_weight,
                "old_effective_A_fit_weight": old_effective_weight,
                "new_presence_fraction_weight": new_weight,
                "effective_A_weight_exact": _close_float(old_effective_weight, new_weight),
                "serialized_lock_weight_equals_fit_weight": _close_float(old_lock_weight, new_weight),
            })

        old_geo_fields = (
            "status", "candidate_id", "interval_frame_bounds_half_open", "frame_ids_physical",
            "boxes_xyxy_normalized", "teacher_boxes_xyxy_normalized", "teacher_mask_present",
            "teacher_mask_missing", "fallback_used", "fallback_reason", "box_source",
            "interpolation_used", "nearest_frame_replacement_used",
        )
        new_geo = result["teacher_geometry"]
        geo_fields_exact = {key: _same(new_geo.get(key), geo_record.get(key)) for key in old_geo_fields}
        if not all(geo_fields_exact.values()):
            raise RuntimeError(f"{case_id}: full-grid TeacherGeometry differs from the sealed original A reference")
        geometry_equal_count += len(geo_record["frame_ids_physical"])

        old_fit_eligible = bool(old["presence_supported"].get("eligible_for_update"))
        new_fit_eligible = not bool(result["no_update"])
        all_fit_eligibility_equal &= old_fit_eligible == new_fit_eligible
        output_cases.append({
            "case_id": case_id,
            "dataset": rr["dataset"],
            "old_selected_id": old_id,
            "old_presence_supported_candidate_id": old["presence_supported"].get("candidate_id"),
            "old_geometry_candidate_id": geo_record.get("candidate_id"),
            "hot_selected_id": id1,
            "hot_parsed_status": c1.get("parsed", {}).get("status"),
            "hot_valid_uncapped_output": c1.get("valid_uncapped_output"),
            "hot_id_equals_old": id_exact,
            "repeat_selected_id_equal": True,
            "repeat_input_bindings_exact": repeat_input_exact,
            "repeat_raw_text_exact": repeat_text_exact,
            "call_input_exact_old": c1.get("exact_input_match_old"),
            "request_sha256": c1["request_sha256"],
            "query_sha256": c1["query_sha256"],
            "frame_clock_sha256": c1["frame_clock_sha256"],
            "source_case_sha256": c1["source_case_sha256"],
            "source_video_sha256": c1["source_video_sha256"],
            "canonical_video_sha256": c1["canonical_video_sha256"],
            "canonical_pixel_sha256": c1["canonical_pixel_sha256"],
            "tracking_acceptance_sha256": c1["tracking_acceptance_sha256"],
            "marker_video_sha256": c1["marker_video_sha256"],
            "mask_inventory_sha256": c1["mask_inventory_sha256"],
            "mask_inventory_file_sha256": hashlib.sha256(inv_bytes).hexdigest(),
            "tracking_acceptance_file_sha256": _sha(accepted_path),
            "selected_RLE_files_sha_verified": result["rle_inventory_files_sha_verified"],
            "anchor_count_old": len(old_anchors),
            "anchor_count_hot": len(new_anchors),
            "anchor_comparisons": case_anchor_comparison,
            "teacher_geometry_grid_count": len(geo_record["frame_ids_physical"]),
            "teacher_geometry_mask_boxes_old": sum(bool(x) for x in geo_record["teacher_mask_present"]),
            "teacher_geometry_frozen_fallbacks_old": sum(bool(x) for x in geo_record["fallback_used"]),
            "teacher_geometry_exact": True,
            "old_fit_eligible": old_fit_eligible,
            "hot_fit_eligible_under_four_anchor_rule": new_fit_eligible,
            "hot_no_update_reason": result["no_update_reason"],
            "hot_QA_selected_masks": result["qa_selected_mask_count"],
            "hot_source_grid_selected_masks": result["source_grid_selected_mask_count"],
            "student_prediction_or_fit_compared": False,
        })

    if len(output_cases) != 8 or input_exact_count != 8 or id_exact_count != 7:
        raise RuntimeError("unexpected original-input or selected-ID equality count")
    if id_changed_cases != ["vidstg_002"]:
        raise RuntimeError(f"unexpected selected-ID change cases: {id_changed_cases}")
    if not all_fit_eligibility_equal:
        # The new adapter has an explicitly different frozen four-anchor
        # no-update threshold; preserve this difference in the receipt rather
        # than treating it as a reference-geometry failure.
        eligibility_changes = [
            row["case_id"] for row in output_cases
            if row["old_fit_eligible"] != row["hot_fit_eligible_under_four_anchor_rule"]
        ]
    else:
        eligibility_changes = []

    source_paths = [
        "scripts/review_stvg_table1_hotkv_reference_v1.py",
        "scripts/stvg_table1_spatial_adapter_v1.py",
        "scripts/prepare_stvg_instance_decoupled_references_v1.py",
        "scripts/prepare_stvg_short_event_tracklets_v1.py",
        "scripts/run_stvg_table1_source_v1.py",
        "scripts/stvg_instance_validated_fit_v1.py",
    ]
    source_hashes = {}
    for rel in source_paths:
        p = ROOT / rel
        if p.is_file():
            source_hashes[rel] = _sha(p)

    return {
        "schema": "stvg-table1-hotkv-spatial-reference-cpu-readback-v1",
        "status": "pass_with_explicit_method_differences",
        "scope": "eight original development cases; selected-ID spatial reference reconstruction only",
        "input_call_count": 16,
        "case_count": 8,
        "original_input_exact_cases": input_exact_count,
        "repeat_selected_ID_equal_cases": repeat_id_count,
        "repeat_input_binding_exact_cases": repeat_input_count,
        "repeat_raw_text_exact_cases": repeat_text_count,
        "old_ID_exact_call_count_of_16": sum(
            bool(barrier_rows[row["path"]]["exact_id_old"]) for row in global_barrier["rows"]
        ),
        "old_raw_text_exact_call_count_of_16": sum(
            bool(barrier_rows[row["path"]]["exact_text_old"]) for row in global_barrier["rows"]
        ),
        "old_ID_exact_cases": id_exact_count,
        "selected_ID_changed_cases": id_changed_cases,
        "selected_ID_changed_call_count": 2,
        "hot_valid_uncapped_parsed_cases": sum(bool(row["hot_valid_uncapped_output"]) for row in output_cases),
        "exact_anchor_geometry_evidence_rows": anchor_equal_count,
        "exact_full_grid_teacher_geometry_rows": geometry_equal_count,
        "selected_RLE_files_sha_verified": verified_mask_file_count,
        "old_effective_A_new_fit_weight_equal_anchor_rows": anchor_equal_count - weight_changed_count,
        "effective_fit_weight_difference_anchor_rows": weight_changed_count,
        "old_lock_serialized_weight_mismatch_anchor_rows": lock_weight_mismatch_count,
        "weight_semantics": "the old A student overwrites the legacy lock weight with reliability.presence_fraction before fitting; new Spatial-A uses that same effective fit weight",
        "reference_anchor_and_teacher_geometry_equal": True,
        "student_outputs_or_fit_equal": None,
        "student_outputs_or_fit_comparison_reason": "This receipt compares pre-fit references and TeacherGeometry only; no new student fits or student readouts were run.",
        "old_vs_new_fit_eligibility_equal": all_fit_eligibility_equal,
        "fit_eligibility_changed_cases": eligibility_changes,
        "eligibility_change_definition": "four slots are potential QA centers; both old A and new Spatial-A update when at least one fixed-center anchor is supported",
        "cases": output_cases,
        "bindings": {
            "hot_global_barrier_sha256": _sha(global_path),
            "hot_runtime_sha256": _sha(runtime_path),
            "old_reference_lock_sha256": _sha(old_ref_path),
            "old_teacher_geometry_sha256": _sha(old_geo_path),
            "source_roster_sha256": _sha(roster_path),
            "tracking_global_barrier_sha256": _sha(track_barrier_path),
            "source_code_sha256": source_hashes,
            "hot_call_receipt_sha256": {
                row["path"]: row["sha256"] for row in global_barrier["rows"]
            },
        },
        "source_case_pickles_opened": False,
        "canonical_query_text_materialized": False,
        "private_teacher_prose_opened": False,
        "video_or_rgb_read": False,
        "GT_read": False,
        "model_or_GPU_used": False,
    }


def _synthetic_stream_contract() -> dict[str, Any]:
    import tempfile

    sample = '{"private_query":"NEVER_EXPORT_ME", "input":{"frame_map":[{"local_frame_index":0,"physical_frame_id":7}],"local_frame_count":1,"caption":"SECRET"},"frame_map":[{"local_frame_index":0,"physical_frame_id":7}],"local_frame_count":1}'
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "request.json"
        p.write_text(sample, encoding="utf-8")
        got = _StreamJSON(p, {"frame_map", "local_frame_count"}).extract()
    assert got == {"frame_map": [{"local_frame_index": 0, "physical_frame_id": 7}],
                   "local_frame_count": 1}
    assert "SECRET" not in json.dumps(got) and "NEVER_EXPORT_ME" not in json.dumps(got)
    return {"stream_selects_only_frame_map_fields": "pass", "query_string_not_returned": "pass"}


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--contracts", action="store_true")
    parser.add_argument("--review-hotkv", action="store_true")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    if args.contracts:
        print(json.dumps({"schema": "stvg-table1-hotkv-reader-contracts-v1",
                          "status": "pass", "synthetic": _synthetic_stream_contract(),
                          "GT_read": False, "CUDA_initialized": False},
                         sort_keys=True, separators=(",", ":")))
        return
    if not args.review_hotkv:
        parser.error("use --contracts or --review-hotkv")
    result = run_readback()
    if args.write:
        out = BASE / "HOTKV003_REFERENCE_CPU_READBACK002.json"
        if out.exists():
            raise FileExistsError(out)
        out.write_text(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
                       encoding="utf-8")
        print(json.dumps({"status": result["status"], "path": str(out.relative_to(ROOT)),
                          "sha256": _sha(out), "case_count": result["case_count"],
                          "anchor_rows": result["exact_anchor_geometry_evidence_rows"],
                          "geometry_rows": result["exact_full_grid_teacher_geometry_rows"]},
                         sort_keys=True, separators=(",", ":")))
    else:
        print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
