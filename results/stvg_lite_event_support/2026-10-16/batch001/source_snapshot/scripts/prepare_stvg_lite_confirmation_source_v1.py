"""Prepare a fixed, metadata-only 32+32 source-confirmation roster.

This preparer reads only the existing opaque source registries, official
one-query-video registries, current Coordinate128 identity lock, and cache
receipts/bindings. It never reads a PLAN row, annotation, GT, prediction value,
REQUEST.json, or query/caption body. Cache selection is strictly downstream of
the fixed identity selection and cannot affect membership or order.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
CONFIRMATION = ROOT / "artifacts/stvg_lite_event_support_v1/confirmation"
FULL_ROSTER = ROOT / "artifacts/stvg_final_table1_v1/FULL/FULL_SOURCE_ROSTER_LOCK.json"
CURRENT_DEV_ROSTER = ROOT / "artifacts/stvg_soft_ep_coordinate128_v1/ROSTER_LOCK.json"
SOURCE_RUNTIME = ROOT / "artifacts/stvg_final_table1_v1/SOURCE_RUNTIME.json"
PREFLIGHT_ROSTER = ROOT / "artifacts/stvg_final_table1_v1/PREFLIGHT_ROSTER_LOCK.json"
ONE_QUERY_ROSTERS = {
    "hc2": ROOT / "artifacts/decota_paper_experiments_v1/hc2/ONE_QUERY_VIDEO_ROSTER.json",
    "vidstg": ROOT / "artifacts/decota_paper_experiments_v1/vidstg/ONE_QUERY_VIDEO_ROSTER.json",
}
SOURCE_CASES = ROOT / "artifacts/stvg_final_table1_v1/source_cases"
TRACKING_CASES = ROOT / "artifacts/stvg_final_table1_v1/tracking_cases"
TEACHER_CASES = ROOT / "artifacts/stvg_final_table1_v1/teacher_cases"
STUDENT_CASES = ROOT / "artifacts/stvg_final_table1_v1/student_cases"
EVIDENCE_SOURCE_CASES = ROOT / "artifacts/stvg_evidence_constrained_v1/FULL/source_cases"
COORDINATE_SOURCE_CASES = ROOT / "artifacts/stvg_soft_ep_coordinate128_v1/source_cases"
OUT_INVENTORY = CONFIRMATION / "ROSTER_INVENTORY.json"
OUT_PROPOSAL = CONFIRMATION / "ROSTER_PROPOSAL.json"
OUT_LOCK = CONFIRMATION / "ROSTER_LOCK.json"

DOMAINS = ("hc2", "vidstg")
TARGET_BINDING = {
    "hc2": {
        "official_split": "validation",
        "source_model": "vidstg",
        "source_direction": "vid_to_hc1",
        "checkpoint_path": "checkpoints/TASTVG_VidSTG.pth",
        "checkpoint_sha256": "5ab12c86363ef0ce0ee006c00fd11c6b659c3a9b2cb01a4f2c613efe22a2aa83",
        "alias_prefix": "HC",
    },
    "vidstg": {
        "official_split": "test",
        "source_model": "hcstvg2",
        "source_direction": "hc2_to_vid",
        "checkpoint_path": "checkpoints/TASTVG_HCSTVG2.pth",
        "checkpoint_sha256": "47d8f15841cd57e7bbf5a10e8bf23b1054d23b753e0becbd38a07f3dd60d5036",
        "alias_prefix": "V",
    },
}
SAFE_ROW_FIELDS = (
    "dataset", "query_ordinal", "parent_sha256", "query_key_sha256",
    "safe_record_sha256", "plan_path", "plan_sha256", "plan_row_pointer",
    "subject_path", "subject_sha256", "video_sha256", "frame_ids", "fps",
    "width", "height", "frame_count", "duration", "video_bounds", "query_type",
)
FORBIDDEN_TEXT_KEYS = {"query_text", "caption", "sentence", "query", "text"}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path.relative_to(ROOT)}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to replace existing roster output: {path.relative_to(ROOT)}")
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")
    os.replace(temp, path)


def _reject_text_fields(row: Mapping[str, Any], source: str) -> None:
    found = FORBIDDEN_TEXT_KEYS.intersection(row)
    if found:
        raise ValueError(f"Unexpected text field in opaque registry {source}: {sorted(found)}")


def _cache_file_summary(directory: Path, *, suffix: str = "*.receipt.json",
                        payload_suffix: str = ".pt") -> dict[str, Any]:
    receipts = sorted(directory.glob(suffix)) if directory.is_dir() else []
    payload_count = 0
    size_matches = 0
    for receipt_path in receipts:
        receipt = _read_json(receipt_path)
        _reject_text_fields(receipt, receipt_path.name)
        payload_path = receipt_path.with_name(receipt_path.name.replace(".receipt.json", payload_suffix))
        if not payload_path.is_file():
            continue
        payload_count += 1
        try:
            if payload_path.stat().st_size == int(receipt.get("bytes", -1)):
                size_matches += 1
        except (TypeError, ValueError):
            pass
    return {
        "directory": str(directory.relative_to(ROOT)),
        "receipt_count": len(receipts),
        "payload_file_count": payload_count,
        "payload_size_matches_receipt_count": size_matches,
        "payload_bytes_hashed_or_deserialized": False,
    }


def _opaque_metadata_sources() -> dict[str, Any]:
    source_files = {
        "full_source_roster": FULL_ROSTER,
        "coordinate128_current_roster": CURRENT_DEV_ROSTER,
        "source_runtime": SOURCE_RUNTIME,
        "preflight_roster": PREFLIGHT_ROSTER,
        "official_hc2_one_query_video_roster": ONE_QUERY_ROSTERS["hc2"],
        "official_vidstg_one_query_video_roster": ONE_QUERY_ROSTERS["vidstg"],
    }
    out = {}
    for name, path in source_files.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        out[name] = {"path": str(path.relative_to(ROOT)), "sha256": _sha256_file(path)}
    return out


def _collect_cache_indexes(selected: list[dict[str, Any]],
                           preflight_records: list[dict[str, Any]],
                           source_runtime_sha256: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected_by_key = {
        (row["dataset"], int(row["query_ordinal"]), row["query_key_sha256"]): row
        for row in selected
    }
    selected_by_qhash = {row["query_key_sha256"]: row for row in selected}
    preflight_by_alias = {row["probe_alias"]: row for row in preflight_records}

    source_cache_candidates: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    final_source_identity_hits = 0
    final_source_receipts = sorted(SOURCE_CASES.glob("*.receipt.json")) if SOURCE_CASES.is_dir() else []
    for receipt_path in final_source_receipts:
        receipt = _read_json(receipt_path)
        _reject_text_fields(receipt, receipt_path.name)
        old = preflight_by_alias.get(receipt.get("case_id"))
        if old is None or receipt.get("query_key_sha256") != old.get("query_key_sha256"):
            continue
        key = (old["dataset"], int(old["query_ordinal"]), old["query_key_sha256"])
        if key not in selected_by_key:
            continue
        final_source_identity_hits += 1
        payload_path = receipt_path.with_name(receipt_path.name.replace(".receipt.json", ".pt"))
        payload_exists = payload_path.is_file()
        payload_bytes_match = payload_exists and payload_path.stat().st_size == int(receipt.get("bytes", -1))
        source_cache_candidates.setdefault(key, []).append({
            "source_registry": "stvg_final_table1_v1/preflight_source_cases",
            "payload_path": str(payload_path.relative_to(ROOT)),
            "receipt_path": str(receipt_path.relative_to(ROOT)),
            "receipt_identity_match": True,
            "payload_file_present": payload_exists,
            "payload_size_matches_receipt": bool(payload_bytes_match),
            "receipt_sha256": receipt.get("sha256"),
            "payload_sha_recomputed": False,
            "source_runtime_sha256": receipt.get("source_runtime_sha256"),
            "source_runtime_matches_locked_preflight": receipt.get("source_runtime_sha256") == source_runtime_sha256,
            "source_case_sha256": receipt.get("source_case_sha256"),
            "old_case_alias": receipt.get("case_id"),
        })

    evidence_receipts = sorted(EVIDENCE_SOURCE_CASES.glob("*.receipt.json")) if EVIDENCE_SOURCE_CASES.is_dir() else []
    evidence_identity_hits = 0
    for receipt_path in evidence_receipts:
        receipt = _read_json(receipt_path)
        _reject_text_fields(receipt, receipt_path.name)
        key = (receipt.get("dataset"), receipt.get("query_ordinal"), receipt.get("query_key_sha256"))
        if key not in selected_by_key:
            continue
        evidence_identity_hits += 1
        payload_path = receipt_path.with_name(receipt_path.name.replace(".receipt.json", ".pt.gz"))
        source_cache_candidates.setdefault(key, []).append({
            "source_registry": "stvg_evidence_constrained_v1/FULL/source_cases",
            "payload_path": str(payload_path.relative_to(ROOT)),
            "receipt_path": str(receipt_path.relative_to(ROOT)),
            "receipt_identity_match": True,
            "payload_file_present": payload_path.is_file(),
            "payload_size_matches_receipt": payload_path.is_file() and payload_path.stat().st_size == int(receipt.get("bytes", -1)),
            "receipt_sha256": receipt.get("sha256"),
            "payload_sha_recomputed": False,
            "source_runtime_sha256": receipt.get("source_runtime_sha256"),
            "source_runtime_matches_locked_preflight": receipt.get("source_runtime_sha256") == source_runtime_sha256,
            "source_case_sha256": receipt.get("source_case_sha256"),
            "old_case_alias": receipt.get("probe_alias"),
        })

    tracking_by_qhash: dict[str, list[dict[str, Any]]] = {}
    tracking_binding_files = sorted(TRACKING_CASES.glob("*/TRACKING_BINDING.json")) if TRACKING_CASES.is_dir() else []
    tracking_identity_hits = 0
    for binding_path in tracking_binding_files:
        binding = _read_json(binding_path)
        _reject_text_fields(binding, binding_path.name)
        qhash = binding.get("query_key_sha256")
        row = selected_by_qhash.get(qhash)
        if row is None or binding.get("dataset") != row["dataset"]:
            continue
        bound_video = binding.get("source_video_sha256", binding.get("video_sha256"))
        if bound_video != row["video_sha256"]:
            continue
        tracking_identity_hits += 1
        tracking_by_qhash.setdefault(qhash, []).append({
            "binding_path": str(binding_path.relative_to(ROOT)),
            "query_key_sha256_match": True,
            "dataset_match": True,
            "video_sha256_match": True,
            "tracking_payload_present": any(binding_path.parent.glob("*.pt")) or any(binding_path.parent.glob("*.pt.gz")),
            "tracking_receipt_present": any(binding_path.parent.glob("*.receipt.json")),
        })

    teacher_aliases = {p.name.removesuffix(".receipt.json") for p in TEACHER_CASES.glob("*.receipt.json")} if TEACHER_CASES.is_dir() else set()
    student_aliases = {p.name.removesuffix(".receipt.json") for p in STUDENT_CASES.glob("*.receipt.json")} if STUDENT_CASES.is_dir() else set()
    old_alias_by_qhash = {row["query_key_sha256"]: row["probe_alias"] for row in preflight_records}
    rows_with_cache: list[dict[str, Any]] = []
    for row in selected:
        key = (row["dataset"], int(row["query_ordinal"]), row["query_key_sha256"])
        old_alias = old_alias_by_qhash.get(row["query_key_sha256"])
        source_hits = source_cache_candidates.get(key, [])
        rows_with_cache.append({
            **row,
            "cache_availability": {
                "exact_identity_source_caches": source_hits,
                "exact_identity_source_cache_count": len(source_hits),
                "tracking_binding_metadata": tracking_by_qhash.get(row["query_key_sha256"], []),
                "tracking_payload_cache_count": sum(
                    bool(item["tracking_payload_present"]) for item in tracking_by_qhash.get(row["query_key_sha256"], [])
                ),
                "matching_teacher_receipt_present": old_alias in teacher_aliases if old_alias else False,
                "matching_student_receipt_present": old_alias in student_aliases if old_alias else False,
                "old_preflight_case_alias_for_same_query": old_alias,
                "raw_video_path_presence": "not_checked; video bytes are resolved by the frozen capture adapter from the PLAN row at execution",
            },
        })

    cache_inventory = {
        "selection_was_frozen_before_cache_inspection": True,
        "payload_policy": "Only receipt/binding metadata, file existence, and file size were inspected; source/teacher/student payloads were not deserialized and payload SHA was not recomputed.",
        "final_table1_preflight_source": _cache_file_summary(SOURCE_CASES),
        "final_table1_tracking": {
            "directory": str(TRACKING_CASES.relative_to(ROOT)),
            "tracking_binding_metadata_count": len(tracking_binding_files),
            "tracking_payload_file_count": sum(1 for _ in TRACKING_CASES.glob("*/*.pt")) + sum(1 for _ in TRACKING_CASES.glob("*/*.pt.gz")) if TRACKING_CASES.is_dir() else 0,
            "tracking_receipt_file_count": sum(1 for _ in TRACKING_CASES.glob("*/*.receipt.json")) if TRACKING_CASES.is_dir() else 0,
        },
        "final_table1_teacher": _cache_file_summary(TEACHER_CASES),
        "final_table1_student": _cache_file_summary(STUDENT_CASES),
        "evidence_constrained_full_source": _cache_file_summary(EVIDENCE_SOURCE_CASES, payload_suffix=".pt.gz"),
        "full_table1_source_cases": {
            "directory": "artifacts/stvg_final_table1_v1/FULL/source_cases",
            "directory_present": (ROOT / "artifacts/stvg_final_table1_v1/FULL/source_cases").is_dir(),
        },
        "coordinate128_current_development_source_cases": {
            "directory": str(COORDINATE_SOURCE_CASES.relative_to(ROOT)),
            "directory_present": COORDINATE_SOURCE_CASES.is_dir(),
            "use_for_confirmation": False,
            "reason": "Current Coordinate128 parent identities are excluded from the locked roster.",
            "payload_files_opened": False,
        },
        "selected_identity_match_counts": {
            "preflight_source_receipt_matches": final_source_identity_hits,
            "evidence_constrained_source_receipt_matches": evidence_identity_hits,
            "tracking_binding_matches": tracking_identity_hits,
            "teacher_receipt_matches_via_same_query_preflight_alias": sum(
                item["cache_availability"]["matching_teacher_receipt_present"] for item in rows_with_cache
            ),
            "student_receipt_matches_via_same_query_preflight_alias": sum(
                item["cache_availability"]["matching_student_receipt_present"] for item in rows_with_cache
            ),
        },
        "matching_cache_choice_affects_roster": False,
    }
    return rows_with_cache, cache_inventory


def prepare() -> tuple[dict[str, Any], dict[str, Any]]:
    full = _read_json(FULL_ROSTER)
    current = _read_json(CURRENT_DEV_ROSTER)
    runtime = _read_json(SOURCE_RUNTIME)
    preflight = _read_json(PREFLIGHT_ROSTER)
    _reject_text_fields(full, FULL_ROSTER.name)
    _reject_text_fields(current, CURRENT_DEV_ROSTER.name)
    if full.get("GT_read") is not False or full.get("historical_exposure") is not True:
        raise ValueError("The full source registry must remain GT-free and disclose historical exposure")
    if full.get("selection_uses_GT_or_scores") is not False:
        raise ValueError("Full source registry does not explicitly rule out GT/score selection")
    if current.get("GT_read") is not False or current.get("count") != 128:
        raise ValueError("Current Coordinate128 identity lock is not the expected GT-free 128-row roster")
    if preflight.get("GT_read") is not False:
        raise ValueError("Prior source preflight roster is not marked GT-free")
    records = full.get("records")
    dev_records = current.get("records")
    if not isinstance(records, list) or not isinstance(dev_records, list):
        raise ValueError("Source registry records must be lists")
    for row in records:
        _reject_text_fields(row, FULL_ROSTER.name)
        missing = [field for field in SAFE_ROW_FIELDS if field not in row]
        if missing:
            raise ValueError(f"Source registry row lacks required opaque fields: {missing}")
    for row in dev_records:
        _reject_text_fields(row, CURRENT_DEV_ROSTER.name)

    source_runtime_sha = _sha256_file(SOURCE_RUNTIME)
    excluded_parents = {
        domain: {row["parent_sha256"] for row in dev_records if row.get("dataset") == domain}
        for domain in DOMAINS
    }
    if any(len(excluded_parents[domain]) == 0 for domain in DOMAINS):
        raise ValueError("Current development roster lacks a domain parent exclusion set")

    fixed_records: list[dict[str, Any]] = []
    domain_inventory: dict[str, Any] = {}
    for domain in DOMAINS:
        video_roster = _read_json(ONE_QUERY_ROSTERS[domain])
        if video_roster.get("GT_used") is not False or video_roster.get("selection_before_predictions") is not True:
            raise ValueError(f"Official one-query-video roster is not prediction-blind and GT-free: {domain}")
        if video_roster.get("official_split") != TARGET_BINDING[domain]["official_split"]:
            raise ValueError(f"Official split binding changed for {domain}")
        ordinals = video_roster.get("parent_ordinals")
        if not isinstance(ordinals, list) or len(ordinals) != len(set(ordinals)):
            raise ValueError(f"Official one-query-video ordinals are malformed for {domain}")
        ordinal_set = set(ordinals)
        official_rows = [row for row in records
                         if row.get("dataset") == domain and row.get("query_ordinal") in ordinal_set]
        if len(official_rows) != len(ordinals) or len({row["query_ordinal"] for row in official_rows}) != len(ordinals):
            raise ValueError(f"Full source registry does not exactly cover official video ordinals for {domain}")
        parents_before = {row["parent_sha256"] for row in official_rows}
        eligible = [row for row in official_rows if row["parent_sha256"] not in excluded_parents[domain]]
        groups: dict[str, list[dict[str, Any]]] = {}
        for row in eligible:
            groups.setdefault(row["parent_sha256"], []).append(row)
        if len(groups) < 32:
            raise ValueError(f"Fewer than 32 eligible source parents in {domain}")

        # Parent SHA order is the fixed source hash order. Query-key SHA order
        # is applied only inside each selected parent. Neither cache state nor
        # GT/outcome data participate in either ordering decision.
        ordered_parents = sorted(groups)
        chosen_parents = ordered_parents[:32]
        domain_records: list[dict[str, Any]] = []
        for offset, parent_sha in enumerate(chosen_parents, start=1):
            children = sorted(groups[parent_sha], key=lambda row: (
                row["query_key_sha256"], row["video_sha256"], int(row["query_ordinal"])
            ))
            row = children[0]
            alias = f"{TARGET_BINDING[domain]['alias_prefix']}{offset:02d}"
            record = {field: row[field] for field in SAFE_ROW_FIELDS}
            record.update({
                "probe_alias": alias,
                "source_case_path": str((CONFIRMATION / "source_cases" / f"{alias}.pt").relative_to(ROOT)),
                "source_registry_probe_alias": row.get("probe_alias"),
                "parent_hash_rank": offset,
                "selected_query_rank_within_parent": 1,
                "eligible_official_query_count_for_parent": len(children),
                "coordinate128_parent_excluded": False,
                "source_model": TARGET_BINDING[domain]["source_model"],
                "source_direction": TARGET_BINDING[domain]["source_direction"],
                "source_checkpoint_path": TARGET_BINDING[domain]["checkpoint_path"],
                "source_checkpoint_sha256": TARGET_BINDING[domain]["checkpoint_sha256"],
                "official_split": TARGET_BINDING[domain]["official_split"],
                "subject_metadata_file_present_stat_only": (ROOT / row["subject_path"]).is_file(),
                "plan_file_present_stat_only": (ROOT / row["plan_path"]).is_file(),
            })
            if record["parent_sha256"] in excluded_parents[domain]:
                raise AssertionError("Selected parent overlaps current Coordinate128 development")
            domain_records.append(record)
        if len({row["parent_sha256"] for row in domain_records}) != 32:
            raise AssertionError("Selected rows are not 32 independent parent identities")
        if len({row["video_sha256"] for row in domain_records}) != 32:
            raise AssertionError("Selected rows do not have 32 distinct clip-video identities")
        fixed_records.extend(domain_records)
        domain_inventory[domain] = {
            "official_one_query_video_registry_path": str(ONE_QUERY_ROSTERS[domain].relative_to(ROOT)),
            "official_split": TARGET_BINDING[domain]["official_split"],
            "official_registry_query_or_video_ordinals": len(ordinals),
            "official_registry_unique_clip_videos": len({row["video_sha256"] for row in official_rows}),
            "official_registry_unique_source_parents": len(parents_before),
            "official_rows_matching_full_source_registry": len(official_rows),
            "coordinate128_excluded_parent_count": len(excluded_parents[domain]),
            "coordinate128_parent_overlap_with_official_registry": len(parents_before & excluded_parents[domain]),
            "eligible_official_rows_after_parent_exclusion": len(eligible),
            "eligible_parent_count_after_parent_exclusion": len(groups),
            "eligible_parents_with_multiple_official_clip_queries": sum(len(v) > 1 for v in groups.values()),
            "selected_parent_count": len(domain_records),
            "selected_query_count": len(domain_records),
            "selected_distinct_clip_video_count": len({row["video_sha256"] for row in domain_records}),
            "selection_order": "lexicographic ascending parent_sha256; within selected parent lexicographic ascending query_key_sha256, then video_sha256, then query_ordinal; first child only",
            "target_source_direction": {
                "target_dataset": domain,
                "source_model": TARGET_BINDING[domain]["source_model"],
                "direction": TARGET_BINDING[domain]["source_direction"],
                "checkpoint_path": TARGET_BINDING[domain]["checkpoint_path"],
                "checkpoint_sha256_from_locked_source_runtime": TARGET_BINDING[domain]["checkpoint_sha256"],
                "checkpoint_file_present_stat_only": (ROOT / TARGET_BINDING[domain]["checkpoint_path"]).is_file(),
            },
            "identity_semantics": {
                "parent_sha256": "independent source-parent identity; the unit used for exclusion and the one-query-per-parent cap",
                "video_sha256": "clip-video identity for the selected row; it remains separate from parent_sha256, especially for HC2 where one parent can contain multiple clip videos",
            },
        }

    if len(fixed_records) != 64:
        raise AssertionError("The fixed confirmation proposal must contain 64 rows")
    preflight_records = preflight.get("records")
    if not isinstance(preflight_records, list):
        raise ValueError("Prior 64-query preflight registry has no records")
    fixed_records, cache_inventory = _collect_cache_indexes(fixed_records, preflight_records, source_runtime_sha)
    metadata_sources = _opaque_metadata_sources()
    proposal = {
        "schema": "stvg-lite-event-support-confirmation-roster-proposal-v1",
        "status": "prepared_for_root_review",
        "selection_unit": "one query per independent parent source",
        "count": 64,
        "domain_counts": {domain: sum(row["dataset"] == domain for row in fixed_records) for domain in DOMAINS},
        "selection": {
            "rule": "Start from official ONE_QUERY_VIDEO_ROSTER parent_ordinals and the opaque full source registry; exclude every current Coordinate128 parent_sha256 in the same domain; sort eligible parent_sha256 ascending and take 32/domain; inside each selected parent sort query_key_sha256 ascending, then video_sha256 and query_ordinal, and take the first query.",
            "uses_ground_truth": False,
            "uses_scores_or_predictions": False,
            "uses_cache_availability": False,
            "source_parent_independent": True,
            "clip_video_is_separately_recorded": True,
            "official_one_query_video_roster_used": True,
            "coordinate128_development_exclusion": {
                "source_roster": str(CURRENT_DEV_ROSTER.relative_to(ROOT)),
                "excluded_unit": "parent_sha256 identity, per target domain",
                "excluded_parent_counts": {domain: len(excluded_parents[domain]) for domain in DOMAINS},
                "selected_parent_overlap_count": 0,
            },
        },
        "history_exposure": {
            "full_source_registry_historical_exposure": full["historical_exposure"],
            "globally_unseen_claim": False,
            "known_current_development_disjointness": "All parent_sha256 identities in the current Coordinate128 roster were excluded per domain.",
            "older_project_history": "The full source registry marks historical_exposure=true. Exposure outside the specifically excluded current Coordinate128 parent set is not proven absent; this roster must not be described as globally unseen.",
        },
        "source_registry_bindings": {
            "full_source_roster_path": str(FULL_ROSTER.relative_to(ROOT)),
            "full_source_roster_sha256": metadata_sources["full_source_roster"]["sha256"],
            "source_runtime_path": str(SOURCE_RUNTIME.relative_to(ROOT)),
            "source_runtime_sha256": source_runtime_sha,
            "source_checkpoint_bindings": runtime.get("source_checkpoints"),
            "official_one_query_registry_hash_salt": "PaperOneQueryVideo-20261005",
        },
        "domains": domain_inventory,
        "records": fixed_records,
        "cache_inventory": cache_inventory,
        "data_boundary": {
            "GT_or_annotations_opened": False,
            "scores_or_statistics_opened": False,
            "PLAN_rows_or_query_caption_bodies_read_by_this_preparer": False,
            "subject_json_bodies_read": False,
            "REQUEST_json_read_by_this_preparer": False,
            "cache_payloads_deserialized": False,
            "cache_payload_sha_recomputed": False,
            "one_prior_schema_probe_boundary_note": "An earlier worker metadata schema probe parsed artifacts/stvg_final_table1_v1/tracking_cases/HC01/REQUEST.json to inspect keys; query_text was materialized in process memory but was not printed, indexed, selected on, or emitted. That REQUEST file is not read by this preparer.",
        },
        "prepared_by": {
            "script_path": str(Path(__file__).resolve().relative_to(ROOT)),
            "script_sha256": _sha256_file(Path(__file__).resolve()),
            "time_unix": time.time(),
        },
        "GT_read": False,
    }
    inventory = {
        "schema": "stvg-lite-event-support-confirmation-roster-inventory-v1",
        "status": "metadata_inventory_complete",
        "scope": "CPU-only metadata inventory and fixed 32/domain source/query proposal; no model, GT, score, annotation, or media execution",
        "metadata_sources": metadata_sources,
        "source_registry": {
            "full_roster_status": full.get("status"),
            "historical_exposure": full.get("historical_exposure"),
            "query_count": full.get("official_query_count"),
            "domain_counts": full.get("domain_counts"),
            "all_manifest_rows_preserved": full.get("all_manifest_rows_preserved"),
            "selection_uses_GT_or_scores": full.get("selection_uses_GT_or_scores"),
            "input_manifest_sha256_claimed_by_roster": full.get("manifest_sha256"),
            "current_coordinate128_parent_counts": {domain: len(excluded_parents[domain]) for domain in DOMAINS},
            "full_source_registry_historical_exposure_disclosed": True,
        },
        "domain_inventory": domain_inventory,
        "fixed_roster": {
            "path": str(OUT_LOCK.relative_to(ROOT)),
            "count": len(fixed_records),
            "domain_counts": proposal["domain_counts"],
            "parent_overlap_with_coordinate128": 0,
            "selection_independent_of_cache_state": True,
        },
        "cache_inventory": cache_inventory,
        "capture_interface": {
            "frozen_adapter_path": "scripts/run_stvg_table1_source_v1.py",
            "entrypoint": "capture_one(model, record, frame_cache)",
            "record_source": str(OUT_LOCK.relative_to(ROOT)),
            "input_resolution": "The adapter resolves the original PLAN row by dataset and query_ordinal at capture time and checks plan, subject, video hash, frame IDs, and geometry against the selected opaque roster record.",
            "source_models": {domain: TARGET_BINDING[domain]["source_model"] for domain in DOMAINS},
            "source_directions": {domain: TARGET_BINDING[domain]["source_direction"] for domain in DOMAINS},
            "existing_cli_limitation": "run_stvg_table1_source_v1.py accepts only the existing fixed 64 preflight aliases; run_stvg_table1_full_source_v1.py requires the complete 13,785-row roster. Neither CLI directly accepts this confirmation roster.",
            "confirmation_runner": "scripts/run_stvg_lite_confirmation_source_v1.py (parameterized --all-confirmation or --case-alias; uses confirmation/ROSTER_LOCK.json and requires a root-authored confirmation/SOURCE_RUNTIME.json bound to this roster and the original locked Source runtime)",
            "not_started": True,
        },
        "history_exposure": proposal["history_exposure"],
        "data_boundary": proposal["data_boundary"],
        "prepared_by": proposal["prepared_by"],
        "GT_read": False,
    }
    return inventory, proposal


def main() -> None:
    inventory, proposal = prepare()
    # The standard-path roster is deliberately not marked locked. Root must
    # independently verify it and create its SOURCE_RUNTIME.json before a
    # source-capture launcher will accept it.
    roster_lock = dict(proposal)
    roster_lock["schema"] = "stvg-lite-event-support-confirmation-roster-v1"
    roster_lock["status"] = "prepared_for_root_review"
    _write_json(OUT_LOCK, roster_lock)
    _write_json(OUT_PROPOSAL, proposal)
    _write_json(OUT_INVENTORY, inventory)
    print("CONFIRMATION_ROSTER_PREPARED 64 metadata rows; root review required; no capture started")


if __name__ == "__main__":
    main()
