"""Capture frozen TA-STVG appearance maps for fixed short physical windows.

This worker is deliberately separate from the old sparse-source captures.  It
uses only the locked short-window manifest and canonical FFV1 frames, makes no
new proposals, performs no backward pass, and writes no query text or labels.
The normal execution entry point is ``capture``; ``cpu-contracts`` is synthetic
and does not initialize CUDA or read experiment inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUT = ROOT / "artifacts/stvg_short_event_tracklets_v1"
MANIFEST_PATH = OUT / "PREPARED_MANIFEST.json"
LOCK_PATH = OUT / "FEATURE_CAPTURE_LOCK.json"
RUNTIME_PATH = OUT / "FEATURE_RUNTIME.json"
CONTRACT_PATH = OUT / "CPU_FEATURE_CAPTURE_CONTRACTS.json"
FEATURE_DIR = OUT / "features"
BARRIER_PATH = OUT / "FEATURE_CAPTURE_BARRIER.json"
RUN_LOCK_PATH = OUT / ".FEATURE_CAPTURE_RUN.lock"
SOURCE_CASES = ROOT / "artifacts/stvg_mllm_candidate_feedback_v1/source_cases"
CANONICAL_BASE = ROOT / "artifacts/stvg_target_teacher_distillation_v1/canonical_inputs"
PLAN_BASE = ROOT / "artifacts/decota_paper_experiments_v1"
CASE_IDS = tuple(
    [f"hc2_{i:03d}" for i in range(4)]
    + [f"vidstg_{i:03d}" for i in range(4)]
)
MANIFEST_SCHEMA = "stvg-short-event-tracklets-plan-v1"
FEATURE_SCHEMA = "stvg-short-window-frozen-appearance-v1"
CONTRACT_SCHEMA = "stvg-short-window-feature-cpu-contract-v1"
MAX_FRAMES_PER_CASE = 20
FEATURE_CHANNELS = 256
MODEL_INPUT_HEIGHT = 224
REQUIRED_CAPTURE_PINS = (
    "scripts/run_decota_paper_main_v1.py",
    "scripts/run_spatial_regression_alignment_v1.py",
    "scripts/run_tastvg_evidence_vulnerability_v2.py",
    "scripts/stvg_mllm_feedback_common_v1.py",
    "vg_tta/stvg_opd_frozen_mask_cache012.py",
    "vg_tta/stvg_opd_frozen_visual_host011.py",
    "vg_tta/stvg_opd_frozen_attention_lifetime010.py",
    "vg_tta/stvg_opd_frozen_module_cpu008.py",
    "vg_tta/stvg_opd_frozen_capture_cpu007.py",
    "methods/decota_final_simplified_v1/backbone.py",
    "methods/decota_final_simplified_v1/replay.py",
    "methods/decota_final_simplified_v1/tensors.py",
)


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _tensor_sha256(value: Any, torch_module: Any) -> str:
    tensor = value.detach().cpu().contiguous()
    return _sha_bytes(tensor.view(torch_module.uint8).numpy().tobytes())


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha(value: Any) -> str:
    raw = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return _sha_bytes(raw)


def _read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_once_json(path: Path, value: Any) -> dict[str, Any]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"refusing to overwrite immutable output: {path}")
    raw = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": _sha_bytes(raw), "bytes": len(raw)}


def _write_once_torch(path: Path, value: Any, torch_module: Any) -> dict[str, Any]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"refusing to overwrite immutable output: {path}")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch_module.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": _sha_file(path), "bytes": path.stat().st_size}


def _root_path(value: str | Path) -> Path:
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (ROOT / path).resolve()
    if resolved != ROOT and ROOT not in resolved.parents:
        raise ValueError(f"path escapes project root: {value}")
    return resolved


def _fixed_case_ids() -> list[str]:
    return list(CASE_IDS)


def _manifest_case_frames(case: Mapping[str, Any]) -> tuple[list[int], dict[int, int]]:
    centers = case.get("centers")
    if not isinstance(centers, list) or len(centers) != 4:
        raise ValueError(f"{case.get('case_id')}: expected exactly four locked centers")
    coverage = [int(v) for v in case["canonical_physical_coverage_half_open"]]
    if len(coverage) != 2 or coverage[1] <= coverage[0]:
        raise ValueError(f"{case.get('case_id')}: invalid canonical physical coverage")
    physical_to_local: dict[int, int] = {}
    all_ids: set[int] = set()
    for center in centers:
        center_id = int(center["center_physical_frame_id"])
        physical_ids = [int(v) for v in center["physical_frame_ids"]]
        local_ids = [int(v) for v in center["local_frame_indices"]]
        expected_window = list(range(max(coverage[0], center_id - 2), min(coverage[1] - 1, center_id + 2) + 1))
        if len(physical_ids) != len(local_ids) or not 1 <= len(physical_ids) <= 5:
            raise ValueError(f"{case.get('case_id')}: malformed clipped +/-2 window")
        if physical_ids != expected_window:
            raise ValueError(f"{case.get('case_id')}: window is not the exact coverage-clipped physical +/-2 window")
        if center_id not in physical_ids:
            raise ValueError(f"{case.get('case_id')}: center is outside its physical window")
        anchor_index = physical_ids.index(center_id)
        if any(b != a + 1 for a, b in zip(local_ids, local_ids[1:])):
            raise ValueError(f"{case.get('case_id')}: local frame mapping is not adjacent")
        for physical_id, local_id in zip(physical_ids, local_ids):
            if physical_id in physical_to_local and physical_to_local[physical_id] != local_id:
                raise ValueError(f"{case.get('case_id')}: inconsistent repeated physical/local mapping")
            physical_to_local[physical_id] = local_id
            all_ids.add(physical_id)
        if not center.get("matched_tracklets"):
            raise ValueError(f"{case.get('case_id')}: center has no locked candidate provenance")
        tracklet_ids = [int(item["candidate_id"]) for item in center["matched_tracklets"]]
        if tracklet_ids != [int(v) for v in case["fixed_displayed_candidate_ids"]]:
            raise ValueError(f"{case.get('case_id')}: tracklet candidate IDs differ from frozen source set")
        for item in center["matched_tracklets"]:
            if [int(v) for v in item["physical_frame_ids"]] != physical_ids:
                raise ValueError(f"{case.get('case_id')}: candidate tracklet uses a different physical window")
            if [int(v) for v in item["local_frame_indices"]] != local_ids:
                raise ValueError(f"{case.get('case_id')}: candidate tracklet uses a different local frame mapping")
            if int(item["anchor_index"]) != anchor_index:
                raise ValueError(f"{case.get('case_id')}: candidate anchor index differs from the center")
    ordered = sorted(all_ids)
    if len(ordered) > MAX_FRAMES_PER_CASE or len(ordered) < 4:
        raise ValueError(f"{case.get('case_id')}: short-window union size outside [4,20]")
    local_values = [physical_to_local[value] for value in ordered]
    if local_values != sorted(set(local_values)):
        raise ValueError(f"{case.get('case_id')}: local-to-physical map is not one-to-one and increasing")
    return ordered, physical_to_local


def _verify_bindings(manifest: Mapping[str, Any], manifest_sha: str) -> dict[str, Any]:
    """Verify runtime and all artifact pins before opening source payloads/media."""
    lock = _read_json(LOCK_PATH)
    runtime = _read_json(RUNTIME_PATH)
    if lock.get("status") != "locked" or lock.get("GT_read") is not False:
        raise ValueError("FEATURE_CAPTURE_LOCK must be locked with GT_read=false")
    if runtime.get("status") != "locked" or runtime.get("GT_read") is not False:
        raise ValueError("FEATURE_RUNTIME must be locked with GT_read=false")

    manifest_bound = lock.get("manifest_sha256", lock.get("prepared_manifest_sha256"))
    if manifest_bound != manifest_sha:
        raise ValueError("feature lock does not bind the exact prepared manifest bytes")
    runtime_manifest_bound = runtime.get("manifest_sha256", runtime.get("prepared_manifest_sha256"))
    if runtime_manifest_bound != manifest_sha:
        raise ValueError("feature runtime does not bind the exact prepared manifest bytes")
    lock_bound = runtime.get("feature_lock_sha256", runtime.get("lock_sha256"))
    if lock_bound != _sha_file(LOCK_PATH):
        raise ValueError("feature runtime does not bind the feature lock bytes")

    expected_script_sha = _sha_file(Path(__file__).resolve())
    expected_contract_sha = _sha_file(CONTRACT_PATH)
    pin_maps = []
    for document in (lock, runtime):
        pins: dict[str, Any] = {}
        for key in ("pins", "implementation_pins", "protected"):
            declared = document.get(key, {})
            if not isinstance(declared, Mapping):
                raise ValueError("lock pins must be ROOT-relative path-to-SHA mappings")
            for relative, expected in declared.items():
                if relative in pins and pins[relative] != expected:
                    raise ValueError(f"conflicting hashes for locked path {relative}")
                pins[str(relative)] = expected
        checked: dict[str, str] = {}
        for relative, expected in pins.items():
            path = _root_path(str(relative))
            if not path.is_file() or _sha_file(path) != expected:
                raise ValueError(f"locked pin changed or is missing: {relative}")
            checked[str(relative)] = str(expected)
        pin_maps.append(checked)
    combined = {**pin_maps[0], **pin_maps[1]}
    if combined.get("scripts/capture_stvg_short_window_features_v1.py") != expected_script_sha:
        raise ValueError("feature lock/runtime must pin this exact worker source")
    if combined.get("artifacts/stvg_short_event_tracklets_v1/CPU_FEATURE_CAPTURE_CONTRACTS.json") != expected_contract_sha:
        raise ValueError("feature lock/runtime must pin the exact CPU feature contracts")
    missing_capture_pins = [relative for relative in REQUIRED_CAPTURE_PINS if relative not in combined]
    if missing_capture_pins:
        raise ValueError(f"feature runtime omits frozen capture dependency pins: {missing_capture_pins}")
    if _sha_file(MANIFEST_PATH) != manifest_sha:
        raise ValueError("prepared manifest changed during lock verification")
    return {
        "lock": lock,
        "runtime": runtime,
        "lock_sha256": _sha_file(LOCK_PATH),
        "runtime_sha256": _sha_file(RUNTIME_PATH),
        "manifest_sha256": manifest_sha,
        "worker_sha256": expected_script_sha,
        "contract_sha256": expected_contract_sha,
    }


def _verify_manifest_structure(manifest: Mapping[str, Any], manifest_sha: str) -> dict[str, Any]:
    if manifest.get("schema") != MANIFEST_SCHEMA or manifest.get("status") != "prepared_manifest_not_executed":
        raise ValueError("unexpected or already-executed short-window manifest")
    if manifest.get("GT_read") is not False or manifest.get("GPU_used") is not False:
        raise ValueError("short-window manifest must declare GT-free, pre-inference preparation")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or [str(row.get("case_id")) for row in cases] != list(CASE_IDS):
        raise ValueError("manifest must contain exactly the fixed old-eight cases in stable order")
    if manifest.get("case_count") != 8 or manifest.get("center_count") != 32:
        raise ValueError("manifest case/center counts do not match the locked old-eight design")

    binding_paths = {
        "source_roster_lock_path": ("artifacts/stvg_adjacent_motion_opd_v1/ROSTER_LOCK.json", "source_roster_lock_sha256"),
        "candidate_input_barrier_path": ("artifacts/stvg_adjacent_motion_opd_v1/CANDIDATE_INPUT_BARRIER.json", "candidate_input_barrier_sha256"),
        "qa_student_barrier_path": ("artifacts/stvg_temporal_caption_query_spatial_opd_v1/STUDENT_GLOBAL_BARRIER.json", "qa_student_barrier_sha256"),
        "track_helper_path": ("scripts/stvg_adjacent_affine_tracklet_v1.py", "track_helper_sha256"),
        "proposal_path": ("protocols/stvg_short_event_tracklets_v1.proposal.json", "proposal_sha256"),
    }
    bindings = manifest.get("source_bindings")
    if not isinstance(bindings, Mapping):
        raise ValueError("manifest lacks its frozen source binding table")
    for path_key, (relative, hash_key) in binding_paths.items():
        path = ROOT / relative
        if bindings.get(path_key) != relative or not path.is_file() or _sha_file(path) != bindings.get(hash_key):
            raise ValueError(f"manifest source binding changed: {path_key}")

    case_index: dict[str, dict[str, Any]] = {}
    for case in cases:
        cid = str(case["case_id"])
        if case.get("cohort") != "historically_exposed_old8" or case.get("GT_read") is not False:
            raise ValueError(f"{cid}: old-eight exposure/no-GT provenance changed")
        expected_source = SOURCE_CASES / f"{cid}.pt"
        if _root_path(case["source_case_path"]) != expected_source.resolve():
            raise ValueError(f"{cid}: source case path escaped the fixed old-eight source directory")
        if not isinstance(case.get("source_case_sha256"), str) or len(case["source_case_sha256"]) != 64:
            raise ValueError(f"{cid}: missing source-case content binding")
        if not expected_source.is_file() or _sha_file(expected_source) != case["source_case_sha256"]:
            raise ValueError(f"{cid}: source case bytes differ from the prepared manifest")
        if not isinstance(case.get("source_model_state_sha256"), str) or len(case["source_model_state_sha256"]) != 64:
            raise ValueError(f"{cid}: missing source model state binding")
        if case.get("dataset") != cid.rsplit("_", 1)[0]:
            raise ValueError(f"{cid}: dataset/case naming mismatch")
        qa_path = _root_path(case["qa_student_path"])
        expected_qa = (ROOT / "artifacts/stvg_temporal_caption_query_spatial_opd_v1/student" / f"{cid}.pt").resolve()
        if qa_path != expected_qa or not qa_path.is_file() or _sha_file(qa_path) != case["qa_student_sha256"]:
            raise ValueError(f"{cid}: reused QA student artifact differs from the prepared manifest")
        if _root_path(case["canonical_request_path"]) != (CANONICAL_BASE / cid / "REQUEST.json").resolve():
            raise ValueError(f"{cid}: canonical request path escaped its fixed directory")
        request_path = _root_path(case["canonical_request_path"])
        if not request_path.is_file() or _sha_file(request_path) != case["canonical_request_sha256"]:
            raise ValueError(f"{cid}: canonical REQUEST bytes differ from the prepared manifest")
        video_path = _root_path(case["canonical_video_path"])
        if video_path.parent != (CANONICAL_BASE / cid).resolve():
            raise ValueError(f"{cid}: canonical video path escaped its fixed directory")
        if video_path.name != f"{cid}.mkv":
            raise ValueError(f"{cid}: canonical video filename changed")
        if case.get("fixed_displayed_candidate_ids", [1])[0] != 1:
            raise ValueError(f"{cid}: frozen Source proposal is not first")
        physical_ids, physical_to_local = _manifest_case_frames(case)
        case_index[cid] = {
            "case": case,
            "physical_frame_ids": physical_ids,
            "physical_to_local": physical_to_local,
        }
    return case_index


def _preflight() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    for path in (MANIFEST_PATH, LOCK_PATH, RUNTIME_PATH, CONTRACT_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"required frozen feature input is missing: {path.relative_to(ROOT)}")
    manifest_sha = _sha_file(MANIFEST_PATH)
    manifest = _read_json(MANIFEST_PATH)
    lock_context = _verify_bindings(manifest, manifest_sha)
    cases = _verify_manifest_structure(manifest, manifest_sha)
    contract = _read_json(CONTRACT_PATH)
    if contract.get("schema") != CONTRACT_SCHEMA or contract.get("status") != "passed":
        raise ValueError("CPU feature contracts are not sealed as passed")
    if contract.get("worker_sha256") != lock_context["worker_sha256"]:
        raise ValueError("CPU contracts bind a different worker revision")
    output_files = [
        path
        for cid in CASE_IDS
        for path in (FEATURE_DIR / f"{cid}.pt", FEATURE_DIR / f"{cid}.json")
    ]
    if BARRIER_PATH.exists() or any(path.exists() for path in output_files) or RUN_LOCK_PATH.exists():
        raise FileExistsError("feature capture outputs already exist; refusing overwrite or implicit resume")
    return manifest, cases, lock_context


def extract_appearance_maps(
    norm: Any,
    views: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    requested_frame_ids: Sequence[int],
    torch_module: Any | None = None,
) -> dict[str, Any]:
    """Extract and merge ``H[:h*w]`` appearance grids by physical frame ID.

    This mirrors the existing SpatialReplay ordering exactly: normalize the
    saved prefix, take the first ``h*w`` tokens, permute ``(L,N,C)`` to
    ``(N,C,L)``, then reshape to ``(frames,256,h,w)``. No temporal query or
    candidate geometry is inferred here.
    """
    if torch_module is None:
        import torch as torch_module
    torch = torch_module
    if len(views) != 2 or len(records) != 2:
        raise ValueError("the frozen native capture must contain exactly two offsets")
    required = [int(value) for value in requested_frame_ids]
    if not required or required != sorted(set(required)):
        raise ValueError("requested physical frame IDs must be nonempty, sorted, and unique")
    by_frame: dict[int, Any] = {}
    by_offset: dict[int, list[int]] = {}
    grid_by_offset: dict[int, list[int]] = {}
    source_dtype_by_offset: dict[int, str] = {}
    for view, record in zip(views, records):
        offset = int(record["offset"])
        if offset not in (0, 1) or offset in by_offset:
            raise ValueError("native offsets must be exactly 0 and 1")
        frame_ids = [int(value) for value in record["frame_ids"]]
        if frame_ids != required[offset::2]:
            raise ValueError("native offset frame IDs do not preserve the original alternating temporal split")
        prefix = view.get("prefix")
        if not torch.is_tensor(prefix) or prefix.ndim != 3:
            raise ValueError("each native prefix must be a rank-three tensor")
        with torch.no_grad():
            H = norm(prefix).detach()
            h, w = (int(v) for v in view["info"]["fea_map_size"])
            count = h * w
            nf = int(H.shape[1])
            if min(h, w) <= 0 or count > int(H.shape[0]):
                raise ValueError("feature-map size is incompatible with the native prefix")
            if int(H.shape[2]) != FEATURE_CHANNELS or len(frame_ids) != nf:
                raise ValueError("prefix channels/frame count do not match the native offset record")
            if len(frame_ids) == 0 or len(frame_ids) != len(set(frame_ids)):
                raise ValueError("native offset frame IDs must be nonempty and unique")
            fa = H[:count].permute(1, 2, 0).reshape(nf, FEATURE_CHANNELS, h, w).detach()
        for index, frame_id in enumerate(frame_ids):
            if frame_id in by_frame:
                raise ValueError(f"physical frame {frame_id} occurs in both native offsets")
            by_frame[frame_id] = fa[index].detach().cpu().contiguous()
        by_offset[offset] = frame_ids
        grid_by_offset[offset] = [h, w]
        source_dtype_by_offset[offset] = str(prefix.dtype)
    if set(by_frame) != set(required):
        missing, extra = sorted(set(required) - set(by_frame)), sorted(set(by_frame) - set(required))
        raise ValueError(f"native offsets do not cover requested frames exactly: missing={missing}, extra={extra}")
    ordered = torch.stack([by_frame[frame_id] for frame_id in required], dim=0).contiguous()
    if ordered.device.type != "cpu" or ordered.requires_grad or not bool(torch.isfinite(ordered).all()):
        raise ValueError("feature maps must be finite detached CPU tensors")
    return {
        "frame_ids": required,
        "maps": ordered,
        "shape": [int(v) for v in ordered.shape],
        "dtype": str(ordered.dtype),
        "offset_frame_ids": {str(offset): by_offset[offset] for offset in sorted(by_offset)},
        "grid_size_by_offset": {str(offset): grid_by_offset[offset] for offset in sorted(grid_by_offset)},
        "prefix_dtype_by_offset": {str(offset): source_dtype_by_offset[offset] for offset in sorted(source_dtype_by_offset)},
        "layout": "physical-time order; H[:h*w].permute(1,2,0).reshape(nf,256,h,w)",
    }


def _replace_row_frame_grid(
    row: Mapping[str, Any], physical_ids: Sequence[int], video_path: str,
    video_sha256: str, width: int, height: int,
) -> dict[str, Any]:
    ids = [int(value) for value in physical_ids]
    if not ids or ids != sorted(set(ids)):
        raise ValueError("capture row frame grid must be nonempty sorted unique physical IDs")
    result = dict(row)
    inputs = dict(row["input"])
    inputs.update(
        video_path=str(video_path), video_sha256=str(video_sha256), frame_ids=ids,
        width=int(width), height=int(height),
    )
    result["input"] = inputs
    result["frame_ids"] = ids
    if list(result["input"]["frame_ids"]) != ids or result["frame_ids"] != ids:
        raise RuntimeError("new physical IDs did not reach both native model row fields")
    return result


def _frozen_prefix_capture(
    model: Any, frames: Any, row: Mapping[str, Any], physical_ids: Sequence[int],
    torch_module: Any, frozen_forward_impl: Any | None = None,
) -> tuple[Any, Any, Any, dict[str, Any]]:
    if frozen_forward_impl is None:
        from vg_tta.stvg_opd_frozen_mask_cache012 import frozen_forward as frozen_forward_impl
    with torch_module.no_grad():
        batch, records, base = frozen_forward_impl(model, frames, row)
        if torch_module.is_grad_enabled():
            raise RuntimeError("frozen full-model prefix ran with gradient tracking enabled")
        feature = extract_appearance_maps(base.norm, base.views, records, physical_ids, torch_module)
    return batch, records, base, feature


def _mock_full_capture_contract(torch_module: Any) -> dict[str, Any]:
    """Exercise the actual capture orchestration with synthetic files/tensors.

    All media decoding, checkpoint loading, and model calls are replaced by
    deterministic CPU doubles. The real _run_capture, source-case validation,
    row replacement, frozen-prefix adapter, feature serialization, and barrier
    construction still execute end to end.
    """
    import tempfile
    from types import SimpleNamespace
    import numpy as np

    from methods.decota_final_simplified_v1.tensors import state_hash
    import scripts.run_tastvg_evidence_vulnerability_v2 as clean_loader
    import scripts.stvg_mllm_feedback_common_v1 as common
    import vg_tta.stvg_opd_frozen_mask_cache012 as capture_module

    global ROOT, OUT, MANIFEST_PATH, LOCK_PATH, RUNTIME_PATH, CONTRACT_PATH
    global FEATURE_DIR, BARRIER_PATH, RUN_LOCK_PATH
    old_globals = {
        key: globals()[key]
        for key in ("ROOT", "OUT", "MANIFEST_PATH", "LOCK_PATH", "RUNTIME_PATH", "CONTRACT_PATH", "FEATURE_DIR", "BARRIER_PATH", "RUN_LOCK_PATH")
    }
    old_preflight = globals()["_preflight"]
    old_load_model = globals()["_load_model"]
    old_capture_case = globals()["_decode_canonical_frames"]
    old_clean_loader = clean_loader.install_clean_loader
    old_guard = common.guard
    old_raw_row = common.raw_row
    old_frozen_forward = capture_module.frozen_forward
    old_cuda_available = torch_module.cuda.is_available
    old_cuda_empty_cache = torch_module.cuda.empty_cache
    old_cuda_sync = torch_module.cuda.synchronize
    source_objects: dict[str, dict[str, Any]] = {}
    try:
        with tempfile.TemporaryDirectory(prefix="stvg_short_feature_mock_") as temporary:
            root = Path(temporary)
            globals()["ROOT"] = root
            globals()["OUT"] = root / "artifacts"
            globals()["MANIFEST_PATH"] = root / "manifest.json"
            globals()["LOCK_PATH"] = root / "feature.lock.json"
            globals()["RUNTIME_PATH"] = root / "feature.runtime.json"
            globals()["CONTRACT_PATH"] = root / "contract.json"
            globals()["FEATURE_DIR"] = root / "features"
            globals()["BARRIER_PATH"] = root / "FEATURE_CAPTURE_BARRIER.json"
            globals()["RUN_LOCK_PATH"] = root / ".capture.lock"
            (root / "sources").mkdir()
            (root / "canonical").mkdir()

            class FakeModel(torch_module.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.weight = torch_module.nn.Parameter(torch_module.tensor([0.25]), requires_grad=False)
                    self.eval()
                    self._fixed_full_state_hash = state_hash(self.state_dict())

            fake_model = FakeModel()
            expected_state = state_hash(fake_model.state_dict())
            fake_sha = lambda char: char * 64
            entries: dict[str, dict[str, Any]] = {}
            manifest_cases = []
            for cid in CASE_IDS:
                dataset = cid.rsplit("_", 1)[0]
                source_case = {
                    "case_id": cid,
                    "dataset": dataset,
                    "source": "synthetic_source",
                    "original_source_only": True,
                    "GT_read": False,
                    "source_model_state_sha256": expected_state,
                    "frame_ids": [0, 1, 2, 3],
                    "native_boxes": [[0.1, 0.1, 0.2, 0.2]] * 4,
                    "input": {"video_sha256": fake_sha("2"), "width": 4, "height": 4, "frame_ids": [0, 1, 2, 3]},
                }
                source_path = root / "sources" / f"{cid}.pt"
                torch_module.save(source_case, source_path)
                source_sha = _sha_file(source_path)
                source_objects[str(source_path)] = source_case
                canonical_dir = root / "canonical" / cid
                canonical_dir.mkdir(parents=True)
                request_path = canonical_dir / "REQUEST.json"
                video_path = canonical_dir / f"{cid}.mkv"
                request = {
                    "video_path": str(video_path),
                    "canonical_video_sha256": fake_sha("3"),
                    "canonical_pixel_sha256": fake_sha("4"),
                    "source_video_sha256": fake_sha("2"),
                }
                request_path.write_text(json.dumps(request), encoding="utf-8")
                case = {
                    "case_id": cid,
                    "dataset": dataset,
                    "cohort": "historically_exposed_old8",
                    "source_case_path": source_path.relative_to(root).as_posix(),
                    "source_case_sha256": source_sha,
                    "source_model_state_sha256": expected_state,
                    "source_grid_count": 4,
                    "source_grid_frame_ids_sha256": _canonical_sha([0, 1, 2, 3]),
                    "source_video_sha256": fake_sha("2"),
                    "canonical_request_path": request_path.relative_to(root).as_posix(),
                    "canonical_request_sha256": _sha_file(request_path),
                    "canonical_video_path": video_path.relative_to(root).as_posix(),
                    "canonical_video_sha256": fake_sha("3"),
                    "canonical_pixel_sha256": fake_sha("4"),
                    "canonical_physical_coverage_half_open": [0, 4],
                    "fixed_displayed_candidate_ids": [1],
                }
                physical_ids = [0, 1, 2, 3]
                entries[cid] = {"case": case, "physical_frame_ids": physical_ids, "physical_to_local": {i: i for i in physical_ids}}
                manifest_cases.append(case)

            manifest = {"cases": manifest_cases}
            context = {
                "manifest_sha256": fake_sha("5"), "lock_sha256": fake_sha("6"),
                "runtime_sha256": fake_sha("7"), "worker_sha256": fake_sha("8"),
                "contract_sha256": fake_sha("9"),
            }
            globals()["_preflight"] = lambda: (manifest, entries, context)
            globals()["_load_model"] = lambda _source, _expected, _torch: fake_model

            def fake_raw_row(record):
                return {
                    "frame_ids": list(record["frame_ids"]),
                    "input": {
                        "frame_ids": list(record["frame_ids"]), "width": 4, "height": 4,
                        "video_path": "old-source-video", "video_sha256": record["input"]["video_sha256"],
                        "caption": "synthetic only",
                    },
                    "parses": {"subject": "synthetic"},
                }

            common.raw_row = fake_raw_row
            clean_loader.install_clean_loader = lambda: None
            common.guard = lambda _event, _args: None

            def fake_decode(request, _request_path, case, physical_ids, _mapping, expected_height, expected_width):
                assert list(physical_ids) == [0, 1, 2, 3]
                assert expected_height == expected_width == 4
                frames = np.zeros((4, 4, 4, 3), dtype=np.uint8)
                frame_rows = [
                    {"physical_frame_id": frame_id, "local_frame_index": frame_id,
                     "source_timestamp_s": float(frame_id), "height_width_channels": [4, 4, 3],
                     "rgb_sha256": _sha_bytes(frames[frame_id].tobytes())}
                    for frame_id in physical_ids
                ]
                return {
                    "frames": frames,
                    "pixel_sha256": _sha_bytes(frames.tobytes()),
                    "per_frame": frame_rows,
                    "backend": "synthetic-cpu",
                    "decoded_local_frame_count": 4,
                    "canonical_video_sha256": case["canonical_video_sha256"],
                    "canonical_pixel_sha256_declared": case["canonical_pixel_sha256"],
                }

            globals()["_decode_canonical_frames"] = fake_decode

            def fake_forward(model, frames, row):
                assert not torch_module.is_grad_enabled()
                assert row["frame_ids"] == row["input"]["frame_ids"] == [0, 1, 2, 3]
                assert row["input"]["video_sha256"] == fake_sha("3")
                views = []
                for ids in ([0, 2], [1, 3]):
                    prefix = torch_module.arange((4 + 3) * len(ids) * FEATURE_CHANNELS, dtype=torch_module.float32)
                    prefix = prefix.reshape(7, len(ids), FEATURE_CHANNELS)
                    views.append({"prefix": prefix, "info": {"fea_map_size": [2, 2]}})
                records = [{"offset": 0, "frame_ids": [0, 2]}, {"offset": 1, "frame_ids": [1, 3]}]
                base = SimpleNamespace(norm=lambda value: value, views=views)
                return None, records, base

            capture_module.frozen_forward = fake_forward
            torch_module.cuda.is_available = lambda: True
            torch_module.cuda.empty_cache = lambda: None
            torch_module.cuda.synchronize = lambda: None
            result = _run_capture()
            assert result["cases"] == 8
            barrier = _read_json(BARRIER_PATH)
            assert barrier["status"] == "sealed" and barrier["count"] == 8
            assert barrier["feature_offset_forwards"] == 16 and barrier["backward_calls"] == 0
            assert torch_module.cuda.is_initialized() is False
            for cid in CASE_IDS:
                saved = torch_module.load(FEATURE_DIR / f"{cid}.pt", map_location="cpu", weights_only=False)
                assert saved["frame_ids"] == [0, 1, 2, 3]
                assert saved["appearance_maps"].shape == (4, FEATURE_CHANNELS, 2, 2)
                assert saved["appearance_maps"].device.type == "cpu"
                assert saved["metadata"]["GT_read"] is False and saved["metadata"]["backward_calls"] == 0
                assert "caption" not in saved and "caption" not in saved["metadata"]
            return {"status": "passed", "cases": 8, "synthetic_full_capture_path": True}
    finally:
        globals()["_preflight"] = old_preflight
        globals()["_load_model"] = old_load_model
        globals()["_decode_canonical_frames"] = old_capture_case
        clean_loader.install_clean_loader = old_clean_loader
        common.guard = old_guard
        common.raw_row = old_raw_row
        capture_module.frozen_forward = old_frozen_forward
        torch_module.cuda.is_available = old_cuda_available
        torch_module.cuda.empty_cache = old_cuda_empty_cache
        torch_module.cuda.synchronize = old_cuda_sync
        for key, value in old_globals.items():
            globals()[key] = value


def _decode_canonical_frames(
    request: Mapping[str, Any],
    request_path: Path,
    case: Mapping[str, Any],
    requested_physical_ids: Sequence[int],
    expected_local_by_physical: Mapping[int, int],
    expected_height: int,
    expected_width: int,
) -> dict[str, Any]:
    import cv2
    import numpy as np

    case_id = str(case["case_id"])
    video_path = _root_path(request["video_path"])
    if video_path != _root_path(case["canonical_video_path"]):
        raise ValueError(f"{case_id}: REQUEST canonical video path differs from manifest")
    if _sha_file(video_path) != case["canonical_video_sha256"]:
        raise ValueError(f"{case_id}: canonical encoded video SHA mismatch")
    if _sha_file(request_path) != case["canonical_request_sha256"]:
        raise ValueError(f"{case_id}: canonical REQUEST bytes changed")
    if request.get("canonical_video_sha256") != case["canonical_video_sha256"]:
        raise ValueError(f"{case_id}: REQUEST video SHA differs from manifest")
    if request.get("canonical_pixel_sha256") != case.get("canonical_pixel_sha256"):
        raise ValueError(f"{case_id}: REQUEST pixel digest metadata differs from manifest")
    if request.get("source_video_sha256") != case["source_video_sha256"]:
        raise ValueError(f"{case_id}: REQUEST source-video binding differs from manifest")

    physical_ids = [int(value) for value in requested_physical_ids]
    if physical_ids != sorted(set(physical_ids)):
        raise ValueError("canonical decode requests must be sorted unique physical IDs")
    frame_map = request.get("frame_map")
    local_count = int(request.get("local_frame_count", -1))
    if not isinstance(frame_map, list) or local_count != len(frame_map) or local_count <= 0:
        raise ValueError(f"{case_id}: REQUEST local frame map/count invalid")
    physical_to_local: dict[int, int] = {}
    local_seen: set[int] = set()
    timestamp_by_physical: dict[int, float] = {}
    for row in frame_map:
        physical_id = int(row["physical_frame_id"])
        local_id = int(row["local_frame_index"])
        timestamp = float(row["source_timestamp_s"])
        if physical_id in physical_to_local or local_id in local_seen or not math.isfinite(timestamp):
            raise ValueError(f"{case_id}: REQUEST frame map has duplicate or invalid rows")
        physical_to_local[physical_id] = local_id
        local_seen.add(local_id)
        timestamp_by_physical[physical_id] = timestamp
    if local_seen != set(range(local_count)):
        raise ValueError(f"{case_id}: REQUEST local indices are not dense and zero-based")
    ordered_physical = sorted(physical_to_local)
    if ordered_physical != list(range(ordered_physical[0], ordered_physical[-1] + 1)):
        raise ValueError(f"{case_id}: REQUEST physical IDs are not contiguous")
    if [physical_to_local[value] for value in ordered_physical] != list(range(local_count)):
        raise ValueError(f"{case_id}: REQUEST local/physical map is not strictly ordered")
    ordered_timestamps = [timestamp_by_physical[value] for value in ordered_physical]
    if any(b <= a for a, b in zip(ordered_timestamps, ordered_timestamps[1:])):
        raise ValueError(f"{case_id}: REQUEST timestamps are not strictly increasing")
    for physical_id in physical_ids:
        if physical_to_local.get(physical_id) != int(expected_local_by_physical[physical_id]):
            raise ValueError(f"{case_id}: manifest physical/local mapping differs from REQUEST")
    selected_local = {physical_to_local[physical_id]: physical_id for physical_id in physical_ids}
    wanted_local = sorted(selected_local)
    if len(wanted_local) != len(physical_ids) or wanted_local[-1] >= local_count:
        raise ValueError(f"{case_id}: requested canonical frame index is unavailable")

    capture = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
    if not capture.isOpened():
        raise RuntimeError(f"{case_id}: OpenCV/FFmpeg could not open canonical video")
    rgb_by_physical: dict[int, Any] = {}
    per_frame: list[dict[str, Any]] = []
    try:
        backend = capture.getBackendName() if hasattr(capture, "getBackendName") else "unknown"
        if backend not in {"FFMPEG", "unknown"}:
            raise RuntimeError(f"{case_id}: unexpected video backend {backend}")
        # Decode from local frame 0 in order. This avoids backend-dependent
        # seek rounding; the REQUEST map is the sole physical/local clock.
        for local_id in range(wanted_local[-1] + 1):
            before = float(capture.get(cv2.CAP_PROP_POS_FRAMES))
            if math.isfinite(before) and int(round(before)) != local_id:
                raise RuntimeError(f"{case_id}: decoder local position drift before frame {local_id}")
            ok, bgr = capture.read()
            after = float(capture.get(cv2.CAP_PROP_POS_FRAMES))
            if not ok or bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3:
                raise RuntimeError(f"{case_id}: canonical decode ended before local frame {local_id}")
            if math.isfinite(after) and int(round(after)) != local_id + 1:
                raise RuntimeError(f"{case_id}: decoder local position drift after frame {local_id}")
            if local_id not in selected_local:
                continue
            physical_id = selected_local[local_id]
            height, width = map(int, bgr.shape[:2])
            if (height, width) != (int(expected_height), int(expected_width)):
                raise ValueError(f"{case_id}: canonical RGB dimensions differ from the source input metadata")
            rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8)
            rgb_by_physical[physical_id] = rgb
            per_frame.append({
                "physical_frame_id": physical_id,
                "local_frame_index": local_id,
                "source_timestamp_s": timestamp_by_physical[physical_id],
                "height_width_channels": [height, width, 3],
                "rgb_sha256": _sha_bytes(rgb.tobytes()),
            })
            del rgb, bgr
    finally:
        capture.release()
    if list(sorted(rgb_by_physical)) != physical_ids:
        raise RuntimeError(f"{case_id}: not every requested physical frame decoded exactly once")
    frames = np.stack([rgb_by_physical[frame_id] for frame_id in physical_ids], axis=0)
    if frames.dtype != np.uint8 or frames.shape != (len(physical_ids), expected_height, expected_width, 3):
        raise RuntimeError(f"{case_id}: canonical RGB batch has unexpected dtype or shape")
    return {
        "frames": frames,
        "pixel_sha256": _sha_bytes(np.ascontiguousarray(frames).tobytes()),
        "per_frame": per_frame,
        "backend": backend,
        "decoded_local_frame_count": wanted_local[-1] + 1,
        "canonical_video_sha256": case["canonical_video_sha256"],
        "canonical_pixel_sha256_declared": case["canonical_pixel_sha256"],
    }


def _validate_source_case(record: Mapping[str, Any], entry: Mapping[str, Any], torch_module: Any) -> None:
    case = entry["case"]
    cid = case["case_id"]
    if record.get("case_id") != cid or record.get("dataset") != case["dataset"]:
        raise ValueError(f"{cid}: loaded source case identity does not match manifest")
    if record.get("GT_read") is not False or record.get("original_source_only") is not True:
        raise ValueError(f"{cid}: source cache is not marked as source-only and no-GT")
    if record.get("source_model_state_sha256") != case["source_model_state_sha256"]:
        raise ValueError(f"{cid}: source model state SHA differs from manifest")
    original_ids = [int(value) for value in record.get("frame_ids", [])]
    if len(original_ids) != int(case["source_grid_count"]) or _canonical_sha(original_ids) != case["source_grid_frame_ids_sha256"]:
        raise ValueError(f"{cid}: original sparse source grid differs from manifest")
    if record.get("input", {}).get("video_sha256") != case["source_video_sha256"]:
        raise ValueError(f"{cid}: source video SHA differs from canonical REQUEST binding")
    if len(record.get("native_boxes", [])) == 0:
        raise ValueError(f"{cid}: original Source boxes are missing; cannot confirm source context")


def _load_model(source: str, expected_state_sha256: str, torch_module: Any):
    from scripts.run_decota_paper_main_v1 import model_for
    from methods.decota_final_simplified_v1.tensors import state_hash

    model = model_for(source)
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("model_for returned a training/unfrozen source model")
    actual = state_hash(model.state_dict())
    if actual != expected_state_sha256 or getattr(model, "_fixed_full_state_hash", None) != expected_state_sha256:
        raise ValueError("loaded source checkpoint state differs from frozen source case")
    return model


def _capture_case(
    manifest: Mapping[str, Any], entry: Mapping[str, Any], lock_context: Mapping[str, Any],
    source_case: Mapping[str, Any], model: Any, torch_module: Any,
) -> dict[str, Any]:
    from scripts.stvg_mllm_feedback_common_v1 import raw_row
    from vg_tta.stvg_opd_frozen_mask_cache012 import frozen_forward
    from methods.decota_final_simplified_v1.tensors import state_hash

    case = entry["case"]
    cid = str(case["case_id"])
    _validate_source_case(source_case, entry, torch_module)
    row = raw_row(source_case)

    request_path = _root_path(case["canonical_request_path"])
    request = _read_json(request_path)
    physical_ids = list(entry["physical_frame_ids"])
    expected_local = entry["physical_to_local"]
    input_width = int(source_case["input"]["width"])
    input_height = int(source_case["input"]["height"])
    decoded = _decode_canonical_frames(
        request, request_path, case, physical_ids, expected_local,
        expected_height=input_height, expected_width=input_width,
    )

    # The new physical window intentionally replaces the old sparse source
    # grid in both locations consumed by the native dataset/model path.
    new_row = _replace_row_frame_grid(
        row, physical_ids, str(_root_path(case["canonical_video_path"])),
        case["canonical_video_sha256"], input_width, input_height,
    )

    model_state_before = state_hash(model.state_dict())
    if model_state_before != case["source_model_state_sha256"]:
        raise ValueError(f"{cid}: model was not at the locked source checkpoint before capture")
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError(f"{cid}: source model is not frozen/eval")
    frames = decoded["frames"]
    batch, records, base, feature = _frozen_prefix_capture(
        model, frames, new_row, physical_ids, torch_module, frozen_forward,
    )
    model_state_after = state_hash(model.state_dict())
    if model_state_after != model_state_before or model_state_after != case["source_model_state_sha256"]:
        raise RuntimeError(f"{cid}: frozen source model state changed during feature capture")

    map_tensor = feature["maps"].detach().cpu().contiguous()
    if map_tensor.requires_grad or map_tensor.device.type != "cpu":
        raise RuntimeError(f"{cid}: output features are not detached CPU tensors")
    model_width = min(int(MODEL_INPUT_HEIGHT * input_width / input_height), int(MODEL_INPUT_HEIGHT * 1.4))
    feature_meta = {
        "schema": FEATURE_SCHEMA,
        "case_id": cid,
        "dataset": case["dataset"],
        "cohort": "historically_exposed_old8",
        "GT_read": False,
        "backward_calls": 0,
        "source_case_sha256": case["source_case_sha256"],
        "source_model_state_sha256": case["source_model_state_sha256"],
        "source_checkpoint_state_sha256": case["source_model_state_sha256"],
        "source_grid_frame_ids_sha256": case["source_grid_frame_ids_sha256"],
        "manifest_sha256": lock_context["manifest_sha256"],
        "manifest_case_sha256": _canonical_sha(case),
        "feature_lock_sha256": lock_context["lock_sha256"],
        "feature_runtime_sha256": lock_context["runtime_sha256"],
        "canonical_request_sha256": case["canonical_request_sha256"],
        "canonical_video_sha256": decoded["canonical_video_sha256"],
        "canonical_pixel_sha256_declared": decoded["canonical_pixel_sha256_declared"],
        "new_physical_frame_ids": list(physical_ids),
        "manifest_local_frame_ids": [int(expected_local[value]) for value in physical_ids],
        "frame_map_subset": [
            {key: value for key, value in row_meta.items() if key in {"physical_frame_id", "local_frame_index", "source_timestamp_s"}}
            for row_meta in decoded["per_frame"]
        ],
        "pixel_sha256": decoded["pixel_sha256"],
        "pixel_sha256_by_physical_frame": {str(item["physical_frame_id"]): item["rgb_sha256"] for item in decoded["per_frame"]},
        "decoded_dimensions_height_width": [input_height, input_width],
        "model_resize_height_width": [MODEL_INPUT_HEIGHT, model_width],
        "model_preprocess": "methods.decota_final_simplified_v1.backbone.make_batch; RGB uint8, bilinear antialias, ImageNet normalization",
        "native_offsets": feature["offset_frame_ids"],
        "feature_grid_size_by_offset": feature["grid_size_by_offset"],
        "prefix_dtype_by_offset": feature["prefix_dtype_by_offset"],
        "feature_map_layout": feature["layout"],
        "feature_maps_shape": feature["shape"],
        "feature_maps_dtype": feature["dtype"],
        "feature_maps_sha256": _tensor_sha256(map_tensor, torch_module),
        "canonical_decode_backend": decoded["backend"],
        "canonical_local_frames_decoded_from_zero_through_max_requested": decoded["decoded_local_frame_count"],
        "full_video_canonical_pixel_sha_recomputed": False,
        "old_sparse_source_native_outputs_compared": False,
        "candidate_proposals_created": False,
        "gradient_enabled_during_capture": False,
        "source_model_state_restored": True,
    }
    output = {
        "schema": FEATURE_SCHEMA,
        "metadata": feature_meta,
        "frame_ids": list(physical_ids),
        "appearance_maps": map_tensor,
    }
    del row, new_row, source_case, frames, decoded["frames"]
    return output


def _run_capture() -> dict[str, Any]:
    manifest, entries, lock_context = _preflight()
    fd = os.open(RUN_LOCK_PATH, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        stream.write(json.dumps({"pid": os.getpid(), "start_time": time.time(), "manifest_sha256": lock_context["manifest_sha256"]}) + "\n")
        stream.flush()
        os.fsync(stream.fileno())

    import gc
    import random
    import numpy as np
    import torch
    from scripts.run_tastvg_evidence_vulnerability_v2 import install_clean_loader
    from scripts.stvg_mllm_feedback_common_v1 import guard as no_gt_guard

    install_clean_loader()
    sys.addaudithook(no_gt_guard)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    torch.set_num_threads(2)
    torch.manual_seed(20261014)
    random.seed(20261014)
    np.random.seed(20261014)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_initialized() is False and not torch.cuda.is_available():
        raise RuntimeError("locked feature capture requires the root-provided CUDA runtime")

    records: list[dict[str, Any]] = []
    model = None
    loaded_source_key: tuple[str, str] | None = None
    total_model_offset_forwards = 0
    started = time.time()
    try:
        for cid in CASE_IDS:
            entry = entries[cid]
            case = entry["case"]
            source_path = _root_path(case["source_case_path"])
            if _sha_file(source_path) != case["source_case_sha256"]:
                raise ValueError(f"{cid}: source case SHA changed before loading")
            # The source field is the same model selector consumed by the
            # established model_for path; the source hash disambiguates it.
            import torch as torch_module
            source_case = torch_module.load(source_path, map_location="cpu", weights_only=False)
            _validate_source_case(source_case, entry, torch_module)
            source_name = str(source_case["source"])
            source_key = (source_name, str(case["source_model_state_sha256"]))
            if loaded_source_key != source_key:
                if model is not None:
                    del model
                    model = None
                    gc.collect()
                    torch.cuda.empty_cache()
                model = _load_model(source_name, case["source_model_state_sha256"], torch)
                loaded_source_key = source_key
            output = _capture_case(manifest, entry, lock_context, source_case, model, torch)
            del source_case
            pt_info = _write_once_torch(FEATURE_DIR / f"{cid}.pt", output, torch)
            receipt = {
                "status": "captured_source_only_feature_maps",
                "case_id": cid,
                "feature_file_sha256": pt_info["sha256"],
                "feature_file_bytes": pt_info["bytes"],
                "source_case_sha256": case["source_case_sha256"],
                "source_model_state_sha256": case["source_model_state_sha256"],
                "manifest_sha256": lock_context["manifest_sha256"],
                "GT_read": False,
                "backward_calls": 0,
            }
            receipt_info = _write_once_json((FEATURE_DIR / f"{cid}.pt").with_suffix(".json"), receipt)
            records.append({"case_id": cid, **pt_info, "receipt": receipt_info})
            total_model_offset_forwards += 2
            del output
            gc.collect()
            torch.cuda.empty_cache()
            if torch.cuda.is_initialized():
                torch.cuda.synchronize()
            print(f"captured short-window frozen features {cid} ({len(records)}/8)", flush=True)
        final_state_sha = str(getattr(model, "_fixed_full_state_hash", ""))
        if not model.training and all(not parameter.requires_grad for parameter in model.parameters()):
            pass
        else:
            raise RuntimeError("source model ended capture in a non-frozen state")
        from methods.decota_final_simplified_v1.tensors import state_hash
        if state_hash(model.state_dict()) != final_state_sha:
            raise RuntimeError("source model state differs from its locked state after final capture")
    finally:
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    if len(records) != 8:
        raise RuntimeError("feature capture did not complete all fixed old-eight cases")
    barrier = {
        "schema": "stvg-short-window-feature-capture-barrier-v1",
        "status": "sealed",
        "scope": "fixed historically exposed old-eight frozen source feature capture on four +/-2 windows",
        "count": 8,
        "case_ids": list(CASE_IDS),
        "center_count": 32,
        "features_case_count": len(records),
        "new_physical_frame_count": sum(len(entry["physical_frame_ids"]) for entry in entries.values()),
        "feature_offset_forwards": total_model_offset_forwards,
        "source_case_sha256_by_case": {cid: entries[cid]["case"]["source_case_sha256"] for cid in CASE_IDS},
        "source_model_state_sha256_by_case": {cid: entries[cid]["case"]["source_model_state_sha256"] for cid in CASE_IDS},
        "manifest_sha256": lock_context["manifest_sha256"],
        "feature_lock_sha256": lock_context["lock_sha256"],
        "feature_runtime_sha256": lock_context["runtime_sha256"],
        "worker_sha256": lock_context["worker_sha256"],
        "cpu_contracts_sha256": lock_context["contract_sha256"],
        "files": {row["case_id"]: {key: row[key] for key in ("path", "sha256", "bytes", "receipt")} for row in records},
        "records": records,
        "GT_read": False,
        "labels_read": False,
        "backward_calls": 0,
        "candidate_proposals_created": False,
        "old_source_native_outputs_compared": False,
        "full_video_canonical_pixel_sha_recomputed": False,
        "source_model_state_restored": True,
        "CPU_only": False,
        "GPU_used": True,
        "started_unix_time": started,
        "finished_unix_time": time.time(),
        "worker_wall_seconds": time.time() - started,
    }
    info = _write_once_json(BARRIER_PATH, barrier)
    return {"barrier": info, "cases": len(records)}


def run_cpu_contracts() -> dict[str, Any]:
    import torch

    before_cuda = torch.cuda.is_initialized()
    if before_cuda:
        raise RuntimeError("CPU contracts must run before CUDA initialization")

    def make_view(ids: list[int], offset: int):
        nf, h, w = len(ids), 2, 2
        # First h*w tokens have a unique offset/frame/channel encoding;
        # extra tokens must not leak into the appearance feature map.
        prefix = torch.arange((h * w + 3) * nf * FEATURE_CHANNELS, dtype=torch.float32)
        prefix = prefix.reshape(h * w + 3, nf, FEATURE_CHANNELS) + 100000.0 * offset
        return {"prefix": prefix, "info": {"fea_map_size": [h, w]}}

    views = [make_view([10, 12], 0), make_view([11], 1)]
    records = [
        {"offset": 0, "frame_ids": [10, 12]},
        {"offset": 1, "frame_ids": [11]},
    ]
    result = extract_appearance_maps(lambda value: value, views, records, [10, 11, 12], torch)
    assert result["frame_ids"] == [10, 11, 12]
    assert result["maps"].shape == (3, FEATURE_CHANNELS, 2, 2)
    assert result["maps"].device.type == "cpu" and not result["maps"].requires_grad
    expected0 = views[0]["prefix"][:4].permute(1, 2, 0).reshape(2, FEATURE_CHANNELS, 2, 2)
    expected1 = views[1]["prefix"][:4].permute(1, 2, 0).reshape(1, FEATURE_CHANNELS, 2, 2)
    assert torch.equal(result["maps"][0], expected0[0])
    assert torch.equal(result["maps"][1], expected1[0])
    assert torch.equal(result["maps"][2], expected0[1])
    assert result["offset_frame_ids"] == {"0": [10, 12], "1": [11]}
    fake_row = {"frame_ids": [7, 9], "input": {"frame_ids": [7, 9], "width": 4, "height": 2}}
    replaced = _replace_row_frame_grid(fake_row, [20, 21, 22, 23], "/canonical/video.mkv", "3" * 64, 4, 2)
    assert replaced["frame_ids"] == replaced["input"]["frame_ids"] == [20, 21, 22, 23]
    assert fake_row["frame_ids"] == [7, 9] and fake_row["input"]["frame_ids"] == [7, 9]

    mock_result = _mock_full_capture_contract(torch)

    def must_reject(call):
        try:
            call()
        except (ValueError, KeyError):
            return
        raise AssertionError("malformed feature contract input was accepted")

    must_reject(lambda: extract_appearance_maps(lambda x: x, views, records, [10, 11, 13], torch))
    must_reject(lambda: extract_appearance_maps(lambda x: x, views, [records[0], {"offset": 0, "frame_ids": [11]}], [10, 11, 12], torch))
    bad_view = {"prefix": views[0]["prefix"], "info": {"fea_map_size": [0, 2]}}
    must_reject(lambda: extract_appearance_maps(lambda x: x, [bad_view, views[1]], records, [10, 11, 12], torch))
    assert torch.cuda.is_initialized() is False
    return {
        "status": "passed",
        "checks": {
            "physical_id_merge_sorted": True,
            "native_two_offset_partition_exact": True,
            "spatial_prefix_slice_and_layout_exact": True,
            "extra_prefix_tokens_excluded": True,
            "output_detached_cpu": True,
            "both_model_row_frame_id_fields_replaced_without_mutating_source": True,
            "missing_or_extra_frame_rejected": True,
            "duplicate_offset_rejected": True,
            "invalid_feature_grid_rejected": True,
            "cuda_initialized": False,
            "full_source_load_to_feature_barrier_mock": mock_result["synthetic_full_capture_path"],
        },
        "GT_read": False,
        "real_RGB_read": False,
        "model_loaded": False,
        "GPU_used": False,
        "mock_full_capture": mock_result,
    }


def write_cpu_contracts() -> dict[str, Any]:
    if LOCK_PATH.exists() or RUNTIME_PATH.exists():
        raise RuntimeError("feature runtime is already locked; do not rewrite CPU contracts after freeze")
    result = run_cpu_contracts()
    worker_sha = _sha_file(Path(__file__).resolve())
    contract = {
        "schema": CONTRACT_SCHEMA,
        "status": result["status"],
        "scope": "synthetic CPU verification of short-window appearance map ABI only",
        "worker_sha256": worker_sha,
        "feature_extraction": {
            "source": "methods.decota_final_simplified_v1.replay.SpatialReplay.reference_values",
            "formula": "H=norm(prefix); fa=H[:h*w].permute(1,2,0).reshape(nf,256,h,w)",
            "merge_key": "physical frame ID from the two original native offset records",
            "saved_order": "ascending requested physical frame ID",
        },
        "expected_run_scope": {
            "case_ids": list(CASE_IDS),
            "centers_per_case": 4,
            "max_unique_frames_per_case": MAX_FRAMES_PER_CASE,
            "source_model_frozen": True,
            "native_offset_forwards_per_case": 2,
            "backward_calls": 0,
            "GT_read": False,
            "candidate_proposals_created": False,
            "compared_with_old_sparse_native_output": False,
        },
        "checks": result["checks"],
        "synthetic_only": True,
        "GT_read": False,
        "real_RGB_read": False,
        "model_loaded": False,
        "GPU_used": False,
    }
    info = _write_once_json(CONTRACT_PATH, contract)
    return {"contract": info, **result}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("cpu-contracts", "write-cpu-contracts", "capture"))
    args = parser.parse_args(argv)
    if args.mode == "cpu-contracts":
        print(json.dumps(run_cpu_contracts(), sort_keys=True))
        return 0
    if args.mode == "write-cpu-contracts":
        print(json.dumps(write_cpu_contracts(), sort_keys=True))
        return 0
    print(json.dumps(_run_capture(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
