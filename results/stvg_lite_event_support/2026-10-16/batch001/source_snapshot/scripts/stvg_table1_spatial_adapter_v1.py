#!/usr/bin/env python3
"""Table 1 Spatial-A adapter over the frozen native-instance reference and OPD.

The adapter keeps reference construction separate from the full-model caller.
It only consumes the caller's already captured source projection, canonical
frame map, selected object ID, and the filtered SAM2 RLE inventory.  It never
opens GT annotations, media frames, or source checkpoints.

Reference geometry is the same selected-ID mask geometry used by the locked
WHO/WHEN-decoupled preparation: four nearest-unique source-grid centers in the
QA half-open interval; each anchor uses its clipped center +/- 2 window;
training anchors require a visible center mask and at least one valid adjacent
same-ID pair.  Fit weights are presence fractions, as specified for this arm.
The teacher-geometry arm uses the same selected ID on the full source grid and
uses exact Frozen boxes wherever that ID has no mask.  It does not interpolate
or substitute a nearby frame.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import prepare_stvg_instance_decoupled_references_v1 as refs
from scripts import prepare_stvg_short_event_tracklets_v1 as center_policy

SHA_RE = re.compile(r"[0-9a-f]{64}\Z")


def _as_numpy(value: Any):
    import numpy as np

    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _integer_sequence(values: Any, name: str) -> list[int]:
    import numpy as np

    array = _as_numpy(values)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional sequence")
    result = []
    for value in array.tolist():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must contain integer-valued entries")
        if not math.isfinite(float(value)) or int(value) != value:
            raise ValueError(f"{name} must contain integer-valued entries")
        result.append(int(value))
    if result != sorted(set(result)):
        raise ValueError(f"{name} must be strictly increasing and unique")
    return result


def _frame_map(request: Mapping[str, Any]) -> tuple[list[dict[str, int]], dict[int, int]]:
    raw = request.get("frame_map")
    count = request.get("local_frame_count")
    if not isinstance(raw, list) or isinstance(count, bool) or not isinstance(count, int):
        raise ValueError("canonical request must contain frame_map and local_frame_count")
    if len(raw) != count or count <= 0:
        raise ValueError("canonical frame_map length differs from local_frame_count")
    rows = []
    by_physical = {}
    for expected_local, row in enumerate(raw):
        if not isinstance(row, Mapping):
            raise ValueError("canonical frame_map rows must be mappings")
        local = row.get("local_frame_index")
        physical = row.get("physical_frame_id")
        if (isinstance(local, bool) or not isinstance(local, int) or local != expected_local
                or isinstance(physical, bool) or not isinstance(physical, int) or physical < 0):
            raise ValueError("canonical local/physical frame binding is malformed")
        if physical in by_physical:
            raise ValueError("canonical physical frame IDs must be unique")
        by_physical[physical] = local
        rows.append({"local_frame_index": local, "physical_frame_id": physical})
    if [row["physical_frame_id"] for row in rows] != sorted(by_physical):
        raise ValueError("canonical physical frame map must be strictly increasing")
    return rows, by_physical


def _source_projection(source_case: Mapping[str, Any]):
    """Read only the small source projection needed by this CPU adapter."""
    import numpy as np

    frame_ids = _integer_sequence(source_case.get("frame_ids"), "source_case.frame_ids")
    width = source_case.get("width")
    height = source_case.get("height")
    if (isinstance(width, bool) or not isinstance(width, int) or width <= 0
            or isinstance(height, bool) or not isinstance(height, int) or height <= 0):
        raise ValueError("source_case must expose positive original width and height")
    boxes = source_case.get("native_boxes_xyxy_normalized")
    array = np.asarray(_as_numpy(boxes), dtype=np.float64)
    if array.shape != (len(frame_ids), 4) or not np.isfinite(array).all():
        raise ValueError("source_case.native_boxes_xyxy_normalized must be finite N x 4")
    if not (np.all(array[:, 0] >= 0) and np.all(array[:, 1] >= 0)
            and np.all(array[:, 2] <= 1) and np.all(array[:, 3] <= 1)
            and np.all(array[:, 0] < array[:, 2]) and np.all(array[:, 1] < array[:, 3])):
        raise ValueError("source Frozen boxes must be legal normalized xyxy")
    return frame_ids, int(width), int(height), array.tolist()


def _qa_interval(value: Any) -> tuple[int, int]:
    if (not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2
            or any(isinstance(x, bool) or not isinstance(x, int) for x in value)):
        raise ValueError("QA interval must be an integer physical-frame half-open pair")
    start, end = int(value[0]), int(value[1])
    if start < 0 or end <= start:
        raise ValueError("QA interval must be nonempty and nonnegative")
    return start, end


def _fixed_centers(
    interval: tuple[int, int], frame_ids: Sequence[int], request_rows: Sequence[Mapping[str, int]],
) -> tuple[list[dict[str, Any]], str | None]:
    """Use the original nearest-unique four-bin source-grid center rule."""
    try:
        selected = center_policy._nearest_unique_grid_centers(interval, frame_ids)
    except ValueError as exc:
        return [], "qa_interval_cannot_supply_four_unique_centers:" + str(exc)
    physical_to_local = {int(row["physical_frame_id"]): int(row["local_frame_index"])
                         for row in request_rows}
    position_by_physical = {int(physical): pos for pos, physical in enumerate(frame_ids)}
    result = []
    local_count = len(request_rows)
    for item in selected:
        physical = int(item["center_physical_frame_id"])
        if physical not in physical_to_local or physical not in position_by_physical:
            raise ValueError("snapped QA center is not bound in source grid and canonical request")
        local = physical_to_local[physical]
        local_window = list(range(max(0, local - 2), min(local_count, local + 3)))
        if len(local_window) < 1 or local not in local_window:
            raise ValueError("fixed center +/- 2 window is malformed")
        result.append({
            "slot": int(item["slot"]),
            "target_physical_frame_id": float(item["target_physical_frame_id"]),
            "center_physical_frame_id": physical,
            "source_grid_position": int(position_by_physical[physical]),
            "center_local_frame_index": int(local),
            "local_frame_indices": local_window,
        })
    if len(result) != 4 or [row["slot"] for row in result] != [0, 1, 2, 3]:
        raise ValueError("original QA center rule did not return exactly four ordered slots")
    return result, None


def _safe_inventory_files(inventory: Mapping[str, Any], request_count: int, root: Path) -> dict[int, dict[str, Any]]:
    if inventory.get("GT_read") is not False:
        raise ValueError("filtered mask inventory must be explicitly GT-free")
    raw_files = inventory.get("files")
    if not isinstance(raw_files, list):
        raise ValueError("filtered mask inventory must contain its files list")
    by_index = {}
    root_resolved = root.resolve()
    for row in raw_files:
        if not isinstance(row, Mapping):
            raise ValueError("mask inventory file rows must be mappings")
        index, relpath, digest, byte_count = row.get("index"), row.get("path"), row.get("sha256"), row.get("bytes")
        if (isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < request_count
                or not isinstance(relpath, str) or not relpath
                or not isinstance(digest, str) or not SHA_RE.fullmatch(digest)
                or isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count <= 0):
            raise ValueError("mask inventory row has invalid index/path/hash/size")
        if index in by_index:
            raise ValueError("mask inventory contains duplicate local-frame indices")
        rel = Path(relpath)
        if rel.is_absolute() or any(part in {"", ".", ".."} for part in rel.parts):
            raise ValueError("mask inventory path must be a safe project-relative path")
        path = (root_resolved / rel).resolve(strict=True)
        if not path.is_file() or not path.is_relative_to(root_resolved):
            raise ValueError("mask inventory path escapes project root or is not a file")
        by_index[index] = {"path": path, "sha256": digest, "bytes": byte_count, "relative_path": relpath}
    if set(by_index) != set(range(request_count)):
        raise ValueError("filtered mask inventory must bind every canonical local frame exactly once")
    if inventory.get("mask_count") != request_count:
        raise ValueError("filtered mask inventory count differs from canonical frame count")
    return by_index


def _read_selected_masks(
    selected_id: int,
    needed_local_indices: set[int],
    inventory_files: Mapping[int, Mapping[str, Any]],
    width: int,
    height: int,
):
    """Decode only the selected filtered object ID on required local frames."""
    import numpy as np
    runtime_root = ROOT / "external/stvg_runtime"
    if not runtime_root.is_dir():
        raise RuntimeError("the repository-pinned CPU RLE decoder is unavailable")
    if str(runtime_root) not in sys.path:
        sys.path.insert(0, str(runtime_root))
    from pycocotools import mask as mask_utils
    from scripts.stvg_r1_native_teacher_v1 import bbox_from_mask_inclusive
    from scripts.prepare_stvg_native_instance_references_v1 import _xyxy_inclusive_to_normalized

    rows: dict[int, dict[str, Any]] = {}
    masks_by_local: dict[int, Any] = {}
    for local in sorted(needed_local_indices):
        entry = inventory_files[int(local)]
        data = entry["path"].read_bytes()
        if len(data) != int(entry["bytes"]) or hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise RuntimeError("filtered RLE file differs from its locked inventory digest")
        decoded_json = json.loads(data)
        objects = decoded_json.get("objects")
        if not isinstance(objects, Mapping):
            raise ValueError("filtered RLE JSON has no objects mapping")
        object_row = objects.get(str(selected_id))
        mask = None
        box = None
        area = 0
        if object_row is not None:
            if not isinstance(object_row, Mapping) or not isinstance(object_row.get("rle"), Mapping):
                raise ValueError("selected filtered ID row has no RLE mapping")
            mask = np.asarray(mask_utils.decode(object_row["rle"]))
            if mask.ndim == 3 and mask.shape[-1] == 1:
                mask = mask[..., 0]
            if mask.ndim != 2 or mask.shape != (height, width):
                raise ValueError("selected filtered RLE dimensions differ from source-case dimensions")
            box, area = bbox_from_mask_inclusive(mask)
            if box is not None:
                xyxy, cxcywh = _xyxy_inclusive_to_normalized(box, width, height)
                masks_by_local[int(local)] = mask.astype(bool, copy=False)
            else:
                xyxy = cxcywh = None
        else:
            xyxy = cxcywh = None
        rows[int(local)] = {
            "local_frame_index": int(local),
            "xyxy_inclusive_pixels": box,
            "xyxy_normalized_continuous": xyxy,
            "cxcywh_normalized_continuous": cxcywh,
            "mask_area_px": int(area),
            "present": box is not None,
        }
    return rows, masks_by_local


def build_reference(
    source_case: Mapping[str, Any],
    qa_interval: Sequence[int],
    canonical_request: Mapping[str, Any],
    selected_object_id: int | None,
    filtered_rle_inventory: Mapping[str, Any],
    *,
    root: str | Path = ROOT,
) -> dict[str, Any]:
    """Build fixed-center presence anchors and a Frozen-fallback geometry arm.

    ``source_case`` is a lightweight caller projection, not a path or loaded
    pickle. It must contain ``frame_ids``, ``width``, ``height``, and
    ``native_boxes_xyxy_normalized``. This boundary avoids reading source RGB
    from the source cache. A selected ID is never inferred here.
    """
    import numpy as np

    case_id = source_case.get("case_id")
    dataset = source_case.get("dataset")
    if not isinstance(case_id, str) or not case_id or not isinstance(dataset, str) or not dataset:
        raise ValueError("source_case must bind case_id and dataset")
    frame_ids, width, height, frozen_boxes = _source_projection(source_case)
    interval = _qa_interval(qa_interval)
    request_rows, by_physical = _frame_map(canonical_request)
    request_count = len(request_rows)
    if any(frame_id not in by_physical for frame_id in frame_ids):
        raise ValueError("every source-grid physical frame must occur in canonical_request.frame_map")

    centers, center_error = _fixed_centers(interval, frame_ids, request_rows)
    inventory_files = _safe_inventory_files(filtered_rle_inventory, request_count, Path(root))
    known_raw = filtered_rle_inventory.get("known_ids")
    if not isinstance(known_raw, list):
        raise ValueError("filtered mask inventory must expose known_ids")
    known_ids = {int(value) for value in known_raw
                 if isinstance(value, int) and not isinstance(value, bool) and value > 0}
    if len(known_ids) != len(known_raw):
        raise ValueError("filtered mask inventory known_ids must contain unique positive integers")

    syntactic_id = (isinstance(selected_object_id, int) and not isinstance(selected_object_id, bool)
                    and selected_object_id > 0)
    selected_id = int(selected_object_id) if syntactic_id else None
    id_in_inventory = selected_id is not None and selected_id in known_ids
    center_windows = [local for center in centers for local in center["local_frame_indices"]]
    qa_locals = [row["local_frame_index"] for row in request_rows
                 if interval[0] <= row["physical_frame_id"] < interval[1]]
    source_grid_locals = [by_physical[physical] for physical in frame_ids]
    needed = set(center_windows) | set(qa_locals) | set(source_grid_locals)

    frames_by_local: dict[int, dict[str, Any]] = {}
    masks_by_local: dict[int, Any] = {}
    if id_in_inventory and selected_id is not None:
        frames_by_local, masks_by_local = _read_selected_masks(
            selected_id, needed, inventory_files, width, height
        )
    else:
        frames_by_local = {
            local: {"local_frame_index": local, "xyxy_inclusive_pixels": None,
                    "xyxy_normalized_continuous": None, "cxcywh_normalized_continuous": None,
                    "mask_area_px": 0, "present": False}
            for local in needed
        }

    if centers and id_in_inventory and selected_id is not None:
        raw_anchors, supported_legacy, center_audit = refs._make_anchor_sets(
            centers, frames_by_local, masks_by_local, selected_id
        )
        supported_anchors = []
        for anchor in supported_legacy:
            adjusted = copy.deepcopy(anchor)
            adjusted["weight"] = float(adjusted["reliability"]["presence_fraction"])
            supported_anchors.append(adjusted)
    else:
        raw_anchors, supported_anchors, center_audit = [], [], []
        for center in centers:
            center_audit.append({
                "slot": int(center["slot"]),
                "position": int(center["source_grid_position"]),
                "physical_frame_id": int(center["center_physical_frame_id"]),
                "local_frame_index": int(center["center_local_frame_index"]),
                "local_frame_indices": list(center["local_frame_indices"]),
                "center_mask_present": False,
                "presence_fraction": 0.0,
                "valid_adjacent_pair_count": 0,
                "mean_adjacent_mask_iou": 0.0,
                "legacy_weight": 0.0,
                "raw_anchor_created": False,
                "presence_supported_anchor_created": False,
            })

    qa_mask_present = any(
        interval[0] <= int(request_rows[local]["physical_frame_id"]) < interval[1]
        and frames_by_local.get(local, {}).get("xyxy_normalized_continuous") is not None
        for local in qa_locals
    )
    raw_geometry_admitted = bool(id_in_inventory and qa_mask_present and raw_anchors)
    # The lock defines four potential center slots. Missing masks can leave
    # one, two, or three supported anchors; the original A fit accepts any
    # nonempty set. Do not turn the four-slot sampling rule into a four-visible
    # anchor eligibility gate.
    no_update = (center_error is not None or selected_id is None or not id_in_inventory
                 or len(supported_anchors) == 0)
    if center_error is not None:
        no_update_reason = center_error
    elif selected_id is None:
        no_update_reason = "selected_object_id_missing_or_invalid"
    elif not id_in_inventory:
        no_update_reason = "selected_object_id_absent_from_filtered_mask_inventory"
    elif len(supported_anchors) == 0:
        no_update_reason = "no_presence_supported_anchor_at_fixed_centers"
    else:
        no_update_reason = None

    teacher_boxes_by_physical = {
        int(physical): frames_by_local.get(by_physical[int(physical)], {}).get("xyxy_normalized_continuous")
        for physical in frame_ids
    }
    if not id_in_inventory:
        geometry_abstain_reason = ("candidate_id_invalid" if selected_id is None
                                   else "candidate_id_absent_from_mask_inventory")
    elif not qa_mask_present:
        geometry_abstain_reason = "selected_id_absent_in_QA_interval"
    elif not raw_anchors:
        geometry_abstain_reason = "no_selected_mask_at_fixed_centers"
    else:
        geometry_abstain_reason = no_update_reason or "selected_id_not_admitted_in_fixed_QA"
    geometry = refs._geometry_arm(
        frame_ids=frame_ids,
        frozen_boxes=frozen_boxes,
        teacher_boxes_by_physical=teacher_boxes_by_physical,
        candidate_id=selected_id,
        admitted=raw_geometry_admitted,
        qa_interval=interval,
        abstain_reason=geometry_abstain_reason,
    )
    geometry["qa_center_slots"] = [int(row["slot"]) for row in centers]
    geometry["geometry_selected_id"] = selected_id if raw_geometry_admitted else None
    qa_selected_mask_count = sum(
        frames_by_local.get(local, {}).get("xyxy_normalized_continuous") is not None
        for local in qa_locals
    )
    source_grid_selected_mask_count = sum(
        frames_by_local.get(local, {}).get("xyxy_normalized_continuous") is not None
        for local in source_grid_locals
    )
    return {
        "schema": "stvg-table1-spatial-a-reference-v1",
        "case_id": case_id,
        "dataset": dataset,
        "qa_interval_frame_bounds_half_open": list(interval),
        "selected_object_id": selected_id,
        "selected_id_in_inventory": bool(id_in_inventory),
        "centers": centers,
        "center_audit": center_audit,
        "anchors": supported_anchors,
        "anchor_count": len(supported_anchors),
        "no_update": bool(no_update),
        "no_update_reason": no_update_reason,
        "teacher_geometry": geometry,
        "qa_selected_mask_count": int(qa_selected_mask_count),
        "source_grid_selected_mask_count": int(source_grid_selected_mask_count),
        "rle_inventory_files_sha_verified": len(needed) if id_in_inventory else 0,
        "geometry_is_full_source_grid": True,
        "missing_mask_policy": "exact_frozen_source_box_per_missing_source_grid_frame",
        "interpolation_used": False,
        "nearest_frame_replacement_used": False,
        "GT_read": False,
    }


def _anchor_tensors(reference: Mapping[str, Any], frame_ids: Sequence[int], torch):
    anchors = []
    positions_by_physical = {int(physical): pos for pos, physical in enumerate(frame_ids)}
    for row in reference.get("anchors", []):
        physical = int(row["physical_frame_id"])
        position = int(row["position"])
        if position < 0 or position >= len(frame_ids) or int(frame_ids[position]) != physical:
            raise ValueError("anchor source-grid position and physical ID disagree")
        if positions_by_physical.get(physical) != position:
            raise ValueError("anchor has an ambiguous source-grid binding")
        raw = torch.tensor(row["box_cxcywh"], dtype=torch.float64)
        if raw.shape != (4,) or not torch.isfinite(raw).all():
            raise ValueError("anchor cxcywh must be finite width four")
        cx, cy, width, height = map(float, raw.tolist())
        if not (0 <= cx - width / 2 < cx + width / 2 <= 1
                and 0 <= cy - height / 2 < cy + height / 2 <= 1):
            raise ValueError("anchor cxcywh is not a legal normalized box")
        weight = float(row["weight"])
        presence = float(row["reliability"]["presence_fraction"])
        if not math.isfinite(weight) or not (0 < weight <= 1) or weight != presence:
            raise ValueError("Spatial-A fit weights must equal the presence fraction")
        anchors.append({"position": position, "physical_frame_id": physical,
                        "box": raw.to(dtype=torch.float32), "weight": weight})
    return anchors


def _assert_readout_matches_fit(readout: Any, fit: Mapping[str, Any]) -> None:
    import numpy as np

    if isinstance(readout, Mapping):
        boxes = readout.get("boxes")
        if boxes is None:
            boxes = readout.get("boxes_xyxy_normalized")
    else:
        boxes = readout
    if boxes is None:
        raise ValueError("full-readout callback must return boxes or a boxes mapping")
    actual = _as_numpy(boxes)
    expected = _as_numpy(fit["final_cached_decoder_boxes"])
    if actual.shape != expected.shape or not np.array_equal(actual, expected):
        raise RuntimeError("full-model final readout differs from the fit cached decoder boxes")


def fit_spatial_a(
    *,
    base: Any,
    initial: Mapping[str, Any],
    reference: Mapping[str, Any],
    frame_ids: Sequence[int],
    key: str,
    config: Mapping[str, Any],
    reference_sha256: str,
    readout_callback: Callable[[Mapping[str, Any]], Any] | None,
    checkpoint_callback: Callable[[Mapping[str, Any]], Any] | None = None,
) -> dict[str, Any]:
    """Run original validated IoU OPD and delegate the unchanged full readout.

    ``readout_callback`` receives the frozen helper's complete fit result and
    must perform the caller's ordinary source-reset full-model decode.  It is
    skipped for exact no-update.  No new optimizer, loss, sampling, or step
    selection is introduced here.
    """
    if key != "caption_query_fixed_space|" + str(reference.get("case_id")):
        raise ValueError("Spatial-A must preserve the original per-case RNG key")
    if int(config.get("steps", -1)) != 5:
        raise ValueError("Spatial-A requires the locked original five OPD steps")
    if not isinstance(reference_sha256, str) or not SHA_RE.fullmatch(reference_sha256):
        raise ValueError("reference_sha256 must be lowercase SHA-256")
    ids = _integer_sequence(frame_ids, "frame_ids")
    if bool(reference.get("no_update")):
        if reference.get("anchors"):
            raise ValueError("no_update reference cannot contain fit anchors")
        return {
            "schema": "stvg-table1-spatial-a-fit-v1",
            "algorithm": "validated_loss_weighted_iou",
            "status": "exact_no_update",
            "fit": None,
            "readout": None,
            "actual_fit_count": 0,
            "actual_final_redecode_calls": 0,
            "optimizer_steps": 0,
            "proposed_optimizer_steps": 0,
            "no_update_reason": reference.get("no_update_reason"),
            "GT_read": False,
        }
    anchors = _anchor_tensors(reference, ids, __import__("torch"))
    centers = reference.get("centers", [])
    if len(centers) != 4 or [int(row["slot"]) for row in centers] != [0, 1, 2, 3]:
        raise ValueError("updating Spatial-A must preserve the four fixed QA center slots")
    if not 1 <= len(anchors) <= 4:
        raise ValueError("an updating Spatial-A reference must contain one to four supported anchors")
    if readout_callback is None:
        raise ValueError("updating Spatial-A requires the caller's full-model readout callback")
    from scripts.stvg_instance_validated_fit_v1 import fit_validated

    fit = fit_validated(
        base, initial, anchors, ids, key, dict(config), validate=True,
        reference_sha256=reference_sha256, checkpoint_callback=checkpoint_callback,
    )
    readout = readout_callback(fit)
    _assert_readout_matches_fit(readout, fit)
    return {
        "schema": "stvg-table1-spatial-a-fit-v1",
        "algorithm": "validated_loss_weighted_iou",
        "status": "fit_and_full_readout",
        "fit": fit,
        "readout": readout,
        "actual_fit_count": 1,
        "actual_final_redecode_calls": 1,
        "optimizer_steps": int(fit["optimizer_steps"]),
        "proposed_optimizer_steps": int(fit["proposed_optimizer_steps"]),
        "no_update_reason": None,
        "GT_read": False,
    }


def _contract_synthetic() -> dict[str, Any]:
    import numpy as np
    import torch

    # Four target bins snap to the original nearest-unique grid centers.
    centers = center_policy._nearest_unique_grid_centers([0, 8], list(range(8)))
    assert [r["center_physical_frame_id"] for r in centers] == [1, 3, 5, 7]
    short, err = _fixed_centers([0, 3], list(range(8)),
                                [{"local_frame_index": i, "physical_frame_id": i} for i in range(8)])
    assert short == [] and err is not None

    masks = {
        0: np.array([[1, 1, 0], [1, 0, 0]], dtype=bool),
        1: np.array([[1, 1, 0], [0, 0, 0]], dtype=bool),
        2: np.array([[1, 0, 0], [0, 0, 0]], dtype=bool),
        3: np.array([[0, 0, 1], [0, 0, 1]], dtype=bool),
        4: np.array([[0, 1, 1], [0, 0, 1]], dtype=bool),
    }
    toy_centers = [{"slot": 0, "source_grid_position": 1, "center_physical_frame_id": 1,
                    "center_local_frame_index": 1, "local_frame_indices": [0, 1, 2]}]
    frames = {}
    for local, mask in masks.items():
        ys, xs = np.nonzero(mask)
        if len(xs):
            b = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
            xy, cx = refs.oldprep._xyxy_inclusive_to_normalized(b, 3, 2)
        else:
            xy = cx = None
        frames[local] = {"local_frame_index": local, "xyxy_normalized_continuous": xy,
                         "cxcywh_normalized_continuous": cx}
    raw, supported, audit = refs._make_anchor_sets(toy_centers, frames, masks, 7)
    assert len(raw) == 1 and len(supported) == 1 and audit[0]["valid_adjacent_pair_count"] == 2
    weighted = copy.deepcopy(supported[0])
    weighted["weight"] = weighted["reliability"]["presence_fraction"]
    assert weighted["weight"] == 1.0 and weighted["weight"] != supported[0]["reliability"]["mean_adjacent_mask_iou"]

    frozen = [[.1, .1, .2, .2], [.2, .2, .4, .4], [.3, .3, .5, .5]]
    geometry = refs._geometry_arm(frame_ids=[10, 11, 12], frozen_boxes=frozen,
        teacher_boxes_by_physical={10: None, 11: [.4, .4, .6, .6], 12: None},
        candidate_id=7, admitted=True, qa_interval=(0, 20), abstain_reason="missing")
    assert geometry["boxes_xyxy_normalized"] == [frozen[0], [.4, .4, .6, .6], frozen[2]]
    assert geometry["fallback_used"] == [True, False, True]
    assert geometry["interpolation_used"] is False and geometry["nearest_frame_replacement_used"] is False

    # Exact no-update cannot invoke fitting or the readout callback.
    called = {"readout": 0}
    no_update_ref = {"case_id": "synthetic", "no_update": True,
                     "no_update_reason": "no_presence_supported_anchor_at_fixed_centers", "anchors": []}
    result = fit_spatial_a(base=None, initial={}, reference=no_update_ref, frame_ids=list(range(8)),
        key="caption_query_fixed_space|synthetic", config={"steps": 5}, reference_sha256="a"*64,
        readout_callback=lambda fit: called.__setitem__("readout", called["readout"] + 1))
    assert result["status"] == "exact_no_update" and result["actual_fit_count"] == 0
    assert result["actual_final_redecode_calls"] == 0 and called["readout"] == 0

    # Exercise the real locked helper on its synthetic CPU student base and
    # the adapter's callback/equality seam; no full model or source data used.
    from scripts import stvg_target_teacher_student_v1 as core
    base = core._synthetic_student_base(8)
    cfg = core.load_original_opd_config("hc2")
    cfg["steps"] = 5
    update_anchors = []
    for position in (1,):
        box = base.zero["boxes"][position].detach().cpu().to(torch.float64)
        update_anchors.append({
            "slot": len(update_anchors), "position": position,
            "physical_frame_id": position, "box_cxcywh": box.tolist(),
            "weight": 1.0,
            "reliability": {"presence_fraction": 1.0},
        })
    fit_reference = {"case_id": "synthetic", "no_update": False,
                     "anchors": update_anchors, "no_update_reason": None,
                     "centers": [{"slot": slot} for slot in range(4)]}
    callback_calls = []
    fitted = fit_spatial_a(
        base=base,
        initial={key: value.detach().clone() for key, value in base.initial.items()},
        reference=fit_reference,
        frame_ids=list(range(8)),
        key="caption_query_fixed_space|synthetic",
        config=cfg,
        reference_sha256="a"*64,
        readout_callback=lambda fit: callback_calls.append(fit) or {"boxes": fit["final_cached_decoder_boxes"]},
    )
    assert fitted["status"] == "fit_and_full_readout" and fitted["actual_fit_count"] == 1
    assert fitted["actual_final_redecode_calls"] == 1 and len(callback_calls) == 1
    assert fitted["fit"]["optimizer_steps"] <= 5
    assert not torch.cuda.is_initialized()
    return {"center_rule": "pass", "short_qa_no_update": "pass",
            "presence_fraction_fit_weight": "pass", "missing_mask_frozen_fallback": "pass",
            "no_update_skips_fit_and_readout": "pass",
            "validated_fit_delegation_and_readout_callback": "pass",
            "GT_read": False, "CUDA_initialized": False}


def _contract_original_eight() -> dict[str, Any]:
    """Rebuild the eight old references from locked RLEs without loading video or source PTs."""
    from scripts import stvg_native_instance_common_v1 as common

    old_ref_path = ROOT / "artifacts/stvg_instance_identity_update_v1/REFERENCE_LOCK.json"
    old_geo_path = ROOT / "artifacts/stvg_instance_identity_update_v1/TEACHER_GEOMETRY_ONLY_READOUT.json"
    roster_path = ROOT / "artifacts/stvg_native_instance_spatial_opd_v1/ROSTER_LOCK.json"
    for path in (old_ref_path, old_geo_path, roster_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    old_ref = common.read(old_ref_path)
    old_geo = common.read(old_geo_path)
    roster = common.read(roster_path)
    tracking_barrier_path = ROOT / "artifacts/stvg_native_instance_spatial_opd_v1/TRACKING_GLOBAL_BARRIER.json"
    tracking_barrier = common.read(tracking_barrier_path)
    if (tracking_barrier.get("status") != "complete" or tracking_barrier.get("count") != 8
            or tracking_barrier.get("GT_read") is not False):
        raise RuntimeError("the original-eight filtered-mask barrier is not complete and GT-free")
    if any(doc.get("GT_read") is not False for doc in (old_ref, old_geo, roster)):
        raise RuntimeError("original-eight no-GT contract inputs must have GT_read=false")
    refs_by_id = {str(row["case_id"]): row for row in old_ref["records"]}
    geos_by_id = {str(row["case_id"]): row for row in old_geo["records"]}
    roster_by_id = {str(row["case_id"]): row for row in roster["records"]}
    if set(refs_by_id) != set(geos_by_id) or set(refs_by_id) != set(roster_by_id) or len(refs_by_id) != 8:
        raise RuntimeError("original-eight reference, geometry, and roster cases do not align")

    from scripts.stvg_native_instance_common_v1 import read as common_read
    import numpy as np
    import json as json_module

    anchor_rows = geometry_rows = presence_weight_rows = no_update_cases = 0
    exact_center_rows = rle_files_sha_verified = 0
    geometry_fallbacks = geometry_mask_boxes = 0
    no_raw_teacher = 0
    for case_id in sorted(refs_by_id):
        locked = refs_by_id[case_id]
        geometry_record = geos_by_id[case_id]["raw_decoupled"]
        roster_record = roster_by_id[case_id]
        expected_anchors = locked["presence_supported"]["anchors"]
        selected_id = locked["raw_decoupled"].get("candidate_id")
        request_path = ROOT / roster_record["request_path"]
        request = common_read(request_path)
        inventory_path = ROOT / "artifacts/stvg_native_instance_spatial_opd_v1" / "TRACKING" / case_id / "MASK_INVENTORY.json"
        inventory = common_read(inventory_path)
        accepted_path = inventory_path.parent / "ACCEPTED.json"
        accepted = common_read(accepted_path)
        if accepted.get("GT_read") is not False or accepted.get("mask_inventory_sha256") != common.sha(inventory_path):
            raise RuntimeError(f"{case_id}: sealed accepted-tracking receipt does not bind its mask inventory")
        barrier_row = next((row for row in tracking_barrier.get("files", []) if row.get("case_id") == case_id), None)
        if barrier_row is None or barrier_row.get("sha256") != common.sha(accepted_path):
            raise RuntimeError(f"{case_id}: tracking global barrier does not bind accepted tracking")
        expected_grid = list(map(int, geometry_record["frame_ids_physical"]))
        frozen_projection = geometry_record["boxes_xyxy_normalized"]
        width = height = None
        if selected_id is not None and selected_id in set(inventory.get("known_ids", [])):
            # Read only one selected-object RLE header to recover the source
            # raster size. No image or annotation is opened.
            first_path = next(ROOT / row["path"] for row in inventory["files"] if row["index"] == 0)
            first_json = json_module.loads(first_path.read_bytes())
            obj = first_json.get("objects", {}).get(str(selected_id))
            if isinstance(obj, Mapping) and isinstance(obj.get("rle"), Mapping):
                size = obj["rle"].get("size")
                if isinstance(size, list) and len(size) == 2:
                    height, width = map(int, size)
        if width is None or height is None:
            # In no-ID/missing-ID cases no RLE is decoded; use the old
            # geometry-only shape binding only for the source dimensions.
            # Actual frozen boxes remain the supplied sealed geometry rows.
            # Dimensions are recoverable from the inventory's first RLE.
            first_path = ROOT / inventory["files"][0]["path"]
            first_json = json_module.loads(first_path.read_bytes())
            any_obj = next(iter(first_json.get("objects", {}).values()), None)
            if not isinstance(any_obj, Mapping) or not isinstance(any_obj.get("rle"), Mapping):
                raise RuntimeError("cannot obtain original filtered mask dimensions from RLE metadata")
            height, width = map(int, any_obj["rle"]["size"])
        source_projection = {
            "case_id": case_id,
            "dataset": locked["dataset"],
            "frame_ids": expected_grid,
            "width": width,
            "height": height,
            # The teacher geometry readback contains the exact full-grid
            # Frozen fallback boxes as well as selected-mask boxes. The
            # builder overwrites every present selected mask; absent entries
            # remain these sealed fallback coordinates.
            "native_boxes_xyxy_normalized": frozen_projection,
        }
        result = build_reference(
            source_projection, locked["qa_interval_frame_bounds_half_open"],
            request, selected_id, inventory,
        )
        roster_centers = roster_record.get("centers", [])
        if len(roster_centers) != 4 or len(result["centers"]) != 4:
            raise AssertionError(f"{case_id}: original roster must bind the same four QA centers")
        for actual_center, expected_center in zip(result["centers"], roster_centers):
            for actual_key, expected_key in (("slot", "slot"),
                                             ("center_physical_frame_id", "center_physical_frame_id"),
                                             ("source_grid_position", "source_grid_position"),
                                             ("local_frame_indices", "local_frame_indices")):
                if actual_center.get(actual_key) != expected_center.get(expected_key):
                    raise AssertionError(f"{case_id}: fixed QA center/window differs from locked roster")
            exact_center_rows += 1
        rle_files_sha_verified += int(result["rle_inventory_files_sha_verified"])
        actual_anchors = result["anchors"]
        if len(actual_anchors) != len(expected_anchors):
            raise AssertionError(f"{case_id}: supported anchor count differs from sealed reference")
        for actual, expected in zip(actual_anchors, expected_anchors):
            for key in ("slot", "position", "physical_frame_id", "local_frame_index",
                        "local_frame_indices", "box_xyxy_normalized", "box_cxcywh",
                        "candidate_id", "coordinate_convention", "reliability"):
                if actual.get(key) != expected.get(key):
                    raise AssertionError(f"{case_id}: anchor {key} differs from sealed reference")
            presence = float(expected["reliability"]["presence_fraction"])
            if actual["weight"] != presence:
                raise AssertionError(f"{case_id}: fit weight is not exact presence fraction")
            if actual["weight"] == expected["weight"] and expected["reliability"]["mean_adjacent_mask_iou"] != 1:
                # This is not an error: the new arm intentionally replaces
                # the legacy presence x adjacent-IoU fit weight.
                pass
            presence_weight_rows += 1
        expected_geometry = geometry_record
        actual_geometry = result["teacher_geometry"]
        for key in ("status", "candidate_id", "interval_frame_bounds_half_open", "frame_ids_physical",
                    "boxes_xyxy_normalized", "teacher_boxes_xyxy_normalized", "teacher_mask_present",
                    "teacher_mask_missing", "fallback_used", "fallback_reason", "box_source",
                    "interpolation_used", "nearest_frame_replacement_used"):
            if actual_geometry.get(key) != expected_geometry.get(key):
                raise AssertionError(f"{case_id}: TeacherGeometry field {key} differs from sealed readback")
        anchor_rows += len(actual_anchors)
        geometry_rows += len(actual_geometry["frame_ids_physical"])
        geometry_fallbacks += sum(bool(x) for x in actual_geometry["fallback_used"])
        geometry_mask_boxes += sum(bool(x) for x in actual_geometry["teacher_mask_present"])
        if result["no_update"]:
            no_update_cases += 1
        if expected_geometry["candidate_id"] is None:
            no_raw_teacher += 1
    return {
        "status": "pass",
        "case_count": 8,
        "center_rows_exact": exact_center_rows,
        "rle_inventory_files_sha_verified": rle_files_sha_verified,
        "anchor_rows_exact_except_intentional_presence_weight": anchor_rows,
        "presence_fraction_weights_exact": presence_weight_rows,
        "teacher_geometry_grid_rows_exact": geometry_rows,
        "teacher_mask_box_rows": geometry_mask_boxes,
        "frozen_fallback_rows": geometry_fallbacks,
        "prelocked_no_update_cases_without_supported_anchor": no_update_cases,
        "all_frozen_teacher_geometry_cases": no_raw_teacher,
        "inputs": {
            "old_reference_lock_sha256": common.sha(old_ref_path),
            "old_teacher_geometry_sha256": common.sha(old_geo_path),
            "roster_sha256": common.sha(roster_path),
            "tracking_barrier_sha256": common.sha(tracking_barrier_path),
            "RLEs_verified_on_required_local_frames": True,
        },
        "source_cache_pickle_opened": False,
        "GT_read": False,
        "RGB_or_video_read": False,
    }


def contracts() -> dict[str, Any]:
    synthetic = _contract_synthetic()
    real = _contract_original_eight()
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError("Spatial-A CPU contracts must not initialize CUDA")
    return {
        "schema": "stvg-table1-spatial-a-cpu-contracts-v1",
        "status": "pass",
        "synthetic": synthetic,
        "original_eight": real,
        "GT_read": False,
        "CUDA_initialized": False,
        "media_frames_read": False,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--contracts", action="store_true")
    args = parser.parse_args()
    if not args.contracts:
        parser.error("only --contracts is supported; this adapter does not launch workers")
    print(json.dumps(contracts(), sort_keys=True, separators=(",", ":"), allow_nan=False))
