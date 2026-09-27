"""Shared-reference PTD decode preserving the official incremental cache path.

The event branch generates semantic-reference/time IDs once.  The spatial
branch then starts from its own fresh multimodal prefill of the original
prompt and runs the official ``generate_ptd`` schedule.  Parser callbacks
replace only semantic/time decisions with the already generated event IDs;
the official semantic, temporal, and box probe/cache schedule remains intact.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import patch

import torch

from vg_tta.desta3d_v2_ptd import branch_injection
from vg_tta.desta3d_v2_shared_reference import (
    SharedPrefixError,
    _generate_event_reference_time,
    _module_ptd,
    validate_shared_reference_time,
)


def _coordinate_token_ids(pg, tokenizer, token_ids) -> tuple[int, ...]:
    ids = tuple(int(pg.get_token_id(tokenizer, f"<{i}>")) for i in range(1001))
    if len(set(ids)) != 1001:
        raise RuntimeError("tokenizer coordinate token IDs are not unique")
    expected = {token_id: value for value, token_id in enumerate(ids)}
    if token_ids.get("coord_id_to_value") != expected:
        raise RuntimeError("tokenizer coordinate IDs disagree with the PTD token registry")
    return ids


@torch.inference_mode()
def _spatial_decode_official_cached(
    model,
    processor,
    inputs,
    fields,
    adapter,
    shared,
    event_token_ids,
    *,
    gate_override=None,
    ptd_attn_implementation="sdpa",
):
    pg = _module_ptd()
    nframes = int(inputs["video_grid_thw"][0, 0])
    token_ids = pg.build_ptd_token_ids(processor.tokenizer, max_time_tokens=nframes)
    validate_shared_reference_time(shared, event_token_ids, pg)
    validate_shared_reference_time(shared, token_ids, pg)
    if token_ids["coord_id_to_value"] != event_token_ids["coord_id_to_value"]:
        raise SharedPrefixError("event and spatial PTD coordinate registries differ")

    reference = tuple(int(x) for x in shared.reference_token_ids)
    anchors = tuple(int(x) for x in shared.time_anchor_ids)
    temporal = tuple(int(x) for x in shared.time_token_ids)
    coordinate_ids = _coordinate_token_ids(pg, processor.tokenizer, token_ids)
    coord_capture: dict[str, Any] = {}
    semantic_cursor = 0
    temporal_calls = 0
    original_semantic = pg._parse_semantic_block
    original_temporal = pg._parse_temporal_block
    original_probe = pg._run_cached_ptd_probe

    def force_event_semantic(_raw, _token_ids, *, first_block, block_size):
        nonlocal semantic_cursor
        if block_size != 6 or semantic_cursor >= len(reference):
            raise SharedPrefixError("unexpected spatial semantic probe after shared reference ended")
        if bool(first_block) != (semantic_cursor == 0):
            raise SharedPrefixError("official spatial semantic block order is inconsistent")
        chunk = reference[semantic_cursor : semantic_cursor + block_size]
        complete = semantic_cursor + len(chunk) == len(reference)
        if not chunk or (not complete and len(chunk) != block_size):
            raise SharedPrefixError("shared reference cannot be partitioned into PTD blocks")
        if complete != (chunk[-1] == int(token_ids["ref_end"])):
            raise SharedPrefixError("shared reference closing marker is not at its final PTD slot")
        semantic_cursor += len(chunk)
        return list(chunk), complete

    def force_event_temporal(_raw, _token_ids, *, block_size):
        nonlocal temporal_calls
        if block_size != 6 or semantic_cursor != len(reference) or temporal_calls:
            raise SharedPrefixError("official spatial temporal probe is out of sequence")
        temporal_calls += 1
        return list(temporal), list(anchors)

    def observe_official_probe(*args, **kwargs):
        query = tuple(
            int(x)
            for x in torch.as_tensor(kwargs["query_token_ids"]).detach().cpu().flatten().tolist()
        )
        if query != anchors:
            return original_probe(*args, **kwargs)

        def capture_lm(*lm_args, **lm_kwargs):
            outputs, logits = original_lm(*lm_args, **lm_kwargs)
            expected = (1, len(anchors) * 6)
            if logits.ndim != 3 or tuple(logits.shape[:2]) != expected:
                raise RuntimeError(
                    f"box-probe logits must have shape [1,{expected[1]},vocab], got {tuple(logits.shape)}"
                )
            if logits.shape[-1] <= max(coordinate_ids):
                raise RuntimeError("box-probe logits do not cover coordinate token IDs")
            coord_index = torch.tensor(coordinate_ids, dtype=torch.long, device=logits.device)
            coord_logits = logits.reshape(1, len(anchors), 6, -1)[0, :, 1:5, :]
            coord_capture["logits"] = coord_logits.index_select(-1, coord_index).detach().to(
                device="cpu", dtype=torch.float32
            ).contiguous()
            return outputs, logits

        original_lm = pg._run_language_model
        with patch.object(pg, "_run_language_model", capture_lm):
            blocks, cache = original_probe(*args, **kwargs)
        coord_capture["raw_blocks"] = blocks.detach().to(device="cpu")
        coord_capture["query"] = torch.as_tensor(kwargs["query_token_ids"]).detach().to(device="cpu")
        coord_capture["positions"] = torch.as_tensor(kwargs["probe_position_starts"]).detach().to(device="cpu")
        coord_capture["contexts"] = torch.as_tensor(kwargs["context_limits"]).detach().to(device="cpu")
        return blocks, cache

    pg.configure_ptd_model(model, block_size=6)
    with branch_injection(
        model, adapter, inputs, fields, "spatial", gate_override=gate_override
    ) as injection:
        with patch.object(pg, "_parse_semantic_block", force_event_semantic), patch.object(
            pg, "_parse_temporal_block", force_event_temporal
        ), patch.object(pg, "_run_cached_ptd_probe", observe_official_probe):
            completion_ids, result = pg.generate_ptd(
                model,
                processor.tokenizer,
                dict(inputs),
                max_new_tokens=1024,
                max_time_tokens=nframes,
                temperature=0.0,
                generation_format="spatio_temporal_grounding",
                ptd_attn_implementation=ptd_attn_implementation,
            )

    completion = processor.tokenizer.decode(completion_ids[0], skip_special_tokens=False)
    spatial = {
        "interval": list(shared.interval),
        "positions": [],
        "boxes": torch.empty((0, 4), dtype=torch.float32),
        "geometry_valid": torch.empty((0,), dtype=torch.bool),
        "format_ok": False,
        "completion": completion,
        "shared_reference_token_ids": list(reference),
        "shared_time_token_ids": list(temporal),
        "spatial_prefix_length": int(inputs["input_ids"].shape[1]) + len(reference) + 1 + len(temporal) + 1,
        "GT_used": False,
    }
    if "raw_blocks" in coord_capture and "logits" in coord_capture:
        spatial.update(
            raw_box_blocks=coord_capture["raw_blocks"],
            logits=coord_capture["logits"],
            coordinate_token_ids=list(coordinate_ids),
            box_probe_query_token_ids=coord_capture["query"],
            box_probe_position_starts=coord_capture["positions"],
            box_probe_context_limits=coord_capture["contexts"],
        )
    if not bool(getattr(result, "stopped", False)):
        spatial["failure"] = "official spatial generation did not stop successfully"
        return spatial, injection
    if semantic_cursor != len(reference) or temporal_calls != 1:
        spatial["failure"] = "official spatial path did not consume the exact shared reference/time"
        return spatial, injection
    if "raw_blocks" not in coord_capture or "logits" not in coord_capture:
        spatial["failure"] = "official spatial path did not expose its final box-probe logits"
        return spatial, injection
    raw_blocks = coord_capture["raw_blocks"]
    try:
        blocks = pg._validate_box_blocks(raw_blocks, token_ids, expected_blocks=len(anchors), block_size=6)
    except (RuntimeError, ValueError) as exc:
        spatial["failure"] = f"official PTD box grammar failure: {exc}"
        return spatial, injection

    coordinate_values = torch.tensor(
        [[token_ids["coord_id_to_value"][int(token)] for token in block[1:5]] for block in blocks],
        dtype=torch.float32,
    )
    from vg_tta.ptd_spatial_adapter_ab_v1 import boxes_from_tokens

    boxes, geometry_valid = boxes_from_tokens(coordinate_values)
    spatial.update(
        positions=list(range(shared.interval[0], shared.interval[1] + 1)),
        boxes=boxes.detach().to(device="cpu"),
        geometry_valid=geometry_valid.detach().to(device="cpu"),
        box_token_blocks=blocks,
        format_ok=True,
    )
    return spatial, injection


@torch.inference_mode()
def decode_shared_reference_two_pass(
    model,
    processor,
    inputs,
    adapter,
    fields,
    *,
    gate_override=None,
    ptd_attn_implementation="sdpa",
):
    """Decode the event IDs once and force them through official spatial PTD."""
    with branch_injection(model, adapter, inputs, fields, "event", gate_override=gate_override) as event_injection:
        event, packed = _generate_event_reference_time(
            model, processor, inputs, ptd_attn_implementation=ptd_attn_implementation
        )
    if not event["format_ok"] or packed is None:
        return {
            "event": event, "spatial": None, "interval": event.get("interval"),
            "format_ok": False, "event_injection": event_injection,
            "spatial_injection": None, "GT_used": False,
            "cache_policy": "event failure retained; spatial pass not started",
        }
    shared, event_token_ids = packed
    try:
        spatial, spatial_injection = _spatial_decode_official_cached(
            model, processor, inputs, fields, adapter, shared, event_token_ids,
            gate_override=gate_override, ptd_attn_implementation=ptd_attn_implementation,
        )
    except SharedPrefixError as exc:
        event["failure"] = f"shared reference/time prefix rejected before spatial decode: {exc}"
        return {
            "event": event, "spatial": None, "interval": event["interval"],
            "format_ok": False, "event_injection": event_injection,
            "spatial_injection": None, "GT_used": False,
            "cache_policy": "invalid shared prefix retained as failure; no fallback",
        }
    return {
        "event": event,
        "spatial": spatial,
        "interval": list(shared.interval),
        "format_ok": bool(spatial["format_ok"]),
        "event_injection": event_injection,
        "spatial_injection": spatial_injection,
        "GT_used": False,
        "cache_policy": "fresh spatial multimodal prefill; official incremental PTD semantic/time/box cache path with semantic/time parser outputs fixed to event-generated IDs",
    }
