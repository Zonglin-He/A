"""PTD decode variant that shares the event-generated reference and time.

The event pass uses the official PTD temporal-localization decoder.  Its actual
semantic and time token IDs are then appended to the original multimodal input
and prefetched again under the spatial visual branch.  Only after this fresh
spatial KV is built do we run the official cached box probe, conditioned on the
same time-anchor token IDs.  The existing independent-reference implementation
in :mod:`vg_tta.desta3d_v2_ptd` is intentionally left available as a control.

This module reads no labels and never reuses the event pass KV cache.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence
from unittest.mock import patch

import torch

from vg_tta.desta3d_v2_ptd import branch_injection


class SharedPrefixError(ValueError):
    """The event-generated response cannot safely seed the spatial pass."""


@dataclass(frozen=True)
class SharedReferenceTime:
    """Validated token-ID decision emitted by one official event pass."""

    reference_token_ids: tuple[int, ...]
    time_token_ids: tuple[int, int, int, int]
    time_anchor_ids: tuple[int, ...]
    interval: tuple[int, int]
    event_completion_ids: tuple[int, ...]

    def spatial_prefix_ids(self, newline_id: int) -> tuple[int, ...]:
        """IDs appended to the original prompt before the fresh spatial prefill."""
        delimiter = int(newline_id)
        return self.reference_token_ids + (delimiter,) + self.time_token_ids + (delimiter,)


def _module_ptd():
    # Match the project PTD callers, which add external/ParallelTubeDecoding/src
    # to sys.path while loading the local checkpoint processor.
    import model.ptd_generation as pg

    return pg


def _parse_semantic_tokens(
    reference: Sequence[int],
    token_ids: Mapping[str, Any],
    parse_semantic,
    *,
    block_size: int,
) -> None:
    if not reference:
        raise SharedPrefixError("shared semantic reference is empty")
    if block_size != 6:
        raise SharedPrefixError(f"PTD semantic block_size must be 6, got {block_size}")
    cursor = 0
    first = True
    reconstructed: list[int] = []
    complete = False
    while cursor < len(reference):
        chunk = list(reference[cursor : cursor + block_size])
        padded = chunk + [int(token_ids["null"])] * (block_size - len(chunk))
        try:
            accepted, complete = parse_semantic(
                padded, token_ids, first_block=first, block_size=block_size
            )
        except (RuntimeError, ValueError, KeyError, TypeError) as exc:
            raise SharedPrefixError(f"invalid shared semantic block: {exc}") from exc
        if tuple(accepted) != tuple(chunk[: len(accepted)]):
            raise SharedPrefixError("semantic parser changed the supplied token prefix")
        reconstructed.extend(accepted)
        if complete:
            if cursor + len(chunk) != len(reference):
                raise SharedPrefixError("tokens follow the semantic reference closing marker")
            break
        if len(chunk) != block_size:
            raise SharedPrefixError("semantic reference ends before its official closing marker")
        cursor += block_size
        first = False
    if not complete or tuple(reconstructed) != tuple(reference):
        raise SharedPrefixError("semantic reference is incomplete or not token-identical")


def validate_shared_reference_time(
    shared: SharedReferenceTime,
    token_ids: Mapping[str, Any],
    ptd_module=None,
    *,
    block_size: int = 6,
) -> SharedReferenceTime:
    """Revalidate token grammar, interval, anchors, and event completion IDs.

    This is called immediately before constructing the spatial prefix. Invalid
    prefixes fail explicitly; the helper never silently regenerates a new
    semantic reference or falls back to independent-reference decoding.
    """
    pg = ptd_module or _module_ptd()
    reference = tuple(int(x) for x in shared.reference_token_ids)
    temporal = tuple(int(x) for x in shared.time_token_ids)
    anchors = tuple(int(x) for x in shared.time_anchor_ids)
    if len(temporal) != 4:
        raise SharedPrefixError(f"temporal segment must contain 4 token IDs, got {len(temporal)}")
    _parse_semantic_tokens(reference, token_ids, pg._parse_semantic_block, block_size=block_size)
    padded_time = list(temporal) + [int(token_ids["null"])] * (block_size - len(temporal))
    try:
        parsed_tokens, parsed_anchors = pg._parse_temporal_block(
            padded_time, token_ids, block_size=block_size
        )
    except (RuntimeError, ValueError, KeyError, TypeError) as exc:
        raise SharedPrefixError(f"invalid shared temporal segment: {exc}") from exc
    if tuple(parsed_tokens) != temporal:
        raise SharedPrefixError("temporal parser changed the supplied token prefix")
    time_to_index = token_ids["time_token_indices"]
    start_id, end_id = temporal[1], temporal[2]
    if start_id not in time_to_index or end_id not in time_to_index:
        raise SharedPrefixError("shared time endpoints are absent from the PTD time vocabulary")
    interval = (int(time_to_index[start_id]), int(time_to_index[end_id]))
    expected_anchors = tuple(int(x) for x in token_ids["ordered_time_tokens"][interval[0] : interval[1] + 1])
    if interval[1] < interval[0] or not expected_anchors:
        raise SharedPrefixError("shared time interval is reversed or empty")
    if anchors != tuple(int(x) for x in parsed_anchors) or anchors != expected_anchors:
        raise SharedPrefixError("shared time-anchor IDs do not match the generated endpoints")
    if tuple(shared.interval) != interval:
        raise SharedPrefixError("stored interval does not match the generated time-token IDs")
    expected_completion = reference + (int(token_ids["newline"]),) + temporal + (int(token_ids["eos"]),)
    if tuple(int(x) for x in shared.event_completion_ids) != expected_completion:
        raise SharedPrefixError("event completion is not the exact reference/time token sequence")
    if reference[0] != int(token_ids["ref_start"]) or reference[-1] != int(token_ids["ref_end"]):
        raise SharedPrefixError("semantic reference lacks official start/end markers")
    return shared


@torch.inference_mode()
def _generate_event_reference_time(model, processor, inputs, *, ptd_attn_implementation: str):
    pg = _module_ptd()
    nframes = int(inputs["video_grid_thw"][0, 0])
    token_ids = pg.build_ptd_token_ids(processor.tokenizer, max_time_tokens=nframes)
    original_semantic = pg._parse_semantic_block
    original_temporal = pg._parse_temporal_block
    semantic_blocks: list[tuple[tuple[int, ...], bool]] = []
    temporal_blocks: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def capture_semantic(*args, **kwargs):
        result = original_semantic(*args, **kwargs)
        semantic_blocks.append((tuple(int(x) for x in result[0]), bool(result[1])))
        return result

    def capture_temporal(*args, **kwargs):
        result = original_temporal(*args, **kwargs)
        temporal_blocks.append((tuple(int(x) for x in result[0]), tuple(int(x) for x in result[1])))
        return result

    with patch.object(pg, "_parse_semantic_block", capture_semantic), patch.object(
        pg, "_parse_temporal_block", capture_temporal
    ):
        generated, result = pg.generate_ptd(
            model,
            processor.tokenizer,
            dict(inputs),
            max_new_tokens=1024,
            max_time_tokens=nframes,
            temperature=0.0,
            generation_format="temporal_localization",
            ptd_attn_implementation=ptd_attn_implementation,
        )

    generated_ids = tuple(int(x) for x in generated[0].detach().cpu().tolist())
    completion = processor.tokenizer.decode(generated[0], skip_special_tokens=False)
    event = {
        "interval": None,
        "format_ok": False,
        "temporal": None,
        "semantic_token_ids": [],
        "completion": completion,
        "completion_token_ids": list(generated_ids),
        "GT_used": False,
    }
    if not bool(getattr(result, "stopped", False)):
        event["failure"] = "official event/reference/time generation did not stop successfully"
        return event, None
    if not semantic_blocks or not semantic_blocks[-1][1] or len(temporal_blocks) != 1:
        event["failure"] = "official event generation omitted a complete reference or time segment"
        return event, None
    reference = tuple(token for block, _done in semantic_blocks for token in block)
    temporal, anchors = temporal_blocks[0]
    expected = reference + (int(token_ids["newline"]),) + temporal + (int(token_ids["eos"]),)
    if generated_ids != expected:
        event["failure"] = "official completion IDs disagree with captured semantic/time decisions"
        return event, None
    time_to_index = token_ids["time_token_indices"]
    if temporal[1] not in time_to_index or temporal[2] not in time_to_index:
        event["failure"] = "generated time endpoints are outside the registered PTD vocabulary"
        return event, None
    interval = (int(time_to_index[temporal[1]]), int(time_to_index[temporal[2]]))
    shared = SharedReferenceTime(
        reference_token_ids=reference,
        time_token_ids=(temporal[0], temporal[1], temporal[2], temporal[3]),
        time_anchor_ids=anchors,
        interval=interval,
        event_completion_ids=generated_ids,
    )
    try:
        validate_shared_reference_time(shared, token_ids, pg)
    except SharedPrefixError as exc:
        event["failure"] = f"invalid generated shared prefix: {exc}"
        return event, None
    event.update(
        interval=list(interval),
        format_ok=True,
        semantic_token_ids=list(reference),
        temporal={"tokens": list(temporal), "anchors": list(anchors)},
        shared_reference_time=shared,
    )
    return event, (shared, token_ids)


def _append_shared_prefix(inputs: Mapping[str, Any], shared: SharedReferenceTime, token_ids):
    if "input_ids" not in inputs or not torch.is_tensor(inputs["input_ids"]):
        raise ValueError("spatial shared-prefix prefill requires tensor input_ids")
    input_ids = inputs["input_ids"]
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
        raise ValueError(f"expected nonempty batch-one input_ids, got {tuple(input_ids.shape)}")
    new_ids = list(shared.spatial_prefix_ids(int(token_ids["newline"])))
    suffix = torch.tensor([new_ids], dtype=input_ids.dtype, device=input_ids.device)
    extended = dict(inputs)
    extended["input_ids"] = torch.cat([input_ids, suffix], dim=1)
    attention = inputs.get("attention_mask")
    if attention is None:
        attention = torch.ones_like(input_ids, dtype=torch.long)
    if not torch.is_tensor(attention) or tuple(attention.shape) != tuple(input_ids.shape):
        raise ValueError("attention_mask must match the original input_ids shape")
    if not bool(attention[0, -1].item()):
        raise ValueError("right-padded source prompts are unsupported for shared-prefix decode")
    suffix_attention = torch.ones((1, len(new_ids)), dtype=attention.dtype, device=attention.device)
    extended["attention_mask"] = torch.cat([attention, suffix_attention], dim=1)
    mm_types = inputs.get("mm_token_type_ids")
    if mm_types is not None:
        if not torch.is_tensor(mm_types) or tuple(mm_types.shape) != tuple(input_ids.shape):
            raise ValueError("mm_token_type_ids must match the original input_ids shape")
        mm_suffix = torch.zeros((1, len(new_ids)), dtype=mm_types.dtype, device=mm_types.device)
        extended["mm_token_type_ids"] = torch.cat([mm_types, mm_suffix], dim=1)
    token_types = inputs.get("token_type_ids")
    if token_types is not None:
        if not torch.is_tensor(token_types) or tuple(token_types.shape) != tuple(input_ids.shape):
            raise ValueError("token_type_ids must match the original input_ids shape")
        token_suffix = torch.zeros((1, len(new_ids)), dtype=token_types.dtype, device=token_types.device)
        extended["token_type_ids"] = torch.cat([token_types, token_suffix], dim=1)
    return extended


def _spatial_prefill_kwargs(inputs: Mapping[str, Any]) -> dict[str, Any]:
    prefill = dict(inputs)
    for stale in (
        "position_ids",
        "cache_position",
        "past_key_values",
        "labels",
        "ptd_position_ids",
        "ptd_prefix_lengths",
        "ptd_context_limits",
        "ptd_prefix_length",
    ):
        prefill.pop(stale, None)
    prefill["use_cache"] = True
    prefill["return_dict"] = True
    prefill["logits_to_keep"] = 1
    return prefill


@torch.inference_mode()
def _spatial_boxes_from_shared_prefix(
    model,
    processor,
    inputs,
    fields,
    adapter,
    shared: SharedReferenceTime,
    token_ids,
    *,
    gate_override=None,
    ptd_attn_implementation: str = "sdpa",
):
    pg = _module_ptd()
    validate_shared_reference_time(shared, token_ids, pg)
    spatial_inputs = _append_shared_prefix(inputs, shared, token_ids)
    prefix_len = int(spatial_inputs["input_ids"].shape[1])
    n_boxes = len(shared.time_anchor_ids)
    if n_boxes <= 0:
        raise SharedPrefixError("shared event interval contains no spatial probes")
    if n_boxes != shared.interval[1] - shared.interval[0] + 1:
        raise SharedPrefixError("shared time anchors do not cover the full generated interval")

    pg.configure_ptd_model(model, block_size=6)
    with branch_injection(model, adapter, spatial_inputs, fields, "spatial", gate_override=gate_override) as injection:
        outputs = model(**_spatial_prefill_kwargs(spatial_inputs))
    past_key_values = getattr(outputs, "past_key_values", None)
    if past_key_values is None:
        raise RuntimeError("fresh spatial multimodal prefill returned no KV cache")

    # Exactly mirror generate_ptd's box-probe positions and context: each block
    # sees the full spatial prefix with the event-generated reference and time.
    query_token_ids = torch.tensor(
        shared.time_anchor_ids, dtype=spatial_inputs["input_ids"].dtype,
        device=spatial_inputs["input_ids"].device,
    )
    positions = torch.arange(n_boxes, dtype=torch.long, device=query_token_ids.device)
    probe_position_starts = prefix_len + 8 * positions
    context_limits = torch.full((n_boxes,), prefix_len, dtype=torch.long, device=query_token_ids.device)
    ptd_attn_implementation = pg.resolve_ptd_attn_implementation(ptd_attn_implementation)
    if ptd_attn_implementation == "flash_attention_2":
        # The official generator converts a dynamic cache before issuing
        # flattened parallel blocks; reserve only this pass's exact box suffix.
        cache_len = int(pg._cache_length(past_key_values))
        core = pg._core_model(model)
        past_key_values = pg.make_reusable_ptd_cache(
            past_key_values,
            config=getattr(core.language_model, "config", None),
            max_cache_len=cache_len + 8 * n_boxes + 6 * n_boxes,
        )
    # Capture the real LM-head scores already computed by this spatial box
    # probe.  Sampling consumes the same tensor, so this wrapper adds no model
    # forward and leaves decode/cache behavior unchanged.
    coord_ids_by_value = tuple(
        int(pg.get_token_id(processor.tokenizer, f"<{value}>"))
        for value in range(1001)
    )
    if len(set(coord_ids_by_value)) != 1001:
        raise RuntimeError("tokenizer coordinate token IDs are not unique")
    registered_coords = token_ids.get("coord_id_to_value")
    expected_coord_map = {token_id: value for value, token_id in enumerate(coord_ids_by_value)}
    if registered_coords != expected_coord_map:
        raise RuntimeError("tokenizer coordinate IDs disagree with the PTD token registry")
    captured: dict[str, torch.Tensor] = {}
    original_run_language_model = pg._run_language_model

    def capture_spatial_coordinate_logits(*args, **kwargs):
        outputs, block_logits = original_run_language_model(*args, **kwargs)
        expected_shape = (1, n_boxes * 6)
        if block_logits.ndim != 3 or tuple(block_logits.shape[:2]) != expected_shape:
            raise RuntimeError(
                "spatial PTD LM logits must have shape "
                f"[1,{n_boxes * 6},vocab], got {tuple(block_logits.shape)}"
            )
        if block_logits.shape[-1] <= max(coord_ids_by_value):
            raise RuntimeError("spatial PTD LM logits do not cover coordinate token IDs")
        if "coordinate_logits" in captured:
            raise RuntimeError("spatial PTD probe invoked the LM wrapper more than once")
        coordinate_ids = torch.tensor(
            coord_ids_by_value, dtype=torch.long, device=block_logits.device
        )
        # In each 6-token block, the query position predicts box_start; the
        # following four positions predict cx, cy, w, h respectively.
        coordinate_logits = block_logits.reshape(1, n_boxes, 6, -1)[0, :, 1:5, :]
        captured["coordinate_logits"] = coordinate_logits.index_select(
            -1, coordinate_ids
        ).detach().to(device="cpu", dtype=torch.float32).contiguous()
        return outputs, block_logits

    with patch.object(pg, "_run_language_model", capture_spatial_coordinate_logits):
        raw_blocks, spatial_cache_after_probe = pg._run_cached_ptd_probe(
            model,
            spatial_inputs["input_ids"],
            token_ids,
            past_key_values,
            query_token_ids=query_token_ids,
            probe_position_starts=probe_position_starts,
            context_limits=context_limits,
            block_size=6,
            temperature=0.0,
            top_p=None,
            top_k=None,
            ptd_attn_implementation=ptd_attn_implementation,
        )
    del spatial_cache_after_probe
    if "coordinate_logits" not in captured:
        raise RuntimeError("spatial PTD probe did not expose its coordinate logits")

    completion_ids = list(shared.reference_token_ids) + [int(token_ids["newline"])]
    completion_ids += list(shared.time_token_ids) + [int(token_ids["newline"])]
    spatial = {
        "interval": list(shared.interval),
        "positions": [],
        "boxes": torch.empty((0, 4), dtype=torch.float32),
        "geometry_valid": torch.empty((0,), dtype=torch.bool),
        "format_ok": False,
        "completion": "",
        "raw_box_blocks": raw_blocks.detach().to(device="cpu"),
        "logits": captured["coordinate_logits"],
        "coordinate_token_ids": list(coord_ids_by_value),
        "shared_reference_token_ids": list(shared.reference_token_ids),
        "shared_time_token_ids": list(shared.time_token_ids),
        "spatial_prefix_input_ids": spatial_inputs["input_ids"].detach().to(device="cpu"),
        "spatial_prefix_length": prefix_len,
        "box_probe_query_token_ids": query_token_ids.detach().to(device="cpu"),
        "box_probe_position_starts": probe_position_starts.detach().to(device="cpu"),
        "box_probe_context_limits": context_limits.detach().to(device="cpu"),
        "GT_used": False,
    }
    try:
        blocks = pg._validate_box_blocks(raw_blocks, token_ids, expected_blocks=n_boxes, block_size=6)
    except (RuntimeError, ValueError) as exc:
        spatial["failure"] = f"official PTD box grammar failure: {exc}"
        spatial["completion"] = processor.tokenizer.decode(
            torch.tensor(completion_ids, dtype=spatial_inputs["input_ids"].dtype),
            skip_special_tokens=False,
        )
        return spatial, injection

    coordinate_values = torch.tensor(
        [[token_ids["coord_id_to_value"][token] for token in block[1:5]] for block in blocks],
        dtype=torch.float32,
    )
    from vg_tta.ptd_spatial_adapter_ab_v1 import boxes_from_tokens

    boxes, geometry_valid = boxes_from_tokens(coordinate_values)
    positions_list = list(range(shared.interval[0], shared.interval[1] + 1))
    for i, (anchor, block) in enumerate(zip(shared.time_anchor_ids, blocks)):
        completion_ids.extend([int(anchor), *block, int(token_ids["eos"] if i == n_boxes - 1 else token_ids["newline"])])
    spatial.update(
        positions=positions_list,
        boxes=boxes.detach().to(device="cpu"),
        geometry_valid=geometry_valid.detach().to(device="cpu"),
        format_ok=True,
        completion=processor.tokenizer.decode(
            torch.tensor(completion_ids, dtype=spatial_inputs["input_ids"].dtype),
            skip_special_tokens=False,
        ),
        box_token_blocks=blocks,
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
    ptd_attn_implementation: str = "sdpa",
):
    """Decode event reference/time once, then spatial boxes with a fresh KV.

    Return keys compatible with ``vg_tta.desta3d_v2_ptd.decode_two_pass`` and
    its source prediction record: ``event``, ``spatial``, ``interval``,
    ``format_ok``, ``event_injection``, ``spatial_injection``, ``GT_used`` and
    ``cache_policy``. The event-generated IDs are inserted into the spatial
    input prefix; the spatial pass never invokes the semantic or time decoder.
    """
    with branch_injection(model, adapter, inputs, fields, "event", gate_override=gate_override) as event_injection:
        event, packed = _generate_event_reference_time(
            model, processor, inputs, ptd_attn_implementation=ptd_attn_implementation
        )
    if not event["format_ok"] or packed is None:
        return {
            "event": event,
            "spatial": None,
            "interval": event.get("interval"),
            "format_ok": False,
            "event_injection": event_injection,
            "spatial_injection": None,
            "GT_used": False,
            "cache_policy": "event failure retained; spatial pass not started",
        }
    shared, token_ids = packed
    try:
        spatial, spatial_injection = _spatial_boxes_from_shared_prefix(
            model,
            processor,
            inputs,
            fields,
            adapter,
            shared,
            token_ids,
            gate_override=gate_override,
            ptd_attn_implementation=ptd_attn_implementation,
        )
    except SharedPrefixError as exc:
        event["failure"] = f"shared reference/time prefix rejected before spatial decode: {exc}"
        return {
            "event": event,
            "spatial": None,
            "interval": event["interval"],
            "format_ok": False,
            "event_injection": event_injection,
            "spatial_injection": None,
            "GT_used": False,
            "cache_policy": "invalid shared prefix retained as failure; no independent fallback",
        }
    return {
        "event": event,
        "spatial": spatial,
        "interval": list(shared.interval),
        "format_ok": bool(spatial["format_ok"]),
        "event_injection": event_injection,
        "spatial_injection": spatial_injection,
        "GT_used": False,
        "cache_policy": "fresh spatial KV prefills original multimodal prompt plus exact event-generated reference/time IDs; event KV discarded",
    }
