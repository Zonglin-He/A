"""CPU-only helpers for source-supervised DESTA-3D v2 PTD loss partitioning.

The helper below reads the official PTD ``labels`` and metadata as-is. It does
not alter the causal labels, input IDs, attention mask, PTD position IDs, or
context limits. Returned masks align with the target sequence used by
``ptd_8b_teacher_feasibility_v1.joint_loss``:

    logit_positions = nonzero(labels[0, 1:] != -100)
    target_positions = logit_positions + 1

Use ``temporal`` for semantic-reference plus time-token CE and ``spatial`` for
per-frame box-row CE. Formatting tokens (newlines and assistant end marker)
remain in ``structural`` so a caller must make an explicit decision about
whether/how to retain them. MTP ``<null>`` padding is retained and assigned to
the type of its originating block, matching the official joint CE.

This module intentionally does no model forward, device transfer, file read,
or CUDA initialization. It is intended to build/inspect masks on CPU before a
training batch is moved to the model device.
"""

from __future__ import annotations

from typing import Any

import torch


IGNORE_INDEX = -100
PTD_BLOCK_SIZE = 6


def _single_token_id(tokenizer: Any, token: str) -> int:
    value = tokenizer.convert_tokens_to_ids(token)
    if value is None:
        raise ValueError(f"tokenizer does not know required PTD token {token!r}")
    value = int(value)
    unknown = getattr(tokenizer, "unk_token_id", None)
    if unknown is not None and value == int(unknown):
        raise ValueError(f"required PTD token {token!r} maps to unk_token_id")
    return value


def _cpu_vector(data: dict[str, Any], name: str) -> torch.Tensor:
    if name not in data:
        raise ValueError(f"PTD source sample is missing {name}")
    value = torch.as_tensor(data[name]).detach().to(device="cpu")
    if value.ndim == 2:
        if value.shape[0] != 1:
            raise ValueError(f"{name} must contain a single source example, got {tuple(value.shape)}")
        value = value[0]
    if value.ndim != 1:
        raise ValueError(f"{name} must be rank 1 or [1,L], got {tuple(value.shape)}")
    return value


def _prefix_response_spans(
    input_ids: list[int],
    labels: list[int],
    prefix_length: int,
    tokenizer: Any,
) -> tuple[list[str], dict[str, Any]]:
    """Parse the official response grammar in the original teacher-forced prefix."""
    ids = {
        name: _single_token_id(tokenizer, token)
        for name, token in {
            "ref_start": "<|object_ref_start|>",
            "ref_end": "<|object_ref_end|>",
            "time_start": "<|time_start|>",
            "time_end": "<|time_end|>",
            "box_start": "<|box_start|>",
            "box_end": "<|box_end|>",
            "im_end": "<|im_end|>",
            "null": "<null>",
        }.items()
    }
    newline_tokens = tokenizer.encode("\n", add_special_tokens=False)
    if not newline_tokens:
        raise ValueError("tokenizer encodes the PTD newline separator as an empty sequence")
    newline_id = int(newline_tokens[0])
    if len(newline_tokens) != 1:
        raise ValueError("official PTD response grammar requires newline to be one token")

    time_ids = {_single_token_id(tokenizer, f"<t{i}>") for i in range(1, 101)}
    coordinate_ids = {_single_token_id(tokenizer, f"<{i}>") for i in range(1001)}

    supervised_prefix = [i for i in range(prefix_length) if labels[i] != IGNORE_INDEX]
    if not supervised_prefix:
        raise ValueError("source PTD prefix has no supervised response tokens")
    response_start, response_end = supervised_prefix[0], supervised_prefix[-1]
    if supervised_prefix != list(range(response_start, prefix_length)):
        raise ValueError("source PTD response labels must be contiguous through ptd_prefix_length")
    if response_end >= len(input_ids):
        raise ValueError("response labels extend beyond input_ids")
    if input_ids[response_start] != ids["ref_start"]:
        raise ValueError("source response must start with <|object_ref_start|>")

    categories = ["structural"] * prefix_length
    ref_end_positions = [
        p for p in range(response_start + 1, prefix_length) if input_ids[p] == ids["ref_end"]
    ]
    if len(ref_end_positions) != 1:
        raise ValueError(f"expected one object reference end token, found {len(ref_end_positions)}")
    ref_end = ref_end_positions[0]
    for p in range(response_start, ref_end + 1):
        categories[p] = "semantic"

    time_start_positions = [
        p for p in range(ref_end + 1, prefix_length) if input_ids[p] == ids["time_start"]
    ]
    if len(time_start_positions) != 1:
        raise ValueError(f"expected one PTD time segment, found {len(time_start_positions)}")
    time_start = time_start_positions[0]
    if time_start + 3 >= prefix_length or [input_ids[time_start], input_ids[time_start + 3]] != [ids["time_start"], ids["time_end"]]:
        raise ValueError("invalid four-token PTD time segment")
    if input_ids[time_start + 1] not in time_ids or input_ids[time_start + 2] not in time_ids:
        raise ValueError("PTD time segment must contain two learned <tN> tokens")
    for p in range(time_start, time_start + 4):
        categories[p] = "time"

    end_positions = [
        p for p in range(time_start + 4, prefix_length) if input_ids[p] == ids["im_end"]
    ]
    if len(end_positions) != 1:
        raise ValueError(f"expected one assistant end token after PTD time segment, found {len(end_positions)}")
    im_end = end_positions[0]

    cursor = time_start + 4
    if cursor < im_end and input_ids[cursor] == newline_id:
        cursor += 1
    box_rows = []
    while cursor < im_end:
        # Newlines separate the time segment and each time-anchored box row;
        # they remain structural rather than being silently dropped.
        if input_ids[cursor] == newline_id:
            cursor += 1
            continue
        if input_ids[cursor] not in time_ids:
            raise ValueError(f"expected a learned <tN> box anchor at response position {cursor}")
        if cursor + 6 >= im_end:
            raise ValueError("truncated seven-token PTD box row")
        row = input_ids[cursor : cursor + 7]
        if row[1] != ids["box_start"] or row[6] != ids["box_end"]:
            raise ValueError(f"invalid time-anchored PTD box row at response position {cursor}")
        if any(value not in coordinate_ids for value in row[2:6]):
            raise ValueError(f"invalid coordinate token in PTD box row at response position {cursor}")
        for p in range(cursor, cursor + 7):
            categories[p] = "box"
        box_rows.append((cursor, cursor + 6))
        cursor += 7
    if not box_rows:
        raise ValueError("source response contains no time-anchored PTD box row")
    if any(input_ids[p] != newline_id for p in range(im_end + 1, prefix_length)):
        raise ValueError("only trailing newline tokens may follow the assistant end token")

    spans = {
        "response_start": response_start,
        "response_end": response_end,
        "semantic_span_inclusive": [response_start, ref_end],
        "time_span_inclusive": [time_start, time_start + 3],
        "box_spans_inclusive": [[a, b] for a, b in box_rows],
        "assistant_end_position": im_end,
        "newline_token_id": newline_id,
    }
    return categories, {**spans, "token_ids": ids}


def split_source_loss_masks(data: dict[str, Any], tokenizer: Any) -> dict[str, Any]:
    """Return CE-target masks split into semantic/time, box, and formatting.

    Args:
        data: One example produced by the official PTD source target builder.
            Required tensors are ``input_ids``, ``labels``, ``ptd_position_ids``,
            and ``ptd_prefix_lengths`` (the singular official pre-collator key
            ``ptd_prefix_length`` is accepted too).
        tokenizer: The matching official PTD tokenizer, with its learned
            ``<t1>`` ... ``<t100>`` and coordinate tokens installed.

    Top-level ``event``/``temporal`` and ``spatial`` are CPU bool tensors with
    the same [1,L] shape as ``labels``; they are true only at official CE label
    positions. A caller may clone ``labels`` and set entries outside a chosen
    mask to -100, while passing every attention/position/context tensor through
    unchanged. ``ce_masks`` gives the same groups aligned to joint_loss's
    compact ``keep``/``targets`` list. Semantic, time, box, and structural
    classes are disjoint and cover all official CE targets, including MTP null
    padding. ``ptd_context_limits`` is intentionally not used to classify loss
    targets: it controls attention visibility, not CE supervision identity.
    """
    input_ids_t = _cpu_vector(data, "input_ids")
    labels_t = _cpu_vector(data, "labels")
    position_ids_t = _cpu_vector(data, "ptd_position_ids")
    if "ptd_prefix_lengths" in data:
        prefix_t = torch.as_tensor(data["ptd_prefix_lengths"]).detach().to(device="cpu").reshape(-1)
    elif "ptd_prefix_length" in data:
        prefix_t = torch.as_tensor(data["ptd_prefix_length"]).detach().to(device="cpu").reshape(-1)
    else:
        raise ValueError("PTD source sample is missing ptd_prefix_lengths")
    if prefix_t.numel() != 1:
        raise ValueError("split_source_loss_masks currently expects one source example")
    prefix_length = int(prefix_t[0])

    if input_ids_t.numel() != labels_t.numel() or input_ids_t.numel() != position_ids_t.numel():
        raise ValueError("input_ids, labels, and ptd_position_ids must have the same sequence length")
    if prefix_length <= 0 or prefix_length >= input_ids_t.numel():
        raise ValueError(f"invalid ptd_prefix_length={prefix_length}")

    input_ids = [int(v) for v in input_ids_t.tolist()]
    labels = [int(v) for v in labels_t.tolist()]
    ptd_position_ids = [int(v) for v in position_ids_t.tolist()]
    prefix_categories, spans = _prefix_response_spans(input_ids, labels, prefix_length, tokenizer)

    # Exactly mirror joint_loss: label positions are shifted one token from
    # their corresponding logit positions; label index zero is never scored.
    target_positions = [p for p in range(1, len(labels)) if labels[p] != IGNORE_INDEX]
    logit_positions = [p - 1 for p in target_positions]
    if not target_positions:
        raise ValueError("official PTD source sample has no CE targets")

    categories: list[str] = []
    mtp_blocks = 0
    for target_pos in target_positions:
        if target_pos < prefix_length:
            categories.append(prefix_categories[target_pos])
            continue
        if target_pos == prefix_length:
            raise ValueError("PTD MTP suffix must keep labels[prefix_length] at IGNORE_INDEX")
        predictor_pos = target_pos - 1
        suffix_offset = predictor_pos - prefix_length
        if suffix_offset < 0:
            raise ValueError("MTP target does not follow its PTD prefix")
        block_start = prefix_length + (suffix_offset // PTD_BLOCK_SIZE) * PTD_BLOCK_SIZE
        block_source_target_pos = ptd_position_ids[block_start] + 1
        if not (0 <= block_source_target_pos < prefix_length):
            raise ValueError("PTD MTP block does not map back into its teacher-forced source prefix")
        source_target_pos = ptd_position_ids[predictor_pos] + 1
        label_id = labels[target_pos]
        if label_id == spans["token_ids"]["null"]:
            # The official builder supervises MTP block padding as <null>.
            # Attach it to the source span that produced this block.
            categories.append(prefix_categories[block_source_target_pos])
        else:
            if not (0 <= source_target_pos < prefix_length):
                raise ValueError("PTD MTP target position does not map back into source prefix")
            categories.append(prefix_categories[source_target_pos])
        if suffix_offset % PTD_BLOCK_SIZE == 0:
            mtp_blocks += 1

    class_names = ("semantic", "time", "box", "structural")
    ce_masks = {
        name: torch.tensor([category == name for category in categories], dtype=torch.bool)
        for name in class_names
    }
    covered = torch.zeros(len(target_positions), dtype=torch.int8)
    for mask in ce_masks.values():
        covered += mask.to(torch.int8)
    if not bool(covered.eq(1).all()):
        raise AssertionError("source CE classes must be disjoint and cover each official target exactly once")
    ntp = torch.tensor([pos < prefix_length for pos in target_positions], dtype=torch.bool)
    ce_masks["temporal"] = ce_masks["semantic"] | ce_masks["time"]
    ce_masks["event"] = ce_masks["temporal"]
    ce_masks["spatial"] = ce_masks["box"]
    ce_masks["ntp"] = ntp
    ce_masks["mtp"] = ~ntp

    label_shape = (1, len(labels))
    position_masks = {
        name: torch.zeros(label_shape, dtype=torch.bool)
        for name in (*class_names, "event", "temporal", "spatial", "ntp", "mtp")
    }
    positions_tensor = torch.tensor(target_positions, dtype=torch.long)
    for name in class_names:
        position_masks[name][0, positions_tensor[ce_masks[name]]] = True
    for name in ("event", "temporal", "spatial", "ntp", "mtp"):
        position_masks[name][0, positions_tensor[ce_masks[name]]] = True

    group_masks = {
        "event": ce_masks["event"],
        "spatial": ce_masks["spatial"],
        "semantic": ce_masks["semantic"],
        "time": ce_masks["time"],
        "box": ce_masks["box"],
    }
    counts = {
        **{name: int(ce_masks[name].sum()) for name in (*class_names, "event", "temporal", "spatial", "ntp", "mtp")},
        "total_ce_targets": len(target_positions),
        "mtp_blocks": mtp_blocks,
    }
    for group_name, group_mask in group_masks.items():
        counts[f"{group_name}_ntp"] = int((group_mask & ntp).sum())
        counts[f"{group_name}_mtp"] = int((group_mask & ~ntp).sum())

    return {
        **position_masks,
        "ce_masks": ce_masks,
        "target_positions": positions_tensor,
        "logit_positions": torch.tensor(logit_positions, dtype=torch.long),
        "target_categories": tuple(categories),
        "counts": counts,
        "source_response_spans": spans,
    }
