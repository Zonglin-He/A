"""Bounded source-only PTD-4B adapter CE interface pilot.

Registration and CPU checks do not load model weights or use a GPU. ``pilot``
is deliberately a separate, explicitly invoked action; this task only
registers and CPU-checks the path. It trains a small Desta3DAdapter through
the stock visual.merger hook while leaving the official PTD response grammar,
decoder, and all 4B parameters frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from scripts.decota_matrix_common_v1 import read, sha, write

OUT = ROOT / "artifacts/desta3d_v1/adapter_train"
PARENT = ROOT / "artifacts/desta3d_v1"
CAP_SECONDS = 28_800
CAP_STAGE_SECONDS = 1_800
TRAIN_SEED = 20260927
PICK_SEED = "desta3d-adapter-source-train-v1|"
PILOT_KEYS = [
    "vidstg_source_query:25863",
    "vidstg_source_query:22341",
    "vidstg_source_query:33631",
    "vidstg_source_query:31098",
]


def json_read(path: Path | str) -> Any:
    return json.loads(Path(path).read_text())


def json_write_once(path: Path | str, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    assert not path.exists(), f"immutable artifact already exists: {path}"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_vidstg_builder():
    path = ROOT / "external/ParallelTubeDecoding/data/prepare_vidstg.py"
    spec = importlib.util.spec_from_file_location("desta3d_official_prepare_vidstg", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _candidate_rows() -> list[dict[str, Any]]:
    rows = json_read(PARENT / "SOURCE_INITIAL_INPUTS.json")
    labels = json_read(PARENT / "SOURCE_LABELS_TRAINING_ONLY.json")
    provenance = json_read(PARENT / "SOURCE_LABEL_PROVENANCE.json")["records"]
    builder = load_vidstg_builder()
    train = [r for r in rows if r.get("split") == "train"]
    assert len(train) == 95, f"expected the registered 95 source-train initial inputs; got {len(train)}"
    train.sort(key=lambda r: hashlib.sha256((PICK_SEED + r["key"]).encode()).hexdigest())
    eligible: list[dict[str, Any]] = []
    for row in train:
        key, q = row["key"], row["input"]
        lab = labels[key]
        assert lab["frame_ids"] == q["frame_ids"], f"source label/input frame mismatch: {key}"
        boxes: list[tuple[int, list[int]]] = []
        box_flags: list[dict[str, Any]] = []
        prov = provenance[key]
        assert prov["source"] == row["source"] and prov["split"] == "train"
        for pos, (fid, xyxy, valid, active) in enumerate(
            zip(lab["frame_ids"], lab["boxes_xyxy"], lab["box_valid"], lab["event_active"]),
            start=1,
        ):
            # A missing target observation is unknown visibility, never a negative.
            # Only positive target boxes within the known source event enter PTD SFT.
            if not (bool(valid) and bool(active)):
                continue
            xy_pixel = [
                float(xyxy[0]) * q["width"],
                float(xyxy[1]) * q["height"],
                float(xyxy[2]) * q["width"],
                float(xyxy[3]) * q["height"],
            ]
            norm = builder.normalize_box(xy_pixel, q["width"], q["height"])
            if norm is None:
                continue
            obs = [
                o
                for fr in prov["frames"]
                if int(fr["frame_id"]) == int(fid)
                for o in fr["observations"]
                if int(o["tid"]) == int(lab["target_id"]) and bool(o["is_referent"])
            ]
            assert len(obs) == 1, f"raw trajectory provenance is not unique for {key}, frame {fid}"
            boxes.append((pos, norm))
            box_flags.append(
                {
                    "sample_position_1based": pos,
                    "source_frame_id": int(fid),
                    "generated": int(obs[0]["generated"]),
                    "tracker": str(obs[0]["tracker"]),
                }
            )
        record = builder.build_record(q["video_path"], q["caption"].strip(), boxes)
        if record is None:
            continue
        eligible.append(
            {
                "key": key,
                "source": row["source"],
                "split": row["split"],
                "frame_count": len(q["frame_ids"]),
                "frame_ids": list(q["frame_ids"]),
                "target_id": int(lab["target_id"]),
                "category": lab["category"],
                "event_interval": lab["event_interval"],
                "supervised_positions_1based": [p for p, _ in boxes],
                "supervised_source_frame_ids": [int(q["frame_ids"][p - 1]) for p, _ in boxes],
                "box_count": len(boxes),
                "raw_referent_flags": box_flags,
                "label_semantics": lab["label_semantics"],
                "response": record["conversations"][1]["value"],
                "video_sha256": q["video_sha256"],
            }
        )
    return eligible


def selected_source_examples() -> list[dict[str, Any]]:
    candidates = _candidate_rows()
    eligible_by_key = {r["key"]: r for r in candidates}
    selected = [eligible_by_key[k] for k in PILOT_KEYS]
    assert len(selected) == 4
    assert all(x["split"] == "train" and x["box_count"] > 0 for x in selected)
    assert len({x["source"] for x in selected}) == 4
    expected = [
        [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        [1, 2, 3, 4, 5, 6, 7, 8, 9],
        list(range(1, 16)),
        list(range(2, 17)),
    ]
    assert [x["supervised_positions_1based"] for x in selected] == expected
    return selected


def gpu_receipts() -> list[Path]:
    return sorted(p for p in PARENT.rglob("*.json") if p.parent.name == "receipts")


def accumulated_gpu_seconds() -> tuple[float, list[dict[str, Any]]]:
    rows = []
    total = 0.0
    for path in gpu_receipts():
        payload = json_read(path)
        seconds = float(payload.get("seconds", 0.0))
        total += seconds
        rows.append({"path": str(path), "seconds": seconds, "stage": payload.get("stage")})
    return total, rows


def _pins() -> dict[str, str]:
    checkpoint_receipt = ROOT / "checkpoints/ParallelTubeDecoding-Qwen3-VL-4B/OFFICIAL_RECEIPT.json"
    pins = [
        Path(__file__),
        ROOT / "scripts/ptd_spatial_adapter_ab_v1.py",
        ROOT / "scripts/ptd_8b_teacher_feasibility_v1.py",
        ROOT / "vg_tta/desta3d_v1.py",
        ROOT / "external/ParallelTubeDecoding/src/dataset/sft_dataset.py",
        ROOT / "external/ParallelTubeDecoding/data/prepare_vidstg.py",
        PARENT / "SOURCE_INITIAL_INPUTS.json",
        PARENT / "SOURCE_LABELS_TRAINING_ONLY.json",
        PARENT / "SOURCE_LABEL_PROVENANCE.json",
        PARENT / "SOURCE_PREPARATION.json",
        PARENT / "BUDGET_AUTHORIZATION.json",
        ROOT / "methods/CURRENT_METHOD.json",
        checkpoint_receipt,
    ]
    return {str(p): sha(p) for p in pins}


def register() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lock_path = OUT / "TRAIN_INTERFACE_LOCK.json"
    if lock_path.exists():
        verify_lock()
        print(json.dumps(json_read(lock_path), indent=2))
        return

    chosen = selected_source_examples()
    selection = {
        "selection_rule": "first four PTD-builder-eligible rows in SHA256(sorted 95 initial source-train rows) order; no target/dev rows",
        "selection_seed": PICK_SEED,
        "examples": chosen,
        "source_label_use": "positive referent boxes with box_valid && event_active only; source raw trajectory flags preserved; no missing/unknown frame used as an absence label",
        "response_builder": "official ParallelTubeDecoding/data/prepare_vidstg.py build_record + original SupervisedDataset._append_ptd_targets",
        "target_GT_read": False,
    }
    selection_path = OUT / "SOURCE_TRAIN_SELECTION.json"
    if selection_path.exists():
        assert json_read(selection_path) == selection, "refusing to replace immutable source selection"
    else:
        json_write_once(selection_path, selection)

    prior_seconds, receipts = accumulated_gpu_seconds()
    assert prior_seconds < CAP_SECONDS, f"already-spent GPU time exceeds authorized cap: {prior_seconds}"
    checkpoint = json_read(ROOT / "checkpoints/ParallelTubeDecoding-Qwen3-VL-4B/OFFICIAL_RECEIPT.json")
    config = {
        "version": "desta3d_adapter_train_v1",
        "status": "registered_not_run",
        "task": "source-only supervised engineering pilot for frozen PTD-Qwen3-VL-4B through visual.merger",
        "data": {
            "selected_source_examples": PILOT_KEYS,
            "selected_parent_count": 4,
            "source_only": True,
            "target_GT_read": False,
            "validation_GT_used": False,
            "source_response_boxes": "known event-active and valid referent boxes in one contiguous sampled interval; no event-external absence labels",
            "query_input_contains_GT": False,
            "query_context": "exact frozen 2560-D mean of the original PTD prompt-prefill language hidden states after the last video token, captured once per source prompt with no grad; no response or target labels included",
            "reader_input": "canonical merger output [1,T,Hm,Wm,2560], exact 32-frame-or-fewer source clean input",
        },
        "model": {
            "checkpoint": checkpoint,
            "backbone_parameters_trainable": False,
            "backbone_decoder_and_grammar": "official stock PTD 4B model, unchanged",
            "adapter": {"class": "vg_tta.desta3d_v1.Desta3DAdapter", "in_channels": 2560, "query_dim": 2560, "hidden_dim": 256, "architecture": "dual3d", "hook": "model.model.visual.merger forward hook"},
            "optimizer_parameters": "adapter parameters only",
            "optimizer": {"name": "AdamW", "lr": 0.0001, "weight_decay": 0.0},
            "steps_per_source_example": 2,
            "source_examples": 4,
            "maximum_optimizer_steps": 8,
            "loss": "official PTD source response cross-entropy via scripts.ptd_8b_teacher_feasibility_v1.joint_loss (NTP+MTP sequence CE)",
            "gradient_checkpointing": {"enabled": True, "use_reentrant": False},
            "dropout": "model in train mode to activate non-reentrant gradient checkpointing; all nn.Dropout layers held in eval for deterministic exact no-op comparison",
            "inference_mode": False,
            "exact_noop": "compare stock and zero-initialized hooked source loss, NTP/MTP stats, and merger tokens before any optimizer step",
            "expected_inactive_decoder_ce_parameters": ["query_proj (feeds reader scalar heads only)", "referent_head", "event_head"],
            "reader_gradient_gate": "on each source example's second CE step, require nonzero gradients for input_proj, stem, short_reader, and long_reader; query_proj and scalar score heads are intentionally inactive under decoder CE",
        },
        "resource_authorization": {
            "cumulative_gpu_seconds_cap": CAP_SECONDS,
            "pilot_stage_gpu_seconds_cap": CAP_STAGE_SECONDS,
            "stage_cap_includes_model_load_decode_training_and_checks": True,
            "includes_all_prior_attempts_and_failures": True,
            "seconds_recorded_before_this_registration": prior_seconds,
            "prior_receipts": receipts,
            "no_gpu_invoked_by_register_or_cpu_check": True,
        },
        "scope": {
            "engineering_interface_only": True,
            "not_an_E2_information_probe_result": True,
            "not_target_TTA_or_deployable_method": True,
            "later_E5_E6_required_comparison": ["native frozen PTD", "source-trained adapter without TTA", "source-trained adapter plus separately specified TTA"],
            "interpretation": "source-fit improvement over native frozen PTD alone cannot be attributed to TTA; E2 representation probe success does not establish decoder adaptation efficacy",
        },
    }
    config_path = OUT / "ADAPTER_TRAIN_CONFIG.json"
    if config_path.exists():
        assert json_read(config_path) == config, "refusing to replace immutable training config"
    else:
        json_write_once(config_path, config)

    lock = {
        "created_unix": time.time(),
        "status": "ready_for_parent_review_and_explicit_GPU_invocation; not run",
        "script_sha256": sha(Path(__file__)),
        "config_sha256": sha(config_path),
        "selection_sha256": sha(selection_path),
        "pins": _pins(),
        "target_GT_read": False,
        "GPU_started": False,
    }
    json_write_once(lock_path, lock)
    print(json.dumps(lock, indent=2))


def verify_lock() -> dict[str, Any]:
    lock = json_read(OUT / "TRAIN_INTERFACE_LOCK.json")
    for path, digest in lock["pins"].items():
        assert sha(path) == digest, f"registered dependency changed: {path}"
    assert sha(Path(__file__)) == lock["script_sha256"], "training script changed after lock"
    assert sha(OUT / "ADAPTER_TRAIN_CONFIG.json") == lock["config_sha256"]
    assert sha(OUT / "SOURCE_TRAIN_SELECTION.json") == lock["selection_sha256"]
    cfg = json_read(OUT / "ADAPTER_TRAIN_CONFIG.json")
    assert cfg["resource_authorization"]["cumulative_gpu_seconds_cap"] == CAP_SECONDS
    assert cfg["resource_authorization"]["pilot_stage_gpu_seconds_cap"] == CAP_STAGE_SECONDS
    assert cfg["model"]["maximum_optimizer_steps"] == 8
    return cfg


def _check_selection_and_grammar(processor) -> dict[str, Any]:
    from scripts.ptd_8b_teacher_feasibility_v1 import append_targets

    chosen = selected_source_examples()
    assert [x["key"] for x in chosen] == PILOT_KEYS
    checked = []
    for item in chosen:
        prompt_ids = processor.tokenizer.encode("prompt", add_special_tokens=False)
        response_ids = processor.tokenizer.encode(item["response"] + "<|im_end|>\n", add_special_tokens=False)
        start = len(prompt_ids)
        input_ids = torch.tensor(prompt_ids + response_ids, dtype=torch.long)
        labels = torch.tensor([-100] * start + response_ids, dtype=torch.long)
        mm = torch.zeros_like(input_ids)
        data = {
            "input_ids": input_ids,
            "labels": labels,
            "mm_token_type_ids": mm,
            "video_grid_thw": torch.tensor([[item["frame_count"], 32, 32]], dtype=torch.long),
            "pixel_values_videos": torch.empty((0,)),
        }
        full = append_targets(processor, data, start)
        assert full["ptd_prefix_length"].item() == len(input_ids)
        assert full["input_ids"].numel() == full["labels"].numel()
        assert full["ptd_position_ids"].numel() == full["input_ids"].numel()
        assert int(full["labels"].ne(-100).sum()) > 0
        checked.append({"key": item["key"], "source": item["source"], "target_tokens": int(full["labels"].ne(-100).sum()), "ptd_tokens": int(full["ptd_position_ids"].numel() - len(input_ids))})
    return {"selected_examples": checked, "official_append_targets_cpu": True}


def cpu_check() -> None:
    cfg = verify_lock()
    from vg_tta.desta3d_v1 import Desta3DAdapter

    torch.manual_seed(TRAIN_SEED)
    adapter = Desta3DAdapter(in_channels=16, query_dim=16, hidden_dim=8, architecture="dual3d").eval()
    x = torch.randn(1, 3, 2, 2, 16)
    q = torch.randn(1, 16)
    with torch.no_grad():
        y = adapter(x, q)
    assert torch.equal(y["updated_tokens"], x), "adapter must be exact identity at zero init"
    assert y["updated_tokens"].shape == x.shape
    from scripts.ptd_spatial_adapter_ab_v1 import processor_load

    processor = processor_load()
    grammar = _check_selection_and_grammar(processor)
    payload = {
        "time_unix": time.time(),
        "GPU_used": False,
        "model_weights_loaded": False,
        "8B_weights_loaded": False,
        "source_selection_and_contiguous_box_gate": True,
        "source_label_unknowns_as_absence": False,
        "adapter_cpu_exact_noop": True,
        **grammar,
        "registered_config_sha256": sha(OUT / "ADAPTER_TRAIN_CONFIG.json"),
    }
    path = OUT / "CPU_CHECK.json"
    if path.exists():
        old = json_read(path)
        for key, value in payload.items():
            if key != "time_unix":
                assert old.get(key) == value, f"CPU check changed for {key}"
    else:
        json_write_once(path, payload)
    print(json.dumps(payload, indent=2))


def _processor_prompt_kwargs(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    keep = {
        "input_ids",
        "attention_mask",
        "mm_token_type_ids",
        "pixel_values_videos",
        "video_grid_thw",
        "second_per_grid_ts",
    }
    return {k: v for k, v in inputs.items() if k in keep}


def capture_exact_query(model, prompt_inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    """Capture the source query mean from an exact frozen prompt prefill."""
    model_inputs = _processor_prompt_kwargs(prompt_inputs)
    input_ids = model_inputs["input_ids"]
    video_mask = input_ids[0].eq(model.config.video_token_id)
    positions = video_mask.nonzero(as_tuple=False).flatten()
    assert len(positions), "video token absent in source prompt"
    query_mask = torch.arange(input_ids.shape[1], device=input_ids.device) > positions[-1]
    query_mask &= model_inputs["attention_mask"][0].bool()
    assert query_mask.any(), "source prompt contains no post-video instruction tokens"
    captured: dict[str, torch.Tensor] = {}

    def language_hook(module, args, output):
        h = output.last_hidden_state
        assert h.shape[1] == input_ids.shape[1]
        captured["query"] = h[0, query_mask].float().mean(0).detach()

    handle = model.model.language_model.register_forward_hook(language_hook)
    try:
        with torch.no_grad():
            model.model(**model_inputs, use_cache=False)
    finally:
        handle.remove()
    assert "query" in captured and captured["query"].numel() == 2560
    assert torch.isfinite(captured["query"]).all()
    return captured["query"].unsqueeze(0)


def build_sft_data(processor, row: dict[str, Any], selection: dict[str, Any]):
    from scripts.ptd_spatial_adapter_ab_v1 import frames_for, inputs_for
    from scripts.ptd_8b_teacher_feasibility_v1 import append_targets

    frames, frame_ids = frames_for(row, "clean")
    assert frame_ids == row["input"]["frame_ids"]
    prompt_inputs, preprocess = inputs_for(row, processor, frames)
    prompt_ids = prompt_inputs["input_ids"][0].detach().cpu()
    response_ids = processor.tokenizer.encode(selection["response"] + "<|im_end|>\n", add_special_tokens=False)
    start = int(prompt_ids.numel())
    response_tensor = torch.tensor(response_ids, dtype=torch.long)
    labels = torch.cat((torch.full((start,), -100, dtype=torch.long), response_tensor))
    mm = prompt_inputs.get("mm_token_type_ids", torch.zeros_like(prompt_inputs["input_ids"]))[0].detach().cpu()
    mm = torch.cat((mm, torch.zeros(len(response_ids), dtype=mm.dtype)))
    data = {
        "input_ids": torch.cat((prompt_ids, response_tensor)),
        "labels": labels,
        "mm_token_type_ids": mm,
        "video_grid_thw": prompt_inputs["video_grid_thw"].detach().cpu(),
        "pixel_values_videos": prompt_inputs["pixel_values_videos"].detach().cpu(),
    }
    data = append_targets(processor, data, start)
    data["ptd_prefix_lengths"] = data.pop("ptd_prefix_length").unsqueeze(0)
    for key in ("input_ids", "labels", "mm_token_type_ids", "attention_mask", "ptd_position_ids", "ptd_context_limits"):
        data[key] = data[key].unsqueeze(0)
    data = {k: v.to("cuda") if torch.is_tensor(v) else v for k, v in data.items()}
    meta = {
        "key": row["key"],
        "source": row["source"],
        "frame_count": len(frame_ids),
        "frame_ids": frame_ids,
        "grid": data["video_grid_thw"].detach().cpu().tolist(),
        "pixel_sha": preprocess["pixel_sha"],
        "prompt_tokens": start,
        "response_tokens": len(response_ids),
        "supervised_label_tokens": int(data["labels"].ne(-100).sum()),
        "source_gt_used": True,
        "target_gt_read": False,
    }
    return data, prompt_inputs, meta


def adapter_group_gradients(adapter) -> dict[str, Any]:
    groups: dict[str, dict[str, Any]] = {}
    for name, param in adapter.named_parameters():
        group_name = name.split(".", 1)[0]
        row = groups.setdefault(group_name, {"parameters": 0, "grad_none": 0, "nonzero_grad_parameters": 0, "grad_norm_sq": 0.0})
        row["parameters"] += param.numel()
        if param.grad is None:
            row["grad_none"] += param.numel()
        else:
            norm = float(param.grad.detach().float().norm())
            row["grad_norm_sq"] += norm * norm
            if norm > 0:
                row["nonzero_grad_parameters"] += param.numel()
    for row in groups.values():
        row["grad_norm"] = math.sqrt(row.pop("grad_norm_sq"))
    return groups


def _assert_budget(started: float) -> tuple[float, list[dict[str, Any]]]:
    spent, receipts = accumulated_gpu_seconds()
    elapsed = time.monotonic() - started
    assert spent + elapsed < CAP_SECONDS, f"cumulative GPU cap reached: completed={spent:.3f}s running={elapsed:.3f}s cap={CAP_SECONDS}s"
    assert elapsed < CAP_STAGE_SECONDS, f"source adapter pilot stage cap reached: elapsed={elapsed:.3f}s cap={CAP_STAGE_SECONDS}s"
    return spent, receipts


def run_pilot() -> None:
    cfg = verify_lock()
    assert torch.cuda.is_available(), "pilot requires the explicitly available GPU"
    assert not (OUT / "RUNNING.json").exists(), "a pilot run was already started; inspect its state/receipt"
    assert not (OUT / "PILOT_COMPLETE.json").exists(), "pilot already completed"
    initial_spend, _ = _assert_budget(time.monotonic())
    json_write_once(OUT / "RUNNING.json", {"started_unix": time.time(), "started_gpu_seconds_before_run": initial_spend, "maximum_optimizer_steps": 8})

    from scripts.ptd_spatial_adapter_ab_v1 import processor_load, model_load
    from scripts.ptd_8b_teacher_feasibility_v1 import joint_loss
    from vg_tta.desta3d_v1 import Desta3DAdapter, reshape_merged_video_tokens

    started = time.monotonic()
    history: list[dict[str, Any]] = []
    receipts = OUT / "receipts"
    receipts.mkdir(parents=True, exist_ok=True)
    try:
        torch.manual_seed(TRAIN_SEED)
        torch.set_num_threads(4)
        torch.cuda.reset_peak_memory_stats()
        processor = processor_load()
        model = model_load()
        model.requires_grad_(False)
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.train()
        # Gradient checkpointing is gated on training mode in Transformers;
        # disable stochastic dropout explicitly to keep the exact no-op check.
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.eval()
        assert all(not p.requires_grad for p in model.parameters())
        frozen_versions = {name: param._version for name, param in model.named_parameters()}

        source_rows = {r["key"]: r for r in json_read(PARENT / "SOURCE_INITIAL_INPUTS.json") if r.get("split") == "train"}
        selected_rows = {r["key"]: r for r in json_read(OUT / "SOURCE_TRAIN_SELECTION.json")["examples"]}
        adapter = Desta3DAdapter(in_channels=2560, query_dim=2560, hidden_dim=256, architecture="dual3d").cuda().eval()
        optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-4, weight_decay=0.0)
        no_op_record = None

        for source_index, key in enumerate(PILOT_KEYS):
            _assert_budget(started)
            row = source_rows[key]
            selection = selected_rows[key]
            data, prompt_inputs, meta = build_sft_data(processor, row, selection)
            query = capture_exact_query(model, prompt_inputs)
            query = query.detach().float()
            times = torch.tensor(meta["frame_ids"], dtype=torch.float32, device="cuda")[None] / float(row["input"]["fps"])
            hook_stats: dict[str, Any] = {"calls": 0, "last_delta_max_abs": None, "last_delta_norm": None}
            stock_tokens: dict[str, torch.Tensor] = {}
            hooked_tokens: dict[str, torch.Tensor] = {}

            if no_op_record is None:
                def stock_merger_hook(module, args, output):
                    stock_tokens["tokens"] = output.detach().cpu().clone()

                stock_handle = model.model.visual.merger.register_forward_hook(stock_merger_hook)
                try:
                    with torch.no_grad():
                        baseline_loss, baseline_stats = joint_loss(model, data)
                finally:
                    stock_handle.remove()
                assert "tokens" in stock_tokens

            def merger_hook(module, args, output):
                grid = reshape_merged_video_tokens(output.float(), data["video_grid_thw"][0])
                result = adapter(grid, query, frame_times=times)
                delta = result["delta"]
                hook_stats["calls"] += 1
                hook_stats["last_delta_max_abs"] = float(delta.detach().abs().max())
                hook_stats["last_delta_norm"] = float(delta.detach().float().norm())
                updated = result["updated_tokens"].reshape_as(output).to(output.dtype)
                if no_op_record is None:
                    hooked_tokens["input"] = output.detach().cpu().clone()
                    hooked_tokens["output"] = updated.detach().cpu().clone()
                return updated

            handle = model.model.visual.merger.register_forward_hook(merger_hook)
            try:
                if no_op_record is None:
                    zero_delta = adapter.out_proj.weight.detach().eq(0).all() and adapter.out_proj.bias.detach().eq(0).all()
                    assert bool(zero_delta), "adapter output projection must begin zero initialized"
                    hook_stats["calls"] = 0
                    with torch.no_grad():
                        zero_loss, zero_stats = joint_loss(model, data)
                    assert hook_stats["calls"] == 1
                    assert float(hook_stats["last_delta_max_abs"]) == 0.0
                    assert torch.equal(stock_tokens["tokens"], hooked_tokens["input"]), "stock merger tokens changed between baseline and zero-hook passes"
                    assert torch.equal(stock_tokens["tokens"], hooked_tokens["output"]), "zero adapter changed stock merger tokens"
                    assert torch.equal(baseline_loss, zero_loss), "zero adapter changed source PTD loss"
                    assert baseline_stats == zero_stats, "zero adapter changed NTP/MTP CE statistics"
                    no_op_record = {
                        "key": key,
                        "source": row["source"],
                        "exact_loss_equal": True,
                        "exact_ntp_mtp_stats_equal": True,
                        "stock_and_zero_hook_merger_tokens_equal": True,
                        "zero_delta_max_abs": 0.0,
                        "baseline_loss": float(baseline_loss),
                        "baseline_stats": baseline_stats,
                        "merger_hook_calls": hook_stats["calls"],
                    }

                for local_step in range(2):
                    _assert_budget(started)
                    if local_step == 1:
                        assert bool(adapter.out_proj.weight.detach().ne(0).any() or adapter.out_proj.bias.detach().ne(0).any()), "first AdamW update did not move zero-initialized out_proj"
                    assert torch.is_grad_enabled() and not torch.is_inference_mode_enabled(), "source CE training must use autograd, not inference_mode"
                    optimizer.zero_grad(set_to_none=True)
                    hook_stats["calls"] = 0
                    loss, stats = joint_loss(model, data)
                    assert torch.isfinite(loss), "source CE must be finite"
                    loss.backward()
                    assert hook_stats["calls"] == 1, "adapter hook must run exactly once per source forward"
                    base_grad_none = all(p.grad is None for p in model.parameters())
                    assert base_grad_none, "frozen PTD backbone received a gradient"
                    assert any(p.grad is not None and float(p.grad.detach().float().norm()) > 0 for p in adapter.out_proj.parameters()), "source CE did not train adapter output projection"
                    gradients = adapter_group_gradients(adapter)
                    if local_step == 1:
                        for group_name in ("input_proj", "stem", "short_reader", "long_reader"):
                            assert gradients.get(group_name, {}).get("grad_norm", 0.0) > 0.0, f"nonzero source CE gradient failed to reach {group_name} after the first out_proj update"
                    optimizer.step()
                    torch.cuda.synchronize()
                    assert all(param._version == frozen_versions[name] for name, param in model.named_parameters()), "frozen PTD weights changed"
                    completed_spend, current_receipts = _assert_budget(started)
                    record = {
                        **meta,
                        "source_index": source_index,
                        "step_within_source": local_step + 1,
                        "global_optimizer_step": len(history) + 1,
                        "source_ce": float(loss.detach()),
                        **stats,
                        "adapter_gradient_groups": gradients,
                        "intentionally_inactive_under_decoder_ce": ["query_proj", "referent_head", "event_head"],
                        "backbone_all_grad_none": base_grad_none,
                        "backbone_versions_unchanged": True,
                        "hook_calls": hook_stats["calls"],
                        "delta_max_abs_before_optimizer_step": hook_stats["last_delta_max_abs"],
                        "delta_norm_before_optimizer_step": hook_stats["last_delta_norm"],
                        "seconds_elapsed_this_run": time.monotonic() - started,
                        "gpu_seconds_completed_before_run": completed_spend,
                        "gpu_seconds_receipts_seen": current_receipts,
                        "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
                    }
                    history.append(record)
                    print("DESTA3D_ADAPTER_SOURCE_CE_STEP", json.dumps(record, sort_keys=True), flush=True)
                    del loss
                handle.remove()
            finally:
                if handle.id in getattr(model.model.visual.merger, "_forward_hooks", {}):
                    handle.remove()
            del data, prompt_inputs, query, times
            torch.cuda.empty_cache()

        assert len(history) == 8
        assert no_op_record is not None
        assert all(param._version == frozen_versions[name] for name, param in model.named_parameters())
        adapter_state = {name: tensor.detach().cpu() for name, tensor in adapter.state_dict().items()}
        adapter_path = OUT / "ADAPTER.pt"
        torch.save(adapter_state, adapter_path)
        restored = torch.load(adapter_path, map_location="cpu", weights_only=True)
        assert adapter_state.keys() == restored.keys() and all(torch.equal(adapter_state[k], restored[k]) for k in adapter_state)
        spent_after, final_receipts = accumulated_gpu_seconds()
        assert spent_after + (time.monotonic() - started) < CAP_SECONDS
        complete = {
            "completed_unix": time.time(),
            "status": "source_training_interface_pilot_complete_engineering_only",
            "registered_lock_sha256": sha(OUT / "TRAIN_INTERFACE_LOCK.json"),
            "config_sha256": sha(OUT / "ADAPTER_TRAIN_CONFIG.json"),
            "source_selected_keys": PILOT_KEYS,
            "steps": history,
            "zero_adapter_exact_noop": no_op_record,
            "adapter_sha256": sha(adapter_path),
            "adapter_state_readback_exact": True,
            "backbone_grad_none_and_unchanged": True,
            "target_GT_read": False,
            "deployable_TTA_claim": False,
            "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
            "worker_elapsed_seconds": time.monotonic() - started,
            "gpu_seconds_prior_to_this_run": spent_after,
            "cumulative_receipt_cap_seconds": CAP_SECONDS,
            "cumulative_receipts": final_receipts,
        }
        json_write_once(OUT / "PILOT_COMPLETE.json", complete)
    except BaseException as exc:
        failure = {"failed_unix": time.time(), "error": repr(exc), "traceback": traceback.format_exc(), "target_GT_read": False, "completed_steps": len(history)}
        json_write_once(OUT / "FAILURE.json", failure)
        raise
    finally:
        prior, _ = accumulated_gpu_seconds()
        elapsed = time.monotonic() - started
        receipt = {
            "stage": "desta3d_source_adapter_interface_pilot",
            "seconds": elapsed,
            "prior_seconds": prior,
            "steps": len(history),
            "peak_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0,
            "target_GT_read": False,
        }
        json_write_once(receipts / f"pilot_{time.time_ns()}.json", receipt)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["register", "cpu-check", "pilot"])
    args = parser.parse_args()
    if args.action == "register":
        register()
    elif args.action == "cpu-check":
        cpu_check()
    else:
        run_pilot()


if __name__ == "__main__":
    main()
