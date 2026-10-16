#!/usr/bin/env python3
"""Run the 64-video HC-STVG2 frozen versus episodic-TTA feasibility study."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vg_tta.augmentations import apply_condition, make_strong_view
from vg_tta.metrics import compute_stvg_metrics, interval_from_logits, paired_bootstrap_ci
from vg_tta.tta import EpisodicHeadAdapter, teacher_interval_mask
from vg_tta.tubedetr_runtime import (
    add_repo_to_path,
    build_model,
    dataset_args,
    forward_video,
    load_official_checkpoint,
)


METRIC_KEYS = ("tIoU", "sIoU", "vIoU_corrected", "vIoU_legacy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=str(PROJECT_ROOT / "external" / "TubeDETR"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--annotation-root", required=True)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "artifacts"))
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--preflight-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--severity", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--conditions", nargs="+", default=["clean", "low_light", "blur"])
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    return parser.parse_args()


def preflight_paths(args: argparse.Namespace) -> None:
    required = [
        Path(args.checkpoint),
        Path(args.repo) / "models" / "tubedetr.py",
        Path(args.video_root) / "video",
        Path(args.annotation_root) / "valv2_proc.json",
    ]
    missing = [str(path.resolve()) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "required checkpoint/data paths are missing; refusing to substitute a dataset:\n"
            + "\n".join(missing)
        )


def load_full_video(sample: tuple[Any, ...]) -> tuple[torch.Tensor, list[dict], dict]:
    if len(sample) == 4:
        _, targets, video_target, full_video = sample
    elif len(sample) == 3:
        full_video, targets, video_target = sample
    else:
        raise ValueError(f"unexpected HC-STVG sample tuple length: {len(sample)}")
    if full_video.ndim != 4 or full_video.shape[0] != 3:
        raise ValueError(f"invalid video tensor shape: {tuple(full_video.shape)}")
    if not torch.isfinite(full_video).all():
        raise FloatingPointError("video contains NaN/Inf after preprocessing")
    if len(targets) != full_video.shape[1]:
        raise ValueError("target count does not match sampled frame count")
    start, end = video_target["inter_idx"]
    if not (0 <= start <= end < len(targets)):
        raise ValueError(f"invalid ground-truth temporal interval: {(start, end)}")
    tube_box_count = 0
    for target in targets:
        boxes = target.get("boxes")
        if boxes is None or len(boxes) == 0:
            continue
        tube_box_count += len(boxes)
        if not torch.isfinite(boxes).all():
            raise FloatingPointError("ground-truth boxes contain NaN/Inf")
        minimum = boxes.min().item()
        maximum = boxes.max().item()
        if minimum < -1e-6 or maximum > 1.0 + 1e-6:
            raise ValueError("normalized ground-truth box coordinates are outside [0, 1]")
        # torchvision arithmetic can put an exact image-boundary coordinate at
        # 1 + one FP32 ULP. Normalize that representation without accepting a
        # materially out-of-frame annotation.
        if minimum < 0.0 or maximum > 1.0:
            boxes.clamp_(0.0, 1.0)
    if tube_box_count == 0:
        raise ValueError("ground-truth tube is empty after preprocessing")
    return full_video, targets, video_target


def prediction_record(
    outputs: dict[str, torch.Tensor],
    targets: list[dict],
    video_target: dict,
    annotation: dict,
    *,
    sample_index: int,
    condition: str,
    method: str,
    runtime_sec: float,
    peak_vram_gb: float,
    losses: dict[str, float] | None = None,
) -> dict[str, Any]:
    from vg_tta.metrics import TEMPORAL_DECODER_VERSION

    pred_indices = interval_from_logits(outputs["pred_sted"])
    gt_indices = tuple(video_target["inter_idx"])
    metrics = compute_stvg_metrics(
        outputs["pred_boxes"],
        targets,
        pred_indices,
        gt_indices,
        frame_ids=video_target["frames_id"],
        gt_frame_interval=(annotation["tube_start_frame"], annotation["tube_end_frame"]),
    )
    record: dict[str, Any] = {
        "sample_index": sample_index,
        "id": video_target["video_id"],
        "query": video_target["caption"],
        "condition": condition,
        "method": method,
        **metrics,
        "peak_vram_gb": peak_vram_gb,
        "runtime_sec": runtime_sec,
        "temporal_decoder_version": TEMPORAL_DECODER_VERSION,
        # Query-only evaluation evidence, never consumed by adaptation. Retain
        # exact stored logits/boxes so future evaluator audits need no GPU rerun.
        "metric_replay": {
            "pred_sted": outputs["pred_sted"].detach().double().cpu().tolist(),
            "pred_boxes": outputs["pred_boxes"].detach().float().cpu().tolist(),
            "target_boxes": [
                target.get("boxes", torch.empty(0, 4)).detach().float().cpu().tolist()
                for target in targets
            ],
            "frame_ids": [int(value) for value in video_target["frames_id"]],
            "gt_indices": list(gt_indices),
            "gt_frame_interval": [int(annotation["tube_start_frame"]), int(annotation["tube_end_frame"])],
        },
    }
    if losses is not None:
        record["tta_losses"] = losses
    return record


def timed_forward(
    model: torch.nn.Module,
    video: torch.Tensor,
    caption: str,
    args: argparse.Namespace,
) -> tuple[dict[str, torch.Tensor], float, float]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        outputs = forward_video(
            model,
            video,
            caption,
            repo=args.repo,
            stride=args.stride,
            device="cuda",
        )
    torch.cuda.synchronize()
    return outputs, time.perf_counter() - start, torch.cuda.max_memory_allocated() / 1024**3


def run_tta(
    model: torch.nn.Module,
    adapter: EpisodicHeadAdapter,
    video: torch.Tensor,
    caption: str,
    args: argparse.Namespace,
    *,
    augmentation_seed: int,
) -> tuple[dict[str, torch.Tensor], dict[str, float], float, float]:
    adapter.reset()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    try:
        with torch.no_grad():
            teacher = forward_video(
                model, video, caption, repo=args.repo, stride=args.stride, device="cuda"
            )
        strong = make_strong_view(video, seed=augmentation_seed)
        student = forward_video(
            model, strong, caption, repo=args.repo, stride=args.stride, device="cuda"
        )
        time_mask = torch.ones_like(student["pred_sted"][..., 0], dtype=torch.bool)
        box_mask = teacher_interval_mask(teacher["pred_sted"], time_mask=time_mask)
        losses = adapter.step(
            student,
            teacher,
            time_mask=time_mask,
            box_mask=box_mask,
        )
        with torch.no_grad():
            prediction = forward_video(
                model, video, caption, repo=args.repo, stride=args.stride, device="cuda"
            )
        torch.cuda.synchronize()
        runtime = time.perf_counter() - start
        peak = torch.cuda.max_memory_allocated() / 1024**3
        return prediction, losses, runtime, peak
    finally:
        adapter.reset()


def validate_output(outputs: dict[str, torch.Tensor], expected_frames: int) -> None:
    for key in ("pred_boxes", "pred_sted"):
        if key not in outputs or not torch.isfinite(outputs[key]).all():
            raise FloatingPointError(f"missing or non-finite {key}")
    if outputs["pred_boxes"].shape != (expected_frames, 4):
        raise ValueError(f"unexpected pred_boxes shape: {tuple(outputs['pred_boxes'].shape)}")
    if outputs["pred_sted"].shape != (1, expected_frames, 2):
        raise ValueError(f"unexpected pred_sted shape: {tuple(outputs['pred_sted'].shape)}")
    if outputs["pred_boxes"].min().item() < 0.0 or outputs["pred_boxes"].max().item() > 1.0:
        raise ValueError("normalized predicted box coordinates are outside [0, 1]")


def summarize(
    records: list[dict[str, Any]],
    output_dir: Path,
    bootstrap_samples: int,
    seed: int,
) -> None:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["condition"], record["method"])].append(record)
    rows = []
    for (condition, method), values in sorted(grouped.items()):
        rows.append(
            {
                "condition": condition,
                "method": method,
                "num_samples": len(values),
                "mean_tIoU": np.mean([value["tIoU"] for value in values]),
                "mean_sIoU": np.mean([value["sIoU"] for value in values]),
                "mean_vIoU": np.mean([value["vIoU"] for value in values]),
                "mean_vIoU_legacy": np.mean([value["vIoU_legacy"] for value in values]),
                "vIoU_at_0.3": np.mean([value["vIoU_at_0.3"] for value in values]),
                "vIoU_at_0.5": np.mean([value["vIoU_at_0.5"] for value in values]),
                "mean_runtime_sec": np.mean([value["runtime_sec"] for value in values]),
                "peak_vram_gb": np.max([value["peak_vram_gb"] for value in values]),
            }
        )
    with (output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    bootstrap: dict[str, Any] = {}
    for condition in sorted({record["condition"] for record in records}):
        frozen = {
            record["sample_index"]: record
            for record in records
            if record["condition"] == condition and record["method"] == "frozen"
        }
        tta = {
            record["sample_index"]: record
            for record in records
            if record["condition"] == condition and record["method"] == "consistency_tta"
        }
        shared = sorted(set(frozen) & set(tta))
        bootstrap[condition] = {
            metric: paired_bootstrap_ci(
                [tta[index][metric] for index in shared],
                [frozen[index][metric] for index in shared],
                n_bootstrap=bootstrap_samples,
                seed=seed,
            )
            for metric in METRIC_KEYS
        }
    (output_dir / "bootstrap_ci.json").write_text(
        json.dumps(bootstrap, indent=2, ensure_ascii=False) + "\n"
    )


def main() -> None:
    args = parse_args()
    preflight_paths(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    add_repo_to_path(args.repo)
    from datasets import build_dataset

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model, model_args = build_model(
        args.repo,
        device="cuda",
        resolution=args.resolution,
        stride=args.stride,
    )
    checkpoint_report = load_official_checkpoint(model, args.checkpoint)
    if checkpoint_report["missing_keys"] or checkpoint_report["unexpected_keys"]:
        raise RuntimeError(f"checkpoint did not restore cleanly: {checkpoint_report}")
    model.eval()
    adapter = EpisodicHeadAdapter(model, learning_rate=args.lr)
    data_args = dataset_args(model_args, args.video_root, args.annotation_root)
    dataset = build_dataset("hcstvg", image_set="val", args=data_args)

    preflight = []
    for index in range(min(args.preflight_samples, len(dataset))):
        video, targets, video_target = load_full_video(dataset[index])
        preflight.append(
            {
                "sample_index": index,
                "id": video_target["video_id"],
                "frames": int(video.shape[1]),
                "shape": list(video.shape),
                "gt_interval_indices": video_target["inter_idx"],
                "target_frames_with_boxes": sum(bool(len(target["boxes"])) for target in targets),
            }
        )
    (output_dir / "data_preflight.json").write_text(
        json.dumps(preflight, indent=2, ensure_ascii=False) + "\n"
    )

    rng = np.random.default_rng(args.seed)
    count = min(args.num_samples, len(dataset))
    indices = sorted(rng.choice(len(dataset), size=count, replace=False).tolist())
    (output_dir / "subset_indices.json").write_text(json.dumps(indices, indent=2) + "\n")
    frozen_path = output_dir / "frozen_predictions.jsonl"
    tta_path = output_dir / "tta_predictions.jsonl"
    records: list[dict[str, Any]] = []
    with frozen_path.open("w") as frozen_file, tta_path.open("w") as tta_file:
        for position, index in enumerate(indices):
            full_video, targets, video_target = load_full_video(dataset[index])
            annotation = dataset.annotations[index]
            for condition_index, condition in enumerate(args.conditions):
                shifted = apply_condition(full_video, condition, severity=args.severity)
                adapter.reset()
                frozen_outputs, runtime, peak = timed_forward(
                    model, shifted, video_target["caption"], args
                )
                validate_output(frozen_outputs, shifted.shape[1])
                frozen_record = prediction_record(
                    frozen_outputs,
                    targets,
                    video_target,
                    annotation,
                    sample_index=index,
                    condition=condition,
                    method="frozen",
                    runtime_sec=runtime,
                    peak_vram_gb=peak,
                )
                frozen_file.write(json.dumps(frozen_record, ensure_ascii=False) + "\n")
                frozen_file.flush()
                records.append(frozen_record)

                tta_outputs, losses, runtime, peak = run_tta(
                    model,
                    adapter,
                    shifted,
                    video_target["caption"],
                    args,
                    augmentation_seed=args.seed + index * 10 + condition_index,
                )
                validate_output(tta_outputs, shifted.shape[1])
                if peak > 31:
                    raise RuntimeError(f"head-only TTA exceeded 31 GB on sample {index}")
                tta_record = prediction_record(
                    tta_outputs,
                    targets,
                    video_target,
                    annotation,
                    sample_index=index,
                    condition=condition,
                    method="consistency_tta",
                    runtime_sec=runtime,
                    peak_vram_gb=peak,
                    losses=losses,
                )
                tta_file.write(json.dumps(tta_record, ensure_ascii=False) + "\n")
                tta_file.flush()
                records.append(tta_record)
                print(
                    f"[{position + 1}/{len(indices)}] {index=} {condition=} "
                    f"frozen_vIoU={frozen_record['vIoU']:.4f} "
                    f"tta_vIoU={tta_record['vIoU']:.4f}"
                )
    summarize(records, output_dir, args.bootstrap_samples, args.seed)
    print(f"wrote feasibility artifacts to {output_dir}")


if __name__ == "__main__":
    main()
