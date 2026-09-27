#!/usr/bin/env python3
"""Phase-1 evaluator controls and recoverability-oracle experiments."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_feasibility import load_full_video, prediction_record, validate_output
from vg_tta.augmentations import apply_condition, make_strong_view
from vg_tta.metrics import cluster_paired_bootstrap_ci, paired_bootstrap_ci
from vg_tta.tta import EpisodicHeadAdapter, UPDATE_SCOPES, teacher_interval_mask
from vg_tta.tubedetr_runtime import (
    add_repo_to_path,
    build_model,
    dataset_args,
    decode_video,
    encode_video,
    forward_video,
    load_official_checkpoint,
)


METRICS = ("tIoU", "sIoU", "vIoU_corrected", "vIoU_legacy")
ORACLES = ("clean_teacher", "gt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("controls", "oracles"), required=True)
    parser.add_argument("--repo", default=str(PROJECT_ROOT / "external" / "TubeDETR"))
    parser.add_argument(
        "--checkpoint",
        default=str(PROJECT_ROOT / "checkpoints" / "tubedetr_hcstvg2_res224_stride2.pth"),
    )
    parser.add_argument("--video-root", default=str(PROJECT_ROOT / "data" / "hcstvg2_subset"))
    parser.add_argument(
        "--annotation-root",
        default=str(PROJECT_ROOT / "data" / "hcstvg2_subset" / "annotations"),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument(
        "--dataset-role",
        choices=("development", "confirmation"),
        default="development",
        help="Evidence role written into the run manifest; confirmation must be explicit.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--severity", type=int, default=3)
    parser.add_argument("--conditions", nargs="+", default=["low_light", "blur"])
    parser.add_argument("--control-lr", type=float, default=1e-5)
    parser.add_argument("--oracles", nargs="+", choices=ORACLES, default=list(ORACLES))
    parser.add_argument("--scopes", nargs="+", choices=UPDATE_SCOPES, default=list(UPDATE_SCOPES))
    parser.add_argument("--lrs", nargs="+", type=float, default=[1e-6, 1e-5, 1e-4, 1e-3])
    parser.add_argument("--steps", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    return parser.parse_args()


def clone_primary_outputs(outputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        "pred_boxes": outputs["pred_boxes"].detach().float().cpu().clone(),
        "pred_sted": outputs["pred_sted"].detach().float().cpu().clone(),
    }


def output_differences(
    outputs: dict[str, torch.Tensor],
    reference: dict[str, torch.Tensor],
) -> dict[str, float]:
    return {
        "max_abs_pred_boxes": float(
            (outputs["pred_boxes"].detach().float().cpu() - reference["pred_boxes"]).abs().max()
        ),
        "max_abs_pred_sted": float(
            (outputs["pred_sted"].detach().float().cpu() - reference["pred_sted"]).abs().max()
        ),
    }


def run_consistency_episode(
    model: torch.nn.Module,
    adapter: EpisodicHeadAdapter,
    *,
    teacher_video: torch.Tensor,
    student_video: torch.Tensor,
    final_video: torch.Tensor,
    caption: str,
    args: argparse.Namespace,
    steps: int,
    teacher_outputs: dict[str, torch.Tensor] | None = None,
    reuse_student_encoder: bool = False,
) -> tuple[dict[str, torch.Tensor], list[dict[str, float]], float, float, dict[str, float]]:
    adapter.reset()
    if model.training:
        raise RuntimeError("TTA must run with model.eval(); training mode would enable dropout")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    losses: list[dict[str, float]] = []
    try:
        if teacher_outputs is None:
            with torch.no_grad():
                teacher = forward_video(
                    model,
                    teacher_video,
                    caption,
                    repo=args.repo,
                    stride=args.stride,
                    device="cuda",
                )
        else:
            teacher = teacher_outputs
        time_mask = torch.ones_like(teacher["pred_sted"][..., 0], dtype=torch.bool)
        box_mask = teacher_interval_mask(teacher["pred_sted"], time_mask=time_mask)
        cached_memory = None
        if reuse_student_encoder and adapter.scope != "heads_ln_projection":
            with torch.no_grad():
                cached_memory = encode_video(
                    model,
                    student_video,
                    caption,
                    repo=args.repo,
                    stride=args.stride,
                    device="cuda",
                )
        for _ in range(steps):
            if cached_memory is None:
                student = forward_video(
                    model,
                    student_video,
                    caption,
                    repo=args.repo,
                    stride=args.stride,
                    device="cuda",
                )
            else:
                student = decode_video(
                    model,
                    cached_memory,
                    duration=int(student_video.shape[1]),
                    caption=caption,
                    device="cuda",
                )
            losses.append(
                adapter.step(student, teacher, time_mask=time_mask, box_mask=box_mask)
            )
        with torch.no_grad():
            if cached_memory is not None and final_video.data_ptr() == student_video.data_ptr():
                prediction = decode_video(
                    model,
                    cached_memory,
                    duration=int(final_video.shape[1]),
                    caption=caption,
                    device="cuda",
                )
            else:
                prediction = forward_video(
                    model,
                    final_video,
                    caption,
                    repo=args.repo,
                    stride=args.stride,
                    device="cuda",
                )
        torch.cuda.synchronize()
        runtime = time.perf_counter() - started
        peak = torch.cuda.max_memory_allocated() / 1024**3
        before_reset = adapter.delta_from_initial()
        return prediction, losses, runtime, peak, before_reset
    finally:
        adapter.reset()
        reset_delta = adapter.delta_from_initial()
        if reset_delta["parameter_delta_max_abs"] != 0.0:
            raise RuntimeError(f"episodic reset was not exact: {reset_delta}")


def run_gt_episode(
    model: torch.nn.Module,
    adapter: EpisodicHeadAdapter,
    *,
    video: torch.Tensor,
    targets: list[dict],
    gt_indices: tuple[int, int],
    caption: str,
    args: argparse.Namespace,
    steps: int,
    final_video: torch.Tensor | None = None,
) -> tuple[dict[str, torch.Tensor], list[dict[str, float]], float, float, dict[str, float]]:
    adapter.reset()
    if model.training:
        raise RuntimeError("GT oracle must run with model.eval()")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    losses: list[dict[str, float]] = []
    try:
        # The encoder is unaffected in the three head-only scopes. Reusing its
        # frozen memory makes the oracle grid substantially faster without
        # changing any decoded output. The projection scope must re-encode.
        cached_memory = None
        if adapter.scope != "heads_ln_projection":
            with torch.no_grad():
                cached_memory = encode_video(
                    model,
                    video,
                    caption,
                    repo=args.repo,
                    stride=args.stride,
                    device="cuda",
                )
        for _ in range(steps):
            if cached_memory is None:
                outputs = forward_video(
                    model,
                    video,
                    caption,
                    repo=args.repo,
                    stride=args.stride,
                    device="cuda",
                )
            else:
                outputs = decode_video(
                    model,
                    cached_memory,
                    duration=int(video.shape[1]),
                    caption=caption,
                    device="cuda",
                )
            time_mask = torch.ones_like(outputs["pred_sted"][..., 0], dtype=torch.bool)
            losses.append(
                adapter.step_ground_truth(
                    outputs,
                    targets,
                    gt_indices,
                    time_mask=time_mask,
                )
            )
        evaluation_video = video if final_video is None else final_video
        with torch.no_grad():
            prediction = forward_video(
                model,
                evaluation_video,
                caption,
                repo=args.repo,
                stride=args.stride,
                device="cuda",
            )
        torch.cuda.synchronize()
        runtime = time.perf_counter() - started
        peak = torch.cuda.max_memory_allocated() / 1024**3
        before_reset = adapter.delta_from_initial()
        return prediction, losses, runtime, peak, before_reset
    finally:
        adapter.reset()
        reset_delta = adapter.delta_from_initial()
        if reset_delta["parameter_delta_max_abs"] != 0.0:
            raise RuntimeError(f"episodic reset was not exact: {reset_delta}")


def trimmed_mean(values: Iterable[float], proportion: float = 0.1) -> float:
    array = np.sort(np.asarray(list(values), dtype=np.float64))
    trim = int(np.floor(len(array) * proportion))
    if trim and 2 * trim < len(array):
        array = array[trim:-trim]
    return float(array.mean())


def base_record(
    outputs: dict[str, torch.Tensor],
    targets: list[dict],
    video_target: dict,
    annotation: dict,
    *,
    sample_index: int,
    condition: str,
    method: str,
    runtime: float,
    peak: float,
) -> dict[str, Any]:
    record = prediction_record(
        outputs,
        targets,
        video_target,
        annotation,
        sample_index=sample_index,
        condition=condition,
        method=method,
        runtime_sec=runtime,
        peak_vram_gb=peak,
    )
    filename = annotation["video_path"]
    stem = Path(filename).stem
    record["video_filename"] = filename
    record["source_cluster"] = stem.split("_", 1)[1] if "_" in stem else stem
    return record


def timed_frozen(
    model: torch.nn.Module,
    video: torch.Tensor,
    caption: str,
    args: argparse.Namespace,
) -> tuple[dict[str, torch.Tensor], float, float]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
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
    return outputs, time.perf_counter() - started, torch.cuda.max_memory_allocated() / 1024**3


def summarize_controls(records: list[dict[str, Any]], output_dir: Path) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record["method"] != "frozen":
            grouped[record["method"]].append(record)
    summary: dict[str, Any] = {}
    for method, values in sorted(grouped.items()):
        metric_delta_pp = [
            100.0 * (value["vIoU_corrected"] - value["reference_vIoU_corrected"])
            for value in values
        ]
        summary[method] = {
            "n": len(values),
            "max_abs_pred_boxes": max(value["max_abs_pred_boxes"] for value in values),
            "max_abs_pred_sted": max(value["max_abs_pred_sted"] for value in values),
            "max_abs_vIoU_delta_pp": max(abs(value) for value in metric_delta_pp),
            "mean_vIoU_delta_pp": float(np.mean(metric_delta_pp)),
            "temporal_interval_change_rate": float(
                np.mean(
                    [
                        value["predicted_interval"] != value["reference_predicted_interval"]
                        for value in values
                    ]
                )
            ),
            "max_parameter_delta_before_reset": max(
                value["parameter_delta_max_abs"] for value in values
            ),
            "max_parameter_delta_after_reset": max(
                value["parameter_delta_after_reset_max_abs"] for value in values
            ),
            "mean_loss_total": float(
                np.mean([value.get("last_loss_total", 0.0) for value in values])
            ),
        }
        summary[method]["passes_exact_noop"] = bool(
            summary[method]["max_abs_pred_boxes"] == 0.0
            and summary[method]["max_abs_pred_sted"] == 0.0
            and summary[method]["max_abs_vIoU_delta_pp"] == 0.0
            and summary[method]["max_parameter_delta_before_reset"] == 0.0
            and summary[method]["max_parameter_delta_after_reset"] == 0.0
        )
    (output_dir / "control_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )


def summarize_oracles(
    records: list[dict[str, Any]],
    output_dir: Path,
    *,
    bootstrap_samples: int,
    seed: int,
) -> None:
    frozen = {
        (record["sample_index"], record["condition"]): record
        for record in records
        if record["method"] == "frozen"
    }
    clean_frozen = {
        record["sample_index"]: record
        for record in records
        if record["method"] == "frozen" and record["condition"] == "clean"
    }
    grouped: dict[tuple[str, str, str, float, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record["method"] == "frozen":
            continue
        key = (
            record["condition"],
            record["method"],
            record["scope"],
            float(record["lr"]),
            int(record["steps"]),
        )
        grouped[key].append(record)

    rows: list[dict[str, Any]] = []
    detailed: dict[str, Any] = {}
    for key, values in sorted(grouped.items()):
        condition, method, scope, lr, steps = key
        values = sorted(values, key=lambda item: item["sample_index"])
        references = [frozen[(value["sample_index"], condition)] for value in values]
        row: dict[str, Any] = {
            "condition": condition,
            "method": method,
            "scope": scope,
            "lr": lr,
            "steps": steps,
            "n": len(values),
            "interval_change_rate": float(
                np.mean(
                    [
                        value["predicted_interval"] != reference["predicted_interval"]
                        for value, reference in zip(values, references)
                    ]
                )
            ),
            "mean_start_abs_error_frames": float(
                np.mean([value["start_abs_error_frames"] for value in values])
            ),
            "mean_end_abs_error_frames": float(
                np.mean([value["end_abs_error_frames"] for value in values])
            ),
            "mean_runtime_sec": float(np.mean([value["runtime_sec"] for value in values])),
            "peak_vram_gb": float(np.max([value["peak_vram_gb"] for value in values])),
            "mean_parameter_delta_l2": float(
                np.mean([value["parameter_delta_l2"] for value in values])
            ),
        }
        key_name = f"{condition}|{method}|{scope}|lr={lr:g}|steps={steps}"
        detailed[key_name] = {}
        for metric in METRICS:
            adapted = np.asarray([value[metric] for value in values], dtype=np.float64)
            baseline = np.asarray([reference[metric] for reference in references], dtype=np.float64)
            differences = adapted - baseline
            clean = np.asarray(
                [clean_frozen[value["sample_index"]][metric] for value in values],
                dtype=np.float64,
            )
            lost = float(clean.mean() - baseline.mean())
            recovered = float(adapted.mean() - baseline.mean())
            recovery_ratio = recovered / lost if lost > 0 else float("nan")
            prefix = metric.replace("vIoU_corrected", "vIoU")
            row.update(
                {
                    f"mean_{prefix}": float(adapted.mean()),
                    f"median_{prefix}": float(np.median(adapted)),
                    f"trimmed_mean_{prefix}": trimmed_mean(adapted),
                    f"delta_{prefix}_pp": 100.0 * recovered,
                    f"recovery_ratio_{prefix}": recovery_ratio,
                }
            )
            detailed[key_name][metric] = {
                "adapted_mean": float(adapted.mean()),
                "frozen_mean": float(baseline.mean()),
                "clean_frozen_mean": float(clean.mean()),
                "lost_performance": lost,
                "recovered_performance": recovered,
                "recovery_ratio": recovery_ratio,
                "paired_difference_median": float(np.median(differences)),
                "paired_difference_trimmed_mean": trimmed_mean(differences),
                "paired_bootstrap": paired_bootstrap_ci(
                    adapted,
                    baseline,
                    n_bootstrap=bootstrap_samples,
                    seed=seed,
                ),
                "cluster_paired_bootstrap": cluster_paired_bootstrap_ci(
                    adapted,
                    baseline,
                    [value["source_cluster"] for value in values],
                    n_bootstrap=bootstrap_samples,
                    seed=seed,
                ),
            }
        rows.append(row)

    if rows:
        with (output_dir / "oracle_summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (output_dir / "oracle_statistics.json").write_text(
        json.dumps(detailed, indent=2, ensure_ascii=False, allow_nan=True) + "\n"
    )


def run_controls(
    model: torch.nn.Module,
    dataset: Any,
    indices: list[int],
    args: argparse.Namespace,
    output_dir: Path,
) -> None:
    records: list[dict[str, Any]] = []
    output_path = output_dir / "control_records.jsonl"
    all_conditions = ["clean", *args.conditions]
    with output_path.open("w") as output_file:
        for position, sample_index in enumerate(indices):
            full_video, targets, video_target = load_full_video(dataset[sample_index])
            annotation = dataset.annotations[sample_index]
            caption = video_target["caption"]
            for condition_index, condition in enumerate(all_conditions):
                shifted = apply_condition(full_video, condition, severity=args.severity)
                frozen_outputs, runtime, peak = timed_frozen(model, shifted, caption, args)
                validate_output(frozen_outputs, shifted.shape[1])
                frozen_record = base_record(
                    frozen_outputs,
                    targets,
                    video_target,
                    annotation,
                    sample_index=sample_index,
                    condition=condition,
                    method="frozen",
                    runtime=runtime,
                    peak=peak,
                )
                records.append(frozen_record)
                output_file.write(json.dumps(frozen_record, ensure_ascii=False) + "\n")
                reference_outputs = clone_primary_outputs(frozen_outputs)

                # Control B: same final-inference path with zero adaptation steps.
                steps0_adapter = EpisodicHeadAdapter(
                    model, learning_rate=args.control_lr, scope="both"
                )
                steps0_adapter.reset()
                steps0_outputs, runtime, peak = timed_frozen(model, shifted, caption, args)
                steps0_delta = steps0_adapter.delta_from_initial()
                steps0_record = base_record(
                    steps0_outputs,
                    targets,
                    video_target,
                    annotation,
                    sample_index=sample_index,
                    condition=condition,
                    method="steps0",
                    runtime=runtime,
                    peak=peak,
                )
                steps0_record.update(
                    output_differences(steps0_outputs, reference_outputs),
                    reference_vIoU_corrected=frozen_record["vIoU_corrected"],
                    reference_predicted_interval=frozen_record["predicted_interval"],
                    **steps0_delta,
                    parameter_delta_after_reset_max_abs=0.0,
                )
                records.append(steps0_record)
                output_file.write(json.dumps(steps0_record, ensure_ascii=False) + "\n")

                # Control A: full teacher/student/backward/step path, but lr=0.
                strong = make_strong_view(
                    shifted,
                    seed=args.seed + sample_index * 10 + condition_index,
                )
                lr0_adapter = EpisodicHeadAdapter(model, learning_rate=0.0, scope="both")
                lr0_outputs, losses, runtime, peak, delta = run_consistency_episode(
                    model,
                    lr0_adapter,
                    teacher_video=shifted,
                    student_video=strong,
                    final_video=shifted,
                    caption=caption,
                    args=args,
                    steps=1,
                )
                lr0_record = base_record(
                    lr0_outputs,
                    targets,
                    video_target,
                    annotation,
                    sample_index=sample_index,
                    condition=condition,
                    method="lr0_full_path",
                    runtime=runtime,
                    peak=peak,
                )
                lr0_record.update(
                    output_differences(lr0_outputs, reference_outputs),
                    reference_vIoU_corrected=frozen_record["vIoU_corrected"],
                    reference_predicted_interval=frozen_record["predicted_interval"],
                    last_loss_total=losses[-1]["loss_total"],
                    **delta,
                    parameter_delta_after_reset_max_abs=lr0_adapter.delta_from_initial()[
                        "parameter_delta_max_abs"
                    ],
                )
                records.append(lr0_record)
                output_file.write(json.dumps(lr0_record, ensure_ascii=False) + "\n")

                # Control C: identical teacher/student video with a normal update.
                raw_noaug_adapter = EpisodicHeadAdapter(
                    model,
                    learning_rate=args.control_lr,
                    scope="both",
                    minimum_loss_for_update=None,
                )
                noaug_outputs, losses, runtime, peak, delta = run_consistency_episode(
                    model,
                    raw_noaug_adapter,
                    teacher_video=shifted,
                    student_video=shifted,
                    final_video=shifted,
                    caption=caption,
                    args=args,
                    steps=1,
                )
                noaug_record = base_record(
                    noaug_outputs,
                    targets,
                    video_target,
                    annotation,
                    sample_index=sample_index,
                    condition=condition,
                    method="no_augmentation_raw_adam",
                    runtime=runtime,
                    peak=peak,
                )
                noaug_record.update(
                    output_differences(noaug_outputs, reference_outputs),
                    reference_vIoU_corrected=frozen_record["vIoU_corrected"],
                    reference_predicted_interval=frozen_record["predicted_interval"],
                    last_loss_total=losses[-1]["loss_total"],
                    **delta,
                    parameter_delta_after_reset_max_abs=raw_noaug_adapter.delta_from_initial()[
                        "parameter_delta_max_abs"
                    ],
                )
                records.append(noaug_record)
                output_file.write(json.dumps(noaug_record, ensure_ascii=False) + "\n")

                # The production path treats sub-1e-6 loss as a numerical zero.
                guarded_noaug_adapter = EpisodicHeadAdapter(
                    model,
                    learning_rate=args.control_lr,
                    scope="both",
                    minimum_loss_for_update=1e-6,
                )
                guarded_outputs, losses, runtime, peak, delta = run_consistency_episode(
                    model,
                    guarded_noaug_adapter,
                    teacher_video=shifted,
                    student_video=shifted,
                    final_video=shifted,
                    caption=caption,
                    args=args,
                    steps=1,
                )
                guarded_record = base_record(
                    guarded_outputs,
                    targets,
                    video_target,
                    annotation,
                    sample_index=sample_index,
                    condition=condition,
                    method="no_augmentation_guarded",
                    runtime=runtime,
                    peak=peak,
                )
                guarded_record.update(
                    output_differences(guarded_outputs, reference_outputs),
                    reference_vIoU_corrected=frozen_record["vIoU_corrected"],
                    reference_predicted_interval=frozen_record["predicted_interval"],
                    last_loss_total=losses[-1]["loss_total"],
                    **delta,
                    parameter_delta_after_reset_max_abs=guarded_noaug_adapter.delta_from_initial()[
                        "parameter_delta_max_abs"
                    ],
                )
                records.append(guarded_record)
                output_file.write(json.dumps(guarded_record, ensure_ascii=False) + "\n")
                output_file.flush()
            print(f"[controls {position + 1}/{len(indices)}] sample={sample_index}", flush=True)
    summarize_controls(records, output_dir)


def run_oracles(
    model: torch.nn.Module,
    dataset: Any,
    indices: list[int],
    args: argparse.Namespace,
    output_dir: Path,
) -> None:
    records: list[dict[str, Any]] = []
    output_path = output_dir / "oracle_records.jsonl"
    configs = list(itertools.product(args.oracles, args.scopes, args.lrs, args.steps))
    with output_path.open("w") as output_file:
        for position, sample_index in enumerate(indices):
            full_video, targets, video_target = load_full_video(dataset[sample_index])
            annotation = dataset.annotations[sample_index]
            caption = video_target["caption"]
            clean_outputs, runtime, peak = timed_frozen(model, full_video, caption, args)
            validate_output(clean_outputs, full_video.shape[1])
            clean_record = base_record(
                clean_outputs,
                targets,
                video_target,
                annotation,
                sample_index=sample_index,
                condition="clean",
                method="frozen",
                runtime=runtime,
                peak=peak,
            )
            records.append(clean_record)
            output_file.write(json.dumps(clean_record, ensure_ascii=False) + "\n")

            for condition in args.conditions:
                shifted = apply_condition(full_video, condition, severity=args.severity)
                frozen_outputs, runtime, peak = timed_frozen(model, shifted, caption, args)
                validate_output(frozen_outputs, shifted.shape[1])
                frozen_record = base_record(
                    frozen_outputs,
                    targets,
                    video_target,
                    annotation,
                    sample_index=sample_index,
                    condition=condition,
                    method="frozen",
                    runtime=runtime,
                    peak=peak,
                )
                records.append(frozen_record)
                output_file.write(json.dumps(frozen_record, ensure_ascii=False) + "\n")

                for config_index, (oracle, scope, lr, steps) in enumerate(configs):
                    adapter = EpisodicHeadAdapter(
                        model,
                        learning_rate=lr,
                        scope=scope,
                    )
                    if oracle == "clean_teacher":
                        outputs, losses, runtime, peak, delta = run_consistency_episode(
                            model,
                            adapter,
                            teacher_video=full_video,
                            student_video=shifted,
                            final_video=shifted,
                            caption=caption,
                            args=args,
                            steps=steps,
                            teacher_outputs=clean_outputs,
                            reuse_student_encoder=True,
                        )
                        method = "clean_teacher_oracle"
                    else:
                        outputs, losses, runtime, peak, delta = run_gt_episode(
                            model,
                            adapter,
                            video=shifted,
                            targets=targets,
                            gt_indices=tuple(video_target["inter_idx"]),
                            caption=caption,
                            args=args,
                            steps=steps,
                        )
                        method = "gt_oracle"
                    validate_output(outputs, shifted.shape[1])
                    record = base_record(
                        outputs,
                        targets,
                        video_target,
                        annotation,
                        sample_index=sample_index,
                        condition=condition,
                        method=method,
                        runtime=runtime,
                        peak=peak,
                    )
                    record.update(
                        oracle=oracle,
                        scope=scope,
                        lr=lr,
                        steps=steps,
                        losses=losses,
                        trainable_parameter_count=adapter.trainable_parameter_count,
                        trainable_parameter_names=adapter.parameter_names,
                        **delta,
                    )
                    records.append(record)
                    output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    output_file.flush()
                    if len(indices) <= 8:
                        print(
                            f"[oracles {position + 1}/{len(indices)}] sample={sample_index} "
                            f"condition={condition} config={config_index + 1}/{len(configs)} "
                            f"oracle={oracle} scope={scope} lr={lr:g} steps={steps} "
                            f"frozen={frozen_record['vIoU_corrected']:.4f} "
                            f"adapt={record['vIoU_corrected']:.4f}",
                            flush=True,
                        )
            if len(indices) > 8:
                print(
                    f"[oracles {position + 1}/{len(indices)}] sample={sample_index}",
                    flush=True,
                )
    summarize_oracles(
        records,
        output_dir,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Phase-1 experiments")
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
    data_args = dataset_args(model_args, args.video_root, args.annotation_root)
    dataset = build_dataset("hcstvg", image_set="val", args=data_args)
    if args.num_samples > len(dataset):
        raise ValueError(
            f"requested {args.num_samples} examples but the selected subset has {len(dataset)}"
        )
    indices = list(range(args.num_samples))
    subset_manifest_path = Path(args.video_root).expanduser().resolve() / "subset_manifest.json"
    subset_manifest = None
    if subset_manifest_path.exists():
        subset_manifest = json.loads(subset_manifest_path.read_text())
    is_confirmation = args.dataset_role == "confirmation"
    confirmation_separation = None
    if is_confirmation:
        if subset_manifest is None:
            raise ValueError("confirmation runs require a subset_manifest.json")
        excluded_manifests = subset_manifest.get("excluded_manifests", [])
        if not excluded_manifests:
            raise ValueError(
                "confirmation subset manifest must identify the excluded development manifest"
            )
        selected_filenames = {
            dataset.annotations[index]["video_path"] for index in indices
        }
        selected_clusters = {
            Path(filename).stem.split("_", 1)[1]
            for filename in selected_filenames
            if "_" in Path(filename).stem
        }
        excluded_filenames: set[str] = set()
        excluded_clusters: set[str] = set()
        resolved_excluded_manifests: list[str] = []
        for excluded_manifest_name in excluded_manifests:
            excluded_manifest_path = Path(excluded_manifest_name).expanduser().resolve()
            if not excluded_manifest_path.exists():
                raise FileNotFoundError(
                    f"excluded development manifest is missing: {excluded_manifest_path}"
                )
            excluded_manifest = json.loads(excluded_manifest_path.read_text())
            filenames = {
                sample["filename"] for sample in excluded_manifest.get("samples", [])
            }
            excluded_filenames.update(filenames)
            excluded_clusters.update(
                Path(filename).stem.split("_", 1)[1]
                for filename in filenames
                if "_" in Path(filename).stem
            )
            resolved_excluded_manifests.append(str(excluded_manifest_path))
        filename_overlap = sorted(selected_filenames & excluded_filenames)
        cluster_overlap = sorted(selected_clusters & excluded_clusters)
        if filename_overlap or cluster_overlap:
            raise ValueError(
                "confirmation subset overlaps development data: "
                f"filenames={filename_overlap[:5]}, clusters={cluster_overlap[:5]}"
            )
        confirmation_separation = {
            "validated": True,
            "excluded_manifests": resolved_excluded_manifests,
            "selected_video_count": len(selected_filenames),
            "selected_source_cluster_count": len(selected_clusters),
            "excluded_video_count": len(excluded_filenames),
            "excluded_source_cluster_count": len(excluded_clusters),
            "filename_overlap_count": 0,
            "source_cluster_overlap_count": 0,
        }
    manifest = {
        "phase": args.phase,
        "dataset_role": args.dataset_role,
        "dev_only": not is_confirmation,
        "confirmation_eligible": is_confirmation,
        "dataset": (
            f"HC-STVG2 official validation {args.num_samples}-video "
            f"{args.dataset_role} subset"
        ),
        "indices": indices,
        "subset_manifest_path": (
            str(subset_manifest_path) if subset_manifest is not None else None
        ),
        "subset_selection": subset_manifest,
        "confirmation_separation": confirmation_separation,
        "args": vars(args),
        "checkpoint_report": checkpoint_report,
        "model_training_mode": model.training,
        "torch_version": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0),
        "evaluator": {
            "primary": "vIoU_corrected with max(pred_end, gt_end)",
            "compatibility_only": "vIoU_legacy with min(pred_end, gt_end)",
            "upstream_fix_commit": "3c32cc92a0fdaa0c770d95a59d8764e0e212424c",
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    if args.phase == "controls":
        run_controls(model, dataset, indices, args, output_dir)
    else:
        run_oracles(model, dataset, indices, args, output_dir)


if __name__ == "__main__":
    main()
