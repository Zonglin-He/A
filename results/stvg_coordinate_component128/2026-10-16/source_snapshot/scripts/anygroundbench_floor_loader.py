#!/usr/bin/env python3
"""Variable-duration CPU loader for the AnyGroundBench floor-check subset.

TubeDETR's official HC-STVG loader assumes every video is exactly 20 seconds
long.  AnyGroundBench's Mouse and American Football clips retain their real
durations (roughly 4--145 seconds in the tiny subset), so feeding the
converted annotations into that loader can silently truncate a query or fail
its frame-count assertion.  This independent adapter keeps TubeDETR's
``C x T x H x W``/target convention while sampling each clip using its own
FPS and full frame count.

This file is a data/format bridge only.  It does not construct a model, move
anything to CUDA, or perform adaptation.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TUBEDETR_ROOT = PROJECT_ROOT / "external" / "TubeDETR"
if str(TUBEDETR_ROOT) not in sys.path:
    sys.path.insert(0, str(TUBEDETR_ROOT))

from datasets.video_transforms import make_video_transforms  # noqa: E402
from datasets.hcstvg import VideoModulatedSTGrounding  # noqa: E402
from datasets.video_transforms import prepare  # noqa: E402


def _sampling_ids(frame_count: int, video_fps: float, fps: float, max_frames: int) -> list[int]:
    if frame_count <= 0 or video_fps <= 0 or fps <= 0:
        raise ValueError("frame_count and FPS values must be positive")
    sampling_rate = fps / video_fps
    if sampling_rate > 1.0 + 1e-6:
        raise ValueError(
            f"floor loader only downsamples: requested {fps} FPS from {video_fps} FPS"
        )
    frame_ids = [0]
    for frame_id in range(frame_count):
        if int(frame_ids[-1] * sampling_rate) < int(frame_id * sampling_rate):
            frame_ids.append(frame_id)
    if len(frame_ids) > max_frames:
        frame_ids = [frame_ids[(index * len(frame_ids)) // max_frames] for index in range(max_frames)]
    return frame_ids


class AnyGroundBenchVariableDuration:
    """Minimal Dataset-like adapter matching the HC-STVG sample tuple."""

    def __init__(
        self,
        tubedetr_root: str | Path,
        annotation_path: str | Path,
        *,
        fps: float = 5.0,
        max_frames: int = 200,
        resolution: int = 224,
        stride: int = 2,
    ) -> None:
        self.tubedetr_root = Path(tubedetr_root).expanduser().resolve()
        self.annotations: list[dict[str, Any]] = json.loads(
            Path(annotation_path).expanduser().resolve().read_text(encoding="utf-8")
        )
        if not isinstance(self.annotations, list) or not self.annotations:
            raise ValueError("AnyGroundBench conversion contains no annotations")
        self.fps = float(fps)
        self.max_frames = int(max_frames)
        self.stride = int(stride)
        self.transform = make_video_transforms("val", cautious=True, resolution=resolution)
        self.frame_ids: list[list[int]] = []
        for annotation in self.annotations:
            self.frame_ids.append(
                _sampling_ids(
                    int(annotation["frame_count"]),
                    float(annotation["fps"]),
                    self.fps,
                    self.max_frames,
                )
            )

    def __len__(self) -> int:
        return len(self.annotations)

    def _decode_selected(self, path: Path, selected_ids: list[int], frame_count: int) -> list[np.ndarray]:
        selected = set(selected_ids)
        frames: dict[int, np.ndarray] = {}
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError(f"OpenCV cannot open {path}")
        try:
            index = 0
            while index < frame_count and len(frames) < len(selected):
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(
                        f"OpenCV stopped at frame {index} while reading {path}; "
                        f"wanted {frame_count} metadata frames"
                    )
                if index in selected:
                    frames[index] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                index += 1
        finally:
            capture.release()
        missing = [frame_id for frame_id in selected_ids if frame_id not in frames]
        if missing:
            raise RuntimeError(f"selected frames missing from {path}: {missing[:8]}")
        return [frames[frame_id] for frame_id in selected_ids]

    def __getitem__(self, index: int):
        annotation = self.annotations[index]
        frame_ids = self.frame_ids[index]
        video_path = self.tubedetr_root / "video" / annotation["video_path"]
        frames = self._decode_selected(
            video_path, frame_ids, int(annotation["frame_count"])
        )
        width = int(annotation["width"])
        height = int(annotation["height"])
        start = int(annotation["tube_start_frame"])
        end = int(annotation["tube_end_frame"])
        trajectory = annotation["trajectory"]
        targets = []
        inter_idx = []
        for output_index, frame_id in enumerate(frame_ids):
            if start <= frame_id < end:
                bbox = trajectory[frame_id - start]
                anns = [{"bbox": bbox}]
                inter_idx.append(output_index)
            else:
                anns = []
            targets.append(prepare(width, height, anns))
        images, targets = self.transform(frames, targets)
        if not inter_idx:
            raise RuntimeError(
                f"selected frames miss the annotated interval for {annotation['video_path']}"
            )
        target = {
            "video_id": int(annotation["video_id"]),
            "inter_idx": [inter_idx[0], inter_idx[-1]],
            "frames_id": frame_ids,
            "caption": annotation["caption"],
        }
        full_images = images
        if self.stride:
            images = images[:, :: self.stride]
        return images, targets, target, full_images


def cpu_smoke_dataset(tubedetr_root: str | Path, annotation_path: str | Path) -> dict[str, Any]:
    """Decode every selected query sample and verify full-duration alignment."""

    dataset = AnyGroundBenchVariableDuration(tubedetr_root, annotation_path)
    rows = []
    for index in range(len(dataset)):
        images, targets, target, full_images = dataset[index]
        if images.ndim != 4 or full_images.ndim != 4:
            raise RuntimeError(f"unexpected tensor shape at index {index}: {images.shape}")
        if full_images.shape[1] != len(targets) or len(target["frames_id"]) != len(targets):
            raise RuntimeError(f"frame/target mismatch at index {index}")
        if target["inter_idx"][0] < 0 or target["inter_idx"][1] >= len(targets):
            raise RuntimeError(f"invalid interval at index {index}: {target['inter_idx']}")
        if not torch.isfinite(images).all() or not torch.isfinite(full_images).all():
            raise FloatingPointError(f"non-finite image tensor at index {index}")
        rows.append(
            {
                "index": index,
                "video_id": target["video_id"],
                "video_path": dataset.annotations[index]["video_path"],
                "sample_shape": list(images.shape),
                "full_shape": list(full_images.shape),
                "inter_idx": list(target["inter_idx"]),
            }
        )
    return {"dataset_len": len(dataset), "samples": rows}


def compare_official_loader_behavior(tubedetr_root: str | Path, annotation_path: str | Path) -> dict[str, Any]:
    """Report the known 20-second HC loader limitation without failing the smoke."""

    environment_bin = str(Path(sys.executable).resolve().parent)
    os.environ["PATH"] = os.pathsep.join([environment_bin, os.environ.get("PATH", "")])
    try:
        dataset = VideoModulatedSTGrounding(
            str(Path(tubedetr_root).resolve()),
            str(Path(annotation_path).resolve()),
            transforms=make_video_transforms("val", cautious=True, resolution=224),
            is_train=False,
            video_max_len=200,
            video_max_len_train=100,
            fps=5,
            tmp_crop=False,
            tmp_loc=True,
            stride=2,
        )
        failures = []
        empty_intervals = []
        for index in range(len(dataset)):
            try:
                _, _, target, _ = dataset[index]
            except Exception as exc:  # noqa: BLE001 - diagnostics are serialized
                failures.append({"index": index, "error": f"{type(exc).__name__}: {exc}"})
                continue
            if target["inter_idx"][0] < 0:
                empty_intervals.append(index)
        return {
            "loader": "external/TubeDETR/datasets/hcstvg.py",
            "assumption": "20-second clips",
            "failures": failures,
            "empty_intervals": empty_intervals,
        }
    except Exception as exc:  # noqa: BLE001 - diagnostics are serialized
        return {
            "loader": "external/TubeDETR/datasets/hcstvg.py",
            "assumption": "20-second clips",
            "constructor_error": f"{type(exc).__name__}: {exc}",
        }

