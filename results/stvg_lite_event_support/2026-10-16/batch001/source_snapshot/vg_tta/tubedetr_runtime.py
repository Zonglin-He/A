"""Small runtime bridge around the upstream TubeDETR repository."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch


def ensure_environment_tools_on_path() -> None:
    """Expose the reproducible project-local executable and model caches."""

    environment_bin = str(Path(sys.executable).resolve().parent)
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    if environment_bin not in path_entries:
        os.environ["PATH"] = os.pathsep.join([environment_bin, *path_entries])
    project_root = Path(__file__).resolve().parents[1]
    local_hf_cache = project_root / ".cache" / "huggingface"
    local_torch_cache = project_root / ".cache" / "torch"
    if local_hf_cache.is_dir():
        os.environ.setdefault("HF_HOME", str(local_hf_cache))
    if local_torch_cache.is_dir():
        os.environ.setdefault("TORCH_HOME", str(local_torch_cache))


def add_repo_to_path(repo: str | Path) -> Path:
    repo_path = Path(repo).expanduser().resolve()
    if not (repo_path / "models" / "tubedetr.py").is_file():
        raise FileNotFoundError(f"not a TubeDETR checkout: {repo_path}")
    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))
    return repo_path


def build_model(
    repo: str | Path,
    *,
    device: str = "cuda",
    resolution: int = 224,
    stride: int = 5,
    video_max_len: int = 200,
    fast: bool = True,
) -> tuple[torch.nn.Module, Any]:
    ensure_environment_tools_on_path()
    repo_path = add_repo_to_path(repo)
    from main import get_args_parser
    from models import build_model as upstream_build_model

    args = get_args_parser().parse_args(
        [
            "--dataset_config",
            str(repo_path / "config" / "hcstvg.json"),
            "--combine_datasets",
            "hcstvg",
            "--combine_datasets_val",
            "hcstvg",
            "--device",
            device,
            "--resolution",
            str(resolution),
            "--stride",
            str(stride),
            "--video_max_len",
            str(video_max_len),
            "--video_max_len_train",
            str(video_max_len),
            "--freeze_backbone",
            "--freeze_text_encoder",
        ]
        + ([] if fast else ["--no_fast"])
    )
    model, _, _ = upstream_build_model(args)
    model.to(torch.device(device))
    return model, args


def load_official_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    # PyTorch 2.6 changed the default to weights_only=True. Official TubeDETR
    # checkpoints contain trusted training metadata in addition to tensor weights.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    key = "model_ema" if "model_ema" in checkpoint else "model"
    if key not in checkpoint:
        raise KeyError("checkpoint has neither 'model_ema' nor 'model'")
    state = dict(checkpoint[key])
    model_state = model.state_dict()
    ignored_compatibility_keys: list[str] = []
    legacy_position_ids_key = "transformer.text_encoder.embeddings.position_ids"
    if legacy_position_ids_key in state and legacy_position_ids_key not in model_state:
        # Older Transformers releases persisted this deterministic arange
        # buffer. Modern RoBERTa recreates it internally and deliberately omits
        # it from the state dict, so retaining the legacy copy only produces a
        # false-positive unexpected key.
        state.pop(legacy_position_ids_key)
        ignored_compatibility_keys.append(legacy_position_ids_key)
    if "query_embed.weight" in state and model.num_queries < state["query_embed.weight"].shape[0]:
        state["query_embed.weight"] = state["query_embed.weight"][: model.num_queries]
    time_key = "transformer.time_embed.te"
    if (
        time_key in state
        and time_key in model_state
        and state[time_key].shape != model_state[time_key].shape
    ):
        # The sinusoidal table is deterministic; rebuild it when max video
        # length differs instead of treating the checkpoint as incompatible.
        state.pop(time_key)
    incompatible = model.load_state_dict(state, strict=False)
    expected_missing = [time_key] if time_key not in state and time_key in model_state else []
    return {
        "state_key": key,
        "missing_keys": [key for key in incompatible.missing_keys if key not in expected_missing],
        "unexpected_keys": list(incompatible.unexpected_keys),
        "rebuilt_deterministic_keys": expected_missing,
        "ignored_compatibility_keys": ignored_compatibility_keys,
    }


def make_nested_video(video: torch.Tensor, repo: str | Path):
    add_repo_to_path(repo)
    from util.misc import NestedTensor

    return NestedTensor.from_tensor_list([video], False)


def forward_video(
    model: torch.nn.Module,
    full_video: torch.Tensor,
    caption: str,
    *,
    repo: str | Path,
    stride: int,
    device: torch.device | str,
    use_bf16: bool = True,
) -> dict[str, torch.Tensor]:
    """Forward one normalized CxTxHxW video through TubeDETR."""

    memory_cache = encode_video(
        model,
        full_video,
        caption,
        repo=repo,
        stride=stride,
        device=device,
        use_bf16=use_bf16,
    )
    return decode_video(
        model,
        memory_cache,
        duration=int(full_video.shape[1]),
        caption=caption,
        device=device,
        use_bf16=use_bf16,
    )


def forward_video_with_features(
    model: torch.nn.Module,
    full_video: torch.Tensor,
    caption: str,
    *,
    repo: str | Path,
    stride: int,
    device: torch.device | str,
    use_bf16: bool = True,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Forward a video and expose final query-conditioned frame features.

    TubeDETR feeds a ``layers x batch x time x hidden`` tensor into
    ``sted_embed``.  A temporary pre-hook captures that tensor without
    modifying the official model source.  The returned features are the final
    decoder layer for batch element zero and are aligned one-to-one with
    ``pred_sted`` frames.
    """

    memory_cache = encode_video(
        model,
        full_video,
        caption,
        repo=repo,
        stride=stride,
        device=device,
        use_bf16=use_bf16,
    )
    return decode_video_with_features(
        model,
        memory_cache,
        duration=int(full_video.shape[1]),
        caption=caption,
        device=device,
        use_bf16=use_bf16,
    )


def forward_video_with_temporal_head_input(
    model: torch.nn.Module,
    full_video: torch.Tensor,
    caption: str,
    *,
    repo: str | Path,
    stride: int,
    device: torch.device | str,
    use_bf16: bool = True,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Forward a video and expose the complete temporal-head input tensor."""

    memory_cache = encode_video(
        model,
        full_video,
        caption,
        repo=repo,
        stride=stride,
        device=device,
        use_bf16=use_bf16,
    )
    return decode_video_with_temporal_head_input(
        model,
        memory_cache,
        duration=int(full_video.shape[1]),
        caption=caption,
        device=device,
        use_bf16=use_bf16,
    )


def encode_video(
    model: torch.nn.Module,
    full_video: torch.Tensor,
    caption: str,
    *,
    repo: str | Path,
    stride: int,
    device: torch.device | str,
    use_bf16: bool = True,
) -> dict[str, Any]:
    """Run the visual/text encoder and return a reusable memory cache."""

    if full_video.ndim != 4 or full_video.shape[0] != 3:
        raise ValueError(f"expected CxTxHxW input, got {tuple(full_video.shape)}")
    duration = int(full_video.shape[1])
    device = torch.device(device)
    full_video = full_video.to(device, non_blocking=True)
    samples_fast = make_nested_video(full_video, repo).to(device) if model.fast else None
    slow_video = full_video[:, ::stride] if stride else full_video
    samples = make_nested_video(slow_video, repo).to(device)
    autocast_enabled = use_bf16 and device.type == "cuda"
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
        memory_cache = model(
            samples,
            [duration],
            [caption],
            encode_and_save=True,
            samples_fast=samples_fast,
        )
    return memory_cache


def decode_video(
    model: torch.nn.Module,
    memory_cache: dict[str, Any],
    *,
    duration: int,
    caption: str,
    device: torch.device | str,
    use_bf16: bool = True,
) -> dict[str, torch.Tensor]:
    """Decode a cached one-video encoder representation."""

    device = torch.device(device)
    autocast_enabled = use_bf16 and device.type == "cuda"
    # TubeDETR normalizes ``samples`` before branching on encode/decode even
    # though the decoded path never reads its value. Supply a minimal valid
    # NestedTensor instead of None.
    from util.misc import NestedTensor

    sample_stub = NestedTensor.from_tensor_list(
        [torch.zeros((3, 1, 1), device=device)]
    )
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
        outputs = model(
            sample_stub,
            [duration],
            [caption],
            encode_and_save=False,
            memory_cache=memory_cache,
        )
    return outputs


def decode_video_with_features(
    model: torch.nn.Module,
    memory_cache: dict[str, Any],
    *,
    duration: int,
    caption: str,
    device: torch.device | str,
    use_bf16: bool = True,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Decode and return final-layer ``T x hidden`` temporal head inputs."""

    outputs, hidden = decode_video_with_temporal_head_input(
        model,
        memory_cache,
        duration=duration,
        caption=caption,
        device=device,
        use_bf16=use_bf16,
    )
    features = hidden[-1, 0]
    if outputs["pred_sted"].shape[1] != len(features):
        raise RuntimeError("decoder features and temporal predictions are not frame-aligned")
    return outputs, features


def decode_video_with_temporal_head_input(
    model: torch.nn.Module,
    memory_cache: dict[str, Any],
    *,
    duration: int,
    caption: str,
    device: torch.device | str,
    use_bf16: bool = True,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Decode and return ``layers x batch x time x hidden`` head inputs."""

    captured: list[torch.Tensor] = []

    def capture_input(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        if len(inputs) != 1:
            raise RuntimeError("unexpected sted_embed input signature")
        captured.append(inputs[0])

    handle = model.sted_embed.register_forward_pre_hook(capture_input)
    try:
        outputs = decode_video(
            model,
            memory_cache,
            duration=duration,
            caption=caption,
            device=device,
            use_bf16=use_bf16,
        )
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError(f"expected one sted_embed hook call, observed {len(captured)}")
    hidden = captured[0]
    if hidden.ndim != 4 or hidden.shape[1] != 1 or hidden.shape[2] != duration:
        raise RuntimeError(
            "unexpected decoder feature shape "
            f"{tuple(hidden.shape)} for duration {duration}"
        )
    if outputs["pred_sted"].shape[1] != hidden.shape[2]:
        raise RuntimeError("temporal head inputs and predictions are not frame-aligned")
    return outputs, hidden


def dataset_args(model_args: Any, video_root: str | Path, annotation_root: str | Path) -> Any:
    values = vars(model_args).copy()
    values.update(
        hcstvg_vid_path=str(Path(video_root).expanduser().resolve()),
        hcstvg_ann_path=str(Path(annotation_root).expanduser().resolve()),
        v2=True,
        test=False,
        tmp_crop=False,
    )
    return SimpleNamespace(**values)


def vidstg_dataset_args(
    model_args: Any,
    video_root: str | Path,
    annotation_root: str | Path,
    *,
    test: bool = True,
) -> Any:
    """Return upstream TubeDETR arguments for a prepared VidSTG split.

    The model architecture is shared between the HC-STVG2 and VidSTG
    checkpoints, but the upstream dataset builder selects its annotation file
    through ``args.test``.  Keeping that choice explicit avoids silently
    loading ``val.json`` when an untouched ``test.json`` confirmation subset
    was prepared.
    """

    values = vars(model_args).copy()
    values.update(
        vidstg_vid_path=str(Path(video_root).expanduser().resolve()),
        vidstg_ann_path=str(Path(annotation_root).expanduser().resolve()),
        test=bool(test),
        tmp_crop=False,
    )
    return SimpleNamespace(**values)
