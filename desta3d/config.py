"""One editable JSON per manual run; no experiment-directory constants here."""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class AdapterConfig:
    in_channels: int = 2560
    query_dim: int = 2560
    hidden_dim: int = 128
    architecture: str = "dual3d"
    p1_enabled: bool = False
    train_stage: str = "frozen"
    freeze_tta_gates: bool = True


@dataclass(frozen=True)
class DirectionConfig:
    feature_dim: int = 128  # Frozen cache width; independent of hidden_dim.
    hidden_dim: int = 128
    output_rank: int = 256
    global_mean: bool = False
    radius: float = 0.13545580427763146  # Used by project(), NOT cosine fit.
    output_init_std: float = 0.001


@dataclass(frozen=True)
class ViewConfig:
    temporal_dim: float = 0.25
    spatial_blur_radius: float = 8.0


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 20260928
    steps: int = 200
    batch_size: int = 4  # Equal-query gradient accumulation; variable THW grids.
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    clip_norm: float = 1.0
    device: str = "cuda:0"
    threads: int = 4
    minimum_free_gib: float = 8.0


@dataclass(frozen=True)
class CacheConfig:
    root: str = "artifacts/desta3d_v3/latent_oracle_v1/a0_fast_screen_v1"
    seal_sha256: str = ""
    basis_file: str = "artifacts/desta3d_v3/latent_oracle_v1/a0_fast_screen_v1/BASIS.pt"
    basis_sha256: str = ""
    channel_basis_file: str = ""  # Optional 256 x rank .npy, e.g. sealed B16.
    channel_basis_sha256: str = ""


def _section(cls, data):
    if not isinstance(data, dict):
        raise ValueError(f"{cls.__name__} must be an object")
    defaults = asdict(cls())
    unknown = set(data) - set(defaults)
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} keys: {sorted(unknown)}")
    for name, value in data.items():
        expected = type(defaults[name])
        valid = (type(value) in (int, float) and math.isfinite(value)
                 if expected is float else type(value) is expected)
        if not valid:
            raise ValueError(f"Invalid type/value for {name}: expected {expected.__name__}")
    return cls(**data)


def load_config(path):
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict) or type(data.get("version")) is not int or data["version"] != 1:
        raise ValueError("Expected config version 1")
    kind = data.get("kind")
    sections = {
        "adapter": {"model": AdapterConfig},
        "cached_direction": {"model": DirectionConfig, "training": TrainingConfig, "cache": CacheConfig},
        "pixel_views": {"views": ViewConfig},
    }
    if kind not in sections:
        raise ValueError(f"kind must be one of {list(sections)}")
    if set(data) - {"version", "kind", *sections[kind]}:
        raise ValueError("Unknown top-level configuration key")
    result = {"version": 1, "kind": kind}
    for name, cls in sections[kind].items():
        result[name] = asdict(_section(cls, data.get(name, {})))
    m = result.get("model", {})
    for name in ("feature_dim", "hidden_dim", "output_rank", "in_channels", "query_dim"):
        if name in m and m[name] < 1:
            raise ValueError(f"{name} must be positive")
    if kind == "adapter":
        if m["architecture"] not in ("dual3d", "shared3d", "early_factorized"):
            raise ValueError("Unknown adapter architecture")
        if m["train_stage"] not in ("frozen", "tta", "A evidence", "B integration"):
            raise ValueError("Unknown adapter train_stage")
    elif kind == "cached_direction":
        t, c = result["training"], result["cache"]
        if m["radius"] < 0 or m["output_init_std"] <= 0:
            raise ValueError("radius >= 0 and output_init_std > 0 required")
        if min(t[k] for k in ("steps", "batch_size", "threads")) < 1 or t["seed"] < 0:
            raise ValueError("Positive steps/batch_size/threads and nonnegative seed required")
        if t["learning_rate"] <= 0 or t["clip_norm"] <= 0 or t["weight_decay"] < 0:
            raise ValueError("Invalid optimizer parameters")
        if t["minimum_free_gib"] < 8:
            raise ValueError("Keep the project 8 GiB free-space floor")
        if t["device"] not in ("cpu", "cuda:0"):
            raise ValueError("device must be cpu or cuda:0")
        for name in ("seal_sha256", "basis_sha256", "channel_basis_sha256"):
            value = c[name]
            if value and (len(value) != 64 or any(x not in "0123456789abcdef" for x in value)):
                raise ValueError(f"Invalid {name}")
        if not c["root"] or not c["basis_file"]:
            raise ValueError("Cache root and basis_file are required")
        if bool(c["channel_basis_file"]) != bool(c["channel_basis_sha256"]):
            raise ValueError("channel basis path and hash must be supplied together")
    else:
        v = result["views"]
        if not 0 <= v["temporal_dim"] <= 1 or v["spatial_blur_radius"] < 0:
            raise ValueError("temporal_dim in [0,1]; spatial_blur_radius >= 0")
    return result
