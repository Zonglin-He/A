#!/usr/bin/env python3
"""Audit and CPU-smoke-test the official TA-STVG second backbone.

This helper is intentionally independent of TubeDETR and vg_tta.  It reads
the official checkout and checkpoint files, audits the two local Python
prefixes, and runs a one-sample synthetic temporal-head smoke test on CPU.
It never calls .cuda(), torch.cuda, or the complete TA-STVG pipeline.  The
latter needs the released video/text model assets and the upstream 2023
dependency stack, which are recorded by the generated report.

The temporal head in the official pipeline is TASTVGNet.temp_embed:
MLP(256, 256, 2, 2, dropout=0.3).  TAStyleTemporalHead below mirrors that
source implementation so the smoke test can run without importing
TA-STVG's optional torchtext/Roberta stack.

Examples:
  .conda/tubedetr/bin/python scripts/prepare_tastvg.py
  .conda/tubedetr/bin/python scripts/prepare_tastvg.py --no-smoke

The output JSON is a machine-readable companion to README_TASTVG.md.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_DIR = PROJECT_ROOT / "external" / "TA-STVG"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "phase3_tastvg_prep.json"
OFFICIAL_REPO = "https://github.com/HengLan/TA-STVG"
EXPECTED_COMMIT = "904ad0c344eb12fe34513f97a33f93118f6d3be4"

# These are the exact public links in the official README at EXPECTED_COMMIT.
# The expected SHA-256 values equal the Hugging Face X-Linked-ETag observed
# while downloading the files on 2026-09-04.
CHECKPOINTS: dict[str, dict[str, Any]] = {
    "HC-STVG2": {
        "filename": "TASTVG_HCSTVG2.pth",
        "hf_revision": "38e89a5895346d77cba87b0273f9ede95874befc",
        "url": (
            "https://huggingface.co/Gstar666/TASTVG/resolve/main/"
            "TASTVG_HCSTVG2.pth?download=true"
        ),
        "expected_size_bytes": 1_951_277_247,
        "expected_sha256": (
            "47d8f15841cd57e7bbf5a10e8bf23b1054d23b753e0becbd38a07f3dd60d5036"
        ),
    },
    "VidSTG": {
        "filename": "TASTVG_VidSTG.pth",
        "hf_revision": "38e89a5895346d77cba87b0273f9ede95874befc",
        "url": (
            "https://huggingface.co/Gstar666/TASTVG/resolve/main/"
            "TASTVG_VidSTG.pth?download=true"
        ),
        "expected_size_bytes": 1_951_230_809,
        "expected_sha256": (
            "5ab12c86363ef0ce0ee006c00fd11c6b659c3a9b2cb01a4f2c613efe22a2aa83"
        ),
    },
    "HC-STVG": {
        "filename": "TASTVG_HCSTVG.pth",
        "hf_revision": "38e89a5895346d77cba87b0273f9ede95874befc",
        "url": (
            "https://huggingface.co/Gstar666/TASTVG/resolve/main/"
            "TASTVG_HCSTVG.pth?download=true"
        ),
        "expected_size_bytes": None,
        "expected_sha256": None,
        "status_note": (
            "Not downloaded: the requested natural-shift pair and clean "
            "guardrail are covered by HC-STVG2 and VidSTG."
        ),
    },
}

MODEL_ASSETS: dict[str, dict[str, Any]] = {
    "swin_tiny_kinetics400": {
        "path": "checkpoints/tastvg_model_zoo/swin_tiny_patch244_window877_kinetics400_1k.pth",
        "url": (
            "https://github.com/SwinTransformer/storage/releases/download/v1.0.4/"
            "swin_tiny_patch244_window877_kinetics400_1k.pth"
        ),
        "note": (
            "Official release has no published checksum; local SHA-256 is an "
            "acquisition fingerprint."
        ),
    },
    "resnet101": {
        "path": ".cache/torch/hub/checkpoints/resnet101-cd907fc2.pth",
        "url": "https://download.pytorch.org/models/resnet101-cd907fc2.pth",
    },
}

REQUIREMENT_PACKAGE_NAMES = {
    "tqdm": "tqdm",
    "yacs": "yacs",
    "torchvision": "torchvision",
    "torchtext": "torchtext",
    "torchdata": "torchdata",
    "timm": "timm",
    "tensorboard": "tensorboard",
    "pytorch-pretrained-bert": "pytorch-pretrained-bert",
    "transformers": "transformers",
    "Pillow": "Pillow",
    "opencv_python": "opencv-python",
    "ffmpeg_python": "ffmpeg-python",
    "scipy": "scipy",
    "cython": "Cython",
    "packaging": "packaging",
    "ftfy": "ftfy",
}


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return a file's SHA-256 without loading the multi-GB checkpoint."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(path: Path) -> str | None:
    """Read a checkout revision without changing its state."""

    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def readme_checkpoint_links(path: Path) -> dict[str, str]:
    """Parse the public Hugging Face checkpoint links from the README."""

    if not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8")
    links = re.findall(
        r"https://huggingface\.co/Gstar666/TASTVG/resolve/main/"
        r"TASTVG_(?:HCSTVG2|VidSTG|HCSTVG)\.pth\?download=true",
        text,
    )
    answer: dict[str, str] = {}
    for link in links:
        match = re.search(r"TASTVG_(HCSTVG2|VidSTG|HCSTVG)\.pth", link)
        if match:
            answer[match.group(1)] = link
    return answer


def checkpoint_records() -> dict[str, dict[str, Any]]:
    """Record URL, expected metadata, and local file verification."""

    records: dict[str, dict[str, Any]] = {}
    for dataset, spec in CHECKPOINTS.items():
        path = CHECKPOINT_DIR / str(spec["filename"])
        item = {
            "path": str(path.relative_to(PROJECT_ROOT)),
            "url": spec["url"],
            "hf_revision": spec.get("hf_revision"),
            "expected_size_bytes": spec["expected_size_bytes"],
            "expected_sha256": spec.get("expected_sha256"),
        }
        if "status_note" in spec:
            item["status_note"] = spec["status_note"]
        if path.is_file():
            actual_size = path.stat().st_size
            item["actual_size_bytes"] = actual_size
            item["actual_sha256"] = sha256_file(path)
            expected_size = spec.get("expected_size_bytes")
            item["size_matches"] = (
                expected_size is not None and actual_size == expected_size
            )
            expected_sha = spec.get("expected_sha256")
            item["sha256_matches"] = (
                expected_sha is not None and item["actual_sha256"] == expected_sha
            )
            item["status"] = (
                "verified"
                if item["size_matches"] and item["sha256_matches"]
                else "present_but_not_verified"
            )
        else:
            item["status"] = "not_present"
        records[dataset] = item
    return records


def model_asset_records() -> dict[str, dict[str, Any]]:
    """Record the non-TA release assets needed by the CPU runtime."""

    records: dict[str, dict[str, Any]] = {}
    for name, spec in MODEL_ASSETS.items():
        path = PROJECT_ROOT / str(spec["path"])
        item: dict[str, Any] = {
            "path": str(spec["path"]),
            "url": spec["url"],
        }
        if "note" in spec:
            item["note"] = spec["note"]
        if path.is_file():
            item.update(
                {
                    "status": "present",
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        else:
            item["status"] = "missing"
        records[name] = item

    roberta_root = PROJECT_ROOT / ".cache" / "huggingface" / "hub" / "models--roberta-base"
    revision_path = roberta_root / "refs" / "main"
    roberta: dict[str, Any] = {
        "path": str(roberta_root.relative_to(PROJECT_ROOT)),
        "url": "https://huggingface.co/FacebookAI/roberta-base",
    }
    if revision_path.is_file():
        revision = revision_path.read_text(encoding="utf-8").strip()
        snapshot = roberta_root / "snapshots" / revision
        required = [
            "config.json",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
        ]
        roberta.update(
            {
                "status": "present" if all((snapshot / f).is_file() for f in required) else "incomplete",
                "revision": revision,
                "snapshot": str(snapshot.relative_to(PROJECT_ROOT)),
                "required_files": required,
            }
        )
    else:
        roberta["status"] = "missing"
    records["roberta_base"] = roberta
    return records


def _versions_for_current_python() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for display_name, distribution_name in REQUIREMENT_PACKAGE_NAMES.items():
        try:
            packages[display_name] = importlib.metadata.version(distribution_name)
        except importlib.metadata.PackageNotFoundError:
            # The current TubeDETR prefix intentionally uses the headless
            # OpenCV wheel; it exposes the same cv2 import without GUI libs.
            if display_name == "opencv_python":
                try:
                    packages[display_name] = importlib.metadata.version(
                        "opencv-python-headless"
                    )
                except importlib.metadata.PackageNotFoundError:
                    packages[display_name] = None
            else:
                packages[display_name] = None
    packages["python"] = sys.version.split()[0]
    packages["ffmpeg_cli"] = shutil.which("ffmpeg")
    try:
        import torch  # noqa: WPS433 - deliberately local, CPU-only audit

        packages["torch"] = torch.__version__
        packages["torch_built_cuda"] = torch.version.cuda
    except Exception as exc:  # pragma: no cover - depends on caller env
        packages["torch_import_error"] = f"{type(exc).__name__}: {exc}"
    return packages


def _versions_for_prefix(prefix: Path) -> dict[str, Any]:
    """Inspect another local prefix without importing it into this process."""

    python_bin = prefix / "bin" / "python"
    if not python_bin.is_file():
        return {"status": "missing", "python": str(python_bin)}
    code = r"""
import importlib.metadata as md
import json
import shutil
import sys
names = {
    "tqdm": "tqdm", "yacs": "yacs", "torchvision": "torchvision",
    "torchtext": "torchtext", "torchdata": "torchdata",
    "timm": "timm", "tensorboard": "tensorboard",
    "pytorch-pretrained-bert": "pytorch-pretrained-bert",
    "transformers": "transformers", "Pillow": "Pillow",
    "opencv_python": "opencv-python", "ffmpeg_python": "ffmpeg-python",
    "scipy": "scipy", "cython": "Cython", "packaging": "packaging",
    "ftfy": "ftfy",
}
out = {"status": "ok", "python": sys.version.split()[0]}
out["ffmpeg_cli"] = shutil.which("ffmpeg")
for display, distribution in names.items():
    try:
        out[display] = md.version(distribution)
    except md.PackageNotFoundError:
        if display == "opencv_python":
            try:
                out[display] = md.version("opencv-python-headless")
            except md.PackageNotFoundError:
                out[display] = None
        else:
            out[display] = None
try:
    import torch
    out["torch"] = torch.__version__
    out["torch_built_cuda"] = torch.version.cuda
except Exception as exc:
    out["torch_import_error"] = f"{type(exc).__name__}: {exc}"
print(json.dumps(out, sort_keys=True))
"""
    try:
        child_env = dict(os.environ)
        child_env["PATH"] = ":".join(
            [str(python_bin.parent), child_env.get("PATH", "")]
        )
        result = subprocess.run(
            [str(python_bin), "-c", code],
            check=True,
            capture_output=True,
            text=True,
            env=child_env,
        )
        return json.loads(result.stdout.strip().splitlines()[-1])
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        return {
            "status": "probe_failed",
            "python": str(python_bin),
            "error": f"{type(exc).__name__}: {exc}",
        }


def environment_audit() -> dict[str, Any]:
    requirements = (
        (REPO_DIR / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if (REPO_DIR / "requirements.txt").is_file()
        else []
    )
    return {
        "official_requirements_file": str(
            (REPO_DIR / "requirements.txt").relative_to(PROJECT_ROOT)
        ),
        "official_requirements": [
            line.strip()
            for line in requirements
            if line.strip() and not line.lstrip().startswith("#")
        ],
        "current_project_prefix": _versions_for_current_python(),
        "isolated_tastvg_prefix": _versions_for_prefix(
            PROJECT_ROOT / ".conda" / "tastvg"
        ),
        "policy": (
            "The existing .conda/tubedetr torch/CUDA stack was not replaced. "
            "The pinned torchtext==0.15.2 stack lives in .conda/tastvg."
        ),
        "known_old_stack_blocker": (
            "A clean install of official transformers==4.5.1 reaches "
            "tokenizers==0.10.3, which has no CPython 3.11 wheel and needs "
            "a Rust compiler for its source build. The isolated audit prefix "
            "therefore keeps transformers==4.52.4 only for import-level "
            "inspection; it is not claimed to be the official runtime."
        ),
    }


def temporal_head_parameter_audit() -> dict[str, Any]:
    """Build source-equivalent MLPs and report exact names/shapes/counts."""

    import torch
    from torch import nn

    class TAStyleMLP(nn.Module):
        def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            num_layers: int,
            dropout: float = 0.0,
        ) -> None:
            super().__init__()
            self.num_layers = num_layers
            hidden = [hidden_dim] * (num_layers - 1)
            self.layers = nn.ModuleList(
                nn.Linear(n, k)
                for n, k in zip([input_dim] + hidden, hidden + [output_dim])
            )
            self.dropout = nn.Dropout(dropout) if dropout else None

        def forward(self, x: Any) -> Any:
            import torch.nn.functional as F

            for index, layer in enumerate(self.layers):
                x = F.relu(layer(x)) if index < self.num_layers - 1 else layer(x)
                if self.dropout is not None and index < self.num_layers:
                    x = self.dropout(x)
            return x

    def records(prefix: str, output_dim: int) -> tuple[list[dict[str, Any]], int]:
        module = TAStyleMLP(256, 256, output_dim, 2, dropout=0.3)
        rows: list[dict[str, Any]] = []
        for name, parameter in module.named_parameters():
            rows.append(
                {
                    "name": f"{prefix}.{name}",
                    "shape": list(parameter.shape),
                    "numel": parameter.numel(),
                }
            )
        return rows, sum(row["numel"] for row in rows)

    boundary, boundary_count = records("temp_embed", 2)
    action, action_count = records("action_embed", 1)
    return {
        "boundary_head": {
            "module": "TASTVGNet.temp_embed",
            "source_constructor": (
                "MLP(hidden_dim=256, hidden_dim=256, output_dim=2, "
                "num_layers=2, dropout=0.3)"
            ),
            "parameters": boundary,
            "trainable_scalar_count": boundary_count,
            "strict_head_only_default": True,
        },
        "actionness_head": {
            "module": "TASTVGNet.action_embed",
            "source_constructor": (
                "MLP(hidden_dim=256, hidden_dim=256, output_dim=1, "
                "num_layers=2, dropout=0.3)"
            ),
            "parameters": action,
            "trainable_scalar_count": action_count,
            "strict_head_only_default": False,
        },
        "boundary_plus_actionness_scalar_count": boundary_count + action_count,
        "do_not_include_for_strict_head_only": (
            "ground_decoder.time_decoder.* is an upstream temporal decoder, "
            "not part of temp_embed; the official optimizer groups it separately."
        ),
    }


def valid_span_entropy(
    logits: Any,
    valid_mask: Any | None = None,
    epsilon: float = 1e-8,
) -> tuple[Any, dict[str, float]]:
    """Compute normalized entropy over legal strict spans start < end.

    logits is B x T x 2. Endpoint distributions are independently softmaxed
    over valid frames, then their joint distribution is renormalized over the
    upper-triangular legal span set. Entropy is divided by
    log(N*(N-1)/2) per sample, matching vg_tta.tta. This strict convention is
    also the intended interpretation of TubeDETR's temporal decoder.
    TA-STVG's postprocessor constructs an inf-valued tril(0), which leaves
    the strict upper triangle (start < end) legal; this function expresses
    the same legal set directly.
    """

    import torch

    logits = logits.float()
    if logits.ndim != 3 or logits.shape[-1] != 2:
        raise ValueError("logits must have shape BxTx2")
    if valid_mask is None:
        valid_mask = torch.ones(
            logits.shape[:2], dtype=torch.bool, device=logits.device
        )
    else:
        valid_mask = valid_mask.to(device=logits.device, dtype=torch.bool)
    if valid_mask.shape != logits.shape[:2]:
        raise ValueError("valid_mask must have shape BxT")
    valid_counts = valid_mask.sum(dim=1)
    if (valid_counts < 2).any():
        raise ValueError("every sample needs at least two valid frames")

    normalized_rows = []
    raw_rows = []
    candidate_counts: list[int] = []
    for batch_index in range(logits.shape[0]):
        positions = torch.nonzero(valid_mask[batch_index], as_tuple=False).flatten()
        selected = logits[batch_index].index_select(0, positions)
        start_probability = selected[:, 0].softmax(dim=0)
        end_probability = selected[:, 1].softmax(dim=0)
        joint = start_probability[:, None] * end_probability[None, :]
        legal = torch.triu(torch.ones_like(joint, dtype=torch.bool), diagonal=1)
        legal_mass = joint.masked_select(legal)
        legal_mass = legal_mass / legal_mass.sum().clamp(min=epsilon)
        raw_entropy = -(
            legal_mass * legal_mass.clamp(min=epsilon).log()
        ).sum()
        count = int(legal.sum().item())
        candidate_counts.append(count)
        raw_rows.append(raw_entropy)
        if count == 1:
            normalized_rows.append(raw_entropy * 0.0)
        else:
            normalized_rows.append(
                raw_entropy / logits.new_tensor(float(count)).log()
            )
    normalized = torch.stack(normalized_rows)
    raw = torch.stack(raw_rows)
    loss = normalized.mean()
    return loss, {
        "loss_total": float(loss.detach().item()),
        "raw_temporal_span_entropy": float(raw.mean().detach().item()),
        "mean_valid_frames": float(valid_counts.float().mean().detach().item()),
        "mean_legal_span_count": float(sum(candidate_counts) / len(candidate_counts)),
    }


@contextlib.contextmanager
def capture_query_conditioned_temporal_features(model: Any) -> Iterator[list[Any]]:
    """Capture final query-conditioned per-frame decoder features.

    The official pipeline calls self.temp_embed(time_hiden_state), where
    time_hiden_state is [decoder_layers, batch, sampled_frames, 256]. A
    pre-hook therefore captures the exact features consumed by both temporal
    heads without changing model outputs:

        with capture_query_conditioned_temporal_features(model) as captured:
            outputs = model(videos, texts, targets, iteration_rate=-1)
        hidden = captured[-1]       # [L, B, T, 256], final decoder invocation
        per_frame = hidden[-1]      # [B, T, 256], final decoder layer
        pred_sted = outputs["pred_sted"]  # [B, T, 2]

    iteration_rate=-1 makes TA-STVG refine and invoke the decoder twice;
    captured[-1] deliberately selects the second/final invocation. Use
    iteration_rate=0 when a single decoder invocation is desired.
    """

    if not hasattr(model, "temp_embed"):
        raise AttributeError("model has no TASTVGNet.temp_embed module")
    captured: list[Any] = []

    def _pre_hook(_module: Any, inputs: tuple[Any, ...]) -> None:
        if not inputs:
            raise RuntimeError("temp_embed received no feature tensor")
        captured.append(inputs[0].detach())

    handle = model.temp_embed.register_forward_pre_hook(_pre_hook)
    try:
        yield captured
    finally:
        handle.remove()


def cpu_temporal_head_smoke() -> dict[str, Any]:
    """Run one synthetic B=1 CPU forward and head-only gradient check."""

    import torch
    from torch import nn

    torch.manual_seed(20260904)
    device = torch.device("cpu")

    class TAStyleTemporalHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.num_layers = 2
            self.layers = nn.ModuleList(
                [nn.Linear(256, 256), nn.Linear(256, 2)]
            )
            self.dropout = nn.Dropout(0.3)

        def forward(self, x: Any) -> Any:
            import torch.nn.functional as F

            for index, layer in enumerate(self.layers):
                x = F.relu(layer(x)) if index == 0 else layer(x)
                x = self.dropout(x)
            return x

    head = TAStyleTemporalHead().to(device)
    hidden = torch.randn(6, 1, 8, 256, device=device)
    hidden.requires_grad_(False)
    head.eval()
    with torch.no_grad():
        all_layer_logits = head(hidden)
    final_logits = all_layer_logits[-1]
    valid_mask = torch.tensor([[True] * 6 + [False] * 2], device=device)
    entropy, entropy_info = valid_span_entropy(final_logits, valid_mask)

    # Uniform logits over four frames have six equiprobable legal spans.
    uniform_loss, _ = valid_span_entropy(torch.zeros(1, 4, 2, device=device))

    # Verify that only the synthetic temporal head receives gradients while
    # the query-conditioned decoder feature tensor remains frozen.
    head.train()
    gradient_logits = head(hidden)
    gradient_loss, _ = valid_span_entropy(gradient_logits[-1], valid_mask)
    gradient_loss.backward()
    grad_names = [
        name
        for name, parameter in head.named_parameters()
        if parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
    ]
    return {
        "status": "passed",
        "device": str(device),
        "cuda_calls": False,
        "synthetic_decoder_features_shape": list(hidden.shape),
        "all_layer_logits_shape": list(all_layer_logits.shape),
        "final_pred_sted_shape": list(final_logits.shape),
        "valid_mask": valid_mask.tolist(),
        "entropy": entropy_info,
        "uniform_four_frame_entropy": float(uniform_loss.detach().item()),
        "head_gradient_parameter_names": grad_names,
        "hidden_requires_grad": bool(hidden.requires_grad),
        "hidden_grad_is_none": hidden.grad is None,
        "head_parameters_cpu": all(
            parameter.device.type == "cpu" for parameter in head.parameters()
        ),
    }


def build_report(run_smoke: bool) -> dict[str, Any]:
    readme_links = readme_checkpoint_links(REPO_DIR / "README.md")
    current_commit = git_commit(REPO_DIR)
    report: dict[str, Any] = {
        "repository": {
            "url": OFFICIAL_REPO,
            "path": str(REPO_DIR.relative_to(PROJECT_ROOT)),
            "commit": current_commit,
            "expected_commit": EXPECTED_COMMIT,
            "commit_matches_expected": current_commit == EXPECTED_COMMIT,
            "license_files": [
                path.name
                for path in REPO_DIR.iterdir()
                if path.is_file()
                and path.name.lower()
                in {"license", "license.md", "license.txt", "copying", "notice"}
            ]
            if REPO_DIR.is_dir()
            else [],
            "license_note": (
                "No LICENSE/COPYING/NOTICE file is present in this official "
                "checkout; GitHub API reported license=null. Do not infer a "
                "permissive license."
            ),
        },
        "readme_checkpoint_links": readme_links,
        "checkpoints": checkpoint_records(),
        "model_assets": model_asset_records(),
        "environment": environment_audit(),
        "temporal_head_parameter_audit": temporal_head_parameter_audit(),
        "smoke": {"status": "not_run"},
    }
    if run_smoke:
        try:
            report["smoke"] = cpu_temporal_head_smoke()
        except Exception as exc:  # pragma: no cover - report should preserve error
            report["smoke"] = {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="JSON report path (default: artifacts/phase3_tastvg_prep.json)",
    )
    parser.add_argument(
        "--no-smoke",
        action="store_true",
        help="skip the CPU-only synthetic temporal-head smoke",
    )
    args = parser.parse_args()

    report = build_report(run_smoke=not args.no_smoke)
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["smoke"]["status"] in {"passed", "not_run"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
