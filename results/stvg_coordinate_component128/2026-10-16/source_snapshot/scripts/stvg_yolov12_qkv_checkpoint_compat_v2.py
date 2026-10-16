"""Per-instance qkv AAttn compatibility with an explicit CUDA SDPA backend.

The pinned STVG checkpoint stores original YOLOv12 v1.0 fused-qkv AAttn
instances, while the pinned local checkout defines a later split qk/v class.
This adapter restores the v1.0 tensor layout and attention math without editing
the shared class or global state. CPU uses the upstream stable softmax branch.
CUDA uses PyTorch SDPA with Q/K/V cast to float16, the original 1/sqrt(d)
scale, no dropout, the original token order, and the original PE/projection
order. The CUDA backend can differ numerically from upstream FlashAttention;
bitwise equivalence is not claimed.

Upstream reference:
  https://github.com/sunsmarterjie/yolov12/blob/
  b1035f8f97cd1a7440454a54f8bc19f989448159/ultralytics/nn/modules/block.py
  SHA-256: 5af268a71abcc9bcba77122cedbee071bdafbe95e5f9ad8f25acf54406f82cf9
"""

from __future__ import annotations

from types import MethodType
from typing import Any


UPSTREAM_REPOSITORY = "https://github.com/sunsmarterjie/yolov12"
UPSTREAM_REVISION = "b1035f8f97cd1a7440454a54f8bc19f989448159"
UPSTREAM_FILE = "ultralytics/nn/modules/block.py"
UPSTREAM_FILE_SHA256 = "5af268a71abcc9bcba77122cedbee071bdafbe95e5f9ad8f25acf54406f82cf9"
CHECKPOINT_SHA256 = "682ce8dadee004dbe964950f1bf3eda451671815a6ed62db80b398916b9b7c6f"
CURRENT_REPOSITORY_REVISION = "2abab7153a065fb2925e8088e9ca2b19016ab7d6"
CURRENT_BLOCK_FILE_SHA256 = "e8ca08816f17d768bfe1613b7f6e009459ef6e9ca4d8d85bd1807dcb7afe72c8"


def _validate_checkpoint_aattn(module: Any, qualified_name: str) -> dict[str, Any]:
    """Validate the saved fused-QKV parameter layout without reading values."""
    if type(module).__name__ != "AAttn":
        raise TypeError(f"{qualified_name}: expected AAttn, got {type(module).__name__}")
    if not type(module).__module__.startswith("ultralytics.nn.modules.block"):
        raise TypeError(f"{qualified_name}: unexpected class module {type(module).__module__}")
    if not all(hasattr(module, key) for key in ("qkv", "proj", "pe", "area", "num_heads", "head_dim")):
        raise ValueError(f"{qualified_name}: missing v1.0 fused-QKV attributes")
    if hasattr(module, "qk") or hasattr(module, "v"):
        raise ValueError(f"{qualified_name}: split-QK/V layout is outside this adapter")

    qkv_conv = getattr(module.qkv, "conv", None)
    proj_conv = getattr(module.proj, "conv", None)
    pe_conv = getattr(module.pe, "conv", None)
    if qkv_conv is None or proj_conv is None or pe_conv is None:
        raise ValueError(f"{qualified_name}: expected Conv wrappers with .conv modules")

    qkv_shape = tuple(int(v) for v in qkv_conv.weight.shape)
    proj_shape = tuple(int(v) for v in proj_conv.weight.shape)
    pe_shape = tuple(int(v) for v in pe_conv.weight.shape)
    channels = int(module.num_heads) * int(module.head_dim)
    if qkv_shape != (3 * channels, channels, 1, 1):
        raise ValueError(f"{qualified_name}: incompatible qkv shape {qkv_shape}")
    if proj_shape != (channels, channels, 1, 1):
        raise ValueError(f"{qualified_name}: incompatible proj shape {proj_shape}")
    if pe_shape != (channels, 1, 7, 7):
        raise ValueError(f"{qualified_name}: incompatible pe shape {pe_shape}")
    if int(module.area) not in (1, 4):
        raise ValueError(f"{qualified_name}: unexpected checkpoint area {module.area}")
    if int(pe_conv.groups) != channels:
        raise ValueError(f"{qualified_name}: positional convolution is not depthwise")

    return {
        "name": qualified_name,
        "area": int(module.area),
        "num_heads": int(module.num_heads),
        "head_dim": int(module.head_dim),
        "qkv_weight_shape": list(qkv_shape),
        "proj_weight_shape": list(proj_shape),
        "pe_weight_shape": list(pe_shape),
    }


def _forward_aattn_backend(self: Any, x: Any, backend: str) -> Any:
    """Run fused-QKV AAttn with upstream stable softmax or half-QKV SDPA."""
    import torch
    import torch.nn.functional as functional

    if backend not in ("upstream_stable", "sdpa_half"):
        raise ValueError(f"unsupported AAttn backend: {backend}")
    if x.ndim != 4:
        raise ValueError(f"AAttn expects BCHW input; got shape {tuple(x.shape)}")
    batch, channels, height, width = x.shape
    token_count = height * width
    area = int(self.area)
    if token_count % area:
        raise ValueError(f"AAttn area {area} does not divide token count {token_count}")

    qkv = self.qkv(x).flatten(2).transpose(1, 2)
    if area > 1:
        qkv = qkv.reshape(batch * area, token_count // area, channels * 3)
        batch, token_count, _ = qkv.shape
    q, k, v = qkv.view(batch, token_count, self.num_heads, self.head_dim * 3).split(
        [self.head_dim, self.head_dim, self.head_dim], dim=3
    )

    if backend == "sdpa_half":
        # PyTorch SDPA consumes [batch, heads, sequence, head_dim]. Keep the
        # original sequence order and scale; cast inputs to half like the
        # upstream FlashAttention branch and restore the projection dtype.
        query = q.permute(0, 2, 1, 3).contiguous().to(torch.float16)
        key = k.permute(0, 2, 1, 3).contiguous().to(torch.float16)
        value = v.permute(0, 2, 1, 3).contiguous().to(torch.float16)
        attended = functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            scale=self.head_dim ** -0.5,
        )
        x = attended.to(q.dtype).permute(0, 2, 1, 3)
    else:
        # Pinned upstream v1.0 CPU arithmetic, including stable max subtraction.
        q = q.permute(0, 2, 3, 1)
        k = k.permute(0, 2, 3, 1)
        v = v.permute(0, 2, 3, 1)
        attention = (q.transpose(-2, -1) @ k) * (self.head_dim ** -0.5)
        maximum = attention.max(dim=-1, keepdim=True).values
        exponentials = torch.exp(attention - maximum)
        attention = exponentials / exponentials.sum(dim=-1, keepdim=True)
        x = v @ attention.transpose(-2, -1)
        x = x.permute(0, 3, 1, 2)
        v = v.permute(0, 3, 1, 2)

    if area > 1:
        x = x.reshape(batch // area, token_count * area, channels)
        v = v.reshape(batch // area, token_count * area, channels)
        batch, token_count, _ = x.shape
    x = x.reshape(batch, height, width, channels).permute(0, 3, 1, 2)
    v = v.reshape(batch, height, width, channels).permute(0, 3, 1, 2)
    x = x + self.pe(v)
    return self.proj(x)


def _installed_forward(self: Any, x: Any) -> Any:
    return _forward_aattn_backend(self, x, "sdpa_half" if x.is_cuda else "upstream_stable")


def install_checkpoint_qkv_aattn_compat_v2(model: Any) -> dict[str, Any]:
    """Bind the v2 AAttn forward to qkv-layout instances only.

    CUDA uses PyTorch SDPA with half Q/K/V; CPU retains upstream stable softmax.
    The operation is idempotent and does not modify shared class methods,
    parameters, state dicts, or torch backend globals.
    """
    if not hasattr(model, "named_modules"):
        raise TypeError("model must be a torch.nn.Module-like object")

    entries = []
    already_installed = 0
    for name, module in model.named_modules():
        if type(module).__name__ != "AAttn" or not type(module).__module__.startswith(
            "ultralytics.nn.modules.block"
        ):
            continue
        entry = _validate_checkpoint_aattn(module, name or "<root>")
        current_forward = getattr(module, "forward", None)
        if getattr(current_forward, "__func__", None) is _installed_forward:
            already_installed += 1
        else:
            module.forward = MethodType(_installed_forward, module)
        entries.append(entry)

    if not entries:
        raise ValueError("model contains no supported ultralytics AAttn instances")
    return {
        "schema": "stvg-yolov12-qkv-aattn-compat-v2",
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "upstream_repository": UPSTREAM_REPOSITORY,
        "upstream_revision": UPSTREAM_REVISION,
        "upstream_file": UPSTREAM_FILE,
        "upstream_file_sha256": UPSTREAM_FILE_SHA256,
        "current_repository_revision": CURRENT_REPOSITORY_REVISION,
        "current_block_file_sha256": CURRENT_BLOCK_FILE_SHA256,
        "aattn_count": len(entries),
        "already_installed_count": already_installed,
        "areas": {str(area): sum(entry["area"] == area for entry in entries) for area in (1, 4)},
        "cuda_attention_backend": "torch.nn.functional.scaled_dot_product_attention; Q/K/V float16",
        "cuda_scale": "head_dim ** -0.5",
        "cuda_dropout_p": 0.0,
        "cuda_is_causal": False,
        "cpu_attention_backend": "upstream-v1.0 max-subtracted stable softmax",
        "qkv_order_area_restore_pe_proj_unchanged": True,
        "bitwise_equivalence_claimed": False,
        "class_method_unchanged": True,
        "state_dict_unchanged_by_adapter": True,
        "gpu_execution_performed": False,
        "inference_success_claimed": False,
        "layers": entries,
    }
