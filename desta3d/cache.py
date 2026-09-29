"""Read only sealed cached tensors; never import a PTD loader or label scorer."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve(workspace, path):
    return (Path(workspace) / path).resolve()


class SealedCache:
    def __init__(self, config, workspace):
        self.root = resolve(workspace, config["root"])
        seal_path = self.root / "CACHE_SEAL.json"
        self.input_hashes = {}

        def verify(path, expected):
            actual = sha256(path)
            if not expected or actual != expected:
                raise ValueError(f"Missing/mismatched input SHA256: {path}")
            self.input_hashes[str(path)] = actual

        verify(seal_path, config["seal_sha256"])
        manifest = json.loads(seal_path.read_text())["files"]
        self.files = {}
        for split in ("train", "dev"):
            prefix = Path("cache") / split
            relative = sorted(
                Path(name) for name in manifest
                if Path(name).parent.parent == prefix and Path(name).name == "CACHE.pt"
            )
            if not relative:
                raise ValueError(f"No sealed {split} cache")
            self.files[split] = []
            for name in relative:
                path = (self.root / name).resolve()
                if not path.is_relative_to(self.root):
                    raise ValueError("Cache file escapes its root")
                verify(path, manifest[str(name)])
                self.files[split].append(path)

        basis_path = resolve(workspace, config["basis_file"])
        verify(basis_path, config["basis_sha256"])
        self.basis = torch.load(basis_path, map_location="cpu", weights_only=True)
        self.channel_basis = None
        if config["channel_basis_file"]:
            path = resolve(workspace, config["channel_basis_file"])
            verify(path, config["channel_basis_sha256"])
            self.channel_basis = torch.from_numpy(np.load(path, allow_pickle=False)).double()
            b = self.channel_basis
            if b.ndim != 2 or b.shape[0] != self.basis.shape[1]:
                raise ValueError("Channel basis does not match the union")
            if not torch.isfinite(b).all() or not torch.allclose(
                b.T @ b, torch.eye(b.shape[1], dtype=b.dtype), atol=2e-6, rtol=0
            ):
                raise ValueError("Channel basis must have orthonormal columns")
            self.basis = (self.basis.double() @ b).float()

    def get(self, split, index, device):
        data = torch.load(self.files[split][index], map_location="cpu", weights_only=True)
        required = ("z", "qT", "qS", "evidence8", "state33", "oracle_coeff256")
        if any(k not in data or not isinstance(data[k], torch.Tensor) for k in required):
            raise ValueError("Incomplete cached tensor contract")
        if data["z"].shape[0] != 1 or any(not torch.isfinite(data[k]).all() for k in required):
            raise ValueError("Expected one finite query per cache")
        target = data["oracle_coeff256"]
        if self.channel_basis is not None:
            # A0.4 contract: project on CPU FP64, then train on FP32 targets.
            target = (target.double() @ self.channel_basis).float()
        inputs = {k: data[k].to(device) for k in required[:-1]}
        return inputs, target.to(device)
