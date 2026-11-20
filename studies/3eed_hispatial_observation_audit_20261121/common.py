"""Shared private/public paths, verified local assets and stable IDs."""
import hashlib
import json
from pathlib import Path
import subprocess


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PRIVATE = HERE / "private"
OUT = HERE / "results"
PROTOCOL = HERE / "PROTOCOL.json"
ASSETS = HERE / "MODELS.json"
PLATFORMS = ("waymo", "drone", "quad")
ARMS = ("moge", "camera_rays", "global_scale", "prior_depth", "wrong_geometry")
SIDE = 448


def read(path):
    return json.loads(Path(path).read_bytes())


def write(path, obj):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def key(*values):
    return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()[:24]


def require_repo(repo_path, name):
    expected = read(PROTOCOL)["revision"][name]
    revision = subprocess.check_output(["git", "-C", str(repo_path), "rev-parse", "HEAD"], text=True).strip()
    if revision != expected:
        raise RuntimeError(f"{name} checkout {revision} instead of {expected}")


def require_weights(root, asset):
    from fetch_models import check_file
    specs = read(ASSETS)[asset]["files"]
    folder = root / asset
    for name, spec in specs.items():
        if not check_file(folder / name, spec):
            raise RuntimeError(f"Missing or invalid pinned asset: {asset}/{name}")
    return folder
