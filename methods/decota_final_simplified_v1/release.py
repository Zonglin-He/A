"""Check the cleaned runtime, independently of historical experiment locks."""

import json
from pathlib import Path

from .config import ROOT
from .tensors import sha256


def verify_release():
    path = Path(__file__).with_name("RELEASE.json")
    release = json.loads(path.read_text())
    for name, expected in release["code_pins"].items():
        if sha256(ROOT / name) != expected:
            raise RuntimeError(f"Clean DeCoTA release dependency changed: {name}")
    return release
