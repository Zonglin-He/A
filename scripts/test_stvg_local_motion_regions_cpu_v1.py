"""Run synthetic-only contracts for the local-motion region generator."""

from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_stvg_local_motion_regions_v1 import ROOT as PROJECT_ROOT, synthetic_contracts


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    result = synthetic_contracts()
    result["source_pins"] = {
        "generator": {
            "path": "scripts/run_stvg_local_motion_regions_v1.py",
            "sha256": sha(PROJECT_ROOT / "scripts/run_stvg_local_motion_regions_v1.py"),
        },
        "contract_runner": {
            "path": "scripts/test_stvg_local_motion_regions_cpu_v1.py",
            "sha256": sha(Path(__file__).resolve()),
        },
    }
    receipt = PROJECT_ROOT / "artifacts/stvg_local_motion_regions_v1/CPU_CONTRACTS.json"
    receipt.parent.mkdir(parents=True, exist_ok=True)
    content = (json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    if receipt.exists():
        if receipt.read_bytes() != content:
            raise RuntimeError("synthetic contract receipt already exists and differs; refusing to overwrite")
    else:
        receipt.write_bytes(content)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))
    print(f"contract_receipt={receipt.relative_to(PROJECT_ROOT).as_posix()} sha256={sha(receipt)}")
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
