"""Static scaffold inspection; deliberately contains no experiment launcher."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / "studies/post_suite_tta_v1"
PRIVATE = ROOT / "outputs/post_suite_tta_v1"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text())


def inspect():
    registry = read(STUDY / "scaffold/sources.lock.json")
    errors = []
    checked = 0
    for method in registry["methods"]:
        inventory = PRIVATE / method["private_inventory"]
        if not inventory.exists():
            errors.append(f"{method['name']}: source inventory missing")
            continue
        if digest(inventory) != method["inventory_sha256"]:
            errors.append(f"{method['name']}: source inventory changed")
            continue
        for row in read(inventory)["files"]:
            file = PRIVATE / row["path"]
            if not file.is_file() or digest(file) != row["sha256"]:
                errors.append(f"{method['name']}: upstream file missing/changed: {row['path']}")
            checked += 1
    return {"source_files_checked": checked, "source_errors": errors,
            "imports_upstream_code": False, "model_or_GPU_actions": 0,
            "experiment_launch_available": False, "runtime_dependencies_installed": False,
            "methods": [{k: v for k, v in m.items() if k not in {"private_inventory", "inventory_sha256"}}
                        for m in registry["methods"]]}


def gate():
    active = read(PRIVATE / "active_intent.json")
    plan = PRIVATE / "plan_v2.json"
    reasons = []
    if digest(plan) != active["sha256"]:
        reasons.append("Active intent plan SHA changed")
    predecessor = PRIVATE / "predecessor_scope_completion.json"
    if not predecessor.exists():
        reasons.append("All previous experiments/publication not yet certified complete")
    else:
        reasons.append("Predecessor receipt needs independent root review and frozen execution-plan binding")
    reasons.append("New scientific/execution plans, exact assets and task mappings not yet frozen")
    return {"release_allowed": False, "reasons": reasons,
            "scope": "Scaffold has no model launcher; these conditions cannot be bypassed by a CLI flag"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["doctor", "gate", "paths"])
    args = parser.parse_args()
    if args.action == "doctor":
        result = inspect()
    elif args.action == "gate":
        result = gate()
    else:
        result = {"public_scaffold": str(STUDY / "scaffold"), "private_sources": str(PRIVATE / "sources")}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.action == "doctor" and result["source_errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
