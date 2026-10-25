"""Static generic TTA source/config checker. No upstream execution."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
PRIVATE = ROOT / "outputs/post_suite_tta_v1"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def doctor():
    registry = read(HERE / "sources.lock.json")
    checked = 0
    failures = []
    methods = []
    for method in registry["methods"]:
        inventory = PRIVATE / method["private_inventory"]
        if sha(inventory) != method["inventory_sha256"]:
            failures.append(method["name"] + ": inventory SHA changed")
            continue
        for row in read(inventory)["files"]:
            path = PRIVATE / row["path"]
            if not path.is_file() or sha(path) != row["sha256"]:
                failures.append(method["name"] + ": source missing/changed: " + row["path"])
            checked += 1
        definitions = []
        for relative in method["core_files"]:
            path = PRIVATE / "generic_sources" / method["name"] / relative
            tree = ast.parse(path.read_text())
            definitions.extend(n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)))
        methods.append({"name": method["name"], "commit": method["commit"],
                        "core_definitions": definitions, "runtime_ready": False})
    return {"source_files_checked": checked, "failures": failures, "methods": methods,
            "upstream_imports": 0, "model_GPU_GT_actions": 0, "experiment_launcher": False}


def gate():
    active = read(PRIVATE / "active_intent.json")
    path = ROOT / active["active_plan"]
    valid = sha(path) == active["sha256"]
    return {"active_intent_hash_valid": valid, "release_allowed": False,
            "predecessor_completion_receipt_exists": (PRIVATE / "predecessor_scope_completion.json").exists(),
            "new_scientific_execution_plan_frozen": False,
            "reason": "Source preparation only. Existing full-suite completion and independently frozen execution required."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["doctor", "configs", "gate"])
    args = parser.parse_args()
    if args.action == "doctor":
        result = doctor()
    elif args.action == "configs":
        result = {p.stem: read(p) for p in sorted((HERE / "configs").glob("*.json"))}
    else:
        result = gate()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.action == "doctor" and result["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
