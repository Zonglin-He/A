#!/usr/bin/env python3
"""Build a code-only dependency manifest for the selected STVG main method.

Default behavior writes a manifest under main_method_release/ and creates the
small extracted pairwise helper once; it never moves, deletes, imports, or
executes project code.  --copy creates byte-identical source copies beneath
main_method_release/code/.  --verify-saved-temporal is optional CPU-only replay.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "main_method_release"
MANIFEST = OUT / "DEPENDENCY_MANIFEST.json"
PAIRWISE_SOURCE = "scripts/stvg_pairwise_caption256_v1.py"
PAIRWISE_DERIVED = "scripts/stvg_pairwise_primitives_v1.py"
PAIRWISE_FUNCTIONS = ("prompt", "combine")

# The release is rooted only in the new main-method package and archive tool.
# Legacy geometry/temporal/spatial routines enter through actual imports or
# executed source dependencies, not as independent historical roots.
ROOTS = ["scripts/research_archive.py"]

SOURCE_DIRS = ("scripts", "vg_tta", "methods", "main_method")
PROTECTED_NAME_PARTS = (
    "soft_ep_coordinate",
    "soft_ep_search",
    "soft_ep_tail",
)
EXCLUDED_CLASSES = (
    "checkpoints and model weights",
    "media, frames, annotations, and GT payloads",
    "raw Teacher responses, query/caption text, and per-source result payloads",
    "cached tensors, prediction files, and fit artifacts",
    "credentials, environment files, and personal conversations",
    "historical reports/results not required to import the runtime code",
)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def source_paths() -> list[Path]:
    found: set[Path] = set()
    for directory in SOURCE_DIRS:
        base = ROOT / directory
        if not base.exists():
            continue
        for path in base.rglob("*.py"):
            if any(part in {"__pycache__", ".pytest_cache"} for part in path.parts):
                continue
            found.add(path.resolve())
    return sorted(found)


def module_candidates(module: str) -> list[Path]:
    """Resolve a project import to source files without importing the module."""
    parts = module.split(".") if module else []
    if not parts or parts[0] not in SOURCE_DIRS:
        return []
    base = ROOT.joinpath(*parts)
    candidates = [base.with_suffix(".py"), base / "__init__.py"]
    return [p.resolve() for p in candidates if p.is_file()]


def imported_modules(tree: ast.AST, path: Path) -> list[tuple[str, str]]:
    package = list(path.relative_to(ROOT).with_suffix("").parts[:-1])
    out: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.append((alias.name, "import"))
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parent = package[: max(0, len(package) - node.level + 1)]
                module = ".".join(parent + ([node.module] if node.module else []))
            else:
                module = node.module or ""
            if module and module not in SOURCE_DIRS:
                out.append((module, "from"))
            elif module in SOURCE_DIRS and module_candidates(module):
                out.append((module, "from-package"))
            # `from scripts import local_module` is a common project pattern.
            if module in SOURCE_DIRS:
                for alias in node.names:
                    if alias.name != "*":
                        out.append((module + "." + alias.name, "from-submodule"))
    return out


def literal_local_py_paths(tree: ast.AST, current: Path) -> list[tuple[str, Path]]:
    """Follow explicit paths used as dynamic source inputs, not hash metadata."""
    out: list[tuple[str, Path]] = []
    dynamic_calls = {
        "open", "exec", "compile", "importlib.util.spec_from_file_location",
        "SourceFileLoader", "runpy.run_path",
    }
    for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
        try:
            name = ast.unparse(call.func)
        except Exception:
            continue
        if name not in dynamic_calls and not name.endswith((".read_text", ".read_bytes")):
            continue
        for node in ast.walk(call):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            value = node.value.strip()
            if not value.endswith(".py"):
                continue
            probes = [ROOT / value, current.parent / value]
            if "/" not in value and "\\" not in value:
                probes.extend((ROOT / "scripts" / value, ROOT / "main_method" / value))
            for probe in probes:
                try:
                    candidate = probe.resolve()
                    candidate.relative_to(ROOT)
                except (OSError, ValueError):
                    continue
                if candidate.is_file() and candidate.suffix == ".py":
                    if candidate.relative_to(ROOT).parts[0] in SOURCE_DIRS:
                        out.append((value, candidate))
                    break
    return out


def inspect_source_calls(tree: ast.AST) -> list[dict[str, object]]:
    rows = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = ast.unparse(node.func) if hasattr(ast, "unparse") else type(node.func).__name__
        if name.endswith("getsource") or name.endswith("exec") or name.endswith("compile"):
            arg = ast.unparse(node.args[0]) if node.args and hasattr(ast, "unparse") else None
            rows.append({"line": getattr(node, "lineno", None), "operation": name, "target": arg})
    return rows


def pairwise_primitive_source() -> tuple[str, dict[str, object]]:
    path = ROOT / PAIRWISE_SOURCE
    if not path.exists():
        # The clean public tree intentionally excludes the old experiment
        # controller. Its already extracted pure helpers are self-contained;
        # bind their exact bytes to the published extraction receipt.
        receipt = ROOT / 'results/stvg_main_qwen/2026-10-17/cleanup/CODE_MANIFEST.json'
        metadata = json.loads(receipt.read_text())['pairwise_helper_extraction']
        derived = (ROOT / PAIRWISE_DERIVED).read_text(encoding='utf-8')
        if hashlib.sha256(derived.encode()).hexdigest() != metadata['derived_sha256']:
            raise RuntimeError('published pure pairwise helper bytes changed')
        ast.parse(derived, filename=PAIRWISE_DERIVED)
        return derived, metadata
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=PAIRWISE_SOURCE)
    defs = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    labels_node = next((
        n for n in tree.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "LABELS" for t in n.targets)
    ), None)
    missing = [name for name in PAIRWISE_FUNCTIONS if name not in defs]
    if missing or labels_node is None:
        raise RuntimeError(f"pairwise primitive definitions missing: {missing}")
    source = path.read_text(encoding="utf-8")
    labels_source = ast.get_source_segment(source, labels_node)
    segments = [ast.get_source_segment(source, defs[name]) for name in PAIRWISE_FUNCTIONS]
    if labels_source is None or any(segment is None for segment in segments):
        raise RuntimeError("could not extract exact pairwise function source")
    derived = (
        '"""Pure prompt/reward helpers extracted from the pairwise experiment.\n'
        'The original training, Teacher, readout, and scoring entry points are excluded.\n'
        '"""\nimport json\n\n'
        + labels_source + "\n\n" + "\n\n".join(segments)
        + "\n"
    )
    ast.parse(derived, filename=PAIRWISE_DERIVED)
    return derived, {
        "source": PAIRWISE_SOURCE,
        "source_sha256": digest(path),
        "constant": "LABELS",
        "functions": list(PAIRWISE_FUNCTIONS),
        "derived_release_path": PAIRWISE_DERIVED,
        "derived_sha256": hashlib.sha256(derived.encode()).hexdigest(),
        "extraction": "exact AST source segments; prompt/combine only",
    }


def gather_main_method_roots() -> list[str]:
    base = ROOT / "main_method"
    if not base.exists():
        return []
    return [rel(p) for p in sorted(base.rglob("*.py")) if p.is_file()]


def active_unit_code_protection() -> dict[str, object]:
    """Best-effort unit path discovery; never returns command arguments."""
    units: list[str] = []
    protected: set[str] = set()
    try:
        raw = subprocess.run(
            ["systemctl", "list-units", "--type=service", "--state=active", "--no-legend", "--no-pager"],
            check=False, capture_output=True, text=True, timeout=3,
        ).stdout
        units = [line.split()[0] for line in raw.splitlines() if line.split()]
        for unit in units:
            if not any(part in unit.lower() for part in ("soft-ep-coordinate", "soft_ep_coordinate")):
                continue
            shown = subprocess.run(
                ["systemctl", "show", unit, "--property=ExecStart", "--value", "--no-pager"],
                check=False, capture_output=True, text=True, timeout=3,
            ).stdout
            for token in re.findall(r"(?:/[^\s;,'\"}]+\.py)", shown):
                candidate = Path(token)
                try:
                    candidate = candidate.resolve()
                    protected.add(rel(candidate))
                except (OSError, ValueError):
                    pass
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        units = []
    # Workspace/root may expose a different systemd namespace. Protect this
    # family even when list-units cannot see the root-owned active services.
    protected.update(
        rel(p) for p in source_paths()
        if p.parent == ROOT / "scripts" and any(part in p.name.lower() for part in PROTECTED_NAME_PARTS)
    )
    return {
        "unit_list_visible_in_this_namespace": units,
        "protected_source_paths": sorted(protected),
        "protection_rule": "never include protected soft-ep-coordinate code in omission/deletion candidates",
        "workspace_deletion_performed": False,
    }


def build_closure() -> dict[str, object]:
    derived, derived_meta = pairwise_primitive_source()
    ensure_pairwise_module(derived)
    roots = list(ROOTS) + gather_main_method_roots() + [PAIRWISE_DERIVED]
    missing_roots = [name for name in roots if not (ROOT / name).is_file()]
    pending = deque(ROOT / name for name in roots if (ROOT / name).is_file())
    included: dict[str, dict[str, object]] = {}
    imports: dict[str, set[str]] = defaultdict(set)
    externals: set[str] = set()
    literal_edges: dict[str, set[str]] = defaultdict(set)
    dynamic_ops: dict[str, list[dict[str, object]]] = {}
    external_path_categories: dict[str, set[str]] = defaultdict(set)
    local_sources = source_paths()
    namespace_top = {p.split(".", 1)[0] for p in SOURCE_DIRS}

    while pending:
        path = pending.popleft().resolve()
        try:
            name = rel(path)
        except ValueError:
            continue
        if name in included:
            continue
        # Python executes every existing package initializer before importing
        # a submodule. Preserve those bytes and their own import dependencies;
        # replacing a real package with an implicit namespace is not cleanup.
        ancestor = path.parent
        while ancestor != ROOT and ancestor.is_relative_to(ROOT):
            initializer = ancestor / '__init__.py'
            if initializer.is_file() and initializer != path:
                imports[name].add(rel(initializer))
                pending.append(initializer)
            ancestor = ancestor.parent
        raw = path.read_bytes()
        text = raw.decode("utf-8")
        tree = ast.parse(text, filename=name)
        included[name] = {
            "path": name,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
            "ast_syntax": "pass",
        }
        dynamic_ops[name] = inspect_source_calls(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str) or not node.value.strip().endswith(".py"):
                continue
            value = node.value.strip().replace("\\", "/")
            first = value.split("/", 1)[0]
            if first in {".conda", ".venv", "external", "artifacts"}:
                external_path_categories[name].add(first)
            elif "private_case" in value:
                external_path_categories[name].add("private_case*")

        for module, kind in imported_modules(tree, path):
            candidates = module_candidates(module)
            if candidates:
                for candidate in candidates:
                    imports[name].add(rel(candidate))
                    pending.append(candidate)
            elif module.split(".", 1)[0] in namespace_top:
                # Missing names such as `from scripts import X` may be
                # attribute imports; only report unresolved if no submodule exists.
                if kind != "from-submodule":
                    imports[name].add("UNRESOLVED:" + module)
            else:
                externals.add(module.split(".", 1)[0])

        for literal, candidate in literal_local_py_paths(tree, path):
            target = rel(candidate)
            literal_edges[name].add(target)
            pending.append(candidate)

    # The extracted pure helpers are a derived release file, not a byte-copy
    # of the larger experimental script.
    included[PAIRWISE_DERIVED] = {
        "path": PAIRWISE_DERIVED,
        "sha256": derived_meta["derived_sha256"],
        "bytes": len(derived.encode()),
        "ast_syntax": "pass",
        "derived_from": PAIRWISE_SOURCE,
        "functions": list(PAIRWISE_FUNCTIONS),
    }
    all_scripts = sorted(p for p in local_sources if p.parent == ROOT / "scripts")
    closure_paths = set(included)
    protection = active_unit_code_protection()
    protected = set(protection["protected_source_paths"])
    omitted = [rel(p) for p in all_scripts if rel(p) not in closure_paths and rel(p) not in protected]
    protected_omitted = [rel(p) for p in all_scripts if rel(p) not in closure_paths and rel(p) in protected]
    return {
        "schema": "stvg-main-method-release-dependency-manifest-v1",
        "scope": "Qwen3.5 temporal pairwise preference + Direct four-anchor 1792-parameter episodic Gaussian OPD + Native marginal-MAP readout",
        "roots": roots,
        "missing_roots": missing_roots,
        "source_files": [included[k] for k in sorted(included)],
        "source_file_count": len(included),
        "internal_import_edges": {k: sorted(v) for k, v in sorted(imports.items())},
        "explicit_local_py_edges": {k: sorted(v) for k, v in sorted(literal_edges.items())},
        "inspect_exec_compile_sites": dynamic_ops,
        "excluded_local_path_dependency_categories": {
            k: sorted(v) for k, v in sorted(external_path_categories.items())
        },
        "external_import_roots": sorted(externals),
        "pairwise_helper_extraction": derived_meta,
        "excluded_payload_classes": list(EXCLUDED_CLASSES),
        "active_unit_protection": protection,
        "omitted_unrelated_top_level_scripts_for_release_only": omitted,
        "protected_omitted_top_level_scripts_not_candidates": protected_omitted,
        "workspace_source_deletion_performed": False,
        "release_materialized": False,
        "verification": {
            "AST_parse": "pass for all included source files",
            "source_hashes": "original runtime sources are byte-hashed without rewriting; pairwise helper has a separate AST-extraction hash",
            "import_resolution": "static import and dynamic-source closure rooted in main_method, research_archive, and extracted pairwise helpers; no project modules imported",
            "temporal_saved_fit_decode_replay": "not run; opt-in --verify-saved-temporal",
            "new_GT_or_model_or_score_calls": 0,
        },
        "known_full_cohort_gaps": [
            "main_method implements an official_clean roster assertion for 13,785 queries (HC2 3,482/237 parents; VidSTG 10,303/732), per-query files, and global barriers; those stages have not been run or independently qualified here.",
            "The engineering32 scope and official_clean scope share the new stage code, but no full-population single-Qwen Teacher/Student execution or post-seal scoring was performed by this builder.",
            "Teacher inputs are hash-verified immediately before use; the scoring seal checks exact unique roster aliases and per-query identity. These are code-reviewed guards, not evidence that new all-query execution has completed.",
            "The official 109-batch dispatcher is implemented but remains gated on actual engineering ROOT, CPU scoring, independent review and remote publication receipts.",
            "No new all-query inference, GT scoring, publication, or CURRENT/PAPER promotion was performed by this builder.",
        ],
        "exact_equivalent_acceleration_candidates": [
            "Reuse decoded source frames and deterministic processor outputs only under content-addressed keys; require exact input tensor hashes against the reference path.",
            "Keep one bounded Qwen GPU queue and overlap only CPU roster validation, hash checks, and immutable input decoding; preserve claim order and one-call semantics.",
            "Reuse the already sealed temporal fit and logits for the decoder-only stage; recompute the full physical-interval mass table on CPU and require exact interval/tie equality.",
            "Do not change dtype, reduction order, tie handling, optimizer, candidate set, prompt, frame sampler, or retry rules under an engineering-only acceleration.",
        ],
    }, derived


def ensure_pairwise_module(derived: str) -> None:
    """Create the importable pure helper once; never overwrite changed source."""
    path = ROOT / PAIRWISE_DERIVED
    expected = hashlib.sha256(derived.encode("utf-8")).hexdigest()
    if path.exists():
        if digest(path) != expected:
            raise FileExistsError(f"refusing to replace non-matching helper source: {rel(path)}")
        return
    path.write_text(derived, encoding="utf-8")


def materialize_code(manifest: dict[str, object], derived: str, *, force: bool) -> None:
    code_root = OUT / "code"
    files = manifest["source_files"]
    expected: set[str] = set()
    for item in files:
        src_name = item["path"]
        expected.add(src_name)
        dst = code_root / src_name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            if digest(dst) == item["sha256"]:
                pass
            elif not force:
                raise FileExistsError(f"refusing to overwrite non-matching release file: {dst}")
            else:
                shutil.copyfile(ROOT / src_name, dst)
        else:
            shutil.copyfile(ROOT / src_name, dst)
        if digest(dst) != item["sha256"]:
            raise RuntimeError(f"copy SHA mismatch: {src_name}")
    derived_dst = code_root / PAIRWISE_DERIVED
    derived_dst.parent.mkdir(parents=True, exist_ok=True)
    if derived_dst.exists() and not force and digest(derived_dst) != hashlib.sha256(derived.encode()).hexdigest():
        raise FileExistsError(f"refusing to overwrite non-matching derived file: {derived_dst}")
    if not derived_dst.exists() or digest(derived_dst) != hashlib.sha256(derived.encode()).hexdigest():
        derived_dst.write_text(derived, encoding="utf-8")
    if digest(derived_dst) != hashlib.sha256(derived.encode()).hexdigest():
        raise RuntimeError("derived helper SHA mismatch")
    pruned: list[str] = []
    protected_release = set(manifest.get("protected_omitted_top_level_scripts_not_candidates", []))
    retained_protected: list[str] = []
    if force and code_root.exists():
        for stale in sorted(code_root.rglob("*.py")):
            stale_rel = stale.relative_to(code_root).as_posix()
            if stale_rel not in expected:
                if stale_rel in protected_release:
                    retained_protected.append(stale_rel)
                    continue
                stale.unlink()
                pruned.append(stale_rel)
        for directory in sorted((p for p in code_root.rglob("*") if p.is_dir()), reverse=True):
            try:
                directory.rmdir()
            except OSError:
                pass
    manifest["release_materialized"] = True
    manifest["release_pruned_stale_python_files"] = pruned
    manifest["protected_release_python_files_retained"] = retained_protected


def verify_saved_temporal(max_cases: int | None = None) -> dict[str, object]:
    """Replay saved pairwise fits/decodes on CPU; no model, scorer, or GT reader."""
    import numpy as np
    import torch
    from scripts.stvg_evidence_common_v1 import load_case
    from scripts.stvg_table1_temporal_adapter_v1 import _hidden_from_source_case
    from scripts import stvg_temporal_marginal_train_fit256_v1 as fit_module
    from scripts.stvg_temporal_marginal_decode_util256_v1 import decode

    base = ROOT / "artifacts/stvg_pairwise_caption256_v1"
    roster = json.loads((base / "ROSTER_LOCK.json").read_text(encoding="utf-8"))["records"]
    jobs = json.loads((base / "INPUT_BARRIER.json").read_text(encoding="utf-8"))["jobs"]
    job_map = {row["case_id"]: row for row in jobs}
    fit_maps = {}
    for kind in ("R1", "QWEN"):
        barrier = json.loads((base / kind / "FIT_BARRIER.json").read_text(encoding="utf-8"))
        fit_maps[kind] = {row["case_id"]: row for row in barrier["files"]}
    torch.set_num_threads(2)
    checked = 0
    checked_fit = 0
    checked_decode = 0
    for record in roster:
        if max_cases is not None and checked >= max_cases:
            break
        cid = record["probe_alias"]
        case, _ = load_case(record)
        if case.get("GT_read") is not False:
            raise RuntimeError("saved temporal replay requires GT-free source case")
        job_ref = job_map[cid]
        job = json.loads((ROOT / job_ref["path"]).read_text(encoding="utf-8"))
        offsets = fit_module.native.offsets(case, _hidden_from_source_case(case))
        pred_path = base / "predictions" / (cid + ".pt")
        pred = torch.load(pred_path, map_location="cpu", weights_only=False)
        for kind in ("R1", "QWEN"):
            teacher = json.loads((base / kind / "teacher" / (cid + ".json")).read_text(encoding="utf-8"))
            stored_ref = fit_maps[kind][cid]
            saved = torch.load(ROOT / stored_ref["path"], map_location="cpu", weights_only=False)["fit"]
            recomputed = fit_module.fit(offsets, job["selection"]["hypotheses"], teacher["rewards"])
            if not np.array_equal(saved["theta"], recomputed["theta"]):
                raise AssertionError(f"saved fit theta mismatch: {cid}/{kind}")
            if saved["optimizer_steps"] != recomputed["optimizer_steps"] or saved["final_interval"] != recomputed["final_interval"]:
                raise AssertionError(f"saved fit state mismatch: {cid}/{kind}")
            if len(saved["final_logits"]) != len(recomputed["final_logits"]) or not all(
                torch.equal(a.cpu(), b.cpu()) for a, b in zip(saved["final_logits"], recomputed["final_logits"])
            ):
                raise AssertionError(f"saved fit logits mismatch: {cid}/{kind}")
            decoded = decode(case["records"], saved["final_logits"])[0]
            if pred["arms"][kind + "_PAIRWISE"]["interval"] != decoded:
                raise AssertionError(f"saved merged-interval decode mismatch: {cid}/{kind}")
            if not all(torch.equal(a.cpu(), b.cpu()) for a, b in zip(pred["actual_integrated_head_logits"][kind], saved["final_logits"])):
                raise AssertionError(f"integrated head output mismatch: {cid}/{kind}")
            checked_fit += 1
            checked_decode += 1
        checked += 1
    return {
        "status": "pass",
        "sources_checked": checked,
        "fits_exact": checked_fit,
        "decodes_exact": checked_decode,
        "GT_reads": 0,
        "model_calls": 0,
        "score_calls": 0,
        "all_sources": max_cases is None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--copy", action="store_true", help="copy only code closure under main_method_release/code")
    parser.add_argument("--force", action="store_true", help="allow replacing matching release outputs")
    parser.add_argument("--verify-saved-temporal", action="store_true", help="replay sealed temporal fits/decodes on CPU, without GT")
    parser.add_argument("--max-replay-cases", type=int, default=None)
    parser.add_argument("--refresh-manifest", action="store_true", help="replace an existing manifest after explicit review")
    args = parser.parse_args()
    if args.max_replay_cases is not None and args.max_replay_cases < 1:
        parser.error("--max-replay-cases must be positive")

    manifest, derived = build_closure()
    if args.verify_saved_temporal:
        manifest["verification"]["temporal_saved_fit_decode_replay"] = verify_saved_temporal(args.max_replay_cases)
    if args.copy:
        materialize_code(manifest, derived, force=args.force)
    OUT.mkdir(parents=True, exist_ok=True)
    if MANIFEST.exists() and not (args.refresh_manifest or args.force):
        raise FileExistsError(f"manifest already exists; pass --refresh-manifest to update: {MANIFEST}")
    temp = MANIFEST.with_suffix(".json.tmp")
    temp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, MANIFEST)
    print(json.dumps({
        "manifest": rel(MANIFEST),
        "source_file_count": manifest["source_file_count"],
        "missing_roots": manifest["missing_roots"],
        "external_import_root_count": len(manifest["external_import_roots"]),
        "omitted_top_level_scripts": len(manifest["omitted_unrelated_top_level_scripts_for_release_only"]),
        "protected_omitted_top_level_scripts": len(manifest["protected_omitted_top_level_scripts_not_candidates"]),
        "release_materialized": manifest["release_materialized"],
        "workspace_deletion_performed": False,
    }, indent=2))
    return 0 if not manifest["missing_roots"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
