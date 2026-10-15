"""Stage and verify the E1 anonymous public export; this script never commits or pushes."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/stvg_reference_recovery_v1/Reference_swap"
OUT = ROOT / "results/stvg_reference_swap/2026-10-15"
DOC = ROOT / "docs/STVG_REFERENCE_SWAP_V1.md"
PROTOCOL = ROOT / "protocols/stvg_reference_swap_v1.md"
CHECKOUT = ROOT.parent / "visual-grounding-public-A"
BRANCH = "research/stvg-opd-paper-hc2-revision-v2"
RESULT_FILES = {
    "E1_POSTSEAL_ROWS.json", "E1_POSTSEAL_SUMMARY.json", "E1_PAIRED_DELTAS.json",
    "E1_REFERENCE_QUALITY.json", "E1_TEACHER_READBACK.json", "E1_EXECUTION_COST.json",
    "E1_OPTIMIZER_MATH_READBACK.json",
    "E1_ROOT_READBACK.json", "E1_SCALAR_READBACK.json", "E1_SOURCE_BINDINGS.json", "REPORT_RECEIPT.json",
    "REFERENCE_SWAP_VIOU.png", "REFERENCE_SWAP_CONTRASTS.png", "README.md",
}
FIXED_FILES = {
    "docs/STVG_REFERENCE_SWAP_V1.md",
    "protocols/stvg_reference_swap_v1.md",
    "scripts/report_stvg_reference_swap_v1.py",
    "scripts/publish_stvg_reference_swap_v1.py",
    "scripts/measure_stvg_reference_swap_v1.py",
    "scripts/review_stvg_reference_swap_v1.py",
    "scripts/review_stvg_reference_swap_scalars_v1.py",
    "scripts/review_stvg_reference_swap_optimizer_v1.py",
    "scripts/review_stvg_reference_swap_costs_v1.py",
    "scripts/review_stvg_reference_swap_reference_quality_v1.py",
}
FORBIDDEN_JSON_TOKENS = (
    b'"query_text"', b'"caption"', b'"video_path"', b'"raw_output"',
    b'"source_path"', b'"frame_coordinates"', b'"optimizer_state"',
    b'"weights"', b'"GT_coordinates"',
)


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=CHECKOUT, text=True).strip()


def source_files() -> set[str]:
    receipt = read(BASE / "STUDENT_GLOBAL_BARRIER.json")
    assert receipt.get("status") == "complete" and receipt.get("GT_read") is False
    paths = set(FIXED_FILES)
    # Include only the E1 code files actually pinned by the sealed runtimes.
    for runtime_name in ("RUNTIME.json", "STUDENT_RUNTIME.json"):
        runtime = read(BASE / runtime_name)
        assert runtime.get("status") == "locked" and runtime.get("GT_read") is False
        for rel in runtime.get("pins", {}):
            if rel.startswith("scripts/") and "reference_swap" in Path(rel).name:
                paths.add(rel)
    paths.update(str(path.relative_to(ROOT)) for path in (ROOT / "scripts").glob("*reference_swap*.py"))
    return paths


def stage(branch: str) -> None:
    report_receipt = read(OUT / "REPORT_RECEIPT.json")
    assert report_receipt.get("status") == "complete" and report_receipt.get("GT_read") is True
    assert read(BASE / "ROOT_POSTSEAL_READBACK.json").get("status") == "complete"
    assert read(BASE / "ROOT_TEACHER_READBACK.json").get("status") == "pass"
    assert read(BASE / "ROOT_POSTSEAL_SCALAR_READBACK.json").get("status") == "pass"
    assert read(BASE / "ROOT_INDEPENDENT_OPTIMIZER_MATH_READBACK.json").get("status") == "pass"
    assert read(BASE / "ROOT_ACTUAL_EXECUTION_COST.json").get("status") == "pass"
    assert read(OUT / "E1_OPTIMIZER_MATH_READBACK.json").get("status") == "pass"
    assert read(OUT / "E1_EXECUTION_COST.json").get("status") == "pass"
    assert read(BASE / "ROOT_VISUAL_REVIEW.json").get("status") == "pass"
    paths = source_files() | {str((OUT / name).relative_to(ROOT)) for name in RESULT_FILES}
    for rel in paths:
        path = ROOT / rel
        assert path.is_file(), rel
        raw = path.read_bytes()
        assert path.suffix in {".py", ".md", ".json", ".png"}, rel
        if path.suffix == ".json":
            for token in FORBIDDEN_JSON_TOKENS:
                assert token not in raw, (rel, token)
        assert (b"/home/" + b"wwww/.codex/" + b"attachments/") not in raw, rel
    assert git("remote", "get-url", "origin") == "https://github.com/Zonglin-He/A.git"
    assert not git("diff", "--name-only") and not git("diff", "--cached", "--name-only")
    assert git("branch", "--show-current") == branch
    assert branch == BRANCH
    subprocess.run(["git", "fetch", "--quiet", "origin", "main", BRANCH], cwd=CHECKOUT, check=True)
    base_commit = git("rev-parse", "HEAD")
    assert base_commit == git("rev-parse", "origin/main") == git("rev-parse", f"origin/{BRANCH}")
    manifest = []
    for rel in sorted(paths):
        source = ROOT / rel
        target = CHECKOUT / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        raw = source.read_bytes()
        manifest.append({"path": rel, "sha256": sha(source), "bytes": len(raw),
                         "mode": "100644",
                         "blob_sha": hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()})
    subprocess.run(["git", "add", "--", *sorted(paths)], cwd=CHECKOUT, check=True)
    staged = set(git("diff", "--cached", "--name-only").splitlines())
    assert staged == paths
    expected_tree = git("write-tree")
    (BASE / "PUBLIC_STAGE.json").write_text(json.dumps({
        "schema": "stvg-reference-swap-public-stage-v1", "status": "staged_for_review",
        "branch": branch, "base_commit": base_commit, "expected_tree": expected_tree,
        "files": manifest, "file_count": len(manifest),
        "bytes": sum(row["bytes"] for row in manifest),
        "commit_or_push_performed": False, "time": time.time(), "GT_read": False,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"staged {len(manifest)} files for review; no commit or push was made")


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "STVG-reference-swap-verifier"})
    with urllib.request.urlopen(request, timeout=50) as response:
        return response.read()


def verify(commit: str) -> None:
    plan = read(BASE / "PUBLIC_STAGE.json")
    assert plan.get("status") == "staged_for_review" and plan.get("commit_or_push_performed") is False
    for branch in ("main", BRANCH):
        ref = json.loads(fetch("https://api.github.com/repos/Zonglin-He/A/git/ref/heads/" +
                               urllib.parse.quote(branch, safe="/")))
        assert ref["object"]["sha"] == commit
    record = json.loads(fetch(f"https://api.github.com/repos/Zonglin-He/A/git/commits/{commit}"))
    assert record["tree"]["sha"] == plan["expected_tree"]
    assert record["parents"][0]["sha"] == plan["base_commit"]
    tree = json.loads(fetch(f"https://api.github.com/repos/Zonglin-He/A/git/trees/{record['tree']['sha']}?recursive=1"))
    assert not tree.get("truncated")
    lookup = {row["path"]: row for row in tree["tree"]}
    def verify_one(row: dict[str, Any]) -> dict[str, Any]:
        entry = lookup[row["path"]]
        assert entry["mode"] == row["mode"] and entry["sha"] == row["blob_sha"]
        remote = fetch(f"https://raw.githubusercontent.com/Zonglin-He/A/{commit}/" +
                       urllib.parse.quote(row["path"], safe="/"))
        assert len(remote) == row["bytes"] and hashlib.sha256(remote).hexdigest() == row["sha256"]
        assert remote == (ROOT / row["path"]).read_bytes()
        assert remote == (CHECKOUT / row["path"]).read_bytes()
        return {"path": row["path"], "sha256": row["sha256"], "bytes": row["bytes"],
                "remote_bytes_identical": True}
    with ThreadPoolExecutor(max_workers=4) as pool:
        verified = list(pool.map(verify_one, plan["files"]))
    (BASE / "PUBLIC_VERIFY.json").write_text(json.dumps({
        "schema": "stvg-reference-swap-public-verify-v1", "status": "pass",
        "commit": commit, "tree_sha": record["tree"]["sha"], "files": verified,
        "file_count": len(verified), "bytes": sum(row["bytes"] for row in verified),
        "branches": ["main", BRANCH], "GT_read": False, "time": time.time(),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"verified {len(verified)} files at {commit}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    stage_parser = sub.add_parser("stage")
    stage_parser.add_argument("--branch", default=BRANCH)
    verify_parser = sub.add_parser("verify")
    verify_parser.add_argument("commit")
    args = parser.parse_args()
    if args.command == "stage":
        stage(args.branch)
    else:
        verify(args.commit)


if __name__ == "__main__":
    main()
