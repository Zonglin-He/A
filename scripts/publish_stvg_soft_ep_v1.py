#!/usr/bin/env python3
"""Stage and locally verify the anonymous Soft EP report; never commits or pushes."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/stvg_soft_ep_opd_v1"
OUT = ROOT / "results/stvg_soft_ep_opd/2026-10-15"
DOC = ROOT / "docs/STVG_SOFT_EP_OPD_V1.md"
PROTOCOL = ROOT / "protocols/stvg_soft_ep_opd_v1.md"
DEFAULT_CHECKOUT = ROOT.parent / "visual-grounding-public-A"
EXPECTED_REMOTE = "https://github.com/Zonglin-He/A.git"

SOURCE_FILES = (
    "scripts/audit_stvg_soft_ep_v1.py",
    "scripts/report_stvg_soft_ep_v1.py",
    "scripts/publish_stvg_soft_ep_v1.py",
    "scripts/lock_stvg_soft_ep_v1.py",
    "scripts/launch_stvg_soft_ep_v1.py",
    "scripts/review_stvg_soft_ep_v1.py",
    "scripts/run_stvg_soft_ep_v1.py",
    "scripts/lock_stvg_soft_ep_trial_v1.py",
    "scripts/launch_stvg_soft_ep_trial_v1.py",
    "scripts/run_stvg_soft_ep_trial_v1.py",
    "scripts/run_stvg_soft_ep_trial_worker_v1.py",
    "scripts/review_stvg_soft_ep_trial_v1.py",
    "scripts/stvg_soft_ep_fit_v1.py",
    "scripts/stvg_soft_ep_response_v1.py",
    "scripts/measure_stvg_soft_ep_v1.py",
    "scripts/stvg_reference_recovery_common_v1.py",
    "scripts/stvg_reference_recovery_fit_v1.py",
    "scripts/run_stvg_table1_student004.py",
    "scripts/stvg_target_teacher_student_v1.py",
    "scripts/stvg_weightfree_coordinate_fit_v1.py",
    "scripts/stvg_evidence_common_v1.py",
    "scripts/stvg_native_instance_common_v1.py",
    "methods/decota_spatial_opd_v1/configs.json",
    "methods/decota_final_simplified_v1/tensors.py",
    "vg_tta/decota_spatial_opd_tunable_v1.py",
    "vg_tta/tastvg_decota_critic_p0_v1.py",
)
RESULT_FILES = (
    "ANONYMOUS_ROWS.json", "ANONYMOUS_SUMMARIES.json",
    "ANONYMOUS_PAIRED_CONTRASTS.json", "ANONYMOUS_PROCESS_READBACK.json",
    "SOURCE_BINDINGS.json", "SOFT_EP_VIOU_MEANS.png", "SOFT_EP_DIAGNOSTICS.png",
    "REPORT_RECEIPT.json",
)
ALLOWED_SUFFIXES = {".py", ".md", ".json", ".png"}
FORBIDDEN_JSON_TOKENS = (
    b'"query_text"', b'"caption"', b'"video_path"', b'"raw_output"',
    b'"source_path"', b'"physical_frame_ids"', b'"box_xyxy"',
    b'"box_cxcywh"', b'"state_before"', b'"gradient"', b'"optimizer_state"',
    b'"GT_coordinates"', b'"weights"',
)


class PublishError(RuntimeError):
    pass


def need(condition: Any, message: str) -> None:
    if not condition:
        raise PublishError(message)


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(checkout: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=checkout, text=True).strip()


def tracked(checkout: Path, rel: str) -> bool:
    result = subprocess.run(["git", "ls-files", "--error-unmatch", "--", rel],
                            cwd=checkout, text=True, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    return result.returncode == 0


def expected_paths() -> set[str]:
    receipt = read(OUT / "REPORT_RECEIPT.json")
    need(receipt.get("status") == "complete" and receipt.get("trial_count") == 3
         and receipt.get("total_arm_rows") == 768
         and receipt.get("total_summary_cells") == 240
         and receipt.get("total_paired_cells") == 360,
         "report receipt is incomplete")
    need(read(BASE / "ROOT_INDEPENDENT_CPU_AUDIT.json").get("status") == "pass"
         and read(BASE / "mu025/ROOT_INDEPENDENT_CPU_AUDIT.json").get("status") == "pass"
         and read(BASE / "mu4/ROOT_INDEPENDENT_CPU_AUDIT.json").get("status") == "pass",
         "one or more independent CPU audits are incomplete")
    need(read(BASE / "ROOT_ROOT_REPEAT_CPU_AUDIT.json").get("status") == "pass"
         and read(BASE / "mu025/ROOT_ROOT_REPEAT_CPU_AUDIT.json").get("status") == "pass"
         and read(BASE / "mu4/ROOT_ROOT_REPEAT_CPU_AUDIT.json").get("status") == "pass",
         "one or more root repeat audits are incomplete")
    return set(SOURCE_FILES) | {str(DOC.relative_to(ROOT)), str(PROTOCOL.relative_to(ROOT))} | {
        str((OUT / name).relative_to(ROOT)) for name in RESULT_FILES
    }


def verify_privacy(paths: set[str]) -> None:
    for rel in paths:
        src = ROOT / rel
        need(src.is_file(), f"missing staged source: {rel}")
        need(src.suffix in ALLOWED_SUFFIXES, f"unapproved public file type: {rel}")
        raw = src.read_bytes()
        if src.suffix == ".json":
            for token in FORBIDDEN_JSON_TOKENS:
                need(token not in raw, f"forbidden private field in {rel}: {token.decode()}")
        attachment_prefix = b"/home/" + b"wwww/.codex/attachments/"
        workspace_prefix = b"/home/" + b"wwww/visual grounding/"
        need(attachment_prefix not in raw, f"private attachment path in {rel}")
        need(workspace_prefix not in raw, f"absolute workspace path in {rel}")
        need(src.suffix not in {".pt", ".pth", ".safetensors", ".npy", ".mp4", ".mkv"},
             f"private tensor/media file in {rel}")


def stage(checkout: Path, branch: str | None, replace: bool) -> None:
    checkout = checkout.resolve()
    need(checkout.is_dir() and (checkout / ".git").exists(), "public checkout is missing or not a Git repository")
    need(git(checkout, "remote", "get-url", "origin") == EXPECTED_REMOTE,
         "public checkout remote differs from the established destination")
    current_branch = git(checkout, "branch", "--show-current")
    need(branch is None or branch == current_branch,
         "stage only supports the already checked-out branch; it will not switch or create branches")
    need(not git(checkout, "diff", "--cached", "--name-only"),
         "public checkout has existing staged changes; refusing to mix exports")
    need(not git(checkout, "diff", "--name-only"),
         "public checkout has tracked modifications; refusing to mix exports")
    paths = expected_paths()
    verify_privacy(paths)
    manifest = []
    staged_paths = []
    for rel in sorted(paths):
        source, target = ROOT / rel, checkout / rel
        raw = source.read_bytes()
        target_same = target.is_file() and target.read_bytes() == raw
        is_tracked = tracked(checkout, rel)
        needs_stage = not (target_same and is_tracked)
        if target.exists() and not target_same:
            need(replace, f"target has different bytes; pass --replace to overwrite only this allowlisted file: {rel}")
        if not target_same:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        blob = hashlib.sha1(f"blob {len(raw)}\0".encode("ascii") + raw).hexdigest()
        manifest.append({"path": rel, "sha256": sha(source), "bytes": len(raw),
                         "mode": "100644", "blob_sha": blob, "staged": needs_stage})
        if needs_stage:
            staged_paths.append(rel)
    if staged_paths:
        subprocess.run(["git", "add", "--", *sorted(staged_paths)], cwd=checkout, check=True)
    staged = set(git(checkout, "diff", "--cached", "--name-only").splitlines())
    need(staged == set(staged_paths), "staged path set differs from the explicit changed-file allowlist")
    receipt = {
        "schema": "stvg-soft-ep-public-stage-v1", "status": "staged_for_review",
        "branch": current_branch, "base_commit": git(checkout, "rev-parse", "HEAD"),
        "expected_tree": git(checkout, "write-tree"), "files": manifest,
        "file_count": len(manifest), "staged_paths": sorted(staged_paths),
        "staged_file_count": len(staged_paths), "bytes": sum(x["bytes"] for x in manifest),
        "commit_performed": False, "push_performed": False,
    }
    (BASE / "PUBLIC_STAGE.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
                                             encoding="utf-8")
    print(json.dumps({"status": "staged_for_review", "file_count": len(manifest),
                      "bytes": receipt["bytes"], "commit_performed": False,
                      "push_performed": False}, ensure_ascii=False))


def verify_stage(checkout: Path) -> None:
    checkout = checkout.resolve()
    plan = read(BASE / "PUBLIC_STAGE.json")
    need(plan.get("status") == "staged_for_review" and plan.get("commit_performed") is False
         and plan.get("push_performed") is False, "no valid local staging receipt")
    need(git(checkout, "remote", "get-url", "origin") == EXPECTED_REMOTE, "wrong public remote")
    need(git(checkout, "branch", "--show-current") == plan["branch"], "branch changed after staging")
    need(git(checkout, "rev-parse", "HEAD") == plan["base_commit"], "base commit changed after staging")
    need(git(checkout, "write-tree") == plan["expected_tree"], "staged tree changed after staging")
    rows = plan.get("files", [])
    paths = {row["path"] for row in rows}
    need(len(paths) == len(rows) == plan.get("file_count"), "stage manifest path/count mismatch")
    staged_paths = set(plan.get("staged_paths", []))
    need(staged_paths == {row["path"] for row in rows if row.get("staged") is True},
         "staged path subset differs from manifest flags")
    need(set(git(checkout, "diff", "--cached", "--name-only").splitlines()) == staged_paths,
         "Git index paths differ from staged manifest")
    verified = []
    for row in rows:
        rel = row["path"]
        source, target = ROOT / rel, checkout / rel
        source_raw, target_raw = source.read_bytes(), target.read_bytes()
        need(len(source_raw) == len(target_raw) == row["bytes"], f"byte count mismatch: {rel}")
        need(hashlib.sha256(source_raw).hexdigest() == hashlib.sha256(target_raw).hexdigest() == row["sha256"],
             f"working-tree copy mismatch: {rel}")
        if row.get("staged") is True:
            blob = git(checkout, "hash-object", rel)
            need(blob == row["blob_sha"], f"Git blob mismatch: {rel}")
            index_raw = subprocess.check_output(["git", "show", f":{rel}"], cwd=checkout)
            need(index_raw == source_raw, f"index bytes mismatch: {rel}")
        else:
            head_raw = subprocess.check_output(["git", "show", f"HEAD:{rel}"], cwd=checkout)
            need(head_raw == source_raw, f"unchanged tracked file differs from HEAD: {rel}")
        verified.append({"path": rel, "sha256": row["sha256"], "bytes": row["bytes"],
                         "staged_bytes_match": True})
    result = {"schema": "stvg-soft-ep-public-local-verify-v1", "status": "pass",
              "files": verified, "file_count": len(verified),
              "staged_paths": sorted(staged_paths), "staged_file_count": len(staged_paths),
              "bytes": sum(x["bytes"] for x in verified),
              "commit_performed": False, "push_performed": False}
    (BASE / "PUBLIC_LOCAL_VERIFY.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                                                   encoding="utf-8")
    print(json.dumps({"status": "pass", "file_count": len(verified),
                      "bytes": result["bytes"], "commit_performed": False,
                      "push_performed": False}, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    stage_parser = sub.add_parser("stage", help="copy allowlisted files and add them to the local index")
    stage_parser.add_argument("--checkout", type=Path, default=DEFAULT_CHECKOUT)
    stage_parser.add_argument("--branch", default=None, help="must equal the already checked-out branch")
    stage_parser.add_argument("--replace", action="store_true")
    verify_parser = sub.add_parser("verify", help="verify local working-tree and index bytes only")
    verify_parser.add_argument("--checkout", type=Path, default=DEFAULT_CHECKOUT)
    args = parser.parse_args()
    if args.command == "stage":
        stage(args.checkout, args.branch, args.replace)
    else:
        verify_stage(args.checkout)


if __name__ == "__main__":
    main()
