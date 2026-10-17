#!/usr/bin/env python3
"""Maintain a read-only-source research ledger, not an experiment scheduler.

`snapshot` mechanically regenerates only the two named archive sidecars.
`check` and `verify-current` never write or import model/GPU packages.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs/RESEARCH_HISTORY.md"
INDEX = ROOT / "docs/RESEARCH_SOURCE_INDEX.md"
SNAPSHOT = ROOT / "docs/research_archive_sources.json"
SELF_GENERATED = {DOC, INDEX, SNAPSHOT}
CURRENT = ROOT / "artifacts/decota_full_case_iteration_v1/analysis/hcstvg1_test"
LINK_RE = re.compile(r"\]\((?:<([^>\n]+)>|([^\n)]+))\)")
JSON_NAMES = re.compile(r"summary|manifest|config|lock|analysis|decision|readout|complete|current_method", re.I)
VOLATILE_NAMES = {"progress.json", "queue.log", "CURRENT_STATUS.md"}
CONVERSATION_ATTACHMENTS = """
6489d32d-33fb-44d2-bf84-5db879918ea4
161b07c7-ad60-4994-9adf-f8c3c19e4ea2
7d600a25-079f-4e2b-b4dd-ac365f3f82d3
051f5f82-80c4-4d19-920a-2845a28771f9
01301aeb-888d-4d68-a21d-114b047965cd
4728ff60-9867-4585-b70e-0c56bf638848
6b97c7e2-1062-494f-8541-323b815a4787
2763feb6-3e90-44d6-af22-1f7b3b56f9e6
c63f693f-0664-4c2e-aa0c-555ef027bbb1
db626603-f896-4acf-9f90-ce20f6221244
5a229871-01f4-408f-856f-04087966bb39
c238c558-4287-4cb6-b918-65785cca693b
ee32532e-f0bc-4243-8196-e57382f6c902
0964b662-91dc-4eb4-906c-e90758b06e6f
d39a8994-86c2-4dba-9512-c7bc77a557c5
799fa6ae-5055-42c9-b977-3c8a1840b780
d4211857-2a19-4c73-b346-80f79f88ac87
aaddbfe4-3ec6-41bc-836d-8b3ae505641c
84dab3d0-6231-4047-a0d6-fca91825c00c
3e0c4efc-490f-4d15-a206-749899550e39
29f433c5-ad47-49dd-8400-0dbec68ba5a8
923a2ba9-b551-444f-95bf-f609b136e24a
bcae10be-84aa-4ed2-89a5-f3d313a6d32a
b7f4595e-880d-4039-bcaf-e39253c0fc3d
70305e8c-ca67-431d-bdaf-6dcf15d12aec
798c6658-6baf-41dc-816f-8393dd3d475e
d47ec2ed-d000-4c21-b71b-1dae30722a3f
33b7b53c-1d40-48b8-91bb-d1e2ef4b44c6
6152e53f-d3ba-4663-ac9f-4db7f9f4e8ab
b8c7e816-a1a9-4624-b15e-dab0b56e5299
e0a11b96-4fd3-4078-a297-38ffb219a084
2d78405b-28e4-47ae-8a2e-9d9e9eed7b35
c22b6585-8b8d-4420-8ff8-1ebbba917d7a
663ac475-c3e3-4c36-95e7-840ec0e9e00b
d5d69c5e-e86f-44f6-8250-2c74b76785ce
""".split()


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def link_targets(text: str):
    for match in LINK_RE.finditer(text):
        target = (match.group(1) or match.group(2)).strip()
        if target.startswith(("http://", "https://", "mailto:")):
            yield "external", target
        elif target.startswith("#"):
            yield "anchor", unquote(target[1:])
        else:
            target = unquote(target.split("#", 1)[0])
            target = re.sub(r":\d+$", "", target)
            yield "local", target


def source_kind(path: Path) -> str:
    name = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
    if "/.codex/attachments/" in name:
        return "user_attachment_not_execution_receipt"
    if "artifacts/decoder_v1/" in name:
        return "historical_mirror_name_hint"
    if any(s in name.lower() for s in ["failed", "superseded", "aborted", "pre_", "smoke", "dry"]):
        return "engineering_or_revision_name_hint"
    if name.startswith("protocols/"):
        return "protocol_not_execution_receipt"
    if path.suffix == ".md":
        return "report_or_method_document"
    if path.suffix == ".py":
        return "implementation"
    return "result_or_configuration"


def collect_sources():
    text = DOC.read_text(encoding="utf-8")
    cited = set()
    for kind, target in link_targets(text):
        if kind == "local":
            p = Path(target)
            p = p if p.is_absolute() else DOC.parent / p
            if p.is_file() and p not in SELF_GENERATED:
                cited.add(p)
    paths = set(cited)
    # Only attachments actually named in this conversation, not other tasks.
    for attachment_id in CONVERSATION_ATTACHMENTS:
        path = Path("/home/wwww/.codex/attachments") / attachment_id / "pasted-text.txt"
        if path.is_file():
            paths.add(path)
    for dirname in ["artifacts", "protocols", "methods"]:
        base = ROOT / dirname
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix == ".md" or (path.suffix == ".json" and JSON_NAMES.search(path.name)):
                paths.add(path)
    for p in (ROOT / "docs").glob("*.md"):
        if p not in SELF_GENERATED:
            paths.add(p)
    rows = []
    for p in sorted(paths):
        info = p.stat()
        # Large per-query files remain locatable without repeatedly hashing GBs.
        checksum = digest(p) if info.st_size <= 8 * 1024 * 1024 else None
        title = None
        if p.suffix in {".md", ".txt"}:
            with p.open(encoding="utf-8", errors="replace") as stream:
                for line in stream:
                    if line.strip():
                        title = line.strip().lstrip("# ")[:180]
                        break
        rows.append(dict(
            path=str(p), relative_path=str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else None,
            kind=source_kind(p), cited_in_main=p in cited, title=title,
            size_bytes=info.st_size, mtime=datetime.fromtimestamp(info.st_mtime).astimezone().isoformat(),
            sha256=checksum, hash_scope="entire_file" if checksum else "metadata_only_over_8MiB",
            volatile=p.name in VOLATILE_NAMES,
            verification="indexed_only_not_an_independent_validation",
        ))
    return rows


def verify_current():
    rows = json.loads((CURRENT / "rows.json").read_text())
    summary = json.loads((CURRENT / "summary.json").read_text())
    sources = {r["source"] for r in rows}
    assert len({r["key"] for r in rows}) == len(rows), "Duplicate query keys"
    assert len(rows) == summary["scored_queries"]
    assert len(sources) == summary["scored_sources"]
    result = {}
    errors = []
    for method in ["frozen", "temporal_only", "mymethod"]:
        result[method] = {}
        for metric in ["vIoU_corrected", "sIoU", "tIoU"]:
            by_source = defaultdict(list)
            values = []
            for r in rows:
                value = r["metrics"][method][metric]
                if value is not None:
                    values.append(value)
                    by_source[r["source"]].append(value)
            query_mean = statistics.fmean(values)
            source_mean = statistics.fmean(statistics.fmean(v) for v in by_source.values())
            expected = summary["groups"]["all"]["methods"][method][metric]
            errors += [abs(query_mean - expected["query_mean"]), abs(source_mean - expected["mean"])]
            result[method][metric] = dict(query_percent=100 * query_mean, source_percent=100 * source_mean)
    assert max(errors) < 1e-12, f"Aggregation mismatch: {max(errors)}"
    for r in rows:
        assert r["metrics"]["frozen"]["sIoU"] == r["metrics"]["temporal_only"]["sIoU"]
        assert r["metrics"]["temporal_only"]["tIoU"] == r["metrics"]["mymethod"]["tIoU"]
    doc = DOC.read_text(encoding="utf-8") if DOC.exists() else ""
    required = [
        f"{result[m]['vIoU_corrected']['query_percent']:.3f} | "
        f"{result[m]['vIoU_corrected']['source_percent']:.3f} | "
        f"{result[m]['sIoU']['source_percent']:.3f} | "
        f"{result[m]['tIoU']['source_percent']:.3f}"
        for m in result
    ]
    if doc:
        assert all(value in doc for value in required), "Main HC1 table does not match current raw rows"
    return dict(
        cohort="hcstvg1_test", queries=len(rows), sources=len(sources),
        nominal=summary["nominal_queries"], unavailable=len(summary["unavailable"]),
        maximum_aggregation_error=max(errors), metrics=result,
        verification_scope="Independent aggregation of saved per-query scores; not new inference or all-history rescoring",
    )


def make_snapshot():
    rows = collect_sources()
    now = datetime.now().astimezone().isoformat()
    verification = verify_current()
    snapshot = dict(
        schema_version=1, as_of=now, main_document=str(DOC), main_sha256=digest(DOC),
        source_count=len(rows), cited_source_count=sum(r["cited_in_main"] for r in rows),
        note="Index and hashes do not certify execution, correctness, independence, or scientific claims.",
        current_result_verification=verification, sources=rows,
        conversation_attachments_expected=len(CONVERSATION_ATTACHMENTS),
        conversation_attachments_found=sum(r["kind"] == "user_attachment_not_execution_receipt" for r in rows),
    )
    # Generated sidecars only; never alter reports, predictions, or the authored ledger.
    SNAPSHOT.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    groups = defaultdict(list)
    for row in rows:
        rel = row["relative_path"]
        group = "/".join(rel.split("/")[:2]) if rel else "本对话用户附件"
        groups[group].append(row)
    lines = [
        "# STVG 研究总档案：来源目录", "",
        f"生成时间：{now}。共 {len(rows)} 个来源文件；正文直接引用 {snapshot['cited_source_count']} 个。", "",
        "主入口：[研究总档案](</home/wwww/visual grounding/docs/RESEARCH_HISTORY.md>)。", "",
        "这是可检索索引，不是实验完成清单。文件名分类只是线索；旧镜像、smoke、失败启动和被替代版本不算独立重复。",
        "正文直接引用也不代表从原视频独立复算；以正文的核验范围为准。历史报告与保留数据留在原目录；经用户授权清理的旧原始缓存以清理收据和正文记录为准。", "",
        "本文件由 scripts/research_archive.py snapshot 生成。人工解释只维护主文，不在这里写另一份当前结论。", "",
    ]
    for group, values in sorted(groups.items()):
        lines += [f"## {group}", ""]
        for r in values:
            label = Path(r["path"]).name
            rel = r["relative_path"]
            if rel and len(rel.split("/")) > 3:
                label = "/".join(rel.split("/")[2:])
            mark = "正文引用" if r["cited_in_main"] else "索引收录"
            title = (r["title"] or "").replace("|", "\\|")
            lines.append(f"- [{label}](<{r['path']}>) — {mark}；{r['kind']}" + (f"；{title}" if title else ""))
        lines.append("")
    INDEX.write_text("\n".join(lines), encoding="utf-8")
    return dict(status="snapshot_written", files=[str(INDEX), str(SNAPSHOT)],
                sources=len(rows), cited=snapshot["cited_source_count"], current_verification=verification)


def check():
    errors, warnings = [], []
    counts = defaultdict(int)
    for file in [DOC, INDEX]:
        if not file.exists():
            errors.append(f"Missing {file}")
            continue
        text = file.read_text(encoding="utf-8")
        ids = set(re.findall(r'<a\s+id="([^"]+)"', text))
        ids |= {re.sub(r"[^\w\- ]", "", s.lower()).replace(" ", "-")
                for s in re.findall(r"^#{1,6} (.+)$", text, flags=re.M)}
        for kind, target in link_targets(text):
            counts[kind] += 1
            if kind == "anchor":
                if target not in ids:
                    errors.append(f"Unknown anchor in {file.name}: {target}")
            elif kind == "local":
                path = Path(target)
                path = path if path.is_absolute() else file.parent / path
                if not path.exists():
                    errors.append(f"Broken link in {file.name}: {target}")
    if SNAPSHOT.exists():
        snap = json.loads(SNAPSHOT.read_text())
        if snap["main_sha256"] != digest(DOC):
            warnings.append("Main document changed after snapshot; regenerate after reviewing edits.")
        for source in snap["sources"]:
            if not source["cited_in_main"] or source["volatile"] or not source["sha256"]:
                continue
            p = Path(source["path"])
            if not p.exists():
                errors.append(f"Cited source disappeared: {p}")
            elif digest(p) != source["sha256"]:
                warnings.append(f"Cited source changed; review affected claims: {p}")
    else:
        warnings.append("No source snapshot yet; run snapshot after the document review.")
    try:
        current = verify_current()
    except (AssertionError, OSError, ValueError, KeyError) as exc:
        errors.append(f"Current-result verification: {exc}")
        current = None
    output = dict(status="pass" if not errors else "fail", errors=errors,
                  warnings=warnings, link_counts=dict(counts), current_result=current)
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 1 if errors else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["snapshot", "check", "verify-current"])
    args = parser.parse_args()
    if args.command == "check":
        return check()
    result = make_snapshot() if args.command == "snapshot" else verify_current()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
