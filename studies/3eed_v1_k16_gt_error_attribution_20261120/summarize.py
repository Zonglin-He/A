"""Compose the complete casebook after the sealed numerical replay and visual review."""
import csv
import html
import json
from math import fsum
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PRIVATE = HERE / "review_PRIVATE"
IDENTITY = {"same_target": "同物体几何误差", "other_instance": "另一物体", "unresolved": "身份不明"}
COHORT = {"frozen_miss": "冻结漏选", "seed_harm": "seed 误改"}
COMPONENTS = ("xy", "z", "size", "yaw")
PART = {"xy": "水平中心", "z": "高度中心", "size": "长宽高", "yaw": "朝向"}
FROZEN = ("prior", "subject_joint", "reference")
SCORE = {"prior": "前端先验", "subject_joint": "主体分", "reference": "直接参照分"}


def load_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def signed(value):
    return f"{value:+.3f}"


def collect():
    labels = load_csv(RESULTS / "identity_review.csv")
    by_case = {r["case_id"]: r for r in labels}
    scores = load_csv(RESULTS / "paired_scores.csv")
    assert {r["case_id"] for r in scores} == set(by_case)
    assert len(scores) == len(labels) == len(by_case) == 28
    rows = []
    for s in scores:
        row = dict(by_case[s["case_id"]])
        for k, v in s.items():
            if k in ("case_id", "cohort", "platform"):
                row[k] = v
            elif v in ("True", "False", ""):
                row[k] = {"True": True, "False": False, "": None}[v]
            else:
                row[k] = float(v)
        row["largest_frozen_term"] = max(FROZEN, key=lambda t: row["gap_" + t])
        if row["identity"] == "same_target":
            row["largest_geometry_component"] = max(COMPONENTS, key=lambda t: row["geometry_" + t])
        rows.append(row)
    return rows


def summarize(rows):
    controls = ("independent", "reference_mismatch", "remove_prior", "remove_subject_unary",
                "remove_direct_relation", "prior_only", "unary_only", "own_text_only",
                "remove_neighbors", "remove_density")
    summary = {"gap_convention": "failed minus best-scored qualifying candidate", "cohorts": {}}
    for name in COHORT:
        group = [r for r in rows if r["cohort"] == name]
        same = [r for r in group if r["identity"] == "same_target"]
        recordings = set()
        for r in group:
            meta = json.loads((PRIVATE / (r["case_id"] + ".json")).read_bytes())["data"]
            recordings.add((r["platform"], meta["recording"]))
        terms = {}
        for col in ("gap_prior", "gap_subject_joint", "gap_reference", "gap_frozen", "front_gap_text",
                    "front_gap_density", "front_gap_neighbors", "gap_seed", "gap_positive_cost",
                    "gap_saved_negative_cost"):
            v = [r[col] for r in group]
            terms[col] = dict(positive=sum(x > 1e-7 for x in v), zero=sum(abs(x) <= 1e-7 for x in v),
                              negative=sum(x < -1e-7 for x in v), mean=fsum(v) / len(v))
        cells = Counter((r["four_has_hit"], r["identity"]) for r in group)
        iou_by_identity = {}
        for identity in IDENTITY:
            members = [r for r in group if r["identity"] == identity]
            if members:
                iou_by_identity[identity] = {
                    "qualifying": fsum(r["good_iou"] for r in members) / len(members),
                    "failing": fsum(r["bad_iou"] for r in members) / len(members)}
        summary["cohorts"][name] = dict(
            rows=len(group), platforms=dict(Counter(r["platform"] for r in group)), recordings=len(recordings),
            identity=dict(Counter(r["identity"] for r in group)),
            four_has_hit=sum(r["four_has_hit"] for r in group),
            original_four_by_identity=[dict(has_hit=h, identity=i, rows=n) for (h,i),n in sorted(cells.items())],
            actual_ref_rows=sum(r["has_reference"] for r in group),
            largest_term_favoring_failed=dict(Counter(r["largest_frozen_term"] for r in group
                                              if r["gap_"+r["largest_frozen_term"]] > 0)),
            term_gaps=terms, all_k16_reread_hits={c: sum(r[c + "_winner_iou"] > .5 for r in group) for c in controls},
            bound_prior_precludes_all_hits=[r["case_id"] for r in group if r["beyond_full_correction_range"]],
            bound_ref_precludes_all_hits_active=[r["case_id"] for r in group if r["has_reference"] and r["beyond_direct_reference_range"]],
            same_geometry_largest=dict(Counter(r["largest_geometry_component"] for r in same)),
            same_geometry_mean={k:fsum(r["geometry_" + k] for r in same)/len(same) for k in COMPONENTS},
            paired_iou_by_identity=iou_by_identity,
            good_is_frozen_for_all_seed_harms=all(r["original_is_best_seed_good"] for r in group) if name=="seed_harm" else None,
            bad_code_strict_subset_of_good=[r["case_id"] for r in group if r["codes_shared"]==r["bad_code_size"]<r["good_code_size"]],
            bad_exact_gt=[r["case_id"] for r in group if r["bad_exact_gt_code"]])
    return summary


def casebook(rows):
    lines = ["# 28 条源域失败记录", "",
             "达标要求有向三维 IoU 严格大于 0.50。每行蓝框为该组读出给分最高的达标框，橙框为模型实际选中的未达标框。分差均为橙减蓝，正值有利于橙框。", "",
             "原四槽从 v1 前端独立重建。「原四槽有好框」只说明四槽内有达标框。K16 的最高分达标框可以是另一个框。", ""]
    sections = []
    for key, title in COHORT.items():
        group = [r for r in rows if r["cohort"] == key]
        lines.extend([f"## {title}（{len(group)} 条）", ""])
        if key == "frozen_miss":
            lines += ["| 编号 | 平台 | 达标框 → 失败赢家 IoU | 身份核验 | 原四槽有好框 | 前端先验差 | 主体分差 | 参照分差 | 同物体时最大几何分量 | 原图与点云证据 |",
                      "|---|---|---:|---|---|---:|---:|---:|---|---|"]
        else:
            lines += ["| 编号 | 平台 | 原赢家 → seed 赢家 IoU | 身份核验 | 冻结总分差 | seed 总分差 | 免去的负类代价差 | 计入的正类代价差 | 同物体时最大几何分量 | 原图与点云证据 |",
                      "|---|---|---:|---|---:|---:|---:|---:|---|---|"]
        for r in group:
            name, identity = r["case_id"], IDENTITY[r["identity"]]
            pair = f'{r["good_iou"]:.4f} → {r["bad_iou"]:.4f}'
            gaps = [r["gap_" + k] for k in FROZEN]
            if key == "frozen_miss":
                values = ["是" if r["four_has_hit"] else "否", *(signed(x) for x in gaps)]
            else:
                fields = ("gap_frozen", "gap_seed", "gap_saved_negative_cost", "gap_positive_cost")
                values = [signed(r[f]) for f in fields]
            part = r.get("largest_geometry_component")
            geometry_part = f'{PART[part]} {signed(r["geometry_"+part])}' if part else "—"
            lines.append("| " + " | ".join([name,r["platform"],pair,identity,*values,geometry_part,r["observation_zh"]]) + " |")
            meta = json.loads((PRIVATE / (name + ".json")).read_bytes())
            point_note = f'失败框内 {r["bad_inside_points"]:.0f} 个点，其中 {r["bad_target_shared_points"]:.0f} 个同时在目标真值框中。'
            detail = f'先验差 {signed(gaps[0])}，主体分差 {signed(gaps[1])}，直接参照分差 {signed(gaps[2])}，冻结总分差 {signed(r["gap_frozen"])}，seed 分差 {signed(r["gap_seed"])}。'
            geometry_note = ""
            if r["identity"] == "same_target":
                geometry_note = "从失败框换入达标框分量的平均 IoU 贡献为" + "、".join(
                    PART[k] + " " + signed(r["geometry_"+k]) for k in COMPONENTS) + "。"
            sections.append(f'<section id="{name}"><h2>{name} · {title} · {identity}</h2>'
                            f'<p>{html.escape(meta["data"]["caption"])}</p><p><b>达标框 → 失败赢家 {pair}</b>　{html.escape(r["observation_zh"])}</p>'
                            f'<p>{detail} {point_note} {geometry_note}</p>'
                            f'<a href="plates/{name}.jpg"><img loading="lazy" alt="{name} 全景、候选投影与点云俯侧视图" src="plates/{name}.jpg"></a></section>')
        lines.append("")
    lines += ["同物体记录的中心、尺寸、朝向平均边际贡献与各反事实全池赢家见[配对分数](results/paired_scores.csv)。seed 总分差等于免去的负类代价差减去计入的正类代价差。符号保留三位小数，完整精度在 CSV 中。", ""]
    (HERE / "CASES_ZH.md").write_text("\n".join(lines))
    nav = " ".join(f'<a href="#{r["case_id"]}">{r["case_id"]}</a>' for r in rows)
    page = ('<!doctype html><html lang="zh"><meta charset="utf-8"><title>K16 逐例图册</title>'
            '<style>body{font:16px/1.65 system-ui;max-width:1450px;margin:2rem auto;padding:0 1rem;color:#243342}img{max-width:100%;height:auto}'
            'nav a{display:inline-block;padding:.2rem}a{color:#245789}section{margin:3rem 0}</style>'
            '<h1>K16 源域逐例图册</h1><p>绿为主体真值，蓝为该读出得分最高的达标框，橙为实际失败赢家，紫为附近的其他标注。'
            '分差以橙减蓝计算。图中回波颜色表示框内外的几何关系。点击图可查看原尺寸。</p>'
            f'<nav>{nav}</nav>' + "".join(sections) + "</html>")
    (PRIVATE / "index.html").write_text(page)


if __name__ == "__main__":
    rows = collect()
    write_json(RESULTS / "summary.json", summarize(rows))
    casebook(rows)
