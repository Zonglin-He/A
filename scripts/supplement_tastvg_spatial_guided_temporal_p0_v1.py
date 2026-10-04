"""Post-score report context from sealed anonymous scalars; no new selections.

Source concentration and source deletion are descriptive, not preregistered
decision rules. Neither changes the original DECISION or scientific code pins.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUB = ROOT / "results/tastvg_spatial_guided_temporal_p0/2026-10-04"
BASE = ROOT / "artifacts/tastvg_spatial_guided_temporal_p0_v1"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    s = json.loads((PUB / "SUMMARY.json").read_text())
    audit = json.loads((PUB / "ROOT_AUDIT.json").read_text())
    assert audit["status"] == "pass"
    d = json.loads((PUB / "DECISION.json").read_text())
    cases = json.loads((PUB / "CASES.json").read_text())
    supplement = {
        "status": "descriptive_post_score_report_context",
        "scientific_decision_unchanged": d["status"],
        "inputs": {f: digest(PUB / f) for f in [
            "SUMMARY.json", "DECISION.json", "CASES.json", "ROOT_AUDIT.json",
            "GT_ORACLE_SCOPE_NOTE.json"]},
        "implementation": {str(Path(__file__).relative_to(ROOT)): digest(Path(__file__))},
        "concentration_denominator": "Sum of positive source-mean differences only, not net mean",
        "scope": "No GT, media, weights, models, new predictions, new CI, tuning or subgroup selection",
        "panels": {},
    }
    lines = [
        "", "<!-- P0_POST_SCORE_CONTEXT -->", "",
        "The decision is a resource/use decision about this fixed crop, **not a demonstrated impossibility of WHERE-to-WHEN guidance**. All four primary A-ROI differences have paired intervals crossing zero; the current implementation therefore lacks the agreed cross-panel evidence to become a DTA teacher. These small, historically exposed expert-source panels do not establish equivalence to Full, either.",
        "",
        "Post-score source heterogeneity is reported below without changing any selection, threshold, crop or decision. Gross-positive concentration divides the two largest positive source-mean changes by the sum of all positive source-mean changes; it must not be described as a fraction of net gain.",
        "",
        "| Corrupt panel | Positive / negative / zero sources | Largest two share of gross positive gain | A-ROI−Full order1 / order2 (pp) |",
        "|---|---:|---:|---:|",
    ]
    for ds in ["vidstg", "hc2"]:
        for sp in ["search", "confirm"]:
            z = s[ds][sp]["corrupt"]
            v = z["metrics"]["A_ROI_minus_Full_t"]["source_values"]
            n = len(v)
            positive = sorted([(k, x) for k, x in v.items() if x > 0], key=lambda kx: (-kx[1], int(kx[0])))
            gross = sum(x for _, x in positive)
            frac = sum(x for _, x in positive[:2]) / gross if gross else None
            loo = {k: sum(x for j, x in v.items() if j != k) / (n - 1) for k in v}
            assert abs(sum(v.values()) / n - z["metrics"]["A_ROI_minus_Full_t"]["mean"]) < 1e-12
            orders = {o: q["metrics"]["A_ROI_minus_Full_t"] for o, q in z["orders"].items()}
            counts = [sum(x > 0 for x in v.values()), sum(x < 0 for x in v.values()), sum(x == 0 for x in v.values())]
            support = {a: z["metrics"][a + "_oracle_t"]["mean"] for a in ["Full", "A_ROI", "GT_ROI"]}
            supplement["panels"][ds + "/" + sp] = {
                "source_counts_positive_negative_zero": counts,
                "source_values": v,
                "largest_two_positive_sources": positive[:2],
                "gross_positive_sum": gross,
                "largest_two_fraction_of_gross_positive": frac,
                "leave_one_source_out_means": loo,
                "order_differences": orders,
                "support_oracle_means": support,
                "A_full_fallback_fraction": z["metrics"]["A_full_fallback_fraction"]["mean"],
            }
            lines.append(f'| {ds}/{sp} | {counts[0]} / {counts[1]} / {counts[2]} | {100 * frac:.2f}% | {100 * orders["order1"]["mean"]:+.4f} / {100 * orders["order2"]["mean"]:+.4f} |')
    h = supplement["panels"]["hc2/confirm"]
    max_source = h["largest_two_positive_sources"][0][0]
    lines += [
        "",
        f'HC confirmation contains only seven independent expert sources: three improve and four worsen. Sources 43 and 34 account for {100 * h["largest_two_fraction_of_gross_positive"]:.2f}% of **gross positive source gain**. Removing the largest positive source ({max_source}) leaves a descriptive mean of {100 * h["leave_one_source_out_means"][max_source]:+.4f} pp. No source is removed from the primary result. This is concentration/influence evidence, not a new validation set or a fitted rule.',
        "",
        "Orders use their original expert schedules and may expose different source populations; differences between their aggregate means do not isolate a causal order effect.",
        "",
        "The proposal-support oracle means (Full / A-ROI / GT-ROI) are:", "",
    ]
    for panel, z in supplement["panels"].items():
        q = z["support_oracle_means"]
        lines.append(f'- {panel}: {100 * q["Full"]:.4f}% / {100 * q["A_ROI"]:.4f}% / {100 * q["GT_ROI"]:.4f}%.')
    lines += [
        "",
        "These descriptive means do not show a uniform increase of support capacity. In HC confirmation the A-ROI support mean is lower even though its top1 mean is higher. This is compatible with a change in choice within the support; it does not identify the causal reason or prove that identity filtering succeeded.",
        "",
        "The deterministic best/worst confirmation examples are retained in CASES.json. Values below are teacher tIoU percentages, not final STVG vIoU:",
        "",
        "| Panel / arm / example | Anonymous source | Condition / order | Full → crop teacher tIoU |",
        "|---|---:|---|---:|",
    ]
    for ds, a in [("vidstg", "A_ROI"), ("hc2", "A_ROI"), ("hc2", "GT_ROI")]:
        for label in ["best", "worst"]:
            r = cases[ds + "/confirm/" + a][label][-1 if label == "best" else 0]
            lines.append(f'| {ds} / {a} / {label} | {r["source_id"]} | {r["condition"]} / {r["order"]} | {100 * r["Full_t"]:.4f}% → {100 * r[a + "_t"]:.4f}% |')
    lines += [
        "",
        "**GT timing limitation:** the tracked-to-extended crop-motion change depends on where spatial annotations exist, which is event-dependent. Nearest-box extension prevents a direct GT event-window/black-frame cue, but indirect temporal cues may remain. This was documented in GT_ORACLE_SCOPE_NOTE.json before temporal scoring. GT-ROI is consequently not an independent spatial-only oracle. An inability of this extended-box input to improve teacher top1 cannot distinguish missing full-clip target tracking from context/scale/motion shift, or frozen-expert input mismatch.",
        "",
        "All A boxes were valid under the fixed rule: the full-frame fallback fraction is zero. Original corrupted pixels were independently verified for every scheduled cell; 460 predetermined cropped frames were reconstructed with a separate pad/slice implementation. These checks protect the measured intervention from decode/crop errors and do not turn its uncertain effect into a mechanistic conclusion.",
        "",
        "ROI inputs are re-encoded through the same frozen visual encoder and temporal expert, so their raw proposal support can change with the view. This is an expert-input intervention, not a new spatial plausibility scorer on a fixed temporal candidate set. Full retains the original saved proposals exactly.",
        "",
        "Run `scripts/supplement_tastvg_spatial_guided_temporal_p0_v1.py` after the original report generator to reproduce this explicitly post-score context. The original executed scientific bindings and decision are unchanged.",
    ]
    out = PUB / "REPORT_SUPPLEMENT.json"
    out.write_text(json.dumps(supplement, ensure_ascii=False, indent=2) + "\n")
    report = ROOT / "docs/TA_SPATIALLY_GUIDED_TEMPORAL_P0_REVIEW.md"
    draft = report.read_text()
    prefix = draft.split("<!-- P0_POST_SCORE_CONTEXT -->")[0].rstrip() + "\n"
    for before, after in {
        "Only288": "Only 288", "240corrupt/48clean": "240 corrupt / 48 clean",
        "Vid16/8": "Vid 16/8", "HC14/7": "HC 14/7",
        "paired10000-source": "paired 10,000-source", "seed20261004": "seed 20261004",
        "Vid138/144": "Vid 138/144", "HC144/144": "HC 144/144",
        "ratio1.5": "ratio 1.5", "lower95CI": "lower 95% CI",
        "Full235": "Full 235", "plus2": "plus 2",
    }.items():
        prefix = prefix.replace(before, after)
    prefix = prefix.replace("Clean teacher differences:\n-", "Clean teacher differences:\n\n-")
    if BASE.exists():
        saved = BASE / "report_drafts/generated_before_supplement.md"
        saved.parent.mkdir(exist_ok=True)
        if not saved.exists():
            saved.write_text(prefix)
    report.write_text(prefix + "\n" + "\n".join(lines) + "\n")
    print("descriptive supplement complete; original scientific decision unchanged")


if __name__ == "__main__":
    run()
