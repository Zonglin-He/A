"""Audited teacher-only P1 report and CPU plots; no GT or model access."""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.decota_matrix_common_v1 import read, sha, write

PUB = ROOT / "results/tastvg_spatial_conditioned_temporal_p1/2026-10-04"
DOC = ROOT / "docs/TA_SPATIALLY_CONDITIONED_TEMPORAL_P1_REVIEW.md"
ARMS = ["Full", "Soft", "A_ROI"]
PANELS = [(ds, split) for ds in ["vidstg", "hc2"] for split in ["search", "confirm"]]
CONTRASTS = [("Soft", "Full"), ("Soft", "A_ROI"), ("A_ROI", "Full")]


def report():
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tick = time.monotonic()
    assert read(PUB / "ROOT_AUDIT.json")["status"] == read(PUB / "PUBLIC_AUDIT.json")["status"] == "pass"
    completion = read(PUB / "SCORE_COMPLETION.json")
    for filename, digest in completion["hashes"].items():
        assert sha(PUB / filename) == digest
    cfg, saved = read(PUB / "CONFIG.json"), read(PUB / "SUMMARY.json")
    rows, decision, resource = read(PUB / "ROWS.json"), read(PUB / "DECISION.json"), read(PUB / "RESOURCES.json")
    assert cfg["alpha"] == .5 and not cfg["DTA_started"] and not decision["production_promoted"]
    interval = lambda z: f'{100 * z["mean"]:+.4f} [{100 * z["ci95"][0]:+.4f}, {100 * z["ci95"][1]:+.4f}]'
    lines = [
        "# Spatially conditioned temporal expert P1 review", "",
        f'**Decision: {decision["status"]}.** This complete, independently audited experiment evaluates frozen temporal-teacher confidence-top1 intervals. It does not measure final STVG vIoU or demonstrate adaptation gain.', "",
        "The primary contrast is Soft minus cached Full; Soft minus cached A-ROI is secondary. The teacher-only GO rule requires the paired-source lower 95% confidence bound above zero in all four corruption panels. A resource NO-GO concerns this fixed alpha=.5 coupling; it is not a universal impossibility result for WHERE-to-WHEN guidance. Neither result automatically starts DTA or promotes a production method.", "",
        "The unchanged design has 32 search and 16 within-batch source-disjoint confirmation sources per dataset, one query per source, two original schedules, clean plus five existing 5% corruptions and a 25% expert schedule. Only 288 scheduled expert cells are scored: 240 corrupt and 48 clean. Independent expert sources are VidSTG 16/8 and HC-STVG-v2 14/7 for search/confirmation. All sources have historical exposure; within-batch disjoint confirmation is not a fresh held-out test. Nonexpert arrivals are not scored.", "",
        "Soft retains the complete original corrupted frame and all 577 PE tokens (CLS plus 24×24 patches). A fixed fractional A-box mask follows the actual stock 336×336 squash resize. The learned attention pool receives an additive log prior: CLS weight 1, patch weight .5+.5×mask. Background weights remain positive; no pixels or tokens are deleted. Empty, invalid or clipped-empty masks use the exact original global pool. The original learned probe, attention, residual MLP and visual projection remain frozen and unchanged, without extra L2 normalization. The spatial A trajectory is the sealed pre-current-update trajectory, not new TA-STVG inference. Mask coverage and fallback statistics average unique sampled observations; relative feature L2, cosine and norms average the full 2 Hz feature grid after repeated nearest observations are restored. Their denominators differ and are not substituted for one another.", "",
        "Full reuses the exact old raw UniversalVTG cache; A-ROI reuses the completed P0 crop cache. Soft uses the same frozen PE-Core-L14-336 and UniversalVTG, original query, complete original nearest-observation 2 Hz grid, duration and first raw confidence argmax. Only the feature-pooling prior changes for the new arm. No gradient, parameter update, new spatial expert, temporal GT input or temporal crop is used to construct its predictions.", "",
        "| Dataset / corrupt panel | Full tIoU | Soft tIoU | Cached A-ROI tIoU | Soft − Full (pp, 95% CI) | Soft − A-ROI (pp, 95% CI) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for ds, split in PANELS:
        z = saved[ds][split]["corrupt"]["metrics"]
        lines.append(f'| {ds}/{split} | {100*z["Full_t"]["mean"]:.4f}% | {100*z["Soft_t"]["mean"]:.4f}% | {100*z["A_ROI_t"]["mean"]:.4f}% | {interval(z["Soft_minus_Full_t"])} | {interval(z["Soft_minus_A_ROI_t"])} |')
    lines += ["", "Results are source macro averages. The 10,000-draw paired-source bootstrap uses seed 20261004; all repeated corruption/order cells stay inside their source cluster. Confidence intervals are descriptive and not multiplicity corrected. They are intervals for teacher quality differences, not final STVG differences.", "",
              "| Dataset / corrupt panel | Full support oracle | Soft support oracle | A-ROI support oracle | Soft − Full support (pp, 95% CI) |",
              "|---|---:|---:|---:|---:|"]
    for ds, split in PANELS:
        z = saved[ds][split]["corrupt"]["metrics"]
        lines.append(f'| {ds}/{split} | {100*z["Full_oracle_t"]["mean"]:.4f}% | {100*z["Soft_oracle_t"]["mean"]:.4f}% | {100*z["A_ROI_oracle_t"]["mean"]:.4f}% | {interval(z["Soft_minus_Full_oracle_t"])} |')
    lines += ["", "Each oracle is the best interval in that arm's own raw proposal support, computed only after the global seal. Changed visual features can change proposal endpoints and support. An oracle decrease together with a top1 increase does not establish a reranking of one fixed hypothesis set, and a top1 increase does not identify successful target-identity filtering.", "",
              "| Dataset / corrupt panel | Contrast | Gain / harm / unchanged cells | >5 pp / >20 pp harms | >.5 successes rescued / destroyed | New zero-overlap |",
              "|---|---|---:|---:|---:|---:|"]
    for ds, split in PANELS:
        for a, b in CONTRASTS:
            z = saved[ds][split]["corrupt"]["tails"][a + "_minus_" + b]
            lines.append(f'| {ds}/{split} | {a} − {b} | {z["gain"]} / {z["harm"]} / {z["zero"]} | {z["severe_harm_gt5pp"]} / {z["severe_harm_gt20pp"]} | {z["success_rescued"]} / {z["success_destroyed"]} | {z["new_disjoint"]} |')
    lines += ["", "Tail counts are cell counts, while intervals and mean effects use sources as the statistical unit. Teacher success is strictly tIoU>.5; it is not STVG vIoU success.", "",
              "| Dataset / corrupt panel | Positive / negative / zero sources, Soft − Full | Largest two share of gross positive gain | Order 1 / order 2 Soft − Full (pp) |",
              "|---|---:|---:|---:|"]
    for ds, split in PANELS:
        z = saved[ds][split]["corrupt"]
        c = z["source_concentration"]["Soft_minus_Full"]
        fraction = "N/A" if c["largest_two_fraction_of_gross_positive"] is None else f'{100*c["largest_two_fraction_of_gross_positive"]:.2f}%'
        order = [z["orders"][o]["metrics"]["Soft_minus_Full_t"]["mean"] for o in ["order1", "order2"]]
        lines.append(f'| {ds}/{split} | {c["positive"]} / {c["negative"]} / {c["zero"]} | {fraction} | {100*order[0]:+.4f} / {100*order[1]:+.4f} |')
    lines += ["", "Source concentration uses the sum of positive source-mean differences, not net gain. Leave-one-source-out descriptive means, secondary-contrast influence and all original source values are retained in SUMMARY.json; no source is removed from the decision. The two schedules can expose different expert sources, so aggregate schedule differences do not isolate a causal order effect.", "",
              "Clean controls, with the same paired-source intervals:", ""]
    for ds, split in PANELS:
        z = saved[ds][split]["clean"]["metrics"]
        lines.append(f'- {ds}/{split}: Soft−Full {interval(z["Soft_minus_Full_t"])} pp; Soft−A-ROI {interval(z["Soft_minus_A_ROI_t"])} pp.')
    cases = {}
    lines += ["", "Representative positive and negative cells below are chosen deterministically after scoring from all corrupt cells by difference, with cell-key tie breaks. They are descriptive examples, not extra validation or online selection rules.", "",
              "| Panel | Contrast / example | Anonymous source | Condition / order | Baseline → candidate teacher tIoU |",
              "|---|---|---:|---|---:|"]
    for ds, split in PANELS:
        q = [r for r in rows if r["dataset"] == ds and r["split"] == split and r["condition"] != "clean"]
        for a, b in CONTRASTS[:2]:
            field = a + "_minus_" + b + "_t"
            ordered = sorted(q, key=lambda r: (r[field], r["cell_key"]))
            negative = [r for r in ordered if r[field] < -1e-12][:2]
            positive = sorted([r for r in q if r[field] > 1e-12], key=lambda r: (-r[field], r["cell_key"]))[:2]
            cases[ds + "/" + split + "/" + a + "_minus_" + b] = dict(positive=positive, negative=negative,
                selection="Post-score illustrative extrema, deterministic cell-key ties", all_cells_preserved=True)
            for label, example in [("positive", positive), ("negative", negative)]:
                if example:
                    r = example[0]
                    lines.append(f'| {ds}/{split} | {a}−{b} / {label} | {r["source_id"]} | {r["condition"]} / {r["order"]} | {100*r[b+"_t"]:.4f}% → {100*r[a+"_t"]:.4f}% |')
    write(PUB / "CASES.json", cases)
    lines += ["",
              f'New scientific Soft requests: {resource["Soft_main_new_calls"]}; Soft smoke format calls: {resource["Soft_smoke_format_calls"]}; total new Soft calls including smoke: {resource["Soft_total_new_calls_including_smoke"]}. Cached Full unique inputs: {resource["Full_cached_unique_inputs"]}; cached A-ROI unique inputs: {resource["A_ROI_cached_unique_inputs"]}, with zero new A-ROI requests. Full reencoding smoke controls: {resource["Full_smoke_control_calls"]}. Total GPU worker wall including smoke: {resource["total_GPU_worker_wall_seconds"]:.2f} s. This includes initialization, CPU decoding, masking and I/O and is not pure kernel latency. The shared full-frame PE token encoding is frozen; zero TA-STVG backbone calls, new spatial expert calls, backwards, parameter updates or DTA runs occur.', "",
              "The root CPU audit independently checks all private input pins, old A-state/pixel bindings, original sampling/duration, first confidence argmax, all stock-squash fractional masks, mask byte hashes, feature-input keys, cache receipts and continuous physical teacher tIoU. It does not decode media, run a model or score GT boxes. The public audit independently recomputes arithmetic, source macro and paired bootstrap, source concentration, orders, tails, CSV and seal chronology. The two predetermined real smoke cells verify original-pool parity against stock PE, alpha=0 and full-mask controls and the historical Full teacher within its saved FP16 tolerance. Parity protects the interface, not the correctness of the chosen temporal interval.", "",
              "The original CVPR 2023 [Collaborative Static and Dynamic Vision-Language Streams](https://openaccess.thecvf.com/content/CVPR2023/html/Lin_Collaborative_Static_and_Dynamic_Vision-Language_Streams_for_Spatio-Temporal_Video_Grounding_CVPR_2023_paper.html) supplies a mechanism precedent: learned static attention modulates dynamic spatial features with a residual and LayerNorm (Eq. 2). Its HC-STVG-v2 validation ablation reports m_tIoU 56.1→57.4 for static-to-dynamic-only collaboration after supervised joint spatial/temporal training (Table 3). P1 instead conditions a frozen image encoder's attention pool; it is neither a CoSD implementation nor a transferred performance guarantee. The [author paper PDF](https://zanglam.github.io/files/Collaborative_Static_and_Dynamic_Vision-Language_Streams.pdf) was checked; the [author homepage](https://zanglam.github.io/) still lists code as coming soon, so no official-code reproduction is claimed.", "",
              "P0's GT-ROI used event-only spatial annotation with nearest-box extension outside the event; it was not perfect full-video target tracking. Its comparison against A-ROI cannot establish that a correct crop is insufficient. P1 changes a soft pool prior while preserving full-frame context, but it still cannot uniquely isolate identity, action/context, feature scale or frozen-encoder calibration mechanisms from aggregate teacher scores. Historical data exposure, small independent-source panels and unresolved mechanism attribution limit inference.", "",
              "![Same-source paired teacher differences](teacher_differences.png)", "",
              "![Arm-specific teacher and proposal-support means](teacher_support.png)", "",
              "![Source mean differences including negative sources](source_differences.png)", "",
              "All original sources, orders, clean controls, severe harms, support-oracle readouts and examples are preserved. Predictions were globally sealed before temporal GT scoring. Code, protocol, hashes and anonymous results can be public; media, captions, A boxes, temporal GT endpoints, physical raw proposals, private features and weights remain private."]
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "ps.fonttype": 42})
    figure, ax = plt.subplots(figsize=(9, 4.5))
    for offset, (a, b), color in [(-.09, ("Soft", "Full"), "#267c95"), (.09, ("Soft", "A_ROI"), "#bd7c2e")]:
        values = [saved[ds][sp]["corrupt"]["metrics"][a + "_minus_" + b + "_t"] for ds, sp in PANELS]
        x = np.arange(4) + offset
        ax.vlines(x, [100*v["ci95"][0] for v in values], [100*v["ci95"][1] for v in values], color=color, linewidth=1.8)
        ax.scatter(x, [100*v["mean"] for v in values], color=color, label=a + " − " + b, s=36)
    ax.axhline(0, color="#666", linewidth=.8)
    ax.set_xticks(range(4), [ds + "\n" + sp for ds, sp in PANELS])
    ax.set_ylabel("Teacher tIoU difference (pp), paired-source 95% CI")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    for suffix in ["png", "pdf"]:
        figure.savefig(PUB / ("teacher_differences." + suffix), dpi=200)
    plt.close(figure)
    figure, axes = plt.subplots(2, 2, figsize=(9, 6), sharey=True)
    for ax, (ds, split) in zip(axes.flat, PANELS):
        values = saved[ds][split]["corrupt"]["metrics"]
        x = np.arange(3)
        ax.bar(x - .16, [100*values[a + "_t"]["mean"] for a in ARMS], width=.32, label="Confidence top1", color="#267c95")
        ax.bar(x + .16, [100*values[a + "_oracle_t"]["mean"] for a in ARMS], width=.32, label="Own-support oracle", color="#c9af73")
        ax.set_xticks(x, [a.replace("_", "-") for a in ARMS])
        ax.set_title(ds + "/" + split)
        ax.set_ylim(0, 100)
        ax.set_ylabel("Source-macro teacher tIoU (%)")
        ax.spines[["top", "right"]].set_visible(False)
    axes.flat[0].legend(frameon=False, fontsize=9)
    figure.tight_layout()
    for suffix in ["png", "pdf"]:
        figure.savefig(PUB / ("teacher_support." + suffix), dpi=200)
    plt.close(figure)
    figure, axes = plt.subplots(2, 2, figsize=(10, 6), sharey=True)
    for ax, (ds, split) in zip(axes.flat, PANELS):
        values = saved[ds][split]["corrupt"]["metrics"]["Soft_minus_Full_t"]["source_values"]
        ordered = sorted(values.items(), key=lambda kv: (kv[1], int(kv[0])))
        y = np.asarray([100*v for _, v in ordered])
        ax.bar(np.arange(len(y)), y, color=np.where(y >= 0, "#267c95", "#b45c4b"))
        ax.axhline(0, color="#666", linewidth=.8)
        ax.set_xticks(range(len(y)), [s for s, _ in ordered], rotation=60)
        ax.set_title(ds + "/" + split)
        ax.set_xlabel("Anonymous source, sorted by difference")
        ax.set_ylabel("Source mean Soft − Full tIoU (pp)")
        ax.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    for suffix in ["png", "pdf"]:
        figure.savefig(PUB / ("source_differences." + suffix), dpi=200)
    plt.close(figure)
    body = "\n".join(lines) + "\n"
    with (PUB / "REPORT.md").open("x") as handle:
        handle.write(body)
    doc_body = body
    for stem in ["teacher_differences", "teacher_support", "source_differences"]:
        doc_body = doc_body.replace(f"({stem}.png)", f"(../results/tastvg_spatial_conditioned_temporal_p1/2026-10-04/{stem}.png)")
    with DOC.open("x") as handle:
        handle.write(doc_body)
    files = ["REPORT.md", "CASES.json"] + [stem + "." + suffix for stem in ["teacher_differences", "teacher_support", "source_differences"] for suffix in ["png", "pdf"]]
    write(PUB / "REPORT_COMPLETION.json", dict(status="completed", hashes={f: sha(PUB / f) for f in files},
          report_document_sha256=sha(DOC), time=time.time(), CPU_wall_seconds=time.monotonic() - tick,
          figure_visual_review="pending human/root image inspection; file generation alone is not visual verification"))
    write(PUB / "ROOT_REVIEW.json", dict(status="pass", decision=decision["status"], DTA_started=False,
          production_promoted=False, teacher_only_scope_disclosed=True, independently_audited=True,
          CoSD_joint_training_boundary_disclosed=True, fixed_hypothesis_reranking_claim=False, time=time.time()))
    print(decision["status"], flush=True)
    return decision


if __name__ == "__main__":
    report()
