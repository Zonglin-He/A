"""Produce paper figures from anonymous numerical evidence only."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


OUT = Path(__file__).resolve().parent / "results"
COHORTS = ("validation", "extension")
COLORS = ("#295D92", "#B96436")
PLATFORMS = ("waymo", "drone", "quad", "macro")


def rows(file):
    with (OUT / file).open(newline="") as stream:
        return list(csv.DictReader(stream))


def head(ax, p, title):
    ax.set_title(f"{p}  {title}", loc="left", pad=12, weight="bold", fontsize=10.5)


def changes(ax):
    x = np.arange(len(PLATFORMS))
    for i, cohort in enumerate(COHORTS):
        data = {r["platform"]: r for r in rows(f"{cohort}_pairs.csv")
                if r["arm"] == "height_only" and r["reference"] == "baseline"}
        ci = json.loads((OUT / f"{cohort}_evaluation.json").read_text())["recording_bootstrap_95pp"]
        mean = np.array([float(data[p]["delta_acc50_pp"]) for p in PLATFORMS])
        bounds = np.array([ci[f"{p}/height_only-baseline"]["acc50"] for p in PLATFORMS])
        xpos = x + (i - .5) * .29
        ax.bar(xpos, mean, .25, color=COLORS[i], label=("Validation" if i == 0 else "Recording-disjoint"))
        for xp, (lo, hi) in zip(xpos, bounds, strict=True):
            ax.plot((xp, xp), (lo, hi), color="#303A43", lw=1)
            ax.hlines((lo, hi), xp - .045, xp + .045, color="#303A43", lw=1)
    ax.set_xticks(x, ("Waymo", "Drone", "Quad", "Macro"))
    ax.axhline(0, color="#66717D", lw=.7)
    ax.set_ylabel("Acc@50 change vs. frozen (pp)")
    ax.legend(loc="best", frameon=False, fontsize=8)
    head(ax, "A", "Localization after roof correction")


def controls(ax):
    arms = ("prior_only", "points_only", "rays_only", "height_only", "soft_search", "swapped_prior", "constant_prior", "no_prior")
    labels = ("Height prior only", "Prior + returns", "Prior + entries", "All evidence",
              "Soft height search", "Shuffled prior", "Other cohort median", "No height prior")
    medians = {r["cohort"]: float(r["acc50"]) for r in rows("constant_prior_metrics.csv") if r["platform"] == "macro"}
    y = np.arange(len(arms))
    for i, cohort in enumerate(COHORTS):
        data = {r["arm"]: float(r["acc50"]) for r in rows(f"{cohort}_metrics.csv") if r["platform"] == "macro"}
        data["constant_prior"] = medians[cohort]
        dx = np.array([data[a] - data["baseline"] for a in arms])
        yy = y + (i - .5) * .26
        ax.hlines(yy, 0, dx, color=COLORS[i], lw=1.4, alpha=.65)
        ax.scatter(dx, yy, color=COLORS[i], s=14, zorder=3)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.axhline(3, color="#D9C56A", lw=8, alpha=.23, zorder=0)
    ax.axvline(0, color="#66717D", lw=.7)
    ax.set_xlabel("Macro Acc@50 change vs. frozen (pp)")
    head(ax, "B", "Fixed-input height controls")


def evidence(ax):
    table = {r["arm"]: r for r in rows("validation_geometry.csv") if r["platform"] == "pooled"}
    groups = ("prior_only", "points_only", "soft_search", "rays25pct", "rays50pct", "height_only")
    labels = ("Height prior only", "Prior + returns", "Soft height search", "25% of entries", "50% of entries", "All entries")
    y = np.arange(len(groups))
    for i, key in enumerate(("selection", "audit")):
        values = [1000 * int(table[a][f"{key}_changed"]) / int(table[a][f"affected_{key}"]) for a in groups]
        yy = y + (i - .5) * .26
        ax.scatter(values, yy, color=COLORS[i], marker=("o" if i == 0 else "s"), s=14,
                   label=("Selection rays" if i == 0 else "Audit rays"), zorder=3)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Changed ray outcomes per 1,000 rays")
    ax.set_xlim(left=-.7)
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    head(ax, "C", "Observable geometry in Validation")


def example(ax):
    examples = rows("example_cases.csv")
    categories = ("rescued", "harmed", "ray_veto", "oracle_still_fails")
    labels = ("Rescue", "Harm", "Entry veto", "Oracle fails")
    key_y = np.arange(len(categories))
    lim = []
    for y, case in zip(key_y, categories, strict=True):
        row, = [r for r in examples if r["cohort"] == "validation" and r["case"] == case and r["target"] == "waymo"]
        lower, final, target = [-float(row[k]) for k in ("slack_m", "trim_m", "top_error_old_m")]
        ax.hlines(y, lower, 0., lw=5, color="#BACBDD", zorder=1)
        ax.scatter(final, y, color=COLORS[0], s=24, zorder=3)
        ax.scatter(target, y, marker="x", color=COLORS[1], s=29, zorder=4)
        lim.extend([lower, target])
        ax.text(1.035, y, f"{float(row['before']):.3f} → {float(row['after']):.3f}", va="center", ha="left",
                fontsize=8, transform=ax.get_yaxis_transform(), clip_on=False)
    ax.axvline(0., color="#66717D", lw=.7)
    ax.set_xlim(min(lim) - .06, max(lim + [.07]) + .05)
    ax.set_yticks(key_y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Roof relative to original top (m)")
    ax.legend(handles=[Line2D([0], [0], color="#BACBDD", lw=4, label="Feasible interval"),
                       Line2D([0], [0], lw=0, marker="o", color=COLORS[0], label="Output", markersize=4),
                       Line2D([0], [0], lw=0, marker="x", color=COLORS[1], label="Annotation", markersize=5)],
              frameon=False, fontsize=7.5, loc="lower center", bbox_to_anchor=(.50, -0.47), ncol=3)
    head(ax, "D", "Example roof intervals and 3D IoU")


def run():
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#677480", "axes.labelcolor": "#28303A", "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 6.4), layout="constrained")
    for draw, ax in zip((changes, controls, evidence, example), axes.ravel(), strict=True):
        draw(ax)
    fig.savefig(OUT / "paper_figure.pdf", bbox_inches="tight")
    fig.savefig(OUT / "paper_figure.png", dpi=280, bbox_inches="tight")


if __name__ == "__main__":
    run()
