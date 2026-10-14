"""Correct shared-legend aliases; retain the original plotted results unchanged."""
import sys
from typing import Any
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.report_stvg_native_instance_v1 import plt,PdfPages,DATASET_LABELS,ARM_ORDER,ARM_COLORS,ARM_SHORT,FIGURE_DIR,_summary_entries,_anonymous_source,_load_scored_rows
PDF_PATH=FIGURE_DIR/'native_instance_summary_reviewed.pdf'
PNG_PATH=FIGURE_DIR/'native_instance_summary_reviewed.png'
def _make_figure(summary: dict[str, Any], indexed: dict[str, dict[str, Any]], cases: dict[str, dict[str, Any]]) -> None:
    summaries = _summary_entries(summary)
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 7.3), sharey=True)
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42})
    for ax, dataset in zip(axes, DATASET_LABELS):
        ids = [cid for cid, value in cases.items() if value["dataset"] == dataset]
        x = list(range(len(ARM_ORDER)))
        for i, arm in enumerate(ARM_ORDER):
            entry = summaries[(dataset, arm, "m_vIoU")]
            mean = entry.get("mean")
            if mean is not None:
                ax.bar(i, 100.0 * float(mean), width=0.7, color=ARM_COLORS[arm], alpha=0.72, zorder=1)
            for j, case_id in enumerate(sorted(ids, key=lambda cid: _anonymous_source(cid, dataset))):
                value = indexed[f"{case_id}\0{arm}"]["_metrics"]["m_vIoU"]
                if value is None:
                    continue
                offset = (j - (len(ids) - 1) / 2) * 0.105
                ax.scatter(
                    [i + offset],
                    [100.0 * value],
                    color=f"C{j}",
                    edgecolor="white",
                    linewidth=0.5,
                    s=30,
                    zorder=3,
                    label=f"HC{j+1:02d} / V{j+1:02d}" if i == 0 else None,
                )
            ax.text(i, 0.985, f"{entry['n_valid']}/4", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=7, color="#333333")
        ax.set_xticks(x, [ARM_SHORT[arm] for arm in ARM_ORDER], rotation=42, ha="right")
        ax.set_title(f"{DATASET_LABELS[dataset]} · four exposed sources")
        ax.set_ylabel("Mean full-video vIoU (%)")
        ax.grid(axis="y", alpha=0.2, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.99))
    fig.suptitle("Five primary arms and frozen/spatial attribution controls", y=0.94, fontsize=13)
    fig.text(
        0.5,
        0.015,
        "Bars show valid-source means; colored points show each anonymous source. n/4 is printed above every arm. Unknowns are retained in the report. No pooled domain estimate.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0.02, 0.06, 0.98, 0.90))
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG_PATH, dpi=180, bbox_inches="tight", metadata={"Title": "Native instance STVG summary"})
    with PdfPages(PDF_PATH, metadata={"Title": "Native instance STVG summary", "Author": "STVG research report"}) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

if __name__=='__main__':
    assert not PDF_PATH.exists() and not PNG_PATH.exists()
    rows,summary,indexed,cases=_load_scored_rows()
    _make_figure(summary,indexed,cases)
