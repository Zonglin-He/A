"""Render audited anonymous summaries only; no model, GT span or latent access."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from pathlib import Path
import json
import hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/tastvg_pr_accessibility/2026-10-03'
FIG = OUT / 'figures'
FITS = ['source_fit', 'target_search_fit']
COLORS = ['#44566B', '#168C88']
LABELS = ['Source fit', 'Target-search fit (supervised)']
ROLES = ['P_A', 'R_A', 'P_W', 'R_W']
ARMS = ['predicted', 'GT_anchor', 'GT_winner', 'GT_precision', 'GT_recall', 'GT_all']


def read(path):
    return json.loads(path.read_text())


def style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#A6AFB8')
    ax.grid(axis='y', color='#E6E9ED', linewidth=.65, zorder=0)
    ax.tick_params(length=3, color='#A6AFB8')
    ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(FIG / (name + '.png'), dpi=220, bbox_inches='tight', facecolor='white')
    fig.savefig(FIG / (name + '.pdf'), bbox_inches='tight', facecolor='white',
                metadata={'CreationDate': None, 'ModDate': None})
    plt.close(fig)


def main():
    assert read(OUT / 'ROOT_AUDIT.json')['status'] == 'pass'
    FIG.mkdir(exist_ok=True)
    data = {ds: read(OUT / ds / 'SUMMARY.json') for ds in ['vidstg', 'hc2']}
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.titlesize': 11, 'axes.labelsize': 10,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    names = {'vidstg': 'VidSTG', 'hc2': 'HC-STVG-v2'}
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 6.1), sharey=True)
    x = np.arange(4)
    for i, ds in enumerate(data):
        for j, panel in enumerate(['search_corrupt', 'confirm_corrupt']):
            ax = axes[i, j]
            z = data[ds][panel]
            for f, fit in enumerate(FITS):
                vals = [z['regression'][fit][r]['metrics']['mae'] for r in ROLES]
                mean = np.array([v['mean'] for v in vals])
                bounds = np.array([v['ci95'] for v in vals]).T
                xx = x + (f - .5) * .34
                ax.bar(xx, mean, .29, color=COLORS[f], alpha=.9, zorder=3, label=LABELS[f])
                ax.errorbar(xx, mean, np.maximum(0., np.vstack([mean - bounds[0], bounds[1] - mean])),
                            fmt='none', ecolor=COLORS[f], capsize=3, lw=1, zorder=4)
            ax.set_xticks(x, [r'$P_A$', r'$R_A$', r'$P_W$', r'$R_W$'])
            ax.set_ylim(0, .62)
            mode = 'Search: in-sample' if j == 0 else 'Confirmation: source-disjoint'
            ax.set_title(f"{names[ds]} | {mode}\n{z['sources']} sources, {z['cells']} corrupt expert cells", pad=9)
            if j == 0:
                ax.set_ylabel('Raw P/R mean absolute error ↓')
            style(ax)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='upper center', ncol=2,
               frameon=False, bbox_to_anchor=(.5, 1.015))
    fig.tight_layout(rect=(0, 0, 1, .965), h_pad=2.0, w_pad=1.9)
    save(fig, 'pr_fit_generalization')

    fig, axes = plt.subplots(2, 2, figsize=(11.8, 6.1), sharey=True)
    x = np.arange(6)
    for i, ds in enumerate(data):
        z = data[ds]['confirm_corrupt']
        for j, metric in enumerate(['auc', 'balanced_accuracy']):
            ax = axes[i, j]
            for f, fit in enumerate(FITS):
                vals = [z['decision'][fit][arm]['metrics'][metric] for arm in ARMS]
                mean = np.array([v['mean'] for v in vals])
                bounds = np.array([v['ci95'] for v in vals]).T
                xx = x + (f - .5) * .15
                ax.errorbar(xx, mean, np.maximum(0., np.vstack([mean - bounds[0], bounds[1] - mean])),
                            fmt='o' if f == 0 else 's', ms=5, color=COLORS[f],
                            capsize=2.5, lw=1.1, zorder=3, label=LABELS[f])
            ax.axhline(.5, color='#A9AFB6', ls='--', lw=.8)
            ax.set_ylim(-.055, 1.065)
            ax.set_xticks(x, ['Pred.', 'GT\nanchor', 'GT\nwinner', 'GT P', 'GT R', 'All GT'])
            ax.set_title(f"{names[ds]} | {'Ranking AUROC' if j == 0 else 'Balanced accuracy at ΔT > 0'}", pad=10)
            ax.set_ylabel('AUROC ↑' if j == 0 else 'Balanced accuracy ↑')
            style(ax)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='upper center', ncol=2,
               frameon=False, bbox_to_anchor=(.5, 1.015))
    fig.tight_layout(rect=(0, 0, 1, .965), h_pad=2.3, w_pad=2)
    save(fig, 'pr_oracle_ladder')

    # Paired changes use the published joint source bootstrap, not independent CI subtraction.
    fig, axes = plt.subplots(1, 2, figsize=(11.3, 3.5), sharex=True, sharey=True)
    for ax, ds in zip(axes, data):
        z = data[ds]['confirm_corrupt']['paired_target_minus_source_regression']
        vals = [z[r]['mae'] for r in ROLES]
        mean = np.array([v['mean'] for v in vals])
        bounds = np.array([v['ci95'] for v in vals]).T
        y = np.arange(4)[::-1]
        ax.errorbar(mean, y, xerr=np.maximum(0., np.vstack([mean - bounds[0], bounds[1] - mean])),
                    fmt='o', color=COLORS[1], ms=6, capsize=4, lw=1.3, zorder=3)
        ax.axvline(0, ls='--', lw=.9, color='#78838F')
        ax.axvspan(-.09, 0, color='#F0F6F5', zorder=0)
        ax.set_yticks(y, [r'$P_A$', r'$R_A$', r'$P_W$', r'$R_W$'])
        ax.set_xlim(-.09, .21)
        assert bounds[0].min() >= -.09 and bounds[1].max() <= .21, 'Do not truncate paired intervals'
        ax.set_ylim(-.5, 3.5)
        ax.set_xlabel('Target-search minus source-fit MAE (lower is better)')
        ax.set_title(f'{names[ds]} | Confirmation corruption', pad=12)
        style(ax)
    fig.tight_layout(w_pad=2.4)
    save(fig, 'pr_paired_confirmation_mae')
    files = {str(p.relative_to(OUT)): {'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
             'bytes': p.stat().st_size} for p in sorted(FIG.iterdir()) if p.suffix in ['.png', '.pdf']}
    manifest = {'status': 'rendered_from_audited_summaries', 'source_bootstrap_draws': 10000,
                'CI': '95% whole-source percentile; undefined draws excluded and reported in SUMMARY',
                'not_new_model_inference': True, 'files': files}
    (OUT / 'FIGURES.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'status': 'rendered', 'files': len(files)}))


if __name__ == '__main__':
    main()
