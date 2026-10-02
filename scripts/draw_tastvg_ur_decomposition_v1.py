"""Paper-style figures from anonymous sealed diagnostic summaries."""
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def draw(directory):
    directory = Path(directory)
    summary = json.loads((directory / 'SUMMARY.json').read_text())
    writes = json.loads((directory / 'WRITE_ROWS.json').read_text())
    colors = dict(U='#3675AD', R='#D17A35', Specific='#7A5CAC')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    for group in ['corruption', 'clean']:
        data = summary[group]
        fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.6), gridspec_kw={'width_ratios': [1.55, 1.25, 1.]})
        roles = ['self', 'next', 'near', 'far', 'broader']
        for j, branch in enumerate(['U', 'R', 'Specific']):
            for i, role in enumerate(roles):
                m = data['roles'][role]['metrics']['delta_'+branch+'_v']
                mean = 100*m['mean']; lo, hi = 100*np.asarray(m['ci95'])
                axes[0].errorbar(i+(j-1)*.16, mean, yerr=[[max(0, mean-lo)], [max(0, hi-mean)]],
                    fmt=['o', 's', '^'][j], markersize=5.5, color=colors[branch],
                    capsize=2.5, linewidth=1.2, label=branch if i == 0 else None)
        for i, name in enumerate(['self_far', 'near_far', 'self_broader', 'near_broader']):
            m = data['contrasts']['metrics']['delta_Specific_'+name+'_v']
            mean = 100*m['mean']; lo, hi = 100*np.asarray(m['ci95'])
            axes[1].errorbar(i, mean, yerr=[[max(0, mean-lo)], [max(0, hi-mean)]],
                fmt='o', markersize=6, color=colors['Specific'], capsize=3, linewidth=1.4)
        cosines = [w['geometry']['cosine_U_R'] for w in writes if
                   (w['condition'] != 'clean' if group == 'corruption' else w['condition'] == group)
                   and w['geometry']['cosine_U_R'] is not None]
        axes[2].hist(cosines, bins=np.linspace(-1, 1, 17), color='#658B9B', edgecolor='white', linewidth=.8)
        axes[2].axvline(0, color='#929292', linestyle='--', linewidth=.8)
        axes[2].set_xlim(-1.05, 1.05); axes[2].set_xlabel(r'$\cos(g_U,g_R)$'); axes[2].set_ylabel('Donor cells')
        axes[0].set_xticks(range(5), ['Self', 'Next', 'Text-near', 'Text-far', 'All future'])
        axes[0].set_ylabel('Fixed-time dense vIoU gain (pp)')
        axes[0].legend(frameon=False, fontsize=8, ncol=3)
        axes[1].set_xticks(range(4), ['Self−far', 'Near−far', 'Self−all', 'Near−all'], rotation=15)
        axes[1].set_ylabel('Paired Specific gain difference (pp)')
        for ax in axes[:2]:
            ax.axhline(0, color='#929292', linestyle='--', linewidth=.8)
            ax.grid(axis='y', color='#E8E8E8', linewidth=.5)
        for ax, title in zip(axes, ['(a) Same-state write utility', '(b) Residual locality', '(c) Gradient agreement']):
            ax.set_title(title, loc='left', fontweight='bold', fontsize=10)
        fig.tight_layout(w_pad=1.9)
        for extension in ['png', 'pdf', 'svg']:
            fig.savefig(directory / (group+'_write_decomposition.'+extension), dpi=250, bbox_inches='tight')
        plt.close(fig)


if __name__ == '__main__':
    draw(sys.argv[1])
