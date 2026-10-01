"""Reproducible static sensitivity/confirmation figure from sealed summaries."""
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def run(folder):
    b = Path(folder)
    read = lambda p: json.loads(p.read_text())
    trials, selection = read(b/'TRIALS.json'), read(b/'SELECTION.json')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'svg.fonttype': 'none', 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1))
    series = [('all', '#315b8a', 'o', 'All arrivals'),
              ('nonexpert', '#b77932', 's', 'Nonexpert arrivals')]
    for ax, stage, key, title in zip(axes[:2], ['A', 'B'],
            ['lr', 'teacher_temperature'],
            ['(a) Learning rate; teacher temperature = 1',
             '(b) Teacher temperature; selected learning rate']):
        valid = sorted([t for t in trials if t['stage'] == stage and t['state'] == 'COMPLETE'],
                       key=lambda t: t['params'][key])
        for subset, color, marker, label in series:
            data = [read(b/t['result_dir']/'SUMMARY.json')['corruption'][subset]['metrics']['delta_m_vIoU'] for t in valid]
            x = [t['params'][key] for t in valid]
            y = [100*m['mean'] for m in data]
            ax.plot(x, y, marker=marker, ms=3, lw=1.3, color=color, label=label)
            if subset == 'all':
                ax.fill_between(x, [100*m['ci95'][0] for m in data],
                                [100*m['ci95'][1] for m in data], color=color, alpha=.12,
                                label='All-arrival 95% CI')
        ax.axvline(selection['stages'][stage]['params'][key], color='#555555',
                   ls=':', lw=1, label='Search-selected value')
        ax.set_xscale('log')
        ax.set_xlabel('Learning rate' if stage == 'A' else 'Teacher temperature')
        ax.set_title(title, fontsize=10, loc='left', pad=10)
        ax.set_ylabel('Corruption ΔvIoU vs. Frozen (pp)')
    ax = axes[2]
    arms = ['default', 'lr_only', 'selected']
    conf = read(b/'CONFIRMATION_SUMMARY.json')
    for subset, color, marker, label in series:
        for i, arm in enumerate(arms):
            if 'summary' not in conf[arm]:
                continue
            m = conf[arm]['summary']['corruption'][subset]['metrics']['delta_m_vIoU']
            mean, (lo, hi) = 100*m['mean'], [100*v for v in m['ci95']]
            ax.errorbar(i + (-.08 if subset == 'all' else .08), mean,
                        yerr=[[mean-lo], [hi-mean]], fmt=marker, ms=5, color=color,
                        capsize=3, lw=1.2)
    ax.set_xticks(range(3), ['Default', 'LR only', 'LR + temp.'])
    ax.set_xlim(-.45, 2.45)
    ax.set_ylabel('Corruption ΔvIoU vs. Frozen (pp)')
    ax.set_title('(c) Separate 16-source confirmation', fontsize=10, loc='left', pad=10)
    for ax in axes:
        ax.axhline(0, color='#777777', lw=.8, zorder=0)
        ax.grid(axis='y', color='#eeeeee', lw=.6)
        ax.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc='upper center', frameon=False,
               bbox_to_anchor=(.5, 1.01))
    fig.subplots_adjust(left=.065, right=.99, bottom=.17, top=.8, wspace=.31)
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(b/f'sensitivity.{ext}', dpi=180, facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    run(sys.argv[1])
