"""Create common-state reports only after actual independent root readback."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.decota_matrix_common_v1 import read, write, sha
from scripts.review_stvg_opd_common_state_postseal_v1 import BASE, REC, PUB, ARMS, require_seal


def run():
    require_seal()
    receipt = read(REC/'ROOT_READBACK.json'); assert receipt['status'] == 'pass'
    s = read(PUB/'INDEPENDENT_ROOT_STATISTICS.json'); assert s['status'] == 'pass'
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    destination = ROOT/'output/pdf/stvg_opd_optimization_geometry/common_state'
    destination.mkdir(parents=True, exist_ok=True)
    keys = ['hc2/cross_clean', 'hc2/same_corruption', 'vidstg/cross_clean', 'vidstg/same_corruption']
    assert set(k+'/excluding_development' for k in keys) <= set(s['settings'])
    labels = ['HC cross', 'HC physical shifts', 'Vid cross', 'Vid physical shifts']
    def statistic(key, field):
        return s['settings'][key+'/excluding_development']['statistics'][field]
    def draw(ax, field, scale, label, offset, color):
        for i, key in enumerate(keys):
            v = statistic(key, field)
            if v['mean'] is None:
                ax.text(0, i+offset, 'undefined', fontsize=8)
                continue
            lo, hi = np.array(v['ci95'])*scale
            mean = v['mean']*scale
            ax.plot([lo, hi], [i+offset]*2, color=color, linewidth=2)
            ax.plot(mean, i+offset, 'o', color=color, label=label if i == 0 else None)
    plt.rcParams.update({'font.size': 11, 'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), constrained_layout=True)
    for ax in axes:
        ax.set_yticks(range(4), labels); ax.invert_yaxis(); ax.axvline(0, color='.6', linewidth=1)
        ax.grid(axis='x', alpha=.2)
    draw(axes[0], 'Full_minus_Direct_v', 100, 'Full - Direct', -.12, '#2464a8')
    draw(axes[0], 'Full_minus_Direct_capped_v', 100, 'Full - constrained Direct', .12, '#bb5a26')
    axes[0].set_xlabel('Paired vIoU difference (percentage points)'); axes[0].legend(fontsize=9)
    axes[0].set_title('Same original Full initial state')
    for j, arm in enumerate(ARMS):
        draw(axes[1], arm+'_current_v', 100, arm, (j-1)*.18, ['#2464a8', '#5d8d44', '#bb5a26'][j])
    axes[1].set_xlabel('After - common Before (percentage points)'); axes[1].legend(fontsize=9)
    axes[1].set_title('Current-query change')
    fig.suptitle('Common-state control: primary non-development historical parents', fontsize=13)
    files = []
    for extension in ['pdf', 'png']:
        p = destination/('paired_effects.'+extension); assert not p.exists()
        fig.savefig(p, dpi=180); files.append(p)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), constrained_layout=True)
    for ax in axes:
        ax.set_yticks(range(4), labels); ax.invert_yaxis(); ax.grid(axis='x', alpha=.2)
    draw(axes[0], 'common_parameter_gradient_cosine', 1, 'Joint parameter gradient cosine', 0, '#2464a8')
    axes[0].set_xlim(-1.05, 1.05); axes[0].set_xlabel('OPD / Direct cosine at the same Full parameter state')
    axes[0].set_title('Actual parameter gradients')
    draw(axes[1], 'Direct_capped_budget_utilization', 1, 'Applied movement / OPD budget', 0, '#bb5a26')
    axes[1].set_xlabel('Mean utilization of nonzero output-movement budgets'); axes[1].set_title('Upper-budget control; no forced amplification')
    fig.suptitle('Parent means and 10,000 paired parent bootstrap intervals', fontsize=13)
    for extension in ['pdf', 'png']:
        p = destination/('parameter_geometry.'+extension); assert not p.exists()
        fig.savefig(p, dpi=180); files.append(p)
    plt.close(fig)
    lines = ['# Common-state OPD / Direct control', '',
        'All 5376 logical fits were globally sealed before GT scoring. Independent root review passed for the complete original Full reference, inputs, state/reset/inheritance/writeback, Adam and cap arithmetic, dense readouts, propagation and parent bootstrap statistics.', '',
        'The primary population excludes the existing development parents. These are historically exposed parents, and the comparison is conditional on original Full history. It does not measure three independent online streams. The constrained Direct arm enforces an upper bound on native output movement; it is not exact norm matching. Intervals are descriptive and unadjusted for multiple comparisons.', '',
        '| Setting | Paired contrast | Parents | Mean (pp) | 95% CI (pp) | Negative parent tail >5 pp |',
        '|---|---|---:|---:|---|---:|']
    for key, label in zip(keys, labels):
        for field in ['Full_minus_Direct_v', 'Full_minus_Direct_capped_v', 'Direct_capped_minus_Direct_v']:
            v = statistic(key, field); lo, hi = [x*100 for x in v['ci95']]
            lines.append(f"| {label} | {field} | {v['parents']} | {v['mean']*100:.3f} | [{lo:.3f}, {hi:.3f}] | {v['negative_parent_tail_5pp']} |")
    lines += ['', 'All historical populations, individual fixed orders, all three arms, negative tails, undefined cosines and cap utilization remain in the complete anonymous scalar files. The independent root statistics also separate observed decoder frames, unobserved decoder frames, interpolated frames and frames outside Native support. The PDF figures still require actual pixel view; publication, remote content verification, archive closure and the true common-state FINAL remain outstanding.']
    p = PUB/'REPORT.md'; assert not p.exists(); p.write_text('\n'.join(lines)+'\n')
    files.append(p)
    write(REC/'FIGURE_EXPORT.json', dict(status='pending_actual_PDF_pixel_view_publication_archive',
          files={str(p.relative_to(ROOT)):dict(sha256=sha(p), bytes=p.stat().st_size) for p in files},
          PDF_count=2, actual_PDF_pixel_view=False, task_complete=False, CPU_only=True, GT_read=True, time=time.time()))


if __name__ == '__main__':
    run()
