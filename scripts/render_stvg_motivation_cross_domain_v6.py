"""White, borderless Figure 1 exports, backed only by the new sealed cross data."""
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.stvg_motivation_cross_domain_common_v6 import BASE, DIRECTIONS, MODELS, read, write, sha, check_global_seal


def setup():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12,
        'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white',
        'svg.fonttype': 'none', 'pdf.fonttype': 42, 'axes.edgecolor': 'white'})
    return plt


def statistics_panel(ax, data):
    from matplotlib.patches import Rectangle
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    ax.text(0, .99, '(b) Cross-domain errors', fontsize=17, weight='semibold', va='top', color='#1D1D1F')
    colors = ['#DAEBE4', '#FFDBC5', '#CADDF6', '#E9E9EB']
    headers = ['T+  S+', 'T+  S−', 'T−  S+', 'T−  S−']
    xs = [.31, .51, .71, .91]
    for x, label in zip(xs, headers):
        ax.text(x, .895, label, ha='center', va='center', fontsize=11, color='#515154')
    labels = ['VidSTG → HC2', 'HC2 → VidSTG']
    for index, direction in enumerate(DIRECTIONS):
        y = .795-index*.355
        ax.text(0, y, labels[index], weight='semibold', fontsize=13, va='center', color='#1D1D1F')
        for k, model in enumerate(MODELS):
            cy = y-.095-k*.105
            name = 'TA-STVG' if model == 'tastvg' else 'TubeDETR'
            ax.text(0, cy, name, va='center', fontsize=12, color='#515154')
            r = data['directions'][direction]['models'][model]
            assert r['N'] == 128 and sum(r['counts']) == 128
            for x, value, count, color in zip(xs, r['percent'], r['counts'], colors):
                ax.add_patch(Rectangle((x-.088, cy-.041), .176, .082,
                    facecolor=color, edgecolor='none', linewidth=0))
                ax.text(x, cy+.008, f'{value:.1f}%', ha='center', va='center', fontsize=15,
                    weight='medium', color='#1D1D1F')
                ax.text(x, cy-.020, str(count), ha='center', va='center', fontsize=8.5, color='#515154')
    ax.text(0, .055, 'T: tIoU > 0.5    S: mean frame IoU > 0.5', fontsize=9.5, color='#6E6E73')
    ax.text(0, .020, '128 parent sources / domain', fontsize=9.5, color='#6E6E73')


def export(fig, stem):
    outputs = {}
    for extension in ['png', 'pdf', 'svg']:
        path = BASE/(stem+'.'+extension)
        assert not path.exists(), 'Preserve previous exports'
        fig.savefig(path, dpi=240, facecolor='white', transparent=False)
        outputs[path.name] = dict(sha256=sha(path), bytes=path.stat().st_size)
    write(BASE/(stem+'_RENDER.json'), dict(status='rendered_pending_actual_root_view', files=outputs,
        cross_domain_only=True, quadrants_sha256=sha(BASE/'QUADRANT_STATISTICS.json'),
        white_background=True, outer_or_panel_borders=False, overall_figure_title=False,
        actual_root_view=False, time=time.time()))


def render_statistics():
    check_global_seal()
    data = read(BASE/'QUADRANT_STATISTICS.json')
    assert data['source_training_only'] and data['native_only'] and data['total_native_cells'] == 512
    plt = setup(); fig = plt.figure(figsize=(8.5, 6.5))
    ax = fig.add_axes([.045, .04, .91, .92]); statistics_panel(ax, data)
    export(fig, 'panel_B_cross_domain'); plt.close(fig)


def render_full(case):
    """Root supplies an audited actual-frame case; no stock/generated media."""
    import numpy as np
    from matplotlib.patches import Rectangle
    from PIL import Image
    check_global_seal()
    data = read(BASE/'QUADRANT_STATISTICS.json')
    plt = setup(); fig = plt.figure(figsize=(18, 7.2))
    ax = fig.add_axes([.03, .05, .59, .90]); ax.axis('off')
    ax.text(0, .99, '(a) Native cross-domain predictions', fontsize=17, weight='semibold', va='top', color='#1D1D1F')
    # Use the actual query, with no invented semantic rewrite.
    ax.text(0, .91, case['query'], fontsize=12, color='#515154', va='top', wrap=True)
    ax.text(.002, .05, 'GT', color='#248F58', fontsize=11)
    ax.text(.072, .05, 'Prediction', color='#E04D4A', fontsize=11)
    frame_x = [.31, .54, .77]
    for k, model in enumerate(MODELS):
        top = .80-k*.33
        ax.text(0, top, 'TA-STVG' if model == 'tastvg' else 'TubeDETR', fontsize=13, weight='semibold', color='#1D1D1F')
        timeline = fig.add_axes([.03, .05+(top-.20)*.90, .155, .115])
        timeline.set_xlim(*case['input_frame_extent']); timeline.set_ylim(0, 1); timeline.axis('off')
        timeline.plot(case['input_frame_extent'], [.15, .15], color='#D2D2D7', lw=1)
        for a, y, color in [(case['GT_interval'], .75, '#248F58'), (case['models'][model]['interval'], .43, '#E04D4A')]:
            if a is not None:
                timeline.plot(a, [y, y], color=color, lw=6, solid_capstyle='round')
        for j, frm in enumerate(case['frames']):
            path = BASE/frm['path']; assert sha(path) == frm['sha256']
            image = np.asarray(Image.open(path).convert('RGB'))
            panel = fig.add_axes([.03+frame_x[j]*.59, .05+(top-.22)*.90, .125, .21])
            panel.imshow(image); panel.axis('off')
            for box, color, style in [(frm.get('GT_box'), '#248F58', '-'),
                    (case['models'][model]['boxes'][j], '#E04D4A', '--')]:
                if box is None:
                    continue
                x1, y1, x2, y2 = box
                panel.add_patch(Rectangle((x1, y1), x2-x1, y2-y1, fill=False,
                    edgecolor=color, lw=2.0, linestyle=style))
    stats = fig.add_axes([.66, .05, .31, .90]); statistics_panel(stats, data)
    export(fig, 'figure1_cross_domain'); plt.close(fig)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'full':
        render_full(read(BASE/'PRIVATE_CASE.json'))
    else:
        render_statistics()
