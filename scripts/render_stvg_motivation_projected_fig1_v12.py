"""Render a three-backbone Fig. 1 layout using only the saved case evidence.

The pending Projected lane deliberately has no interval. An expert frame or
box may only be added in a separately source-bound revision, not from GT.
This renderer performs no inference, training, expert calls, or GT scoring.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import hashlib
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/stvg_motivation_projected_fig1_v12'
SOURCE = ROOT / 'artifacts/stvg_motivation_sketch_v10'
MODELS = [('tastvg', 'TA-STVG'), ('tubedetr', 'TubeDETR'),
          ('ptd', 'PTD / Qwen3-VL')]
GT = '#338263'
NATIVE = '#D58C3F'
PROJECTED = '#527DAF'
INK = '#20262D'
MUTED = '#75808C'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(data, handle, indent=2)
        handle.write('\n')


def render():
    lock = json.loads((BASE / 'DESIGN_LOCK.json').read_text())
    for rel, digest in lock['source_pins'].items():
        assert sha(ROOT / rel) == digest, rel
    prior = json.loads((BASE / 'LAYOUT_RUNTIME002.json').read_text())
    revision = json.loads((BASE / 'LAYOUT_RUNTIME003.json').read_text())
    assert prior['prior_renderer_sha256'] == lock['renderer_sha256']
    assert revision['prior_renderer_sha256'] == prior['renderer_sha256']
    assert sha(Path(__file__)) == revision['renderer_sha256']
    case = json.loads((SOURCE / 'PRIVATE_CASE.json').read_text())
    assert len(case['frames']) == 5
    assert set(case['models']) == {model for model, _ in MODELS}
    assert lock['Projected_intervals_available'] == 0
    assert lock['actual_DINO_observations_available'] == 0

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42,
                         'ps.fonttype': 42, 'svg.fonttype': 'none'})
    fig = plt.figure(figsize=(16.5, 14.4), facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    ax.text(.035, .97, 'Spatio-Temporal Grounding under Domain Shift',
            fontsize=22, weight='bold', color=INK, va='top')
    ax.text(.035, .931, 'Query: "' + case['query'] + '"',
            fontsize=18, color=INK, va='top')
    ax.text(.969, .969, 'LAYOUT PREVIEW', fontsize=12, color=MUTED,
            ha='right', va='top', weight='bold')
    handles = [Line2D([0], [0], color=GT, lw=2.7, ls=(0, (3, 2))),
               Line2D([0], [0], color=NATIVE, lw=2.7),
               Line2D([0], [0], color=PROJECTED, lw=2.7)]
    ax.legend(handles, ['GT', 'Native prediction',
                        'DINO / Projected style (reserved)'],
              loc='upper left', bbox_to_anchor=(.034, .902),
              ncol=3, frameon=False, fontsize=13, handlelength=2.2,
              columnspacing=3, borderpad=0, labelspacing=.4)

    frame_x, frame_w, frame_gap = .16, .156, .006
    frame_h = frame_w * (16.5 / 14.4) * (480 / 640)
    bar_x, bar_w = .16, .805
    start, stop = case['timeline_extent']
    x_for = lambda frame: bar_x + bar_w * (frame - start) / (stop - start)
    media = []
    for row, (model, title) in enumerate(MODELS):
        top = .833 - row * .268
        bottom = top - frame_h
        ax.text(.035, top - .011, title, fontsize=18 if row < 2 else 16,
                weight='bold', color=INK, va='top')
        ax.text(.035, top - .046, 'Spatial', fontsize=12, color=MUTED,
                va='top')
        for col, record in enumerate(case['frames']):
            image_path = SOURCE / record['path']
            assert sha(image_path) == record['sha256']
            rgb = Image.open(image_path).convert('RGB')
            assert hashlib.sha256(rgb.tobytes()).hexdigest() == record['pixel_sha256']
            x = frame_x + col * (frame_w + frame_gap)
            image_ax = fig.add_axes([x, bottom, frame_w, frame_h])
            image_ax.imshow(rgb)
            image_ax.set_xlim(0, rgb.width)
            image_ax.set_ylim(rgb.height, 0)
            image_ax.set_xticks([])
            image_ax.set_yticks([])
            for spine in image_ax.spines.values():
                spine.set_color('#CFD5DC')
                spine.set_linewidth(.7)
            for box, color, dashed in [(record['GT_box'], GT, True),
                                       (record['predictions'][model], NATIVE, False)]:
                x1, y1, x2, y2 = box
                rectangle = Rectangle((x1, y1), x2 - x1, y2 - y1,
                                      fill=False, ec=color, lw=2.2,
                                      linestyle=(0, (3, 2)) if dashed else '-')
                rectangle.set_path_effects([
                    patheffects.Stroke(linewidth=3.5, foreground='white', alpha=.92),
                    patheffects.Normal()])
                image_ax.add_patch(rectangle)
            gx, gy, _, _ = record['GT_box']
            image_ax.text(gx, max(0, gy - 6), 'GT', fontsize=9, color='white',
                          va='bottom', weight='bold',
                          bbox=dict(facecolor=GT, edgecolor='none', pad=1.4))
            ax.text(x + frame_w / 2, top + .008,
                    f'$t_{{{col + 1}}}$  (f{record["frame_id"]})',
                    fontsize=11.5, ha='center', color=INK)
            media.append(dict(model=model, frame_sha256=record['sha256'],
                              pixel_sha256=record['pixel_sha256']))

        ax.text(.035, bottom - .017, 'Temporal', fontsize=12, color=MUTED,
                va='top')
        lanes = [('GT', case['GT_interval'], GT),
                 ('Native', case['models'][model]['interval'], NATIVE),
                 ('Projected', None, PROJECTED)]
        for lane, (label, interval, color) in enumerate(lanes):
            y = bottom - .025 - lane * .021
            ax.text(.146, y, label, ha='right', va='center',
                    fontsize=12.5, weight='bold', color=INK)
            if interval is not None:
                a, b = interval
                ax.add_patch(Rectangle((bar_x, y - .006), bar_w, .012,
                                       fc='#F2F4F6', ec='none', zorder=1))
                ax.add_patch(Rectangle((x_for(a), y - .006),
                                       x_for(b) - x_for(a), .012,
                                       fc=color, ec='none', zorder=3))
                for point in [a, b]:
                    ax.plot([x_for(point), x_for(point)],
                            [y - .009, y + .009], color=color, lw=1.1, zorder=4)
            else:
                ax.text(bar_x + .012, y, 'Not yet measured - no interval drawn',
                        va='center', fontsize=11.5, color=MUTED, style='italic')
        axis_y = bottom - .082
        ax.plot([bar_x, bar_x + bar_w], [axis_y, axis_y],
                color='#C9D0D7', lw=.8)
        for value in [start, *[f['frame_id'] for f in case['frames']], stop]:
            tx = x_for(value)
            ax.plot([tx, tx], [axis_y, axis_y + .004], color='#9AA4AF', lw=.8)
            ax.text(tx, axis_y - .006, str(value), color=MUTED,
                    fontsize=9, va='top', ha='center')
        if row == 2:
            ax.text(.964, axis_y - .022, 'Original frame index', color=MUTED,
                    fontsize=9, va='top', ha='right')

    ax.text(.035, .039,
            'GT and Native are the original measured outputs. '
            'This case has no recorded DINO observations; no DINO boxes or dots are drawn.',
            fontsize=11.5, color=MUTED, va='top')
    ax.text(.035, .019,
            'Projected is reserved for matched measurements. '
            'PTD / Qwen3-VL uses a joint-trained checkpoint; '
            'the other two rows use HC-STVG2 source-only checkpoints.',
            fontsize=11.5, color=MUTED, va='top')
    out = BASE / 'layout_preview003'
    out.mkdir(parents=True, exist_ok=True)
    output_files = {}
    for suffix in ['png', 'pdf', 'svg']:
        path = out / ('figure1_three_backbone_projected_preview.' + suffix)
        assert not path.exists(), path
        fig.savefig(path, dpi=240 if suffix == 'png' else 300,
                    facecolor='white', metadata={'Creator': 'STVG Figure renderer'})
        output_files[suffix] = dict(path=str(path.relative_to(ROOT)),
                                    sha256=sha(path), bytes=path.stat().st_size)
    plt.close(fig)
    receipt = dict(status='layout_preview_rendered_pending_actual_visual_review',
                   scope='three-backbone GT/Native layout; unmeasured additions left empty',
                   model_count=3, frame_count=5, image_panels=15,
                   GT_overlays=15, Native_overlays=15,
                   GT_intervals=3, Native_intervals=3,
                   Projected_intervals_drawn=0, DINO_boxes_drawn=0,
                   DINO_observation_dots_drawn=0,
                   source_case_sha256=sha(SOURCE / 'PRIVATE_CASE.json'),
                   private_media=True, public_media_exported=False,
                   actual_supplemental_measurements_complete=False,
                   new_model_calls=0, new_GPU_calls=0, new_DINO_calls=0,
                   new_GT_scoring=0, active_P7A_payload_access=False,
                   output_files=output_files, time=time.time())
    write_new(out / 'RENDER_RECEIPT.json', receipt)
    print(json.dumps({k: receipt[k] for k in ['status', 'model_count',
                      'image_panels', 'Projected_intervals_drawn', 'DINO_boxes_drawn']}))


if __name__ == '__main__':
    render()
