"""Offline aligned Source/Ours/GT figures from sealed, scored predictions."""
import argparse
import json
import os
from pathlib import Path
import sys
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True)
    p.add_argument('--labels', required=True)
    p.add_argument('--metrics', required=True)
    p.add_argument('--predictions', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    from stvg.runtime import sha
    from stvg.evaluation import read_rows
    from stvg.metrics_base import xyxy
    from stvg.tube_decode import decode
    from stvg.tube_runner import write
    root = Path(args.predictions).resolve()
    seal = json.loads((root/'sealed.json').read_text())
    metrics = json.loads(Path(args.metrics).read_text())
    if metrics['manifest_sha256'] != sha(args.manifest) or metrics['labels_sha256'] != sha(args.labels) or seal['manifest_sha256'] != sha(args.manifest):
        raise ValueError('Scored population does not match supplied inputs/labels')
    for entry in seal['files'].values():
        path = (root/entry['path']).resolve()
        if not path.is_relative_to(root) or sha(path) != entry['sha256']:
            raise ValueError('Prediction seal mismatch')
    inputs = {r['id']: r for r in read_rows(args.manifest)}
    labels = {r['id']: r for r in read_rows(args.labels)}
    candidates = {k: [] for k in ['spatial_correction', 'temporal_correction', 'joint_improvement', 'failure']}
    for row in metrics['rows']:
        a, b = row['values']['Source'], row['values']['Full']
        dv, dt, ds = [b[k]-a[k] for k in ['m_vIoU', 'm_tIoU', 'm_sIoU']]
        key = row['id']
        if ds > 0 and dt <= 0:
            candidates['spatial_correction'].append((ds, key))
        if dt > 0 and ds <= 0:
            candidates['temporal_correction'].append((dt, key))
        if ds > 0 and dt > 0:
            candidates['joint_improvement'].append((min(ds, dt), key))
        if dv < 0:
            candidates['failure'].append((-dv, key))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    selections = {}
    for category, choices in candidates.items():
        if not choices:
            selections[category] = dict(status='no_qualifying_case_in_complete_population')
            continue
        _, key = sorted(choices, key=lambda x: (-x[0], x[1]))[0]
        record, gt = inputs[key], labels[key]
        meta = record['metadata']
        frames, ids = decode(meta, record['dataset'])
        with np.load(root/seal['files'][key]['path'], allow_pickle=False) as z:
            boxes = [xyxy(z['Source'], meta['width'], meta['height']), xyxy(z['Live'], meta['width'], meta['height'])]
            spans = [z['Native_interval'].tolist(), z['Strong_interval'].tolist(), gt['interval']]
            observed = {ids[i] for i in z['positions8'].tolist()}
        # Predeclared display locations, independent of frame-level IoU values.
        g, h = gt['interval']
        anchors = [ids[0], g, (g+h)/2, h-1, sum(spans[0])/2, sum(spans[1])/2, ids[-1]]
        indices = sorted({min(range(len(ids)), key=lambda i: (abs(ids[i]-a), i)) for a in anchors})
        fig = plt.figure(figsize=(3*len(indices), 10), layout='constrained')
        grid = fig.add_gridspec(4, len(indices), height_ratios=[1, 1, 1, .32])
        gt_boxes = {int(k): b for k, b in gt['boxes'].items()}
        colors = ['#d45500', '#006eae', '#17823b']
        for row, name in enumerate(['Source', 'Ours', 'GT']):
            for col, index in enumerate(indices):
                fid = ids[index]
                ax = fig.add_subplot(grid[row, col])
                ax.imshow(frames[index])
                box = gt_boxes.get(fid) if row == 2 else boxes[row][index] if spans[row][0] <= fid < spans[row][1] else None
                if box is not None:
                    x, y, x1, y1 = box
                    ax.add_patch(Rectangle((x, y), x1-x, y1-y, fill=False, edgecolor=colors[row], linewidth=2))
                ax.set_title(f'{name} | {fid/meta["fps"]:.2f}s | {"observed" if fid in observed else "unobserved"}', fontsize=10)
                ax.set_axis_off()
        ax = fig.add_subplot(grid[3, :])
        for i, (name, span, color) in enumerate(zip(['Source', 'Ours', 'GT'], spans, colors)):
            a, b = [v/meta['fps'] for v in span]
            ax.barh(i, b-a, left=a, color=color, height=.5)
        ax.set_yticks([0, 1, 2], ['Source', 'Ours', 'GT'])
        ax.set_xlabel('Physical video time (seconds); dataset endpoint conventions preserved')
        ax.set_xlim(min(ids[0], *(s[0] for s in spans))/meta['fps'], max(ids[-1]+1, *(s[1] for s in spans))/meta['fps'])
        fig.suptitle(f'{category}: {meta["caption"]}', wrap=True)
        dest = output/f'{category}.png'
        fig.savefig(dest, dpi=150)
        plt.close(fig)
        selections[category] = dict(id=key, file=dest.name, frames=[ids[i] for i in indices],
                                    disclosure='post-scoring descriptive selection, not independent quantitative evidence')
        del frames
    write(output/'selection.json', dict(status='rendered_available_categories', selections=selections,
        selection_rule='largest positive component change under category sign constraints; largest vIoU loss for failure; ID tie break',
        all_examples_from_same_sealed_population=True, model_inference_performed=False))


if __name__ == '__main__':
    main()
