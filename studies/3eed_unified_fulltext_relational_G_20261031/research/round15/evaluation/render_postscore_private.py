"""Max four PRIVATE post-score RGB/GT diagnostics, never a prediction method.

Two fixed selection rules per platform: most harmful proposal change (if any,
otherwise largest source-pool oracle gap), then largest remaining pool gap.
All selection is explicitly post-score; no new task score, model or G is run.
"""
from pathlib import Path
import argparse
import hashlib
import itertools
import json
import sys
import textwrap
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'crossplatform/round7'), str(ROOT / 'research/round5/audit')]
from camera_projection import CameraSpec, project_model_points
from independent_core_v2 import transform_gt


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def check_seal(folder, digest):
    if sha(folder / 'SEALED.json') != digest:
        raise ValueError('Changed diagnostic input seal')
    value = read(folder / 'SEALED.json')
    for rel, expected in value['files'].items():
        if Path(rel).is_absolute() or '..' in Path(rel).parts or sha(folder / rel) != expected:
            raise ValueError('Changed diagnostic input member')
    return value


def corners(box):
    box = np.asarray(box, np.float64)
    dim = np.maximum(box[3:6], 1e-6)
    signs = np.array(list(itertools.product((-1., 1.), repeat=3)))
    local = signs * dim / 2
    angle = float(box[6]) if len(box) == 7 else 0.
    co, si = np.cos(angle), np.sin(angle)
    rotation = np.array([[co, -si, 0], [si, co, 0], [0, 0, 1]])
    return local @ rotation.T + box[:3]


def draw(ax, box, camera, wh, name, color, style):
    points = project_model_points(corners(box), camera, wh)
    valid = (points['depth'] > .1) & np.isfinite(points['uv']).all(1)
    edges = 0
    for i, j in itertools.combinations(range(8), 2):
        if (i ^ j).bit_count() == 1 and valid[i] and valid[j]:
            xy = points['uv'][[i, j]]
            ax.plot(xy[:, 0], xy[:, 1], color=color, ls=style, lw=1.8)
            edges += 1
    ax.plot([], [], color=color, ls=style, label=name + (' [partly projected]' if edges < 12 else ''))
    return {'projectable_corners': int(valid.sum()), 'drawn_edges': edges}


def main(args):
    if sha(args.release) != args.release_sha256:
        raise ValueError('Exact local render release required')
    release = read(args.release)
    if release.get('LOCAL_POSTSCORE_RGB_GT_RELEASE') is not True or release['renderer_sha256'] != sha(__file__):
        raise ValueError('Root-only local post-score render scope')
    predictions = ROOT / release['predictions']
    scored = ROOT / release['scored']
    semantic = ROOT / release['semantic']
    ps = check_seal(predictions, release['prediction_seal_sha256'])
    ss = check_seal(scored, release['scored_seal_sha256'])
    check_seal(semantic, release['semantic_seal_sha256'])
    if ps['n_samples'] != 204 or ss['n_samples'] != 204 or ss['prediction_seal_sha256'] != sha(predictions / 'SEALED.json'):
        raise ValueError('Actual complete score/prediction chain')
    for path, expected in release['input_bindings'].items():
        if sha(ROOT / path) != expected:
            raise ValueError('Diagnostic input binding')
    if args.output.exists():
        raise FileExistsError('New diagnostic directory required')
    # All actual score/target/sensor access is below the explicit boundary.
    result = read(scored / 'RESULTS_PRIVATE.json')
    manifests = {r['id']: r for r in read(ROOT / 'research/round15/input/FORMAL204_GPU_INPUTS.json')}
    cases = {r['id']: r for r in read(ROOT / 'research/round15/input/FORMAL_CASE204_PRIVATE.json')}
    frames = read(ROOT / 'research/round15/input/G_FRAME_INDEX_PRIVATE.json')
    index = {r['id']: r for r in read(ROOT / 'crossplatform/round5/baseline_engineering/SCORING_INDEX_PRIVATE.json')}
    requests = []
    for platform in ('drone', 'quad'):
        rows = [r for r in result['rows_PRIVATE'] if r['platform'] == platform and r['case_role'] == 'fresh64_recent_content_relative_only']
        gap = lambda r: r['offline_source_pool_oracle']['max_iou_all256'] - r['arms']['source']['iou']
        delta = lambda r: r['arms']['proposal_geo']['iou'] - r['arms']['source']['iou']
        harm = sorted(rows, key=lambda r: (delta(r), r['id']))[0]
        first = harm if delta(harm) < -1e-8 else sorted(rows, key=lambda r: (-gap(r), r['id']))[0]
        rule = 'most_negative_proposal_geo_minus_source' if delta(harm) < -1e-8 else 'largest_offline_source_pool_gap_no_negative_proposal_change'
        remaining = [r for r in rows if r['id'] != first['id']]
        second = sorted(remaining, key=lambda r: (-gap(r), r['id']))[0]
        requests.extend([(first, rule), (second, 'largest_remaining_offline_source_pool_gap')])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    args.output.mkdir(parents=True)
    records = []
    for row, rule in requests:
        sid = row['id']; manifest = manifests[sid]; case = cases[sid]; alias = case['case_alias']
        frame = frames[case['frame_alias']]; item = index[sid]
        if sha(item['meta_path']) != item['metadata_container_sha256'] or sha(manifest['imagepath']) != frame['image_sha256']:
            raise ValueError('Exact original image/target join')
        metadata = read(item['meta_path'])
        target = transform_gt(metadata['ground_info'][item['object_index']]['bbox_3d'], manifest['platform'], item.get('waymo_view_suffix'))
        fields = {k: frame[k] for k in ('image_intrinsic', 'image_extrinsic')}
        if frame['camera_distortion'] is not None:
            fields['camera_distortion'] = frame['camera_distortion']
        camera = CameraSpec.from_camera_fields(manifest['platform'], fields, missing_distortion_policy='assume_zero_declared')
        image = np.asarray(Image.open(manifest['imagepath']).convert('RGB'))
        panel = np.asarray(Image.open(semantic / 'panels_PRIVATE' / (alias + '__order0.png')).convert('RGB'))
        with np.load(predictions / 'cases_PRIVATE' / (alias + '.npz'), allow_pickle=False) as data:
            boxes = data['arm_boxes_raw6'].copy(); names = data['arm_names'].tolist()
        fig, axes = plt.subplots(1, 2, figsize=(16, 8.8))
        fig.subplots_adjust(top=.88, bottom=.32, left=.02, right=.99, wspace=.025)
        for ax, pixels in zip(axes, (image, panel)):
            ax.imshow(pixels); ax.set_xlim(0, image.shape[1]); ax.set_ylim(image.shape[0], 0); ax.axis('off')
        projections = {}
        for name, color, style in [('source', '#00cbe6', '--'), ('graph_geo', '#38e76a', '-'), ('proposal_geo', '#ffb24e', ':')]:
            projections[name] = draw(axes[0], boxes[names.index(name)], camera, (image.shape[1], image.shape[0]), name, color, style)
        projections['GT7D'] = draw(axes[0], target, camera, (image.shape[1], image.shape[0]), 'GT7D', '#fa4545', '-.')
        axes[0].legend(fontsize=7, loc='upper right')
        fig.suptitle('PRIVATE post-score diagnostic | ' + alias + ' | ' + rule, fontsize=11)
        lines = []
        for name in ('source', 'proposal_geo', 'graph_assoc', 'graph_geo'):
            arm = row['arms'][name]
            lines.append(f"{name}: IoU={arm['iou']:.6f}; center={arm['coordinate_errors']['center_L2_m']:.3f}m; size={arm['coordinate_errors']['size_L2_m']:.3f}m; {arm['status']}")
        fig.text(.02, .28, '\n'.join(lines), fontsize=8, va='top')
        fig.text(.02, .16, '\n'.join(textwrap.wrap('PRIVATE input description: ' + manifest['caption'], 165)), fontsize=8, va='top')
        fig.text(.02, .035, 'Post-score selection is descriptive only. Right: original sealed numbered full scene. No new prediction, G, IoU, fitting or parameter update.', fontsize=8)
        path = args.output / (alias + '.jpg')
        fig.savefig(path, dpi=120); plt.close(fig)
        records.append({'case_alias': alias, 'selection': rule, 'image_sha256': sha(path), 'projection': projections,
                        'native_yaw0_predictions_GT_yaw_preserved': True, 'raw_caption_and_image_private': True})
    dump(args.output / 'SUMMARY_PRIVATE.json', {'state': 'POSTSCORE_PRIVATE_DIAGNOSTIC_ONLY', 'n': len(records), 'records': records,
          'release_sha256': sha(args.release), 'renderer_sha256': sha(__file__), 'not_performance_evidence_or_identity_certification': True})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--release-sha256', required=True); parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
