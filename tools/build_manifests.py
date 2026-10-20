"""Offline official-split intake. Never imported by the predictor.

No query is dropped for missing media, track holes, or model behavior. A formal
manifest is written only when its whole expected population is available.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stvg.runtime import sha
from stvg.tube_runner import write, population_hash, digest

HC_SHA = '61f7a2a74d39c13dd380540d4ac819edb7abcde1826d44229e539d128bf3260c'
VID_SHA = 'ef769b9823ce2a4da8ff190ab69ccc77b8965181c58384fadc1f81e96fc4f2a7'


def hc_grid(frame_count):
    # Exact TA-STVG input support: range(0, frame_count-1), 6.4 fps,
    # append the last supported node. It is fixed before any model output.
    support = list(range(frame_count-1))
    rate = 6.4/(frame_count/20.)
    out = [0]
    for fid in support:
        if int(out[-1]*rate) < int(fid*rate):
            out.append(fid)
    if out[-1] != support[-1]:
        out.append(support[-1])
    return out


def vid_grid(begin, end):
    support = list(range(begin, end+1))
    if len(support) > 128:
        return [support[int(i)] for i in np.linspace(0, len(support)-1, 128)]
    return support


def media_index(root):
    out = {}
    for p in Path(root).rglob('*'):
        if p.suffix.lower() in {'.mp4', '.mkv', '.avi'}:
            if p.name in out:
                raise ValueError(f'Ambiguous media basename: {p.name}')
            out[p.name] = p.resolve()
    return out


def vidor_records(path):
    records = {}
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            if name.endswith('.json'):
                value = json.loads(z.read(name))
                if value['video_id'] in records:
                    raise ValueError('Duplicate VidOR video ID')
                records[value['video_id']] = value
    return records


def build(dataset, annotations, video_root, output, vidor_annotations=None, inventory=False):
    annotations, output = Path(annotations), Path(output)
    expected_sha = HC_SHA if dataset == 'hc2' else VID_SHA
    if sha(annotations) != expected_sha:
        raise ValueError('Annotation bytes differ from the registered canonical release; do not substitute a same-name split')
    raw = json.loads(annotations.read_text())
    media = media_index(video_root)
    rows, labels, missing = [], [], []
    media_sha = {}

    def path_for(name):
        if name not in media:
            missing.append(name)
            return str(Path(video_root).resolve()/name), None
        path = media[name]
        if name not in media_sha:
            media_sha[name] = sha(path)
        return str(path), media_sha[name]

    if dataset == 'hc2':
        if len(raw) != 1902:
            raise ValueError('Expected all 1902 user-confirmed author-release HC2 validation records')
        for i, (name, value) in enumerate(raw.items()):
            width = value['width'] if 'width' in value else value['img_size'][1]
            height = value['height'] if 'height' in value else value['img_size'][0]
            path, h = path_for(name)
            key = f'hc2-{i:05d}'
            metadata = dict(caption=value['English'].lower(), index=i, width=width, height=height,
                fps=value['img_num']/20., frame_count=value['img_num'], duration=20.,
                original_video_id=name, video_path=path, video_sha256=h, frame_ids=hc_grid(value['img_num']))
            rows.append(dict(id=key, dataset=dataset, key=key, metadata=metadata))
            start = int(value['st_frame'])-1
            boxes = {start+j: [x, y, min(x+w, width), min(y+h, height)]
                     for j, (x, y, w, h) in enumerate(value['bbox'])}
            labels.append(dict(id=key, dataset=dataset, parent=Path(name).stem.split('_', 1)[1],
                               interval=[start, start+len(value['bbox'])-1], boxes=boxes))
    else:
        if not vidor_annotations:
            raise ValueError('VidOR validation trajectories are required for offline labels')
        trajectories = vidor_records(vidor_annotations)
        for annotation_index, value in enumerate(raw):
            original = trajectories[value['vid']]
            path, h = path_for(Path(original['video_path']).name)
            for kind in ['captions', 'questions']:
                for query_index, query in enumerate(value[kind]):
                    i = len(rows)
                    key = f'vidstg-{annotation_index:05d}-{kind[0]}-{query_index:02d}'
                    begin, end = value['used_segment']['begin_fid'], value['used_segment']['end_fid']
                    metadata = dict(caption=query['description'], index=i, width=value['width'], height=value['height'],
                        fps=value['fps'], frame_count=value['frame_count'], original_video_id=value['vid'],
                        start_frame=begin, end_frame=end, kind=kind,
                        video_path=path, video_sha256=h, frame_ids=vid_grid(begin, end))
                    rows.append(dict(id=key, dataset=dataset, key=key, metadata=metadata))
                    span = [value['temporal_gt']['begin_fid'], value['temporal_gt']['end_fid']]
                    boxes = {}
                    for fid in range(*span):
                        found = [b for b in original['trajectories'][fid] if b['tid'] == query['target_id']]
                        if len(found) != 1:
                            raise ValueError('Official target track hole; never silently drop or fill a query')
                        b = found[0]['bbox']
                        boxes[fid] = [b['xmin'], b['ymin'], b['xmax'], b['ymax']]
                    labels.append(dict(id=key, dataset=dataset, parent=value['vid'], interval=span, boxes=boxes))
    expected = (1902, 227) if dataset == 'hc2' else (10303, 732)
    if (len(rows), len({g['parent'] for g in labels})) != expected:
        raise ValueError('Canonical query/parent count mismatch')
    output.mkdir(parents=True, exist_ok=True)
    summary = dict(dataset=dataset, queries=len(rows), parents=expected[1], annotation_sha256=sha(annotations),
        missing_media=len(set(missing)), missing_media_names_sha256=digest(sorted(set(missing))),
        official_queries_filtered=0, labels_used_only_during_offline_intake=True,
        endpoint_policy='preserve existing dataset-specific scoring convention',
        input_grid='hc default FFmpeg output indices, 6.4 fps' if dataset == 'hc2' else 'zero-based used_segment inclusive support, linspace floor up to 128',
        status='intake_inventory_only' if inventory or missing else 'complete_manifest')
    write(output/f'{dataset}_intake.json', summary)
    if inventory:
        return summary
    if missing:
        raise FileNotFoundError(f'{len(set(missing))} required videos missing; no partial manifest was written')
    manifest, label_path = output/f'{dataset}.jsonl', output/f'{dataset}_labels.jsonl'
    if manifest.exists() or label_path.exists():
        raise ValueError('Refusing to overwrite an evaluation manifest')
    manifest.write_text(''.join(json.dumps(r, ensure_ascii=False, separators=(',', ':'))+'\n' for r in rows))
    label_path.write_text(''.join(json.dumps(r, separators=(',', ':'))+'\n' for r in labels))
    summary.update(population_sha256=population_hash(rows), manifest_sha256=sha(manifest), labels_sha256=sha(label_path))
    write(output/f'{dataset}_intake.json', summary)
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=['hc2', 'vidstg'], required=True)
    p.add_argument('--annotations', required=True)
    p.add_argument('--video-root', required=True)
    p.add_argument('--vidor-annotations')
    p.add_argument('--output', default='data/evaluation')
    p.add_argument('--inventory', action='store_true')
    args = p.parse_args()
    print(json.dumps(build(args.dataset, args.annotations, args.video_root, args.output, args.vidor_annotations, args.inventory), indent=2))


if __name__ == '__main__':
    main()
