"""Independent CPU source-Frozen geometry readback, without shared metric code."""
import hashlib
import json
import math
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts/desta3d_v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def iou_xyxy(a, b):
    inter = max(0., min(a[2], b[2])-max(a[0], b[0])) * max(0., min(a[3], b[3])-max(a[1], b[1]))
    area_a = max(0., a[2]-a[0]) * max(0., a[3]-a[1])
    area_b = max(0., b[2]-b[0]) * max(0., b[3]-b[1])
    return inter / max(area_a+area_b-inter, 1e-7)


def score(frame_ids, interval, positions, tokens, label, format_ok=True):
    """Physical half-open interval; sparse decoder boxes stay sparse."""
    assert frame_ids == label['frame_ids']
    valid = [i for i in range(len(frame_ids)) if label['box_valid'][i] and label['event_active'][i]]
    good_interval = (interval is not None and len(interval) == 2 and
                     0 <= interval[0] <= interval[1] < len(frame_ids))
    if not (format_ok and good_interval):
        return dict(vIoU=0., sIoU=0., tIoU=0., spatial_support_count=len(valid), format_ok=False)
    assert len(positions) == len(tokens) and len(set(positions)) == len(positions)
    boxes = {i: [float(v)/1000 for v in b] for i, b in zip(positions, tokens)}
    assert all(0 <= i < len(frame_ids) for i in boxes)
    pred = [frame_ids[interval[0]], frame_ids[interval[1]]+1]
    gt = [label['event_interval']['begin_fid'], label['event_interval']['end_fid']]
    ts, te = max(pred[0], gt[0]), min(pred[1], gt[1])
    us, ue = min(pred[0], gt[0]), max(pred[1], gt[1])
    per_frame = {i: iou_xyxy(boxes.get(i, [0., 0., 0., 0.]), label['boxes_xyxy'][i]) for i in valid}
    denominator = sum(us <= f < ue for f in frame_ids)
    numerator = sum(v for i, v in per_frame.items() if ts <= frame_ids[i] < te)
    return dict(vIoU=numerator/max(denominator, 1), sIoU=sum(per_frame.values())/max(len(valid), 1),
                tIoU=max(0, te-ts)/max(ue-us, 1), spatial_support_count=len(valid),
                temporal_union_sample_count=denominator, intersection_iou_sum=numerator,
                predicted_interval=pred, gt_interval=gt, format_ok=True)


def main():
    torch.set_num_threads(1)
    # Hand-computed unequal physical spacings, native box absent at one GT frame.
    lab = dict(frame_ids=[0, 2, 8, 10], box_valid=[True]*4,
               event_active=[False, True, True, False], boxes_xyxy=[[0.,0.,1.,1.]]*4,
               event_interval=dict(begin_fid=2, end_fid=9))
    test = score(lab['frame_ids'], [0, 3], [1], [[0,0,1000,1000]], lab)
    assert test['vIoU'] == .25 and test['sIoU'] == .5 and math.isclose(test['tIoU'], 7/11)
    rows = json.loads((ART/'SOURCE_INPUTS.json').read_text())
    initial = json.loads((ART/'SOURCE_INITIAL_INPUTS.json').read_text())
    initial_val = {r['key'] for r in initial if r['split'] == 'validation'}
    labels = json.loads((ART/'SOURCE_LABELS_TRAINING_ONLY.json').read_text())
    barrier = json.loads((ART/'SOURCE_FULL_FEATURE_BARRIER.json').read_text())
    result = []
    for r in rows:
        if r['split'] != 'validation':
            continue
        for condition in ['clean'] + (['noise_medium', 'defocus_extreme'] if r['key'] in initial_val else []):
            path = ART/'source_features'/condition/(hashlib.sha256(r['key'].encode()).hexdigest()+'.pt')
            assert sha(path) == barrier['files'][str(path)], path
            cache = torch.load(path, map_location='cpu', weights_only=False)
            assert cache['key'] == r['key'] and cache['frame_ids'] == r['input']['frame_ids']
            values = score(cache['frame_ids'], cache['interval'], cache['spatial_positions'],
                           cache['box_logits'].argmax(-1).tolist(), labels[r['key']], cache['format_ok'])
            result.append(dict(key=r['key'], source=r['source'], condition=condition,
                               cache_sha=barrier['files'][str(path)], metrics=values))
    summaries = {}
    for condition in ['clean', 'noise_medium', 'defocus_extreme']:
        group = [x for x in result if x['condition'] == condition]
        parents = sorted({x['source'] for x in group})
        parent_rows = [dict(source=p, **{m: sum(x['metrics'][m] for x in group if x['source']==p)/sum(x['source']==p for x in group)
                                        for m in ['vIoU','sIoU','tIoU']}) for p in parents]
        summaries[condition] = dict(queries=len(group), parents=len(parents),
                                    parent_macro={m:sum(x[m] for x in parent_rows)/len(parents) for m in ['vIoU','sIoU','tIoU']},
                                    parent_rows=parent_rows)
    payload = dict(scope='independent Frozen source-validation geometry; no target labels, no fitted adapter',
                   code_sha=sha(__file__), label_sha=sha(ART/'SOURCE_LABELS_TRAINING_ONLY.json'),
                   barrier_sha=sha(ART/'SOURCE_FULL_FEATURE_BARRIER.json'),
                   metric='normalized xyxy independent arithmetic; corrected sampled physical-time union',
                   rows=result, summaries=summaries)
    out = ART/'SOURCE_FROZEN_INDEPENDENT_METRICS.json'
    assert not out.exists(), out
    out.write_text(json.dumps(payload, indent=2)+'\n')
    print(json.dumps({k:{x:y for x,y in v.items() if x!='parent_rows'} for k,v in summaries.items()},indent=2))


if __name__ == '__main__':
    main()
