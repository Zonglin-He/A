"""Local-only real RGB/GT overlays after the independent cross-domain root audit."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
from scripts.stvg_motivation_cross_domain_common_v6 import BASE, MODELS, read, write, sha, check_global_seal


def prepare(direction='hc2_to_vidstg', ordinal=42):
    check_global_seal()
    assert read(BASE/'ROOT_AUDIT.json')['status'] == 'pass'
    assert not (BASE/'PRIVATE_CASE.json').exists()
    from scripts.score_stvg_motivation_cross_domain_v6 import read_truths
    from vg_tta.exact_frame_decode_audit_v2 import decode
    roster = read(BASE/direction/'ROSTER.json'); row = roster['rows'][ordinal]
    assert row['ordinal'] == ordinal
    scalar = {(z['model'], z['parent']): z for z in read(BASE/'SCALAR_ROWS.json') if z['direction'] == direction}
    predictions = {m: read(BASE/direction/m/'predictions'/f'{ordinal:05}.json') for m in MODELS}
    assert all(scalar[m, ordinal]['tIoU'] > .5 and scalar[m, ordinal]['sIoU'] <= .5 for m in MODELS)
    truth, spans, provenance = read_truths(direction, roster)
    gt = truth[ordinal]; span = spans[ordinal]
    pixels, ids = decode(row['input'])
    assert ids == row['frame_ids']
    digest = hashlib.sha256(pixels.tobytes()).hexdigest()
    common = read(BASE/direction/'common_pixels'/f'{ordinal:05}.json')
    assert digest == common['pixel_sha256'] and all(z['pixel_sha256'] == digest for z in predictions.values())
    available = [fid for fid in ids if fid in gt and span[0] <= fid < span[1]
        and all(z['interval'][0] <= fid < z['interval'][1] for z in predictions.values())]
    assert len(available) >= 3
    chosen = [available[0], available[len(available)//2], available[-1]]
    assert len(set(chosen)) == 3
    folder = BASE/'private_frames'; folder.mkdir(exist_ok=True)
    w, h = row['input']['width'], row['input']['height']
    frames = []; models = {m: dict(interval=z['interval'], boxes=[], tIoU=scalar[m, ordinal]['tIoU'],
        sIoU=scalar[m, ordinal]['sIoU']) for m, z in predictions.items()}
    for fid in chosen:
        image = pixels[ids.index(fid)]
        p = folder/f'frame_{fid:05}.png'; assert not p.exists(); Image.fromarray(image).save(p)
        frames.append(dict(frame_id=fid, path=str(p.relative_to(BASE)), sha256=sha(p),
            pixel_sha256=hashlib.sha256(image.tobytes()).hexdigest(), GT_box=list(map(float, gt[fid]))))
        for m, z in predictions.items():
            boxes = np.asarray(z['boxes'], dtype=float) * [w, h, w, h]
            if direction == 'vidstg_to_hc2': boxes = np.maximum(boxes, 0)
            models[m]['boxes'].append([float(np.interp(fid, z['box_frame_ids'], boxes[:, j])) for j in range(4)])
    pins = {str((BASE/direction/m/'predictions'/f'{ordinal:05}.json').relative_to(ROOT)):
        sha(BASE/direction/m/'predictions'/f'{ordinal:05}.json') for m in MODELS}
    pins.update(provenance)
    record = dict(direction=direction, ordinal=ordinal, parent_source=row['source'], query=row['input']['caption'],
        width=w, height=h, fps=row['input']['fps'], input_frame_extent=[ids[0], ids[-1]],
        GT_interval=span, frames=frames, models=models, input_pins=pins, canonical_RGB_sha256=digest,
        selection='post-statistics illustrative common T+/S- case with multiple children; ordinal 42; not a frequency estimate',
        frame_selection='first, middle, last shared sampled frames inside both predictions and the annotated event',
        real_model_outputs_not_generated=True, new_model_calls=0, active_OPD_P1_payload_access=False,
        local_only_private_media_query_GT_overlay=True, time=time.time())
    write(BASE/'PRIVATE_CASE.json', record)
    print(json.dumps(dict(status='actual_case_bound', direction=direction, ordinal=ordinal,
        frame_ids=chosen, native_scores={m:{k:models[m][k] for k in ['tIoU','sIoU']} for m in MODELS})))


if __name__ == '__main__':
    prepare()
