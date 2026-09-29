"""Two independent frozen evidence providers. No source labels are opened."""
import argparse
import gc
import hashlib
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.desta_native_common import D, DINO, SAM, read, write, sha, stage, complete, verified, check_pins
from vg_tta.desta3d_v3_oracle_io import allocation


def temporal(name):
    repo = ROOT / 'external/UniversalVTG'
    dest = stage(name, [Path(__file__), D/'EXPERT_ASSETS.json', repo/'universal_vtg_inference.py'])
    with allocation(dest) as (cfg, guard):
        import numpy as np
        import torch
        from PIL import Image
        from vg_tta.exact_frame_decode_audit_v2 import decode
        sys.path[:0] = [str(repo), str(repo/'perception_models')]
        from universal_vtg_inference import UniversalVTG
        torch.set_num_threads(4); torch.manual_seed(cfg['seed'])
        check_pins(read(D/'EXPERT_ASSETS.json')['pins'])
        expert = UniversalVTG(experiment_name=str(ROOT/'checkpoints/universalvtg'), device='cuda', enable_query_unifier=False)
        expert._ensure_video_encoder(); expert._ensure_text_encoder()
        for m in [expert.model, expert._video_extractor, expert._text_model]:
            m.eval().requires_grad_(False)
        rows = read(D/'DEV64.json')
        for i, row in enumerate(rows):
            ep = D/'experts/temporal'/f'{i:02}'
            if verified(ep): continue
            guard(); frames, ids = decode(row['input']); start = time.monotonic()
            duration = (ids[-1] - ids[0] + 1) / row['input']['fps']
            slots = np.linspace(ids[0], ids[-1], max(1, int(duration*2)))
            pick = np.abs(np.asarray(ids)[None, :] - slots[:, None]).argmin(1)
            feat = []
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
                for at in range(0, len(frames), 16):
                    pixels = torch.stack([expert._video_preprocess(Image.fromarray(f)) for f in frames[at:at+16]]).cuda().half()
                    feat.append(expert._video_extractor(pixels).float().cpu())
            features = torch.cat(feat)[pick].T.contiguous()
            text = expert.encode_text(row['input']['caption'])
            with torch.inference_mode():
                result = expert.predict(features, text, fps=None, feature_fps=2., duration=duration, use_unifier=False)
            segments = result['segments'][0].float().cpu().numpy()
            scores = result['scores'][0].float().cpu().numpy()
            valid = [j for j, s in enumerate(segments) if np.isfinite(s).all() and s[1]>=s[0] and np.isfinite(scores[j])]
            winner = max(valid, key=lambda j: (float(scores[j]), -j)) if valid else None
            interval = (segments[winner] * row['input']['fps'] + ids[0]).tolist() if winner is not None else None
            write(ep/'EVIDENCE.json', dict(key=row['key'], frame_ids=ids, pixel_sha256=hashlib.sha256(frames.tobytes()).hexdigest(),
                segments_seconds=segments.tolist(), scores=scores.tolist(), selected=winner, interval_physical=interval,
                slots_physical=slots.tolist(), nearest_observation_indices=pick.tolist(), duration=duration,
                GT_read=False, label_free=True, seconds=time.monotonic()-start))
            torch.save({'video_features': features, 'text_features': text.cpu()}, ep/'FEATURES.pt')
            complete(ep, index=i, GT_read=False)
            print('TEMPORAL_COMPLETE', i+1, 64, 'seconds', time.monotonic()-start, flush=True)
            del frames, feat, features, text, result, pixels
            gc.collect(); torch.cuda.empty_cache()
        complete(D/'experts/temporal', queries=64, GT_read=False)
        write(dest/'COMPLETE.json', {'queries':64})


def spatial(name):
    dest = stage(name, [Path(__file__), D/'EXPERT_ASSETS.json'])
    with allocation(dest) as (cfg, guard):
        import numpy as np
        import torch
        from PIL import Image
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        from vg_tta.exact_frame_decode_audit_v2 import decode
        sys.path.insert(0, str(ROOT/'external/sam2'))
        from sam2.build_sam import build_sam2_video_predictor
        torch.set_num_threads(4); torch.manual_seed(cfg['seed'])
        check_pins(read(D/'EXPERT_ASSETS.json')['pins'])
        processor = AutoProcessor.from_pretrained(str(DINO), local_files_only=True)
        detector = AutoModelForZeroShotObjectDetection.from_pretrained(str(DINO), local_files_only=True).cuda().eval().requires_grad_(False)
        tracker = build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_l.yaml', str(SAM), device='cuda').eval().requires_grad_(False)
        rows = read(D/'DEV64.json')
        for i, row in enumerate(rows):
            ep = D/'experts/spatial'/f'{i:02}'
            if verified(ep): continue
            guard(); frames, ids = decode(row['input']); start = time.monotonic()
            detections = []; best = None
            text = row['input']['caption'].lower().strip().rstrip('.') + '.'
            for j, frame in enumerate(frames):
                image = Image.fromarray(frame)
                batch = processor(images=image, text=text, return_tensors='pt').to('cuda')
                with torch.inference_mode(): result = detector(**batch)
                parsed = processor.post_process_grounded_object_detection(result, batch.input_ids,
                    threshold=cfg['dino_box_threshold'], text_threshold=cfg['dino_text_threshold'], target_sizes=[image.size[::-1]])[0]
                boxes = parsed['boxes'].float().cpu().tolist(); scores = parsed['scores'].float().cpu().tolist()
                labels = parsed.get('text_labels', parsed.get('labels', []))
                detections.append(dict(position=j, physical=ids[j], boxes=boxes, scores=scores, labels=labels))
                for k, (box, score) in enumerate(zip(boxes, scores)):
                    valid = np.isfinite(box).all() and box[2]>box[0] and box[3]>box[1]
                    if valid and (best is None or score > best['score']):
                        best = dict(position=j, candidate=k, box=box, score=score)
                del batch, result, parsed
            # Raw detector evidence is preserved before tracking starts.
            write(ep/'DETECTIONS.json', dict(text=text, frames=detections, selected=best))
            boxes_by_frame = {}; mask_records = {}
            if best is not None:
                frame_dir = ep/'lossless_frames'; frame_dir.mkdir()
                for j, frame in enumerate(frames):
                    # SAM2's loader filters .jpg but Pillow detects PNG bytes losslessly.
                    Image.fromarray(frame).save(frame_dir/f'{j:05}.jpg', format='PNG')
                with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                    state = tracker.init_state(str(frame_dir), offload_video_to_cpu=True, offload_state_to_cpu=True)
                    tracker.add_new_points_or_box(state, frame_idx=best['position'], obj_id=1, box=np.asarray(best['box'], np.float32))
                    for reverse in (False, True):
                        for j, obj_ids, logits in tracker.propagate_in_video(state, start_frame_idx=best['position'], reverse=reverse):
                            if str(ids[j]) in mask_records: continue
                            mask = (logits[0,0] > cfg['mask_threshold']).cpu().numpy()
                            ys, xs = np.where(mask); h, w = mask.shape
                            box = [float(xs.min()/w), float(ys.min()/h), float((xs.max()+1)/w), float((ys.max()+1)/h)] if len(xs) else None
                            boxes_by_frame[str(ids[j])] = box
                            packed = np.packbits(mask.reshape(-1))
                            mask_records[str(ids[j])] = dict(shape=[h,w], packed=packed, box=box)
                torch.save(mask_records, ep/'MASKS.pt')
                # Materialized frames are retained as physical-input evidence.
                del state, logits, mask, packed
            write(ep/'EVIDENCE.json', dict(key=row['key'], frame_ids=ids,
                pixel_sha256=hashlib.sha256(frames.tobytes()).hexdigest(), boxes_by_frame=boxes_by_frame,
                anchor=best, missing_count=len(ids)-sum(b is not None for b in boxes_by_frame.values()),
                GT_read=False, label_free=True, seconds=time.monotonic()-start))
            # Seal nested input frames too.
            write(ep/'FRAME_SEAL.json', {str(p.relative_to(ep)):sha(p) for p in ep.rglob('*.jpg')})
            complete(ep, index=i, GT_read=False)
            print('SPATIAL_COMPLETE', i+1, 64, 'seconds', time.monotonic()-start, flush=True)
            del frames, mask_records, detections
            gc.collect(); torch.cuda.empty_cache()
        complete(D/'experts/spatial', queries=64, GT_read=False)
        write(dest/'COMPLETE.json', {'queries':64})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('expert', choices=['temporal','spatial']); p.add_argument('--run', required=True)
    args = p.parse_args()
    globals()[args.expert](args.run)
