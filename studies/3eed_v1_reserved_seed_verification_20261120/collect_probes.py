"""Run real frozen 3EED measurements. This entry point never imports a GT scorer."""
import argparse
import hashlib
import os
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CAMERA = HERE.parent / '3eed_v1_camera_word_replacement_20261120'
sys.path.insert(0, str(CAMERA))
from calibrate import digest, read, save
from predict import archive_npz, frame, inputs
from crossplatform.ground import Grounder
from crossplatform.candidate_frontend import WAYMO_VIEWS, sensor_frame_box
from crossplatform.round7.camera_projection import raw_to_model_points
from probes import ARMS, NATIVE, cpu, decode, evidence, terminal_boxes, word_probability

WORK = HERE / 'run_PRIVATE'
PUBLIC = HERE / 'results'
CONFIG = read(HERE / 'PROTOCOL.json')


def native_forward(model, inp):
    from model51 import SENTINEL, firstnoun_mask
    from runner_helpers import sample_points
    from utils.transform_waymo import transform_to_front_view
    sampled, _, _ = sample_points(dict(platform=inp['target'], seed=inp['seed'], pointpath=inp['points']))
    if inp['target'] == 'waymo':
        sampled, _ = transform_to_front_view(sampled, np.empty((0, 7)), inp['waymo_view'])
    text = inp['text'].strip() + SENTINEL
    device = next(model.parameters()).device
    mask = firstnoun_mask(model, text)[0].to(device)
    points = torch.as_tensor(np.ascontiguousarray(sampled, dtype=np.float32), device=device)
    raw = model(dict(point_clouds=points[None], text=[text]))
    if raw['text_attention_mask'].any() or raw['fp2_xyz'].shape != (1, 1024, 3):
        raise ValueError('Unexpected native token or seed convention')
    return raw, mask


def checked_anchors(native, sensor_boxes, inp):
    # The PCA production head preserves native centers and dimensions, flooring invalid sizes only.
    anchors = sensor_boxes[:, :6].copy()
    anchors[:, :3] = raw_to_model_points(sensor_boxes[:, :3], inp['target'], WAYMO_VIEWS.index(inp['waymo_view']))
    actual = native.copy().astype(np.float64)
    actual[:, 3:] = np.maximum(actual[:, 3:], np.float32(1e-6))
    distance = abs(actual[None] - anchors[:, None]).max(axis=-1)
    nearest = distance.argmin(axis=-1)
    error = distance[np.arange(len(anchors)), nearest].max()
    if error > 2e-5:
        raise ValueError(f'Frozen slate does not match recomputed native boxes: {error}')
    return native[nearest].astype(np.float64), float(error)


def fresh_front(grounder, inp):
    scene, ev = grounder._read_observation(inp)
    scores = cpu(grounder._decoder(ev)['subject_logits'][0]).astype(np.float64)
    anchors = cpu(ev['subject_boxes'][0])
    boxes = np.stack([sensor_frame_box(b, scene.target, scene.view_index) for b in anchors])
    return scores, boxes


@torch.inference_mode()
def observe(grounder, inp, base, boxes, key, check):
    if check:
        fresh_scores, fresh_boxes = fresh_front(grounder, inp)
        np.testing.assert_array_equal(fresh_scores, base)
        np.testing.assert_array_equal(fresh_boxes, boxes)
    model = grounder._frontend.base
    raw, mask = native_forward(model, inp)
    native_ids = raw['query_points_sample_inds'][0].long()
    gate = raw['seeds_obj_cls_logits'].sigmoid()[0, 0]
    torch.testing.assert_close(native_ids, gate.topk(NATIVE).indices, atol=0, rtol=0)
    torch.testing.assert_close(raw['fp2_xyz'][0, native_ids], raw['query_points_xyz'][0], atol=0, rtol=0)
    keep = torch.ones(len(gate), dtype=torch.bool, device=native_ids.device)
    keep[native_ids] = False
    reserve_ids = keep.nonzero().flatten()
    ids = torch.cat([native_ids, reserve_ids])
    assert len(reserve_ids) == 768 and len(ids.unique()) == 1024
    original = (cpu(terminal_boxes(raw)), cpu(word_probability(raw['last_proj_queries'], raw['proj_tokens'], mask)))
    anchors, anchor_error = checked_anchors(original[0], boxes, inp)
    elapsed = []
    results = []
    for isolated in (True, False):
        torch.cuda.synchronize()
        start = perf_counter()
        geom, probabilities, early = decode(model, raw, ids, mask, isolated)
        torch.cuda.synchronize()
        elapsed.append(perf_counter()-start)
        results.append((geom, probabilities))
    private, opened = results
    native_errors = [np.abs(result[k][:NATIVE]-original[k]).max() for result in results for k in (0,1)]
    # Native float32 batched layers can use different reductions at the expanded shape.
    if native_errors[0] > .002 or native_errors[1] > 1e-4:
        raise ValueError(f'Isolated context drift exceeds budget: {native_errors}')
    single_error = [-1., -1.]
    if check:
        j = int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], 'big') % len(reserve_ids)
        single_ids = torch.cat([native_ids, reserve_ids[j:j+1]])
        one_box, one_p, _ = decode(model, raw, single_ids, mask, isolated=True)
        single_error = [float(np.abs(a[-1]-b[NATIVE+j]).max()) for a,b in zip((one_box,one_p), private)]
        if single_error[0] > .002 or single_error[1] > 1e-4:
            raise ValueError(f'Each probe should equal its singleton calculation: {single_error}')
    pr = tuple(x[NATIVE:] for x in private)
    op = tuple(x[NATIVE:] for x in opened)
    logits, totals, valid = evidence(anchors, original, pr, op, early, base, key)
    if not np.isfinite(totals).all() or not np.isfinite(logits).all():
        raise ValueError('Non-finite support or logits')
    return dict(base=base, boxes=boxes, choices=logits.argmax(axis=1).astype(np.int8), logits=logits,
                totals=totals, valid=valid, native_error=native_errors, single_error=single_error,
                anchor_error=anchor_error, seconds=elapsed,
                reserve_center_shift=np.linalg.norm(pr[0][:,:3]-op[0][:,:3],axis=1).mean(),
                reserve_word_change=np.abs(pr[1]-op[1]).mean())


def roster_id(rows):
    return hashlib.sha256('\n'.join(f"{r['pointpath']}|{r['metapath']}|{r['caption']}|{r.get('seed',0)}"
                                    for r in rows).encode()).hexdigest()


def run(stage, smoke=False):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    roster = inputs(stage)
    archived = CAMERA / 'run_PRIVATE' / stage
    historical = None
    if stage == 'ordinary':
        if not read(PUBLIC/'pilot_decision.json')['expand']:
            raise ValueError('The registered expansion condition was not met')
    else:
        assert digest(archived/'SEALED.json') == CONFIG[stage+'_seal']
        historical = read(archived/'SEALED.json')
        assert historical['lock'] == digest(archived/'LOCK.json') and not historical['task_labels_opened']
        old_roster = read(archived/'LOCK.json')['roster']
    assert tuple(CONFIG['arms']) == ARMS and digest(ROOT/'crossplatform/single_frame_sources.json') == CONFIG['asset_register']
    work = WORK/stage
    work.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock = dict(stage=stage, assets=CONFIG['asset_register'], arms=ARMS,
                code={n:digest(HERE/n) for n in ('PROTOCOL.json','collect_probes.py','probes.py')},
                roster={f'{s}_to_{t}':roster_id(rows) for (s,t),rows in roster.items()})
    lock['arms'] = list(lock['arms'])
    if historical:
        assert lock['roster'] == old_roster
    if not smoke:
        if (work/'LOCK.json').exists():
            assert lock == read(work/'LOCK.json')
        else:
            save(work/'LOCK.json',lock)
    model = None
    loaded = None
    receipts = {}
    start_time = perf_counter()
    for (src,tgt),rows in roster.items():
        direction = f'{src}_to_{tgt}'
        group = work/direction
        group.mkdir(exist_ok=True)
        for start in range(0,len(rows),64):
            if smoke and start:
                break
            name = f'{direction}/{start:05d}.npz'
            dest = work/name
            receipt = dest.with_suffix('.json')
            batch = rows[start:start+64]
            if smoke or not receipt.exists():
                if loaded != src:
                    if model is not None:
                        del model
                        torch.cuda.empty_cache()
                    model = Grounder(src)
                    loaded = src
                    assert not any(m.training for m in model._frontend.base.modules())
                if historical:
                    assert digest(archived/name) == historical['files'][name]['sha256']
                    with np.load(archived/name,allow_pickle=False) as old:
                        base, boxes = old['base'], old['boxes']
                records = []
                for j,row in enumerate(batch[:1] if smoke else batch):
                    inp=frame(row)
                    with torch.inference_mode():
                        logits, candidates = (base[j],boxes[j]) if historical else fresh_front(model,inp)
                        obs = observe(model,inp,logits,candidates,f'{direction}|{start+j}', j==0)
                    records.append(obs)
                if smoke:
                    first = records[0]
                    print(direction, 'geometry/probability drift private then open', first['native_error'],
                          'singleton', first['single_error'], 'seconds', first['seconds'], flush=True)
                    continue
                values = {key:np.stack([r[key] for r in records]) for key in records[0]}
                archive_npz(dest,**values)
                save(receipt,dict(n=len(batch),sha256=digest(dest),lock=digest(work/'LOCK.json')))
            if not smoke:
                control = read(receipt)
                assert control['n']==len(batch) and control['sha256']==digest(dest) and control['lock']==digest(work/'LOCK.json')
                receipts[name] = control
                print(stage,direction,start+len(batch),'/',len(rows),round(perf_counter()-start_time,1),flush=True)
    if not smoke:
        total = sum(r['n'] for r in receipts.values())
        assert total == {'source':384, 'pilot':3072,'ordinary':18620}[stage]
        save(work/'SEALED.json',dict(n=total,files=receipts,lock=digest(work/'LOCK.json'),task_labels_opened=False,
                                    FINAL_access=False,seconds=perf_counter()-start_time))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=('source','pilot','ordinary'))
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    run(args.stage,args.smoke)
