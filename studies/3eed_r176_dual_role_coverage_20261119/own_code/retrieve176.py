"""One prediction-only replacement of the original R175 anchor diversity rule."""
import gc
import time

import numpy as np
import torch

from scope176 import (HERE, N, ORDER, ORIGINAL, PLATFORMS, PRED_ARMS, PRIOR,
                      PRIVATE, R165_DATA, array_sha, json_atomic, npz_atomic,
                      ordinary_rows, points_for, read, sha, state)
from predict175 import BATCH, encoder, evidence_batch, eval_lock
from model175 import RelationalDecoder
from crossplatform.ground_single_frame import orient_box

OUTPUT = PRIVATE/'retrieval_predictions'
NEW_SPEC = HERE/'retrieval_protocol.json'


def overlap_anchors(prior, scores, seeds):
    """Keep native scores and four slots, penalizing overlapping support."""
    lo = prior[..., :3] - prior[..., 3:]/2
    hi = prior[..., :3] + prior[..., 3:]/2
    volume = prior[..., 3:].prod(-1)
    support = ((seeds[:, None] >= lo[:, :, None]) &
               (seeds[:, None] <= hi[:, :, None])).all(-1).float()
    size = support.sum(-1)
    ranks = scores.clone()
    indices = []
    batch = torch.arange(len(prior), device=prior.device)[:, None]
    for _ in range(4):
        chosen = ranks.argmax(-1)
        indices.append(chosen)
        intersection = (torch.minimum(hi[:, None], hi[batch, chosen, None]) -
                        torch.maximum(lo[:, None], lo[batch, chosen, None])).clamp_min(0).prod(-1)
        box_iou = intersection/(volume[:, None] + volume[batch, chosen, None] - intersection)
        and_support = support[batch, chosen] @ support.transpose(1,2)
        picked_size = size[batch, chosen, None]
        jaccard = and_support/(size[:, None] + picked_size - and_support).clamp_min(1)
        jaccard = torch.where((size[:, None] >= 4) & (picked_size >= 4), jaccard, 0.)
        ranks = ranks - 4.*torch.maximum(box_iou, jaccard)
        ranks.scatter_(-1, chosen[..., None], -torch.inf)
    return torch.stack(indices, -1)


def lock():
    gate = read(PRIVATE/'RETRIEVAL_GATE.json')
    assert gate['retrieval_is_largest'] and not gate['FINAL_access']
    previous = eval_lock()
    registered = read(PRIOR/'EVALUATION_LOCK.json')
    code = ('retrieve176.py', 'scope176.py', 'retrieval_protocol.json')
    value = dict(anchors_per_role=4, gate=sha(PRIVATE/'RETRIEVAL_GATE.json'),
                 code={p:sha(HERE/p) for p in code}, r175_eval_lock=previous,
                 original_predictions=sha(ORIGINAL/'PREDICTION_SEAL.json'),
                 weights={s:registered['models'][f'{s}/pair_only'] for s in PLATFORMS},
                 n=N, FINAL_access=False, target_task_label_reads=0, target_updates=0)
    out = PRIVATE/'RETRIEVAL_LOCK.json'
    if out.exists():
        assert read(out) == value
    else:
        json_atomic(out, value)
    return sha(out)


def model_for(source):
    model = RelationalDecoder('pair_only')
    checkpoint = PRIOR/source/'pair_only.pth'
    assert sha(checkpoint) == read(PRIVATE/'RETRIEVAL_LOCK.json')['weights'][source]
    model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location='cpu'))
    return model.cuda().eval().requires_grad_(False)


@torch.inference_mode()
def predict_block(source, target, bank, old, loaded, model, first):
    groups = []
    native_arm = PRED_ARMS.index('pair_only')
    rows = ordinary_rows(target)
    for start in range(0, len(bank['ordinals']), BATCH):
        stop = min(start+BATCH, len(bank['ordinals']))
        chunk = slice(start,stop)
        f, pca = evidence_batch(source,target,bank,loaded,start,stop,first and start==0)
        original_ids = f['anchors'].cpu().numpy()
        np.testing.assert_array_equal(original_ids, old['anchor_ids'][chunk])
        np.testing.assert_array_equal(f['xyz'].cpu().numpy(), old['xyz'][chunk])
        np.testing.assert_array_equal(pca, old['slot_boxes'][chunk])
        np.testing.assert_array_equal(f['relation'].cpu().numpy(), old['relation'][chunk])
        control = model(f)['choices'].cpu().numpy()
        np.testing.assert_array_equal(control,old['choice'][chunk,native_arm])
        new = overlap_anchors(f['prior'], f['prior_scores'], f['xyz'])
        new_ids = new.cpu().numpy()
        np.testing.assert_array_equal(original_ids[...,0], new_ids[...,0])
        assert np.all(np.diff(np.sort(new_ids, axis=-1), axis=-1) > 0)
        direct = model(dict(f, anchors=new))
        choice = direct['choices'].cpu().numpy().astype(np.int8)
        slots = np.empty_like(pca)
        for j, i in enumerate(range(start,stop)):
            known = dict(zip(original_ids[j].ravel(),pca[j].reshape(-1,7),strict=True))
            diff = set(new_ids[j].ravel()) - known.keys()
            if diff:
                sample, ids = points_for(rows[int(bank['ordinals'][i])])
                assert array_sha(sample)==bank['points_sha'][i].tobytes()
                assert array_sha(ids)==bank['indices_sha'][i].tobytes()
                known.update({q:orient_box(sample,bank['raw'][i,q]) for q in sorted(diff)})
            slots[j] = np.stack([known[k] for k in new_ids[j].ravel()]).reshape(2,4,7)
        raw = bank['raw'][chunk][np.arange(len(choice))[:,None,None],new_ids]
        np.testing.assert_array_equal(slots[...,:3],raw[...,:3])
        np.testing.assert_array_equal(slots[...,3:6],raw[...,3:6].clip(min=1e-6))
        unchanged = (new_ids == original_ids).all(axis=(1,2))
        np.testing.assert_array_equal(choice[unchanged],control[unchanged])
        boxes = slots[np.arange(len(choice))[:,None],np.arange(2)[None],choice]
        groups.append(dict(anchors=new_ids, pca=slots, boxes=boxes, choice=choice,
                           relation=f['relation'].cpu().numpy()))
    return {k:np.concatenate([p[k] for p in groups]) for k in groups[0]}


def direction(source, target, loaded, model, pinned):
    route = f'{source}_to_{target}'
    inputs = R165_DATA/route
    r175 = ORIGINAL/route
    src, old = read(inputs/'SEAL.json'), read(r175/'SEAL.json')
    old_global = read(ORIGINAL/'PREDICTION_SEAL.json')
    assert old_global['directions'][route]['seal']==sha(r175/'SEAL.json')
    assert sha(inputs/'SEAL.json')==old['archive'] and src['n']==old['n']
    assert not src['protected_payload_access'] and not src['oracle_target_class_read']
    out = OUTPUT/route
    out.mkdir(parents=True,exist_ok=True,mode=0o700)
    offset, files = 0, {}
    started = time.monotonic()
    for name, digest in sorted(src['files'].items()):
        p = inputs/name
        old_path = r175/name
        assert sha(p)==digest and sha(old_path)==old['files'][name]
        dest = out/name
        if not dest.exists():
            with np.load(p,allow_pickle=False) as bank, np.load(old_path,allow_pickle=False) as before:
                cols = ('ordinals','offsets','tokens','q','raw','points_sha','indices_sha')
                rows = {k:bank[k] for k in cols}
                data = {k:before[k] for k in ('anchor_ids','choice','xyz','slot_boxes','relation')}
            pred = predict_block(source,target,rows,data,loaded,model,offset==0)
            npz_atomic(dest,**pred,ordinals=rows['ordinals'],source_block=digest,
                       original_block=old['files'][name],lock=pinned)
        with np.load(dest,allow_pickle=False) as saved:
            assert str(saved['lock'])==pinned and str(saved['source_block'])==digest
            assert str(saved['original_block'])==old['files'][name]
            idx = saved['ordinals']
            np.testing.assert_array_equal(idx,np.arange(offset,offset+len(idx)))
            offset += len(idx)
        files[name] = sha(dest)
        print('R176_RETRIEVAL',route,offset,src['n'],round(time.monotonic()-started,1),flush=True)
    assert offset == src['n'] == len(ordinary_rows(target))
    stamp = out/'SEAL.json'
    json_atomic(stamp,dict(n=offset,files=files,lock=pinned,r175_seal=sha(r175/'SEAL.json'),
                          target_task_label_reads=0,target_updates=0,FINAL_access=False))
    return dict(n=offset,seal=sha(stamp))


def verify():
    stamp = read(OUTPUT/'PREDICTION_SEAL.json')
    assert stamp['lock']==lock() and stamp['n']==N and not stamp['FINAL_access']
    assert not stamp['target_task_label_reads'] and not stamp['target_updates']
    assert set(stamp['directions']) == {f'{s}_to_{t}' for s,t in ORDER}
    for route, item in stamp['directions'].items():
        p = OUTPUT/route/'SEAL.json'
        assert sha(p)==item['seal'] and read(p)['n']==item['n']
        assert all(sha(p.parent/name)==digest for name,digest in read(p)['files'].items())
    return stamp


def main():
    pinned = lock()
    state('frozen_retrieval', 'Same R175 pair scorer and four slots, one overlap retrieval policy')
    results = {}
    for source in PLATFORMS:
        loaded, model = encoder(source), model_for(source)
        for s, target in ORDER:
            if s == source:
                results[f'{source}_to_{target}'] = direction(source,target,loaded,model,pinned)
        del loaded, model
        gc.collect()
        torch.cuda.empty_cache()
    assert sum(v['n'] for v in results.values())==N
    json_atomic(OUTPUT/'PREDICTION_SEAL.json', dict(n=N,directions=results,lock=pinned,
                target_task_label_reads=0,target_updates=0,FINAL_access=False))
    print('R176_RETRIEVAL_SEALED',N,flush=True)


if __name__ == '__main__':
    main()
