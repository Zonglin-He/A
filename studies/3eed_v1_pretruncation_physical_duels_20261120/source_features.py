"""Recover matched physical alternatives on Waymo before training the small readout."""
from time import perf_counter

from common import (HERE, PREV, WORK, affine, anon, archive_npz, atomic, capture,
                    digest, engine, fixed_prior, np, pin, read, setup, torch)
from diagnose_source import natural_anchors
from features import observation
from paired import many_iou
from scope import canonical, source_rows
from source_pairs import catalog, source_identity
from roles import inventory


def ownership(iou, identities):
    hits = (iou[:,identities] > .25) & ((iou > .25).sum(1) == 1)[:,None]
    slot = np.where(hits, iou[:,identities], -1).argmax(0)
    slot[~hits.any(0)] = -1
    return hits.T, slot


def native_boxes(ev):
    return ev['subject_boxes'][0].cpu().numpy().copy()


def common_nominations(first, second):
    # Each sentence gets its turn at each rank. Suppress near-duplicates in physical space.
    ordered = np.stack((first, second),axis=1).reshape(-1,7)
    keep = []
    for i, box in enumerate(ordered):
        if not keep or (many_iou(ordered[keep],box) <= .35).all():
            keep.append(i)
            if len(keep) == 4:
                break
    if len(keep) < 4:
        for i, box in enumerate(ordered):
            if all(not np.array_equal(box,ordered[k]) for k in keep):
                keep.append(i)
                if len(keep) == 4:
                    break
    if len(keep) < 4:
        keep.extend(i for i in range(len(ordered)) if i not in keep)
    return ordered[keep[:4]]


def pair_bank(pool, endpoints, native, score):
    selected = list(endpoints)
    assert len(set(selected)) == 2
    extras = []
    for box in native:
        idx = np.flatnonzero((pool == box).all(-1))
        if len(idx):
            extras.append(int(idx[0]))
    order = np.argsort(-score,kind='stable').tolist()
    for idx in extras + order:
        if idx not in selected and (many_iou(pool[selected],pool[idx]) <= .35).all():
            selected.append(idx)
            if len(selected) == 4:
                break
    if len(selected) < 4:
        selected.extend(i for i in order if i not in selected)
    chosen = np.array(sorted(selected[:4],key=lambda i:(-float(score[i]),int(i))))
    return pool[chosen], chosen


def reencode(model, boxes, by_sentence, pair):
    hs, js, supports = [], [], []
    for index in pair:
        scene, data = by_sentence[index]
        prior, known = fixed_prior(boxes,scene)
        dev = data['subject_boxes'].device
        ev = dict(data, subject_boxes=torch.as_tensor(boxes[None],device=dev,dtype=torch.float32),
                  base_logit=torch.as_tensor(prior[None],device=dev))
        h,j = capture(model._decoder,ev)
        hs.append(h)
        js.append(j)
        supports.append(known)
    return dict(h=np.stack(hs),j=np.stack(js),votes=np.stack(supports))


@torch.inference_mode()
def run():
    setup()
    model = engine()
    w0 = affine(model).numpy()
    np.testing.assert_array_equal(w0,np.load(PREV/'run_PRIVATE/initial.npz')['head'])
    splits = {s:source_rows('waymo',s) for s in ('fit','calibration')}
    for field in ('recording','point_sha256'):
        assert not ({r[field] for r in splits['fit']} & {r[field] for r in splits['calibration']})
    pin(WORK/'SOURCE_INPUTS.json',dict(code=digest(HERE/'source_features.py'),common=digest(HERE/'common.py'),
        config=digest(HERE/'PROTOCOL.json'),previous=digest(PREV/'run_PRIVATE/SOURCE_SEAL.json')))
    totals, native_cal = {}, []
    start = perf_counter()
    for split,rows in splits.items():
        records = []
        for n,(row,(pool,anchor_gt)) in enumerate(zip(rows,natural_anchors(split),strict=True)):
            objects,pairs = catalog(row)
            if not pairs and split != 'calibration':
                continue
            frame = observation(row)
            original = int(row['object_index'])
            assert canonical(objects[original]['caption']) == frame['text']
            scene,ev = model._read_observation(frame)
            native = native_boxes(ev)
            if split == 'calibration':
                h,j = capture(model._decoder,ev)
                native_cal.append(dict(h=h,j=j))
            if not pairs:
                continue
            np.testing.assert_array_equal(pool[:,:3],scene.boxes[:,:3])
            assert all(np.any((pool == b).all(-1)) for b in native)
            annotated,_ = inventory(row)
            np.testing.assert_array_equal(anchor_gt,annotated[0])
            full,inv = source_identity(row,objects,pool)
            prior,_ = fixed_prior(pool,scene)
            by_sentence = {original:(scene,ev)}
            for i in sorted({i for p in pairs for i in p} - {original}):
                by_sentence[i] = model._read_observation(dict(frame,text=canonical(objects[i]['caption'])))
                np.testing.assert_array_equal(scene.sampled_points,by_sentence[i][0].sampled_points)
            for a,b in pairs:
                identity = inv[[a,b]]
                assert identity[0] != identity[1]
                _, endpoints = ownership(full,identity)
                if (endpoints >= 0).all():
                    shared,indices = pair_bank(pool,endpoints,native,prior)
                    hits,slots = ownership(full[indices],identity)
                    # Guarantee supervision uses precisely the original physical representative.
                    slots = np.array([np.where(indices == i)[0][0] for i in endpoints])
                    assert hits[0,slots[0]] and hits[1,slots[1]]
                else:
                    shared = native
                    measured,_ = source_identity(row,objects,shared)
                    hits,slots = ownership(measured,identity)
                rec = dict(key=anon('pair',row['point_sha256'],a,b),frame_key=anon('scene',row['point_sha256']),
                           rec_key=anon('rec',row['recording']),category=objects[a]['class'].lower(),
                           slots=slots,hits=hits,available=(endpoints>=0),primary=original in (a,b),
                           **reencode(model,shared,by_sentence,(a,b)))
                left,right = sorted((a,b),key=lambda i:canonical(objects[i]['caption']))
                free = common_nominations(native_boxes(by_sentence[left][1]),native_boxes(by_sentence[right][1]))
                mat,_ = source_identity(row,objects,free)
                fhits,fslots = ownership(mat,identity)
                rec.update(free_hits=fhits,free_slots=fslots)
                rec.update({'free_'+k:v for k,v in reencode(model,free,by_sentence,(a,b)).items()})
                records.append(rec)
            if n%40 == 0:
                print(split,n,len(records),round(perf_counter()-start,1),flush=True)
        assert len(records) == {'fit':213,'calibration':44}[split]
        path = WORK/(split+'.npz')
        arrays = {k:np.stack([r[k] for r in records]) for k in records[0]}
        archive_npz(path,**arrays)
        tot = int(arrays['available'].all(1).sum())
        assert tot == {'fit':181,'calibration':41}[split]
        totals[split] = dict(hash=digest(path),potential=len(records),physical=tot,
                             nominees=int((arrays['free_slots']>=0).all(1).sum()))
        print(split,totals[split],flush=True)
    assert len(native_cal) == 160
    archive_npz(WORK/'native_cal.npz', **{k:np.stack([r[k] for r in native_cal]) for k in ('h','j')})
    atomic(WORK/'SOURCE_SEAL.json',dict(splits=totals,native_cal=digest(WORK/'native_cal.npz'),
                                inputs=digest(WORK/'SOURCE_INPUTS.json'),elapsed=perf_counter()-start))


if __name__ == '__main__':
    run()
