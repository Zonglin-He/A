"""Capture source-only phrase and candidate reads for local physical identity."""
import argparse
from collections import Counter, defaultdict, deque
import fcntl
import gc
import hashlib
from pathlib import Path
from time import perf_counter

import numpy as np

from binding_collect import HERE, PLATFORMS, ROOT, SOURCE_SETTINGS, atomic_json, read, sha
from edges import point_support

DATA = HERE / 'private/appearance'
PROTOCOL = HERE / 'appearance_protocol.json'
MIN_SEEDS = 3
NUM_FRAMES = 180


def order(s):
    return hashlib.sha256(str(s).encode()).hexdigest()


def prepare_source():
    from calibrate168 import ordinary_content, sources

    dest = DATA / 'source'
    dest.mkdir(parents=True, mode=0o700, exist_ok=True)
    file = dest / 'ROSTER.json'
    check = dest / 'ROSTER_SEAL.json'
    assert not file.exists() and not check.exists()
    framekeys, imkeys, forbidden_frames, forbidden_recordings = ordinary_content()
    # R159 is the already registered ordinary exposure beyond the original ordinary pool.
    from scope170 import REC_INPUT, R159
    continued = read(R159 / 'RECORDING_INPUT_SEAL.json')
    assert sha(REC_INPUT) == continued['input_sha256'] and not continued['FINAL_access']
    extended_recordings = set()
    for row in read(REC_INPUT):
        if 'row' in row:
            row = row['row']
        for key, keys in [('point_sha256',framekeys),('image_sha256',imkeys)]:
            if key in row:
                keys.add(row[key])
        if 'metapath' in row:
            p = Path(row['metapath']).resolve()
            forbidden_frames.add(p)
            extended_recordings.add(p.parent.parent)
    result = []
    reused_extended = set()
    seen_frame, seen_points, seen_images = set(), set(),set()
    pins = {}
    counts = {}
    for platform in PLATFORMS:
        m = ROOT/'crossplatform/round161'/f'{platform}_train_PRIVATE.json'
        a = sources(platform)
        pins[platform] = dict(train=sha(m),registered=sha(a))
        own = {Path(x['metapath']).resolve() for x in read(m)}
        grouped = defaultdict(deque)
        split = Counter()
        for r in sorted(read(a),key=lambda r:order('candidate-appearance-v1|'+r['id'])):
            frame = Path(r['metapath']).resolve()
            assert frame in own and r['platform']==platform
            parent = frame.parent.parent
            if frame in forbidden_frames or parent in forbidden_recordings:
                split['ordinary_overlap'] += 1
                continue
            grouped[parent].append(r)
        pools = [v for _,v in sorted(grouped.items(),key=lambda t:order(t[0]))]
        pnum = 0
        while pnum<NUM_FRAMES and any(pools):
            for q in pools:
                if not q or pnum>=NUM_FRAMES:
                    continue
                r = q.popleft()
                path = Path(r['metapath']).resolve()
                if path in seen_frame:
                    split['duplicate_frame'] += 1
                    continue
                scan, cam = sha(Path(r['pointpath'])),sha(Path(r['imagepath']))
                if scan in framekeys or cam in imkeys:
                    split['ordinary_content'] += 1
                    continue
                if scan in seen_points or cam in seen_images:
                    split['duplicate_content'] += 1
                    continue
                seen_frame.add(path)
                seen_points.add(scan)
                seen_images.add(cam)
                if path.parent.parent in extended_recordings:
                    reused_extended.add(path.parent.parent)
                    split['historical_r159_recording'] += 1
                result.append(dict(case=len(result)+1,source=platform,row=r,group=order(path.parent.parent),
                                   scan=scan,cam=cam))
                pnum+=1
        split['selected']=pnum
        counts[platform]=dict(split)
    atomic_json(file,result)
    atomic_json(check,dict(roster=sha(file),source_manifests=pins,selection=counts,
                           protected_data_access=False,original_ordinary_recording_overlap=0,
                           r159_recordings_shared=len(reused_extended),validation_content_overlap=0))
    print('Registered source-only unique frames',counts,flush=True)


def normalized_rows(x):
    return x / np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)


def observations(ev, original):
    """Attach the original calibrated image reads to the same lidar seeds."""
    seed_ids = np.sort(np.unique(ev.seeds,axis=0,return_index=True)[1])
    idx = ev.p2i[seed_ids]
    w = ev.p2i_weight[seed_ids] * (idx >= 0)
    img = np.einsum('nj,njd->nd',w,ev.ionly[np.maximum(idx,0)])
    img /= np.maximum(w.sum(axis=1,keepdims=True),1e-12)
    point = original['vis0'][0,seed_ids]
    assert img.shape[1]==64 and point.shape[1]==288
    usable = ev.visible[seed_ids] & (w.sum(1)>0)
    return (ev.seeds[seed_ids], usable, normalized_rows(point), normalized_rows(img))


def pool(mask, value):
    return normalized_rows(mask.astype(np.float32) @ value)


def read_regions(boxes, cloud):
    seeds, visible, point, rgb = cloud
    inside = point_support(boxes,seeds)
    exterior = boxes.copy()
    exterior[:,3:6] *= 1.5
    boundary = point_support(exterior,seeds) & ~inside
    inside &= visible
    boundary &= visible
    count_in, count_ext = inside.sum(1),boundary.sum(1)
    a = np.c_[pool(inside,point),pool(inside,rgb)]
    b = np.c_[pool(boundary,point),pool(boundary,rgb)]
    return dict(inside=a, exterior=b, inner_count=count_in, outer_count=count_ext)


def phrase_features(z, roles):
    text = z['lang0'][0]
    assert text.shape[0]==roles.shape[1]
    return normalized_rows(roles.astype(np.float32) @ text)


def source_ref(entry, parsed):
    from calibrate168 import edges_of, source_instances
    from panel169 import rewrite
    from retrieval167 import direct_edges

    direct = direct_edges(edges_of(parsed['graph']))
    row = entry['row']
    if not direct:
        return None
    candidate = source_instances(row,parsed,sorted({i for i,_ in direct}))
    for i,sign in sorted(direct):
        if i not in candidate:
            continue
        views = rewrite(row['caption'],parsed,direct,i,sign)
        if views['status']=='supported':
            return i,sign,views['D'],candidate[i]
    return None


def source_frame(net, entry, params):
    from calibrate168 import parsed
    from calibrate168 import frozen_pool
    from interface66 import capture
    from retrieval167 import direct_edges, repaired
    from requery169 import native_donors, native_forward
    from crossplatform.ground_single_frame import orient_box
    from replay168 import oriented
    from labels166 import occupied_owners, physical_labels

    row = entry['row']
    assert sha(Path(row['pointpath']))==entry['scan'] and sha(Path(row['imagepath']))==entry['cam']
    started = perf_counter()
    ev = net.evidence_frame(row)
    z, _ = capture(net.base,ev.points,ev.indices,row['caption'])
    np.testing.assert_array_equal(z['boxes'],ev.boxes)
    np.testing.assert_array_equal(z['xyz'][0],ev.seeds)
    nr = len(ev.boxes)
    native = np.c_[ev.boxes,np.zeros(nr)]
    if direct_edges(ev.edges) and ev.role_mask.size:
        fixed = repaired(ev,frozen_pool(ev,params),z['query'],z['tokens'])
        body = fixed['bank'][0]>=0
        ids = fixed['bank'][0,body]
        extra = fixed['novel'][np.isfinite(fixed['novel']).all(1)]
        raw_s = np.r_[native,extra][ids]
        roi_s = oriented(row,fixed,ev.boxes,ids,lambda _:ev.points)
        score = fixed['score'][0,body]
    else:
        ids, score, _ = native_donors(dict(z=z,utility=ev.utility,subject_winner=ev.old_winner))
        raw_s = native[ids]
        roi_s = np.stack([orient_box(ev.points,b) for b in ev.boxes[ids]])
    fields = observations(ev,z)
    role = ev.role_mask
    text = phrase_features(z,role)
    subject = text[0] if len(text) else np.zeros(288,np.float32)
    desc = read_regions(raw_s,fields)
    names, _, points, labels = physical_labels(row,ev.seeds)
    owned = occupied_owners(roi_s,points,labels,len(names))
    category = np.array([names[j] == names[0] if j>=0 else False for j in owned])
    data = {f's_{k}':v for k,v in desc.items()}
    data.update(s_txt=subject,s_base=score,s_owners=owned,s_class=category,s_truth=0)
    ann = source_ref(entry,parsed(row))
    if ann is not None:
        ri,_,text_spec,truth = ann
        d = native_forward(net,row,ev.points,ev.indices,None,text_spec)
        assert np.array_equal(d['z']['xyz'],z['xyz'])
        refs,ranks,_ = native_donors(d)
        raw = d['z']['boxes'][refs]
        roi_r = np.stack([orient_box(ev.points,box) for box in raw])
        ref_owners = occupied_owners(roi_r,points,labels,len(names))
        eligible_class = np.array([names[j] == names[truth] if j>=0 else False for j in ref_owners])
        desc_ref = read_regions(np.c_[raw,np.zeros(len(raw))],fields)
        data.update({f'r_{k}':v for k,v in desc_ref.items()})
        assert ri<len(text)
        data.update(r_txt=text[ri],r_base=ranks,r_owners=ref_owners,r_class=eligible_class,r_truth=truth)
    data.update(seconds=perf_counter()-started)
    return data


def capture_source():
    import torch
    from common160 import npz_atomic
    from evidence162 import EvidenceGrounder
    from model51 import fingerprint
    from predict171 import read_npz

    path = DATA/'source'
    sealed = read(path/'ROSTER_SEAL.json')
    assert sealed['roster']==sha(path/'ROSTER.json') and not sealed['protected_data_access']
    lock = path/'CAPTURE_LOCK.json'
    inputs = dict(code=sha(Path(__file__)),design=sha(PROTOCOL),roster=sealed['roster'],
                  settings=sha(SOURCE_SETTINGS),source_pins=sha(ROOT/'crossplatform/single_frame_sources.json'))
    if lock.exists():
        assert read(lock)==inputs
    else:
        atomic_json(lock,inputs)
    rows = read(path/'ROSTER.json')
    files = {}
    settings = read(SOURCE_SETTINGS)['sources']
    complete = path/'CAPTURE_SEAL.json'
    assert not complete.exists()
    with (ROOT/'research/GPU_RESOURCE.lock').open('a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        torch.set_num_threads(1)
        for p in PLATFORMS:
            if not any(r['source']==p for r in rows):
                continue
            net = EvidenceGrounder(p)
            before = fingerprint(net.base),fingerprint(net.matcher)
            count = 0
            for entry in rows:
                if p!=entry['source']:
                    continue
                tag = f"{entry['case']:03d}_{p}.npz"
                out = path/tag
                if out.exists():
                    assert str(read_npz(out)['pin'])==sha(lock)
                else:
                    with torch.inference_mode():
                        f = source_frame(net,entry,settings[p])
                    npz_atomic(out,**f,pin=sha(lock))
                files[tag] = sha(out)
                count+=1
                if count%30==0:
                    print(p,count,flush=True)
            assert before==(fingerprint(net.base),fingerprint(net.matcher))
            del net
            gc.collect()
            torch.cuda.empty_cache()
    assert len(files)==len(rows)
    atomic_json(complete,dict(lock=sha(lock),files=files,target_labels_read=False,protected_data_access=False))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('prepare_source','capture_source'))
    globals()[parser.parse_args().phase]()
