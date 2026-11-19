"""Reconstruct the old head inputs and capture aligned instance evidence on sealed pools."""
import fcntl
import gc
import hashlib
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from scipy.fft import dct

from catalog import HOME, ROOT, PRIVATE, PROTOCOL, OVERNIGHT, atomic_json, read, sha, source_roster


def rgb_bins(row, uv):
    with Image.open(row['imagepath']) as im:
        pixels = np.asarray(im.convert('RGB'))
    h,w = pixels.shape[:2]
    clean = np.nan_to_num(uv)
    ix = np.clip(np.rint(clean[:,0]*(w-1)).astype(int),0,w-1)
    iy = np.clip(np.rint(clean[:,1]*(h-1)).astype(int),0,h-1)
    # Square-root normalized distributions reduce domination by a single color bin.
    channels = pixels[iy,ix].astype(float)/256. * 8.
    ticks = np.arange(8, dtype=float)+.5
    out = np.maximum(1. - np.abs(channels[:,:,None]-ticks), 0.)
    total = np.maximum(out.sum(-1,keepdims=True), 1.e-12)
    return (out / total).reshape(-1,24).astype(np.float32)


def histogram(mask, colors):
    counts = mask.astype(np.float32) @ colors
    counts = counts.reshape(-1,3,8)
    counts /= np.maximum(1.,counts.sum(-1,keepdims=True))
    return np.sqrt(counts).reshape(-1,24)/np.sqrt(3.)


def geometry_evidence(ev, boxes):
    from visual162 import projection

    subject = ev.old_pca
    extent = boxes[:,:3] - subject[:3]
    distance = np.linalg.norm(extent[:,:2],axis=1)
    scale = np.log(np.maximum(.05,boxes[:,3:5]) / np.maximum(.05,subject[3:5]))
    scene_xy = projection(ev,np.r_[subject[None,:3],boxes[:,:3]])
    screen = np.nan_to_num(scene_xy[1:] - scene_xy[0],nan=0.,posinf=0.,neginf=0.)
    return np.c_[np.tanh(extent / [12.,12.,3.]),np.tanh(distance/12.),np.tanh(scale),
                 np.tanh(screen*3.)].astype(np.float32)


def shuffled_rgb(local, observed, key):
    result = local.copy()
    active = np.flatnonzero(observed)
    if len(active)>1:
        h=hashlib.sha256(('candidate-rgb-v1|'+key).encode()).digest()
        shift=1+int.from_bytes(h[:8],'little')%(len(active)-1)
        result[active,288:344] = local[np.roll(active,shift),288:344]
    return result


def capture(net, item, packet, key):
    from appearance_collect import MIN_SEEDS, observations, phrase_features, read_regions
    from common160 import array_sha
    from edges import point_support
    from interface66 import capture as base_capture
    from requery169 import native_donors, native_forward

    row = item['row']
    assert sha(Path(row['pointpath']))==row['point_sha256'] and sha(Path(row['imagepath']))==row['image_sha256']
    ev = net.evidence_frame(row)
    original, _ = base_capture(net.base,ev.points,ev.indices,row['caption'])
    np.testing.assert_array_equal(original['xyz'][0],ev.seeds)
    np.testing.assert_array_equal(original['boxes'],ev.boxes)
    assert str(packet['sample_sha'])==array_sha(ev.indices)
    assert str(packet['points_sha'])==array_sha(ev.points)
    rewritten = native_forward(net,row,ev.points,ev.indices,None,item['views']['D'])
    identifiers, ranks, _ = native_donors(rewritten)
    np.testing.assert_array_equal(rewritten['z']['boxes'][identifiers],packet['raw'])
    np.testing.assert_array_equal(ranks-ranks.max(),packet['unaries'][:-1])
    np.testing.assert_array_equal(original['xyz'],rewritten['z']['xyz'])

    cloud = observations(ev,original)
    seeds, visible = cloud[:2]
    boxes = np.c_[packet['raw'],np.zeros(len(identifiers))]
    local = read_regions(boxes,cloud)
    exp = phrase_features(original,ev.role_mask)
    role = item['role']
    assert 0 < role < len(exp)
    old = (local['inside'] - local['exterior']).astype(np.float32)
    availability = (local['inner_count']>=MIN_SEEDS)&(local['outer_count']>=MIN_SEEDS)&(np.linalg.norm(exp[role])>0)

    _, ids = np.unique(ev.seeds,axis=0,return_index=True)
    np.testing.assert_array_equal(ev.seeds[np.sort(ids)],seeds)
    per_seed = rgb_bins(row,ev.uv[np.sort(ids)])
    interior = point_support(boxes,seeds)
    enlarged = boxes.copy()
    enlarged[:,3:6] *= 1.5
    exterior = point_support(enlarged,seeds)&~interior
    descriptor = histogram(interior&visible,per_seed)-histogram(exterior&visible,per_seed)
    compressed = dct(old[:,288:],axis=1,type=2,norm='ortho')[:,:32]
    extra = np.c_[old[:,:288],compressed,descriptor,geometry_evidence(ev,packet['boxes'])].astype(np.float32)
    assert extra.shape==old.shape==(len(boxes),352) and exp[role].shape==(288,)
    muted=extra.copy()
    muted[:,288:344]=0.
    return dict(old=old,extra=extra,rgb_off=muted,rgb_shuffled=shuffled_rgb(extra,availability,key),
                phrase=exp[role].astype(np.float32),usable=availability,original_pool_sha=array_sha(packet['raw']),
                sample_sha=packet['sample_sha'],frozen_forward_path=True)


def run():
    import torch
    from common160 import npz_atomic
    from evidence162 import EvidenceGrounder
    from model51 import fingerprint
    from predict171 import read_npz

    sys.path.insert(0,str(OVERNIGHT))
    roster, seals = source_roster()
    path=PRIVATE/'features'
    path.mkdir(mode=0o700,parents=True,exist_ok=True)
    lock=path/'LOCK.json'
    intent=dict(inputs=seals,protocol=sha(PROTOCOL),producer=sha(Path(__file__)),
                prior_features=sha(OVERNIGHT/'appearance_collect.py'),checkpoint_pin=sha(ROOT/'crossplatform/single_frame_sources.json'),
                protected_data_access=False,labels_loaded=False)
    if lock.exists():
        assert read(lock)==intent
    else:
        atomic_json(lock,intent)
    done=path/'SEAL.json'
    assert not done.exists()
    packets = {}
    counts = {}
    torch.set_num_threads(1)
    with (ROOT/'research/GPU_RESOURCE.lock').open('a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for source in ('waymo','drone','quad'):
            net = EvidenceGrounder(source)
            before = fingerprint(net.base),fingerprint(net.matcher)
            n = 0
            for key,(item,example) in roster.items():
                if item['row']['platform']!=source:
                    continue
                p = path/(key+'.npz')
                if p.exists():
                    assert read_npz(p)['lock'].item()==sha(lock)
                else:
                    with torch.inference_mode():
                        answer = capture(net,item,read_npz(example),key)
                    npz_atomic(p,**answer,lock=sha(lock))
                packets[key]=sha(p)
                n+=1
                if n%12==0:
                    print(source,n,'verified pools',flush=True)
            assert before==(fingerprint(net.base),fingerprint(net.matcher))
            counts[source]=n
            del net
            gc.collect()
            torch.cuda.empty_cache()
    assert len(packets)==len(roster)==144
    atomic_json(done,dict(lock=sha(lock),files=packets,counts=counts,
                          source_weights_unchanged=True,protected_data_access=False,labels_loaded=False))
    print('All fixed-pool features sealed',counts,flush=True)


if __name__ == '__main__':
    run()
