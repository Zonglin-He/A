"""Matched beam, RGB-path, competition, overlap and geometry readouts on official Validation."""
from collections import Counter
from functools import lru_cache
import argparse
import gc
import hashlib
import sys
import time

import numpy as np
import torch

from scope165 import HERE, ORDER, PLATFORMS, ROOT, W97, WORK, atomic_json, check_lock, ordinary, read, sha
from common160 import array_sha, npz_atomic

sys.path[:0] = [str(ROOT), str(ROOT/'research/round158'),str(ROOT/'research/round66'),
                str(ROOT/'crossplatform/round51'),str(ROOT/'crossplatform/round7/experiment'),
                str(ROOT/'crossplatform/round3/p0/upstream')]
from crossplatform.ground_single_frame import SingleFrameGrounder, WAYMO_VIEWS, orient_box
from grounding158 import select_grounding_box
from model51 import SENTINEL, fingerprint
from runner_helpers import sample_points
from utils.transform_waymo import transform_to_front_view

RAW_ARMS = ('native_text_axis','native_text_pca','r76_lidar_axis','r76_lidar_pca',
            'r76_rgb_axis','r76_rgb_pca','r76_overlap_axis','r76_overlap_pca',
            'r91_axis','r91_pca','r158_axis','r158_pca')
BLOCK = 128
archived_file_sha = lru_cache(maxsize=10000)(sha)


def points_for(row):
    pts,ids,_ = sample_points(row)
    if row['platform']=='waymo':
        pts,_ = transform_to_front_view(pts,np.empty((0,7)),WAYMO_VIEWS[int(row['suffix'] or 0)])
    return np.ascontiguousarray(pts,np.float32),ids


def axis_box(box):
    result = np.r_[box,np.float32(0)]
    result[3:6] = np.maximum(result[3:6],np.float32(1e-6))
    return result


@lru_cache(maxsize=6)
def old_files(source,target):
    path = W97/f'{source}_to_{target}'
    parent = read(W97/'MATRIX_SEALED.json')
    old = read(path/'SEALED.json')
    assert sha(path/'SEALED.json')==parent['seals_sha256'][path.name]
    assert old['source']==source and old['target']==target and not old['target_task_GT_read']
    return path,old['files']


def cached(source,row,z,capture_receipt,pilot):
    path,files = old_files(source,row['platform'])
    a,b = [path/f'{kind}_PRIVATE'/(row['id']+suffix) for kind,suffix in (('samples','.npz'),('records','.json'))]
    for f in (a,b):
        assert sha(f)==files[f.relative_to(path).as_posix()]
    rec = read(b)
    for key in ('point_sample_sha256','point_indices_sha256','actual_model_text_sha256'):
        assert capture_receipt[key]==rec['input_receipt'][key]
    with np.load(a,allow_pickle=False) as f:
        assert int(f['anno_index'])==row['anno_index']
        np.testing.assert_array_equal(f['point_indices'],z['point_indices'])
        np.testing.assert_array_equal(f['source_boxes'],z['boxes'])
        names = f['arm_names'].tolist()
        original,scores = f['scores'][[names.index('original_source'),names.index('prepared_frozen')]]
    geo = rec['geometry']
    key = hashlib.sha256(z['xyz'].tobytes()+z['point_indices'].tobytes()).hexdigest()
    assert key==geo['geometry_seed_and_sample_sha256']
    assert archived_file_sha(row['imagepath'])==geo['image_sha256']
    assert archived_file_sha(row['metapath'])==geo['meta_sha256']
    assert archived_file_sha(row['pointpath'])==geo['point_sha256']
    lookup = W97/'shared_physical_geometry_PRIVATE'/f"{row['id']}_{key[:20]}.npz"
    with np.load(lookup,allow_pickle=False) as data:
        g = dict(data)
    g['query_anchor_indices'] = np.asarray(z['query_points_sample_inds'],np.int64)
    if pilot:
        from prepare76 import geometry
        live,_ = geometry(row,z)
        for name in g:
            np.testing.assert_array_equal(g[name],live[name])
    return g,original,scores


def live_r91(model,row,z,g,p0,attention,prefix,pixels,use_old_parser):
    from core89 import route89, tensor_route89, token_roles88
    from parse97 import dependency_document, parse85, parse_subject, repair_inventory87

    text = row['caption']
    if use_old_parser:
        previous = read(W97/'roles_PRIVATE'/(row['id']+'.json'))
        assert previous['caption_sha256']==hashlib.sha256(text.encode()).hexdigest()
        graph,doc = previous['graph'],previous['doc']
    else:
        doc = dependency_document(model.parser(text))
        graph = repair_inventory87(text,doc,parse85(text,doc,parse_subject(text,doc)))
    token_roles,offs = token_roles88(model.base,row,z,graph,doc)
    route,_ = route89(row,dict(z,frozen_probability=p0.cpu().numpy()),g,graph,token_roles,attention,offs,doc)
    m = model.matcher
    m.route = tensor_route89(route)
    m.use_interface = bool(len(token_roles))
    try:
        p,_,details = m.main(model.base,prefix,pixels,'bidirectional')
    finally:
        m.route = None
        m.use_interface = False
        m.local = {}
    torch.testing.assert_close(details['base_probability'],p0,atol=0.,rtol=0.)
    return p.cpu().numpy()


@torch.inference_mode()
def evaluate(model,row,*,pilot=False,reuse=True):
    from interface66 import capture
    from model76 import tensor_data, tensor_prefix
    from prepare76 import geometry

    pts,ids = points_for(row)
    z,receipt = capture(model.base,pts,ids,row['caption'])
    if reuse:
        g,orig,p91 = cached(model.source,row,z,receipt,pilot)
    else:
        g,_ = geometry(row,z)
        p91 = None
        orig = None
    prefix,pixels = tensor_prefix(z),tensor_data(g,row)
    matcher = model.matcher
    assert matcher.route is None and not matcher.use_interface
    attention = []
    handle = None
    if pilot or not reuse:
        layer = model.base.cross_encoder.layers[-1].cross_layer.cross_lv
        handle = layer.register_forward_hook(lambda m,a,out:attention.append(out[1][0].cpu().numpy()))
    try:
        p76,_,_ = matcher.main(model.base,prefix,pixels,'bidirectional')
    finally:
        if handle is not None:
            handle.remove()
    no_rgb = dict(pixels,image=torch.full_like(pixels['image'],.5))
    for name in ('p2i','i2p'):
        no_rgb[name]=torch.full_like(pixels[name],-1)
        no_rgb[name+'_weight']=torch.zeros_like(pixels[name+'_weight'])
    p_lidar,_,detail = matcher.main(model.base,prefix,no_rgb,'independent')
    assert not bool(detail['late_RGB_delta'].count_nonzero())
    if handle is not None:
        actual=live_r91(model,row,z,g,p76,attention[0],prefix,pixels,reuse)
        if reuse:
            np.testing.assert_allclose(actual,p91,atol=1e-10,rtol=0.)
            assert int(actual.argmax())==int(p91.argmax())
        p91=actual
    q = z['query']
    t = z['tokens']
    mask = torch.as_tensor(z['token_mask'],device='cuda').bool()
    # Rank-only comparison uses the same full sentence and same text mask in both native/readout arms.
    s = torch.as_tensor(q,device='cuda') @ torch.as_tensor(t,device='cuda').T / .07
    logits = torch.logsumexp(s[:,mask],-1)-torch.logsumexp(s[:,~mask],-1)
    pnative = logits.double().softmax(0).cpu().numpy()
    if reuse:
        np.testing.assert_allclose(pnative,orig,atol=1e-11,rtol=0.)
    boxes=z['boxes']
    b76=p76.cpu().numpy()
    candidates = (int(pnative.argmax()),int(p_lidar.argmax()),int(b76.argmax()),
                  select_grounding_box(boxes,b76)[0],int(p91.argmax()),select_grounding_box(boxes,p91)[0])
    result={}
    for i in set(candidates):
        result[i] = (axis_box(boxes[i]),orient_box(pts,boxes[i]))
        np.testing.assert_array_equal(result[i][0][:6],result[i][1][:6])
    return dict(q=q,tokens=t,raw=boxes,boxes=np.stack([b for i in candidates for b in result[i]]),
                winners=np.repeat(np.asarray(candidates,dtype=np.int16),2),
                indices_sha=np.frombuffer(bytes.fromhex(array_sha(ids)),np.uint8),
                points_sha=np.frombuffer(bytes.fromhex(array_sha(pts)),np.uint8),
                visible=int(g['visible'].sum()))


def write_block(path,batches,ordinals):
    fields={k:np.stack([a[k] for a in batches]) for k in batches[0] if k!='tokens'}
    size=np.array([len(a['tokens']) for a in batches],dtype=int)
    npz_atomic(path,**fields,tokens=np.concatenate([a['tokens'] for a in batches]), offsets=np.r_[0,np.cumsum(size)],
               ordinals=np.asarray(ordinals,dtype=int),arms=np.asarray(RAW_ARMS),protocol=check_lock()['protocol_sha256'])


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--pilot',action='store_true')
    a=p.parse_args()
    check_lock()
    samples=ordinary()
    start=time.perf_counter()
    written={}
    for source in PLATFORMS:
        model=SingleFrameGrounder(source)
        fp=(fingerprint(model.base),fingerprint(model.matcher))
        for s,target in ORDER:
            if s!=source:
                continue
            folder=WORK/('pilot' if a.pilot else 'ordinary')/f'{s}_to_{target}'
            folder.mkdir(parents=True,exist_ok=True)
            rows=samples[target]
            selected=np.unique(np.linspace(0,len(rows)-1,8,dtype=int)).tolist() if a.pilot else list(range(len(rows)))
            files={}
            for k in range(0,len(selected),BLOCK):
                ordinals=selected[k:k+BLOCK]
                nm=f'{k//BLOCK:04d}.npz'
                f=folder/nm
                if f.exists():
                    with np.load(f,allow_pickle=False) as existing:
                        assert existing['ordinals'].tolist()==ordinals and str(existing['protocol'])==check_lock()['protocol_sha256']
                else:
                    values=[evaluate(model,rows[i],pilot=a.pilot) for i in ordinals]
                    write_block(f,values,ordinals)
                files[nm]=sha(f)
                print('R165',source,target,k+len(ordinals),'/',len(selected),
                      'seconds',round(time.perf_counter()-start,1),flush=True)
            atomic_json(folder/'SEAL.json',dict(source=source,target=target,n=len(selected),files=files,
                    protocol_sha256=check_lock()['protocol_sha256'],file_layout='blocks with zero-based R97 ordinal',
                    oracle_target_class_read=False,protected_payload_access=False,frozen_candidate_parity=len(selected),
                    R91_live_parity=len(selected) if a.pilot else 0))
            written[s,target]=len(selected)
        assert fp==(fingerprint(model.base),fingerprint(model.matcher))
        del model
        gc.collect()
        torch.cuda.empty_cache()
    assert len(written)==6
    atomic_json(HERE/('PILOT_SEAL.json' if a.pilot else 'ORDINARY_FORWARD_SEAL.json'),
                dict(n=sum(written.values()),by_direction={'_to_'.join(k):v for k,v in written.items()},
                     paired_frozen_forward=True,seconds=time.perf_counter()-start,FINAL_access=False))


if __name__=='__main__':
    main()
