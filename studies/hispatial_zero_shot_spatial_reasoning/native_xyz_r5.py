"""Intervene on the object coordinates using HiSpatial's frozen native XYZ stream."""
import argparse
from collections import Counter
from pathlib import Path
import time
from unittest.mock import patch

import cv2
import numpy as np

from download import ROOT, STUDY, read_json, sha256, write_json
from geometry_cohort import stable
from height_geometry import UP, inside_rect
from height_transfer import BGR_ON_RGB
from object_pair_r4 import (GEOMETRY_SEAL, MODEL_SEAL, RUN as R4, check_seal as r4_seal,
                            index_rows, locked_roster, read_rgb)
from object_pair_r4_direction import CONFIG as CLOCK_CONFIG, SEAL as CLOCK_SEAL
from object_pair_r4_geometry import basis, clock_error, geometric_median, relationship
from object_region_plan import valid_xyz
from score import jsonl


CONFIG=STUDY/'NATIVE_XYZ_R5.json'
RUN=ROOT/'outputs'/'hispatial_r5_native_xyz_20261122'
ARMS=('native','assigned','swapped')
H=448


def source_seals():
    names=(MODEL_SEAL,GEOMETRY_SEAL,CLOCK_SEAL)
    for name in names:
        r4_seal(name)
    return {name:sha256(R4/name) for name in names}


def rects(row):
    scales=np.tile(H/np.asarray(row['size'],dtype=np.float64),2)
    return np.clip(np.asarray(row['boxes'][:2],dtype=np.float64)*scales,0,H)


def exclusive_masks(xyz,boxes):
    x,y=np.meshgrid(np.arange(H),np.arange(H))
    a,b=(inside_rect(x,y,box) for box in boxes)
    valid=valid_xyz(xyz)
    return (a&~b&valid,b&~a&valid)


def direct_clock(centers):
    return relationship(centers[0],centers[1],'direction',UP)[0]


def center_geometry(xyz,boxes,up):
    grid=xyz[:,2::4,2::4]
    points=grid[:3].transpose(1,2,0).astype(np.float64)
    x,y=np.meshgrid(np.arange(2,H,4),np.arange(2,H,4))
    keep=valid_xyz(grid)
    centers=np.stack([geometric_median(points[inside_rect(x,y,r)&keep]) for r in boxes])
    aligned=centers@basis(np.asarray(up,dtype=np.float64))
    aligned[:,1]*=-1
    return aligned


def load_xyz(key,digests):
    path=R4/'pointmaps'/f'{key}.npy'
    assert sha256(path)==digests[key]
    xyz=np.load(path,allow_pickle=False)
    assert xyz.shape==(4,H,H) and xyz.dtype==np.float32
    return xyz


def prepare():
    if (RUN/'input_seal.json').exists():
        raise FileExistsError(RUN/'input_seal.json')
    RUN.mkdir(parents=True,exist_ok=True,mode=0o700)
    seals=source_seals()
    roster=locked_roster()
    geometry=index_rows(R4/'geometry.jsonl')
    previous_inputs=index_rows(R4/'direction_control_inputs.jsonl')
    maps=read_json(R4/'pointmap_manifest.json')
    suffix=read_json(CLOCK_CONFIG)['with_frame']
    rows=[]
    for q,item in roster.items():
        if item['category']!='direction':
            continue
        old=previous_inputs[f'{q}/with_frame']
        assert old['prompt']==item['prompt']+'\n'+suffix
        assert old['image']==item['image'] and old['boxes']==item['boxes']
        r4=geometry[q]
        g=r4['geometry']['window_robust']
        row=dict(id=q,image=item['image'],source=item['source'],boxes=item['boxes'],size=item['size'],
                 prompt=old['prompt'],xyz_sha256=maps[item['image']],r4_direct=g['estimate'],
                 status='no_r4_geometry',anchors=None,direct_assigned=None,direct_swapped=None,exclusive_counts=None)
        if g['estimate'] is not None:
            xyz=load_xyz(row['image'],maps)
            boxes=rects(row)
            anchors=center_geometry(xyz,boxes,r4['support']['up'])
            np.testing.assert_allclose(anchors[:,[0,2]],g['evidence'],rtol=1e-9,atol=1e-9)
            anchors=anchors.astype(np.float32)
            assert np.isfinite(anchors).all() and (anchors[:,2]>0).all()
            assert direct_clock(anchors)==g['estimate']
            swapped=direct_clock(anchors[::-1])
            assert clock_error(g['estimate'],swapped)==6
            counts=[int(m.sum()) for m in exclusive_masks(xyz,boxes)]
            row.update(exclusive_counts=counts,status='no_exclusive_pixels')
            if min(counts)>0:
                row.update(status='intervened',anchors=anchors.tolist(),
                           direct_assigned=g['estimate'],direct_swapped=swapped)
        rows.append(row)
    assert len(rows)==107 and sum(r['r4_direct'] is not None for r in rows)==94
    jsonl(RUN/'inputs.jsonl',rows)
    write_json(RUN/'input_seal.json',dict(config=sha256(CONFIG),source_seals=seals,labels=False,
        sources=sha256(R4/'input_seal.json'),rows=len(rows),images=len({r['image'] for r in rows}),
        planner_code=sha256(Path(__file__)),inputs=sha256(RUN/'inputs.jsonl'),
        availability=dict(Counter(r['status'] for r in rows))))
    print('ROI coverage',dict(Counter(r['status'] for r in rows)),flush=True)


def validate_inputs():
    seal=read_json(RUN/'input_seal.json')
    assert seal['config']==sha256(CONFIG) and not seal['labels']
    assert seal['source_seals']==source_seals() and seal['sources']==sha256(R4/'input_seal.json')
    assert seal['inputs']==sha256(RUN/'inputs.jsonl') and seal['planner_code']==sha256(Path(__file__))
    rows=index_rows(RUN/'inputs.jsonl')
    assert len(rows)==seal['rows']==107
    return rows


def overlay(row):
    rgb=read_rgb(row['image']).copy()
    assert list(rgb.shape[1::-1])==row['size']
    for i,(x1,y1,x2,y2) in enumerate(row['boxes']):
        color=BGR_ON_RGB[i] if i<len(BGR_ON_RGB) else (255,255,0)
        cv2.rectangle(rgb,(x1,y1),(x2,y2),color,2)
    return cv2.resize(rgb,(H,H))


def edit_coordinates(xyz,masks,anchors):
    out=xyz.copy()
    for mask,anchor in zip(masks,anchors,strict=True):
        out[:3,mask]=anchor[:,None]
    footprint=masks[0]|masks[1]
    assert np.array_equal(out[3],xyz[3])
    assert np.array_equal(out[:,~footprint],xyz[:,~footprint])
    assert all(np.array_equal(out[:3,mask],np.broadcast_to(anchor[:,None],(3,int(mask.sum()))))
               for mask,anchor in zip(masks,anchors,strict=True))
    return out


def measure_feature_change(before,after,inside):
    import torch
    assert before.shape==after.shape and before.shape[0]==len(inside)
    delta=(after-before).float()
    changed=delta.ne(0).any(dim=1)
    assert not changed[~inside].any().item()
    d=delta[inside]
    return dict(changed_tokens=int(changed.sum().item()),outside_changed_tokens=0,
               delta_rms_inside=float(d.square().mean().sqrt().item()),
               delta_peak=float(d.abs().max().item()))


def prepare_features(predictor,row,patch_size,full_reference=False):
    import torch

    xyz=load_xyz(row['image'],{row['image']:row['xyz_sha256']})
    rgb=overlay(row)
    model=predictor.model
    processor=predictor.processor(text='<image> ',images=rgb,return_tensors='pt').to(predictor.device)
    pixel_values=processor['pixel_values']
    modes={'native':xyz}
    masks=None
    if row['anchors'] is not None:
        masks=exclusive_masks(xyz,rects(row))
        assert [int(v.sum()) for v in masks]==row['exclusive_counts']
        anchors=np.asarray(row['anchors'],dtype=np.float32)
        modes['assigned']=edit_coordinates(xyz,masks,anchors)
        modes['swapped']=edit_coordinates(xyz,masks,anchors[::-1])
    feature_maps={}
    depth_maps={}
    diagnostics={}
    with torch.inference_mode():
        depth_outputs=[]
        vision_outputs=[]
        hook=model.depth_encoder.register_forward_hook(lambda m,i,o:depth_outputs.append(o))

        def encode(value):
            depth_outputs.clear()
            cpu_xyz=torch.from_numpy(value)
            fused=model.get_image_features(pixel_values,cpu_xyz[None].to(predictor.device))
            assert len(depth_outputs)==1
            depth=depth_outputs.pop().squeeze(0).flatten(start_dim=1).T
            return (rgb,cpu_xyz,fused),depth

        try:
            capture=model.vision_tower.register_forward_hook(lambda m,i,o:vision_outputs.append(o))
            try:
                feature_maps['native'],depth_maps['native']=encode(xyz)
            finally:
                capture.remove()
            assert len(vision_outputs)==1
            with patch.object(model.vision_tower,'forward',return_value=vision_outputs[0]) as rgb_call:
                for arm in ('assigned','swapped'):
                    if arm in modes:
                        feature_maps[arm],depth_maps[arm]=encode(modes[arm])
                if full_reference:
                    replay=encode(xyz)
                    assert torch.equal(feature_maps['native'][2],replay[0][2])
                    assert torch.equal(depth_maps['native'],replay[1])
                    diagnostics['captured_rgb_native_replay_bitwise']=True
                assert rgb_call.call_count==len(modes)-1+int(full_reference)
        finally:
            hook.remove()
        if masks is not None:
            n=H//patch_size
            inside=torch.from_numpy((masks[0]|masks[1]).reshape(n,patch_size,n,patch_size).any(axis=(1,3)).ravel()).to(predictor.device)
            diagnostics['target_tokens']=int(inside.sum().item())
            for arm in ('assigned','swapped'):
                depth=measure_feature_change(depth_maps['native'],depth_maps[arm],inside)
                fusion=measure_feature_change(feature_maps['native'][2][0],feature_maps[arm][2][0],inside)
                assert depth['changed_tokens']>0 and fusion['changed_tokens']>0
                diagnostics[arm]=dict(depth=depth,fusion=fusion)
            contra_depth=measure_feature_change(depth_maps['assigned'],depth_maps['swapped'],inside)
            contra_fusion=measure_feature_change(feature_maps['assigned'][2][0],feature_maps['swapped'][2][0],inside)
            assert contra_depth['changed_tokens']>0 and contra_fusion['changed_tokens']>0
            diagnostics['assignment_vs_swap']=dict(depth=contra_depth,fusion=contra_fusion)
    return feature_maps,diagnostics


def predict(cache):
    import torch
    from predict import BATCH_N, batch_query, cached_single, release

    if (RUN/'prediction_seal.json').exists():
        raise FileExistsError(RUN/'prediction_seal.json')
    rows=list(validate_inputs().values())
    started=time.monotonic()
    torch.manual_seed(20261122)
    torch.set_num_threads(3)
    torch.backends.cudnn.benchmark=False
    cv2.setNumThreads(1)
    predictor,moge,runtime=release(cache)
    assert runtime==read_json(R4/'native_runtime.json')['model']
    patch_size=predictor.model.config.vision_config.patch_size
    conv=predictor.model.depth_encoder.conv
    assert conv.kernel_size==conv.stride==(patch_size,patch_size) and conv.padding=='valid'
    assert conv.dilation==(1,1) and H%patch_size==0
    del moge
    torch.cuda.empty_cache()

    check=sorted([r for r in rows if r['anchors'] is not None],key=lambda r:stable('native-xyz-preflight/'+r['id']))[:8]
    parity=[]
    encoding_replays=0
    for i in range(0,len(check),BATCH_N):
        cohort=check[i:i+BATCH_N]
        features=[prepare_features(predictor,r,patch_size,full_reference=True) for r in cohort]
        encoding_replays+=sum(f[1]['captured_rgb_native_replay_bitwise'] for f in features)
        for arm in ARMS:
            prepared=[bundle[0][arm] for bundle in features]
            single=[cached_single(predictor,f,r['prompt']) for f,r in zip(prepared,cohort,strict=True)]
            batched=batch_query(predictor,prepared,cohort)
            parity.extend(dict(arm=arm,same=a==b) for a,b in zip(single,batched,strict=True))
    use=BATCH_N if all(r['same'] for r in parity) else 1
    print('native stream parity',sum(r['same'] for r in parity),'/',len(parity),'batch',use,flush=True)
    old=index_rows(R4/'direction_controls.jsonl')
    payload=[]
    counts=Counter()
    for i in range(0,len(rows),use):
        group=rows[i:i+use]
        features=[prepare_features(predictor,r,patch_size) for r in group]
        decoded={arm:{} for arm in ARMS}
        for arm in ARMS:
            ready=[(j,bundle[0][arm]) for j,bundle in enumerate(features) if arm in bundle[0]]
            if ready:
                prompts=[group[j] for j,_ in ready]
                batch=[f for _,f in ready]
                res=batch_query(predictor,batch,prompts) if len(batch)>1 else [
                        cached_single(predictor,batch[0],prompts[0]['prompt'])]
                decoded[arm].update({r['id']:t for r,t in zip(prompts,res,strict=True)})
                counts[arm]+=len(res)
        for row,bundle in zip(group,features,strict=True):
            q=row['id']
            native=decoded['native'][q]
            payload.append(dict(id=q,response={arm:decoded[arm].get(q,native) for arm in ARMS},
                   r4_native_equal=native==old[f'{q}/with_frame']['response'],feature_audit=bundle[1]))
        if len(payload)%20<use or len(payload)==len(rows):
            print('decoded',len(payload),'/',len(rows),dict(counts),round(time.monotonic()-started,1),'s',flush=True)
    assert not any(p.requires_grad for p in predictor.model.parameters())
    jsonl(RUN/'predictions.jsonl',payload)
    write_json(RUN/'runtime.json',dict(model=runtime,calls=counts,preflight=parity,patch_size=patch_size,
        outside_features_bitwise_verified=True,shared_rgb_features=True,xyz_encoder='original',
        capture_replays_bitwise=encoding_replays,seconds=time.monotonic()-started,
        peak_cuda_bytes=torch.cuda.max_memory_allocated(),batch_size=use))
    write_json(RUN/'prediction_seal.json',dict(labels=False,training_steps=0,rows=len(payload),
        input=sha256(RUN/'input_seal.json'),files={f:sha256(RUN/f) for f in ('predictions.jsonl','runtime.json')}))
    print('sealed',len(payload),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('prepare','predict'))
    p.add_argument('--model-cache',type=Path)
    a=p.parse_args()
    if a.action=='prepare':
        prepare()
    else:
        if a.model_cache is None:
            p.error('--model-cache is required for predict')
        predict(a.model_cache)
