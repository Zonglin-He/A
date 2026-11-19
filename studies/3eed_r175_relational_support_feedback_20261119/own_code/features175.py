"""Reuse natural frozen tensors, and encode only new source relation sentences."""
import argparse
from functools import lru_cache
import gc
import time

import numpy as np
import torch
import torch.nn.functional as F

from scope175 import (NEW_CACHE, OLD_CACHE, OLD_PRIVATE, PLATFORMS, PRIVATE, ROOT,
                      SENTINEL, camera_view, check_locked, json_atomic, npz_atomic, points_for,
                      read, sha, state)
from evidence import FIELDS, encoder, extract
from source175 import inventory_for
from camera_projection import CameraSpec, model_to_camera_matrix, read_camera_allowlist
from crossplatform.ground_single_frame import orient_box
from batch_yaw_iou import many_iou

STATIC = (*FIELDS,'in_image')
SUPERVISION = ('relation','depth_plane','slot_boxes','gt','known_ref','quality','foil','synthetic')
TOKEN_DIM = {'text':1,'valid':1,'role_mask':2}
SHARD = 128
CAPTURE_BATCH = 16


@lru_cache(maxsize=256)
def camera_plane(metadata_path, platform, view):
    cam=CameraSpec.from_camera_fields(platform,read_camera_allowlist(metadata_path),
                                     waymo_view_suffix=view,missing_distortion_policy='assume_zero_declared')
    return model_to_camera_matrix(cam)[2].astype(np.float32)


def plane(row):
    return camera_plane(row['metapath'],row['platform'],camera_view(row))


def annotate(row, sample, raw_slots):
    boxes=np.asarray(row['boxes'],np.float32)
    oriented=np.stack([orient_box(sample,b) for b in raw_slots.reshape(-1,6)]).reshape(2,4,7)
    quality=np.full((2,4),0.,np.float32)
    quality[0]=many_iou(oriented[0],boxes[0])
    known=row['known_reference']
    foil=np.zeros((2,4),bool)
    if known:
        quality[1]=many_iou(oriented[1],boxes[1])
        scene=inventory_for(row)
        if scene is not None:
            names, table, visibility, *_=scene
            for role in (0,1):
                overlap=many_iou(table,boxes[role])
                j=int(overlap.argmax())
                category=(names==names[j])&visibility&(overlap<.5)
                if overlap[j]>.75 and category.any():
                    other=table[category]
                    values=np.stack([many_iou(other,b) for b in oriented[role]])
                    foil[role]=(values.max(1)>.10)&(values.max(1)>quality[role]+.03)
    return dict(relation=np.int8(row['relation']),depth_plane=plane(row),slot_boxes=oriented,
                gt=boxes,known_ref=bool(known),quality=quality,foil=foil,
                synthetic=row['origin']=='synthetic_labelled_pair')


def span_masks(tokenizer, rows, width):
    texts=[r['caption']+SENTINEL for r in rows]
    tok=tokenizer(texts,padding='longest',return_offsets_mapping=True,return_tensors='np')
    a,b=tok['offset_mapping'].transpose(2,0,1)
    result=np.zeros((len(rows),3,width),bool)
    for i,row in enumerate(rows):
        for j,(start,stop) in enumerate(row['spans']):
            if stop>start:
                result[i,j,:a.shape[1]]=(a[i]<stop)&(b[i]>start)&(b[i]>a[i])
    return result, np.pad(tok['attention_mask'].astype(bool),((0,0),(0,width-a.shape[1])))


def add_cached_labels(tokenizer, rows, old, width):
    fields=[]
    added=np.full((len(rows),3,width),False)
    updated=np.asarray(['spans' in x for x in rows])
    for i,r in enumerate(rows):
        samples,_=points_for(r)
        raw=old['prior'][i][old['anchors'][i]]
        fields.append(annotate(r,samples,raw))
    if updated.any():
        masks,pad=span_masks(tokenizer,[r for r,u in zip(rows,updated,strict=True) if u],width)
        np.testing.assert_array_equal(pad,old['valid'][updated])
        # The reference span is the same entity that supplied the frozen reference noun logits.
        np.testing.assert_array_equal(masks[:,1],old['role_mask'][updated,1])
        added[updated]=masks
    return {k:np.asarray([r[k] for r in fields]) for k in fields[0]} | {
        'replace_role_mask':updated,'new_roles':added}


def add_synthetic(loaded, rows, width, first):
    pieces=[]
    for start in range(0,len(rows),CAPTURE_BATCH):
        part=rows[start:start+CAPTURE_BATCH]
        sampled=np.stack([points_for(x)[0] for x in part])
        f=extract(loaded,part,sampled,coordinate_audit=first and start==0)
        assert all(bool(z) for z in f['has_reference'])
        extent=f['valid'].shape[1]
        extractable={k:f[k] for k in STATIC}
        for k in TOKEN_DIM:
            axes=(0,0,0,width-extent) if k=='text' else (0,width-extent)
            extractable[k]=F.pad(f[k],axes)
        fields={k:v.cpu().numpy() for k,v in extractable.items()}
        raw=fields['prior'][np.arange(len(part))[:,None,None],fields['anchors']]
        annotations=[annotate(row,p,slots) for row,p,slots in zip(part,sampled,raw,strict=True)]
        fields.update({k:np.asarray([r[k] for r in annotations]) for k in annotations[0]})
        pieces.append(fields)
    return {k:np.concatenate([p[k] for p in pieces]) for k in pieces[0]}


def capture_source(source):
    check_locked()
    folder=PRIVATE/source
    p=NEW_CACHE/source
    p.mkdir(exist_ok=True,parents=True,mode=0o700)
    roster=read(folder/'ROSTER_SEAL.json')
    old=read(OLD_PRIVATE/source/'FEATURE_SEAL.json')
    assert roster['source']==source and roster['input']==sha(PRIVATE/'INPUT_LOCK.json')
    loaded=encoder(source)
    tokenizer=loaded[0].tokenizer
    files, widths, sizes={}, {}, {}
    counts={}
    for split in ('train','monitor'):
        path=folder/f'{split}_PRIVATE.json'
        assert sha(path)==roster['files'][path.name]
        all_rows=read(path)
        natural=[r for r in all_rows if r['origin']!='synthetic_labelled_pair']
        synthetic=all_rows[len(natural):]
        assert all(r['origin']=='synthetic_labelled_pair' for r in synthetic)
        width=max(old['widths'][split], *(len(s) for s in tokenizer(
            [r['caption']+SENTINEL for r in synthetic])['input_ids'])) if synthetic else old['widths'][split]
        widths[split]=width
        # Natural features keep the original frozen source token width.
        assert width==old['widths'][split]
        sizes[split]=dict(natural=len(natural),synthetic=len(synthetic))
        sums={k:0 for k in ('known_ref','both_covered','subject_foil_slots','ref_foil_slots')}
        for start in range(0,len(natural),SHARD):
            name=f'{split}_{start//SHARD:04d}.npz'
            source_path=OLD_CACHE/source/name
            assert sha(source_path)==old['files'][name]
            out=folder/f'natural_{name}'
            with np.load(source_path,allow_pickle=False) as storage:
                limit=min(SHARD,len(natural)-start)
                needed={k:storage[k][:limit] for k in ('prior','anchors','valid','role_mask','gt')}
                b=natural[start:start+limit]
                np.testing.assert_array_equal(needed['gt'][:,0],np.asarray([r['boxes'][0] for r in b],np.float32))
            if not out.exists():
                packed=add_cached_labels(tokenizer,b,needed,width)
                npz_atomic(out,**packed)
            files[out.name]=sha(out)
            with np.load(out,allow_pickle=False) as z:
                collect(sums,z)
            print('NATURAL_FEATURE',source,split,start+limit,len(natural),flush=True)
        for start in range(0,len(synthetic),SHARD):
            out=p/f'synthetic_{split}_{start//SHARD:04d}.npz'
            if not out.exists():
                made=add_synthetic(loaded,synthetic[start:start+SHARD],width,start==0)
                npz_atomic(out,**made)
            files[out.name]=sha(out)
            with np.load(out,allow_pickle=False) as z:
                collect(sums,z)
            print('SYNTHETIC_FEATURE',source,split,min(start+SHARD,len(synthetic)),len(synthetic),flush=True)
        counts[split]=sums
    json_atomic(folder/'FEATURE_SEAL.json',dict(source=source,files=files,roster=sha(folder/'ROSTER_SEAL.json'),
                        entry=sha(__file__),sizes=sizes,widths=widths,supervision=counts,
                        source_annotations_only=True,FINAL_access=False))
    del loaded
    gc.collect()
    torch.cuda.empty_cache()


def collect(summary, z):
    valid=z['known_ref']
    summary['known_ref']+=int(valid.sum())
    summary['both_covered']+=int((valid&(z['quality'].max(-1)>.1).all(-1)).sum())
    summary['subject_foil_slots']+=int(z['foil'][:,0].sum())
    summary['ref_foil_slots']+=int(z['foil'][:,1].sum())


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('sources',choices=PLATFORMS,nargs='+')
    cli=p.parse_args()
    state('source_features','Reuse natural frozen features and encode only newly labelled source prompts.')
    for source in cli.sources:
        capture_source(source)
