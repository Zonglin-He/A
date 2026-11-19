"""Replay the frozen full R158 path on the authorized source roster."""
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
import hashlib
import time

import numpy as np
import torch

from scope174 import (OLD_CACHE, OLD_PRIVATE, PLATFORMS, PRIVATE, ROOT, SENTINEL, SYNTH_CACHE,
                      check_locked, json_atomic, npz_atomic, points_for, read, register, sha, stable, state)
from anchors174 import prepare
from crossplatform.ground_single_frame import SingleFrameGrounder
from evidence import anchor_seeds, natural_graphs, pixel_proof, text_masks
from batch_yaw_iou import many_iou


def rephrase(row):
    if not row['known_reference']:
        return row
    # The declared side and the original leftmost/unique anchor remain explicit.
    text=row['caption']
    s,r,_=row['spans']
    subj=text[s[0]:s[1]].lower()
    ref=text[r[0]:r[1]].strip().lower()
    side='left' if row['side']<0 else 'right'
    layout=int(stable('R174_relation_wording_v1',row['metapath'],text)[:8],16)%3
    prefix,link,end=(
        ('Look for ',f' immediately to the {side} of ',' in this view.'),
        ('',f' is the one nearest on the {side} of ',' in the image.'),
        ('Select ',f' closest on the {side} of ',' looking at the image.')
    )[layout]
    caption=prefix+subj+link+ref+end
    a=len(prefix)
    b=a+len(subj)
    c=b+len(link)
    return dict(row,caption=caption,spans=[[a,b],[c,c+len(ref)],[b,c]])


def prepare_parse(model, rows):
    from parse97 import dependency_document, parse85, parse_subject, repair_inventory87

    out=[]
    for r,doc in zip(rows,model.parser.bulk_process([r['caption'] for r in rows]),strict=True):
        text=r['caption']
        d=dependency_document(doc)
        g=repair_inventory87(text,d,parse85(text,d,parse_subject(text,d)))
        out.append((g,d))
    return out


@torch.inference_mode()
def r158_source(model,row,sampled,indices,parsed):
    from core89 import route89, tensor_route89, token_roles88
    from interface66 import capture
    from model76 import tensor_data, tensor_prefix
    from prepare76 import geometry

    z,_=capture(model.base,sampled,indices,row['caption'])
    g,_=geometry(row,z)
    prefix,pixels=tensor_prefix(z),tensor_data(g,row)
    matcher=model.matcher
    captured=[]
    layer=model.base.cross_encoder.layers[-1].cross_layer.cross_lv
    h=layer.register_forward_hook(lambda m,a,out:captured.append(out[1][0].cpu().numpy()))
    try:
        initial,_,_=matcher.main(model.base,prefix,pixels,'bidirectional')
    finally:
        h.remove()
    graph,doc=parsed
    tokens,offs=token_roles88(model.base,row,z,graph,doc)
    route,_=route89(row,dict(z,frozen_probability=initial.cpu().numpy()),g,graph,tokens,captured[0],offs,doc)
    matcher.route=tensor_route89(route)
    matcher.use_interface=bool(len(tokens))
    try:
        p,_,det=matcher.main(model.base,prefix,pixels,'bidirectional')
        torch.testing.assert_close(det['base_probability'],initial,rtol=0.,atol=0.)
    finally:
        matcher.route=None
        matcher.use_interface=False
        matcher.local={}
    return z,p.cpu().numpy()


def synthetic_features(model,row,z,sampled):
    tokens=model.base.tokenizer([row['caption']+SENTINEL],padding='longest',
                                 return_offsets_mapping=True,return_tensors='np')
    role,native,has_ref,is_lr=text_masks([row],tokens['offset_mapping'],tokens['attention_mask'].astype(bool))
    proj=z['query']@z['tokens'].T/.07
    rs=(proj*role[0,1]).sum(1)/max(1,int(role[0,1].sum()))
    uv_tuple=pixel_proof(row,z['xyz'][0],sampled)
    return dict(points=z['vis0'][0].astype(np.float16),xyz=z['xyz'][0],text=z['lang0'][0].astype(np.float16),
                valid=tokens['attention_mask'][0].astype(bool), role_mask=role[0],has_reference=has_ref[0],
                left_right=is_lr[0], uv=uv_tuple[1].astype(np.float16),in_image=uv_tuple[2],visible=uv_tuple[3],
                ref_scores=rs.astype(np.float32))


def ref_rank(a, score):
    base=a.copy()
    base[:,3:]=np.maximum(base[:,3:],.03)
    aa=torch.from_numpy(base)[None]
    rr=torch.from_numpy(np.broadcast_to(score,(1,2,256)).copy())
    return anchor_seeds(aa,rr)[0,1].numpy()


def source_one(model,row,old,idx,parsed):
    sampled,ids=points_for(row)
    z,p=r158_source(model,row,sampled,ids,parsed)
    raw=z['boxes']
    synthetic=bool(row['known_reference'])
    if synthetic:
        f=synthetic_features(model,row,z,sampled)
        scores=f.pop('ref_scores')
    else:
        f=None
        np.testing.assert_array_equal(z['xyz'][0],old['xyz'][idx])
        scores=old['prior_scores'][idx,1]
    cand=prepare(raw,p,sampled,ref_rank(raw,scores))
    cand['ref_logit']=scores[cand['reference_ids']]
    cand['gt']=np.asarray(row['boxes'],np.float32)
    cand['known_ref']=synthetic
    cand['quality']=np.stack([many_iou(cand['subject_boxes'],cand['gt'][0]),
                              (many_iou(cand['reference_boxes'],cand['gt'][1]) if synthetic else np.zeros(4))]).astype(np.float32)
    return cand,f


def source_capture(source):
    p=PRIVATE/source
    p.mkdir(exist_ok=True,mode=0o700)
    tmp=SYNTH_CACHE/source
    tmp.mkdir(exist_ok=True,parents=True,mode=0o700)
    lock=read(OLD_PRIVATE/source/'FEATURE_SEAL.json')
    roster=read(OLD_PRIVATE/source/'ROSTER_SEAL.json')
    contract=sha(PRIVATE/'INPUT_LOCK.json')
    done=p/'ANCHORS_SEAL.json'
    if done.exists():
        st=read(done)
        assert st['input']==contract
        assert all(sha(p/f)==h for f,h in st['files'].items())
        assert all(sha(tmp/f)==h for f,h in st['synthetic'].items())
        return
    model=SingleFrameGrounder(source)
    start=time.perf_counter()
    files,syn={},{}
    for split in ('train','monitor'):
        path=OLD_PRIVATE/source/f'{split}_PRIVATE.json'
        assert sha(path)==roster['rosters'][path.name]
        rows=[rephrase(r) for r in read(path)]
        for offset in range(0,len(rows),128):
            nm=f'{split}_{offset//128:04d}.npz'
            dest=p/nm
            synfile=tmp/nm
            group=rows[offset:offset+128]
            if not dest.exists():
                previous=OLD_CACHE/source/nm
                assert sha(previous)==lock['files'][nm]
                arrays=[]
                replacements={}
                with np.load(previous,allow_pickle=False) as store:
                    old={k:store[k] for k in ('xyz','prior_scores')}
                for j in range(0,len(group),8):
                    batch=group[j:j+8]
                    parses=prepare_parse(model,batch)
                    for l,(row,d) in enumerate(zip(batch,parses,strict=True),j):
                        local=dict(row,id=stable('R174-source-utterance',source,split,offset+l,row['caption']))
                        cand,f=source_one(model,local,old,l,d)
                        arrays.append(cand)
                        if f is not None:
                            for k,v in f.items():
                                replacements[f'{l}/{k}']=v
                if replacements:
                    npz_atomic(synfile,**replacements)
                pack={k:np.asarray([x[k] for x in arrays]) for k in arrays[0]}
                npz_atomic(dest,**pack,input=contract)
            else:
                with np.load(dest,allow_pickle=False) as f:
                    assert str(f['input'])==contract and len(f['quality'])==len(group)
            files[nm]=sha(dest)
            if any(r['known_reference'] for r in group):
                syn[nm]=sha(synfile)
            print('SOURCE_R158',source,split,min(offset+128,len(rows)),len(rows),
                  round(time.perf_counter()-start,1),flush=True)
    json_atomic(done,dict(source=source,input=contract, files=files,synthetic=syn,
                         used_original_source_matcher=True, target_labels_used=False, FINAL_access=False))


def main():
    import argparse
    import gc
    q=argparse.ArgumentParser()
    q.add_argument('sources',nargs='+',choices=PLATFORMS)
    args=q.parse_args()
    register()
    state('source_features','Preparing exact R158 scores plus fixed candidate supervision on the source roster.')
    for src in args.sources:
        source_capture(src)
        gc.collect()
        torch.cuda.empty_cache()


if __name__=='__main__':
    main()
