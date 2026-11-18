"""Decode the released conditional-span baseline after R158 and native outputs are sealed."""
import hashlib
import os
import re
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import ijson
import numpy as np
import torch
from transformers import RobertaTokenizerFast
from common160 import (ROOT, HERE, WORK, SPEC, DEV, EXTRA, REC, PLATFORMS, check_protocol, sha, array_sha,
                       npz_atomic, json_new, read, rows, canonical, prediction_file)

sys.path.insert(0,str(ROOT/'crossplatform/round3/p0/upstream'))
from src.joint_det_dataset import Joint3DDataset, WAYMO_SYNONYMS, M3ED_SYNONYMS, get_positive_map
from src import grounding_evaluator
from src.grounding_evaluator import GroundingEvaluator

ARMS=('official_span','native_text_only','prepared_axis','prepared_pca','r158_full')


def object_indices():
    # Only locator fields from the existing expression registries.
    ids={r['id']:(Path(r['meta_path']).resolve(),r['object_index']) for r in
         read(ROOT/'research/round69/input/SCORING_INDEX_PRIVATE.json')}
    ids.update({r['id']:(Path(r['metapath']).resolve(),r['object_index']) for r in
                read(ROOT/'research/round76/input/SOURCE_TRAIN512_PRIVATE.json')})
    return ids


def object_index(e,ids):
    r=e['r']
    if e['cohort']!=REC:return int(r['object_index'])
    meta,i=ids[r['id']]
    assert meta==Path(r['metapath']).resolve()
    return int(i)


def source_span_registry(target,split):
    if target=='waymo' and split=='TRAIN':
        return ROOT/'research/round6/source_data_audit/OFFICIAL_TRAIN_ANNO_ORDER_PRIVATE.json'
    return ROOT/f'research/round96/experiment/OFFICIAL_{target.upper()}_{split}_ORDER_PRIVATE.json'


@lru_cache(maxsize=None)
def reference_maps(target,split):
    # Positive-map digests were verified against Joint3DDataset in the original data audit.
    # Stream only path, expression index, digest and annotated-token-map digest.
    allowed={'meta_path','object_index','caption_sha256','positive_map_sha256'}
    index={}
    with source_span_registry(target,split).open('rb') as stream:
        for prefix,event,value in ijson.parse(stream):
            if prefix=='item' and event=='start_map':
                row={}
            elif prefix.startswith('item.') and prefix[5:] in allowed and event in ('string','number'):
                row[prefix[5:]]=value
            elif prefix=='item' and event=='end_map':
                k=(Path(row['meta_path']).resolve(),int(row['object_index']))
                assert k not in index
                index[k]=(row['caption_sha256'],row['positive_map_sha256'])
    return index


@lru_cache(maxsize=8192)
def expression_class(meta):
    columns=[]
    for field in ('caption','class'):
        with Path(meta).open('rb') as f:
            columns.append(list(ijson.items(f,'ground_info.item.'+field)))
    if len(columns[0])!=len(columns[1]):raise ValueError('Malformed native expression table')
    return list(zip(*columns))


def annotated_mask(e,idx,tok):
    r=e['r'];target=e['target'];mp=Path(r['metapath']).resolve()
    caption,category=expression_class(mp)[idx]; category=category.lower()
    assert canonical(caption)==canonical(r['caption'])
    ref=reference_maps(target,'TRAIN' if e['cohort']==REC else 'VAL')[(mp,idx)]
    if hashlib.sha256(caption.encode()).hexdigest()!=ref[0]:
        raise ValueError('Official category/span registry is joined to a different expression')
    phrase=Joint3DDataset._format_caption(None,caption)
    mapping=WAYMO_SYNONYMS if target=='waymo' else M3ED_SYNONYMS
    # Official loader orders matches by its synonym list, then by occurrence within each word.
    chosen=next((m.span() for w in mapping.get(category,[category])
                 for m in re.finditer(r'\b'+re.escape(w)+r'\b',phrase,flags=re.IGNORECASE)),None)
    if chosen is None:
        raise ValueError('Previously eligible official description unexpectedly lacks a span')
    encoded=tok.batch_encode_plus([phrase],padding='longest',return_tensors='pt')
    pm=get_positive_map(encoded,[torch.tensor(chosen,dtype=torch.long)])[0]
    if array_sha(pm)!=ref[1] or not (pm>0).any():
        raise ValueError('Native token mask differs from the official loader registry')
    return pm


def top_ten(out, mask):
    logits=out['last_proj_queries']@out['proj_tokens'].transpose(-1,-2)/0.07
    probability=logits.softmax(-1)
    padded=probability.new_zeros((1,256,256))
    padded[:,:,:probability.shape[2]]=probability
    scores=(padded[0]*(mask>0).float()).sum(-1)
    return scores.unsqueeze(0).argsort(1,True)[0,:10].cpu().numpy(),int((scores==scores.max()).sum())


def skip_metric(gt,selected):
    # The upstream evaluator passes the already chosen ten boxes to this final metric call.
    # Its result cannot affect sorting. The real common geometry is run only after the winner seal.
    assert gt.shape==(1,7) and selected.shape==(10,6)
    return selected.new_zeros((1,10)),None


@torch.inference_mode()
def main():
    from eval_lock import verify
    verify()
    check_protocol()
    seal=read(HERE/'FORWARD_SEAL.json');manifest=rows()
    frozen_file=WORK/'REPRESENTATIONS_MANIFEST_PRIVATE.json'
    assert seal['protocol']==sha(SPEC) and seal['n']==len(manifest) and sha(frozen_file)==seal['manifest_sha256']
    reps=read(frozen_file)
    assert len(reps)==len(manifest)
    tok=RobertaTokenizerFast.from_pretrained(str(ROOT/'crossplatform/upstream/data/roberta_base'),local_files_only=True)
    ids=object_indices();device='cuda'
    torch.backends.cuda.matmul.allow_tf32=False
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    native=GroundingEvaluator(prefixes=['last_'])
    stub={'gt_bboxes':torch.zeros((1,1,7),device=device),'meta_path':['dummy/p/f/m'],'utterances':['']}
    endpoint=np.empty((len(manifest),len(ARMS),7),np.float32)
    indices=np.empty((len(manifest),len(ARMS)),np.int16)
    flags=np.empty((len(manifest),4),np.int16)
    with patch.object(grounding_evaluator,'iou3d_rotated_vs_aligned',skip_metric):
        for i,e in enumerate(manifest):
            fn=prediction_file(i)
            assert sha(fn)==reps[fn.name]
            with np.load(fn,allow_pickle=False) as f:
                raw,q,t,proxy,prep,who=(f[k] for k in ('raw','q','t','text_proxy_mask','prepared','winners'))
            out={'last_proj_queries':torch.as_tensor(q,device=device)[None],
                 'proj_tokens':torch.as_tensor(t,device=device)[None],
                 'last_center':torch.as_tensor(raw[:,:3],device=device)[None],
                 'last_pred_size':torch.as_tensor(raw[:,3:],device=device)[None].clamp(min=1e-6),
                 'box_label_mask':torch.ones((1,1),device=device)}
            pm=annotated_mask(e,object_index(e,ids),tok)
            out['positive_map']=torch.as_tensor(pm,device=device)[None,None]
            winner,ties=top_ten(out,out['positive_map'][0])
            pad=np.zeros(256,np.float32);pad[:len(proxy)]=proxy
            pure,pties=top_ten(out,torch.as_tensor(pad,device=device)[None])
            native.evaluate_bbox_by_contrast(stub,out,'last_')
            literal=native.prediction_records.pop()
            clamped=raw.copy();clamped[:,3:]=np.maximum(clamped[:,3:],np.float32(1e-6))
            if not np.array_equal(np.array(literal['pred_box'],np.float32),clamped[winner[0]]):
                raise ValueError('Different box from the unmodified upstream winner selection')
            endpoint[i,0,:6]=clamped[winner[0]];endpoint[i,1,:6]=clamped[pure[0]]
            endpoint[i,:2,6]=0
            endpoint[i,2]=prep[0];endpoint[i,2,6]=0
            endpoint[i,3:]=prep
            indices[i]=[winner[0],pure[0],who[0],who[0],who[1]]
            flags[i]=[ties,pties,np.count_nonzero((raw[:,3:]<1e-6).any(1)),int(winner[0]!=pure[0])]
            if (i+1)%2000==0:print('official top1',i+1,flush=True)
    dest=WORK/'BOXES_PRIVATE.npz'
    assert not dest.exists()
    npz_atomic(dest,boxes=endpoint,winners=indices,flags=flags,arms=ARMS)
    json_new(HERE/'PREDICTION_SEAL.json',dict(n=len(manifest),sha256=sha(dest),raw_forward_seal=sha(HERE/'FORWARD_SEAL.json'),
           arms=ARMS,all_official_masks_match_native_registry=len(manifest),upstream_pred_box_matches=len(manifest),
           official_top_score_ties=int((flags[:,0]>1).sum()),proxy_top_score_ties=int((flags[:,1]>1).sum()),
           records_with_native_dimension_clamp=int((flags[:,2]>0).sum()),annotated_span_changed_winner_from_proxy=int(flags[:,3].sum()),
           no_target_boxes_for_prediction=True,official_benchmark_uses_target_class=True,
           target_class_used_by_R158=False,source_training_runs=0,target_updates=0,FINAL_access=False))
    print('winners sealed',len(manifest),flush=True)


if __name__=='__main__':main()
