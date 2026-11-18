"""Re-run the original frozen network on the exact input used for each comparison."""
import gc
import os
import sys
import time
from functools import lru_cache
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from common160 import (DEV, EXTRA, REC, HERE, ROOT, WORK, SPEC, PLATFORMS, rows, reference, read,
                       sha, array_sha, check_protocol, json_new, npz_atomic, prediction_file, canonical)

sys.path[:0] = [str(ROOT),str(ROOT/'research/round158'),str(ROOT/'crossplatform/round96'),
                str(ROOT/'crossplatform/round51'),str(ROOT/'crossplatform/round7/experiment'),
                str(ROOT/'crossplatform/round3/p0/upstream')]
from crossplatform.ground_single_frame import SingleFrameGrounder, WAYMO_VIEWS, orient_box
from model51 import SENTINEL, firstnoun_mask
from runner_helpers import sample_points
from utils.transform_waymo import transform_to_front_view

hash_raw=lru_cache(maxsize=8192)(sha)


def points_for(row):
    pts,idx,_=sample_points(row)
    if row['platform']=='waymo':
        pts,_=transform_to_front_view(pts,np.empty((0,7)),WAYMO_VIEWS[int(row['suffix'] or 0)])
    return np.ascontiguousarray(pts,dtype=np.float32),idx


def legacy_parity(e, boxes, pts, idx, actual_model_text):
    c=e['cohort'];r=e['r'];direction=e['source']+'_to_'+e['target']
    if c==REC:return
    key=r['id']
    if c==EXTRA:
        d=ROOT/'crossplatform/round97'/direction
        with np.load(d/'samples_PRIVATE'/(key+'.npz'),allow_pickle=False) as f:
            assert np.array_equal(f['source_boxes'],boxes) and np.array_equal(idx,f['point_indices'])
        receipt=read(d/'records_PRIVATE'/(key+'.json'))['input_receipt']
    else:
        z=ROOT/'crossplatform/round158/evidence_PRIVATE'/f"{e['ordinal']-1:04}.npz"
        with np.load(z,allow_pickle=False) as f:
            assert np.array_equal(f['boxes'],boxes)
        d=ROOT/'crossplatform/round131/formal'/direction
        receipt=read(d/'records_PRIVATE'/(key+'.json'))['source_prefix']
    import hashlib
    assert (receipt['point_sample_sha256'],receipt['point_indices_sha256'],receipt['actual_model_text_sha256'])==(
        array_sha(pts),array_sha(idx),hashlib.sha256(actual_model_text.encode()).hexdigest())


@torch.inference_mode()
def process(model, i, e, lock_id):
    row=e['r'];src=e['source'];target=e['target']
    assert src==model.source and hash_raw(row['pointpath'])==e['point_sha']
    phrase=canonical(row['caption']);text=phrase+SENTINEL
    pts,idx=points_for(row)
    fit_new=phrase!=row['caption']
    if fit_new:
        assert e['cohort']==REC and target=='waymo' and hash_raw(row['imagepath'])==e['image_sha']
        calls=[]
        def collect(_m,_args,out):
            calls.append({k:out[k] for k in ('last_center','last_pred_size','last_proj_queries','proj_tokens')})
        hook=model.base.register_forward_hook(collect)
        try:
            answer=model.score_frame(target=target,points=row['pointpath'],rgb=row['imagepath'],
               calibration=row['metapath'],text=phrase,seed=row['seed'],waymo_view=WAYMO_VIEWS[int(row['suffix'] or 0)])
        finally:
            hook.remove()
        assert len(calls)==1 and np.array_equal(pts,answer.sampled_points)
        out=calls[0]; selected=np.array([answer.probability.argmax(),answer.winner()],dtype=np.int16)
        prepared=np.stack([orient_box(pts,answer.boxes[q]) for q in selected])
    else:
        out=model.base({'point_clouds':torch.as_tensor(pts,device='cuda')[None],'text':[text]})
        selected,expected=reference(e)
    raw=torch.cat((out['last_center'],out['last_pred_size']),dim=-1)[0].cpu().numpy()
    if fit_new:
        assert np.array_equal(raw,answer.boxes)
    else:
        prepared=np.stack([orient_box(pts,raw[q]) for q in selected])
        if not np.array_equal(prepared,expected):
            raise ValueError(f"Sealed winner box differs on {e['cohort']} ordinal {e['ordinal']}")
    legacy_parity(e,raw,pts,idx,text)
    q=out['last_proj_queries'][0].cpu().numpy();t=out['proj_tokens'][0].cpu().numpy()
    mask=firstnoun_mask(model.base,text)[0].cpu().numpy()
    assert raw.shape==(256,6) and q.shape==(256,64) and t.shape==(len(mask),64)
    assert all(np.isfinite(arr).all() for arr in (raw,q,t,prepared))
    npz_atomic(prediction_file(i),raw=raw,q=q,t=t,text_proxy_mask=mask,prepared=prepared,winners=selected,
               sampled=bytes.fromhex(array_sha(pts)),indices=bytes.fromhex(array_sha(idx)),id_in_lock=i,
               lock=lock_id,canonical_rerun=fit_new)


def main():
    check_protocol()
    lock_id=sha(SPEC)
    manifest=rows()
    (WORK/'representations_PRIVATE').mkdir(parents=True,exist_ok=True)
    tic=time.monotonic();files={};sources={}
    for s in PLATFORMS:
        ids=[i for i,e in enumerate(manifest) if e['source']==s]
        model=SingleFrameGrounder(s)
        assert not any(p.requires_grad for p in model.base.parameters())
        for n,i in enumerate(ids,1):
            path=prediction_file(i)
            if path.exists():
                with np.load(path,allow_pickle=False) as f:
                    assert str(f['lock'])==lock_id and int(f['id_in_lock'])==i
            else:process(model,i,manifest[i],lock_id)
            files[path.name]=sha(path)
            if n%256==0:print('forward',s,n,len(ids),'elapsed_s',round(time.monotonic()-tic),flush=True)
        sources[s]=len(ids)
        del model
        gc.collect();torch.cuda.empty_cache()
    assert len(files)==sum(sources.values())==10812
    private=WORK/'REPRESENTATIONS_MANIFEST_PRIVATE.json'
    json_new(private,files)
    json_new(HERE/'FORWARD_SEAL.json',dict(n=len(files),sources=sources,protocol=lock_id,
        manifest_sha256=sha(private),seconds=round(time.monotonic()-tic,3),no_target_annotation_used_in_forward=True,
        exact_A256_point_indices_and_token_input_legacy_val=7740,
        canonical_R158_live_forwards=sum(canonical(e['r']['caption'])!=e['r']['caption'] for e in manifest),
        FINAL_access=False))
    print('forward sealed',len(files),flush=True)


if __name__=='__main__':main()
