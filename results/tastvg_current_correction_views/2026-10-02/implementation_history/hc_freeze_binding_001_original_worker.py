"""Receipted Sa2VA observation and a genuine shifted UniversalVTG view."""
import os,sys,time,hashlib,gc,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
from scripts.tastvg_correction_views_common_v1 import *

def run(stage,split):
    import torch,numpy as np
    from PIL import Image
    from scripts.run_final_simplification_v1 import lease
    from scripts.run_tastvg_full_b1_experts_v1 import observation
    from vg_tta.tastvg_reference_selection_v1 import student_frames
    from vg_tta.tastvg_current_correction_views_v1 import temporal,uniform_second
    from vg_tta.tastvg_spatial_expansion_s0_v1 import mask_boxes
    from vg_tta.exact_frame_decode_audit_v2 import decode as vd
    from vg_tta.tastvg_paper48_hc2_decode_v1 import decode as hd
    verify();sys.addaudithook(guard);lh=lease();tick=time.monotonic();calls=reused=0
    torch.set_num_threads(4);torch.manual_seed(20260929);np.random.seed(20260929)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=stage!='temporal'
    registry={};model=tokenizer=None
    def spatial_model():
        nonlocal model,tokenizer
        if model is not None:return
        from transformers import AutoModel,AutoTokenizer
        mp=ROOT/'checkpoints/Sa2VA-4B'
        for r in read(mp/'DOWNLOAD_RECEIPT.json')['files']:assert sha(mp/r['file'])==r['sha256']
        for f,h in read(mp/'OFFICIAL_CODE_RECEIPT.json')['files'].items():assert sha(mp/f)==h['sha256']
        model,info=AutoModel.from_pretrained(str(mp),torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,
            use_flash_attn=False,trust_remote_code=True,local_files_only=True,output_loading_info=True)
        assert not any(info[k] for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs'])
        model=model.eval().cuda().requires_grad_(False)
        tokenizer=AutoTokenizer.from_pretrained(str(mp),trust_remote_code=True,use_fast=False,local_files_only=True)
        model.preparing_for_generation(tokenizer,max_new_tokens=256,torch_dtype=torch.bfloat16)
        assert all(float(m.fill_hole_area)==0 for m in model.modules() if hasattr(m,'fill_hole_area'))
    if stage=='temporal':
        repo=ROOT/'external/UniversalVTG';sys.path[:0]=[str(repo),str(repo/'perception_models')]
        from universal_vtg_inference import UniversalVTG
        model=UniversalVTG(experiment_name=str(ROOT/'checkpoints/universalvtg'),device='cuda',enable_query_unifier=False)
        model._ensure_video_encoder();model._ensure_text_encoder()
        for m in [model.model,model._video_extractor,model._text_model]:m.eval().requires_grad_(False)
    else:
        # Only metadata receipts are indexed; no old outcome scores are opened.
        for ds in DATASETS:
            directories=[ROOT/'artifacts/tastvg_reference_selection_v1'/ds/'receipts',OLD/ds/'expert_receipts',BASE/ds/'expert_receipts']
            for directory in directories:
                for f in directory.rglob('*.json'):
                    r=read(f)
                    if 'input_sha256' not in r or 'cache' not in r:continue
                    raw=r['cache'];poss=[ROOT/raw,ROOT/'artifacts/tastvg_reference_selection_v1'/raw,OLD/raw,BASE/raw]
                    cf=next((q for q in poss if q.exists()),None)
                    if cf is not None and sha(cf)==r['cache_sha256']:registry[r['input_sha256']]=(cf,sha(f))
    jobs=[r for r in read(BASE/'COHORT.json')['cells'] if r['split']==split and r['scheduled']]
    for done,cell in enumerate(jobs,1):
        budget();ds=cell['dataset'];p=plan(ds);row=p['rows'][cell['parent']];cond=cell['condition']
        x=oldcell(ds,split,cond,cell['order'],cell['arrival']);tc=x['temporal']['candidates']
        prefix=f'{split}_{cond}_{cell["order"]}_{cell["arrival"]:05}'
        frames,ids=(hd if ds=='hc2' else vd)(row['input']);assert ids==row['frame_ids']
        shifted,pixel,_=observation(row,cond,frames);assert pixel==cell['pixel_sha256']
        if stage=='temporal':
            path=BASE/ds/'views'/f'{prefix}.json'
            if path.exists():assert sha(ROOT/read(path)['cache'])==read(path)['cache_sha256'];continue
            e0,r0=expert(ds,'temporal',cell['parent'],cond,pixel);duration=e0['duration'];phase=.25
            assert duration>phase;slots=phase+np.arange(max(1,int(duration*2)))/2
            pick=np.abs(np.asarray(ids)[None,:]-(ids[0]+slots*row['input']['fps'])[:,None]).argmin(1)
            digest=hashlib.sha256(row['input']['caption'].encode()+str(ids).encode()+str(pick.tolist()).encode()+str(phase).encode()+b''.join(shifted[i].tobytes() for i in np.unique(pick))).hexdigest()
            cf=BASE/ds/'temporal_cache'/f'{digest}.pt';fresh=not cf.exists();start=time.monotonic()
            if fresh:
                unique,inverse=np.unique(pick,return_inverse=True);features=[]
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
                    for at in range(0,len(unique),16):
                        pixels=torch.stack([model._video_preprocess(Image.fromarray(shifted[i])) for i in unique[at:at+16]]).cuda().half()
                        features.append(model._video_extractor(pixels).float().cpu())
                video=torch.cat(features)[inverse].T.contiguous()
                with torch.inference_mode():raw=model.predict(video,e0['text_features'],return_raw=True,fps=None,feature_fps=2.,duration=duration-phase,use_unifier=False)
                seg,scores=raw['raw_segments'],raw['raw_scores']
                if isinstance(seg,list):seg,scores=seg[0],scores[0]
                sec=model._convert_segments_to_seconds(seg,fps=None,feature_fps=2.).clamp(0,duration-phase).float().cpu().numpy();conf=scores.float().cpu().numpy();valid=sec[:,1]>sec[:,0]
                assert np.isfinite(sec).all() and np.isfinite(conf).all()
                save(cf,dict(proposals=((sec[valid]+phase)*row['input']['fps']+ids[0]).tolist(),proposal_confidence=conf[valid].tolist(),
                    picked_observations=pick.tolist(),actual_frame_ids=[ids[i] for i in pick],phase=phase,duration=duration-phase,
                    input_sha256=digest,video_features=video,GT_read=False,seconds=time.monotonic()-start))
                calls+=1
            else:reused+=1
            value=load(cf);rule=temporal(tc,[e0,value]);oldpick=np.asarray(e0['picked_observations'])
            assert oldpick.shape==pick.shape
            write(path,dict(cache=str(cf.relative_to(ROOT)),cache_sha256=sha(cf),original_view=r0,
                input_sha256=digest,pixel_sha256=pixel,rule=rule,phase=.25,new_call=fresh,
                actual_observation_difference=float(np.mean(oldpick!=pick)),original_pick=oldpick.tolist(),
                shifted_pick=pick.tolist(),seconds=time.monotonic()-start,GT_read=False))
        else:
            routes={'R':student_frames(tc,ids)['positions']} if stage=='round1' else {
                'Rnew':student_frames([tc[read(BASE/ds/'views'/f'{prefix}.json')['rule']['selected']]],ids)['positions'],
                'U2':uniform_second(len(ids))}
            for branch,pos in routes.items():
                path=BASE/ds/'expert_receipts'/f'{prefix}_{branch}.json'
                if path.exists():assert sha(ROOT/read(path)['cache'])==read(path)['cache_sha256'];continue
                samples=[shifted[i] for i in pos]
                digest=hashlib.sha256(row['input']['caption'].encode()+str(pos).encode()+str(shifted.shape).encode()+b''.join(a.tobytes() for a in samples)).hexdigest()
                start=time.monotonic();fresh=digest not in registry
                if fresh:
                    spatial_model();attempt=BASE/ds/'attempts'/f'{prefix}_{branch}.json'
                    write(attempt,dict(input_sha256=digest,positions=pos,GT_read=False,time=time.time()))
                    prompt='<image>Please segment the object described by: '+row['input']['caption'].rstrip('.')+'.'
                    with torch.inference_mode():raw=model.predict_forward(video=[Image.fromarray(a) for a in samples],text=prompt,tokenizer=tokenizer)
                    masks=raw['prediction_masks'];mask=masks[0] if len(masks) else None;valid,boxes=mask_boxes(mask,pos,len(ids))
                    value=dict(valid=valid,boxes=boxes,positions=pos,mask_count=len(masks),
                        mask_shape=list(np.asarray(mask).shape) if mask is not None else None,
                        mask_bits=np.packbits(np.asarray(mask,dtype=bool)) if mask is not None else None,
                        prediction_text=raw['prediction'],input_sha256=digest,GT_read=False)
                    cf=BASE/ds/'spatial_cache'/f'{digest}.pt';save(cf,value);registry[digest]=(cf,None);calls+=1
                else:cf,original=registry[digest];value=load(cf);assert value['input_sha256']==digest;reused+=1
                assert value['positions']==pos
                write(path,dict(cache=str(cf.relative_to(ROOT)),cache_sha256=sha(cf),input_sha256=digest,
                    pixel_sha256=pixel,positions=pos,branch=branch,new_call=fresh,valid_frames=int(value['valid'].sum()),
                    seconds=time.monotonic()-start,origin='new_inference' if fresh else 'input_hash_matched_cache',GT_read=False))
        status(BASE/'WORKER_STATUS.json',dict(stage=stage,split=split,status='running',done=done,total=len(jobs),
            worker_pid=os.getpid(),dataset=ds,new_calls=calls,GT_read=False,time=time.time()))
        print('OBSERVATIONS',stage,split,ds,done,len(jobs),flush=True);del frames,shifted;gc.collect();torch.cuda.empty_cache()
    write(BASE/f'{stage}_{split}_RESOURCES.json',dict(stage=stage,split=split,new_calls=calls,reused=reused,
        worker_wall_seconds=time.monotonic()-tick,wall_includes_loading_IO=True,GT_read=False,time=time.time()))
    status(BASE/'WORKER_STATUS.json',dict(status='completed',stage=stage,split=split,GT_read=False,time=time.time()))
    lh.close()
if __name__=='__main__':
    try:run(sys.argv[1],sys.argv[2])
    except BaseException as e:
        status(BASE/'WORKER_STATUS.json',dict(status='failed',error=repr(e),traceback=traceback.format_exc(),time=time.time()));raise
