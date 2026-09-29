"""Offline cached specialists for the full locked panel. No labels or selection metrics."""
import os,sys,time,gc,hashlib,traceback,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,save,load,sha,status
OUT=ROOT/'artifacts/tastvg_full_b1_v1'

def observation(row,condition,frames):
 from vg_tta.tastvg_deployment_corruption_v2 import burst_spec,apply_burst
 from scripts.c1_controlled_corruption_v1 import pixelhash
 if condition=='clean':return frames,pixelhash(frames),None
 family,severity=condition.rsplit('_',1);spec=burst_spec(row['source'],row['input']['frame_count'],row['frame_ids'],int(severity))
 shifted=apply_burst(frames,row['input'],spec,family,row['source']);return shifted,pixelhash(shifted),spec

def verify():
 p=read(OUT/'EXECUTION_LOCK.json')
 pins=dict(p['pins'])
 revision=OUT/'IMPLEMENTATION_REVISION_001.json'
 if revision.exists():
  r=read(revision);assert r['original_execution_lock_sha256']==sha(OUT/'EXECUTION_LOCK.json');pins.update(r['pin_overrides'])
 for f,h in pins.items():assert sha(ROOT/f)==h,f
 assert sha(OUT/'ROSTER_LOCK.json')==p['roster_sha256']
 return read(OUT/'ROSTER_LOCK.json')

def guard(event,args):
 if event=='open' and args and isinstance(args[0],(str,bytes)) and any(x in str(args[0]) for x in ['labels_diagnostic','GT_SUBSET','/ROWS.json','/SUMMARY.json','test_annotations.json']):raise PermissionError('full B1 inference forbids GT/metrics')

def budget(tick):
 prior=sum(read(f)['seconds'] for f in (OUT/'allocations').glob('*.json'))
 assert prior+time.monotonic()-tick<7*86400-60,'full-run engineering time ceiling'
 assert shutil.disk_usage(ROOT).free>8*2**30,'free space floor'

def run(stage,limit=0):
 p=verify();assert read(ROOT/'artifacts/tastvg_matched_ablation_a1_v1/DECISION.json')['status']=='completed'
 tick=time.monotonic();done=created=0;failure=None;state='failed';lease=None
 try:
  os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
  import numpy as np,torch
  from PIL import Image
  from vg_tta.exact_frame_decode_audit_v2 import decode
  from scripts.run_final_simplification_v1 import lease as gpu_lease
  from methods.decota_final_simplified_v1.tensors import state_hash
  sys.addaudithook(guard);lease=gpu_lease();torch.set_num_threads(4);torch.manual_seed(20260929);np.random.seed(20260929);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=(stage=='spatial')
  print('BACKEND',stage,'deterministic',torch.backends.cudnn.deterministic,flush=True)
  if stage=='spatial':
   from transformers import AutoModel,AutoTokenizer
   from vg_tta.tastvg_spatial_expansion_s0_v1 import mask_boxes
   model_path=ROOT/'checkpoints/Sa2VA-4B';receipt=read(model_path/'DOWNLOAD_RECEIPT.json');code=read(model_path/'OFFICIAL_CODE_RECEIPT.json')
   for f in receipt['files']:assert sha(model_path/f['file'])==f['sha256']
   for f,h in code['files'].items():assert sha(model_path/f)==h['sha256']
   model,info=AutoModel.from_pretrained(str(model_path),torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,use_flash_attn=False,trust_remote_code=True,local_files_only=True,output_loading_info=True)
   assert not any(info[k] for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']);model=model.eval().cuda().requires_grad_(False)
   tokenizer=AutoTokenizer.from_pretrained(str(model_path),trust_remote_code=True,use_fast=False,local_files_only=True);model.preparing_for_generation(tokenizer,max_new_tokens=256,torch_dtype=torch.bfloat16)
   assert all(float(m.fill_hole_area)==0 for m in model.modules() if hasattr(m,'fill_hole_area'))
  else:
   repo=ROOT/'external/UniversalVTG';sys.path[:0]=[str(repo),str(repo/'perception_models')]
   from universal_vtg_inference import UniversalVTG
   model=UniversalVTG(experiment_name=str(ROOT/'checkpoints/universalvtg'),device='cuda',enable_query_unifier=False);model._ensure_video_encoder();model._ensure_text_encoder()
   for m in [model.model,model._video_extractor,model._text_model]:m.eval().requires_grad_(False)
  total=len(p['expert_needed'])*len(p['conditions'])
  for parent in p['expert_needed']:
   row=p['rows'][parent];frames=None;text=None
   if all((OUT/stage/c/f'{parent:05}.json').exists() for c in p['conditions']):
    for c in p['conditions']:
     z=read(OUT/stage/c/f'{parent:05}.json');assert sha(OUT/z['cache'])==z['cache_sha256']
    done+=len(p['conditions']);continue
   for cond in p['conditions']:
    f=OUT/stage/cond/f'{parent:05}.json'
    if f.exists():z=read(f);assert sha(OUT/z['cache'])==z['cache_sha256'];done+=1;continue
    if limit and created>=limit:break
    budget(tick)
    if frames is None:frames,ids=decode(row['input']);assert ids==row['frame_ids']
    shifted,pixel,spec=observation(row,cond,frames);start=time.monotonic()
    if stage=='spatial':
     pos=np.rint(np.linspace(0,len(ids)-1,5)).astype(int).tolist();samples=[shifted[i] for i in pos]
     digest=hashlib.sha256(row['input']['caption'].encode()+str(pos).encode()+str(shifted.shape).encode()+b''.join(x.tobytes() for x in samples)).hexdigest()
    else:
     duration=(ids[-1]-ids[0]+1)/row['input']['fps'];slots=np.arange(max(1,int(duration*2)))/2;physical=ids[0]+slots*row['input']['fps'];pick=np.abs(np.asarray(ids)[None,:]-physical[:,None]).argmin(1)
     digest=hashlib.sha256(row['input']['caption'].encode()+str(ids).encode()+str(row['input']['fps']).encode()+str(shifted.shape).encode()+b''.join(shifted[i].tobytes() for i in np.unique(pick))).hexdigest()
    cache=OUT/'expert_cache'/stage/f'{digest}.pt';fresh=not cache.exists()
    if fresh:
     if stage=='spatial':
      prompt='<image>Please segment the object described by: '+row['input']['caption'].rstrip('.')+'.'
      with torch.inference_mode():raw=model.predict_forward(video=[Image.fromarray(im) for im in samples],text=prompt,tokenizer=tokenizer)
      mm=raw['prediction_masks'];mask=mm[0] if len(mm) else None;valid,boxes=mask_boxes(mask,pos,len(ids))
      value=dict(valid=valid,boxes=boxes,positions=pos,mask_count=len(mm),mask_shape=list(np.asarray(mask).shape) if mask is not None else None,mask_bits=np.packbits(np.asarray(mask,dtype=bool)) if mask is not None else None,prediction_text=raw['prediction'])
     else:
      if text is None:text=model.encode_text(row['input']['caption'])
      unique,inverse=np.unique(pick,return_inverse=True);features=[]
      with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
       for at in range(0,len(unique),16):
        pixels=torch.stack([model._video_preprocess(Image.fromarray(shifted[i])) for i in unique[at:at+16]]).cuda().half();features.append(model._video_extractor(pixels).float().cpu())
      video=torch.cat(features)[inverse].T.contiguous()
      with torch.inference_mode():raw=model.predict(video,text,return_raw=True,fps=None,feature_fps=2.,duration=duration,use_unifier=False)
      seg=raw['raw_segments'];scores=raw['raw_scores']
      if isinstance(seg,list):seg=seg[0];scores=scores[0]
      sec=model._convert_segments_to_seconds(seg,fps=None,feature_fps=2.).clamp(0,duration).float().cpu().numpy();conf=scores.float().cpu().numpy();valid=sec[:,1]>sec[:,0]
      assert np.isfinite(sec).all() and np.isfinite(conf).all()
      value=dict(proposals=(sec[valid]*row['input']['fps']+ids[0]).tolist(),proposal_confidence=conf[valid].tolist(),duration=duration,picked_observations=pick.tolist(),video_features=video,text_features=text.cpu())
     save(cache,dict(**value,GT_read=False,input_sha256=digest,seconds=time.monotonic()-start));created+=1
    write(f,dict(parent=parent,condition=cond,pixel_sha256=pixel,cache=str(cache.relative_to(OUT)),cache_sha256=sha(cache),new_inference=fresh,seconds=time.monotonic()-start,GT_read=False));done+=1
    status(OUT/f'{stage.upper()}_STATUS.json',dict(status='running',done=done,total=total,new_inferences_this_run=created,seconds=time.monotonic()-tick));print(stage,done,total,'new',created,'sec',round(time.monotonic()-tick,2),flush=True)
    del shifted;gc.collect();torch.cuda.empty_cache()
   if limit and created>=limit:break
  verify();state='completed' if done==total else 'bounded_batch_complete'
  if state=='completed':write(OUT/f'{stage.upper()}_BARRIER.json',dict(cells=total,files={str(f.relative_to(OUT)):sha(f) for f in (OUT/stage).rglob('*.json')},GT_read=False,time=time.time()))
 except BaseException as e:failure=dict(error=repr(e),traceback=traceback.format_exc());raise
 finally:
  receipt=dict(stage=stage,status=state,done=done,new_inferences=created,seconds=time.monotonic()-tick,failure=failure,time=time.time());write(OUT/'allocations'/f'{time.time_ns()}.json',receipt);status(OUT/f'{stage.upper()}_STATUS.json',receipt)
  if lease:lease.close()

if __name__=='__main__':
 import argparse
 a=argparse.ArgumentParser();a.add_argument('stage',choices=['spatial','temporal']);a.add_argument('--limit',type=int,default=0);x=a.parse_args();run(x.stage,x.limit)
