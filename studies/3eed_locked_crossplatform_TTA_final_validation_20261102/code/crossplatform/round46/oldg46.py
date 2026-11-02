"""Original old-G algorithms with explicit run-local I/O, no evidence redesign."""
import sys,time,importlib.util,argparse,hashlib
from pathlib import Path
import numpy as np
from common46 import *

SCRIPTS={'parse':'parseM24.py','prepare':'prepareM24.py','detect':'detectM24.py','evidence':'evidenceM24.py','bank':'rankM24.py'}
def detect_fresh(m,plan,rows,base,observed,script):
 import torch,re
 from PIL import Image
 from transformers import AutoProcessor,AutoModelForZeroShotObjectDetection
 folder=base/'dinoM64_attempt001';folder.mkdir();(folder/'cases_PRIVATE').mkdir();(folder/'calls_PRIVATE').mkdir()
 assets=ROOT/'crossplatform/round20/assets/grounding_dino';receipt=read(assets.parent/'DINO_ASSET_RECEIPT.json');assert sha(assets/'model.safetensors')==receipt['model_sha256'];torch.set_num_threads(4);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 processor=AutoProcessor.from_pretrained(assets,local_files_only=True);model,info=AutoModelForZeroShotObjectDetection.from_pretrained(assets,local_files_only=True,use_safetensors=True,output_loading_info=True,disable_custom_kernels=True);assert not any(info.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']);model.eval().requires_grad_(False).cuda();versions=[(n,p._version) for n,p in model.named_parameters()];calls=0;start=time.perf_counter()
 def infer(rgb,query):
  nonlocal calls
  assert calls<plan['max_DINO_calls'];calls+=1;path=folder/'calls_PRIVATE'/(f'{calls:06d}.json');rec={'status':'started','attempt':calls,'query_sha256':hashlib.sha256(query.encode()).hexdigest(),'model_forward_entered':False};dump(path,rec)
  try:
   text=query.casefold().strip(' .')+'.';inputs=processor(images=Image.fromarray(rgb),text=text,return_tensors='pt');assert inputs.input_ids.shape[1]<=256;inputs=inputs.to('cuda');rec['model_forward_entered']=True;dump(path,rec)
   with torch.inference_mode():p=model(**inputs)
   b=p.pred_boxes[0].cpu().numpy();s=p.logits[0].sigmoid().max(-1).values.cpu().numpy();np.savez_compressed(folder/'calls_PRIVATE'/(f'{calls:06d}_raw.npz'),pred_boxes_normalized_cxcywh=b,scores=s);rec['status']='forward_completed_raw_saved';dump(path,rec)
   b=np.c_[b[:,:2]-b[:,2:]/2,b[:,:2]+b[:,2:]/2]*np.array([rgb.shape[1],rgb.shape[0]]*2);take=np.flatnonzero(s>=.25);take=take[m.nms(b[take],s[take])[:10]];result={'boxes_PRIVATE':b[take],'scores':s[take],'query_sha256':hashlib.sha256(text.encode()).hexdigest()};rec['status']='completed';dump(path,rec);return result
  except BaseException as e:rec.update(status='error_preserved_no_retry',error_type=type(e).__name__,error=str(e));dump(path,rec);raise
 for r in rows:
  g=read(base/'textM64_attempt001/cases_PRIVATE'/(r['id']+'.json'));prep=read(base/'preparedM64_attempt001/cases_PRIVATE'/(r['id']+'.json'));payload={'alias':r['alias'],'objects':{}}
  if prep['objects']:
   frame,_,_=observed(r)
   for ref in g['references']:
    rgb=frame['rgb'];aa=infer(rgb,ref['own_query']);allowed=not bool(re.search(r'\b(left|right|clockwise)\b',ref['own_query'],re.I));bb=infer(rgb[:,::-1].copy(),ref['own_query']) if allowed else None
    if bb:
     b=np.array(bb['boxes_PRIVATE']);W=rgb.shape[1];bb['boxes_PRIVATE']=np.c_[W-b[:,2],b[:,1],W-b[:,0],b[:,3]]
    stable=[j for j,box in enumerate(aa['boxes_PRIVATE']) if bb is None or max([m.rect_iou(box,b) for b in bb['boxes_PRIVATE']],default=0)>=.5];payload['objects'][ref['entity']]={'original':aa,'flip':bb,'stable_indices':stable,'reused_exact_query':False}
  dump(folder/'cases_PRIVATE'/(r['id']+'.json'),payload)
 assert versions==[(n,p._version) for n,p in model.named_parameters()];seal(folder,'R46_LITERAL_FRESH_OLDG_DINO_NO_GT',n=len(rows),calls=calls,reused_queries=0,seconds=time.perf_counter()-start,model_unchanged=True,original_program_sha256=sha(script),per_call_started_and_raw_saved=True)
def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=list(SCRIPTS),required=True)
 for k in ('plan','authorization','output'):p.add_argument('--'+k,required=True)
 a=p.parse_args();plan,rows,out=context(a);verify(out/'source');base=out/'oldG';base.mkdir(exist_ok=True)
 originals={r['id']:dict(r,stage=r.get('stage','F')) for r in rows};derived=[dict(r,stage='M',alias=r.get('alias',r['id'])) for r in rows]
 if not (base/'input/M64_PRIVATE.json').exists():dump(base/'input/M64_PRIVATE.json',derived)
 else:assert read(base/'input/M64_PRIVATE.json')==derived
 if not (base/'capture336_attempt001').exists():(base/'capture336_attempt001').symlink_to(out/'source',target_is_directory=True)
 sys.path.insert(0,str(ROOT/'crossplatform/round24'))
 import common24
 # common24's existing chain resolves the pinned old observation/geometry modules.
 from observations import load as original_load
 def observed(r):return fresh_observation(originals[r['id']],out) if plan.get('fresh_frames') else original_load(originals[r['id']])
 script=ROOT/'crossplatform/round24'/SCRIPTS[a.stage];spec=importlib.util.spec_from_file_location('R46_original_'+a.stage,script);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 m.BASE=base;m.EXP=base/'experiment';m.feature=lambda r:npz(out/'source/samples_PRIVATE'/(r['id']+'.npz'));m.load=observed
 def locked(_name):
  # Legacy lock path is replaced only by the outer complete R46 code/input lock.
  for rel,h in plan['program_sha256'].items():assert sha(ROOT/rel)==h,rel
 m.lock=locked
 legacyseal=m.seal
 def actualseal(folder,state,**kw):
  if 'n' in kw:kw['legacy_engine_n_metadata']=kw['n'];kw['n']=len(rows)
  kw.update(original_program_sha256=sha(script),run_local_io_adapter=True,all_current_G_uses_frozen_source=True)
  return legacyseal(folder,state,**kw)
 m.seal=actualseal
 if a.stage=='detect':detect_fresh(m,plan,derived,base,observed,script);return
 if a.stage!='evidence':m.main();return
 # Literal fresh M-style evidence loop; only the old cohort-sized 5000-call
 # execution budget is replaced by the registered 68*n upper bound.
 from PIL import Image
 old=read(ROOT/'research/round15/experiment/SEMANTIC_PLAN_V4_LOCK.json');modeldir=Path(old['model_dir'])
 assert all(sha(modeldir/f)==h for f,h in old['model_files_sha256'].items())
 engine=m.Evidence(modeldir,m.DEFAULT_CONFIG);engine.setup_labels();folder=base/'evidenceM64_attempt001';folder.mkdir();(folder/'cases_PRIVATE').mkdir();(folder/'numeric_calls_PRIVATE').mkdir();start=time.perf_counter();calls=0
 originalscore=engine.score;actualnumeric=0;active={}
 def before_forward(module,inputs):
  active['record']['model_forward_entered']=True;dump(active['path'],active['record'])
 def after_forward(module,inputs,result):
  t=engine.torch;v=result.logits[0,-1].float();yt=t.logsumexp(v[engine.ids['yes']],0);nt=t.logsumexp(v[engine.ids['no']],0);active['record'].update(status='forward_completed_raw_numeric_saved',yes_logsumexp=float(yt),no_logsumexp=float(nt),raw_yes_no_logodds=float(yt-nt));dump(active['path'],active['record'])
 hooks=[engine.model.register_forward_pre_hook(before_forward),engine.model.register_forward_hook(after_forward)]
 def scored(image,query):
  nonlocal actualnumeric
  assert actualnumeric<plan['max_numeric_Qwen_calls'],'Stop before exceeding the locked forward cap'
  actualnumeric+=1;active['path']=folder/'numeric_calls_PRIVATE'/(f'{actualnumeric:06d}.json');active['record']={'status':'started','attempt':actualnumeric,'query_PRIVATE':query,'numeric_forward':True,'generated_tokens':0,'model_forward_entered':False};dump(active['path'],active['record'])
  try:
   result=originalscore(image,query);active['record'].update(result,status='completed');dump(active['path'],active['record']);return result
  except BaseException as e:active['record'].update(status='error_preserved_no_retry',error_type=type(e).__name__,error=str(e));dump(active['path'],active['record']);raise
 engine.score=scored
 for i,r in enumerate(derived):
  prep=read(base/'preparedM64_attempt001/cases_PRIVATE'/(r['id']+'.json'));g=read(base/'textM64_attempt001/cases_PRIVATE'/(r['id']+'.json'));d=read(base/'dinoM64_attempt001/cases_PRIVATE'/(r['id']+'.json'));payload={'alias':r['alias'],'targets_PRIVATE':{},'references':{}}
  if prep['mask_pool_PRIVATE']:
   frame,scene,binding=observed(r);assert binding==prep['bindings']
   for mi in prep['mask_pool_PRIVATE']:
    views=[]
    for image in m.crops(frame['rgb'],scene['objects'][mi]):
     full=engine.score(image,g['subject_query']);gray=engine.score(Image.new('RGB',image.size,(127,127,127)),g['subject_query']);calls+=2;views.append({'full':full,'gray':gray,'visual_evidence':full['log_odds_yes_no']-gray['log_odds_yes_no']})
    payload['targets_PRIVATE'][mi]=views
   for ref in g['references']:
    det=d['objects'][ref['entity']];obs=[]
    for j in det['stable_indices'][:3]:
     box=det['original']['boxes_PRIVATE'][j];image=m.reference_crop(frame['rgb'],box);full=engine.score(image,ref['phrase']);gray=engine.score(Image.new('RGB',image.size,(127,127,127)),ref['phrase']);calls+=2;obs.append({'proposal_index':j,'box_PRIVATE':box,'detector_score':float(det['original']['scores'][j]),'full':full,'gray':gray,'visual_evidence':full['log_odds_yes_no']-gray['log_odds_yes_no']})
    payload['references'][ref['entity']]=obs
  dump(folder/'cases_PRIVATE'/(r['id']+'.json'),payload);assert calls<=68*len(rows)
  if (i+1)%16==0:print('R46_OLDG_EVIDENCE',i+1,calls,flush=True)
 for hook in hooks:hook.remove()
 assert engine.verify_unchanged() and calls==actualnumeric;seal(folder,'R46_LITERAL_FRESH_OLDG_EVIDENCE_NO_GT',n=len(rows),new_calls=calls,reused_numeric_forwards=0,non_numeric_VLM_calls=0,non_numeric_basis='typed_subject_and_reference_parser_is_local_Stanza_no_VLM_generate;Evidence_uses_forward_logits_only',model_unchanged=True,seconds=time.perf_counter()-start,peak_bytes=engine.torch.cuda.max_memory_allocated(),original_program_sha256=sha(script),only_budget_replaced='historical5000executionguard_to_68times_registeredn',evidence_prompt_and_full_gray_reference_rules_unchanged=True)
 print('R46_OLDG_EVIDENCE_SEALED',calls,flush=True)
if __name__=='__main__':main()
