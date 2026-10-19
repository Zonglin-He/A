"""Finite confirmation of frozen lambda. No scoring/annotations imported here."""
from pathlib import Path
import sys,time,gc,collections,hashlib,json
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,save,sha
D=R/'artifacts/stvg_temporal_increment_confirm109_v1';P=R/'results/stvg_error_state/2026-10-19/temporal_increment_confirm109'
DEV=R/'artifacts/stvg_temporal_increment128_v1';FINAL=R/'artifacts/stvg_referdino_finalize_v1'

def prepare():
 import torch
 assert not(D/'PLAN_LOCK.json').exists();D.mkdir(exist_ok=True);P.mkdir(parents=True,exist_ok=True)
 old=read(DEV/'PLAN_LOCK.json');sel=read(DEV/'SELECTION_LOCK.json');assert sel['lambda_value']==.5
 raw=read(R/'artifacts/decota_paper_experiments_v1/hc2/PLAN.json')['rows'];rf={r['query_ordinal']:r for r in read(FINAL/'PLAN_LOCK.json')['formal_records']};known={}
 sp=R/'artifacts/stvg_dino_supported_source256_v1'
 for r in read(sp/'PLAN_LOCK.json')['records']:
  if r['dataset']=='hc2':known[r['query_ordinal']]=dict(path=str((sp/'source'/f"{r['alias']}.pt").relative_to(R)),kind='source256')
 for r in read(R/'artifacts/stvg_posterior_decision_tree_v1/PLAN_LOCK.json')['files']:
  if r['dataset']=='hc2'and r['query_ordinal']not in known:known[r['query_ordinal']]=dict(path=r['path'],kind='P6')
 records=[]
 for r in old['confirmation_reserved']:
  q=r['query_ordinal'];path=R/f'artifacts/stvg_opd_paper_hc2_revision_v2/inputs/vidstg_to_hc2/clean/{q:05}.pt';z=torch.load(path,map_location='cpu',weights_only=False);assert z['GT_read']==False and z['parent']==q;assert z['frame_ids']==raw[q]['frame_ids']
  sub=R/f'artifacts/decota_paper_experiments_v1/hc2/subjects/{q:05}.json';sr=read(sub);assert not sr['GT_read'];reuse=None;prior=None
  if q in rf:
   x=rf[q];ip=FINAL/'formal/inputs'/f"{x['alias']}.pt";pp=FINAL/'formal/predictions'/f"{x['alias']}.pt";zz=load(ip);ss=load(pp);assert torch.equal(zz['Source'],z['native_boxes']);assert all(f'Live_{s}'in ss['outputs']for s in [41,42,43]);reuse=dict(input=str(ip.relative_to(R)),input_sha256=sha(ip),prediction=str(pp.relative_to(R)),prediction_sha256=sha(pp),sampling_key=ss['sampling_key'])
  if q in known:
   v=known[q];x=torch.load(R/v['path'],map_location='cpu',weights_only=False);c=x['capture']if v['kind']=='P6'else x;b=c['zero']['boxes']if v['kind']=='P6'else c['Source'];assert torch.equal(b.cpu(),z['native_boxes']);prior=dict(**v,sha256=sha(R/v['path']))
  records.append(dict(r,old_input=str(path.relative_to(R)),old_input_sha256=sha(path),subject_path=str(sub.relative_to(R)),subject_sha256=sha(sub),reuse_spatial=reuse,reuse_prior=prior,frame_count=raw[q]['input']['frame_count']))
 assert len(records)==109 and len({r['parent']for r in records})==109 and not({r['parent']for r in records}&{r['parent']for r in old['records']})
 code=['scripts/run_stvg_temporal_increment_confirm109_v1.py','protocols/stvg_temporal_increment_confirm109_v1.md','vg_tta/stvg_temporal_increment128_v1.py','vg_tta/stvg_where_when_event128_v1.py','vg_tta/stvg_posterior_decision_tree_v1.py','methods/stvg_referdino_opd_v1/config.json','methods/stvg_referdino_opd_v1/policy.py','methods/stvg_referdino_opd_v1/state.py','methods/stvg_referdino_opd_v1/_source_forward.py','vg_tta/stvg_referdino_condition_hc128_v1.py']
 p=dict(records=records,count=109,seeds=[41,42,43],lambdas=[0.,.5],selected_lambda=.5,selection_lock_sha256=sha(DEV/'SELECTION_LOCK.json'),development_original_result_sha256=sel['results_sha256'],original_roster_plan_sha256=sha(DEV/'PLAN_LOCK.json'),config=read(R/'methods/stvg_referdino_opd_v1/config.json'),pins={f:sha(R/f)for f in code},created_epoch=time.time(),deadline_epoch=time.time()+4*3600,GPU_seconds_cap=1800,stage_cap=900,new_teacher_video_cap=109,new_teacher_image_cap=872,new_DINO=0,new_training=0,new_temporal_OPD=False,spatial_reuse=sum(r['reuse_spatial']is not None for r in records),prior_reuse=sum(r['reuse_prior']is not None for r in records),GT_read=False)
 put(D/'PLAN_LOCK.json',p);put(P/'PROTOCOL_PUBLIC.json',{k:v for k,v in p.items()if k not in ['records','pins']});put(P/'EXPOSURE_PUBLIC.json',dict(parents=109,disjoint_from_lambda_selection=True,not_untouched_method_test=True,exposure='earlier full Source/old-method exposure; RH/RF/NH membership per alias below',rows=[dict(alias=r['alias'],prior_panels=r['prior_panels'],exact_spatial_reuse=bool(r['reuse_spatial']))for r in records]));status('prepared',0);print('prepared',p['spatial_reuse'],p['prior_reuse'],flush=True)

def status(stage,done,**kw):put(D/'STATUS.json',dict(stage=stage,done=done,total=109,pid=__import__('os').getpid(),epoch=time.time(),**kw),True)
def verify():
 from main_method.runtime import resource_guard
 p=read(D/'PLAN_LOCK.json');assert time.time()<p['deadline_epoch'];assert all(sha(R/f)==h for f,h in p['pins'].items());assert sha(DEV/'SELECTION_LOCK.json')==p['selection_lock_sha256'];resource_guard();return p

def runlease(stage):
 from methods.stvg_referdino_opd_v1.runtime import lease
 p=verify();used=sum(read(f)['GPU_seconds']for f in (D/'cost').glob('*.json'));lock=lease();start=time.perf_counter()
 def check():
  verify();elapsed=time.perf_counter()-start;assert elapsed+3<p['stage_cap']and used+elapsed+3<p['GPU_seconds_cap'];import torch;assert torch.cuda.mem_get_info()[0]>4*2**30
 return lock,start,check

def source():
 import torch,numpy as np
 from methods.stvg_referdino_opd_v1.runtime import setup,load_source,decode
 from methods.stvg_referdino_opd_v1._source_forward import frozen_forward
 from methods.stvg_referdino_opd_v1.initialization import encode,estimate,apply
 from methods.stvg_referdino_opd_v1.experts import positions8
 from methods.decota_final_simplified_v1.tensors import detached,state_hash
 from scripts.run_decota_paper_main_v1 import unpack_expert
 from scripts.stvg_temporal_marginal_decode_util256_v1 import probabilities,mass_table
 from vg_tta.stvg_posterior_decision_tree_v1 import final_mbr
 setup();p=verify();raw=read(R/'artifacts/decota_paper_experiments_v1/hc2/PLAN.json')['rows'];lock,start,check=runlease('source');model=None;calls=0;files=[]
 try:
  for r in p['records']:
   a=r['alias'];f=D/'inputs'/f'{a}.pt'
   if f.exists():load(f);files.append(read(f.with_suffix('.receipt.json')));continue
   check();assert sha(R/r['old_input'])==r['old_input_sha256'];old=torch.load(R/r['old_input'],map_location='cpu',weights_only=False);src=old['native_boxes'];ex=unpack_expert(old['expert']);pool=[]
   for v in ex['anchors']['single4']:
    pos=v['position'];pr=ex['observations'][('original',pos)]['probe'];assert pr['accepted'];pool.append(dict(position=pos,boxes=np.asarray(pr['boxes'],float).reshape(-1,4),scores=np.asarray(pr['target_scores'],float)))
   cal=p['config']['E_calibration'];est=estimate([encode(v['boxes'])-encode(src.double().numpy())[v['position']]for v in pool],[v['scores']for v in pool],cal['scale'],cal['truncation'],1/16);E=apply(src.double().numpy(),est['delta']);p8=positions8(old['frame_ids'],ex['positions4']);recs=None;features=None;logits=None
   if r['reuse_spatial']:
    v=r['reuse_spatial'];assert sha(R/v['input'])==v['input_sha256'];cached=load(R/v['input']);assert np.array_equal(E,cached['E'])and p8==cached['planned_positions'];features=cached['spatial_inputs'];recs=cached['records']
   if r['reuse_prior']:
    v=r['reuse_prior'];assert sha(R/v['path'])==v['sha256'];xx=torch.load(R/v['path'],map_location='cpu',weights_only=False);cc=xx['capture']if v['kind']=='P6'else xx;logits=cc['zero']['logits']if v['kind']=='P6'else cc['logits'];recs0=cc['records'];assert recs is None or recs==recs0;recs=recs0
   seconds=0.
   if features is None or logits is None:
    if model is None:model=load_source();mh=state_hash(model.state_dict())
    row=dict(raw[r['query_ordinal']]);assert sha(R/r['subject_path'])==r['subject_sha256'];row['parses']=read(R/r['subject_path'])['parses'];tick=time.perf_counter();frames,ids=decode(row['input']);batch,rr,base=frozen_forward(model,frames,row);assert torch.equal(base.zero['boxes'].cpu(),src);assert recs is None or rr==recs;recs=rr;ll=[x.detach().cpu()for x in base.zero['logits']]
    if logits is not None:assert all(torch.equal(x,y)for x,y in zip(ll,logits))
    logits=ll;features=detached(base.spatial_inputs,'cpu');calls+=1;seconds=time.perf_counter()-tick;base.restore(base.initial);del frames,batch,base
   ids,q=mass_table(probabilities(recs,logits));iv=final_mbr(ids,q)['interval'];assert ids==old['frame_ids']
   files.append(save(f,dict(alias=a,Source=src,E=E,frame_ids=ids,interval=iv,planned_positions=p8,original_positions4=ex['positions4'],spatial_inputs=features,records=recs,logits=logits,q0=q,Source_exact=True,old_DINO_reused=True,Source_seconds=seconds,GT_read=False)));status('source',len(files),new_Source=calls);print('source',a,calls,round(time.perf_counter()-start,1),flush=True);gc.collect();torch.cuda.empty_cache()
  if model is not None:assert state_hash(model.state_dict())==mh
  put(D/'SOURCE_BARRIER.json',dict(count=109,files=files,new_Source=calls,all_Source_exact=True,GT_read=False))
 finally:put(D/'cost'/f'source_{time.time_ns()}.json',dict(GPU_seconds=time.perf_counter()-start+3,new_Source=calls,Source_offset_forwards=2*calls,new_ReferDINO=0,new_DINO=0));lock.close()

def teacher():
 import torch,numpy as np
 from methods.stvg_referdino_opd_v1.runtime import setup,decode
 from vg_tta.stvg_referdino_condition_hc128_v1 import ConditionalRefer
 from methods.decota_final_simplified_v1.tensors import state_hash
 setup();torch.manual_seed(42);np.random.seed(42);p=verify();assert read(D/'SOURCE_BARRIER.json')['count']==109;raw=read(R/'artifacts/decota_paper_experiments_v1/hc2/PLAN.json')['rows'];lock,start,check=runlease('teacher');files=[];calls=0
 try:
  model=ConditionalRefer();mh=state_hash(model.model.state_dict())
  for r in p['records']:
   a=r['alias'];f=D/'semantics'/f'{a}.pt'
   if f.exists():load(f);files.append(read(f.with_suffix('.receipt.json')));continue
   check();priorcalls=sum(read(v).get('new_ReferDINO',0)for v in (D/'cost').glob('*.json'));assert priorcalls+calls<109
   z=load(D/'inputs'/f'{a}.pt');row=raw[r['query_ordinal']];tick=time.perf_counter();frames,ids=decode(row['input']);rgb=frames[z['planned_positions']];decode_s=time.perf_counter()-tick;cap={}
   def hook(module,args,result):cap.update(premax=result.detach().cpu().clone(),mask=args[1]['text_token_mask'].detach().cpu().clone())
   h=model.model.class_embed[-1].register_forward_hook(hook)
   try:new=model.run(rgb,row['input']['caption']);calls+=1
   finally:h.remove()
   tr=new['trace'];assert torch.equal(cap['premax'].max(-1,keepdim=True).values,tr['all_logits']);legacy=False
   if r['reuse_spatial']:
    cached=load(R/r['reuse_spatial']['input'])['teacher_native'];assert torch.equal(new['boxes'],cached['boxes']);assert new['selected_query']==cached['selected_query'];legacy=True
   text=' '.join(row['input']['caption'].lower().split());text=text if text.endswith('.')else text+'.';tk=model.model.tokenizer([text],padding='longest',return_tensors='pt',return_offsets_mapping=True);L=min(tk['input_ids'].shape[-1],model.model.max_text_len);assert torch.equal(cap['mask'],tk['attention_mask'][:,:L].bool().expand(8,-1))
   files.append(save(f,dict(alias=a,premax=cap['premax'],text_mask=cap['mask'],token_ids=tk['input_ids'][0,:L].cpu(),offsets=tk['offset_mapping'][0,:L].tolist(),all_boxes=tr['all_boxes'],all_logits=tr['all_logits'],selected_query=new['selected_query'],selected_boxes=new['boxes'],tracked_lineages=tr['tracked_lineages'],matching_permutations=tr['matching_permutations'],frame_ids8=[ids[i]for i in z['planned_positions']],legacy_comparison_available=legacy,legacy_selected_exact=legacy,token_max_exact=True,cost=dict(new['cost'],decode_seconds=decode_s),GT_read=False)));status('teacher',len(files),new_ReferDINO=calls);print('teacher',a,round(time.perf_counter()-start,1),flush=True);del frames,rgb,new,cap,z;gc.collect();torch.cuda.empty_cache()
  assert state_hash(model.model.state_dict())==mh and all(not x.requires_grad and x.grad is None for x in model.model.parameters());put(D/'SEMANTIC_BARRIER.json',dict(count=109,files=files,weights_unchanged=True,new_ReferDINO=calls,GT_read=False))
 finally:put(D/'cost'/f'teacher_{time.time_ns()}.json',dict(GPU_seconds=time.perf_counter()-start+3,new_ReferDINO=calls,new_ReferDINO_image_frames=8*calls,new_Source=0,new_DINO=0));lock.close()

def spatial():
 import torch,numpy as np
 from methods.stvg_referdino_opd_v1.runtime import setup,load_source
 from methods.stvg_referdino_opd_v1.predictor import ReferDINOOPDPredictor
 from methods.decota_final_simplified_v1.tensors import state_hash
 setup();p=verify();assert read(D/'SEMANTIC_BARRIER.json')['count']==109;lock,start,check=runlease('spatial');model=None;files=[];updates=0;queries=0
 try:
  for r in p['records']:
   a=r['alias'];f=D/'spatial'/f'{a}.pt'
   if f.exists():load(f);files.append(read(f.with_suffix('.receipt.json')));continue
   check();z=load(D/'inputs'/f'{a}.pt');sem=load(D/'semantics'/f'{a}.pt')
   if r['reuse_spatial']:
    v=r['reuse_spatial'];assert sha(R/v['prediction'])==v['prediction_sha256'];s=load(R/v['prediction']);outputs={f'Live{x}':np.asarray(s['outputs'][f'Live_{x}'])for x in p['seeds']};out=dict(outputs=outputs,reused=True,sampling_key=v['sampling_key'],actual_updates=0,new_fit_seconds=0.)
   else:
    if model is None:model=load_source();mh=state_hash(model.state_dict());pred=ReferDINOOPDPredictor(p['config'],source=model,check=check)
    z['teacher_native']=dict(boxes=sem['selected_boxes']);tick=time.perf_counter();s=pred.adapt(z,key=a,arms=('Live',),seeds=p['seeds']);outputs={f'Live{x}':np.asarray(s['outputs'][f'Live_{x}'])for x in p['seeds']};updates+=s['cost']['actual_updates'];queries+=1;out=dict(outputs=outputs,reused=False,sampling_key=a,actual_updates=s['cost']['actual_updates'],new_fit_seconds=time.perf_counter()-tick,fit_cost=s['cost'],fits=s['fits'])
   out.update(alias=a,GT_read=False);files.append(save(f,out));status('spatial',len(files),actual_updates=updates);print('spatial',a,'reuse',out['reused'],round(time.perf_counter()-start,1),flush=True);del s,z,sem,out;gc.collect();torch.cuda.empty_cache()
  if model is not None:assert state_hash(model.state_dict())==mh and all(not x.requires_grad and x.grad is None for x in model.parameters())
  put(D/'SPATIAL_BARRIER.json',dict(count=109,files=files,new_queries=queries,actual_updates=updates,GT_read=False))
 finally:put(D/'cost'/f'spatial_{time.time_ns()}.json',dict(GPU_seconds=time.perf_counter()-start+3,new_spatial_queries=queries,actual_temporary_state_updates=updates,new_ReferDINO=0,new_Source=0,new_DINO=0,network_weight_updates=0));lock.close()

def predict():
 import numpy as np
 from vg_tta.stvg_where_when_event128_v1 import event_words,word_tokens,semantic_curve,associate,candidates
 from vg_tta.stvg_temporal_increment128_v1 import feedback,tilt
 from vg_tta.stvg_posterior_decision_tree_v1 import final_mbr,brute_utility
 p=verify();assert read(D/'SPATIAL_BARRIER.json')['count']==109;raw=read(R/'artifacts/decota_paper_experiments_v1/hc2/PLAN.json')['rows'];files=[]
 for r in p['records']:
  a=r['alias'];z=load(D/'inputs'/f'{a}.pt');sem=load(D/'semantics'/f'{a}.pt');s=load(D/'spatial'/f'{a}.pt');assert sha(R/r['subject_path'])==r['subject_sha256'];tokens=event_words(read(R/r['subject_path'])['parses']['context']['tokens']);groups,reason=word_tokens(raw[r['query_ordinal']]['input']['caption'],tokens,sem['offsets']);assert not groups or all(sem['text_mask'][:,g].all()for g in groups)
  ids=z['frame_ids'];edges,_=candidates(ids,z['planned_positions'],r['frame_count'],z['interval']);trajectories=dict(E=np.asarray(z['E']),**s['outputs']);outputs={};audits={};assocs={};curves={};support=np.triu_indices(len(ids),1)
  for arm,boxes in trajectories.items():
   j,meta=associate(boxes[z['planned_positions']],sem['all_boxes'].double().numpy());assocs[arm]=dict(query=j,**meta);curve=semantic_curve(sem['premax'],groups,j);curves[arm]=None if curve is None else curve.tolist();f,meta=feedback(ids,edges,curve);audits[arm]=meta
   for lam in p['lambdas']:
    post=tilt(z['q0'],f,lam);m=final_mbr(ids,post)
    if lam==0:assert np.array_equal(post,z['q0'])and m['interval']==list(z['interval'])
    picks=np.unique(np.linspace(0,len(support[0])-1,17).astype(int));err=float(abs(brute_utility(ids,post,picks)-m['utility_grid'][support[0][picks],support[1][picks]]).max());assert err<1e-11
    outputs[f'{arm}_L{lam:g}']=dict(interval=m['interval'],posterior=post,utility=m['utility'],changed=m['interval']!=list(z['interval']),independent_MBR_error=err,KL_posterior_to_prior=float(np.sum(post[post>0]*(np.log(post[post>0])-np.log(z['q0'][post>0])))),L1_prior_change=float(abs(post-z['q0']).sum()))
  files.append(save(D/'predictions'/f'{a}.pt',dict(alias=a,query_ordinal=r['query_ordinal'],frame_ids=ids,positions8=z['planned_positions'],original_interval=z['interval'],trajectories=trajectories,outputs=outputs,feedback_audits=audits,association=assocs,curves=curves,edges=edges.tolist(),event_status=reason,q0=z['q0'],GT_read=False)));status('prediction',len(files));print('sealed',a,flush=True)
 put(D/'GLOBAL_PREDICTION_BARRIER.json',dict(count=109,files=files,plan_sha256=sha(D/'PLAN_LOCK.json'),all_before_new_GT=True,selection_unchanged=True));status('sealed_before_GT',109)

if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','source','teacher','spatial','predict']);a=ap.parse_args()
 try:globals()[a.stage]()
 except BaseException as e:
  D.mkdir(exist_ok=True,parents=True);put(D/f'FAILURE_{time.time_ns()}.json',dict(stage=a.stage,error=str(e),type=type(e).__name__,partials_preserved=True));raise
