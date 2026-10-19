"""Finite what-role interface repair; seal actual DINO->E outputs before GT."""
import os,sys,time,json,hashlib,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,save,sha
from vg_tta.stvg_what_referring_condition_v1 import rewrite,VERSION
D=ROOT/'artifacts/stvg_what_referring_repair32_v1';P=ROOT/'results/stvg_error_state/2026-10-19/what_referring_repair32'
PRE=ROOT/'artifacts/stvg_dino_endpoint_confirm32_v1';FOCUS=['NV003','NV006','NV009','NV013','NV015']

def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def verify(full=False):
 from main_method.runtime import resource_guard
 l=read(D/'RUNTIME_LOCK.json');assert sha(D/'PLAN_LOCK.json')==l['plan_sha256'];resource_guard()
 for f,h in l['pins'].items():assert sha(ROOT/f)==h,f
 return read(D/'PLAN_LOCK.json')

def prepare():
 assert not D.exists();D.mkdir();P.mkdir();rs=read(PRE/'PLAN_LOCK.json')['records'];conditions={};normal=0
 from scripts.run_decota_paper_main_v1 import read_row
 for r in rs:
  row=read_row(r['dataset'],r['query_ordinal']);par,info=rewrite(row['parses'])
  assert info['changed']==(r['alias'] in FOCUS)
  if not info['changed']:assert par==row['parses'];normal+=1
  conditions[r['alias']]=dict(parses=par,info=info,caption_sha256=hashlib.sha256(row['input']['caption'].encode()).hexdigest(),subject_sha256=row['subject_sha256'])
 put(D/'CONDITIONS_PRIVATE.json',conditions)
 own=['vg_tta/stvg_what_referring_condition_v1.py','scripts/run_stvg_what_referring_repair32_v1.py','protocols/stvg_what_referring_repair32_v1.md','methods/decota_final_simplified_v1/observations.py','methods/decota_final_simplified_v1/config.py','scripts/run_decota_paper_main_v1.py','vg_tta/stvg_dino_supported_correction32_v1.py','vg_tta/stvg_joint_robust_correction32_v1.py','artifacts/stvg_joint_robust_correction32_v1/SOURCE_CALIBRATION.json','methods/CURRENT_METHOD.json','methods/MAIN_METHOD.json','methods/PAPER_METHOD.json',str((D/'CONDITIONS_PRIVATE.json').relative_to(ROOT))]
 for r in rs:
  for sub in ['source','E']:own.append(str((PRE/sub/(r['alias']+'.pt')).relative_to(ROOT)))
  own.append(f'artifacts/decota_paper_experiments_v1/{r["dataset"]}/subjects/{r["query_ordinal"]:05}.json')
 put(D/'PLAN_LOCK.json',dict(status='locked_before_new_DINO_and_GT',records=rs,count=32,focus=FOCUS,condition_version=VERSION,normal_branches_unchanged=normal,DINO_forwards_cap=20,GPU_seconds_cap=900,wall_seconds_cap=3600,created_epoch=time.time(),new_Source_training_OPD=0,no_automatic_followup=True))
 put(D/'RUNTIME_LOCK.json',dict(plan_sha256=sha(D/'PLAN_LOCK.json'),pins={f:sha(ROOT/f)for f in own}))
 put(P/'CONFIG_PUBLIC.json',dict(condition_version=VERSION,count=32,per_domain=16,changed_aliases=FOCUS,target_entity='object',target_span=[4,10],one_prompt_per_query=True,normal_branch_unchanged=normal,four_original_slots=True,new_DINO_forwards_cap=20,role_counts=dict(nsubj=3,obj=1,obl=1),cached_particle_root_repair_count=1,score_threshold=.35,margin=.05,beta=1/16,temporal='fixed existing finalMBR',new_Source_training_OPD=0,development_exposed=True))
 print('LOCKED',32,'focus',5,'normal unchanged',normal,flush=True)

def pools(ex,source):
 import numpy as np
 from vg_tta.stvg_error_state_correction_v1 import encode
 out=[];scores=[]
 for a in ex['anchors']['single4']:
  pr=ex['observations'][('original',a['position'])]['probe'];assert pr['accepted'];out.append(encode(np.asarray(pr['boxes'],float))-encode(source)[a['position']]);scores.append(np.asarray(pr['target_scores'],float))
 return out,scores

def capture():
 import numpy as np,torch
 from scripts.run_decota_paper_main_v1 import gpu,read_row,frames_for
 from methods.decota_final_simplified_v1.observations import SpatialExpert,observations
 from methods.decota_final_simplified_v1.config import EXPERT_SNAPSHOT,EXPERT_SHA256
 from methods.decota_final_simplified_v1.tensors import detached,state_hash
 from vg_tta.stvg_dino_supported_correction32_v1 import estimate
 from vg_tta.stvg_joint_robust_correction32_v1 import apply
 p=verify();assert not (D/'GLOBAL_PREDICTION_BARRIER.json').exists();assert sha(ROOT/EXPERT_SNAPSHOT/'model.safetensors')==EXPERT_SHA256
 lease=gpu();start=time.perf_counter();cache=collections.OrderedDict();files=[];calls=0;forward_seconds=0
 try:
  expert=SpatialExpert(ROOT/EXPERT_SNAPSHOT);expert.model.eval().requires_grad_(False);mh=state_hash(expert.model.state_dict());conds=read(D/'CONDITIONS_PRIVATE.json');cal=read(ROOT/'artifacts/stvg_joint_robust_correction32_v1/SOURCE_CALIBRATION.json')
  for r in p['records']:
   verify();assert time.perf_counter()-start<900;assert torch.cuda.mem_get_info()[0]>4*2**30;a=r['alias'];s=load(PRE/'source'/(a+'.pt'));e=load(PRE/'E'/(a+'.pt'));orig=s['Source'].double().numpy();changed=a in FOCUS;condition=conds[a];ex=s['expert'];est=e['estimate'];out=e['E'];cost=0.;key=None
   if changed:
    row=read_row(r['dataset'],r['query_ordinal']);frames,ids,_=frames_for(r['dataset'],row,cache);prior=torch.load(ROOT/r['prior_input'],map_location='cpu',weights_only=False)
    assert hashlib.sha256(frames.tobytes()).hexdigest()==prior['pixel_sha256'];assert ids==s['frame_ids'];assert condition['caption_sha256']==hashlib.sha256(row['input']['caption'].encode()).hexdigest()
    key=digest(dict(version=VERSION,parses=condition['parses'],subject_sha256=row['subject_sha256'],pixel_sha256=prior['pixel_sha256'],expert_sha256=EXPERT_SHA256,positions=e['planned_positions']))
    tick=time.perf_counter();ex=observations(expert,condition['parses'],frames,ids,prior['indices'],audit=True);cost=time.perf_counter()-tick
    assert ex['positions4']==e['planned_positions']and ex['new_DINO']==4 and ex['actual_observation_positions']==e['planned_positions'];calls+=4
    for (_,pos),v in ex['observations'].items():
     assert v['receipt']['rgb_sha256']==hashlib.sha256(frames[pos].tobytes()).hexdigest();assert v['receipt']['target_span']==[4,10];assert v['receipt']['text'][4:10]=='object';forward_seconds+=v['receipt']['forward_seconds']
    pp,ss=pools(ex,orig);setting=cal['source_domains']['hc2'];est=estimate(pp,ss,setting['scale'],setting['truncation'],1/16);out=apply(orig,est['delta'])
   else:assert condition['info']['changed']==False
   files.append(save(D/'predictions'/(a+'.pt'),dict(alias=a,dataset=r['dataset'],query_ordinal=r['query_ordinal'],frame_ids=e['frame_ids'],planned_positions=e['planned_positions'],interval=e['interval'],outputs=dict(E_MBR=e['E'],Repair_E_MBR=out),Source=orig,expert=detached(ex,'cpu'),estimate=est,changed_condition=changed,condition_info=condition['info'],condition_cache_key=key,GT_read=False,cost=dict(new_DINO=4 if changed else 0,seconds=cost))))
   print(a,'new DINO',4 if changed else 0,'accepted',len(ex['anchors']['single4']),'elapsed',round(time.perf_counter()-start,2),flush=True)
  assert calls==20 and state_hash(expert.model.state_dict())==mh;assert time.perf_counter()-start<900;verify()
  put(D/'GLOBAL_PREDICTION_BARRIER.json',dict(status='complete',count=32,files=files,GT_read=False,new_DINO=calls,expert_sha256=EXPERT_SHA256,model_state_before_after=mh,GPU_allocation_seconds=time.perf_counter()-start,forward_seconds=forward_seconds,peak_allocated_bytes=torch.cuda.max_memory_allocated()))
 finally:lease.close()

def score():
 import torch,numpy as np
 torch.set_num_threads(4);p=verify();bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');assert bar['count']==32 and not torch.cuda.is_initialized();zs=[load(ROOT/f['path'])for f in bar['files']]
 from scripts.score_stvg_opd_p1_v1 import truths
 from scripts.run_stvg_dino_local_residual32_v1 import metrics
 from scripts.score_stvg_dino_mask_align32_v1 import iou_gt
 from scripts.score_stvg_joint_robust_correction32_v1 import paired
 plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in ['hc2','vidstg']};gt={ds:truths(ds,dict(parents=[r['query_ordinal']for r in p['records']if r['dataset']==ds]))for ds in plans};rows=[];frames=[]
 for z in zs:
  ds=z['dataset'];q=z['query_ordinal'];dense,spans,_=gt[ds];row=plans[ds][q];val=metrics(z,row,dense[q],spans[q]);a=z['alias'];old=load(PRE/'source'/(a+'.pt'))['expert'];ex=z['expert']
  rows.append(dict(alias=a,dataset=ds,changed_condition=z['changed_condition'],condition_info=z['condition_info'],new_DINO=z['cost']['new_DINO'],accepted_before=len(old['anchors']['single4']),accepted_after=len(ex['anchors']['single4']),rejection_reasons=dict(collections.Counter(v['probe']['reason']for v in ex['observations'].values())),**val))
  if z['changed_condition']:
   for (_,pos),v in sorted(ex['observations'].items()):
    pr=v['probe'];known=int(z['frame_ids'][pos])in dense[q];bs=np.asarray(pr['boxes'],float);sc=np.asarray(pr['target_scores'],float);order=np.argsort(-sc,kind='stable');ious=[iou_gt(b,dense[q][int(z['frame_ids'][pos])],row,ds)for b in bs]if known else None;base=iou_gt(z['outputs']['E_MBR'][pos],dense[q][int(z['frame_ids'][pos])],row,ds)if known else None
    frames.append(dict(alias=a,position=pos,accepted=pr['accepted'],reason=pr['reason'],margin=pr['margin'],candidate_count=len(bs),scores=sc.tolist(),GT_known=known,E_IoU=base,top_score_IoU=None if not known or not len(bs)else ious[int(order[0])],retained_pool_best_IoU=None if not known or not len(bs)else max(ious),target_span_is_object=v['receipt']['text'][4:10]=='object',new_DINO=1,forward_seconds=v['receipt']['forward_seconds']))
 groups={}
 for name,rr in [('HC16',[r for r in rows if r['dataset']=='hc2']),('Vid16',[r for r in rows if r['dataset']=='vidstg']),('Affected5',[r for r in rows if r['changed_condition']])]:
  groups[name]=dict(parents=len(rr),new_DINO=sum(r['new_DINO']for r in rr),accepted_before=sum(r['accepted_before']for r in rr),accepted_after=sum(r['accepted_after']for r in rr),paired={k:paired([None if r['values']['E_MBR'][k]is None or r['values']['Repair_E_MBR'][k]is None else 100*(r['values']['Repair_E_MBR'][k]-r['values']['E_MBR'][k])for r in rr])for k in ['v','t','s','observed_actual_s','unobserved_actual_s']})
 put(P/'RESULTS_PUBLIC.json',dict(status='completed',count=32,groups=groups,rows=rows,new_GT_after_all32_seal=True,development_only=True,no_OPD_or_training=True));put(P/'FRAME_DIAGNOSTICS_PUBLIC.json',dict(count=len(frames),rows=frames))
 put(P/'COST_BREAKDOWN.json',dict(GPU_allocation_seconds=bar['GPU_allocation_seconds'],DINO_forward_seconds=bar['forward_seconds'],peak_allocated_bytes=bar['peak_allocated_bytes'],total_Source_captures_completed=0,final_worker_DINO_encodings_completed=20,engineering_DINO_encodings_completed=0,new_training_student_updates_OPD=0,old_Vid16_DINO=44,new_total_Vid16_DINO=64,original_DINO_increase_percent=100*20/44));print(json.dumps(groups),flush=True)

if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','capture','score']);a=ap.parse_args()
 try:globals()[a.stage]()
 except BaseException as e:
  D.mkdir(exist_ok=True,parents=True);put(D/('FAILURE_'+str(time.time_ns())+'.json'),dict(stage=a.stage,type=type(e).__name__,error=str(e),partial_preserved=True));raise
