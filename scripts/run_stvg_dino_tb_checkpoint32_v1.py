"""Official DINO-T/B checkpoint-only full-image comparison; no student adaptation."""
import collections,gc,hashlib,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,save,sha
D=ROOT/'artifacts/stvg_dino_tb_checkpoint32_v1';P=ROOT/'results/stvg_error_state/2026-10-19/dino_tb_checkpoint32'
PRE=ROOT/'artifacts/stvg_dino_endpoint_confirm32_v1';BS=ROOT/'checkpoints/groundingdino-base-hf'
VER=None

def verify(full=False):
 global VER
 from main_method.runtime import PinVerifier,resource_guard
 l=read(D/'RUNTIME_LOCK.json');assert sha(D/'PLAN_LOCK.json')==l['plan_sha256']
 if VER is None:VER=PinVerifier(l['pins'])
 VER.verify(full=full);resource_guard();return read(D/'PLAN_LOCK.json')

def frame_pools(ex):
 import numpy as np
 fs=[]
 for a in ex['anchors']['single4']:
  pos=int(a['position']);pr=ex['observations'][('original',pos)]['probe'];assert pr['accepted']
  b=np.asarray(pr['boxes'],float).reshape(-1,4);sc=np.asarray(pr['target_scores'],float)
  assert 0<len(b)<=3 and np.array_equal(b[sc.argmax()],np.asarray(a['box']))
  fs.append(dict(position=pos,boxes=b,scores=sc,candidate_ids=pr['candidate_ids']))
 return fs

def reconstruct(ex,src,setting):
 from vg_tta.stvg_error_state_correction_v1 import encode
 from vg_tta.stvg_dino_supported_correction32_v1 import estimate
 from vg_tta.stvg_joint_robust_correction32_v1 import apply
 fs=frame_pools(ex);es=estimate([encode(f['boxes'])-encode(src)[f['position']]for f in fs],[f['scores']for f in fs],setting['scale'],setting['truncation'],1/16)
 return apply(src,es['delta']),es

def prepare():
 import numpy as np
 from transformers import AutoProcessor
 from methods.decota_final_simplified_v1.config import EXPERT_SNAPSHOT,EXPERT_SHA256,MethodConfig
 assert not D.exists();D.mkdir();P.mkdir(parents=True)
 b=read(BS/'DOWNLOAD_RECEIPT.json');assert b['revision']=='12bdfa3120f3e7ec7b434d90674b3396eccf88eb'
 for f in b['files']:assert sha(BS/f['file'])==f['sha256']
 tproc=AutoProcessor.from_pretrained(ROOT/EXPERT_SNAPSHOT,local_files_only=True);bproc=AutoProcessor.from_pretrained(BS,local_files_only=True)
 assert tproc.image_processor.to_dict()==bproc.image_processor.to_dict(),'Processor must be identical'
 rs=read(PRE/'PLAN_LOCK.json')['records'];cal=read(ROOT/'artifacts/stvg_joint_robust_correction32_v1/SOURCE_CALIBRATION.json');checks=[];expected=0
 from scripts.run_decota_paper_main_v1 import read_row
 for r in rs:
  a=r['alias'];s=load(PRE/'source'/(a+'.pt'));e=load(PRE/'E'/(a+'.pt'));row=read_row(r['dataset'],r['query_ordinal']);ctx=row['parses']['context'];old=row['parses']['old']
  text=ctx['context']if ctx['eligible']else old['phrase'].lower().strip()+'.'
  assert tproc.tokenizer(text,return_offsets_mapping=True)==bproc.tokenizer(text,return_offsets_mapping=True)
  out,es=reconstruct(s['expert'],s['Source'].double().numpy(),cal['source_domains']['vidstg'if r['dataset']=='hc2'else'hc2'])
  assert np.array_equal(out,e['E']) and es['selected']==e['estimate']['selected'];assert s['expert']['positions4']==e['planned_positions']
  expected+=len(s['expert']['observations']);checks.append(dict(alias=a,positions=e['planned_positions'],cached_calls=len(s['expert']['observations']),old_E_exact=True))
 assert expected==108
 cfg=dict(arms=['E_MBR','B_E_MBR'],parents=32,per_domain=16,full_RGB=True,four_original_positions=True,teacher_T_sha256=EXPERT_SHA256,teacher_B_sha256=next(f['sha256']for f in b['files']if f['file']=='model.safetensors'),B_repository=b['repository'],B_revision=b['revision'],processor_tokenization_exact=True,threshold=.35,margin=.05,nms=.5,max_pool=3,beta=1/16,source_calibration_unchanged=True,time='original finalMBR frozen',GT_after_all32_seal=True,development_exposed=True,no_target_tuning=True,OPD_not_run=True,new_models_trained=0,B_calls=108,T_timing_replay_calls=4,no_what_repair=True,quality_thresholds_offline=[.3,.5],cost_T_comparison_scope='NH001 same four RGB forwards only; other T timing is historical',GPU_seconds_cap=900,no_automatic_followup=True)
 put(P/'CONFIG_PUBLIC.json',cfg);put(P/'MODEL_PROVENANCE.json',dict(B_official_HF=b,existing_official_original_B=read(ROOT/'checkpoints/groundingdino-b/DOWNLOAD_RECEIPT.json'),T_snapshot=EXPERT_SNAPSHOT,T_sha256=EXPERT_SHA256,TA={ds:dict(checkpoint=c.checkpoint,sha256=c.checkpoint_sha256)for ds,c in [('hc2',MethodConfig.for_direction('vid_to_hc1')),('vidstg',MethodConfig.for_direction('hc2_to_vid'))]},official_sources=['https://github.com/IDEA-Research/GroundingDINO','https://huggingface.co/IDEA-Research/grounding-dino-base'],HF_reason='Use official Transformers B conversion to preserve identical implementation/processor to existing T; original .pth already local but not mixed into HF inference.'))
 put(D/'PLAN_LOCK.json',dict(status='locked_before_inference_and_new_GT',records=rs,checks=checks,configuration=cfg,created_epoch=time.time(),deadline_epoch=time.time()+7200))
 own=['scripts/audit_stvg_dino_tb_checkpoint32_v1.py','scripts/report_stvg_dino_tb_checkpoint32_v1.py','scripts/publish_stvg_dino_tb_checkpoint32_v1.py','scripts/run_stvg_dino_tb_checkpoint32_v1.py','protocols/stvg_dino_tb_checkpoint32_v1.md','methods/decota_final_simplified_v1/observations.py','methods/decota_final_simplified_v1/config.py','methods/decota_final_simplified_v1/tensors.py','scripts/run_decota_paper_main_v1.py','vg_tta/stvg_dino_supported_correction32_v1.py','vg_tta/stvg_joint_robust_correction32_v1.py','vg_tta/stvg_error_state_correction_v1.py','scripts/run_stvg_dino_local_residual32_v1.py','scripts/score_stvg_opd_p1_v1.py','scripts/score_stvg_dino_mask_align32_v1.py','methods/CURRENT_METHOD.json','methods/MAIN_METHOD.json','methods/PAPER_METHOD.json','artifacts/stvg_joint_robust_correction32_v1/SOURCE_CALIBRATION.json']
 own += [str((BS/f['file']).relative_to(ROOT))for f in b['files']]+[str((ROOT/EXPERT_SNAPSHOT/'model.safetensors').relative_to(ROOT))]
 for r in rs:
  own += [str((PRE/sub/(r['alias']+'.pt')).relative_to(ROOT))for sub in ['source','E']]+[f'artifacts/decota_paper_experiments_v1/{r["dataset"]}/subjects/{r["query_ordinal"]:05}.json']
 put(D/'RUNTIME_LOCK.json',dict(plan_sha256=sha(D/'PLAN_LOCK.json'),pins={f:sha(ROOT/f)for f in own}))
 put(D/'STATUS.json',dict(status='prepared',done=0,GT_read=False),True);print('LOCKED 32 parents/108 B forwards/4 T timing forwards, old E32 exact',flush=True)

def capture():
 import torch,numpy as np
 from scripts.run_decota_paper_main_v1 import gpu,read_row,frames_for
 from methods.decota_final_simplified_v1.observations import SpatialExpert,observations
 from methods.decota_final_simplified_v1.config import EXPERT_SNAPSHOT
 from methods.decota_final_simplified_v1.tensors import detached,state_hash
 p=verify(True);assert not (D/'GLOBAL_PREDICTION_BARRIER.json').exists();lease=gpu();begin=time.perf_counter();cache=collections.OrderedDict();files=[];calls=0
 def check():
  verify();assert time.perf_counter()-begin+3<900 and time.time()<p['deadline_epoch'];assert torch.cuda.mem_get_info()[0]>4*2**30
 try:
  check();torch.cuda.reset_peak_memory_stats();tp=SpatialExpert(ROOT/EXPERT_SNAPSHOT);tmh=state_hash(tp.model.state_dict());r=p['records'][0];assert r['alias']=='NH001'
  row=read_row(r['dataset'],r['query_ordinal']);frames,ids,_=frames_for(r['dataset'],row,cache);s=load(PRE/'source'/(r['alias']+'.pt'));prior=torch.load(ROOT/r['prior_input'],map_location='cpu',weights_only=False)
  tt=time.perf_counter();tex=observations(tp,row['parses'],frames,ids,prior['indices'],audit=True);twall=time.perf_counter()-tt;assert tex['positions4']==s['expert']['positions4'];maxbox=maxscore=0.
  for key,v in tex['observations'].items():
   old=s['expert']['observations'][key];assert v['receipt']['inputs']==old['receipt']['inputs'];assert v['probe']['candidate_ids']==old['probe']['candidate_ids']
   maxbox=max(maxbox,float(np.max(abs(np.asarray(v['probe']['boxes'])-np.asarray(old['probe']['boxes'])))));maxscore=max(maxscore,float(np.max(abs(np.asarray(v['probe']['target_scores'])-np.asarray(old['probe']['target_scores'])))))
   assert v['probe']['accepted']==old['probe']['accepted']
  assert maxbox<1e-6 and maxscore<1e-6 and state_hash(tp.model.state_dict())==tmh
  put(D/'T_TIMING_REPLAY.json',dict(alias='NH001',calls=4,forward_seconds=sum(v['receipt']['forward_seconds']for v in tex['observations'].values()),observation_seconds=twall,peak_allocated_bytes=torch.cuda.max_memory_allocated(),parameter_count=sum(v.numel()for v in tp.model.parameters()),max_box_error_vs_cached=maxbox,max_score_error_vs_cached=maxscore,inputs_and_candidate_ids_exact=True,weights_unchanged=True))
  del tp,tex;cache.clear();del frames;gc.collect();torch.cuda.empty_cache();check();torch.cuda.reset_peak_memory_stats();expert=SpatialExpert(BS);mh=state_hash(expert.model.state_dict());cal=read(ROOT/'artifacts/stvg_joint_robust_correction32_v1/SOURCE_CALIBRATION.json')
  for r in p['records']:
   check();tick=time.perf_counter();a=r['alias'];s=load(PRE/'source'/(a+'.pt'));e=load(PRE/'E'/(a+'.pt'));row=read_row(r['dataset'],r['query_ordinal']);frames,ids,_=frames_for(r['dataset'],row,cache);assert ids==s['frame_ids'];prior=torch.load(ROOT/r['prior_input'],map_location='cpu',weights_only=False)
   assert hashlib.sha256(frames.tobytes()).hexdigest()==prior['pixel_sha256'];raw=[]
   def hook(m,args,out):raw.append(dict(boxes=out.pred_boxes[0].detach().cpu(),logits=out.logits[0].detach().float().cpu()))
   h=expert.model.register_forward_hook(hook);ot=time.perf_counter()
   try:ex=observations(expert,row['parses'],frames,ids,prior['indices'],audit=True)
   finally:h.remove()
   obsseconds=time.perf_counter()-ot;assert ex['positions4']==e['planned_positions'];assert set(ex['observations'])==set(s['expert']['observations']);assert len(raw)==ex['new_DINO'];calls+=ex['new_DINO']
   for key,v in ex['observations'].items():
    old=s['expert']['observations'][key]['receipt'];assert v['receipt']['text']==old['text'] and v['receipt']['target_span']==old['target_span'] and v['receipt']['rgb_sha256']==old['rgb_sha256'];assert v['receipt']['inputs']==old['inputs']
   out,est=reconstruct(ex,s['Source'].double().numpy(),cal['source_domains']['vidstg'if r['dataset']=='hc2'else'hc2']);assert np.isfinite(out).all() and(out[:,2:]>0).all()
   files.append(save(D/'predictions'/(a+'.pt'),dict(alias=a,dataset=r['dataset'],query_ordinal=r['query_ordinal'],frame_ids=ids,planned_positions=e['planned_positions'],interval=e['interval'],outputs=dict(E_MBR=e['E'],B_E_MBR=out),Source=s['Source'].double().numpy(),expert=detached(ex,'cpu'),B_raw=raw,estimate=est,GT_read=False,cost=dict(new_DINO=ex['new_DINO'],forward_seconds=sum(v['receipt']['forward_seconds']for v in ex['observations'].values()),observation_seconds=obsseconds,query_seconds=time.perf_counter()-tick))))
   put(D/'STATUS.json',dict(status='running',done=len(files),total=32,last=a,calls=calls,GT_read=False,pid=os.getpid()),True);print(a,'B calls',ex['new_DINO'],'accepted',len(ex['anchors']['single4']),'elapsed',round(time.perf_counter()-begin,2),flush=True)
   cache.clear();del frames,raw,ex;gc.collect()
  check();assert calls==108 and state_hash(expert.model.state_dict())==mh;verify(True)
  put(D/'GLOBAL_PREDICTION_BARRIER.json',dict(status='complete',count=32,files=files,GT_read=False,new_B_calls=calls,new_T_timing_calls=4,GPU_allocation_seconds=time.perf_counter()-begin+3,B_peak_allocated_bytes=torch.cuda.max_memory_allocated(),B_state_sha256=mh,B_parameter_count=sum(v.numel()for v in expert.model.parameters()),B_peak_reserved_bytes=torch.cuda.max_memory_reserved(),hardware=torch.cuda.get_device_name(),torch_version=torch.__version__,weights_unchanged=True))
  put(D/'STATUS.json',dict(status='sealed_before_scoring',done=32,total=32,GT_read=False),True)
 finally:lease.close()

def score():
 import numpy as np,torch
 torch.set_num_threads(4);p=verify(True);bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');assert bar['count']==32 and not torch.cuda.is_initialized();zs=[load(ROOT/f['path'])for f in bar['files']]
 from scripts.score_stvg_opd_p1_v1 import truths
 from scripts.run_stvg_dino_local_residual32_v1 import metrics
 from scripts.score_stvg_dino_mask_align32_v1 import iou_gt
 from scripts.score_stvg_joint_robust_correction32_v1 import paired
 plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in ['hc2','vidstg']};gt={ds:truths(ds,dict(parents=[r['query_ordinal']for r in p['records']if r['dataset']==ds]))for ds in plans};rows=[];fr=[]
 for z in zs:
  ds=z['dataset'];q=z['query_ordinal'];dense,spans,_=gt[ds];row=plans[ds][q];values=metrics(z,row,dense[q],spans[q]);s=load(PRE/'source'/(z['alias']+'.pt'));e=load(PRE/'E'/(z['alias']+'.pt'));teachers=dict(T=(s['expert'],e['estimate'],'E_MBR'),B=(z['expert'],z['estimate'],'B_E_MBR'))
  for pos in z['planned_positions']:
   fid=int(z['frame_ids'][pos]);known=fid in dense[q];rec=dict(alias=z['alias'],dataset=ds,position=pos,GT_known=known,teachers={})
   for name,(ex,est,arm)in teachers.items():
    ob=ex['observations'].get(('original',pos));fs=frame_pools(ex);selected={f['position']:k for f,k in zip(fs,est['selected'])};top=sel=None
    if ob:
     pr=ob['probe'];boxes=np.asarray(pr['boxes'],float).reshape(-1,4);sc=np.asarray(pr['target_scores'],float);idx=int(sc.argmax())if len(sc)else None
     top=(iou_gt(boxes[idx],dense[q][fid],row,ds)if idx is not None else 0.)if known else None
     k=selected.get(pos,-1);sel=iou_gt(boxes[k],dense[q][fid],row,ds)if known and k>=0 else None
     tr=dict(called=True,accepted=pr['accepted'],reason=pr['reason'],margin=pr['margin'],pool_size=len(sc),top_score=None if idx is None else float(sc[idx]),top_IoU=top,selected_index=k,selected_IoU=sel,forward_seconds=ob['receipt']['forward_seconds'],pool_oracle_IoU=max(iou_gt(b,dense[q][fid],row,ds)for b in boxes)if known and len(boxes)else None)
    else:tr=dict(called=False,accepted=False,reason='original_unresolved_condition',pool_size=0,top_IoU=None,selected_index=-1,selected_IoU=None,forward_seconds=0.)
    tr['propagated_IoU']=iou_gt(z['outputs'][arm][pos],dense[q][fid],row,ds)if known else None
    tr['wrong_accepted_IoU_lt_03']=bool(tr['accepted']and top is not None and top<.3)if known else None
    tr['wrong_accepted_IoU_lt_05']=bool(tr['accepted']and top is not None and top<.5)if known else None;rec['teachers'][name]=tr
   fr.append(rec)
  rows.append(dict(alias=z['alias'],dataset=ds,accepted_T=len(s['expert']['anchors']['single4']),accepted_B=len(z['expert']['anchors']['single4']),E_T_selected=sum(k>=0 for k in e['estimate']['selected']),E_B_selected=sum(k>=0 for k in z['estimate']['selected']),B_cost=z['cost'],exact_noop_vs_T=bool(np.array_equal(z['outputs']['B_E_MBR'],z['outputs']['E_MBR'])),**values))
 groups={}
 for ds in plans:
  rr=[r for r in rows if r['dataset']==ds];ff=[f for f in fr if f['dataset']==ds];topdiff=[];stage={}
  for r in rr:
   kk=[f for f in ff if f['alias']==r['alias']and f['GT_known']and f['teachers']['T']['called']and f['teachers']['B']['called']];topdiff.append(float(np.mean([f['teachers']['B']['top_IoU']-f['teachers']['T']['top_IoU']for f in kk]))*100 if kk else None)
  for name in ['T','B']:
   ss=[f['teachers'][name]for f in ff];k=[f for f in ff if f['GT_known']and f['teachers'][name]['called']];selected=[f for f in k if f['teachers'][name]['selected_IoU']is not None]
   def query_stage(a,b):
    vv=[]
    for r in rr:
     v=[f['teachers'][name][a]-f['teachers'][name][b]for f in selected if f['alias']==r['alias']];vv.append(float(np.mean(v))*100 if v else None)
    return paired(vv)
   stage[name]=dict(called=sum(t['called']for t in ss),no_call=sum(not t['called']for t in ss),accepted=sum(t['accepted']for t in ss),parents_with_reference=sum(r['accepted_'+name]>0 for r in rr),reasons=dict(collections.Counter(t['reason']for t in ss)),known_called=len(k),known_selected=len(selected),wrong_accepted_lt03=sum(f['teachers'][name]['wrong_accepted_IoU_lt_03']for f in k),wrong_accepted_lt05=sum(f['teachers'][name]['wrong_accepted_IoU_lt_05']for f in k),selected_minus_top=query_stage('selected_IoU','top_IoU'),propagated_minus_selected=query_stage('propagated_IoU','selected_IoU'))
  groups[ds]=dict(parents=len(rr),means={arm:{ep:float(np.mean([r['values'][arm][ep]for r in rr if r['values'][arm][ep]is not None]))for ep in ['v','t','s','observed_actual_s','unobserved_actual_s']}for arm in ['E_MBR','B_E_MBR']},B_minus_T={ep:paired([None if r['values']['E_MBR'][ep]is None or r['values']['B_E_MBR'][ep]is None else 100*(r['values']['B_E_MBR'][ep]-r['values']['E_MBR'][ep])for r in rr])for ep in ['v','t','s','observed_actual_s','unobserved_actual_s']},actual_top_B_minus_T=paired(topdiff),stages=stage,exact_noops=sum(r['exact_noop_vs_T']for r in rr))
 put(P/'RESULTS_PUBLIC.json',dict(status='completed',groups=groups,rows=rows,all32_sealed_before_new_GT=True));put(P/'FRAME_DIAGNOSTICS_PUBLIC.json',dict(rows=fr,planned_slots=128,unknown_not_zero_filled=True))
 costs=dict(GPU_allocation_seconds=bar['GPU_allocation_seconds'],B_forwards=108,T_timing_replay_forwards=4,new_Source_forwards=0,training_OPD_updates=0,B_peak_allocated_bytes=bar['B_peak_allocated_bytes'],B_parameter_count=bar['B_parameter_count'],B_peak_reserved_bytes=bar['B_peak_reserved_bytes'],hardware=bar['hardware'],torch_version=bar['torch_version'],T_timing_replay=read(D/'T_TIMING_REPLAY.json'),B_forward_seconds=sum(r['B_cost']['forward_seconds']for r in rows),B_observation_seconds=sum(r['B_cost']['observation_seconds']for r in rows),B_query_seconds=sum(r['B_cost']['query_seconds']for r in rows),B_NH001_forward_seconds=rows[0]['B_cost']['forward_seconds'],old_T_forward_seconds=sum(f['teachers']['T']['forward_seconds']for f in fr),historical_T_not_matched_timing=True,download_cost=read(BS/'DOWNLOAD_RECEIPT.json')['seconds'])
 put(P/'COST_BREAKDOWN.json',costs);print(json.dumps(groups),flush=True)

if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','capture','score']);a=ap.parse_args()
 try:globals()[a.stage]()
 except BaseException as e:
  D.mkdir(parents=True,exist_ok=True);put(D/('FAILURE_'+str(time.time_ns())+'.json'),dict(stage=a.stage,type=type(e).__name__,error=str(e)));raise
