"""Independent CPU raw-logit, E propagation, sealed-score and parent readback."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_dino_tb_checkpoint32_v1 import D,P,PRE,verify,frame_pools
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha

def main():
 import torch,numpy as np
 torch.set_num_threads(4)
 from methods.decota_final_simplified_v1.observations import probe_from_detection,choose_candidate,STOP_WORDS,target_tokens
 from transformers import AutoProcessor
 from scripts.run_stvg_dino_tb_checkpoint32_v1 import BS
 from vg_tta.stvg_error_state_correction_v1 import encode
 from vg_tta.stvg_dino_supported_correction32_v1 import estimate
 from vg_tta.stvg_joint_robust_correction32_v1 import apply
 from scripts.score_stvg_opd_p1_v1 import truths
 from vg_tta.tastvg_oracle_event5_v1 import DenseTube
 from scripts.score_stvg_dino_mask_align32_v1 import iou_gt
 start=time.perf_counter();p=verify(True);bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');assert bar['count']==32 and not torch.cuda.is_initialized();res=read(P/'RESULTS_PUBLIC.json');fs=read(P/'FRAME_DIAGNOSTICS_PUBLIC.json')['rows'];proc=AutoProcessor.from_pretrained(BS,local_files_only=True)
 plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in ['hc2','vidstg']};gt={ds:truths(ds,dict(parents=[r['query_ordinal']for r in p['records']if r['dataset']==ds]))for ds in plans};cal=read(ROOT/'artifacts/stvg_joint_robust_correction32_v1/SOURCE_CALIBRATION.json')
 count=dict(raw_feedback=0,E_reconstruction=0,physical_inputs=0,full_metric=0,actual_metric=0,frame_quality=0,paired_bootstrap=0);err=dict(raw_scores=0.,E=0.,metric=0.)
 for receipt,r in zip(bar['files'],p['records']):
  assert sha(ROOT/receipt['path'])==receipt['sha256'];z=load(ROOT/receipt['path']);a=r['alias'];ds=r['dataset'];q=r['query_ordinal'];s=load(PRE/'source'/(a+'.pt'));e=load(PRE/'E'/(a+'.pt'));row=plans[ds][q];dense,spans,_=gt[ds];pub=next(x for x in res['rows']if x['alias']==a)
  assert z['interval']==e['interval'] and np.array_equal(z['outputs']['E_MBR'],e['E']) and z['GT_read']==False
  for raw,(key,ob)in zip(z['B_raw'],z['expert']['observations'].items()):
   rc=ob['receipt'];old=s['expert']['observations'][key]['receipt'];assert rc['inputs']==old['inputs']and rc['text']==old['text']and rc['target_span']==old['target_span']and rc['rgb_sha256']==old['rgb_sha256'];count['physical_inputs']+=1
   text=rc['text'];tok=proc.tokenizer(text,return_offsets_mapping=True);span=rc['target_span'];row_subject=read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/subjects/{q:05}.json')['parses']['old']
   ent=target_tokens(text,tok['offset_mapping'],span) if span is not None else [i for i,(lo,hi)in enumerate(tok['offset_mapping'])if hi>lo and lo<text.index(row_subject['entity'])+len(row_subject['entity'])and hi>text.index(row_subject['entity'])];whole=[i for i,(lo,hi)in enumerate(tok['offset_mapping'])if hi>lo and text[lo:hi].strip().isalpha()and text[lo:hi]not in STOP_WORDS]
   prob=raw['logits'].sigmoid();target=prob[:,ent].mean(-1);mixed=torch.minimum(target,prob[:,whole].mean(-1));det=dict(all_boxes=raw['boxes'],all_phrase_scores=mixed,target_all_scores=target)
   
   if rc['context_active']:pr=probe_from_detection(det,key[1],z['frame_ids'][key[1]])
   else:
    dd=choose_candidate(raw['boxes'],mixed);pr=dict(candidate_ids=[0]if dd['box']else [],accepted=dd['accepted'],reason=dd['reason'],boxes=torch.tensor([dd['box']])if dd['box']else torch.empty(0,4),target_scores=[dd['score']]if dd['box']else [])
   saved=ob['probe'];assert pr['candidate_ids']==saved['candidate_ids'] and pr['accepted']==saved['accepted'] and pr['reason']==saved['reason'];assert torch.equal(pr['boxes'],saved['boxes'])
   ec=float(np.max(abs(np.asarray(pr['target_scores'])-np.asarray(saved['target_scores']))));err['raw_scores']=max(err['raw_scores'],ec);assert ec<1e-6;count['raw_feedback']+=1
  for ex,est,arm in [(s['expert'],e['estimate'],'E_MBR'),(z['expert'],z['estimate'],'B_E_MBR')]:
   pool=frame_pools(ex);setting=cal['source_domains']['vidstg'if ds=='hc2'else'hc2'];re=estimate([encode(f['boxes'])-encode(z['Source'])[f['position']]for f in pool],[f['scores']for f in pool],setting['scale'],setting['truncation'],1/16);out=apply(z['Source'],re['delta']);ee=float(np.max(abs(out-z['outputs'][arm])));err['E']=max(err['E'],ee);assert ee==0 and re['selected']==est['selected'];count['E_reconstruction']+=1
   tube=DenseTube(out,row,dense[q],spans[q],clip=ds=='hc2');met=tube.score(z['interval'])
   for k in ['v','t','s']:
    em=abs(met[k]-pub['values'][arm][k]);err['metric']=max(err['metric'],em);assert em<2e-10;count['full_metric']+=1
   for key,observed in [('observed_actual_s',True),('unobserved_actual_s',False)]:
    vals=[iou_gt(out[i],dense[q][int(fid)],row,ds)for i,fid in enumerate(z['frame_ids'])if int(fid)in dense[q]and((i in z['planned_positions'])==observed)];vv=float(np.mean(vals))if vals else None;pv=pub['values'][arm][key];assert(vv is None and pv is None)or abs(vv-pv)<2e-10;count['actual_metric']+=1
  for f in [f for f in fs if f['alias']==a]:
   fid=int(z['frame_ids'][f['position']]);assert f['GT_known']==(fid in dense[q])
   for name,ex,est,arm in [('T',s['expert'],e['estimate'],'E_MBR'),('B',z['expert'],z['estimate'],'B_E_MBR')]:
    ff=f['teachers'][name];ob=ex['observations'].get(('original',f['position']));assert ff['called']==(ob is not None)
    if not f['GT_known']:assert ff['top_IoU']is None and ff['selected_IoU']is None and ff['propagated_IoU']is None;continue
    if ob:
     sc=np.asarray(ob['probe']['target_scores']);b=np.asarray(ob['probe']['boxes']);ti=iou_gt(b[sc.argmax()],dense[q][fid],row,ds)if len(sc)else 0.;assert abs(ti-ff['top_IoU'])<1e-12;count['frame_quality']+=1
 for ds,g in res['groups'].items():
  rr=[r for r in res['rows']if r['dataset']==ds]
  for ep,stat in g['B_minus_T'].items():
   vals=np.array([100*(r['values']['B_E_MBR'][ep]-r['values']['E_MBR'][ep])for r in rr if r['values']['B_E_MBR'][ep]is not None and r['values']['E_MBR'][ep]is not None]);assert np.allclose(vals,stat['per_parent_pp'],rtol=0,atol=1e-12);ix=np.random.default_rng(20261018).integers(len(vals),size=(10000,len(vals)));ci=np.quantile(vals[ix].mean(1),[.025,.975]);assert abs(float(vals.mean())-stat['mean'])<1e-12 and np.allclose(ci,stat['CI95'],rtol=0,atol=1e-12);count['paired_bootstrap']+=1
 assert count['raw_feedback']==108 and count['E_reconstruction']==64 and len(fs)==128
 verify(True);put(P/'ROOT_INDEPENDENT_AUDIT.json',dict(status='pass',checks=count,max_errors=err,CPU_seconds=time.perf_counter()-start,raw_B_logits_and_NMS_replayed=True,all32_original_E_exact=True,GT_read_only_after_prediction_seal=True,no_new_GPU_inference=True));print(count,err,flush=True)

if __name__=='__main__':main()
