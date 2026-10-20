"""Select whole-panel configurations, seal selected predictions and score five rows."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,time,json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_single_referdino_parameter_search256_common_v1 import ROOT,D,P,BASE,read,put,sha,load,verify
from scripts.score_stvg_single_referdino_bidirectional256_v1 import difference

def select():
 p=verify();assert read(D/'FOUR_PHASE_COMPLETION.json')['status']=='completed_all_four_phases_and_independent_scoring';out={};predfiles={}
 for ds in ['hc2','vidstg']:
  results={ph:read(P/ds/ph/'RESULTS_PUBLIC.json')for ph in ['coarse','fine']};candidates=[]
  for ph,r in results.items():
   for audit in ['AUDIT_PUBLIC.json','ROOT_AUDIT_PUBLIC.json','READOUT_AUDIT_PUBLIC.json']:assert read(P/ds/ph/audit)['status']=='pass'
   candidates += [(r['means'][a]['v'],ph,a,r['definitions'][a])for a in r['definitions']]
  candidates.sort(key=lambda x:(-x[0],abs(x[3]['alpha']-1),x[3]['steps'],x[3]['lr'],-x[3]['tau'],x[1]))
  value,phase,arm,c=candidates[0];r=results[phase];cfg=dict(p['configs'][ds][p['baseline']],lr=c['lr'],tau=c['tau'],steps=c['steps'],output_alpha=c['alpha']);cfg['selection_record']=str((P/'SELECTION_PUBLIC.json').relative_to(ROOT));cfg['selection']='finite original128 development search; no independent confirmation';Evalue=r['means']['E_Strong']['v']
  out[ds]=dict(config=cfg,phase=phase,arm=arm,v=value,E_v=Evalue,adaptive_winner_exceeds_E=value>Evalue,adaptive_candidates=len(candidates),baseline_arm=p['baseline']+f"_S{p['baseline_steps'][ds]:02}_A100",original_baseline_v=r['means'][p['baseline']+f"_S{p['baseline_steps'][ds]:02}_A100"]['v'],paired_vs_E=r['paired'][arm+'_minus_E_Strong'],paired_vs_original=r['paired'].get(arm+'_minus_'+p['baseline']+f"_S{p['baseline_steps'][ds]:02}_A100"))
  for rec in [v for v in p['records']if v['dataset']==ds]:
   alias=rec['alias'];old=load(BASE/f'predictions/{ds}/{alias}.pt')
   with np.load(D/ds/phase/'predictions'/f'{alias}.npz')as z:b=z[arm].copy();assert np.array_equal(z['E_Strong'],old['E_Ref'])
   f=D/'selected_predictions'/ds/f'{alias}.npz';f.parent.mkdir(parents=True,exist_ok=True);assert not f.exists();np.savez(f,Source=old['Source'],E_Ref=old['E_Ref'],Live=b,frame_ids=np.array(old['frame_ids']),positions8=np.array(old['planned_positions']),Native_interval=np.array(old['intervals']['Native']),Strong_interval=np.array(old['intervals']['Strong']));predfiles[str(f.relative_to(ROOT))]=sha(f)
 put(D/'SELECTED_PREDICTION_BARRIER.json',dict(status='sealed_after_offline_development_selection_before_five_row_scoring',parents=256,files=predfiles,selection_GT_used=True,GT_not_used_inside_any_fit=True,epoch=time.time()))
 selection=dict(status='selected_dataset_specific_development_configurations',datasets=out,seed=41,expert_count=1,per_video_choice=False,independent_confirmation=False,automatic_further_search=False,production_promotion=False)
 put(P/'SELECTION_PUBLIC.json',selection);put(P/'SELECTED_CONFIGS_PUBLIC.json',{ds:q['config']for ds,q in out.items()});put(D/'STATE.json',dict(status='selected_predictions_sealed_pending_five_row_score',datasets={ds:dict(phase=q['phase'],arm=q['arm'])for ds,q in out.items()},automatic_followup=False));print('DATASET_SELECTION',json.dumps({ds:{k:q[k]for k in ['phase','arm','v','E_v','adaptive_winner_exceeds_E']}for ds,q in out.items()}),flush=True)

def score():
 import torch
 torch.set_num_threads(2)
 from scripts.score_stvg_opd_p1_v1 import truths
 from vg_tta.tastvg_oracle_event5_v1 import DenseTube,official,box_iou
 from vg_tta.tastvg_paper48_metrics_v1 import xyxy
 p=verify();selection=read(P/'SELECTION_PUBLIC.json');bar=read(D/'SELECTED_PREDICTION_BARRIER.json');assert len(bar['files'])==256
 for f,h in bar['files'].items():assert sha(ROOT/f)==h
 oldr=read(ROOT/'results/stvg_error_state/2026-10-20/single_referdino_bidirectional256/RESULTS_PUBLIC.json');res={};checks=0;maxerr=0.;beg=time.perf_counter()
 for ds in ['hc2','vidstg']:
  records=[v for v in p['records']if v['dataset']==ds];raw=read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows'];gt,spans,prov=truths(ds,dict(parents=[v['query_ordinal']for v in records]));rows=[];old={v['alias']:v for v in oldr['directions'][ds]['rows']};sel=selection['datasets'][ds];selectedrows={v['alias']:v for v in read(P/ds/sel['phase']/'RESULTS_PUBLIC.json')['rows']}
  for rec in records:
   alias,q=rec['alias'],rec['query_ordinal'];r=raw[q];values={}
   with np.load(D/'selected_predictions'/ds/f'{alias}.npz')as z:
    ids=z['frame_ids'].tolist();pos=z['positions8'].tolist();lookup={f:i for i,f in enumerate(ids)};fixed=[f for f in ids if f in gt[q]];obs=[f for f in fixed if f in {ids[i]for i in pos}];unobs=[f for f in fixed if f not in obs];private=dict(GT_span=spans[q],observed_GT_ids=obs,per_arm={})
    for arm,b,kind in [('Source',z['Source'],'Native'),('Spatial_only',z['Live'],'Native'),('Temporal_only',z['Source'],'Strong'),('Full',z['Live'],'Strong'),('Without_spatial_adaptation',z['E_Ref'],'Strong')]:
     iv=z[kind+'_interval'].tolist();tube=DenseTube(b,r,gt[q],spans[q],clip=ds=='hc2');v=tube.score(iv);off=official(b,r,gt[q],spans[q],iv,ds);e=max(abs(v[k]-off[k])for k in ['v','t','s']);assert e<2e-10;checks+=1;maxerr=max(maxerr,e);pix=xyxy(b,r['input']['width'],r['input']['height']);pix=np.maximum(pix,0)if ds=='hc2'else pix;node={f:float(box_iou(pix[lookup[f]],gt[q][f]))for f in fixed}
     for name,group in [('fixed_actual_sIoU',fixed),('observed_actual_sIoU',obs),('unobserved_actual_sIoU',unobs)]:v[name]=float(np.mean([node[f]for f in group]))if group else None
     v['boundary_MAE']=(abs(iv[0]-spans[q][0])+abs(iv[1]-spans[q][1]))/2;values[arm]=v;private['per_arm'][arm]=dict(interval=iv,actual_IoU=node,dense_GT_ids=tube.fids.tolist(),dense_GT_IoU=tube.iou.tolist())
     if arm in ['Source','Temporal_only','Without_spatial_adaptation']:
      for k in ['v','t','s','fixed_actual_sIoU','unobserved_actual_sIoU']:assert abs(v[k]-old[alias]['values'][arm][k])<2e-10
     if arm=='Full':
      for k in ['v','t','s','fixed_actual_sIoU','unobserved_actual_sIoU']:assert abs(v[k]-selectedrows[alias]['values'][sel['arm']][k])<2e-10
   put(D/f'selected_private_receipts/{ds}/{alias}.json',private);rows.append(dict(alias=alias,values=values,historical=dict(original_Full=old[alias]['values']['Full']),observed_actual_nodes=len(obs),unobserved_actual_nodes=len(unobs)))
  metrics=list(rows[0]['values']['Source']);means={a:{k:float(np.mean([v['values'][a][k]for v in rows if v['values'][a][k]is not None]))for k in metrics}for a in rows[0]['values']};pairs={}
  for a,b in [('Full','Source'),('Full','Without_spatial_adaptation'),('Full','Spatial_only'),('Spatial_only','Source')]:pairs[a+'_minus_'+b]=difference(rows,lambda v:v['values'][a],lambda v:v['values'][b])
  pairs['Full_minus_original_Full']=difference(rows,lambda v:v['values']['Full'],lambda v:v['historical']['original_Full']);res[ds]=dict(parents=128,queries=128,seed=41,means=means,paired=pairs,rows=rows,configuration=sel['config'],independent_confirmation=False,development_selection_bias=True,GT_provenance=prov)
 put(P/'SELECTED_FIVE_ROW_RESULTS_PUBLIC.json',dict(status='completed_selected_configurations_same_exposed_panels',directions=res,expert_count=1,new_independent_on_policy_claim=False,independent_confirmation=False));put(P/'SELECTED_FIVE_ROW_AUDIT_PUBLIC.json',dict(status='pass',official_dense_checks=checks,max_abs=maxerr,all256_retained=True,original_Source_Strong_ERef_unchanged=True,selected_Full_exact_phase_scores=True,seconds=time.perf_counter()-beg));print('SELECTED_FIVE_ROWS_COMPLETE',flush=True)
if __name__=='__main__':
 if sys.argv[1]=='select':select()
 elif sys.argv[1]=='score':score()
