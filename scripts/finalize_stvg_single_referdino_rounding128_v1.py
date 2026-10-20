"""Preserve exact results and evaluate the one accepted rounded HC configuration."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='2'
import sys,json,copy,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_single_referdino_rounding128_v1 import D,P,MD,MP,read,put,sha,verify
from scripts.score_stvg_single_referdino_bidirectional256_v1 import difference

def main():
 import torch
 torch.set_num_threads(2)
 from scripts.score_stvg_opd_p1_v1 import truths
 from vg_tta.tastvg_oracle_event5_v1 import DenseTube,official,box_iou
 from vg_tta.tastvg_paper48_metrics_v1 import xyxy
 p=verify();decision=read(P/'ROUNDING_DECISION_PUBLIC.json');assert decision['adopt_rounded'];cfg=decision['rounded_configuration'];old=read(MP/'SELECTED_FIVE_ROW_RESULTS_PUBLIC.json');pred={}
 # Zero/old fields are copied byte-for-byte; only the sealed rounded Live changes.
 for rec in p['records']:
  ds,a=rec['dataset'],rec['alias'];f=MD/'selected_predictions'/ds/f'{a}.npz';dest=D/'selected_predictions'/ds/f'{a}.npz';dest.parent.mkdir(parents=True,exist_ok=True)
  with np.load(f) as z:out={k:z[k].copy()for k in z.files}
  if ds=='hc2':
   with np.load(D/'hc2/fine/predictions'/f'{a}.npz')as z:out['Live']=z['RND_S40_A100'].copy()
  assert not dest.exists();np.savez(dest,**out);pred[str(dest.relative_to(ROOT))]=sha(dest)
 put(D/'SELECTED_PREDICTION_BARRIER.json',dict(status='sealed_after_single_rounded_trial_selection_before_five_row_evaluation',files=pred,parents=256,new_rounded_HC_predictions=128,unchanged_Vid_predictions=128))
 records=[r for r in p['records']if r['dataset']=='hc2'];raw=read(ROOT/'artifacts/decota_paper_experiments_v1/hc2/PLAN.json')['rows'];gt,spans,prov=truths('hc2',dict(parents=[r['query_ordinal']for r in records]));rows=[];checks=0;maxerr=0.;independent=0;expected={r['alias']:r for r in read(P/'hc2/fine/RESULTS_PUBLIC.json')['rows']}
 for rec in records:
  a,q=rec['alias'],rec['query_ordinal'];row=raw[q];vals={}
  with np.load(D/'selected_predictions/hc2'/f'{a}.npz')as z:
   ids=z['frame_ids'].tolist();lookup={f:i for i,f in enumerate(ids)};fixed=[f for f in ids if f in gt[q]];obs=[f for f in fixed if f in {ids[i]for i in z['positions8']}];unobs=[f for f in fixed if f not in obs]
   for arm,b,kind in [('Source',z['Source'],'Native'),('Spatial_only',z['Live'],'Native'),('Temporal_only',z['Source'],'Strong'),('Full',z['Live'],'Strong'),('Without_spatial_adaptation',z['E_Ref'],'Strong')]:
    iv=z[kind+'_interval'].tolist();tube=DenseTube(b,row,gt[q],spans[q],clip=True);v=tube.score(iv);off=official(b,row,gt[q],spans[q],iv,'hc2');e=max(abs(v[k]-off[k])for k in ['v','t','s']);assert e<2e-10;checks+=1;maxerr=max(maxerr,e);pix=np.maximum(xyxy(b,row['input']['width'],row['input']['height']),0);node={f:float(box_iou(pix[lookup[f]],gt[q][f]))for f in fixed}
    for name,group in [('fixed_actual_sIoU',fixed),('observed_actual_sIoU',obs),('unobserved_actual_sIoU',unobs)]:v[name]=float(np.mean([node[f]for f in group]))if group else None
    v['boundary_MAE']=(abs(iv[0]-spans[q][0])+abs(iv[1]-spans[q][1]))/2;vals[arm]=v
    # Independent physical support aggregation, distinct from DenseTube/official.
    start,end=iv;g,h=spans[q];mask=(tube.fids>=max(start,g))&(tube.fids<min(end,h));den=max(end,h)-min(start,g);overlap=max(0,min(end,h)-max(start,g));union=end-start+h-g-overlap
    for measured,saved in [(float(tube.iou[mask].sum()/max(den,1)),v['v']),(overlap/union,v['t'])]:assert abs(measured-saved)<1e-10;independent+=1
    if arm=='Full':assert abs(v['v']-expected[a]['values']['RND_S40_A100']['v'])<1e-10
  historical=old['directions']['hc2']['rows'][len(rows)]['historical'];rows.append(dict(alias=a,values=vals,historical=historical,observed_actual_nodes=len(obs),unobserved_actual_nodes=len(unobs)))
 metrics=list(rows[0]['values']['Source']);means={a:{k:float(np.mean([r['values'][a][k]for r in rows if r['values'][a][k]is not None]))for k in metrics}for a in rows[0]['values']};paired={}
 for a,b in [('Full','Source'),('Full','Without_spatial_adaptation'),('Full','Spatial_only'),('Spatial_only','Source')]:paired[a+'_minus_'+b]=difference(rows,lambda r:r['values'][a],lambda r:r['values'][b])
 paired['Without_spatial_adaptation_minus_original_ERef_Strong']=difference(rows,lambda r:r['values']['Without_spatial_adaptation'],lambda r:r['historical']['original_ERef_Strong']);paired['Full_minus_original_Full']=difference(rows,lambda r:r['values']['Full'],lambda r:r['historical']['original_Full']);res=copy.deepcopy(old);res['directions']['hc2']=dict(parents=128,queries=128,seed=41,means=means,paired=paired,rows=rows,configuration=cfg,GT_provenance=prov,independent_confirmation=False,development_selection_bias=True);res.update(rounding_revision004=True,Vid_exact_reused=True,status='accepted_rounded_HC_and_unchanged_Vid_five_rows');put(P/'FINAL_FIVE_ROW_RESULTS_PUBLIC.json',res)
 # Primary bootstrap independently recomputed from gathered saved parent deltas.
 ix=np.random.default_rng(20261020).integers(0,128,(10000,128))
 for name,m in paired.items():
  aa,bb=name.split('_minus_');x=[]
  for r in rows:
   x.append(100*(r['values'][aa]['v']-(r['historical']['original_ERef_Strong'if bb=='original_ERef_Strong'else'original_Full']['v']if bb.startswith('original_')else r['values'][bb]['v'])))
  assert np.max(abs(np.quantile(np.asarray(x)[ix].mean(1),[.025,.975])-m['v']['CI95']))<1e-10
 put(P/'FINAL_FIVE_ROW_AUDIT_PUBLIC.json',dict(status='pass',official_dense_checks=checks,independent_physical_support_checks=independent,primary_bootstrap_independent=True,max_abs=maxerr,Vid_exact_reused=True,all256_retained=True))
 configs={ds:dict(g['configuration'])for ds,g in res['directions'].items()};configs['hc2']=dict(cfg,selection_record=str((P/'ROUNDING_DECISION_PUBLIC.json').relative_to(ROOT)));put(P/'SELECTED_CONFIGS_PUBLIC.json',configs);rt=read(ROOT/'methods/stvg_single_referdino_opd_v1/tuned_config_20261020.json');rt.update(revision='dataset_tuned_rounded_20261020',rounding_revision004=True,datasets=configs,rounding_record=str((P/'ROUNDING_DECISION_PUBLIC.json').relative_to(ROOT)));put(ROOT/'methods/stvg_single_referdino_opd_v1/rounded_config_20261020.json',rt)
 print('ROUNDED_COMPLETE_FIVE_ROWS_AND_RUNTIME_CONFIG',means['Full']['v'],flush=True)
if __name__=='__main__':main()
