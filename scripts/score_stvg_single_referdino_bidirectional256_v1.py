"""Five-row official evaluation after both single-expert prediction barriers."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='2'
import sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_single_referdino_bidirectional256_common_v1 import ROOT,D,P,OLD,ROWS,read,put,sha,load,verify
def paired(x):
 x=np.asarray(x,float);rng=np.random.default_rng(20261020);indices=rng.integers(0,len(x),(10000,len(x)));v=x[indices].mean(1)
 return dict(mean=float(x.mean()),CI95=np.quantile(v,[.025,.975]).tolist(),n=len(x))
def main():
 import torch
 torch.set_num_threads(2)
 from scripts.score_stvg_opd_p1_v1 import truths
 from vg_tta.tastvg_oracle_event5_v1 import DenseTube,official,box_iou
 from vg_tta.tastvg_paper48_metrics_v1 import xyxy
 from methods.stvg_single_referdino_opd_v1.initialization import evidence_hash
 p=verify();bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');assert bar['parents']==256 and bar['all_before_GT']
 assert read(P/'INDEPENDENT_SPATIAL_AUDIT_PUBLIC.json')['status']=='pass'
 for f,h in read(D/'SCORING_IMPLEMENTATION_LOCK.json')['pins'].items():assert sha(ROOT/f)==h
 for ds in ['hc2','vidstg']:
  b=read(D/f'BARRIER_{ds}.json');assert sha(D/f'BARRIER_{ds}.json')==bar['barriers'][ds]
  for f,h in b['files'].items():assert sha(ROOT/f)==h
 put(D/'GT_EXPOSURE.json',dict(epoch=time.time(),all256_sealed=True,barrier_sha256=sha(D/'GLOBAL_PREDICTION_BARRIER.json')))
 original=read(ROOT/'results/stvg_error_state/2026-10-20/spatial_strong_bidirectional256/RESULTS_PUBLIC.json');old30={r['alias']:r for r in read(ROOT/'results/stvg_error_state/2026-10-20/vid_saved_parameters128/RESULTS_PUBLIC.json')['rows']};results={};checks=0;oldchecks=0;maxerr=0.;start=time.perf_counter()
 for ds in ['hc2','vidstg']:
  records=[r for r in p['records']if r['dataset']==ds];raw=read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows'];gt,spans,prov=truths(ds,dict(parents=[r['query_ordinal']for r in records]));prior={r['alias']:r for r in original['directions'][ds]['rows']};rows=[]
  for rec in records:
   a,q=rec['alias'],rec['query_ordinal'];z=load(D/f'predictions/{ds}/{a}.pt');tr=load(D/f'traces/{ds}/{a}.pt');assert evidence_hash(tr['fit']['target'].numpy())==z['shared_ReferDINO_evidence_hash'];r=raw[q];ids=z['frame_ids'];lookup={f:i for i,f in enumerate(ids)};fixed=[f for f in ids if f in gt[q]];observed=[f for f in fixed if f in {ids[i]for i in z['planned_positions']}];unobserved=[f for f in fixed if f not in set(observed)];values={};private=dict(frame_ids=ids,actual_GT_ids=fixed,observed_GT_ids=observed,GT_span=spans[q],per_arm={})
   for arm,boxes,kind in [('Source',z['Source'],'Native'),('Spatial_only',z['Live'],'Native'),('Temporal_only',z['Source'],'Strong'),('Full',z['Live'],'Strong'),('Without_spatial_adaptation',z['E_Ref'],'Strong')]:
    iv=z['intervals'][kind];tube=DenseTube(boxes,r,gt[q],spans[q],clip=ds=='hc2');v=tube.score(iv);off=official(boxes,r,gt[q],spans[q],iv,ds);err=max(abs(v[k]-off[k])for k in ['v','t','s']);assert err<2e-10;maxerr=max(maxerr,err);checks+=1
    pix=xyxy(boxes,r['input']['width'],r['input']['height']);pix=np.maximum(pix,0)if ds=='hc2'else pix;node={f:float(box_iou(pix[lookup[f]],gt[q][f]))for f in fixed}
    for name,group in [('fixed_actual_sIoU',fixed),('observed_actual_sIoU',observed),('unobserved_actual_sIoU',unobserved)]:v[name]=float(np.mean([node[f]for f in group]))if group else None
    v['boundary_MAE']=(abs(iv[0]-spans[q][0])+abs(iv[1]-spans[q][1]))/2;values[arm]=v;private['per_arm'][arm]=dict(interval=iv,actual_IoU=node,dense_GT_ids=tube.fids.tolist(),dense_GT_IoU=tube.iou.tolist())
    if arm in ['Source','Temporal_only']:
     for k in prior[a]['values'][arm]:assert abs(prior[a]['values'][arm][k]-v[k])<2e-10;oldchecks+=1
   assert values['Source']['t']==values['Spatial_only']['t'];assert values['Temporal_only']['t']==values['Full']['t']==values['Without_spatial_adaptation']['t']
   put(D/f'private_metric_receipts/{ds}/{a}.json',private)
   historical=dict(old_two_expert_E_Strong=prior[a]['values']['Without_spatial_adaptation'],old_two_expert_Full40=prior[a]['values']['Full'],old_two_expert_Full_matched_steps=prior[a]['values']['Full']if ds=='hc2'else old30[a]['values']['Step30_A100'])
   rows.append(dict(alias=a,values=values,historical=historical,GT_supported_actual_nodes=len(fixed),observed_actual_nodes=len(observed),unobserved_actual_nodes=len(unobserved),initialization_support=z['initialization']['support_frames'],expert_count=1))
  keys=list(rows[0]['values']['Source']);means={arm:{k:float(np.mean([r['values'][arm][k]for r in rows if r['values'][arm][k]is not None]))for k in keys}for arm in ROWS};pairs=[('Spatial_only','Source'),('Temporal_only','Source'),('Full','Source'),('Full','Without_spatial_adaptation'),('Full','Spatial_only')];comparisons={}
  for a,b in pairs:
   comparisons[a+'_minus_'+b]=difference(rows,lambda r:r['values'][a],lambda r:r['values'][b])
  for name,a,b in [('New_ERef_minus_old_E','Without_spatial_adaptation','old_two_expert_E_Strong'),('New_Full_minus_old_Full40','Full','old_two_expert_Full40'),('New_Full_minus_old_Full_matched_steps','Full','old_two_expert_Full_matched_steps')]:comparisons[name]=difference(rows,lambda r:r['values'][a],lambda r:r['historical'][b])
  results[ds]=dict(direction='VidSTG_to_HC2'if ds=='hc2'else'HC2_to_VidSTG',parents=128,queries=128,seed=41,means=means,paired=comparisons,rows=rows,GT_provenance=prov,exposure='original exposed panel; Vid stopping budget selected on old-E development scores; not independent confirmation')
 out=dict(status='completed_single_expert_five_row_evaluation',directions=results,expert_count=1,same_eight_ReferDINO_evidence_for_E_and_OPD=True,configs=p['configs'],new_independent_GroundingDINO_calls=0,independent_confirmation=False,costs=dict(CPU_scoring_seconds=time.perf_counter()-start))
 put(P/'RESULTS_PUBLIC.json',out);put(P/'AUDIT_PUBLIC.json',dict(status='pass',official_dense_checks=checks,max_abs=maxerr,unchanged_Source_and_Temporal_identity_checks=oldchecks,all256_predictions_sealed_before_GT=True,same_evidence_hash_checked=True));print('SINGLE_REF_SCORED',flush=True)
def difference(rows,fn,base):
 out={k:paired([100*(fn(r)[k]-base(r)[k])for r in rows])for k in ['v','t','s','fixed_actual_sIoU','unobserved_actual_sIoU']};x=np.array([100*(fn(r)['v']-base(r)['v'])for r in rows]);out['tail']=dict(positive=int((x>1e-10).sum()),negative=int((x< -1e-10).sum()),unchanged=int((abs(x)<=1e-10).sum()),below_minus5=int((x< -5).sum()),positive_contribution_pp=float(np.maximum(x,0).mean()),negative_contribution_pp=float(np.minimum(x,0).mean()),worst=[dict(alias=rows[i]['alias'],delta_pp=float(x[i]))for i in np.argsort(x)[:8]]);return out
if __name__=='__main__':main()
