"""CPU post-global-seal scores for one frozen temporal-memory mechanism."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,time,numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
from scripts.run_stvg_temporal_object_memory128_v1 import D,P,DEV
from scripts.score_stvg_where_when_event128_v1 import avg,paired

def main():
 import torch
 torch.set_num_threads(4);p=read(D/'PLAN_LOCK.json');bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');assert bar['count']==128 and bar['all_before_new_GT']and bar['plan_sha256']==sha(D/'PLAN_LOCK.json')
 for f,h in p['pins'].items():assert sha(R/f)==h
 for f in bar['files']:assert sha(R/f['path'])==f['sha256']
 put(D/'GT_EXPOSURE.json',dict(epoch=time.time(),global_barrier_sha256=sha(D/'GLOBAL_PREDICTION_BARRIER.json'),all128_before_GT=True))
 from scripts.score_stvg_opd_p1_v1 import truths
 from vg_tta.tastvg_oracle_event5_v1 import DenseTube,official
 gt,spans,prov=truths('hc2',dict(parents=[r['query_ordinal']for r in p['records']]))
 raw=read(R/'artifacts/decota_paper_experiments_v1/hc2/PLAN.json')['rows'];old={r['alias']:r for r in read(R/'results/stvg_error_state/2026-10-19/temporal_increment128/RESULTS_PUBLIC.json')['rows']};rows=[];checks=0
 for r in p['records']:
  a=r['alias'];z=load(D/'predictions'/f'{a}.pt');row=raw[r['query_ordinal']];g=gt[r['query_ordinal']];span=spans[r['query_ordinal']];values={}
  for space,boxes in z['trajectories'].items():
   tube=DenseTube(boxes,row,g,span,clip=True)
   for arm,o in z['outputs'].items():
    iv=o['interval'];v=tube.score(iv);off=official(boxes,row,g,span,iv,'hc2');assert max(abs(v[k]-off[k])for k in ['v','t','s'])<2e-10;checks+=1
    if arm=='Native':assert max(abs(v[k]-old[a]['values'][f'{space}_L0'][k])for k in ['v','t','s'])<2e-10
    assert abs(v['s']-tube.score(z['original_interval'])['s'])<1e-12
    values[f'{space}_{arm}']=dict(**v,interval=iv,boundary_MAE=(abs(iv[0]-span[0])+abs(iv[1]-span[1]))/2)
  for arm in ['Native','E_ROI','LiveOwnROI','ReferROI','ShiftedReferROI']:
   vs=[values[f'Live{s}_'+(f'Live{s}_ROI'if arm=='LiveOwnROI'else arm)]for s in p['seeds']];values[f'LiveMean_{arm}']={k:avg([v[k]for v in vs])for k in ['v','t','s','boundary_MAE']}
  changed={k:dict(changed=o['interval']!=z['original_interval'],max_logit_change=o['max_logit_change'])for k,o in z['outputs'].items()}
  support={k:dict(empty_frames=sum(sum(v['empty'])for v in vv),padding_mask_disagreement=sum(v['padding_mask_disagreement']for v in vv),mean_min_prior=avg([float(v['weights'].min())for v in vv]),mean_max_prior=avg([float(v['weights'].max())for v in vv]))for k,vv in z['prior_meta'].items()}
  rows.append(dict(alias=a,values=values,temporal_changes=changed,prior_support=support));print('scored',a,flush=True)
 arms=['Native','E_ROI','LiveOwnROI','ReferROI','ShiftedReferROI'];means={k:{m:avg([r['values'][k][m]for r in rows])for m in ['v','t','s','boundary_MAE']}for k in [f'LiveMean_{a}'for a in arms]+['E_Native','E_ReferROI']};comparisons={}
 contrasts=[(f'LiveMean_{a}','LiveMean_Native')for a in arms[1:]]+[('LiveMean_ReferROI','LiveMean_ShiftedReferROI'),('LiveMean_ReferROI','LiveMean_E_ROI'),('LiveMean_LiveOwnROI','LiveMean_E_ROI'),('E_ReferROI','E_Native')]
 for a,b in contrasts:
  d={k:paired([(100 if k in ['v','t','s']else 1)*(r['values'][a][k]-r['values'][b][k])for r in rows])for k in ['v','t','s','boundary_MAE']};dv=np.array([100*(r['values'][a]['v']-r['values'][b]['v'])for r in rows]);d['tails']=dict(positive=int((dv>1e-10).sum()),negative=int((dv< -1e-10).sum()),flat=int((abs(dv)<=1e-10).sum()),below_minus3=int((dv< -3).sum()),below_minus5=int((dv< -5).sum()),positive_contribution_pp=float(np.maximum(dv,0).mean()),negative_contribution_pp=float(np.minimum(dv,0).mean()),worst=sorted([dict(alias=r['alias'],delta_pp=float(v))for r,v in zip(rows,dv)],key=lambda x:x['delta_pp'])[:8]);comparisons[a+'_minus_'+b]=d
 result=dict(status='completed_development_capability_probe',parents=128,seeds=p['seeds'],means=means,paired=comparisons,rows=rows,cost=read(D/'COST.json'),primary='LiveMean_ReferROI minus LiveMean_Native',configuration_selection=False,new_temporal_OPD=False,spatial_frozen=True,GT_offline_only=True,independent_confirmation=False)
 put(P/'RESULTS_PUBLIC.json',result);put(P/'INDEPENDENT_AUDIT.json',dict(status='pass',official_DenseTube_checks=checks,all_original_Native_metrics_exact=True,spatial_s_identical=True,all128_sealed_before_GT=True,GT_provenance=prov));put(D/'STATUS.json',dict(stage='scored',done=128,total=128),True);print(comparisons['LiveMean_ReferROI_minus_LiveMean_Native'],flush=True)
if __name__=='__main__':main()
