"""CPU post-global-seal scores for one frozen temporal-memory mechanism."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,time,numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
from scripts.run_stvg_temporal_view128_v1 import D,P,DEV
from scripts.score_stvg_where_when_event128_v1 import avg,paired

def main():
 import torch
 torch.set_num_threads(4);p=read(D/'PLAN_LOCK.json');bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');assert bar['count']==128 and bar['all_before_new_GT']and bar['plan_sha256']==sha(D/'PLAN_LOCK.json');cb=read(D/'CONTROL_BARRIER.json');assert cb['count']==128 and cb['all_before_new_GT'];assert cb['main_barrier_sha256']==sha(D/'GLOBAL_PREDICTION_BARRIER.json')
 for f in cb['files']:assert sha(R/f['path'])==f['sha256']
 for f,h in p['pins'].items():assert sha(R/f)==h
 for f in bar['files']:assert sha(R/f['path'])==f['sha256']
 put(D/'GT_EXPOSURE.json',dict(epoch=time.time(),global_barrier_sha256=sha(D/'GLOBAL_PREDICTION_BARRIER.json'),all128_before_GT=True))
 from scripts.score_stvg_opd_p1_v1 import truths
 from vg_tta.tastvg_oracle_event5_v1 import DenseTube,official
 gt,spans,prov=truths('hc2',dict(parents=[r['query_ordinal']for r in p['records']]))
 raw=read(R/'artifacts/decota_paper_experiments_v1/hc2/PLAN.json')['rows'];old={r['alias']:r for r in read(R/'results/stvg_error_state/2026-10-19/temporal_increment128/RESULTS_PUBLIC.json')['rows']};rows=[];checks=0
 for r in p['records']:
  a=r['alias'];z=load(D/'predictions'/f'{a}.pt');ctrl=load(D/'no_RGB_controls'/f'{a}.pt');z['outputs'].update(ctrl['outputs']);row=raw[r['query_ordinal']];g=gt[r['query_ordinal']];span=spans[r['query_ordinal']];values={}
  for space,boxes in z['trajectories'].items():
   tube=DenseTube(boxes,row,g,span,clip=True)
   for arm,o in z['outputs'].items():
    iv=o['interval'];v=tube.score(iv);off=official(boxes,row,g,span,iv,'hc2');assert max(abs(v[k]-off[k])for k in ['v','t','s'])<2e-10;checks+=1
    if arm=='Native':assert max(abs(v[k]-old[a]['values'][f'{space}_L0'][k])for k in ['v','t','s'])<2e-10
    assert abs(v['s']-tube.score(z['original_interval'])['s'])<1e-12
    values[f'{space}_{arm}']=dict(**v,interval=iv,boundary_MAE=(abs(iv[0]-span[0])+abs(iv[1]-span[1]))/2)
  for arm in p['arms']+['CoarseNoRGBDirect','CoarseNoRGBMix','WindowNoRGBDirect','WindowNoRGBMix']:
   vs=[values[f'Live{s}_{arm}']for s in p['seeds']];values[f'LiveMean_{arm}']={k:avg([v[k]for v in vs])for k in ['v','t','s','boundary_MAE']}
  changed={k:dict(changed=o['interval']!=z['original_interval'])for k,o in z['outputs'].items()}
  support=z['view_meta']
  rows.append(dict(alias=a,values=values,temporal_changes=changed,prior_support=support));print('scored',a,flush=True)
 arms=p['arms']+['CoarseNoRGBDirect','CoarseNoRGBMix','WindowNoRGBDirect','WindowNoRGBMix'];means={k:{m:avg([r['values'][k][m]for r in rows])for m in ['v','t','s','boundary_MAE']}for k in [f'LiveMean_{a}'for a in arms]+['E_Native','E_CoarseMix','E_WindowMix']};comparisons={}
 contrasts=[(f'LiveMean_{a}','LiveMean_Native')for a in arms[1:]]+[(f'LiveMean_{v}{d}',f'LiveMean_{v}NoRGB{d}')for v in ['Coarse','Window']for d in ['Direct','Mix']]+[('LiveMean_CoarseMix','LiveMean_CoarseDirect'),('LiveMean_WindowMix','LiveMean_WindowDirect'),('LiveMean_WindowMix','LiveMean_CoarseMix'),('E_CoarseMix','E_Native'),('E_WindowMix','E_Native')]
 for a,b in contrasts:
  d={k:paired([(100 if k in ['v','t','s']else 1)*(r['values'][a][k]-r['values'][b][k])for r in rows])for k in ['v','t','s','boundary_MAE']};dv=np.array([100*(r['values'][a]['v']-r['values'][b]['v'])for r in rows]);d['tails']=dict(positive=int((dv>1e-10).sum()),negative=int((dv< -1e-10).sum()),flat=int((abs(dv)<=1e-10).sum()),below_minus3=int((dv< -3).sum()),below_minus5=int((dv< -5).sum()),positive_contribution_pp=float(np.maximum(dv,0).mean()),negative_contribution_pp=float(np.minimum(dv,0).mean()),worst=sorted([dict(alias=r['alias'],delta_pp=float(v))for r,v in zip(rows,dv)],key=lambda x:x['delta_pp'])[:8]);comparisons[a+'_minus_'+b]=d
 result=dict(status='completed_development_capability_probe',parents=128,seeds=p['seeds'],means=means,paired=comparisons,rows=rows,cost=read(D/'COST.json'),primary='predeclared actual-view candidates versus Native, all reported',configuration_selection=False,new_temporal_OPD=False,spatial_frozen=True,GT_offline_only=True,independent_confirmation=False)
 put(P/'RESULTS_PUBLIC.json',result);put(P/'INDEPENDENT_AUDIT.json',dict(status='pass',official_DenseTube_checks=checks,all_original_Native_metrics_exact=True,spatial_s_identical=True,all128_sealed_before_GT=True,GT_provenance=prov));put(D/'STATUS.json',dict(stage='scored',done=128,total=128),True);print({k:v['v']for k,v in comparisons.items()},flush=True)
if __name__=='__main__':main()
