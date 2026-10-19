"""Post-seal validation only; no new predictions, selection, or fitting."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import sys,time,numpy as np
R=Path(__file__).resolve().parents[4];sys.path.insert(0,str(R))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
from scripts.run_stvg_temporal_foreground_opd128_v1 import D,P,HR
from scripts.score_stvg_opd_p1_v1 import truths
from vg_tta.tastvg_oracle_event5_v1 import DenseTube,official
from main_method.runtime import resource_guard

def main():
 import torch
 torch.set_num_threads(4);beg=time.perf_counter();p=read(D/'PLAN_LOCK.json');bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');assert bar['count']==128 and read(P/'ROOT_AUDIT.json')['status']=='pass'
 assert all(sha(R/f['path'])==f['sha256']for f in bar['files'])
 gt,spans,prov=truths('hc2',dict(parents=[r['query_ordinal']for r in p['records']]));raw=read(R/'artifacts/decota_paper_experiments_v1/hc2/PLAN.json')['rows'];results={r['alias']:r for r in read(P/'RESULTS_PUBLIC.json')['rows']};ii,jj=np.triu_indices(65,1);checks=0;scalar_checks=0;maxerr=0.
 for r in p['records']:
  resource_guard();assert time.perf_counter()-beg<900;a=r['alias'];z=load(D/'predictions'/f'{a}.pt');rows=[];trows=[];ids=np.asarray(z['frame_ids']);g,h=spans[r['query_ordinal']]
  for space,boxes in z['trajectories'].items():
   if space=='E':continue
   tube=DenseTube(boxes,raw[r['query_ordinal']],gt[r['query_ordinal']],spans[r['query_ordinal']],clip=True)
   # Direct DenseTube interval calls bypass scorer vector prefix arithmetic.
   scores=[tube.score([int(ids[i]),int(ids[j])+1])for i,j in zip(ii,jj)];v=np.array([x['v']for x in scores]);t=np.array([x['t']for x in scores]);rows.append(v);trows.append(t);scalar_checks+=len(scores)
   for k in np.linspace(0,len(ii)-1,9).astype(int):
    off=official(boxes,raw[r['query_ordinal']],gt[r['query_ordinal']],spans[r['query_ordinal']],[int(ids[ii[k]]),int(ids[jj[k]])+1],'hc2');e=max(abs(scores[k][m]-off[m])for m in ['v','t']);assert e<2e-10;maxerr=max(maxerr,e);checks+=1
  for name in ['Native','TeacherDirect','WrongTeacherDirect','HiResAuxMix']:
   for key,curves in [('v',rows),('t',trows)]:
    x=float(z['outputs'][name]['q'][ii,jj]@np.mean(curves,0));e=abs(x-results[a]['expected_candidate_quality'][name][key]);assert e<2e-10;maxerr=max(maxerr,e)
 put(P/'EXPECTED_QUALITY_INDEPENDENT_AUDIT.json',dict(status='pass',all128_sealed_before_any_fresh_GT=True,DenseTube_scalar_candidate_calls=scalar_checks,official_spot_checks=checks,maxerror=maxerr,CPU_seconds=time.perf_counter()-beg,GPU_seconds=0,new_predictions=0,new_updates=0,GT_provenance=prov))
 print('expected quality audit pass',scalar_checks,checks,maxerr,flush=True)
if __name__=='__main__':main()
