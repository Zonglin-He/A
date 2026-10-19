"""Saved-support convex projection, counterfactual readouts then separate GT score."""
from pathlib import Path
import os,sys,time,argparse
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,save,sha
D=ROOT/'artifacts/stvg_fixed_support_projection32_v1';P=ROOT/'results/stvg_error_state/2026-10-19/fixed_support_projection32';CENTER=ROOT/'artifacts/stvg_fine_roi_scoped_opd32_v1';SENS=ROOT/'artifacts/stvg_fine_roi_sensitivity32_v1'
MODES=['Center_live','Tau_low','Tau_high','Step_low','Step_high'];CODE=['vg_tta/stvg_fixed_support_projection32_v1.py','scripts/run_stvg_fixed_support_projection32_v1.py','protocols/stvg_fixed_support_projection32_v1.md']
def prepare():
 from vg_tta.stvg_fixed_support_projection32_v1 import properties
 p=read(SENS/'PLAN_LOCK.json');D.mkdir(parents=True);P.mkdir(parents=True);put(D/'CPU_PROPERTIES.json',properties());put(D/'PLAN_LOCK.json',dict(status='locked',records=p['records'],count=32,modes=MODES,readout='same E +.25Ewh*optimal_mean; all other nodesE; frozen support step1/2 projections separate counterfactuals, not new on-policy',tol=1e-10,maxiter=120,scope='existing allowed & original informative & any real valid',null_component='preserve saved eta_after nullspace',GT_after_all_outputs=True,CPU_cap_seconds=900,created_epoch=time.time(),deadline_epoch=time.time()+1800,development_exposed=True,no_new_GPU_or_teacher_or_training=True));files=CODE+['methods/CURRENT_METHOD.json','methods/MAIN_METHOD.json','methods/PAPER_METHOD.json']+[str((b/'predictions'/(r['alias']+'.pt')).relative_to(ROOT))for b in [CENTER,SENS]for r in p['records']];put(D/'RUNTIME_LOCK.json',dict(plan_sha256=sha(D/'PLAN_LOCK.json'),pins={f:sha(ROOT/f)for f in files}))
def verify():
 from main_method.runtime import resource_guard
 p=read(D/'PLAN_LOCK.json');l=read(D/'RUNTIME_LOCK.json');assert sha(D/'PLAN_LOCK.json')==l['plan_sha256']
 for f,h in l['pins'].items():assert sha(ROOT/f)==h
 resource_guard();return p

def measure():
 import torch,numpy as np
 from vg_tta.stvg_fixed_support_projection32_v1 import project
 torch.set_num_threads(4);p=verify();started=time.perf_counter();files=[]
 for r in p['records']:
  assert time.perf_counter()-started<p['CPU_cap_seconds'];s=load(SENS/'predictions'/(r['alias']+'.pt'));c=load(CENTER/'predictions'/(r['alias']+'.pt'));E=np.array(s['outputs']['E_MBR']);fits=dict(Center_live=c['fits']['Fine_live'],**s['fits']);outputs=dict(s['outputs']);paths={}
  for mode,f in fits.items():
   paths[mode]=[]
   for st,t in enumerate(f['rounds']):
    start=t['eta_after'].numpy();mu=start.copy();valid=t['feedback']['valid'].numpy();active=s['allowed'].numpy()&t['feedback']['informative'].numpy()&valid.any(1);cert=[]
    for j in np.flatnonzero(active):
     z=project(t['standardized_candidates'][j].numpy(),t['feedback']['prob'][j].numpy(),valid[j],start[j],p['tol'],p['maxiter']);mu[j]=z['mu'];cert.append(dict(node=int(j),**z))
    out=E.copy();out[:,:2]+=.25*E[:,2:]*mu;out[~s['allowed'].numpy()]=E[~s['allowed'].numpy()];assert np.isfinite(out).all()and np.array_equal(out[:,2:],E[:,2:]);outputs[mode+'_project'+str(st+1)]=out;paths[mode].append(dict(step=st+1,mu=mu,active=active,certificates=cert,no_feedback_keeps_actual_state=True,frozen_teacher_not_requeried=True))
  files.append(save(D/'predictions'/(r['alias']+'.pt'),dict(alias=r['alias'],dataset=r['dataset'],frame_ids=s['frame_ids'],planned_positions=s['planned_positions'],interval=s['interval'],allowed=s['allowed'],outputs=outputs,paths=paths,GT_read=False)));print(r['alias'],flush=True)
 verify();put(D/'GLOBAL_PREDICTION_BARRIER.json',dict(status='complete',count=32,files=files,all_outputs_before_NEW_GT=True,CPU_seconds=time.perf_counter()-started))

def score():
 import numpy as np,torch
 from scripts.score_stvg_opd_p1_v1 import truths
 from scripts.run_stvg_dino_local_residual32_v1 import metrics,mean
 from scripts.score_stvg_joint_robust_correction32_v1 import paired
 p=verify();bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');assert bar['count']==32
 for f in bar['files']:assert sha(ROOT/f['path'])==f['sha256']
 gt={ds:truths(ds,dict(parents=[r['query_ordinal']for r in p['records']if r['dataset']==ds]))for ds in ['hc2','vidstg']};plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in gt};rows=[]
 for r in p['records']:
  o=load(D/'predictions'/(r['alias']+'.pt'));v=metrics(o,plans[r['dataset']][r['query_ordinal']],gt[r['dataset']][0][r['query_ordinal']],gt[r['dataset']][1][r['query_ordinal']]);summary={}
  for m,steps in o['paths'].items():
   summary[m]=[]
   for st in steps:
    cc=st['certificates'];summary[m].append(dict(active_nodes=len(cc),KL_at_saved_after=mean([x['KL_at_start']for x in cc]),KL_optimum=mean([x['KL_optimum']for x in cc]),removable_KL=mean([x['removable_KL']for x in cc]),max_gradient_inf=max([x['projected_gradient_inf']for x in cc],default=0.),max_null_error=max([x['null_error']for x in cc],default=0.),extrapolated_axis_nodes=sum(x['outside_candidate_axis_range']for x in cc),rank0_nodes=sum(x['rank']==0 for x in cc)))
  rows.append(dict(alias=r['alias'],dataset=r['dataset'],projection=summary,**v))
 arms=list(rows[0]['values']);pairs=[(m+'_project2',m)for m in MODES]+[(m+'_project2','E_MBR')for m in MODES];ep=['v','t','unobserved_actual_s','observed_actual_s'];groups={}
 for ds in gt:
  rr=[r for r in rows if r['dataset']==ds];groups[ds]=dict(count=len(rr),metrics={a:{e:mean([r['values'][a][e]for r in rr])for e in ep}for a in arms},paired={a+'_minus_'+b:{e:paired([None if r['values'][a][e]is None or r['values'][b][e]is None else 100*(r['values'][a][e]-r['values'][b][e])for r in rr])for e in ep if e!='t'}for a,b in pairs})
 put(P/'RESULTS_PUBLIC.json',dict(status='complete',count=32,groups=groups,rows=rows,all_outputs_before_NEW_GT=True,projections_are_frozen_support_counterfactual_not_new_OPD=True,development_exposed=True));put(D/'SCORE_COMPLETION.json',dict(status='complete',count=32));print('score complete',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','measure','score']);a=p.parse_args()
 try:globals()[a.stage]()
 except BaseException as e:
  D.mkdir(exist_ok=True,parents=True);put(D/('FAILURE_'+str(time.time_ns())+'.json'),dict(stage=a.stage,type=type(e).__name__,error=str(e),all_partials_preserved=True));raise
