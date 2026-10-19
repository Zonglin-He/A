"""CPU post-seal evaluation and predeclared temperature choice; never imported by predictor."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,numpy as np,time
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
D=R/'artifacts/stvg_referdino_finalize_v1';P=R/'results/stvg_error_state/2026-10-19/referdino_final';OLD=R/'results/stvg_error_state/2026-10-19/referdino_dualscale_hc128'
EPS=['v','t','s','observed_actual_s','unobserved_actual_s','benchmark_interpolated_s']
def avg(xs):
 x=[float(v)for v in xs if v is not None];return float(np.mean(x))if x else None

def summarize(rows,arms,pairs):
 from scripts.score_stvg_joint_robust_correction32_v1 import paired
 out=dict(count=len(rows),means={a:{e:avg([r['values'][a][e]for r in rows])for e in EPS}for a in arms},paired={})
 for a,b in pairs:
  dif={e:[None if r['values'][a][e]is None or r['values'][b][e]is None else 100*(r['values'][a][e]-r['values'][b][e])for r in rows]for e in EPS}
  out['paired'][a+'_minus_'+b]={e:paired(xs)for e,xs in dif.items()}
  vv=dif['v'];out['paired'][a+'_minus_'+b]['tails']=dict(positive=sum(x>1e-10 for x in vv),negative=sum(x< -1e-10 for x in vv),unchanged=sum(abs(x)<=1e-10 for x in vv),below_minus3=sum(x< -3 for x in vv),below_minus5=sum(x< -5 for x in vv),worst=[dict(alias=r['alias'],delta_pp=x)for x,r in sorted(zip(vv,rows),key=lambda p:p[0])[:5]])
 return out

def main(stage):
 import torch
 from scripts.score_stvg_opd_p1_v1 import truths
 from scripts.run_stvg_dino_local_residual32_v1 import metrics
 torch.set_num_threads(4);plan=read(D/'PLAN_LOCK.json');mode={'development':'Live','fixed':'Fixed','formal':'formal'}[stage];bar=read(D/mode/'GLOBAL_PREDICTION_BARRIER.json');assert bar['count']==(32 if stage=='formal'else 128)
 for f in bar['files']:assert sha(R/f['path'])==f['sha256']
 records=plan['formal_records']if stage=='formal'else plan['records'];gt,spans,_=truths('hc2',dict(parents=[r['query_ordinal']for r in records]));raw=read(R/'artifacts/decota_paper_experiments_v1/hc2/PLAN.json')['rows'];old={r['alias']:r for r in read(OLD/'RESULTS_PUBLIC.json')['rows']};rows=[];trace=[]
 for r in records:
  z=load(D/mode/'predictions'/f"{r['alias']}.pt");m=metrics(z,raw[r['query_ordinal']],gt[r['query_ordinal']],spans[r['query_ordinal']]);values=m['values']
  from methods.stvg_referdino_opd_v1.score import score_one
  independent=score_one(z,dict(boxes_xyxy_pixels=gt[r['query_ordinal']],metadata=raw[r['query_ordinal']]['input'],interval=spans[r['query_ordinal']]))
  assert all((independent[a][e] is None and values[a][e]is None)or abs(independent[a][e]-values[a][e])<2e-10 for a in independent for e in EPS)
  if stage=='development':
   vv=old[r['alias']]['values'];assert values['E_MBR']==vv['E_MBR'];values['Direct']=vv['Direct']
   for arm in ['Live','Fixed']:
    for s in plan['seeds']:values[f'T05_{arm}_{s}']=vv[f'N005F025_{arm}_{s}']
  if stage=='fixed':
   new_values=dict(values);before=read(P/'TEMPERATURE_RESULTS_PUBLIC.json');values.update(next(x['values']for x in before['rows']if x['alias']==r['alias']))
   # New fixed output must override the pre-selection values only for selected tau.
   values.update(new_values)
  for k,f in z['fits'].items():
   rr=[]
   for x in f['rounds']:
    w=x['rollout']['weights']if x['rollout']is not None else None
    rr.append(dict(step=x['step'],loss=x['loss'],updated=x['updated'],ESS=None if w is None else float((1/w.square().sum(-1)).mean()),gradient_L2=float(x['gradient'].norm())if x['gradient']is not None else None,state_step_L2=float(x['state_step'].norm())))
   trace.append(dict(alias=r['alias'],arm=k,updates=f['actual_updates'],seconds=f['seconds'],steps=rr))
  names=[f'T{int(t*100):02}_Live'for t in plan['temperatures']]if stage!='formal'else['Fixed','Live']
  if stage=='fixed':names += [f"T{int(read(D/'SELECTION_LOCK.json')['tau']*100):02}_Fixed"]
  if stage=='development':names+=['T05_Fixed']
  for a in names:
   values[a+'_seedmean']={e:avg([values[f'{a}_{s}'][e]for s in plan['seeds']])for e in EPS}
  rows.append(dict(alias=r['alias'],dataset='hc2',**m))
 if stage=='development':
  arms=['E_MBR','Source_MBR','Direct','T05_Fixed_seedmean']+[f'T{int(t*100):02}_Live_seedmean'for t in plan['temperatures']];pairs=[(a,b)for a in arms[4:]for b in ['E_MBR','Direct','T05_Live_seedmean']if a!=b]
 elif stage=='fixed':
  tau=read(D/'SELECTION_LOCK.json')['tau'];a=f'T{int(tau*100):02}';arms=['E_MBR','Source_MBR','Direct',a+'_Fixed_seedmean',a+'_Live_seedmean'];pairs=[(arms[-1],x)for x in arms[:-1]]+[(arms[-2],'E_MBR')]
 else:arms=['Source_MBR','E_MBR','Direct','Fixed_seedmean','Live_seedmean'];pairs=[('Live_seedmean',x)for x in arms[:-1]]+[('Fixed_seedmean','E_MBR'),('Direct','E_MBR')]
 out=dict(status='postseal_evaluated',count=len(rows),seeds=plan['seeds'],primary=summarize(rows,arms,pairs),rows=rows,GT_only_after_global_seal=True,seed_average_is_metric_not_box_average=True,historical_exposure=True,current_round_development=stage!='formal',formal_prior_exposure='earlier full Source/old method; no final-round tuning'if stage=='formal'else None)
 name={'development':'TEMPERATURE','fixed':'SELECTED_DEVELOPMENT','formal':'FORMAL'}[stage];put(P/(name+'_RESULTS_PUBLIC.json'),out);put(P/(name+'_STEP_DIAGNOSTICS_PUBLIC.json'),dict(rows=trace));print(name,{a:100*v['v']for a,v in out['primary']['means'].items()},flush=True)
 if stage=='development':
  baseline=out['primary']['means']['T05_Live_seedmean']['v'];ranking=sorted([(out['primary']['means'][f'T{int(t*100):02}_Live_seedmean']['v'],t)for t in plan['temperatures']],key=lambda x:(-x[0],x[1]));best=ranking[0][1]if ranking[0][0]>baseline+1e-12 else .05
  sel=dict(status='frozen_before_formal_prediction_GT',tau=best,ranking=[dict(tau=t,mean_full_vIoU=v)for v,t in ranking],selection='predeclared full128 within-parent seed mean then mean parents; ESS not used',results_sha256=sha(P/(name+'_RESULTS_PUBLIC.json')),formal_plan_sha256=sha(D/'PLAN_LOCK.json'),no_more_algorithm_search=True);put(D/'SELECTION_LOCK.json',sel);put(P/'SELECTION_PUBLIC.json',sel)
  cfg=read(R/'methods/stvg_referdino_opd_v1/config.json');cfg.update(tau=best,status='frozen_research_configuration',selection_record=str((P/'SELECTION_PUBLIC.json').relative_to(R)));put(R/'methods/stvg_referdino_opd_v1/config.json',cfg,True)
if __name__=='__main__':main(sys.argv[1])
