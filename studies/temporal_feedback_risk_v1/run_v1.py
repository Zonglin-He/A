"""Whole-domain seal, independent CPU official score and independent cold audit."""
from common import *
import argparse,gc,traceback
def predict(ds):
 import numpy as np,torch
 from core_v1 import derive
 from stvg.runtime import guard
 sys.addaudithook(guard);torch.set_num_threads(1);p=validate();assert read(D/'preflight_receipt_v1.json')['status']=='passed_four_actual_cached_full_chains_all_controls_repeat_exact';folder=D/ds;assert not(folder/'predict_worker.json').exists();who=identity();atomic(folder/'predict_worker.json',who);start=time.perf_counter();files={}
 for i,row in enumerate(records(ds),1):
  resources(start);tick=time.perf_counter();a,proof,diag=derive(ds,row);key=row['id'];out=folder/'predictions'/(key+'.npz');rp=folder/'receipts'/(key+'.json');assert not out.exists()and not rp.exists();out.parent.mkdir(parents=True,exist_ok=True)
  with out.with_suffix('.tmp').open('wb')as f:np.savez_compressed(f,**a)
  os.replace(out.with_suffix('.tmp'),out)
  with np.load(out)as z:assert set(a)==set(z.files)and all(np.array_equal(v,z[k])for k,v in a.items())
  rec=dict(path=str(out.relative_to(D)),sha256=sha(out),provenance=proof,diagnostics=diag,classification_sha256=sha(D/'text_classification_v4.json'),worker=who,conditional_elapsed_seconds=time.perf_counter()-tick,GT_read=False,new_Source=0,new_expert=0,new_optimizer=0,plan_fingerprint=p['fingerprint']);atomic(rp,rec);files[key]=dict(rec,receipt_path=str(rp.relative_to(D)),receipt_sha256=sha(rp));atomic(folder/'predict_progress.json',dict(completed=i,total=COUNTS[ds],worker=who,GT_read=False));del a;gc.collect()
 assert len(files)==COUNTS[ds];atomic(folder/'sealed.json',dict(queries=len(files),files=files,GT_read=False,worker=who,elapsed_wall_seconds=time.perf_counter()-start,classification_sha256=sha(D/'text_classification_v4.json'),plan_fingerprint=p['fingerprint']))
def severe(rr,a,b):
 dt=[100*(r['values'][a]['t']-r['values'][b]['t'])for r in rr];dv=[100*(r['values'][a]['v']-r['values'][b]['v'])for r in rr]
 return dict(queries=len(rr),severe_t=sum(v< -5 for v in dt),severe_v=sum(v< -5 for v in dv),severe_union_once=sum(t< -5 or v< -5 for t,v in zip(dt,dv)),rescue_union_once=sum(t>5 or v>5 for t,v in zip(dt,dv)))
def score(ds):
 import torch,numpy as np
 from full_score import metric,cluster
 torch.set_num_threads(1);validate();folder=D/ds;sp=folder/'sealed.json';seal=read(sp);assert seal['queries']==COUNTS[ds]and not seal['GT_read'];truth=selected.labels(ds);groups=fixed(D/'text_classification_v4.json')['datasets'][ds]['queries'];out=[];checks=0;maxerr=0.;start=time.perf_counter()
 for i,row in enumerate(records(ds),1):
  resources(start);e=seal['files'][row['id']];assert sha(D/e['path'])==e['sha256']and sha(D/e['receipt_path'])==e['receipt_sha256'];g=truth[row['id']];vals={}
  with np.load(D/e['path'])as z:
   inp={**row,'metadata':{**row['metadata'],'frame_ids':z['frame_ids'].tolist()}}
   for a in ARMS:vals[a],err=metric(z[a],inp,{int(k):v for k,v in g['boxes'].items()},g['interval'],z[a+'_interval'].tolist(),z['positions8']);checks+=1;maxerr=max(maxerr,err)
  assert vals['NoVisualUpdate']==vals['TemperaturePrior2'];out.append(dict(id=row['id'],parent=str(g['parent']),duration_group='short_le5s'if(g['interval'][1]-g['interval'][0])/row['metadata']['fps']<=5 else'long_gt5s',text_group=groups[row['id']]['group'],explicit_sequence_and_multiple_action_cues=groups[row['id']]['explicit_sequence_and_multiple_action_cues'],values=vals,diagnostics=e['diagnostics']))
  if i%100==0:atomic(folder/'score_progress.json',dict(completed=i,total=COUNTS[ds],worker=identity()))
 pub=P/ds;report=selected.suite().parent_report(out,ARMS,PAIRS,pub);selected.suite().signed_tails(out,PAIRS,pub);strata={}
 partitions={g:[r for r in out if r['text_group']==g]for g in ['action_centric','appearance_centric','mixed','uncertain']};partitions['explicit_sequence_and_multiple_action_cues']=[r for r in out if r['explicit_sequence_and_multiple_action_cues']]
 # These score-derived risk bins are strictly offline and never choose parameters.
 partitions.update({name:[r for r in out if lo<=r['values']['NoVisualUpdate']['t']<hi]for name,lo,hi in [('initial_joint_t_below025',0,.25),('initial_joint_t_025_05',.25,.5),('initial_joint_t_atleast05',.5,1.000001)]})
 for name,rr in partitions.items():
  strata[name]=dict(queries=len(rr),parents=len({r['parent']for r in rr}),means={a:{k:float(np.mean([r['values'][a][k]for r in rr if r['values'][a][k]is not None]))if any(r['values'][a][k]is not None for r in rr)else None for k in ['t','v','vIoU_ge_03','vIoU_ge_05','boundary_MAE_seconds','interval_length_seconds']}for a in ARMS},paired={a+'-'+b:cluster(rr,a,b,[('t',100),('v',100),('boundary_MAE_seconds',1)])for a,b in PAIRS}if rr else{},tails={a+'-'+b:severe(rr,a,b)for a,b in PAIRS},prior_wrong_reinforcement_association=dict(queries_with_prior_MAP_mass_increase=sum(r['diagnostics']['prior_MAP_probability_increased']for r in rr),queries_with_t_damage_and_prior_MAP_mass_increase=sum(r['diagnostics']['prior_MAP_probability_increased']and r['values']['FullMAP']['t']<r['values']['NoVisualUpdate']['t']for r in rr),not_a_causal_identification=True))
 atomic(pub/'text_and_initial_risk_strata.json',strata);atomic(folder/'metrics.json',dict(private_rows=out,report=report,strata=strata));rec=dict(status='scored_whole_domain_after_seal',queries=len(out),checks=checks,max_error=maxerr,seal_sha256=sha(sp),classification_sha256=sha(D/'text_classification_v4.json'),worker=identity(),elapsed_wall_seconds=time.perf_counter()-start);atomic(folder/'score_receipt.json',rec);atomic(pub/'score_receipt.json',rec)
def audit(ds):
 import numpy as np,torch
 from core_v1 import derive
 from stvg.runtime import guard
 sys.addaudithook(guard);torch.set_num_threads(1);validate();folder=D/ds;seal=read(folder/'sealed.json');assert read(folder/'score_receipt.json')['seal_sha256']==sha(folder/'sealed.json');start=time.perf_counter();sha_checks=0
 for i,row in enumerate(records(ds),1):
  resources(start);e=seal['files'][row['id']];assert sha(D/e['path'])==e['sha256']and sha(D/e['receipt_path'])==e['receipt_sha256'];a,proof,diag=derive(ds,row);assert proof==e['provenance']and diag==e['diagnostics']
  with np.load(D/e['path'])as z:assert set(a)==set(z.files)and all(np.array_equal(v,z[k])for k,v in a.items())
  sha_checks+=2;atomic(folder/'audit_progress.json',dict(completed=i,total=COUNTS[ds],worker=identity(),GT_read=False));del a;gc.collect()
 rr=read(folder/'metrics.json')['private_rows'];report=read(P/ds/'report.json');assert len(rr)==COUNTS[ds]
 for a,mm in report['means'].items():
  for k,v in mm.items():
   vv=[r['values'][a][k]for r in rr if r['values'][a][k]is not None];assert(v is None and not vv)or abs(v-float(np.mean(vv)))<1e-12
 strata=read(P/ds/'text_and_initial_risk_strata.json');assert sum(strata[g]['queries']for g in ['action_centric','appearance_centric','mixed','uncertain'])==len(rr)
 for name,e in strata.items():
  if name in ['action_centric','appearance_centric','mixed','uncertain']:sub=[r for r in rr if r['text_group']==name]
  elif name=='explicit_sequence_and_multiple_action_cues':sub=[r for r in rr if r['explicit_sequence_and_multiple_action_cues']]
  else:continue
  assert len(sub)==e['queries']
  for a,b in PAIRS:assert severe(sub,a,b)==e['tails'][a+'-'+b]
 rec=dict(status='verified_whole_domain_all_arrays_parent_receipts_means_and_text_counts',queries=len(rr),prediction_receipt_SHA_checks=sha_checks,all_arrays_cold_rederived_exact=True,all_means_recomputed=True,score_sha256=sha(folder/'score_receipt.json'),seal_sha256=sha(folder/'sealed.json'),metrics_sha256=sha(folder/'metrics.json'),worker=identity(),elapsed_wall_seconds=time.perf_counter()-start,new_model=0,new_Source=0,new_expert=0,new_GT_scoring=0,root_signed_analysis_and_publication_pending=True);atomic(folder/'reporting_receipt.json',rec);atomic(P/ds/'reporting_receipt.json',rec)
def preflight():
 import numpy as np,torch
 from core_v1 import derive,smooth
 from stvg.prototype import update
 torch.set_num_threads(1);validate();start=time.perf_counter();proof=[]
 for ds in COUNTS:
  for row in records(ds)[:2]:
   a,m,d=derive(ds,row);b,mm,dd=derive(ds,row);assert m==mm and d==dd and set(a)==set(b)and all(np.array_equal(v,b[k])for k,v in a.items());proof.append(dict(dataset=ds,current_FullMAP_and_Source_and_joint_readout_exact=True,all_eight_controls_repeat_array_exact=True,prior_temperature_MAP_invariant=True,original_input_binding_SHA=m['sha256']))
 q=np.triu(np.ones((4,4),dtype=np.float64));q/=q.sum();assert np.array_equal(update(q,np.zeros_like(q)),q)and abs(smooth(q).sum()-1)<1e-12 and not(smooth(q)[q==0]>0).any()
 rec=dict(status='passed_four_actual_cached_full_chains_all_controls_repeat_exact',cases=proof,worker=identity(),elapsed_wall_seconds=time.perf_counter()-start,new_model=0,new_Source=0,new_expert=0,new_GT=0);atomic(D/'preflight_receipt_v1.json',rec);atomic(P/'preflight_receipt_v1.json',rec)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('command',choices=['predict','score','audit','preflight']);p.add_argument('--dataset',choices=COUNTS);a=p.parse_args()
 try:preflight()if a.command=='preflight'else globals()[a.command](a.dataset)
 except BaseException:atomic(D/'failures'/(a.command+'_'+str(a.dataset)+'.json'),dict(error=traceback.format_exc(),worker=identity()));raise
