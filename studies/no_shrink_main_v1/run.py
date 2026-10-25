"""Sealed CPU recalculation for the selected kappa0 main method."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
from ns_common import *
import argparse,traceback,subprocess,gc,shutil
TASKS=['components','CG','budget','corruption','spatial']
def prepare():
 assert not(D/'plan.json').exists();check();inputs=[R/'CURRENT_METHOD.json',R/'profiles/no_shrink_map_v1.json',RUN/'plan.json',PREV/'plan.json',PREV/'spatial_plan.json',PREV/'spatial_runtime_v2_plan.json',OLD/'selection.json',OLD/'supports_sealed.json',OLD/'budget_v2_confirm_sealed.json',R/'outputs/map_components_v1/teacher_index.json',PREV/'corruptions/sealed.json',PREV/'corruptions/reporting_receipt.json']
 for ds in COUNTS:inputs += [RUN/ds/'sealed.json',PREV/f'temporal_v2_{ds}_sealed.json',PREV/f'temporal_v2_{ds}_score_receipt.json',PREV/f'temporal_v2_{ds}_reporting_receipt.json',CG/f'{ds}_sealed.json']
 code=[*S.glob('*.py'),R/'stvg/current.py',R/'configs/default.json',R/'stvg/prototype.py',R/'stvg/spatial.py',R/'studies/map_components_v1/run.py',R/'studies/map_paper_completion_v1/spatial_core.py',R/'studies/map_paper_completion_v1/spatial.py',R/'studies/map_paper_completion_v1/spatial_runtime_v2.py',R/'studies/map_paper_completion_v1/common.py',R/'studies/paper_followup_v1/common.py',R/'tools/full_score.py']
 p=dict(schema='selected_no_shrink_main_v1',authorization='Direct user selects no shrinkage as main method and requests recalculation. This selection follows exposed full results.',science=dict(kappa=0,other_TA_parameters_unchanged=True,CG_other_domain_profiles_unchanged=True,final='q1 jointMAP and new-interval exactDP',budget='Own K contexts; restore Torch4 upstream only; all other CPU1thread',structural='Consume committed old collector compacts after3SHA/inputbinding and legacy64 array replay; explicit0 in every changed branch'),tasks=TASKS,cohorts=COUNTS,budget_cohort=128,corruption_views=768,limits=dict(CPU_jobs=2,CPU_affinity=[14,15],nice=19,minimum_RAM=8*2**30,minimum_disk=80*2**30,domain_seconds=259200),paid_grid_collector=dict(existing_HC=read(PREV/'spatial/hc2_worker.json'),old_dispatcher_stopped=True,Vid='Launch original frozen collector exactly once after HC and finite current-method timing. No new expert; old limits and paidraw checkpoint rules retained.'),timing='192actual measuredSource/96expert plus4Source2expert warmup; unchanged fixed16/domain3repeats; new kappa0 output checks. Legacy64 timings remain historical.',code_sha256={str(f.relative_to(R)):sha(f)for f in code},input_sha256={str(f):sha(f)for f in inputs},failure='Preserve all outputs/attempts and source. No autoretry/skip/fallback/hash waiver. All previous64 scientific sources and artifacts remain immutable.')
 p['fingerprint']=digest(p);atomic(D/'plan.json',p);atomic(P/'protocol.json',{k:v for k,v in p.items()if k!='input_sha256'});
 for f in code:q=D/'source_snapshot'/f.relative_to(R);q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(f.read_bytes())
 print(p['fingerprint'])
def preflight():
 import torch,numpy as np
 from stvg.runtime import guard
 from core import derive
 sys.addaudithook(guard);torch.set_num_threads(1);p=validate();proof=[]
 for task in TASKS:
  for ds in COUNTS:
   rr=rows(task,ds)
   if task=='spatial':
    if ds=='vidstg':continue
    rr=[x for x in rr if(PREV/'spatial'/ds/'receipts'/(x['key']+'.json')).exists()]
   item=rr[0];a,meta=derive(task,ds,item);b,_=derive(task,ds,item);assert all(np.array_equal(a[k],b[k])for k in a);proof.append(dict(task=task,dataset=ds,all_arrays_repeat_exact=True,no_Source=0,no_expert=0))
 assert not torch.cuda.is_initialized();atomic(D/'preflight_receipt.json',dict(status='passed',cases=proof,fingerprint=p['fingerprint'],GT_read=False));atomic(P/'preflight_receipt.json',read(D/'preflight_receipt.json'))
def predict(task,ds):
 import torch,numpy as np
 from core import derive
 from stvg.runtime import guard
 sys.addaudithook(guard);torch.set_num_threads(1);p=validate();assert read(D/'preflight_receipt.json')['status']=='passed';who=identity();assert not(D/f'{task}_{ds}_worker.json').exists();atomic(D/f'{task}_{ds}_worker.json',who);rr=rows(task,ds);files={};dest=D/task/ds;start=time.perf_counter()
 for i,item in enumerate(rr,1):
  assert time.perf_counter()-start<259200 and shutil.disk_usage(R).free>80*2**30
  if task=='spatial':
   while not(PREV/'spatial'/ds/'receipts'/(item['key']+'.json')).exists():
    worker=PREV/'spatial'/f'{ds}_worker.json';assert not worker.exists()or not exited(read(worker))or(PREV/'spatial'/f'{ds}_sealed.json').exists(),'Collector exited without complete seal';assert time.perf_counter()-start<259200;time.sleep(5)
  out=dest/'predictions'/(item['key']+'.npz');rp=dest/'receipts'/(item['key']+'.json');assert not out.exists()and not rp.exists();t=time.perf_counter();a,meta=derive(task,ds,item);out.parent.mkdir(parents=True,exist_ok=True)
  with out.with_suffix('.tmp').open('wb')as f:np.savez_compressed(f,**a)
  os.replace(out.with_suffix('.tmp'),out)
  with np.load(out)as z:assert all(np.array_equal(a[k],z[k])for k in a)
  rec=dict(path=str(out.relative_to(D)),sha256=sha(out),provenance=meta,worker=who,conditional_CPU_seconds=time.perf_counter()-t,GT_read=False,new_Source=0,new_expert=0,fingerprint=p['fingerprint']);atomic(rp,rec);files[item['key']]=dict(rec,receipt_path=str(rp.relative_to(D)),receipt_sha256=sha(rp));atomic(D/f'{task}_{ds}_progress.json',dict(completed=i,total=len(rr),worker=who,seconds=time.perf_counter()-start,GT_read=False));del a,meta;gc.collect()
 parentseal=None
 if task=='spatial':
  sp=PREV/'spatial'/f'{ds}_sealed.json'
  while not sp.exists():assert not exited(read(PREV/'spatial'/f'{ds}_worker.json'));time.sleep(2)
  parent=read(sp);assert parent['queries']==COUNTS[ds]
  for key,e in files.items():assert e['provenance']['sha256']==parent['files'][key]['sha256'] and e['provenance']['compact_sha256']==parent['files'][key]['compact_sha256']
  parentseal=sha(sp)
 atomic(D/f'{task}_{ds}_sealed.json',dict(queries=len(files),files=files,GT_read=False,worker=who,seconds=time.perf_counter()-start,parent_collector_seal_sha256=parentseal,fingerprint=p['fingerprint']))
def score(task,ds):
 import numpy as np,torch
 from full_score import metric,cluster
 torch.set_num_threads(1);validate();sp=D/f'{task}_{ds}_sealed.json';seal=read(sp);items=rows(task,ds);assert seal['queries']==len(items)and not seal['GT_read'];truth=labels(ds);out=[];checks=0;maxerr=0.
 for item in items:
  row=item['row'];e=seal['files'][item['key']];assert sha(D/e['path'])==e['sha256']and sha(D/e['receipt_path'])==e['receipt_sha256'];g=truth[row['id']];vals={}
  with np.load(D/e['path'])as z:
   row={**row,'metadata':{**row['metadata'],'frame_ids':z['frame_ids'].tolist()}}
   for a in arms(task):
    pos=z[a+'_positions']if task=='budget'and a!='Source'else z['positions8'];vals[a],err=metric(z[a],row,{int(k):v for k,v in g['boxes'].items()},g['interval'],z[a+'_interval'].tolist(),pos);assert err<2e-10;maxerr=max(maxerr,err);checks+=1
  out.append(dict(id=item['key'],parent=str(g['parent']),duration_group='short_le5s'if(g['interval'][1]-g['interval'][0])/row['metadata']['fps']<=5 else'long_gt5s',view=item.get('view'),values=vals))
 folder=P/task/ds;res=suite().parent_report(out,arms(task),pairs(task),folder);suite().signed_tails(out,pairs(task),folder)
 if task=='corruption':
  for v in ['Blur','JPEG','Dim']:suite().parent_report([x for x in out if x['view']==v],arms(task),pairs(task),folder/v);suite().signed_tails([x for x in out if x['view']==v],pairs(task),folder/v)
 if task=='spatial':
  synthetic=[]
  for r in out:
   v=r['values'];delta={k:v['S1T1'][k]-v['S1T0'][k]-v['S0T1'][k]+v['S0T0'][k]if all(v[a][k]is not None for a in ['S1T1','S1T0','S0T1','S0T0'])else None for k in v['S1T1']};synthetic.append(dict(parent=r['parent'],values={'Interaction':delta,'Zero':{k:0. if x is not None else None for k,x in delta.items()}}))
  atomic(folder/'interaction.json',dict(definition='(S11-S10)-(S01-S00)',paired=cluster(synthetic,'Interaction','Zero')))
 atomic(D/task/ds/'metrics.json',dict(private_rows=out,report=res));rec=dict(status='scored',queries=len(out),checks=checks,max_error=maxerr,seal_sha256=sha(sp),whole_domain_sealed_before_GT=True);atomic(D/f'{task}_{ds}_score_receipt.json',rec);atomic(folder/'score_receipt.json',rec)
def audit(task,ds):
 import numpy as np,torch
 from core import derive
 from stvg.runtime import guard
 sys.addaudithook(guard);torch.set_num_threads(1);validate();seal=read(D/f'{task}_{ds}_sealed.json');score=read(D/f'{task}_{ds}_score_receipt.json');assert score['seal_sha256']==sha(D/f'{task}_{ds}_sealed.json');segments={};checks=0
 for item in rows(task,ds):
  e=seal['files'][item['key']];assert sha(D/e['path'])==e['sha256']and sha(D/e['receipt_path'])==e['receipt_sha256'];a,meta=derive(task,ds,item);assert meta==e['provenance']
  with np.load(D/e['path'])as z:assert all(np.array_equal(a[k],z[k])for k in a)
  checks+=2;w=e['worker'];k=f"{w['pid']}:{w['startticks']}";segments[k]=segments.get(k,0.)+e['conditional_CPU_seconds']
 rr=read(D/task/ds/'metrics.json')['private_rows'];res=read(P/task/ds/'report.json')
 for a,mm in res['means'].items():
  for k,v in mm.items():
   vv=[x['values'][a][k]for x in rr if x['values'][a][k]is not None];assert(v is None and not vv)or abs(v-float(np.mean(vv)))<1e-12
 rec=dict(status='verified_whole_domain',queries=len(rr),prediction_receipt_SHA_checks=checks,all_arrays_cold_rederived_exact=True,all_means_recomputed=True,worker_CPU_seconds=segments,score_sha256=sha(D/f'{task}_{ds}_score_receipt.json'),new_Source=0,new_expert=0,new_GT_scoring=0,scope='Cached conditional CPU replay; inherited paid acquisition remains nonzero; not deployment latency.');atomic(D/f'{task}_{ds}_reporting_receipt.json',rec);atomic(P/task/ds/'reporting_receipt.json',rec)
 export(task,ds)
def export(task,ds):
 validate();folder=P/task/ds;assert(D/f'{task}_{ds}_reporting_receipt.json').exists();r=read(folder/'report.json');text=f'# Selected no-shrinkage {task}: {ds}\n\nComplete sealed, official scored, all-array exact replay audited. Kappa0 selected by direct user after exposed full results; no independent unseen validation claim. All other settings retained.\n\n|Configuration|tIoU %|vIoU %|R.3 %|R.5 %|Severe t/v/union vs native Source|\n|---|---:|---:|---:|---:|---|\n'
 for a,m in r['means'].items():
  v=r['severe_vs_Source'][a];text+=f"|{a}|{100*m['t']:.6f}|{100*m['v']:.6f}|{100*m['vIoU_ge_03']:.6f}|{100*m['vIoU_ge_05']:.6f}|{v['t']}/{v['v']}/{v['union']}|\n"
 text+='\nAll signed positive/negative paired outcomes, pointwise parent-cluster CIs, short/long splits, severe losses and rescues are preserved in report.json and TSVs. Source retains native per-offset MAP/hull. SourceJointMAP isolates readout when present. Conditional cached CPU costs are not end-to-end latency; inherited Source/expert costs are not zero.\n';(folder/'ANALYSIS.md').write_text(text);atomic(D/f'{task}_{ds}_export_receipt.json',dict(status='exported',analysis_sha256=sha(folder/'ANALYSIS.md')))
def domain(task,ds):
 validate();assert not(D/f'{task}_{ds}_domain_worker.json').exists();atomic(D/f'{task}_{ds}_domain_worker.json',identity())
 try:
  for command in ['predict','score','audit']:
   atomic(D/f'{task}_{ds}_state.json',dict(status=command));log=(D/f'{task}_{ds}_{command}.log').open('ab');c=subprocess.Popen([sys.executable,'-B',__file__,command,'--task',task,'--dataset',ds],stdout=log,stderr=subprocess.STDOUT);atomic(D/f'{task}_{ds}_{command}_child.json',identity(c.pid));rc=c.wait();log.close();atomic(D/f'{task}_{ds}_{command}_exit.json',dict(code=rc));assert rc==0,'Preserved failure, no retry'
  atomic(D/f'{task}_{ds}_state.json',dict(status='complete_scored_audited_exported'))
 except BaseException:atomic(D/f'{task}_{ds}_state.json',dict(status='failed_preserved',error=traceback.format_exc()));raise
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('command',choices=['prepare','preflight','predict','score','audit','export','domain']);a.add_argument('--task',choices=TASKS);a.add_argument('--dataset',choices=COUNTS);z=a.parse_args();globals()[z.command]()if z.command in ['prepare','preflight']else globals()[z.command](z.task,z.dataset)
