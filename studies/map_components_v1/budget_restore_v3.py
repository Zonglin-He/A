"""Restore original per-K four-thread arithmetic; preserve failed one-thread run."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import sys,json,time,signal,subprocess,traceback,gc,shutil
from pathlib import Path
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R))
from studies.map_components_v1 import run as old
D=R/'outputs/map_budget_restore_v3';P=R/'results/map_budget_restore_v3'
F=old.D/'failure_budget_HC00344_Uniform4_20261024'

def validate():
 old.validate();p=old.read(D/'plan.json');assert p['fingerprint']==old.digest({k:v for k,v in p.items()if k!='fingerprint'})
 assert old.sha(__file__)==p['code_sha256'];assert old.sha(old.D/'plan.json')==p['parent_plan_sha256']
 for f,h in p['bound_inputs'].items():assert old.sha(f)==h,f
 return p

def derive(task,ds,row):
 assert task=='budget'
 import torch
 prev=torch.get_num_threads();torch.set_num_threads(4)
 try:return old.budget_arrays(ds,row)
 finally:torch.set_num_threads(prev)

def prepare():
 old.validate();assert not(D/'plan.json').exists();D.mkdir(parents=True);P.mkdir(parents=True,exist_ok=True)
 files=sorted((old.D/'budget/hc2/receipts').glob('*.json'));assert len(files)==67
 source_files={str(p):old.sha(p)for p in [old.D/'plan.json',F/'preservation_receipt.json',F/'diagnostic_summary.json',*files]}
 for p in files:source_files[str(old.D/old.read(p)['path'])]=old.sha(old.D/old.read(p)['path'])
 p=dict(schema='MAP_budget_numeric_restore_v3',parent_scientific_fingerprint=old.read(old.D/'plan.json')['fingerprint'],parent_plan_sha256=old.sha(old.D/'plan.json'),code_sha256=old.sha(__file__),bound_inputs=source_files,
  reason='Old budget_v2 uses runtime.setup Torch4 threads. One-thread reconstruction failed strict MBR replay at HC00344 Uniform4 (same interval,4 box elements max2.569361390314384e-12); four-thread reference bitwise exact. Preserve failure; do not relax comparison.',
  math='Exact unchanged old.budget_arrays; Torch4 only inside numerical derive to reproduce original budget arithmetic; same observations/calibration/geometry/repool/q0/q1/MAP/DP. No source/expert/GPU. Full exact old MBR prerequisite retained.',
  limits=dict(max_compute_processes=2,CPU=[14,15],nice=19,io='idle class3',threads_in_derive=4,threads_elsewhere=1,minimum_RAM_bytes=8*2**30,resume_RAM_bytes=10*2**30,max_seconds_each=259200,minimum_disk_free_bytes=80*2**30),
  evaluation='Same original fixed128parents/domain,Hybrid/Uniform2/4/8/16,eachownjointteacher. Whole domainseal before officiallabels,allarrays exactreplay before export. Historical exposure/no tuning.',
  inherited_component_domain=old.read(old.D/'components_vidstg_domain_child.json'),previous_dispatcher=old.read(old.D/'parallel_v2_controller.json'))
 p['fingerprint']=old.digest(p);old.atomic(D/'plan.json',p);old.atomic(P/'protocol.json',{k:v for k,v in p.items()if k!='bound_inputs'});shutil.copy2(__file__,D/'source_snapshot.py');print(p['fingerprint'],flush=True)

def preflight():
 p=validate();import torch,numpy as np
 from stvg.runtime import guard
 sys.addaudithook(guard);torch.set_num_threads(1);who=old.identity();proof={};rows={r['id']:r for r in old.rows('budget','hc2')}
 for rp in sorted((old.D/'budget/hc2/receipts').glob('*.json')):
  rec=old.read(rp);key=rp.stem;t=time.perf_counter();a,meta=derive('budget','hc2',rows[key])
  with np.load(old.D/rec['path'])as z:assert set(z.files)==set(a)and all(np.array_equal(z[k],a[k])for k in a),('Previously committed MAP changed',key)
  proof[key]=dict(original_receipt=str(rp),original_receipt_sha256=old.sha(rp),original_prediction=str(old.D/rec['path']),original_prediction_sha256=rec['sha256'],all_arrays_exact=True,canonical_recheck_seconds=time.perf_counter()-t,worker=who)
  print('verified_prior',key,flush=True);gc.collect()
 failed=rows['HC00344'];a,_=derive('budget','hc2',failed);b,_=derive('budget','hc2',failed);assert all(np.array_equal(a[k],b[k])for k in a)
 first=old.rows('budget','vidstg')[0];a,_=derive('budget','vidstg',first);b,_=derive('budget','vidstg',first);assert all(np.array_equal(a[k],b[k])for k in a)
 assert not torch.cuda.is_initialized();old.atomic(D/'preflight_receipt.json',dict(status='passed',prior_predictions=proof,all67_old_prediction_bytes_SHA_unchanged=True,failed_query_all8_MBR_exact_and_MAP_repeated_exact=True,Vid_first_all8_MBR_exact_and_MAP_repeated_exact=True,new_Source=0,new_expert=0,new_GPU=0,GT_read=False,fingerprint=p['fingerprint']));old.atomic(P/'preflight_receipt.json',{k:v for k,v in old.read(D/'preflight_receipt.json').items()if k!='prior_predictions'})

def predict(task,ds):
 p=validate();assert task=='budget';import torch,numpy as np
 from stvg.runtime import guard
 sys.addaudithook(guard);torch.set_num_threads(1);pre=old.read(D/'preflight_receipt.json');assert pre['status']=='passed';who=old.identity();started=time.perf_counter();files={};rr=old.rows(task,ds);old.atomic(D/f'{task}_{ds}_worker.json',who)
 for i,row in enumerate(rr,1):
  assert time.perf_counter()-started<p['limits']['max_seconds_each']and shutil.disk_usage(R).free>p['limits']['minimum_disk_free_bytes'];key=row['id'];out=D/task/ds/'predictions'/(key+'.npz');rp=D/task/ds/'receipts'/(key+'.json');assert not out.exists()and not rp.exists();out.parent.mkdir(parents=True,exist_ok=True);t=time.perf_counter()
  prior=pre['prior_predictions'].get(key)if ds=='hc2'else None
  if prior:
   assert old.sha(prior['original_prediction'])==prior['original_prediction_sha256']and old.sha(prior['original_receipt'])==prior['original_receipt_sha256'];rec=old.read(prior['original_receipt']);shutil.copy2(prior['original_prediction'],out)
   assert old.sha(out)==prior['original_prediction_sha256'];rec=dict(rec,path=str(out.relative_to(D)),fingerprint=p['fingerprint'],worker=prior['worker'],conditional_CPU_wall_seconds=prior['canonical_recheck_seconds'],imported_original_receipt=prior['original_receipt'],imported_original_receipt_sha256=prior['original_receipt_sha256'],original_one_thread_cost_provenance=rec['conditional_CPU_wall_seconds'],canonical_Torch4_all_arrays_exact=True)
  else:
   a,meta=derive(task,ds,row)
   with out.with_suffix('.tmp').open('wb')as f:np.savez_compressed(f,**a)
   os.replace(out.with_suffix('.tmp'),out)
   with np.load(out)as z:assert all(np.array_equal(z[k],a[k])for k in a)
   rec=dict(path=str(out.relative_to(D)),sha256=old.sha(out),provenance=meta,worker=who,conditional_CPU_wall_seconds=time.perf_counter()-t,GT_read=False,new_Source=0,new_expert=0,new_GPU=0,fingerprint=p['fingerprint'],canonical_Torch4=True);del a,meta
  old.atomic(rp,rec);files[key]=dict(rec,receipt_path=str(rp.relative_to(D)),receipt_sha256=old.sha(rp));old.atomic(D/f'{task}_{ds}_progress.json',dict(completed=i,total=len(rr),worker=who,seconds=time.perf_counter()-started,GT_read=False));print(task,ds,i,flush=True);gc.collect()
 old.atomic(D/f'{task}_{ds}_sealed.json',dict(queries=len(files),files=files,GT_read=False,worker=who,seconds=time.perf_counter()-started,fingerprint=p['fingerprint']))

# Original score/report/domain code objects, rebound only to the new namespace,
# strict version validator and the restored original numerical derive above.
import types
scope=dict(vars(old));scope.update(D=D,P=P,validate=validate,derive=derive,predict=predict,__file__=__file__)
score=types.FunctionType(old.score.__code__,scope,'score');reporting=types.FunctionType(old.reporting.__code__,scope,'reporting');scope.update(score=score,reporting=reporting)
domain=types.FunctionType(old.domain.__code__,scope,'domain')

def queue():
 p=validate();assert old.exited(p['previous_dispatcher']);assert not(D/'controller.json').exists();old.atomic(D/'controller.json',old.identity());baseline=old.read(old.D/'Vid_baseline.json');last=baseline['completed'];paused=False;pause_at=0.;active={};pending=['hc2','vidstg'];children={};logs={};done=[];failed=[];cg=old.read(old.MAP/'CG_vidstg_domain_child.json');component=p['inherited_component_domain']
 if not old.exited(component):active['components_vidstg']=component
 try:
  with(D/'preflight.log').open('a')as log:
   c=subprocess.Popen([sys.executable,'-B',__file__,'preflight'],stdout=log,stderr=subprocess.STDOUT);old.atomic(D/'preflight_child.json',old.identity(c.pid));assert c.wait()==0,'Preflight mismatch preserved; no budgets'
  while active or pending:
   for key,w in list(active.items()):
    child=children.get(key);ended=child.poll()is not None if child else old.exited(w)
    if not ended:continue
    if child:logs[key].close()
    if key=='components_vidstg':ok=(old.D/'components_vidstg_reporting_receipt.json').exists()
    else:ok=child.returncode==0 and(D/f'budget_{key}_reporting_receipt.json').exists()
    (done if ok else failed).append(key);del active[key]
   current=old.rate();low=old.mem()<p['limits']['minimum_RAM_bytes'];slow=current['completed']-last>=25 and current['mean']>1.35*baseline['mean'];groups=[*active.values(),cg]
   if not paused and(low or slow):
    for w in groups:old.group_signal(w,signal.SIGSTOP)
    paused=True;pause_at=time.time();last=current['completed'];old.atomic(D/'resource_guard.json',dict(status='own_CPU_groups_paused',time=pause_at,current=current))
   elif paused and old.mem()>p['limits']['resume_RAM_bytes']and time.time()-pause_at>=60:
    for w in groups:old.group_signal(w,signal.SIGCONT)
    paused=False;last=current['completed'];old.atomic(D/'resource_guard.json',dict(status='own_CPU_groups_resumed',time=time.time(),current=current))
   if not slow and current['completed']-last>=25:last=current['completed']
   if failed:pending=[]
   while pending and len(active)<2 and not paused and not low:
    ds=pending.pop(0);assert not(D/f'budget_{ds}_domain_child.json').exists();log=(D/f'budget_{ds}_domain.log').open('a');c=subprocess.Popen([sys.executable,'-B',__file__,'domain','--task','budget','--dataset',ds],stdout=log,stderr=subprocess.STDOUT,start_new_session=True);w=old.identity(c.pid);old.atomic(D/f'budget_{ds}_domain_child.json',w);active[ds]=w;children[ds]=c;logs[ds]=log
   state=dict(status='running'if active else'complete_scored_reported_pending_export_publication',active=active,pending=pending,completed=done,failures=failed,paused=paused,fingerprint=p['fingerprint']);old.atomic(D/'state.json',state);old.atomic(D/'guard_latest.json',dict(time=time.time(),Vid=current,baseline=baseline,RAM=old.mem(),paused=paused))
   if active or pending:time.sleep(5)
  assert not failed,'Failure preserved'
 except BaseException as e:old.atomic(D/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),fingerprint=p['fingerprint']));raise
 finally:
  if paused:
   for w in [*active.values(),cg]:old.group_signal(w,signal.SIGCONT)

if __name__=='__main__':
 import argparse
 a=argparse.ArgumentParser();a.add_argument('command',choices=['prepare','preflight','queue','domain','predict','score','report']);a.add_argument('--task',default='budget');a.add_argument('--dataset',choices=['hc2','vidstg']);v=a.parse_args();dict(prepare=prepare,preflight=preflight,queue=queue,domain=lambda:domain(v.task,v.dataset),predict=lambda:predict(v.task,v.dataset),score=lambda:score(v.task,v.dataset),report=lambda:reporting(v.task,v.dataset))[v.command]()
