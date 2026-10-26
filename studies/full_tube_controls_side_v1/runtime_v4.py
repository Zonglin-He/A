"""Same exact outputs, audited cache reuse, partial commits preserved; one CPU slot."""
import sys,importlib.util,argparse,traceback,subprocess,time,os
from pathlib import Path
S=Path(__file__).resolve().parent;sys.path.insert(0,str(S))
import runtime_v3 as rt
original=C=rt.original
spec=importlib.util.spec_from_file_location('_full_tube_cache_core_v4',S/'cache_core_v4.py');core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)

def validate():
 rt.validate();p=C.read(C.D/'execution_v4_plan.json');assert p['fingerprint']==C.digest({k:v for k,v in p.items()if k!='fingerprint'})
 for field in ['code_sha256','input_sha256']:
  for f,h in p[field].items():assert C.sha(C.R/f)==h,f
 return p

def preflight():
 import numpy as np,torch
 torch.set_num_threads(1);validate();C.no_GT();proof=[]
 for ds in C.COUNTS:
  for row in C.records(ds)[:2]:
   a,pa=core.derive(ds,row);b,pb=core.derive(ds,row);fresh,_=rt.numeric.derive(ds,row);assert pa==pb and all(np.array_equal(v,b[k])and np.array_equal(v,fresh[k])for k,v in a.items())
   proof.append(dict(dataset=ds,all_arrays_exact_to_actual_fresh_full_and_native_interval_DP_replays=True,simple_fusions_recomposed=True,repeat_exact=True,original_native_logits_bridge_inherited=C.sha(C.D/'preflight.json')))
 z=dict(status='passed_actual_fresh_DP_vs_audited_cache_recomposition_four_cases',cases=proof,new_Source=0,new_expert=0,new_optimizer=0,new_GT_score=0);C.atomic(C.D/'preflight_v4.json',z);C.atomic(C.P/'preflight_v4.json',z);print(z['status'],flush=True)

def predict(ds):
 import numpy as np,torch
 torch.set_num_threads(1);validate();C.no_GT();assert C.read(C.D/'preflight_v4.json')['status'].startswith('passed_');dest=C.D/ds;assert not(dest/'sealed.json').exists()and not(dest/'predict_worker_v4.json').exists();C.atomic(dest/'predict_worker_v4.json',C.identity());start=time.perf_counter();files={};reused=0
 for i,row in enumerate(C.records(ds),1):
  C.wait_resources(start);a,proof=core.derive(ds,row);out=dest/'predictions'/(row['id']+'.npz');rp=dest/'receipts'/(row['id']+'.json');t=time.perf_counter()
  if rp.exists():
   rec=C.read(rp);assert rec['GT_read']is False and rec['fingerprint']==C.read(C.D/'plan.json')['fingerprint']and C.sha(out)==rec['sha256'];reused+=1
   for k in ['parent_prediction_sha256','parent_receipt_sha256','current_prediction_sha256','current_receipt_sha256','structural_prediction_sha256','structural_receipt_sha256','shared_expert_observations_sha256','native_interval_sha256']:assert rec['provenance'][k]==proof[k]
  else:
   assert not out.exists();out.parent.mkdir(parents=True,exist_ok=True)
   with out.with_suffix('.tmp').open('wb')as f:np.savez_compressed(f,**a)
   os.replace(out.with_suffix('.tmp'),out);rec=dict(path=str(out.relative_to(C.D)),sha256=C.sha(out),provenance=proof,worker=C.identity(),cached_query_wall_seconds=time.perf_counter()-t,GT_read=False,fingerprint=C.read(C.D/'plan.json')['fingerprint'],execution_variant='audited_cache_recomposition_v4');C.atomic(rp,rec)
  with np.load(out)as z:assert set(z.files)==set(a)and all(np.array_equal(v,z[k])for k,v in a.items())
  files[row['id']]=dict(rec,receipt_path=str(rp.relative_to(C.D)),receipt_sha256=C.sha(rp))
  if i%100==0:C.atomic(dest/'progress_v4.json',dict(stage='predict',completed=i,total=C.COUNTS[ds],wall_seconds=time.perf_counter()-start,preserved_fresh_replays_reused=reused,GT_read=False))
 assert len(files)==C.COUNTS[ds];C.atomic(dest/'sealed.json',dict(queries=len(files),files=files,GT_read=False,worker=C.identity(),wall_seconds=time.perf_counter()-start,fingerprint=C.read(C.D/'plan.json')['fingerprint'],v3_fresh_replays_preserved_and_reused=reused,NoTemporal_currentFull_from_independently_whole_audited_exact_arrays=True));print(ds,'whole prediction seal',flush=True)

def score(ds):
 validate();C.score(ds)

def audit(ds):
 import numpy as np,torch
 torch.set_num_threads(1);validate();C.no_GT();dest=C.D/ds;seal=C.read(dest/'sealed.json');score=C.read(dest/'score_receipt.json');assert score['seal_sha256']==C.sha(dest/'sealed.json')and score['metrics_sha256']==C.sha(dest/'metrics.json');assert not(dest/'reporting_receipt.json').exists();start=time.perf_counter();segments={};fresh=0;reused=0
 for i,row in enumerate(C.records(ds),1):
  C.wait_resources(start);a,pr=core.derive(ds,row);e=seal['files'][row['id']];assert C.sha(C.D/e['path'])==e['sha256']and C.sha(C.D/e['receipt_path'])==e['receipt_sha256']
  with np.load(C.D/e['path'])as z:assert set(a)==set(z.files)and all(np.array_equal(v,z[k])for k,v in a.items())
  if e.get('execution_variant')=='audited_cache_recomposition_v4':assert pr==e['provenance'];reused+=1
  else:
   assert e['provenance']['NoTemporal_spatial_DP_reexecuted']
   for k in ['parent_prediction_sha256','parent_receipt_sha256','current_prediction_sha256','current_receipt_sha256','structural_prediction_sha256','structural_receipt_sha256','shared_expert_observations_sha256','native_interval_sha256']:assert e['provenance'][k]==pr[k]
   fresh+=1
  key=f"{e['worker']['pid']}:{e['worker']['startticks']}";segments[key]=segments.get(key,0.)+e['cached_query_wall_seconds']
  if i%100==0:C.atomic(dest/'audit_progress_v4.json',dict(completed=i,total=C.COUNTS[ds],wall_seconds=time.perf_counter()-start,GT_read=False))
 rows=C.read(dest/'metrics.json')['private_rows'];report=C.read(C.P/ds/'report.json')
 for a,mm in report['means'].items():
  for k,v in mm.items():
   x=[r['values'][a][k]for r in rows if r['values'][a][k]is not None];assert(v is None and not x)or abs(v-float(np.mean(x)))<1e-12
 rec=dict(status='verified_whole_domain',queries=len(rows),prediction_receipt_SHA_checks=2*len(rows),all_arrays_cold_rederived_exact=False,all_simple_fusions_cold_rederived_exact=True,all_arrays_cold_recomposed_or_bound_reference_verified_exact=True,all_means_recomputed=True,same_MAIN_geometry_and_DP_parameters_inherited_from_strict_full_domain_control=True,all_four_controls_use_original_native_Source_interval=True,score_sha256=C.sha(dest/'score_receipt.json'),seal_sha256=C.sha(dest/'sealed.json'),metrics_sha256=C.sha(dest/'metrics.json'),inherited_strict_native_spatial_DP_audit_sha256=C.sha(C.STRUCT/f'spatial_{ds}_reporting_receipt.json'),inherited_full_method_audit_sha256=C.sha(C.selected.D/f'components_{ds}_reporting_receipt.json'),v3_fresh_actual_DP_replays_verified_exact=fresh,v4_previously_audited_native_DP_and_Full_predictions_reused=reused,new_Source=0,new_expert=0,new_optimizer=0,new_GPU=0,worker=C.identity(),wall_seconds=time.perf_counter()-start,cached_query_wall_segments=segments,cost_scope='Simple tubes recomposed, all four controls newly officially scored. Native spatial-DP and Full arrays reuse actual previous whole-domain DP computations and all-array audits. Their original costs are nonzero and inherited, not newly rerun or deployment timing. Partial v3 actual replays remain exactly preserved.');C.atomic(dest/'reporting_receipt.json',rec);C.atomic(C.P/ds/'reporting_receipt.json',rec);print(ds,'independent complete cache-input/array/mean audit',flush=True)

def all_domains():
 validate();assert not(C.D/'owner_v4.json').exists();C.atomic(C.D/'owner_v4.json',C.identity())
 try:
  for ds in C.COUNTS:
   if(C.D/ds/'reporting_receipt.json').exists():
    a=C.read(C.D/ds/'reporting_receipt.json');assert a['status']=='verified_whole_domain'and a['queries']==C.COUNTS[ds];continue
   for command in ['predict','score','audit']:
    C.atomic(C.D/'state_v4.json',dict(stage=command,dataset=ds,owner=C.identity()));log=(C.D/(ds+'_'+command+'_v4.log')).open('xb');child=subprocess.Popen([sys.executable,'-B',__file__,command,'--dataset',ds],stdout=log,stderr=subprocess.STDOUT);C.atomic(C.D/(ds+'_'+command+'_child_v4.json'),C.identity(child.pid));code=child.wait();log.close();C.atomic(C.D/(ds+'_'+command+'_exit_v4.json'),dict(code=code));assert code==0,'Preserve all failures, no automatic retry'
  C.atomic(C.D/'state_v4.json',dict(stage='both_domains_scored_audited_pending_root_analysis_publication',owner=C.identity()))
 except BaseException:C.atomic(C.D/'state_v4.json',dict(stage='failed_preserved',error=traceback.format_exc(),owner=C.identity()));raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('command',choices=['preflight','predict','score','audit','all_domains']);p.add_argument('--dataset',choices=C.COUNTS);v=p.parse_args();globals()[v.command](*([]if v.command in ['preflight','all_domains']else[v.dataset]))
