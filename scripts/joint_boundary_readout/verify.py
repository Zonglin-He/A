"""Independent saved-boundary arithmetic and concurrent-formal integrity audit."""
import os,sys,json
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.temporal_signal_audit.extract import read,write,sha

def run():
 results=[];old_predictions={}
 for version in ['v1','v2']:
  out=ROOT/('outputs/joint_boundary_readout_'+version);pub=ROOT/('results/joint_boundary_readout_'+version);p=read(out/'plan.json');seal=read(out/'sealed.json');r=read(pub/'report.json');receipt=read(out/'score_receipt.json')
  assert seal['plan_sha256']==sha(out/'plan.json')and seal['GT_read']is False;assert sha(pub/'report.json')==receipt['report_sha256']and sha(pub/'signed_queries.tsv')==receipt['signed_sha256']
  for k,v in p['sources'].items():assert sha(ROOT/k)==v
  labels={c:{g['id']:g for g in [json.loads(l)for l in (ROOT/v['path']).read_text().splitlines()if l.strip()]}for c,v in p['labels'].items()}
  checks=0;maxerror=0.;bounds=0
  for item,se,metrics in zip(p['items'],seal['files'],r['queries_anonymous']):
   assert sha(out/se['path'])==se['sha256'];assert metrics['query']==item['anonymous'];z=np.load(out/se['path']);a,b=labels[item['cohort']][item['id']]['interval'];allarms=['Source','Baseline',*p['protocol']['arms']]
   for arm in allarms:
    x,y=z[arm+'_interval'];inter=max(0,min(y,b)-max(x,a));t=inter/((y-x)+(b-a)-inter);error=abs(t-metrics['values'][arm]['t']);assert error<2e-10;maxerror=max(maxerror,error);checks+=1
   for arm,(weight,kind,gated,perm)in p['protocol']['arms'].items():
    k=se['selected'][arm];assert np.array_equal(z[arm+'_interval'],z['candidate_intervals'][k])
    if gated and not se['ambiguous']:assert k==0 and np.array_equal(z[arm],z['Baseline'])
    if not kind.startswith('evidence_'):
     assert z['candidate_utilities'][0]-z['candidate_utilities'][k]<=weight+1e-12;bounds+=1
   if version=='v1':old_predictions[item['anonymous']]={k:z[k].copy()for k in ['Source','Source_interval','Baseline','Baseline_interval','candidate_intervals','candidate_utilities','candidate_features','permuted_features']}
   else:
    for k,v in old_predictions[item['anonymous']].items():assert np.array_equal(v,z[k]),k
  assert r['proof']['official_dense_max_error']<2e-10 and r['proof']['network_calls']==0
  for c in ['HC64','HC32','Vid32']:
   rows=[x for x in r['queries_anonymous']if x['cohort']==c]
   for name in allarms:
    for metric,published in [('v','m_vIoU'),('t','m_tIoU')]:assert abs(np.mean([x['values'][name][metric]for x in rows])-r['cohorts'][c]['means'][name][published])<1e-12
  result=dict(version=version,status='pass',independent_scalar_tIoU_checks=checks,scalar_max_error=maxerror,prior_utility_loss_bound_checks=bounds,gated_unchanged_exact=True,baseline_Source_and_features_unchanged_between_attempts=True,all_predictions_before_GT=True)
  results.append(result);write(pub/'verification.json',result)
 formal=read(ROOT/'outputs/boxfixed_full_v1/plan.json');assert all(sha(ROOT/k)==v for k,v in formal['sources'].items());progress=read(ROOT/'outputs/boxfixed_full_v1/vidstg/progress.json');stat=Path('/proc/'+str(progress['pid'])+'/stat').read_text();assert int(stat[stat.rfind(')')+2:].split()[19])==progress['startticks']
 assert progress['completed']>5276
 guard=dict(status='pass',formal_frozen_source_hashes_unchanged=len(formal['sources']),formal_plan_sha256=sha(ROOT/'outputs/boxfixed_full_v1/plan.json'),formal_worker_identity_verified=True,formal_progress_before=5276,formal_progress_after=progress['completed'],no_signal_or_new_process=True)
 for version in ['v1','v2']:write(ROOT/f'results/joint_boundary_readout_{version}/formal_guard.json',guard)
 print(json.dumps(dict(trials=results,formal=guard),indent=2))
if __name__=='__main__':run()
