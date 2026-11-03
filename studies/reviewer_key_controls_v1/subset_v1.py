"""User-reduced fixed128 CPU scope. Original full plans and outputs stay immutable."""
from common_v1 import *
import numpy as np, torch
import full_core_v1 as core

U=D/'subset128_v1'; V=P/'subset128_v1'; OLD_FULL=core.F
ARMS=core.ARMS+['Current']; PAIRS=core.PAIRS

def rows(ds): return read(U/'selection.json')[ds]

def validate_subset():
 p=read(U/'plan_v1.json'); assert p['fingerprint']==digest({k:v for k,v in p.items() if k!='fingerprint'})
 for group in ['code_sha256','input_sha256']:
  for f,h in p[group].items(): assert sha(R/f)==h,f
 old=read(OLD_FULL/'plan_v1.json')
 for group in ['code_sha256','input_sha256']:
  for f,h in old[group].items(): assert sha(R/f)==h,f
 return p

def prepare():
 assert not (U/'plan_v1.json').exists()
 selection=read(R/'outputs/matched_pipeline_timing_v1/selection.json'); atomic(U/'selection.json',selection)
 inventory={}; reuse=read(selected.PREV/'raw_reuse_index.json')
 for ds,rr in selection.items():
  assert len(rr)==128 and len({r['id'] for r in rr})==128
  inventory[ds]={r['id']:reuse[ds][r['id']] for r in rr}
  assert len({r['metadata']['original_video_id'] for r in rr})==128
 atomic(U/'raw_inventory.json',inventory)
 inputs=[U/'selection.json',U/'raw_inventory.json',OLD_FULL/'plan_v1.json',D/'full_to_subset_preserved_20261103/owned_exit.json']
 for ds in selection:
  for task in ['regions','temporal']:
   inputs += [D/task/ds/'reporting_receipt.json',D/task/ds/'metrics.json',D/task/ds/'sealed.json']
 p=dict(schema='reviewer_fixed128_cached_ROI_uniform_gamma_v1',authorization='2026-11-03 user: 不用跑全量其实，跑一些subset就行了',
  selection='Unchanged original matched-pipeline128 queries /128 distinct source-video parents per domain. Fixed before this study; no outcome-based selection. Previously exposed development panel, not independent unknown confirmation.',
  controls='Same frozen full_core_v1 math: five pooling regions at fixed E path, uniform gamma .5 and1 both directions with actual E/native ROI/postfusion geometry/own posterior/interval/final DP. No parameter or current-method change.',
  acquisition='All256 have original SHA-bound retained raw grids. CPU only, no model/Source/expert/optimizer/video decode. Committed full-study compacts reused where available; otherwise actual pooling/geometry from retained raw. Partial289 full predictions remain unscored and preserved.',
  score='Seal both128-query domains individually before independent CPU scoring of only two new gamma arms; reuse already independently audited Source/region/Current per-query metrics by identity and SHA. Separate all-array/cold/raw-origin/means audit. Queryequal10000parentbootstrap; full signed tails/shortlong/severe uniononce/rescues.',
  counts={ds:128 for ds in selection},new_Source=0,new_expert=0,new_optimizer=0,new_model=0,GPU=False,
  resource='max2 own CPU stages, CPU14-15 nice19 idleIO, Torch1; CUDA hidden. RAM8GiB/disk80GiB/private160GiB/72h.',
  code_sha256={str(f.relative_to(R)):sha(f) for f in [Path(__file__),S/'subset_controller_v1.py',S/'full_core_v1.py',S/'common_v1.py']},
  input_sha256={str(f.relative_to(R)):sha(f) for f in inputs})
 p['fingerprint']=digest(p);atomic(U/'plan_v1.json',p);atomic(V/'protocol_v1.json',p);print(p['fingerprint'],flush=True)

def derive(ds,row):
 key=row['id'];oldrp=OLD_FULL/ds/'receipts'/(key+'.json')
 if oldrp.exists():
  e=read(oldrp)
  for f,h in [('path','sha256'),('compact_path','compact_sha256')]:assert sha(OLD_FULL/e[f])==e[h]
  c=torch.load(OLD_FULL/e['compact_path'],map_location='cpu',weights_only=False,mmap=True);a=core.replay(c)
  with np.load(OLD_FULL/e['path']) as z:assert set(a)==set(z.files) and all(np.array_equal(v,z[k]) for k,v in a.items())
  return a,c,dict(kind='committed_full_compact_reuse',receipt_sha256=sha(oldrp),receipt_path=str(oldrp.relative_to(R)),compact_sha256=e['compact_sha256']),e['diagnostics']
 e=read(U/'raw_inventory.json')[ds][key];assert sha(e['path'])==e['sha256']
 raw=torch.load(e['path'],map_location='cpu',weights_only=False,mmap=True);cap,par=selected.suite().load_parent(ds,row)
 a,c,proof,diag=core.build(ds,row,cap,raw)
 return a,c,dict(kind='retained_raw_actual_repool',raw_path=e['path'],raw_sha256=e['sha256'],parent=par,proof=proof),diag

def preflight():
 validate_subset();infer_guard();torch.set_num_threads(1);cases=[]
 for ds in ['hc2','vidstg']:
  for row in rows(ds)[:2]:
   a,c,p,d=derive(ds,row);b=core.replay(c);aa,cc,pp,dd=derive(ds,row)
   assert set(a)==set(b)==set(aa) and all(np.array_equal(v,b[k]) and np.array_equal(v,aa[k]) for k,v in a.items())
   cases.append(dict(dataset=ds,all_actual_region_gamma_current_arrays_repeat_cold_exact=True))
 atomic(U/'preflight.json',dict(status='passed_four_real_cached_cases',cases=cases,worker=identity(),new_Source=0,new_expert=0,new_model=0,new_GT=0));print('subset preflight passed',flush=True)

def predict(ds):
 plan=validate_subset();assert read(U/'preflight.json')['status']=='passed_four_real_cached_cases';infer_guard();torch.set_num_threads(1)
 dest=U/ds;assert not(dest/'sealed.json').exists();start=time.perf_counter();files={};counts={}
 oldseal=fixed(D/'regions'/ds/'sealed.json')
 for i,row in enumerate(rows(ds),1):
  resources(start);key=row['id'];tick=time.perf_counter();a,c,origin,diag=derive(ds,row);b=core.replay(c)
  assert set(a)==set(b) and all(np.array_equal(v,b[k])for k,v in a.items())
  old=oldseal['files'][key];assert sha(D/old['path'])==old['sha256']
  with np.load(D/old['path']) as z:
   for arm in ['Source',*core.REGIONS]:assert np.array_equal(a[arm],z[arm]) and np.array_equal(a[arm+'_interval'],z[arm+'_interval'])
  out=dest/'predictions'/(key+'.npz');cp=dest/'compact'/(key+'.pt');rp=dest/'receipts'/(key+'.json');out.parent.mkdir(parents=True,exist_ok=True);cp.parent.mkdir(parents=True,exist_ok=True)
  assert not out.exists() and not cp.exists();torch.save(c,cp)
  with out.open('wb') as h:np.savez_compressed(h,**a)
  with np.load(out) as z:assert set(a)==set(z.files) and all(np.array_equal(v,z[k])for k,v in a.items())
  e=dict(path=str(out.relative_to(U)),sha256=sha(out),compact_path=str(cp.relative_to(U)),compact_sha256=sha(cp),origin=origin,diagnostics=diag,GT_read=False,new_Source=0,new_expert=0,worker=identity(),elapsed_wall_seconds=time.perf_counter()-tick)
  atomic(rp,e);files[key]=dict(e,receipt_path=str(rp.relative_to(U)),receipt_sha256=sha(rp));counts[origin['kind']]=counts.get(origin['kind'],0)+1
  atomic(dest/'progress.json',dict(completed=i,total=128,counts=counts,GT_read=False,worker=identity()))
 atomic(dest/'sealed.json',dict(queries=128,files=files,counts=counts,GT_read=False,plan_fingerprint=plan['fingerprint'],elapsed_wall_seconds=time.perf_counter()-start,worker=identity()));print(ds,'sealed128',flush=True)

def previous_metrics(ds):
 out={}
 for task in ['regions','temporal']:
  folder=D/task/ds;ack=read(folder/'reporting_receipt.json');assert ack['status']=='verified_complete_task_domain' and ack['metrics_sha256']==sha(folder/'metrics.json')
  out[task]={r['id']:r for r in read(folder/'metrics.json')['private_rows']}
 return out

def score(ds):
 validate_subset();torch.set_num_threads(1);seal=read(U/ds/'sealed.json');assert seal['queries']==128 and not seal['GT_read'];prior=previous_metrics(ds);truth=selected.labels(ds);rr=[];checks=0;mx=0
 from full_score import metric
 for row in rows(ds):
  key=row['id'];e=seal['files'][key];assert sha(U/e['path'])==e['sha256'] and sha(U/e['receipt_path'])==e['receipt_sha256'];g=truth[key];old=prior['regions'][key];cur=prior['temporal'][key]
  assert old['parent']==cur['parent']==str(g['parent']);vals={k:old['values'][k] for k in ['Source',*core.REGIONS]};vals['Current']=cur['values']['Ours']
  with np.load(U/e['path']) as z:
   inp={**row,'metadata':{**row['metadata'],'frame_ids':z['frame_ids'].tolist()}}
   for arm,gamma in core.GAMMAS:
    vals[arm],err=metric(z[arm],inp,{int(k):v for k,v in g['boxes'].items()},g['interval'],z[arm+'_interval'].tolist(),z['positions8']);checks+=1;mx=max(mx,err)
   default='UniformGamma05' if ds=='hc2' else 'UniformGamma1'
   assert all((v is None and vals['Current'][k] is None)or abs(v-vals['Current'][k])<1e-12 for k,v in vals[default].items())
  rr.append(dict(id=key,parent=old['parent'],duration_group=old['duration_group'],values=vals))
 assert len({r['parent']for r in rr})==128
 atomic(U/ds/'metrics.json',dict(private_rows=rr));report=selected.suite().parent_report(rr,ARMS,PAIRS,V/ds);selected.suite().signed_tails(rr,PAIRS,V/ds)
 atomic(U/ds/'score_receipt.json',dict(status='scored_whole128',queries=128,parents=128,new_official_checks=checks,max_discrepancy=mx,existing_Source_regions_Current_metrics_reused=True,seal_sha256=sha(U/ds/'sealed.json'),metrics_sha256=sha(U/ds/'metrics.json'),worker=identity()));print(ds,'scored',flush=True)

def audit(ds):
 validate_subset();infer_guard();torch.set_num_threads(1);seal=read(U/ds/'sealed.json');sc=read(U/ds/'score_receipt.json');assert sc['seal_sha256']==sha(U/ds/'sealed.json') and sc['metrics_sha256']==sha(U/ds/'metrics.json');prior=previous_metrics(ds)
 for key,e in seal['files'].items():
  for f,h in [('path','sha256'),('compact_path','compact_sha256'),('receipt_path','receipt_sha256')]:assert sha(U/e[f])==e[h]
  c=torch.load(U/e['compact_path'],map_location='cpu',weights_only=False,mmap=True);a=core.replay(c)
  with np.load(U/e['path'])as z:assert set(a)==set(z.files) and all(np.array_equal(v,z[k])for k,v in a.items())
  orig=e['origin']
  if orig['kind']=='retained_raw_actual_repool':assert sha(orig['raw_path'])==orig['raw_sha256']
  else:assert sha(R/orig['receipt_path'])==orig['receipt_sha256']
 rr=read(U/ds/'metrics.json')['private_rows'];report=read(V/ds/'report.json')
 for r in rr:
  for arm in ['Source',*core.REGIONS]:assert r['values'][arm]==prior['regions'][r['id']]['values'][arm]
  assert r['values']['Current']==prior['temporal'][r['id']]['values']['Ours']
 for arm,mm in report['means'].items():
  for k,v in mm.items():
   vv=[r['values'][arm][k]for r in rr if r['values'][arm][k]is not None];assert (v is None and not vv) or abs(v-float(np.mean(vv)))<1e-12
 rec=dict(status='verified_fixed128_ROI_uniform_gamma_CPU',queries=128,parents=128,all_arrays_cold_exact=True,all_file_SHA_inputs_reused_metrics_means_exact=True,score_sha256=sha(U/ds/'score_receipt.json'),report_sha256=sha(V/ds/'report.json'),new_Source=0,new_expert=0,new_model=0,GPU=False,worker=identity())
 atomic(U/ds/'reporting_receipt.json',rec);atomic(V/ds/'reporting_receipt.json',rec);print(ds,'audited',flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','preflight','predict','score','audit']);ap.add_argument('--dataset',choices=COUNTS);x=ap.parse_args();globals()[x.stage](*([]if x.stage in ['prepare','preflight']else[x.dataset]))
