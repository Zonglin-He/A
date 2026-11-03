from study_common import *
import argparse
def prepare():
 assert not(D/'plan.json').exists();guard();torch.set_num_threads(1);manifest={}
 for ds in COUNTS:
  manifest[ds]={}
  for row in rows('regions',ds):
   c,raw,p=region_inputs(ds,row);_,m=random_features(ds,row,c,raw);manifest[ds][row['id']]=m
 atomic(D/'random_manifest.json',manifest)
 cases=[]
 for ds in COUNTS:
  for row in rows('regions',ds)[:2]:
   a,p=combination(ds,row);b,q=combination(ds,row);assert p==q and all(np.array_equal(v,b[k])for k,v in a.items())
   # Real cached Source-only posterior and strict pure interpolation bridge.
   c,_=legacy.source(ds,row);qq,_=one_step(c['features']['AuxSourceROI'],c['q0'],kappa=0,beta=1);assert np.array_equal(a['SimpleCombination_interval'],np.asarray(map_interval(c['ids'],qq)))
   pure_boxes=pure(c['ids'],c['pos'],c['geometries']['main']['native_boxes']);assert np.array_equal(pure_boxes,a['SimpleCombination'])
   r,cc,p=region(ds,row);r2,cc2,p2=region(ds,row);cold=replay_random(cc);assert p==p2 and all(np.array_equal(v,r2[k])and np.array_equal(v,cold[k])for k,v in r.items());cases.append(dict(dataset=ds,pure8_and_SourceTR_posterior_actual_cache_exact=True,random_actual_grid_repool_repeat_and_cold_exact=True))
 atomic(D/'preflight.json',dict(status='passed_four_actual_cache_cases',cases=cases,new_Source=0,new_expert=0,new_model=0,new_GPU=0,new_GT=0))
 inputs=[D/'intent.json',D/'random_manifest.json',D/'preflight.json',R/'CURRENT_METHOD_v2.json',R/'profiles/shared_corrected_roi_v1.json',R/'configs/default.json',SUB/'raw_inventory.json',SUB/'selection.json']
 for ds in COUNTS:
  for root in [OLD/'temporal'/ds,PURE/ds,SUB/ds]:
   inputs.extend(root/x for x in ['sealed.json','metrics.json','reporting_receipt.json','score_receipt.json'])
 files=list(S.glob('*.py'))+[R/'studies/reviewer_key_controls_v1/common_v1.py',R/'studies/pure_expert_interpolation_v1/run.py',R/'stvg/representation.py',R/'stvg/prototype.py',R/'stvg/current_shared_roi.py',R/'tools/full_score.py']
 plan=dict(schema='full_pure_SourceTR_combination_and_subset_random_region_v1',authorization=read(D/'intent.json'),counts_full=COUNTS,counts_regions={'hc2':128,'vidstg':128},SourceTemporal='original Source boxes + Source-box pretext features + own one-step jointMAP; no expert',SimpleCombination='exact original raw8center-logsize interpolation + nearest expert tails, SourceTemporal own interval, expert required, no transport/DP',InitializedTemporal='existing expert shared-corrected E boxes + E-ROI temporal interval, no propagation/DP',Full='unchanged original ReferDINO current shared-corrected-ROI kappa0/beta1/context1.5/no firstaux/finalDP; existing soft observed candidates, not repaired or intended-hardanchor validated',fixed_interval='reuse full scored SourceMatched/PureMatched/E-initialized/Full at same Full event each query, t exactly equal, no newGT scoring',regions='same original128parents/domain, fixed same E output trajectory/q0/prototypeupdate/readout; reuse SourceROI/CorrectedROI/Global/Background/Shuffled; only new RandomRegion scored',random='seed20261104, each native physicalframe SHA256 UTF8 seed|dataset|queryalias|physicalframeid modulo row-major feasible placement count; translate exact original expanded1.5ROI weight template uniformly over valid grid positions. Exact per-frame area/shape/nonzero weights multiset retained. Identity/wholeframe/fallback cases retained/count, overlap allowed, no GT/target-score selection. Full manifest precommitted before new scoring.',new_Source=0,new_expert=0,new_model=0,new_GPU=0,new_optimizer=0,whole_domain_seal_then_separate_CPUscore_then_separate_allarray_inputSHA_reusedmeans_audit=True,statistics='same queryequal10000parentpointwiseCI/all signed severe t/vuniononce>5pp/rescue/short<=5slong; exposed cohorts not unknown confirmation, no familywise/equivalence/action-semantics claim',resource='max2 ownCPU jobs CPU14-15nice19idleIO1thread/CUDAhidden,RAM8GiB/disk80GiB/private160GiB/72h; no services or unrelated signals',paper_and_method_changed=False,old_queues_not_resumed=True,code_sha256={str(p.relative_to(R)):sha(p)for p in files},input_sha256={str(p.relative_to(R)):sha(p)for p in inputs})
 plan['fingerprint']=digest(plan);atomic(D/'plan.json',plan);atomic(P/'protocol.json',{k:v for k,v in plan.items()if k!='input_sha256'});print(plan['fingerprint'],flush=True)
def predict(task,ds):
 plan=validate();guard();torch.set_num_threads(1);dest=D/task/ds;assert not(dest/'sealed.json').exists();start=time.perf_counter();entries={}
 for i,row in enumerate(rows(task,ds),1):
  resources(start);key=row['id']
  if task=='combination':a,proof=combination(ds,row);compact=None
  else:a,compact,proof=region(ds,row)
  pp=dest/'predictions'/(key+'.npz');pp.parent.mkdir(parents=True,exist_ok=True);assert not pp.exists()
  with pp.open('wb')as f:np.savez_compressed(f,**a)
  with np.load(pp)as z:assert set(a)==set(z.files)and all(np.array_equal(v,z[k])for k,v in a.items())
  e=dict(path=str(pp.relative_to(D)),sha256=sha(pp),proof=proof,GT_read=False,new_Source=0,new_expert=0,new_model=0,new_GPU=0,worker=identity())
  if compact is not None:
   cp=dest/'compact'/(key+'.pt');cp.parent.mkdir(parents=True,exist_ok=True);torch.save(compact,cp);cold=torch.load(cp,map_location='cpu',weights_only=False,mmap=True);again=replay_random(cold);assert all(np.array_equal(v,again[k])for k,v in a.items());e.update(compact_path=str(cp.relative_to(D)),compact_sha256=sha(cp))
  rp=dest/'receipts'/(key+'.json');atomic(rp,e);entries[key]={**e,'receipt_path':str(rp.relative_to(D)),'receipt_sha256':sha(rp)}
  if i%25==0 or i==len(rows(task,ds)):atomic(dest/'progress.json',dict(completed=i,total=len(rows(task,ds)),GT_read=False,worker=identity()))
 atomic(dest/'sealed.json',dict(queries=len(entries),files=entries,GT_read=False,worker=identity(),elapsed_wall_seconds=time.perf_counter()-start,plan_fingerprint=plan['fingerprint']));print(task,ds,'sealed',len(entries),flush=True)
def prior_metrics(task,ds):
 root=OLD/'temporal'/ds if task=='combination'else SUB/ds;ack=read(root/'reporting_receipt.json');sc=read(root/'score_receipt.json');assert sc['metrics_sha256']==sha(root/'metrics.json');return{r['id']:r for r in read(root/'metrics.json')['private_rows']}
def score(task,ds):
 validate();torch.set_num_threads(1);dest=D/task/ds;seal=read(dest/'sealed.json');assert seal['queries']==len(rows(task,ds))and not seal['GT_read'];truth=selected.labels(ds);prior=prior_metrics(task,ds);records=[];checks=0;mx=0.
 from full_score import metric
 for row in rows(task,ds):
  key=row['id'];e=seal['files'][key];assert sha(D/e['path'])==e['sha256'];g=truth[key];old=prior[key]
  if task=='combination':v={a:old['values'][b]for a,b in [('Source','Source'),('SourceTemporal','SourceTR'),('InitializedTemporal','ExpertInitTR'),('Full','Ours')]};new='SimpleCombination'
  else:v={a:old['values'][a]for a in REGIONS if a!='RandomRegion'};new='RandomRegion'
  with np.load(D/e['path'])as z:
   inp={**row,'metadata':{**row['metadata'],'frame_ids':z['frame_ids'].tolist()}};v[new],err=metric(z[new],inp,{int(k):b for k,b in g['boxes'].items()},g['interval'],z[new+'_interval'].tolist(),z['positions8']);mx=max(mx,err);checks+=1
  if task=='combination':assert v[new]['t']==v['SourceTemporal']['t']and v['InitializedTemporal']['t']==v['Full']['t']
  records.append(dict(id=key,parent=old['parent'],duration_group=old['duration_group'],values=v))
 arms,pairs=(MAIN,MPAIRS)if task=='combination'else(REGIONS,RPAIRS);selected.suite().parent_report(records,arms,pairs,P/task/ds);selected.suite().signed_tails(records,pairs,P/task/ds);atomic(dest/'metrics.json',dict(private_rows=records));atomic(dest/'score_receipt.json',dict(status='whole_sealed_domain_new_arm_official_CPUscore_existing_controls_reused',queries=len(records),new_official_checks=checks,max_discrepancy=mx,seal_sha256=sha(dest/'sealed.json'),metrics_sha256=sha(dest/'metrics.json'),worker=identity()));print(task,ds,'scored',checks,flush=True)
def audit(task,ds):
 validate();guard();torch.set_num_threads(1);dest=D/task/ds;seal=read(dest/'sealed.json');sc=read(dest/'score_receipt.json');assert sc['seal_sha256']==sha(dest/'sealed.json')and sc['metrics_sha256']==sha(dest/'metrics.json');prior=prior_metrics(task,ds)
 for row in rows(task,ds):
  if task=='combination':a,proof=combination(ds,row)
  else:a,c,proof=region(ds,row)
  e=seal['files'][row['id']];assert proof==e['proof']and sha(D/e['path'])==e['sha256']and sha(D/e['receipt_path'])==e['receipt_sha256']
  with np.load(D/e['path'])as z:assert set(a)==set(z.files)and all(np.array_equal(v,z[k])for k,v in a.items())
  if task=='regions':assert sha(D/e['compact_path'])==e['compact_sha256'];cc=torch.load(D/e['compact_path'],map_location='cpu',weights_only=False,mmap=True);aa=replay_random(cc);assert all(np.array_equal(v,aa[k])for k,v in a.items())
 rr=read(dest/'metrics.json')['private_rows'];report=read(P/task/ds/'report.json');n=0
 for r in rr:
  mapping=[('Source','Source'),('SourceTemporal','SourceTR'),('InitializedTemporal','ExpertInitTR'),('Full','Ours')]if task=='combination'else[(a,a)for a in REGIONS if a!='RandomRegion']
  for a,b in mapping:assert r['values'][a]==prior[r['id']]['values'][b]
 for a,m in report['means'].items():
  for k,v in m.items():
   vals=[r['values'][a][k]for r in rr if r['values'][a][k]is not None];assert(v is None and not vals)or abs(v-float(np.mean(vals)))<1e-12;n+=1
 rec=dict(status='whole_domain_independent_allarray_input_receipt_SHA_reusedcontrols_parent_means_audited',queries=len(rr),parents=len({r['parent']for r in rr}),new_official_checks=sc['new_official_checks'],means_checks=n,seal_sha256=sha(dest/'sealed.json'),score_sha256=sha(dest/'score_receipt.json'),metrics_sha256=sha(dest/'metrics.json'),report_sha256=sha(P/task/ds/'report.json'),all_cold_arrays_exact=True,worker=identity(),new_Source=0,new_expert=0,new_model=0,new_GPU=0,new_GT=0);atomic(dest/'reporting_receipt.json',rec);atomic(P/task/ds/'reporting_receipt.json',rec);print(task,ds,'audited',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','predict','score','audit']);p.add_argument('--task',choices=['combination','regions']);p.add_argument('--dataset',choices=COUNTS);a=p.parse_args();globals()[a.stage](*([]if a.stage=='prepare'else[a.task,a.dataset]))
