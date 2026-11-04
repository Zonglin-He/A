from common import *
import argparse,collections,csv

def prepare():
 assert not(D/'plan.json').exists();torch.set_num_threads(1);teacher_roots=[R.parent/'stvg-opd/outputs/boxfixed_full_v1/teacher_cache',R.parent/'stvg-mainline/outputs/tastvg_paper_v1/teacher_cache',R.parent/'stvg-opd/outputs/superseded/formal_raw3482_20261020/teacher_cache',R.parent/'stvg-opd/outputs/superseded/formal_raw3482_20261020/teacher_cache_v2'];index=collections.defaultdict(list);runtime=fixed(RUN/'plan.json')['provider_runtime_fingerprint']
 for root in teacher_roots:
  for p in sorted(root.glob('*.json')):
   x=read(p);c=x['context']
   assert x['key']==p.stem==hashlib.sha256(json.dumps(c,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
   k=context_key(c['dataset'],c['caption'],c['frame_ids'],c['positions']);index[k].append(dict(payload=str(p.with_suffix('.pt')),receipt=str(p),payload_sha256=x['sha256'],receipt_sha256=sha(p),RGB_sha256=c['RGB_sha256'],producer_runtime_fingerprint=c['fingerprint']))
 inventory={};coverage={}
 for ds in COUNTS:
  rr=rows(ds);keys={r['id']for r in rr};assert len(keys)==len(rr)==COUNTS[ds];sets=[set(fixed(x)['files'])for x in [RUN/ds/'sealed.json',TEMP/'temporal'/ds/'sealed.json',PURE/'pure_full'/ds/'sealed.json',COMBO/'combination'/ds/'sealed.json']];assert all(s==keys for s in sets);inventory[ds]={};coverage[ds]=dict(queries=len(rr),all_four_manifests_exact_full_query_set=True,missing=0,extra=0,duplicates=0)
  for row in rr:
   e=fixed(RUN/ds/'sealed.json')['files'][row['id']]
   with np.load(RUN/ds/e['path'])as z:k=context_key(ds,row['metadata']['caption'],z['frame_ids'],z['positions8']);original_raw=z['native_boxes'].copy()
   hits=index[k];assert hits,(ds,row['id'],'missing raw teacher');current=[h for h in hits if h['producer_runtime_fingerprint']==runtime];original=current if current else hits;matched=[]
   for h in original:
    if not current:
     assert sha(h['payload'])==h['payload_sha256'];native=torch.load(h['payload'],map_location='cpu',weights_only=False,mmap=True)
     if not np.array_equal(np.asarray(native['boxes'],dtype=float),original_raw):continue
    matched.append(h)
   assert matched,(ds,row['id'],'no exact archived original raw expert readout');assert len({h['RGB_sha256']for h in matched})==1,(ds,row['id'],'conflicting RGB context');h=matched[0];inventory[ds][row['id']]={**h,'context_without_RGB_key':k,'duplicate_receipts':len(hits),'exact_original_readout_matches':len(matched),'original_archived_producer_used':not bool(current)}
 atomic(D/'teacher_inventory.json',inventory);atomic(P/'coverage_preflight.json',coverage);cases=[]
 for ds in COUNTS:
  for row in rows(ds)[:2]:
   a,p=inputs(ds,row);b,q=inputs(ds,row);assert p==q and all(np.array_equal(v,b[k])for k,v in a.items());cases.append(dict(dataset=ds,raw_expert_payload_production_exact=True,both_independent_interpolation_definitions_repeat_exact=True,old_centerlogsize_array_close=True))
 atomic(D/'preflight.json',dict(status='four_actual_raw_teacher_cases_passed',cases=cases,new_Source=0,new_expert=0,new_model=0,new_GPU=0,new_GT_score=0))
 files=[D/'teacher_inventory.json',D/'preflight.json',RUN/'plan.json',R/'CURRENT_METHOD_v2.json',R/'profiles/shared_corrected_roi_v1.json']
 for ds in COUNTS:
  j=fixed(RUN/'plan.json')['jobs'][ds];files.extend([old.selected.PROVIDER/j['manifest'],old.selected.PROVIDER/j['labels']])
  files.extend([root/name for root in [RUN/ds,TEMP/'temporal'/ds,PURE/'pure_full'/ds,COMBO/'combination'/ds]for name in ['sealed.json']]);files.extend([TEMP/'temporal'/ds/'metrics.json',PURE/'pure_full'/ds/'metrics.json',COMBO/'combination'/ds/'metrics.json'])
 p=dict(schema='independent_full_two_spatial_tables_recheck_v1',authorization='2026-11-04 user suspects implementation and imbalance; independently redo/check full simple combination and same-interval spatial tables. CPU cached rawexpert only, no method/paper change.',counts=COUNTS,coverage='Every arm exact same original full query roster inside each dataset; queryequal, paired; different dataset sizes1902vs10303 are not reweighting imbalance.',input='Source-state provenance lookup prefers current provider runtime; when the paid original teacher was archived under older producer fingerprints, resolve that original content-addressed payload by exact dataset/caption/full physicalIDs/positions/RGBcontext and all32rawboxcoordinates bitwise equal to immutable production. Bind exact original producer fingerprint/receipt/payloadSHA; no hash is modified or waived, no model runs or GT-based choice. Read original rawteacher boxes after receipt/payloadSHA/context(actualRGBhash,caption,nativephysicalIDs,positions,runtime) checks, production NPZ rawboxes exact, all historical pure/current/SourceTR/captions/gridbindings verified.',interpolation=dict(primary='Independent np.interp center+logwidth/logheight, nearest expert endpoint tails, exact8anchor overwrite; no E/Source input. Compare historical loop interpolation within2e-14 coordinate tolerance for independent arithmetic roundoff.',secondary='Prespecified linear xyxycorner interpolation, same raw8/physicaltimes/nearesttails; report both definitions, never select by target score or relabel historical log-size results.'),arms=ARMS,whole_domain_prediction_seal_before_new_labels=True,score='Independent fresh dense pixel-corner IoU and half-open interval evaluator for all7arms. Cross-check actual original official evaluator for Full,ComboLog,PureLogFixed everyquery. Original Source/SourceTR/pure/Full stored perquery metrics compared, no means-only test. Same GT annotated frames and clipping rules for botharms within each dataset. HC vsVid official clipping rules retained.',independent_audit='Separate process rederive allNPZs/inputSHA/teacheranchors/coverage/pairedmeans, cold replay and scalarpixelIoUreference for all denseGTframes; cached original method unchanged.',observed_contract='Current still can output E/rawexpert at anchors; no hard-anchor repair/overwrite/new method. Numerical table correctness and intended-method implementation correctness reported separately.',new_Source=0,new_expert=0,new_model=0,new_GPU=0,new_optimizer=0,stats='Queryequal means10000pairedparentbootstrap seed20261022, allsigned/severe/rescue/shortlong, pointwise/exposeddata/no target tuning',resource='max2CPU14-15nice19idleIO1threadCUDAhidden,RAM8GiB/disk80/private160/72h',code_sha256={str(x.relative_to(R)):sha(x)for x in list(S.glob('*.py'))+[R/'stvg/metrics.py',R/'stvg/metrics_base.py',R/'stvg/metrics_hc.py',R/'tools/full_score.py',R/'stvg/spatial.py',R/'stvg/current_shared_roi.py']},input_sha256={str(x.relative_to(R)):sha(x)for x in files});p['fingerprint']=digest(p);atomic(D/'plan.json',p);atomic(P/'protocol.json',{k:v for k,v in p.items()if k!='input_sha256'});print(p['fingerprint'],flush=True)

def predict(ds):
 plan=validate();old.infer_guard();torch.set_num_threads(1);start=time.perf_counter();dest=D/ds;assert not(dest/'sealed.json').exists();files={};rr=rows(ds)
 for i,row in enumerate(rr,1):
  resources(start);a,p=inputs(ds,row);pp=dest/'predictions'/(row['id']+'.npz');pp.parent.mkdir(parents=True,exist_ok=True);assert not pp.exists()
  with pp.open('wb')as f:np.savez_compressed(f,**a)
  with np.load(pp)as z:assert set(z.files)==set(a)and all(np.array_equal(v,z[k])for k,v in a.items())
  e=dict(path=str(pp.relative_to(D)),sha256=sha(pp),proof=p,GT_read=False,worker=identity());rp=dest/'receipts'/(row['id']+'.json');atomic(rp,e);files[row['id']]={**e,'receipt_path':str(rp.relative_to(D)),'receipt_sha256':sha(rp)}
  if i%50==0 or i==len(rr):atomic(dest/'progress.json',dict(completed=i,total=len(rr),GT_read=False,worker=identity()))
 atomic(dest/'sealed.json',dict(files=files,queries=len(rr),GT_read=False,worker=identity(),elapsed_wall_seconds=time.perf_counter()-start,plan_fingerprint=plan['fingerprint']));print(ds,'sealed',len(rr),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','predict']);p.add_argument('--dataset',choices=COUNTS);a=p.parse_args();globals()[a.stage](*([]if a.stage=='prepare'else[a.dataset]))
