"""Target128 cache-only pilot, released only by audited SOURCE selection."""
from source_calibration import *
M=D/'matching_pilot';V=P/'matching_pilot'
ARMS=['SourceMatched','PureMatched','Matching','Current']
PAIRS=[('Current','Matching'),('Matching','PureMatched'),('Current','PureMatched')]
def panel(ds):return read(M/'selection.json')[ds]
def prepare_pilot():
 assert not(M/'plan.json').exists();ack=read(K/'root_acceptance.json');assert ack['source_functioning_accepted']and ack['no_target_scores_used']
 assert ack['source_audit_sha256']==sha(K/'audit_receipt.json')and ack['source_report_sha256']==sha(P/'source_validation'/'report.json')
 selection=read(R/'outputs/matched_pipeline_timing_v1/selection.json');atomic(M/'selection.json',selection);reuse=read(SEL.PREV/'raw_reuse_index.json');inventory={ds:{r['id']:reuse[ds][r['id']]for r in rr}for ds,rr in selection.items()};atomic(M/'raw_inventory.json',inventory)
 for ds,rr in selection.items():assert len(rr)==128 and len({r['metadata']['original_video_id']for r in rr})==128
 p=dict(schema='fixed128_same_source_features_masks_current_time_propagation_v1',source_config=ack['selected_configs'],cohort='Unchanged original confirm128 queries/128 distinct parents per direction. Exposed targets; no configuration or source acceptance based on target pilot scores.',feature_layer='Exact original postfusion visual prefix tokens+valid mask used by Current SpatialGeometry, same7x10 grid/native224 preprocessing and same8expert Source-grid softmasks/rawboxes/physicalpositions. No different encoder or teacher. No E/Sourcebox/CurrentDP input to Matching.',readout='All spatial arms inherit exactly the same current interval. Pure uses only raw8expertbox interpolation/nearest endpoint outside; Matching runs source-frozen VideoWalk module adapter, no full original encoder reproduction claim.',score='All128 seal before new matching official scoring; Source/current/pure scalars reused only by bound identity. Actual GT-supported unobserved native-frame IoUs and distance/event groups recomputed for all spatial arms. Queryequal parent10000pointwiseCI, frame counts explicit. Separate allarray/cold/raw/SHA/means/distance audit. Do not tune from target outcomes.',new_Source=0,new_expert=0,new_optimizer=0,new_GPU=0,
  code_sha256={str(f.relative_to(R)):sha(f)for f in [Path(__file__),S/'matching.py',S/'cpu.py',S/'source_calibration.py',R/'stvg/evidence.py']},input_sha256={str(f.relative_to(R)):sha(f)for f in [K/'root_acceptance.json',K/'audit_receipt.json',M/'selection.json',M/'raw_inventory.json',D/'cpu_plan.json']})
 p['fingerprint']=digest(p);atomic(M/'plan.json',p);atomic(V/'protocol.json',p);print(p['fingerprint'],flush=True)
def check_pilot():
 p=read(M/'plan.json');assert p['fingerprint']==digest({k:v for k,v in p.items()if k!='fingerprint'})
 for group in ['code_sha256','input_sha256']:
  for f,h in p[group].items():assert sha(R/f)==h,f
 return p
def derive_pilot(ds,row):
 from stvg.evidence import extract_visual_grids
 plan=check_pilot();e=fixed(M/'raw_inventory.json')[ds][row['id']];assert sha(e['path'])==e['sha256'];raw=torch.load(e['path'],map_location='cpu',weights_only=False,mmap=True);cap,binding=SEL.suite().load_parent(ds,row);assert list(raw['frame_ids'])==list(cap['frame_ids'])
 features,valid,ids,shape,_=extract_visual_grids(raw['spatial_inputs'],raw['frame_ids']);pos=cap['positions8'];native=np.asarray(cap['geometry']['native_boxes']);seed=cap['geometry']['side_maps'][pos,0];cfg=plan['source_config'][ds];boxes,diag=matching.propagate(features,valid,pos,seed,native,shape,(row['metadata']['height'],row['metadata']['width']),**cfg)
 pe=fixed(D/'pure_full'/ds/'sealed.json')['files'][row['id']];assert sha(D/pe['path'])==pe['sha256']
 with np.load(D/pe['path'])as z:
  a={k:z[k].copy()for k in ['frame_ids','positions8','SourceMatched','SourceMatched_interval','PureMatched','PureMatched_interval','Current','Current_interval']};a['Matching']=boxes;a['Matching_interval']=a['Current_interval'].copy()
  assert np.array_equal(a['frame_ids'],ids)and np.array_equal(a['positions8'],pos)and np.array_equal(a['Matching'][pos],native)
 compact=dict(features=features,valid=valid,frame_ids=ids,positions8=pos,seeds=seed,native_boxes=native,shape=shape,image_size=(row['metadata']['height'],row['metadata']['width']),GT_read=False)
 return a,compact,dict(raw_path=e['path'],raw_sha256=e['sha256'],parent_binding=binding,prior_pure_prediction_sha256=pe['sha256'],diagnostics=diag)
def preflight():
 check_pilot();parent.infer_guard();torch.set_num_threads(1);cases=[]
 for ds in COUNTS:
  for row in panel(ds)[:2]:
   a,c,p=derive_pilot(ds,row);b,d,q=derive_pilot(ds,row);assert all(np.array_equal(v,b[k])for k,v in a.items());cases.append(dict(dataset=ds,all_arrays_actual_same_source_feature_repeat_exact=True,all8observedboxes_exact=True,new_Source=0))
 atomic(M/'preflight.json',dict(status='passed_four_real_source_selected_target_cache_cases',cases=cases,GT_read=False));print('target pilot preflight passed noGT',flush=True)
def predict(ds):
 plan=check_pilot();assert read(M/'preflight.json')['status']=='passed_four_real_source_selected_target_cache_cases';parent.infer_guard();torch.set_num_threads(1);dest=M/ds;assert not(dest/'sealed.json').exists();files={};start=time.perf_counter()
 for i,row in enumerate(panel(ds),1):
  parent.resources(start);a,c,proof=derive_pilot(ds,row);out=dest/'predictions'/(row['id']+'.npz');cp=dest/'compact'/(row['id']+'.pt');rp=dest/'receipts'/(row['id']+'.json');out.parent.mkdir(parents=True,exist_ok=True);cp.parent.mkdir(parents=True,exist_ok=True)
  assert not out.exists()and not cp.exists()
  with out.open('wb')as f:np.savez_compressed(f,**a)
  save(cp,c);cold=torch.load(cp,map_location='cpu',weights_only=False,mmap=True);bb,_=matching.propagate(cold['features'],cold['valid'],cold['positions8'],cold['seeds'],cold['native_boxes'],cold['shape'],cold['image_size'],**plan['source_config'][ds]);assert np.array_equal(a['Matching'],bb)
  with np.load(out)as z:assert set(a)==set(z.files)and all(np.array_equal(v,z[k])for k,v in a.items())
  rec=dict(path=str(out.relative_to(M)),sha256=sha(out),compact_path=str(cp.relative_to(M)),compact_sha256=sha(cp),proof=proof,GT_read=False,worker=identity());atomic(rp,rec);files[row['id']]=dict(rec,receipt_path=str(rp.relative_to(M)),receipt_sha256=sha(rp));atomic(dest/'progress.json',dict(completed=i,total=128,GT_read=False,worker=identity()))
 atomic(dest/'sealed.json',dict(queries=128,files=files,GT_read=False,plan_fingerprint=plan['fingerprint'],elapsed_wall_seconds=time.perf_counter()-start,worker=identity()));print(ds,'pilot128 sealed',flush=True)
def distance_rows(ds,seal):
 truth=SEL.labels(ds);out=[]
 from stvg.metrics import box_iou
 from stvg.metrics_base import xyxy
 for row in panel(ds):
  key=row['id'];e=seal['files'][key];g=truth[key];gt={int(k):v for k,v in g['boxes'].items()}
  with np.load(M/e['path'])as z:
   ids=z['frame_ids'];pos=z['positions8'];obs=set(map(int,pos));pixels={a:xyxy(z[a],row['metadata']['width'],row['metadata']['height'])for a in ARMS}
   if ds=='hc2':pixels={a:np.maximum(b,0)for a,b in pixels.items()}
   for mode in ['all_unobserved','common_predicted_event','GT_event']:
    span=None if mode=='all_unobserved'else z['Current_interval']if mode=='common_predicted_event'else g['interval']
    for bi in range(-1,5):
     values={a:[]for a in ARMS}
     for t,f in enumerate(ids):
      f=int(f)
      if t in obs or f not in gt or(span is not None and not(span[0]<=f<span[1])):continue
      d=float(np.min(np.abs(ids[pos]-f)))/row['metadata']['fps'];b=min(int(np.searchsorted([0,.5,1,2,4,float('inf')],d,side='right'))-1,4)
      if bi>=0 and bi!=b:continue
      for arm in ARMS:values[arm].append(float(box_iou(pixels[arm][t],gt[f])))
     n=len(values['Current']);out.append(dict(id=key,parent=str(g['parent']),group=mode+'/'+('all_distances'if bi<0 else f'bin{bi}'),annotated_frames=n,values={a:100*float(np.mean(v))if v else None for a,v in values.items()}))
 return out
def distance_report(rr):
 rep={}
 for group in sorted({r['group']for r in rr}):
  qq=[r for r in rr if r['group']==group and r['annotated_frames']>0];z=[]
  for r in qq:
   v=r['values'];z.append(dict(parent=r['parent'],**v,**{a+'-'+b:v[a]-v[b]for a,b in PAIRS}))
  rep[group]=dict(annotated_frames=sum(r['annotated_frames']for r in qq),statistics=stats(z,list(z[0])[1:]if z else[]))
 return rep
def score(ds):
 check_pilot();torch.set_num_threads(1);dest=M/ds;seal=read(dest/'sealed.json');assert seal['queries']==128 and not seal['GT_read'];truth=SEL.labels(ds);prior={r['id']:r for r in read(D/'pure_full'/ds/'metrics.json')['private_rows']};rr=[];checks=0;mx=0
 from full_score import metric
 for row in panel(ds):
  key=row['id'];e=seal['files'][key];assert sha(M/e['path'])==e['sha256'];g=truth[key];values={a:prior[key]['values'][a]for a in ARMS if a!='Matching'}
  with np.load(M/e['path'])as z:
   inp=dict(row,metadata=dict(row['metadata'],frame_ids=z['frame_ids'].tolist()));values['Matching'],err=metric(z['Matching'],inp,{int(f):b for f,b in g['boxes'].items()},g['interval'],z['Matching_interval'].tolist(),z['positions8']);assert values['Matching']['t']==values['Current']['t'];checks+=1;mx=max(mx,err)
  rr.append(dict(id=key,parent=prior[key]['parent'],duration_group=prior[key]['duration_group'],values=values))
 atomic(dest/'metrics.json',dict(private_rows=rr));SEL.suite().parent_report(rr,ARMS,PAIRS,V/ds);SEL.suite().signed_tails(rr,PAIRS,V/ds);dr=distance_rows(ds,seal);atomic(dest/'distance_private_rows.json',dr);atomic(V/ds/'distance_report.json',distance_report(dr));atomic(dest/'score_receipt.json',dict(queries=128,new_official_checks=checks,max_discrepancy=mx,seal_sha256=sha(dest/'sealed.json'),metrics_sha256=sha(dest/'metrics.json'),distance_rows_sha256=sha(dest/'distance_private_rows.json'),worker=identity()));print(ds,'pilot scored',flush=True)
def audit(ds):
 plan=check_pilot();torch.set_num_threads(1);dest=M/ds;seal=read(dest/'sealed.json');sc=read(dest/'score_receipt.json');assert sc['seal_sha256']==sha(dest/'sealed.json')and sc['metrics_sha256']==sha(dest/'metrics.json')
 for row in panel(ds):
  a,c,p=derive_pilot(ds,row);e=seal['files'][row['id']];assert p==e['proof']
  for f,h in [('path','sha256'),('compact_path','compact_sha256'),('receipt_path','receipt_sha256')]:assert sha(M/e[f])==e[h]
  with np.load(M/e['path'])as z:assert set(a)==set(z.files)and all(np.array_equal(v,z[k])for k,v in a.items())
 rr=read(dest/'metrics.json')['private_rows'];rep=read(V/ds/'report.json')
 for arm,mm in rep['means'].items():
  for field,v in mm.items():
   vv=[r['values'][arm][field]for r in rr if r['values'][arm][field]is not None];assert(v is None and not vv)or abs(v-float(np.mean(vv)))<1e-12
 assert distance_rows(ds,seal)==read(dest/'distance_private_rows.json');assert distance_report(read(dest/'distance_private_rows.json'))==read(V/ds/'distance_report.json')
 ack=dict(status='verified_source_selected_matching_pilot128_allarrays_raw_input_SHA_means_distance',queries=128,report_sha256=sha(V/ds/'report.json'),distance_report_sha256=sha(V/ds/'distance_report.json'),metrics_sha256=sha(dest/'metrics.json'),source_selected_config=plan['source_config'][ds],worker=identity(),new_Source=0,new_expert=0);atomic(dest/'reporting_receipt.json',ack);atomic(V/ds/'reporting_receipt.json',ack);print(ds,'pilot allarray audit passed',flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare_pilot','preflight','predict','score','audit']);ap.add_argument('--dataset',choices=COUNTS);x=ap.parse_args();globals()[x.stage](*([x.dataset]if x.stage in ['predict','score','audit']else[]))
