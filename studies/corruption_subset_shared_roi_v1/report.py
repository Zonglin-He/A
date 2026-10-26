"""Independent whole128 official scoring, separate cold audit, actual root bridges."""
from subset_common import *
import argparse
ARMS=['Source','FullMAP','CleanSource','CleanFullMAP'];PAIRS=[('FullMAP','Source'),('FullMAP','CleanFullMAP'),('Source','CleanSource')]
def bridge(ds,name):
 import torch,numpy as np
 from stvg.runtime import guard
 from stvg.readout import prediction
 from stvg.teacher_bridge import evidence_key
 from stvg.evidence import SourceEvidence
 from stvg.spatial import build_geometry
 sys.addaudithook(guard);torch.set_num_threads(4);validate();root=unit(ds,name);wait=read(root/'bridge_waiting.json');assert wait['paid_count']==2;checks=[];started=time.perf_counter()
 for key,e in wait['files'].items():
  row=next(r for r in rows(ds)if r['id']==key);scratch=root/'active'/key;rp=scratch/'processed_raw.pt';assert sha(rp)==e['processed_raw_sha256'];raw=torch.load(rp,map_location='cpu',weights_only=False,mmap=True);frames,ids=__import__('stvg.decode',fromlist=['decode']).decode(row['metadata'],ds);view,meta=operators.apply(frames,ids,row['metadata'],ds,condition(name));assert hashlib.sha256(view.tobytes()).hexdigest()==e['transformed_RGB_sha256']and meta==e['transform'];te=e['teacher'];assert sha(te['path'])==te['sha256']and sha(te['receipt'])==te['receipt_sha256'];teacher=torch.load(te['path'],map_location='cpu',weights_only=False);tr=read(te['receipt']);compact=torch.load(D/e['compact_path'],map_location='cpu',weights_only=False,mmap=True);tk,context=evidence_key(view[compact['positions8']],row['metadata']['caption'],positions=compact['positions8'],frame_ids=ids,dataset=ds,fingerprint=fixed(parent.OLD/'shifts_plan.json')['runtime_fingerprint']);assert tr['key']==tk and tr['context']==context
  if e['kind']=='legacy_prefix':
   a,proof=from_legacy(ds,key,condition(name)['name'],raw);b,_=from_legacy(ds,key,condition(name)['name'],raw);a=replay(a);b=replay(b)
  else:
   full=scratch/'full_Source_checkpoint.pt';assert sha(full)==e['full_Source_checkpoint_sha256'];whole=torch.load(full,map_location='cpu',weights_only=False,mmap=True);nat=prediction(whole['zero']['logits'],whole['zero']['boxes'],whole['records'],ids);assert np.array_equal(nat['boxes'].numpy(),raw['Source'])
   cap={**raw,'teacher_native':teacher};model=checkpoint_class()(ds);cap=model.initialize_cached(cap);ev=SourceEvidence.from_capture(raw['spatial_inputs'],ids,raw['planned_positions'],teacher['native_mask_logits'],teacher['input_shape'][-2:],selected_query=teacher['selected_query']);cap['geometry']=build_geometry(ev,cap['E'],teacher['boxes'],collect_mask_diagnostics=True);a=replay(from_capture(cap,nat['physical_interval']));b=replay(from_capture(cap,nat['physical_interval']))
  cold=replay(compact)
  with np.load(D/e['path'])as z:assert all(np.array_equal(a[k],b[k])and np.array_equal(a[k],cold[k])and np.array_equal(a[k],z[k])for k in a)
  checks.append(dict(query=key,raw_sha256=e['processed_raw_sha256'],full_Source_checkpoint_sha256=e['full_Source_checkpoint_sha256'],actual_same_RGB_teacher_context=True,actual_raw_E_ROI_repool_two_repeats_cold_all_arrays_exact=True,prediction_sha256=e['sha256']))
  del raw,teacher,frames,view,compact;gc.collect()
 assert len(checks)==2 and not torch.cuda.is_initialized();rec=dict(status='passed_actual_two_paid_raw_repool_cold_exact',queries=[x['query']for x in checks],cases=checks,bridge_waiting_sha256=sha(root/'bridge_waiting.json'),worker=identity(),seconds=time.perf_counter()-started,new_model=0,new_Source=0,new_expert=0,new_GT=0);atomic(root/'root_actual_bridge.json',rec);mode='legacy_prefix'if name.startswith('legacy_')else'new_Source';atomic(D/f'{ds}_{mode}_root_bridge.json',dict(unit=name,receipt_path=str(root/'root_actual_bridge.json'),sha256=sha(root/'root_actual_bridge.json')));atomic(P/'bridges'/ds/(name+'.json'),rec);print(rec['status'],flush=True)
def score(ds,name):
 import torch,numpy as np
 from full_score import metric
 torch.set_num_threads(1);validate();root=unit(ds,name);seal=read(root/'sealed.json');assert seal['queries']==128 and not seal['GT_read'];assert set(seal['files'])=={r['id']for r in rows(ds)};truth=selected.labels(ds);cleanrows={r['id']:r for r in fixed(selected.D/'components'/ds/'metrics.json')['private_rows']};rr=[];checks=0;maxerr=0.;started=time.perf_counter()
 for row in rows(ds):
  key=row['id'];e=seal['files'][key];assert sha(D/e['path'])==e['sha256'];g=truth[key];v={}
  with np.load(D/e['path'])as z:
   assert np.array_equal(z['frame_ids'],row['metadata']['frame_ids'])
   for a in ['Source','FullMAP']:v[a],err=metric(z[a],row,{int(k):b for k,b in g['boxes'].items()},g['interval'],z[a+'_interval'].tolist(),z['positions8']);assert err<2e-10;maxerr=max(maxerr,err);checks+=1
  cl=cleanrows[key];assert cl['parent']==str(g['parent']);v.update(CleanSource=cl['values']['Source'],CleanFullMAP=cl['values']['FullMAP']);rr.append(dict(id=key,parent=cl['parent'],duration_group=cl['duration_group'],values=v,sampled_hit=not e['transform'].get('no_sampled_hit',False),pixel_identity=e['kind']=='clean_identity'))
 folder=P/name/ds;report=parent.suite().parent_report(rr,ARMS,PAIRS,folder);parent.suite().signed_tails(rr,PAIRS,folder)
 for field in ['sampled_hit','pixel_identity']:
  for value in [False,True]:
   sub=[r for r in rr if r[field]==value]
   if sub:parent.suite().parent_report(sub,ARMS,PAIRS,folder/(field+'_'+str(value)))
 atomic(root/'metrics.json',dict(private_rows=rr,report=report));rec=dict(status='scored_whole128',checks=checks,max_error=maxerr,seal_sha256=sha(root/'sealed.json'),whole_unit_sealed_before_GT=True,worker=identity(),seconds=time.perf_counter()-started);atomic(root/'score_receipt.json',rec);atomic(folder/'score_receipt.json',rec)
def audit(ds,name):
 import torch,numpy as np
 torch.set_num_threads(4);validate();root=unit(ds,name);seal=read(root/'sealed.json');sr=read(root/'score_receipt.json');assert sr['seal_sha256']==sha(root/'sealed.json');checks=0;started=time.perf_counter()
 for key,e in seal['files'].items():
  for f,h in [('path','sha256'),('receipt_path','receipt_sha256'),('compact_path','compact_sha256')]:assert sha(D/e[f])==e[h];checks+=1
  if e['teacher']:
   te=e['teacher'];assert sha(te['path'])==te['sha256']and sha(te['receipt'])==te['receipt_sha256']
  c=torch.load(D/e['compact_path'],map_location='cpu',weights_only=False,mmap=True);a=replay(c)
  with np.load(D/e['path'])as z:assert all(np.array_equal(a[k],z[k])for k in a)
 rr=read(root/'metrics.json')['private_rows'];report=read(P/name/ds/'report.json')
 for a,fields in report['means'].items():
  for f,v in fields.items():
   values=[r['values'][a][f]for r in rr if r['values'][a][f]is not None];assert(v is None and not values)or abs(v-float(np.mean(values)))<1e-12
 assert len(rr)==128 and len({r['parent']for r in rr})==128;rec=dict(status='verified_whole128_all_arrays_inputs_teacher_SHA_means',queries=128,parents=128,SHA_checks=checks,all_arrays_coldreplay_exact=True,all_means_recomputed=True,score_sha256=sha(root/'score_receipt.json'),seal_sha256=sha(root/'sealed.json'),worker=identity(),seconds=time.perf_counter()-started,new_GT=0);atomic(root/'reporting_receipt.json',rec);atomic(P/name/ds/'reporting_receipt.json',rec)
def export(ds,name):
 import numpy as np
 validate();root=unit(ds,name);audit=read(root/'reporting_receipt.json');assert audit['status']=='verified_whole128_all_arrays_inputs_teacher_SHA_means'and audit['score_sha256']==sha(root/'score_receipt.json');seal=read(root/'sealed.json');rr=read(root/'metrics.json')['private_rows'];r=read(P/name/ds/'report.json');tails={}
 for a,b in PAIRS:
  dt=[100*(x['values'][a]['t']-x['values'][b]['t'])for x in rr];dv=[100*(x['values'][a]['v']-x['values'][b]['v'])for x in rr];tails[a+'-'+b]=dict(severe_t=sum(x< -5 for x in dt),severe_v=sum(x< -5 for x in dv),severe_union_once=sum(x< -5 or y< -5 for x,y in zip(dt,dv)),rescue_union_once=sum(x>5 or y>5 for x,y in zip(dt,dv)))
 folder=P/name/ds;atomic(folder/'all_paired_tails.json',tails);atomic(folder/'cost_segments.json',dict(counts=seal['counts'],whole_prediction_worker_seconds=seal['seconds'],model_load=read(root/'model_load.json')if(root/'model_load.json').exists()else None,workers={f'{e["worker"]["pid"]}:{e["worker"]["startticks"]}'for e in []}if False else [seal['worker']],conditional_query_seconds=sum(e['query_seconds']for e in seal['files'].values()),source_capture_finish_request_IO_query_nested_not_additive=True,original_Source_expert_acquisition_nonzero=True,uncached_deployment=False));atomic(folder/'coverage.json',dict(requested=condition(name),queries=128,parents=128,sampled_hit=sum(x['sampled_hit']for x in rr),pixel_identity=sum(x['pixel_identity']for x in rr),perquery_transform=[dict(index=i,**e['transform'])for i,e in enumerate(seal['files'].values())]))
 (folder/'ANALYSIS.md').write_text(f'# {ds} / {name}\n\nFixed128 query/parents; current shared-corrected-box time pooling, kappa0, final trajectory DP, no first auxiliary DP. All data were previously exposed, not unseen validation.\n\nFull minus matched corrupted Source and matched audited clean methods use pointwise paired-parent CIs; all signed losses and >5pp temporal/spatial union tails retained in TSV/JSON. Simulated coverage differs from cross-family physical magnitude. No-sampled-hit and pixel identity are retained. Conditional cached/reconstructed wall is not uncached deployment, original acquisitions nonzero; nested costs must not be added.\n\n'+json.dumps(dict(means=r['means'],paired=r['paired'],tails=tails),indent=2)+'\n')
 rec=dict(status='root_complete_signed_coverage_cost_analysis_pending_publication',dataset=ds,condition=name,audit_sha256=sha(root/'reporting_receipt.json'),report_sha256=sha(folder/'report.json'),tails_sha256=sha(folder/'all_paired_tails.json'),code_sha256=sha(Path(__file__)));atomic(root/'root_export_receipt.json',rec);atomic(folder/'root_export_receipt.json',rec);print(json.dumps(dict(dataset=ds,condition=name,means=r['means'],paired=r['paired'],tails=tails)),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('command',choices=['bridge','score','audit','export']);p.add_argument('--dataset',choices=COUNTS,required=True);p.add_argument('--condition',required=True);a=p.parse_args();globals()[a.command](a.dataset,a.condition)
