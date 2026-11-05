"""Separate CPU input/mask, official, independent pixel and statistical audits."""
from common import *
import argparse,ast,csv
def independent_box(mask):
 h,w=mask.shape;yy=np.flatnonzero(mask.any(axis=1));xx=np.flatnonzero(mask.any(axis=0))
 if not len(xx):return np.zeros(4,np.float64)
 return np.array([(int(xx[0])+int(xx[-1])+1)/(2*w),(int(yy[0])+int(yy[-1])+1)/(2*h),(int(xx[-1])-int(xx[0])+1)/w,(int(yy[-1])-int(yy[0])+1)/h],np.float64)
def entry(ds,key,sealed=True):
 return fixed(D/'propagation'/ds/'sealed.json')['files'][key]if sealed else read(D/'propagation'/ds/'receipts'/(key+'.json'))
def checked(ds,row,e,raw=False):
 from studies.review_strengthening_v1.sam2_input_adapter_v1 import selected_prompts,lossless_rgb_loader
 a,t,ref=reference(ds,row);pos=a['positions8'];m=selected_prompts(t,row['metadata']['height'],row['metadata']['width']);binding=e['input_binding']
 assert binding['teacher_payload_sha256']==ref['current_teacher']['payload_sha256']and binding['reference_prediction_sha256']==ref['sha256']and binding['mask_prompt_sha256']==hashlib.sha256(m.tobytes()).hexdigest()
 assert binding['Full_interval_sha256']==hashlib.sha256(a['Ours_interval'].tobytes()).hexdigest()and e['ReferDINO_mask_input']and not e['Source_prompt_input']
 assert e['new_Source']==e['new_ReferDINO']==e['new_optimizer']==0 and e['raw_before_validation']and e['actual_CUDA_postprocessor_calls']>0 and e['actual_image_encoder_calls']>0 and e['observed_mask_boxes_not_overwritten']
 for k,h in [('path','sha256'),('mask_path','mask_sha256')]:assert sha(D/e[k])==e[h]
 n=len(a['frame_ids']);h,w=row['metadata']['height'],row['metadata']['width'];boxes=[];empty=[]
 with np.load(D/e['mask_path'])as z:
  assert np.array_equal(z['native_shape'],[n,h,w])and np.array_equal(z['positions8'],pos)and z['packed'].shape==(n,(h*w+7)//8)
  for i,x in enumerate(z['packed']):
   mask=np.unpackbits(x)[:h*w].reshape(h,w).astype(bool);boxes.append(independent_box(mask))
   if not mask.any():empty.append(i)
  if raw:
   frames,source,teacher,proof=data(ds,row);proof['mask_prompt_sha256']=hashlib.sha256(m.tobytes()).hexdigest();assert proof==binding
   images,_,_=lossless_rgb_loader(frames[pos],1024,True);assert hashlib.sha256(images.contiguous().numpy().tobytes()).hexdigest()==e['actual_preprocessing']['eight_preprocessed_sha256'];seen=[]
   for re in e['raw_float_checkpoints']:
    assert sha(D/re['path'])==re['sha256'];v=torch.load(D/re['path'],map_location='cpu',weights_only=False,mmap=True);i=int(v['frame_index']);assert v['input_binding']==binding and not v['GT_read']and torch.isfinite(v['scores']).all();mask=(v['scores'][0,0]>0).numpy();assert np.array_equal(mask,np.unpackbits(z['packed'][i])[:h*w].reshape(h,w).astype(bool));seen.append(i)
   assert sorted(seen)==list(range(n))
 assert len(empty)==e['empty_frames']and sum(i in set(pos)for i in empty)==e['empty_conditioning_frames']
 with np.load(D/e['path'])as z:
  assert np.array_equal(z['Full'],a['Ours'])and np.array_equal(z['SAM2MatchedMask'],np.stack(boxes))and np.array_equal(z['shared_interval'],a['Ours_interval'])and np.array_equal(z['frame_ids'],a['frame_ids'])and np.array_equal(z['positions8'],pos)
 return dict(prediction_sha256=e['sha256'],teacher_sha256=binding['teacher_payload_sha256'],allmask_halfopen_boxes_times_input_SHA_exact=True,actual_RGB_preprocessed_and_full_paid_raw_checked=raw)
def bridge(ds):
 plan=check();assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';torch.set_num_threads(1)
 from stvg.runtime import guard
 sys.addaudithook(guard);dest=D/'propagation'/ds;wait=read(dest/'bridge_input.json');assert len(wait['keys'])==2 and not(dest/'actual_bridge.json').exists();by={r['id']:r for r in rows(ds)}
 cases=[checked(ds,by[k],entry(ds,k,False),True)for k in wait['keys']];assert not torch.cuda.is_initialized()
 atomic(dest/'actual_bridge.json',dict(status='passed_actual_two_new_mask_prompt_SAM21_RGB_teacher_track_masks_preprocess_allfloat_allmask_nativeboxes_sharedFulltime_exact',cases=cases,plan_fingerprint=plan['fingerprint'],worker=identity(),new_Source=0,new_expert=0,new_GT=0));print(ds,'actual mask-prompt bridge passed',flush=True)
def input_audit(ds):
 check();assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';torch.set_num_threads(1);dest=D/'propagation'/ds;seal=read(dest/'sealed.json');assert seal['queries']==128 and not seal['GT_read'];start=time.perf_counter()
 from stvg.runtime import guard
 sys.addaudithook(guard)
 for row in rows(ds):
  e=entry(ds,row['id']);assert sha(D/e['receipt_path'])==e['receipt_sha256'];checked(ds,row,e)
 atomic(dest/'input_audit.json',dict(status='all128_sameexpert_mask_nativeboxes_Fulltime_inputSHA_exact',queries=128,seal_sha256=sha(dest/'sealed.json'),worker=identity(),new_GT=0,elapsed_wall_seconds=time.perf_counter()-start));print(ds,'all128 input/mask audit passed',flush=True)
def metric_functions():
 path=R/'studies/shared_time_spatial_recalc_v2/score.py';tree=ast.parse(path.read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name in ['pixels_at_gt','measure']];assert len(nodes)==2
 scope=dict(np=np,xyxy=lambda b,w,h:np.column_stack([b[:,:2]-b[:,2:]/2,b[:,:2]+b[:,2:]/2])*[w,h,w,h]);exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),scope);return scope['measure']
def derive(ds,manual=False):
 dest=D/'propagation'/ds;assert read(dest/'sealed.json')['queries']==128 and read(dest/'input_audit.json')['seal_sha256']==sha(dest/'sealed.json')
 from studies.map_paper_completion_v1.common import labels
 from stvg.metrics import official
 truth=labels(ds);measure=metric_functions();out=[];checks=0;mx=0.
 for row in rows(ds):
  key=row['id'];e=entry(ds,key);assert sha(D/e['path'])==e['sha256'];g=truth[key];values={};gt={int(k):b for k,b in g['boxes'].items()}
  with np.load(D/e['path'])as z:
   for arm in ARMS:
    v=measure(z,arm,row['metadata'],gt,g['interval'],ds,reference=manual)
    if not manual:
     officialv=official(z[arm],dict(input=row['metadata'],frame_ids=z['frame_ids'].tolist()),gt,g['interval'],z['shared_interval'],ds)
     for field in ['v','t']:mx=max(mx,abs(v[field]-officialv[field]));assert abs(v[field]-officialv[field])<2e-10
    v['R03']=v.pop('vIoU_ge_03');v['R05']=v.pop('vIoU_ge_05');values[arm]=v;checks+=1
  for f in ['t','event_GT_frames','event_observed_frames','event_unobserved_frames']:assert values['Full'][f]==values['SAM2MatchedMask'][f]
  out.append(dict(id=key,parent=str(g['parent']),duration_group='short_le5s'if(g['interval'][1]-g['interval'][0])/row['metadata']['fps']<=5 else'long_gt5s',values=values))
 return out,checks,mx
def stats(rr,fields):
 if not rr:return dict(queries=0,parents=0,statistics={})
 parents=sorted({r['parent']for r in rr});ix={p:i for i,p in enumerate(parents)};n=len(parents);s=np.zeros((n,len(fields)));c=np.zeros_like(s)
 for r in rr:
  for j,k in enumerate(fields):
   if r.get(k)is not None:s[ix[r['parent']],j]+=r[k];c[ix[r['parent']],j]+=1
 w=np.random.default_rng(20261022).multinomial(n,[1/n]*n,size=10000);den=w@c;b=np.divide(w@s,den,out=np.full_like(den,np.nan),where=den>0);out={}
 for j,k in enumerate(fields):
  vals=[r[k]for r in rr if r.get(k)is not None];boot=b[:,j];boot=boot[np.isfinite(boot)];out[k]=dict(mean=float(np.mean(vals))if vals else None,CI95=np.quantile(boot,[.025,.975]).tolist()if len(boot)else None,queries=len(vals),parents=int((c[:,j]>0).sum()))
 return dict(queries=len(rr),parents=n,statistics=out)
def pair(rr):
 zz=[]
 for r in rr:
  a,b=r['values']['Full'],r['values']['SAM2MatchedMask'];dt=100*(a['t']-b['t']);dv=100*(a['v']-b['v']);z=dict(parent=r['parent'],delta_v_pp=dv,delta_t_pp=dt,improved_v_percent=100.*(dv>1e-10),harmed_v_percent=100.*(dv< -1e-10),severe_union_once_percent=100.*(dt< -5 or dv< -5),rescue_union_once_percent=100.*(dt>5 or dv>5))
  for f in ['event_unobserved_IoU','event_observed_IoU']:z['delta_'+f+'_pp']=None if a[f]is None or b[f]is None else 100*(a[f]-b[f])
  zz.append(z)
 return stats(zz,list(zz[0])[1:])if zz else stats([],[])
def summary(rr):
 means={a:{k:float(np.mean(vv))if(vv:=[r['values'][a][k]for r in rr if r['values'][a][k]is not None])else None for k in rr[0]['values'][a]}for a in ARMS}
 return dict(queries=len(rr),parents=len({r['parent']for r in rr}),means=means,paired={g:pair([r for r in rr if g=='all'or r['duration_group']==g])for g in ['all','short_le5s','long_gt5s']},statistics='queryequal10000parentclusterbootstrap seed20261022 pointwise historicallyexposed panel; null empty conditional support retained denominator for primaryv',scope='Exact common Full event/selected raw8ReferDINO mask evidence and physicaltimes. SAM2 has additional encoder; different representations, knowledge and compute budget. Current softanchors unchanged.')
def score(ds):
 check();torch.set_num_threads(1);start=time.perf_counter();rr,n,mx=derive(ds);dest=D/'propagation'/ds;rep=summary(rr);atomic(dest/'metrics.json',dict(private_rows=rr));atomic(P/'propagation'/ds/'report.json',rep);atomic(dest/'score_receipt.json',dict(status='whole128_official_plus_dense_conditional_scored',queries=128,official_checks=n,official_vs_dense_max_error=mx,seal_sha256=sha(dest/'sealed.json'),metrics_sha256=sha(dest/'metrics.json'),report_sha256=sha(P/'propagation'/ds/'report.json'),worker=identity(),elapsed_wall_seconds=time.perf_counter()-start));print(ds,'official score complete',flush=True)
def audit(ds):
 check();torch.set_num_threads(1);start=time.perf_counter();dest=D/'propagation'/ds;sc=read(dest/'score_receipt.json');assert sc['seal_sha256']==sha(dest/'sealed.json')and sc['metrics_sha256']==sha(dest/'metrics.json');rr,n,_=derive(ds,True);old=read(dest/'metrics.json')['private_rows'];mx=0.
 for x,y in zip(rr,old):
  assert all(x[k]==y[k]for k in ['id','parent','duration_group'])
  for a in ARMS:
   for f,v in x['values'][a].items():
    w=y['values'][a][f];assert(v is None)==(w is None)
    if v is not None:mx=max(mx,abs(v-w));assert abs(v-w)<2e-10
 assert len(rr)==128 and summary(old)==read(P/'propagation'/ds/'report.json')
 ack=dict(status='whole128_independent_normalized_pixel_IoU_all_input_mask_arraySHA_means_parentCI_verified',queries=128,manual_checks=n,max_discrepancy=mx,input_audit_sha256=sha(dest/'input_audit.json'),score_receipt_sha256=sha(dest/'score_receipt.json'),metrics_sha256=sha(dest/'metrics.json'),report_sha256=sha(P/'propagation'/ds/'report.json'),worker=identity(),elapsed_wall_seconds=time.perf_counter()-start);atomic(dest/'reporting_receipt.json',ack);atomic(P/'propagation'/ds/'reporting_receipt.json',ack);print(ds,'independent score/stat audit passed',flush=True)
def export(ds):
 check();dest=D/'propagation'/ds;pub=P/'propagation'/ds;ack=read(dest/'reporting_receipt.json');assert ack['report_sha256']==sha(pub/'report.json');rr=read(dest/'metrics.json')['private_rows'];rep=read(pub/'report.json');assert summary(rr)==rep;cost=[]
 for i,row in enumerate(rows(ds),1):
  e=entry(ds,row['id']);cost.append(dict(query=f'Q{i:05}',active_query_wall_seconds=e['active_query_wall_seconds'],request_wall_including_checkpoint_seconds=e['request_wall_including_checkpoint_seconds'],checkpoint_IO_seconds=e['checkpoint_IO_seconds'],image_encoder_calls=e['actual_image_encoder_calls'],postprocessor_calls=e['actual_CUDA_postprocessor_calls'],native_frames=e['all_native_frames'],empty_native_masks=e['empty_frames'],empty_conditioning_masks=e['empty_conditioning_frames'],empty_input_masks=e['empty_input_masks'],worker=e['worker']))
 scope=dict(not_deployment_latency=True,nested_timers_not_additive=True,Source_ReferDINO_acquisition_inherited_nonzero=True,additional_SAM2_model_budget=True,paused_other_GPU_context_excluded=True)
 atomic(pub/'all_actual_query_costs.json',dict(rows=cost,scope=scope));atomic(pub/'cost_empty_summary.json',dict(queries=128,mean_active_query_seconds=float(np.mean([c['active_query_wall_seconds']for c in cost])),mean_request_including_checkpoints_seconds=float(np.mean([c['request_wall_including_checkpoint_seconds']for c in cost])),empty_native_masks=sum(c['empty_native_masks']for c in cost),native_masks=sum(c['native_frames']for c in cost),empty_input_masks=sum(c['empty_input_masks']for c in cost),empty_conditioning_outputs=sum(c['empty_conditioning_masks']for c in cost),queries_any_output_empty=sum(c['empty_native_masks']>0 for c in cost),scope=scope))
 pm={p:f'P{i:04}'for i,p in enumerate(sorted({r['parent']for r in rr}),1)}
 with(pub/'all_signed_scalars.tsv').open('w')as f:
  writer=csv.writer(f,delimiter='\t',lineterminator='\n');fields=['t','v','R03','R05','event_unobserved_IoU','event_observed_IoU'];writer.writerow(['query','parent','duration_group']+[a+'_'+k for a in ARMS for k in fields])
  for i,r in enumerate(rr,1):writer.writerow([f'Q{i:05}',pm[r['parent']],r['duration_group']]+[None if r['values'][a][k]is None else 100*r['values'][a][k]for a in ARMS for k in fields])
 with(pub/'all_signed_tails.tsv').open('w')as f:
  writer=csv.writer(f,delimiter='\t',lineterminator='\n');writer.writerow(['query','delta_v_Full_minus_SAM2_pp','harm_gt5pp','rescue_gt5pp'])
  for i,r in enumerate(rr,1):
   v=100*(r['values']['Full']['v']-r['values']['SAM2MatchedMask']['v']);writer.writerow([f'Q{i:05}',v,v< -5,v>5])
 text=[f'# {ds}: matched ReferDINO evidence, common Full interval','',f'Whole128/{rep["parents"]} parents independently sealed, official scored and manual-pixel/input/mask/SHA/means/CI audited.','', '|Arm|tIoU|vIoU|R.3|R.5|Conditional dense unobserved IoU|','|---|---:|---:|---:|---:|---:|']
 for a in ARMS:text.append('|'+a+'|'+'|'.join('null'if rep['means'][a][k]is None else f"{100*rep['means'][a][k]:.6f}"for k in ['t','v','R03','R05','event_unobserved_IoU'])+'|')
 text+=['','Eight original selected ReferDINO masks prompt official SAM2.1 Hiera-L, all native RGB frames, exact same Full event and same128 queries. No Source/expert/optimizer new calls. SAM2 native mask boxes at prompted frames too; no Source, expert or E output overwrite. Current soft E/expert anchor selection unchanged. The additional SAM2 encoder/model/precision/compute differ, so this is matched expert evidence, not equal models or same internal features.','', 'Conditional dense unobserved IoU uses only common predicted-event/GT-event annotated intersection excluding eight exact observed physical times. Empty support null, allqueries retained in primaryv. All positive/negative tails and short/long pointwise parentCIs retained. Checkpoint/provenance IO is inside experimental clock; not deployment latency.','',f"Full-SAM2 vIoU: {rep['paired']['all']['statistics']['delta_v_pp']}."]
 (pub/'ANALYSIS.md').write_text('\n'.join(text)+'\n');atomic(dest/'export_receipt.json',dict(status='whole128_signed_scalars_tails_cost_empty_analysis_exported_PENDING_ROOT',report_sha256=sha(pub/'report.json'),audit_sha256=sha(dest/'reporting_receipt.json'),worker=identity(),new_GT=0));print(ds,'signed/cost export complete',flush=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('task',choices=['bridge','input_audit','score','audit','export']);a.add_argument('--dataset',choices=['hc2','vidstg'],required=True);x=a.parse_args();globals()[x.task](x.dataset)
