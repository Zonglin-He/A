"""Independent masks/input/official/manual score audits and signed exports."""
from common import *
import argparse,csv
OLD=R/'outputs/reviewer_key_controls_v1'
def independent_box(mask):
 h,w=mask.shape;yy=np.flatnonzero(mask.any(axis=1));xx=np.flatnonzero(mask.any(axis=0))
 if yy.size==0:return np.array([0.,0.,0.,0.])
 left,right,top,bottom=int(xx[0]),int(xx[-1])+1,int(yy[0]),int(yy[-1])+1
 return np.array([(left+right)/w/2,(top+bottom)/h/2,(right-left)/w,(bottom-top)/h],np.float64)
def entry(stage,ds,key,sealed=True):
 return fixed(D/stage/ds/'sealed.json')['files'][key]if sealed else read(D/stage/ds/'receipts'/(key+'.json'))
def parent_arrays(ds,key):
 e=fixed(OLD/'temporal'/ds/'sealed.json')['files'][key];assert sha(OLD/e['path'])==e['sha256']
 with np.load(OLD/e['path'])as z:return {k:z[k].copy()for k in ['Source','Source_interval','SourceTR_interval','frame_ids','positions8']},e['sha256']
def checked(stage,ds,row,e,raw=False):
 a,source=input_data(ds,row);reference,reference_sha=parent_arrays(ds,row['id']);assert all(np.array_equal(a[k],reference[k])for k in a);assert source['parent_temporal_prediction_sha256']==reference_sha
 for p,h in [('path','sha256'),('mask_path','mask_sha256')]:assert sha(D/e[p])==e[h]
 assert e['input_binding']['source_input_sha256']==source['sha256']and e['input_binding']['video_sha256']==row['metadata']['video_sha256'];assert not e['ReferDINO_input']and e['new_Source']==e['new_ReferDINO']==e['new_optimizer']==0 and e['raw_before_validation']and e['actual_CUDA_postprocessor_calls']>0 and e['observed_mask_boxes_not_overwritten']
 n=len(a['frame_ids']);h=row['metadata']['height'];w=row['metadata']['width'];boxes=[];empty=[]
 with np.load(D/e['mask_path'])as z:
  assert np.array_equal(z['native_shape'],[n,h,w])and np.array_equal(z['positions8'],a['positions8']);assert z['packed'].shape==(n,(h*w+7)//8)
  for i,x in enumerate(z['packed']):
   mask=np.unpackbits(x)[:h*w].reshape(h,w).astype(bool);boxes.append(independent_box(mask))
   if not mask.any():empty.append(i)
  if raw:
   frames,sourcea,binding=data(ds,row);assert binding==e['input_binding']and all(np.array_equal(sourcea[k],a[k])for k in a)
   from studies.review_strengthening_v1.sam2_input_adapter_v1 import lossless_rgb_loader
   images,_,_=lossless_rgb_loader(frames[a['positions8']],1024,True);assert hashlib.sha256(images.contiguous().numpy().tobytes()).hexdigest()==e['actual_preprocessing']['eight_preprocessed_sha256'];seen=[]
   for re in e['raw_float_checkpoints']:
    assert sha(D/re['path'])==re['sha256'];v=torch.load(D/re['path'],map_location='cpu',weights_only=False,mmap=True);i=int(v['frame_index']);assert v['input_binding']==binding and not v['GT_read']and torch.isfinite(v['scores']).all();mask=(v['scores'][0,0]>0).numpy();packed=np.unpackbits(z['packed'][i])[:h*w].reshape(h,w).astype(bool);assert np.array_equal(mask,packed);seen.append(i)
   assert sorted(seen)==list(range(n))
 boxes=np.stack(boxes);assert len(empty)==e['empty_frames']and sum(i in set(a['positions8'])for i in empty)==e['empty_conditioning_frames']
 with np.load(D/e['path'])as z:
  for k,v in a.items():assert np.array_equal(v,z[k])
  assert np.array_equal(boxes,z['SAM2Native'])and np.array_equal(boxes,z['SAM2TR']);assert np.array_equal(z['SAM2Native_interval'],a['Source_interval'])and np.array_equal(z['SAM2TR_interval'],a['SourceTR_interval'])
 return dict(prediction_sha256=e['sha256'],mask_sha256=e['mask_sha256'],all_native_masks_Source_only_input_time_and_boxes_exact=True,full_raw_scores_and_actual_RGB_preprocess_checked=raw)
def bridge(stage,ds):
 plan=check();assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';torch.set_num_threads(1);sys.addaudithook(__import__('stvg.runtime',fromlist=['guard']).guard);dest=D/stage/ds;wait=read(dest/'bridge_input.json');assert len(wait['keys'])==2 and not(dest/'actual_bridge.json').exists();lookup={r['id']:r for r in rows(stage,ds)};cases=[]
 for key in wait['keys']:cases.append(checked(stage,ds,lookup[key],entry(stage,ds,key,False),raw=True))
 assert not torch.cuda.is_initialized();atomic(dest/'actual_bridge.json',dict(status='passed_actual_two_new_SAM21_raw_scores_allmask_Source_only_RGB_official_preprocess_nativeboxes_times_independent_CPU_bridge',cases=cases,worker=identity(),plan_fingerprint=plan['fingerprint'],actual_new_GPU=0,new_Source=0,new_ReferDINO=0,new_GT=0));print(stage,ds,'actual independent two-query bridge passed',flush=True)
def input_audit(stage,ds):
 check();assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';torch.set_num_threads(1);dest=D/stage/ds;seal=read(dest/'sealed.json');assert seal['queries']==len(rows(stage,ds))and not seal['GT_read'];start=time.perf_counter();n=0
 for row in rows(stage,ds):
  e=entry(stage,ds,row['id']);assert sha(D/e['receipt_path'])==e['receipt_sha256'];checked(stage,ds,row,e);n+=1
 atomic(dest/'input_audit.json',dict(status='all_whole_domain_Source_only_inputs_SHA_binarymasks_independent_boxes_nativecoverage_times_exact',queries=n,seal_sha256=sha(dest/'sealed.json'),worker=identity(),new_GPU=0,new_Source=0,new_expert=0,new_GT=0,elapsed_wall_seconds=time.perf_counter()-start));print(stage,ds,'input/mask audit complete',flush=True)
def metric(boxes,ids,iv,pos,meta,g,ds,manual=False):
 from stvg.metrics import official,DenseTube,box_iou
 from stvg.metrics_base import xyxy
 gt={int(k):v for k,v in g['boxes'].items()};span=g['interval'];row=dict(input=meta,frame_ids=list(map(int,ids)))
 if manual:
  bb=np.asarray(boxes,np.float64);scale=np.array([meta['width'],meta['height']]);pix=np.concatenate([(bb[:,:2]-bb[:,2:]/2)*scale,(bb[:,:2]+bb[:,2:]/2)*scale],1)
  if ds=='hc2':pix=np.maximum(pix,0)
  ff=np.asarray(sorted(gt));pred=np.array([np.interp(ff,ids,pix[:,k])for k in range(4)]).T;truth=np.array([gt[int(f)]for f in ff]);inter=np.maximum(np.minimum(pred[:,2:],truth[:,2:])-np.maximum(pred[:,:2],truth[:,:2]),0).prod(1);den=np.maximum(pred[:,2:]-pred[:,:2],0).prod(1)+np.maximum(truth[:,2:]-truth[:,:2],0).prod(1)-inter;iou=np.divide(inter,den,out=np.zeros_like(inter),where=den>0);iou[(ff<ids[0])|(ff>ids[-1])]=0
  a,b=map(int,iv);lo,hi=span;over=max(0,min(b,hi)-max(a,lo));v=dict(v=float(iou[(ff>=max(a,lo))&(ff<min(b,hi))].sum()/max(max(b,hi)-min(a,lo),1)),t=over/(b-a+hi-lo-over),s=float(iou.mean()))
 else:
  v=official(boxes,row,gt,span,iv,ds);ref=DenseTube(boxes,row,gt,span,clip=ds=='hc2').score(iv);assert max(abs(v[k]-ref[k])for k in ['v','t','s'])<2e-10
 pix=xyxy(boxes,meta['width'],meta['height']);pix=np.maximum(pix,0)if ds=='hc2'else pix;observed=set(map(int,pos));ii=[i for i,f in enumerate(ids)if int(f)in gt and i not in observed];v['unobserved_actual_sIoU']=float(np.mean([box_iou(pix[i],gt[int(ids[i])])for i in ii]))if ii else None;v.update(R03=float(v['v']>=.3),R05=float(v['v']>=.5));return v
def derive(stage,ds,manual=False):
 seal=read(D/stage/ds/'sealed.json');assert read(D/stage/ds/'input_audit.json')['seal_sha256']==sha(D/stage/ds/'sealed.json');assert seal['queries']==len(rows(stage,ds))and not seal['GT_read']
 from studies.map_paper_completion_v1.common import labels
 truth=labels(ds);prior={r['id']:r for r in fixed(OLD/'temporal'/ds/'metrics.json')['private_rows']};out=[];checks=0
 pilot={r['id']:r for r in fixed(D/'pilot'/ds/'metrics.json')['private_rows']}if stage=='full'else{}
 for row in rows(stage,ds):
  key=row['id'];e=entry(stage,ds,key);assert sha(D/e['path'])==e['sha256'];g=truth[key];values={}
  for arm,name in [('Source','Source'),('SourceTR','SourceTR'),('Current','Ours')]:
   v=prior[key]['values'][name];values[arm]={k:v[k]for k in ['v','t','s','unobserved_actual_sIoU']};values[arm].update(R03=float(v['v']>=.3),R05=float(v['v']>=.5))
  with np.load(D/e['path'])as z:
   for arm in ['SAM2Native','SAM2TR']:
    if key in pilot:
     values[arm]=pilot[key]['values'][arm];continue
    values[arm]=metric(z[arm],z['frame_ids'],z[arm+'_interval'],z['positions8'],row['metadata'],g,ds,manual);checks+=1
  assert values['SAM2Native']['t']==values['Source']['t']and values['SAM2TR']['t']==values['SourceTR']['t'];out.append(dict(id=key,parent=str(g['parent']),duration_group='short_le5s'if(g['interval'][1]-g['interval'][0])/row['metadata']['fps']<=5 else'long_gt5s',values=values))
 return out,checks
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
def pair(rr,a,b):
 z=[]
 for r in rr:
  x,y=r['values'][a],r['values'][b];dt=100*(x['t']-y['t']);dv=100*(x['v']-y['v']);du=None if x['unobserved_actual_sIoU']is None or y['unobserved_actual_sIoU']is None else 100*(x['unobserved_actual_sIoU']-y['unobserved_actual_sIoU']);z.append(dict(parent=r['parent'],delta_t_pp=dt,delta_v_pp=dv,delta_unobserved_IoU_pp=du,improved_v_percent=100.*(dv>1e-10),harmed_v_percent=100.*(dv< -1e-10),severe_t_percent=100.*(dt< -5),severe_v_percent=100.*(dv< -5),severe_union_once_percent=100.*(dt< -5 or dv< -5),rescue_union_percent=100.*(dt>5 or dv>5)))
 return stats(z,list(z[0])[1:])if z else stats([],[])
def summary(rr):
 means={a:{f:float(np.mean([r['values'][a][f]for r in rr if r['values'][a][f]is not None]))if any(r['values'][a][f]is not None for r in rr)else None for f in rr[0]['values'][a]}for a in ARMS};groups={g:{a+'-'+b:pair([r for r in rr if g=='all'or r['duration_group']==g],a,b)for a,b in PAIRS}for g in ['all','short_le5s','long_gt5s']};return dict(queries=len(rr),parents=len({r['parent']for r in rr}),means=means,paired=groups,CI_scope='10000 parent clusters, query equal, seed20261022, pointwise, exposed cohorts, no target configuration selection',unobserved_scope='GT-supported native sampled unobserved frames over video, query equal; not conditional dense event metric')
def score(stage,ds):
 check(True);torch.set_num_threads(1);start=time.perf_counter();dest=D/stage/ds;rr,n=derive(stage,ds);rep=summary(rr);atomic(dest/'metrics.json',dict(private_rows=rr));atomic(P/stage/ds/'report.json',rep);atomic(dest/'score_receipt.json',dict(status='whole_domain_new_SAM2_native_and_SourceTR_intervals_officially_scored',queries=len(rr),new_official_checks=n,pilot_scores_reused=256 if stage=='full'else 0,seal_sha256=sha(dest/'sealed.json'),metrics_sha256=sha(dest/'metrics.json'),report_sha256=sha(P/stage/ds/'report.json'),worker=identity(),elapsed_wall_seconds=time.perf_counter()-start));print(stage,ds,'official score complete pending independent audit',flush=True)
def audit(stage,ds):
 check(True);torch.set_num_threads(1);start=time.perf_counter();dest=D/stage/ds;sc=read(dest/'score_receipt.json');assert sc['seal_sha256']==sha(dest/'sealed.json')and sc['metrics_sha256']==sha(dest/'metrics.json');rr,n=derive(stage,ds,True);old=read(dest/'metrics.json')['private_rows'];assert len(rr)==len(old)==len(rows(stage,ds));mx=0.
 for x,y in zip(rr,old):
  assert all(x[k]==y[k]for k in ['id','parent','duration_group'])
  for a in ARMS:
   for f,v in x['values'][a].items():
    w=y['values'][a][f];assert(v is None)==(w is None)
    if v is not None:mx=max(mx,abs(v-w));assert abs(v-w)<2e-10
 assert summary(old)==read(P/stage/ds/'report.json');ack=dict(status='whole_domain_independent_manual_scores_all_input_mask_array_SHA_means_and_parentCI_verified',queries=len(rr),new_manual_checks=n,max_scalar_discrepancy=mx,input_audit_sha256=sha(dest/'input_audit.json'),score_receipt_sha256=sha(dest/'score_receipt.json'),report_sha256=sha(P/stage/ds/'report.json'),metrics_sha256=sha(dest/'metrics.json'),worker=identity(),elapsed_wall_seconds=time.perf_counter()-start);atomic(dest/'reporting_receipt.json',ack);atomic(P/stage/ds/'reporting_receipt.json',ack);print(stage,ds,'whole independent audit passed',flush=True)
def export(stage,ds):
 check(True);dest=D/stage/ds;pub=P/stage/ds;ack=read(dest/'reporting_receipt.json');assert ack['report_sha256']==sha(pub/'report.json');rr=read(dest/'metrics.json')['private_rows'];rep=read(pub/'report.json');assert summary(rr)==rep;cost=[];empties=0;obs_empty=0;anyempty=0;maskframes=0
 for i,row in enumerate(rows(stage,ds),1):
  e=entry(stage,ds,row['id']);cost.append(dict(query=f'Q{i:05}',active_query_wall_seconds=e['active_query_wall_seconds'],request_wall_including_checkpoint_seconds=e['request_wall_including_checkpoint_seconds'],checkpoint_IO_seconds=e['checkpoint_IO_seconds'],image_encoder_calls=e['actual_image_encoder_calls'],postprocessor_calls=e['actual_CUDA_postprocessor_calls'],worker=e['worker'],reused_from_pilot=stage=='full'and row['id']in fixed(D/'pilot'/ds/'sealed.json')['files']));empties+=e['empty_frames'];obs_empty+=e['empty_conditioning_frames'];anyempty+=e['empty_frames']>0;maskframes+=e['all_native_frames']
 efficiency=dict(queries=len(cost),mean_experimental_active_query_wall_seconds=float(np.mean([r['active_query_wall_seconds']for r in cost])),mean_SAM_request_including_checkpoints_seconds=float(np.mean([r['request_wall_including_checkpoint_seconds']for r in cost])),mean_raw_checkpoint_seconds=float(np.mean([r['checkpoint_IO_seconds']for r in cost])),empty_native_masks=empties,empty_conditioning_masks=obs_empty,queries_with_any_empty=anyempty,total_native_masks=maskframes,one_external_model_not_equal_compute=True,not_deployment_latency=True,nested_clocks_not_additive=True,Source_inherited_acquisition_nonzero=True);atomic(pub/'efficiency_and_empty.json',efficiency);atomic(pub/'all_actual_query_costs.json',dict(rows=cost,scope=efficiency));pm={v:f'P{i:04}'for i,v in enumerate(sorted({r['parent']for r in rr}),1)}
 with(pub/'all_signed_scalars.tsv').open('w')as f:
  w=csv.writer(f,delimiter='\t',lineterminator='\n');fields=['v','t','R03','R05','unobserved_actual_sIoU'];w.writerow(['query','parent','duration_group']+[a+'_'+k for a in ARMS for k in fields])
  for i,r in enumerate(rr,1):w.writerow([f'Q{i:05}',pm[r['parent']],r['duration_group']]+[None if r['values'][a][k]is None else 100*r['values'][a][k]for a in ARMS for k in fields])
 with(pub/'all_signed_tails.tsv').open('w')as f:
  w=csv.writer(f,delimiter='\t',lineterminator='\n');w.writerow(['query','comparison','delta_t_pp','delta_v_pp','severe_t','severe_v','severe_union_once','rescue_union'])
  for i,r in enumerate(rr,1):
   for a,b in PAIRS:
    dt=100*(r['values'][a]['t']-r['values'][b]['t']);dv=100*(r['values'][a]['v']-r['values'][b]['v']);w.writerow([f'Q{i:05}',a+'-'+b,dt,dv,dt< -5,dv< -5,dt< -5 or dv< -5,dt>5 or dv>5])
 text=[f'# {stage} {ds}: Source-prompt SAM2.1 Hiera-L + Source-box TR','',f"Whole {len(rr)} queries/{rep['parents']}parents sealed, independent input/mask audit, official two-interval scores and separate manual-score/SHA/means/parentCI audit complete. No new Source/ReferDINO/optimizer. Source/SourceTR/Current scalar references reused exactly.",'','|Method|tIoU|vIoU|R.3|R.5|Native unobserved IoU|','|---|---:|---:|---:|---:|---:|']
 for a in ARMS:text.append('|'+a+'|'+'|'.join('null'if rep['means'][a][k]is None else f"{100*rep['means'][a][k]:.6f}"for k in ['t','v','R03','R05','unobserved_actual_sIoU'])+'|')
 text+=['','SAM2 receives only original Source boxes at the original eight physical positions and all native sampled RGB frames. All native mask boxes are used, including prompted frames, without Source writeback. SourceTR event is inherited from expert-free original Source-box evidence. No ReferDINO/E/Current time/DP enters SAM2. One external model per pipeline is controlled, not evidence identity, frame-forward count, model knowledge or precision. Official BF16 autocast/TF32,1024 resolution; no target tuning.','', 'All signed negative/severe/rescue and short-long paired CIs are retained. Experimental query clocks include checkpoint/hash/audit provenance and cannot be compared directly with matched deployment timings.','']
 for name,q in rep['paired']['all'].items():
  v=q['statistics']['delta_v_pp'];text.append(f"{name}: delta vIoU {v['mean']:+.6f} pp, pointwise95%CI {v['CI95']}.")
 (pub/'ANALYSIS.md').write_text('\n'.join(text)+'\n');atomic(dest/'export_receipt.json',dict(status='whole_domain_signed_results_cost_empty_and_analysis_exported',report_sha256=sha(pub/'report.json'),efficiency_sha256=sha(pub/'efficiency_and_empty.json'),audit_sha256=sha(dest/'reporting_receipt.json'),worker=identity(),new_GT=0));print(stage,ds,'all signed results/cost exported',flush=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('task',choices=['bridge','input_audit','score','audit','export']);a.add_argument('--stage',choices=['pilot','full'],required=True);a.add_argument('--dataset',choices=list(COUNTS),required=True);x=a.parse_args();globals()[x.task](x.stage,x.dataset)
