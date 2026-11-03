from common_v1 import *
import adapter_runtime_v2 as adapter
import argparse,importlib.util,csv

def compare_values(a,b):
 if torch.is_tensor(a):return torch.equal(a,b)
 if isinstance(a,np.ndarray):return np.array_equal(a,b)
 if isinstance(a,dict):return set(a)==set(b)and all(compare_values(v,b[k])for k,v in a.items())
 if isinstance(a,(list,tuple)):return len(a)==len(b)and all(compare_values(x,y)for x,y in zip(a,b))
 return a==b

def checked(expert,ds,row,e):
 folder=D/expert/ds
 if expert in REUSE:
  arrays,c,proof=cached_reuse_derive(expert,ds,row);assert e['input_binding']==proof and sha(D/e['path'])==e['sha256']and sha(D/e['receipt_path'])==e['receipt_sha256']and sha(D/e['compact_path'])==e['compact_sha256']
  cold=torch.load(D/e['compact_path'],map_location='cpu',weights_only=False,mmap=True);again=from_compact(cold)
  with np.load(D/e['path'])as z:
   assert set(arrays)==set(z.files)==set(again)
   for k,v in arrays.items():assert np.array_equal(v,z[k])and np.array_equal(v,again[k])
   assert np.array_equal(z['expert_boxes'],z['PureSource'][z['positions8']])
  return dict(receipt_sha256=e['receipt_sha256'],existing_Current_and_prior_full_input_array_audit_reused=True,new_pure_all_arrays_exact=True,new_Source=0,new_expert=0,new_GT=0)
 cap,raw,frames,current,refs,proof=read_inputs(ds,row);assert e['input_binding']==proof and not e['GT_read']
 assert sha(D/e['path'])==e['sha256']and sha(D/e['compact_path'])==e['compact_sha256']and sha(D/e['receipt_path'])==e['receipt_sha256']
 if expert in EXPERTS:
  for name in ['raw_expert_output','expert_payload']:assert sha(D/e[name]['path'])==e[name]['sha256']
  full=torch.load(D/e['raw_expert_output']['path'],map_location='cpu',weights_only=False,mmap=True);assert full['input_binding']==proof and full['expert']==expert and full['worker']==e['worker']
  processed=adapter.preprocess(expert,frames[cap['positions8']]);assert list(processed.shape)==full['input_shape']and list(frames.shape[1:3])==full['original_shape']and hashlib.sha256(processed.numpy().tobytes()).hexdigest()==full['processed_input_sha256']
  teacher=torch.load(D/e['expert_payload']['path'],map_location='cpu',weights_only=False,mmap=True);selected=adapter.select(expert,full['outputs'],full['input_shape'],full['original_shape'])
  for k,v in selected.items():assert compare_values(v,teacher[k]),k
  assert e['raw_expert_output']['before_selected_query_validation']and e['raw_expert_output']['before_ROI_or_geometry']and e['expert_payload']['before_both_inference_arms']
  assert e['official_CUDA_kernel_calls']>0
  if expert=='sgmg_videoswin_t':
   # Independent NumPy corner extraction after the official selected mask readout.
   import torch.nn.functional as F
   o=full['outputs'];ll=o['audit_all_native_query_logits'][0];j=int(ll.sigmoid().mean(0).max(-1).values.argmax());assert j==teacher['selected_query']and torch.equal(ll[:,j],o['pred_logits'][0,:,0]);mask=o['pred_masks'][0,:,0];binary=(F.interpolate(mask[None,:,:processed.shape[-2],:processed.shape[-1]],size=frames.shape[1:3],mode='bilinear',align_corners=False).sigmoid()>.5)[0].numpy();h,w=frames.shape[1:3];box=[];empty=0
   for b in binary:
    yy,xx=np.nonzero(b)
    if len(xx):left,right,top,bottom=xx.min(),xx.max()+1,yy.min(),yy.max()+1
    else:left,right,top,bottom=0,w,0,h;empty+=1
    box.append([(left+right)/(2*w),(top+bottom)/(2*h),(right-left)/w,(bottom-top)/h])
   assert np.array_equal(np.asarray(box,dtype=np.float32),teacher['boxes'].numpy())and empty==teacher['empty_box_conversion']
  else:
   assert expert=='onlinerefer_r50'and e['actual_model_steps']==8 and len(e['raw_steps'])==8
   o=full['outputs'];assert compare_values(o['initial_track'],o['steps'][0]['track_before'])and compare_values(o['final_track'],o['steps'][-1]['track_after']);bb=[]
   for i,(st,info)in enumerate(zip(o['steps'],e['raw_steps'])):
    assert sha(D/info['path'])==info['sha256'];p=torch.load(D/info['path'],map_location='cpu',weights_only=False,mmap=True);assert p['frame_index']==i and p['native_state_update_not_yet_applied']and p['input_binding']==proof and compare_values(p['outputs'],st['outputs'])and compare_values(p['track_before'],st['track_before'])
    logits=st['outputs']['pred_logits'][0];idx=int(logits.sigmoid().mean(0).max(-1).values.argmax());assert idx==teacher['selected_queries'][i];bb.append(st['outputs']['pred_boxes'][0,0,idx].numpy())
    assert compare_values(st['track_before'],o['steps'][i-1]['track_after'])if i else len(st['track_before']['pos_embedding'])==5
   assert np.array_equal(np.stack(bb),teacher['boxes'].numpy())and teacher['native_causal_state_carried_all8']and not teacher['native_same_track_all8']

 else:cap,raw,frames,teacher,refs,proof,origin=reuse_payload(expert,ds,row)
 arrays,c=build(cap,raw,teacher,ds);twice,cc=build(cap,raw,teacher,ds);cold=torch.load(D/e['compact_path'],map_location='cpu',weights_only=False,mmap=True);again=from_compact(cold)
 with np.load(D/e['path'])as z:
  assert set(arrays)==set(z.files)==set(twice)==set(again)
  for k,v in arrays.items():assert np.array_equal(v,twice[k])and np.array_equal(v,z[k])and np.array_equal(v,again[k]),k
  assert np.array_equal(z['expert_boxes'],z['PureSource'][z['positions8']])and np.array_equal(z['PureSource'],z['PureOwn'])and np.array_equal(z['PureOwn_interval'],z['Current_interval'])
 assert e['new_Source']==e['new_original_expert']==e['new_optimizer']==0
 return dict(receipt_sha256=e['receipt_sha256'],all_input_RGB_teacher_Source_grid_Epool_geometry_posterior_finalDP_arrays_cold_exact=True,new_Source=0,new_expert=0,new_GT=0)
def bridge(expert,ds):
 validate();guard();torch.set_num_threads(1);folder=D/expert/ds;waiting=folder/'bridge_waiting.json';assert not(folder/'root_actual_bridge.json').exists();w=read(waiting);assert len(w['files'])==2 and w['model_load_sha256']==sha(D/expert/'model_load.json');start=time.perf_counter();cases=[]
 for row,e in zip(rows(ds)[:2],w['files']):cases.append(checked(expert,ds,row,e))
 assert not torch.cuda.is_initialized();atomic(folder/'root_actual_bridge.json',dict(status='passed_actual_two_new_expert_full_input_selection_Epool_geometry_posterior_finalDP_cold_exact',waiting_sha256=sha(waiting),cases=cases,worker=identity(),elapsed_seconds=time.perf_counter()-start,new_Source=0,new_expert=0,new_model=0,new_GT=0));print(expert,ds,'root actual bridge passed',flush=True)
def input_audit(expert,ds):
 validate();guard();torch.set_num_threads(1);folder=D/expert/ds;seal=read(folder/'sealed.json');assert seal['queries']==len(seal['files'])==128 and not seal['GT_read'];start=time.perf_counter();cases=[]
 for row in rows(ds):cases.append(checked(expert,ds,row,seal['files'][row['id']]))
 atomic(folder/'input_audit.json',dict(status='all128_inputs_paidraw_selected8_masks_original_Source_Epool_posterior_finalDP_pure_cold_exact',queries=128,cases=cases,seal_sha256=sha(folder/'sealed.json'),worker=identity(),elapsed_wall_seconds=time.perf_counter()-start,new_expert=0,new_Source=0,new_GT=0));print(expert,ds,'all128 noGT input/math audit passed',flush=True)
def score(expert,ds):
 validate();torch.set_num_threads(1);folder=D/expert/ds;seal=read(folder/'sealed.json');audit=read(folder/'input_audit.json');assert audit['seal_sha256']==sha(folder/'sealed.json')and audit['queries']==128
 from full_score import metric
 truth=old.rc.selected.labels(ds);prior={r['id']:r for r in read(R/'outputs/fresh_external_comparison_v1/referformer_v4/small_v1'/ds/'metrics.json')['private_rows']};rr=[];checks=0;mx=0.
 for row in rows(ds):
  key=row['id'];e=seal['files'][key];g=truth[key];values={'Source':prior[key]['values']['Source']}
  with np.load(D/e['path'])as z:
   inp={**row,'metadata':{**row['metadata'],'frame_ids':z['frame_ids'].tolist()}}
   for arm in ['PureSource','PureOwn','Current']:
    if arm=='Current'and expert in REUSE:values[arm]=prior[key]['values']['CurrentOurs'if expert=='referdino'else'AlternativeOurs'];continue
    values[arm],err=metric(z[arm],inp,{int(k):b for k,b in g['boxes'].items()},g['interval'],z[arm+'_interval'].tolist(),z['positions8']);assert err<2e-10;mx=max(mx,err);checks+=1
  assert values['Source']['t']==values['PureSource']['t']and values['PureOwn']['t']==values['Current']['t']
  rr.append(dict(id=key,parent=str(g['parent']),duration_group='short_le5s'if(g['interval'][1]-g['interval'][0])/row['metadata']['fps']<=5 else'long_gt5s',values=values))
 assert len({r['parent']for r in rr})==128;out=P/expert/ds;report=old.rc.selected.suite().parent_report(rr,ARMS,PAIRS,out);old.rc.selected.suite().signed_tails(rr,PAIRS,out);atomic(folder/'metrics.json',dict(private_rows=rr));atomic(folder/'score_receipt.json',dict(status='whole128_independent_official_score',queries=128,parents=128,new_official_checks=checks,max_error=mx,metrics_sha256=sha(folder/'metrics.json'),input_audit_sha256=sha(folder/'input_audit.json'),seal_sha256=sha(folder/'sealed.json'),worker=identity()));print(expert,ds,'official score complete',checks,flush=True)
def audit(expert,ds):
 validate();guard();torch.set_num_threads(1);folder=D/expert/ds;sc=read(folder/'score_receipt.json');assert sc['metrics_sha256']==sha(folder/'metrics.json')and sc['input_audit_sha256']==sha(folder/'input_audit.json')and sc['seal_sha256']==sha(folder/'sealed.json');rr=read(folder/'metrics.json')['private_rows'];report=read(P/expert/ds/'report.json');checks=0
 for arm,m in report['means'].items():
  for k,v in m.items():
   vals=[r['values'][arm][k]for r in rr if r['values'][arm][k]is not None];assert(v is None and not vals)or abs(v-float(np.mean(vals)))<1e-12;checks+=1
 seal=read(folder/'sealed.json')
 for e in seal['files'].values():assert sha(D/e['path'])==e['sha256']and sha(D/e['receipt_path'])==e['receipt_sha256']and sha(D/e['compact_path'])==e['compact_sha256']
 rec=dict(status='all128_independent_input_coldarrays_fileSHA_parent_and_allmeans_verified',queries=128,parents=128,means_checks=checks,seal_sha256=sha(folder/'sealed.json'),score_sha256=sha(folder/'score_receipt.json'),metrics_sha256=sha(folder/'metrics.json'),report_sha256=sha(P/expert/ds/'report.json'),worker=identity(),new_Source=0,new_expert=0,new_GT=0);atomic(folder/'reporting_receipt.json',rec);atomic(P/expert/ds/'reporting_receipt.json',rec);print(expert,ds,'final audit passed',flush=True)
def export(expert,ds):
 validate();guard();folder=D/expert/ds;ack=read(folder/'reporting_receipt.json');assert ack['report_sha256']==sha(P/expert/ds/'report.json');rr=read(folder/'metrics.json')['private_rows'];rep=read(P/expert/ds/'report.json');tails={}
 for aa,bb in PAIRS:
  dt=np.array([100*(r['values'][aa]['t']-r['values'][bb]['t'])for r in rr]);dv=np.array([100*(r['values'][aa]['v']-r['values'][bb]['v'])for r in rr]);tails[aa+'-'+bb]=dict(severe_t=int((dt< -5).sum()),severe_v=int((dv< -5).sum()),severe_tv_union_once=int(((dt< -5)|(dv< -5)).sum()),rescue_tv_union_once=int(((dt>5)|(dv>5)).sum()),v_improve=int((dv>0).sum()),v_harm=int((dv<0).sum()),v_tie=int((dv==0).sum()))
 seal=read(folder/'sealed.json');calls=list(seal['files'].values());cost=dict(new_Source=0,new_original_expert=0,new_optimizer=0,new_expert=sum(x.get('new_expert_queries',0)for x in calls),empty_mask_to_full_frame_boxes=sum(x.get('empty_mask_to_full_frame_boxes',0)for x in calls),actual_model_steps=sum(x.get('actual_model_steps',0)for x in calls),Source_and_reused_expert_acquisition_inherited_nonzero=True,nested_times_not_additive=True,cached_pipeline_not_deployment=True)
 if expert in EXPERTS:
  cost['model_load_once_per_worker']=read(D/expert/'model_load.json');cost['nested_segment_sums']={k:sum(x['cost'][k]for x in calls)for k in calls[0]['cost']};cost['peak_allocated_GPU_bytes']=max(x['peak_allocated_GPU_bytes']for x in calls)
 else:cost['new_model_or_GPU']=0;cost['cached_derive_wall_sum_seconds']=sum(x['cached_derive_wall_seconds']for x in calls)
 atomic(P/expert/ds/'tails.json',tails);atomic(P/expert/ds/'actual_cost.json',cost)
 text=f'# Expert subset: {expert}, {ds}\n\n128queries/128distinctparents, same original Source and eight physical expert frames; no target tuning/new Source or original ReferDINO. PureSource uses Source native event; PureOwn and Current share the expert-specific recomputed event. Pure is raw-eight center/logsize + nearest tails, no shared E tails. Current is unchanged soft-anchor implementation, not a hard-anchor fix.\n\n|Arm|tIoU %|vIoU %|R.3 %|R.5 %|Unobserved IoU %|\n|---|---:|---:|---:|---:|---:|\n'
 for arm,mm in rep['means'].items():text+='|'+arm+'|'+'|'.join(f'{100*mm[f]:.6f}'for f in ['t','v','vIoU_ge_03','vIoU_ge_05','unobserved_actual_sIoU'])+'|\n'
 text+='\nAll paired pointwise parent10000CIs, signed positives/negatives, severe/rescue union once and short<=5s/long groups retained. No claim of expert independence, equal compute/knowledge, unseen confirmation or familywise significance. Source/expert historical acquisition remains nonzero; request/forward/checkpoint/finish/query clocks nest, not uncached deployment. OnlineRefer uses eight native causal steps with original query/reference-state carry and direct boxhead. SgMg uses one native jointly selected refined-mask clip and an explicit halfopen mask-to-box adapter; empty masks become disclosed full-frame neutral boxes. Both use FP32/noAMP/noTF32; SgMg official example AMP is not enabled here. No extra frame, hidden Source fill, detector or optimizer. Existing R50/R101 remain same family.\n'
 (P/expert/ds/'ANALYSIS.md').write_text(text);atomic(folder/'root_export_receipt.json',dict(status='root_export_ready_pending_root_interpretation_publication',audit_sha256=sha(folder/'reporting_receipt.json'),analysis_sha256=sha(P/expert/ds/'ANALYSIS.md'),cost_sha256=sha(P/expert/ds/'actual_cost.json'),code_sha256=sha(Path(__file__)),new_Source=0,new_expert=0,new_GT=0));print(expert,ds,'all signed/cost export complete',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['bridge','input_audit','score','audit','export']);p.add_argument('--expert',required=True,choices=EXPERTS+REUSE);p.add_argument('--dataset',required=True,choices=['hc2','vidstg']);q=p.parse_args();globals()[q.stage](q.expert,q.dataset)
