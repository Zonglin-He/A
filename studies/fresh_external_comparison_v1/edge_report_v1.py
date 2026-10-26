"""Independent actual-output bridge, fixed-cohort official scorer and cold audit."""
from edge_common_v1 import *
import argparse

def checked(ds,row,e,raw=False):
 import torch,numpy as np,importlib.util
 spec=importlib.util.spec_from_file_location('_EdgeTAM_fixed_output_rules',S/'adapter.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
 frames,pos,native,reference,proof=payload(ds,row);assert e['input_binding']==proof
 for path,h in [('path','sha256'),('support_path','support_sha256')]:assert sha(D/e[path])==e[h]
 c=torch.load(D/e['support_path'],map_location='cpu',weights_only=False,mmap=True);N,H,W=c['native_shape'];assert[N,H,W]==list(frames.shape[:3])and not c['GT_read']and np.array_equal(c['positions8'],pos)and np.array_equal(c['native_boxes'],native)
 assert c['packed_masks'].shape==(N,(H*W+7)//8);boxes=[]
 for i,p in enumerate(c['packed_masks']):
  mask=np.unpackbits(p)[:H*W].reshape(H,W).astype(bool);boxes.append(a.mask_box(mask))
  if raw:
   rec=e['raw_mask_checkpoints'][i];assert sha(D/rec['path'])==rec['sha256'];v=torch.load(D/rec['path'],map_location='cpu',weights_only=False,mmap=True);assert v['frame_index']==i and v['input_binding']==proof and not v['GT_read'];assert torch.isfinite(v['scores']).all()and np.array_equal((v['scores'][0,0]>0).numpy(),mask)
 edge=a.keep_observed(np.stack(boxes),native,pos);reference['EdgeTAM']=edge;reference['EdgeTAM_interval']=reference['Ours_interval'].copy()
 with np.load(D/e['path'])as z:assert set(z.files)==set(reference)and all(np.array_equal(v,z[k])for k,v in reference.items())
 assert e['new_Source']==e['new_original_expert']==0 and e['actual_strict_postprocess_calls']>0 and e['all_raw_masks_checkpointed_before_validation']and e['protocol']['full_native_frame_coverage']and e['protocol']['observed_outputs_exact_input_boxes']
 assert all(np.array_equal(reference[a+'_interval'],reference['Ours_interval'])for a in ARMS)
 return dict(prediction_sha256=e['sha256'],support_sha256=e['support_sha256'],actual_native_RGB_and_eight_expert_context_exact=True,all_mask_boxes_and_observed_writeback_exact=True,all_reference_arrays_and_common_interval_exact=True,full_raw_masks_verified=raw)

def bridge(ds):
 import torch
 torch.set_num_threads(1);validate();parent.gt_guard();folder=D/'pilot_v1'/ds;wait=folder/'bridge_waiting.json';w=read(wait);assert w['status']=='waiting_root_first_two_actual_EdgeTAM_queries'and len(w['files'])==2;assert not(folder/'root_actual_bridge.json').exists();load=read(D/'pilot_v1/model_load_v1.json');assert load['load']['all_official_state_tensors_exact']and load['actual_CUDA_kernel_contract_passed'];start=time.perf_counter()
 proof=[checked(ds,next(r for r in rows(ds)if r['id']==Path(e['path']).stem),e,raw=True)for e in w['files']];assert not torch.cuda.is_initialized();atomic(folder/'root_actual_bridge.json',dict(status='passed_first_two_actual_EdgeTAM_queries',waiting_sha256=sha(wait),cases=proof,model_load_sha256=sha(D/'pilot_v1/model_load_v1.json'),worker=identity(),elapsed_wall_seconds=time.perf_counter()-start,new_model=0,new_Source=0,new_original_expert=0,new_GT_scoring=0))

def score(ds):
 import torch,numpy as np
 from full_score import metric
 torch.set_num_threads(1);validate();folder=D/'pilot_v1'/ds;seal=read(folder/'sealed.json');assert seal['queries']==128 and not seal['GT_read'];assert read(folder/'root_actual_bridge.json')['status']=='passed_first_two_actual_EdgeTAM_queries';truth=parent.selected.labels(ds);rr=[];checks=0;maxerr=0.;start=time.perf_counter()
 for row in rows(ds):
  e=committed(ds,row['id']);assert e==seal['files'][row['id']];g=truth[row['id']];values={}
  with np.load(D/e['path'])as z:
   inp={**row,'metadata':{**row['metadata'],'frame_ids':z['frame_ids'].tolist()}}
   for a in ARMS:values[a],err=metric(z[a],inp,{int(k):v for k,v in g['boxes'].items()},g['interval'],z[a+'_interval'].tolist(),z['positions8']);assert err<2e-10;checks+=1;maxerr=max(maxerr,err)
  assert len({values[a]['t']for a in ARMS})==1
  rr.append(dict(id=row['id'],parent=str(g['parent']),duration_group='short_le5s'if(g['interval'][1]-g['interval'][0])/row['metadata']['fps']<=5 else'long_gt5s',values=values))
 # Alias only within the reporting helper; never inject a native Source time.
 report_rows=[{**r,'values':{**r['values'],'Source':r['values']['SourceMatchedTime']}}for r in rr];out=P/'pilot_v1'/ds;report=parent.selected.suite().parent_report(report_rows,ARMS,PAIRS,out);report['severe_vs_SourceMatchedTime']=report.pop('severe_vs_Source');report['baseline_semantics']='All four rows share the current Ours q1 event; only spatial trajectories compared, not native Source time.';atomic(out/'report.json',report);parent.selected.suite().signed_tails(rr,PAIRS,out);atomic(folder/'metrics.json',dict(private_rows=rr,report=report));rec=dict(status='scored_fixed128_after_whole_seal',queries=128,parents=report['parents'],checks=checks,max_error=maxerr,seal_sha256=sha(folder/'sealed.json'),worker=identity(),elapsed_wall_seconds=time.perf_counter()-start,whole_domain_small_unit_sealed_before_GT=True);atomic(folder/'score_receipt.json',rec);atomic(out/'score_receipt.json',rec)

def audit(ds):
 import torch,numpy as np
 torch.set_num_threads(1);validate();parent.gt_guard();folder=D/'pilot_v1'/ds;seal=read(folder/'sealed.json');sp=folder/'score_receipt.json';assert read(sp)['seal_sha256']==sha(folder/'sealed.json');start=time.perf_counter();proof=[]
 for row in rows(ds):
  e=committed(ds,row['id']);assert e==seal['files'][row['id']];proof.append(checked(ds,row,e))
 rr=read(folder/'metrics.json')['private_rows'];report=read(P/'pilot_v1'/ds/'report.json');assert len(rr)==128 and report['parents']==128
 for a,means in report['means'].items():
  for k,v in means.items():
   x=[r['values'][a][k]for r in rr if r['values'][a][k]is not None];assert(v is None and not x)or abs(v-float(np.mean(x)))<1e-12
 rec=dict(status='verified_fixed128_all_mask_support_parent_teacher_predictions_and_means',queries=128,parents=128,prediction_receipt_SHA_checks=256,all_mask_support_and_reference_arrays_cold_exact=True,all_means_recomputed=True,score_sha256=sha(sp),seal_sha256=sha(folder/'sealed.json'),metrics_sha256=sha(folder/'metrics.json'),worker=identity(),elapsed_wall_seconds=time.perf_counter()-start,new_Source=0,new_original_expert=0,new_model=0,new_GT_scoring=0,full12205_complete=False);atomic(folder/'reporting_receipt.json',rec);atomic(P/'pilot_v1'/ds/'reporting_receipt.json',rec)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('command',choices=['bridge','score','audit']);a.add_argument('--dataset',choices=['hc2','vidstg'],required=True);q=a.parse_args();globals()[q.command](q.dataset)
