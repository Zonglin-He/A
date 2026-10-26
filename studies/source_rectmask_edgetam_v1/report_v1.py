"""Independent full-mask Source-only-prompt output checks, no GT in bridge."""
from mask_common import *
import argparse

def checked(ds,row,e,raw=False):
 import torch,numpy as np,importlib.util
 spec=importlib.util.spec_from_file_location('_EdgeTAM_fixed_output_rules',HERE/'adapter_v1.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
 frames,pos,native,reference,proof=payload(ds,row);assert e['input_binding']==proof
 for path,h in [('path','sha256'),('support_path','support_sha256')]:assert sha(D/e[path])==e[h]
 c=torch.load(D/e['support_path'],map_location='cpu',weights_only=False,mmap=True);N,H,W=c['native_shape'];assert[N,H,W]==list(frames.shape[:3])and not c['GT_read']and np.array_equal(c['positions8'],pos)and np.array_equal(c['native_boxes'],native)
 assert c['packed_masks'].shape==(N,(H*W+7)//8);boxes=[]
 for i,p in enumerate(c['packed_masks']):
  mask=np.unpackbits(p)[:H*W].reshape(H,W).astype(bool);boxes.append(a.mask_box(mask))
  if raw:
   rec=e['raw_mask_checkpoints'][i];assert sha(D/rec['path'])==rec['sha256'];v=torch.load(D/rec['path'],map_location='cpu',weights_only=False,mmap=True);assert v['frame_index']==i and v['input_binding']==proof and not v['GT_read'];assert torch.isfinite(v['scores']).all()and np.array_equal((v['scores'][0,0]>0).numpy(),mask)
 edge=a.keep_observed(np.stack(boxes),native,pos);reference['EdgeTAM']=edge;reference['EdgeTAM_interval']=reference['Source_interval'].copy()
 with np.load(D/e['path'])as z:assert set(z.files)==set(reference)and all(np.array_equal(v,z[k])for k,v in reference.items())
 assert e['protocol']['all_eight_rectangular_mask_prompts']and e['protocol']['no_box_prompt_API_used']
 assert e['new_Source']==e['new_original_expert']==0 and e['actual_strict_postprocess_calls']>0 and e['all_raw_masks_checkpointed_before_validation']and e['protocol']['full_native_frame_coverage']and e['protocol']['observed_outputs_exact_input_boxes']
 assert all(np.array_equal(reference[a+'_interval'],reference['Source_interval'])for a in ARMS)
 return dict(prediction_sha256=e['sha256'],support_sha256=e['support_sha256'],actual_native_RGB_and_Source_prompt_boxes_exact=True,all_mask_boxes_and_observed_writeback_exact=True,all_reference_arrays_and_common_interval_exact=True,full_raw_masks_verified=raw)

def bridge(ds):
 import torch
 torch.set_num_threads(1);validate();parent.gt_guard();folder=D/'pilot_v1'/ds;wait=folder/'bridge_waiting.json';w=read(wait);assert w['status']=='waiting_root_first_two_actual_EdgeTAM_queries'and len(w['files'])==2;assert not(folder/'root_actual_bridge.json').exists();load=read(D/'pilot_v1/model_load_v1.json');assert load['load']['all_official_state_tensors_exact']and load['actual_CUDA_kernel_contract_passed'];start=time.perf_counter()
 proof=[checked(ds,next(r for r in rows(ds)if r['id']==Path(e['path']).stem),e,raw=True)for e in w['files']];assert not torch.cuda.is_initialized();atomic(folder/'root_actual_bridge.json',dict(status='passed_first_two_actual_EdgeTAM_queries',waiting_sha256=sha(wait),cases=proof,model_load_sha256=sha(D/'pilot_v1/model_load_v1.json'),worker=identity(),elapsed_wall_seconds=time.perf_counter()-start,new_model=0,new_Source=0,new_original_expert=0,new_GT_scoring=0))

def score(ds):
 import torch,numpy as np
 from full_score import metric
 torch.set_num_threads(1);validate();folder=D/'pilot_v1'/ds;seal=read(folder/'sealed.json');assert seal['queries']==128 and not seal['GT_read'];truth=parent.selected.labels(ds);rr=[];checks=0;maxerr=0.;start=time.perf_counter();n=R/'outputs/fresh_external_comparison_v1';audit=read(n/ds/'current_audit_receipt_v1.json');assert audit['queries']==128
 for row in rows(ds):
  e=seal['files'][row['id']];assert sha(D/e['path'])==e['sha256'];current=read(n/ds/'sealed.json')['files'][row['id']];assert sha(n/current['path'])==current['sha256'];values={};g=truth[row['id']]
  with np.load(D/e['path'])as z,np.load(n/current['path'])as c:
   arrays={a:(z[a],z[a+'_interval'])for a in ['Source','EdgeTAM']};old=R/'outputs/source_prompt_edgetam_v1';oe=read(old/'pilot_v1'/ds/'sealed.json')['files'][row['id']];assert sha(old/oe['path'])==oe['sha256'];baseline=np.load(old/oe['path']);assert np.array_equal(baseline['Source'],z['Source'])and np.array_equal(baseline['Source_interval'],z['Source_interval']);arrays['BoxPromptEdgeTAM']=(baseline['EdgeTAM'],baseline['EdgeTAM_interval']);arrays.update(CurrentSimple=(c['Simple'],c['Source_interval']),CurrentOurs=(c['Ours'],c['Ours_interval']));assert np.array_equal(z['Source_interval'],c['Source_interval'])and np.array_equal(z['Source'],c['Source']);inp={**row,'metadata':{**row['metadata'],'frame_ids':z['frame_ids'].tolist()}}
   for a,(boxes,iv)in arrays.items():values[a],err=metric(boxes,inp,{int(k):v for k,v in g['boxes'].items()},g['interval'],iv.tolist(),z['positions8']);assert err<2e-10;maxerr=max(maxerr,err);checks+=1
  assert values['Source']['t']==values['EdgeTAM']['t']==values['CurrentSimple']['t']
  rr.append(dict(id=row['id'],parent=str(g['parent']),duration_group='short_le5s'if(g['interval'][1]-g['interval'][0])/row['metadata']['fps']<=5 else'long_gt5s',values=values))
 arms=['Source','CurrentSimple','CurrentOurs','BoxPromptEdgeTAM','EdgeTAM'];pairs=[('EdgeTAM','BoxPromptEdgeTAM'),('EdgeTAM','Source'),('CurrentOurs','EdgeTAM'),('CurrentOurs','CurrentSimple')];out=P/ds;rep=parent.selected.suite().parent_report(rr,arms,pairs,out);rep['semantics']='Single external expert per branch. EdgeTAM receives Source-derived rectangular masks and Source native time only; BoxPromptEdgeTAM is the independently audited same-cohort box-prompt baseline. Only prompt representation changes between these two EdgeTAM arms. CurrentOurs uses ReferDINO once and its own time. Distinct interfaces/compute, not isolated equal-compute propagation attribution.';atomic(out/'report.json',rep);parent.selected.suite().signed_tails(rr,pairs,out);atomic(folder/'metrics.json',dict(private_rows=rr,report=rep));rec=dict(status='scored_whole128_single_expert_branches',queries=128,parents=128,checks=checks,max_error=maxerr,seal_sha256=sha(folder/'sealed.json'),worker=identity(),elapsed_wall_seconds=time.perf_counter()-start);atomic(folder/'score_receipt.json',rec);atomic(out/'score_receipt.json',rec)

def audit(ds):
 import torch,numpy as np
 torch.set_num_threads(1);validate();parent.gt_guard();folder=D/'pilot_v1'/ds;seal=read(folder/'sealed.json');score=read(folder/'score_receipt.json');assert score['seal_sha256']==sha(folder/'sealed.json');start=time.perf_counter()
 for row in rows(ds):
  e=seal['files'][row['id']];assert sha(D/e['receipt_path'])==e['receipt_sha256'];checked(ds,row,e)
 rr=read(folder/'metrics.json')['private_rows'];rep=read(P/ds/'report.json');assert len(rr)==128 and rep['parents']==128
 for a,m in rep['means'].items():
  for k,v in m.items():
   z=[x['values'][a][k]for x in rr if x['values'][a][k]is not None];assert(v is None and not z)or abs(v-float(np.mean(z)))<1e-12
 rec=dict(status='verified_whole128_Source_prompt_only_all_mask_input_arrays_and_means',queries=128,parents=128,all_mask_boxes_and_reference_arrays_cold_exact=True,all_means_recomputed=True,score_sha256=sha(folder/'score_receipt.json'),metrics_sha256=sha(folder/'metrics.json'),worker=identity(),elapsed_wall_seconds=time.perf_counter()-start);atomic(folder/'reporting_receipt.json',rec);atomic(P/ds/'reporting_receipt.json',rec)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('command',choices=['bridge','score','audit']);p.add_argument('--dataset',choices=['hc2','vidstg'],required=True);a=p.parse_args();globals()[a.command](a.dataset)
