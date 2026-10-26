"""Same actual output math, missing raw path redirected only to exact original-byte restore."""
from pilot_common_v1 import *
import pilot_report_v1 as original
import argparse

def raw_path(rec):
 p=D/rec['path']
 if p.exists():return p
 q=read(D/'first_raw_restore_v1/completion.json');assert q['status']=='one_exact_original_float_raw_SHA_restored_all_native_frames_prediction_exact'and q['all_original_raw_mask_bytes_SHA_exact']
 x=[x for x in q['restored']if x['original_path']==rec['path']];assert len(x)==1 and x[0]['sha256']==rec['sha256'];p=D/x[0]['restored_path'];assert sha(p)==rec['sha256'];return p

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
   rec=e['raw_mask_checkpoints'][i];assert sha(raw_path(rec))==rec['sha256'];v=torch.load(raw_path(rec),map_location='cpu',weights_only=False,mmap=True);assert v['frame_index']==i and v['input_binding']==proof and not v['GT_read'];assert torch.isfinite(v['scores']).all()and np.array_equal((v['scores'][0,0]>0).numpy(),mask)
 edge=a.keep_observed(np.stack(boxes),native,pos);reference['EdgeTAM']=edge;reference['EdgeTAM_interval']=reference['Ours_interval'].copy()
 with np.load(D/e['path'])as z:assert set(z.files)==set(reference)and all(np.array_equal(v,z[k])for k,v in reference.items())
 assert e['new_Source']==e['new_original_expert']==0 and e['actual_strict_postprocess_calls']>0 and e['all_raw_masks_checkpointed_before_validation']and e['protocol']['full_native_frame_coverage']and e['protocol']['observed_outputs_exact_input_boxes']
 assert all(np.array_equal(reference[a+'_interval'],reference['Ours_interval'])for a in ARMS)
 return dict(prediction_sha256=e['sha256'],support_sha256=e['support_sha256'],actual_native_RGB_and_eight_expert_context_exact=True,all_mask_boxes_and_observed_writeback_exact=True,all_reference_arrays_and_common_interval_exact=True,full_raw_masks_verified=raw)

def validate_bridge():
 validate();p=read(D/'root_raw_bridge_plan_v2.json');assert p['fingerprint']==digest({k:v for k,v in p.items()if k!='fingerprint'})
 for field in ['code_sha256','input_sha256']:
  for f,h in p[field].items():assert sha(R/f)==h,f
 return p

if __name__=='__main__':
 p=validate_bridge();a=argparse.ArgumentParser();a.add_argument('command',choices=['bridge','score','audit']);a.add_argument('--dataset',choices=['hc2','vidstg'],required=True);q=a.parse_args();original.checked=checked;getattr(original,q.command)(q.dataset)
 if q.command=='bridge':atomic(D/'pilot_v1'/q.dataset/'root_raw_restore_bridge_receipt_v2.json',dict(status='original_full_float_SHA_exact_root_bridge_passed',root_actual_bridge_sha256=sha(D/'pilot_v1'/q.dataset/'root_actual_bridge.json'),restore_completion_sha256=sha(D/'first_raw_restore_v1/completion.json'),bridge_execution_fingerprint=p['fingerprint'],all_numeric_statements_reverse_exact=True,one_additional_engineering_tracking_query_disclosed=True,new_Source=0,new_original_expert=0,new_GT=0))
