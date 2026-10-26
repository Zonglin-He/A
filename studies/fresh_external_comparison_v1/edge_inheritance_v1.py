"""No-forward proof that all16 prior paid EdgeTAM outputs fit the fresh cohort."""
from edge_common_v1 import *
def run():
 import torch,numpy as np
 import pilot_common_v1 as old
 import pilot_report_runtime_v2 as previous_report
 validate();release();torch.set_num_threads(1);parent.gt_guard();start=time.perf_counter();files={}
 for row in rows('hc2')[:16]:
  resources(start);e=old.committed('hc2',row['id']);previous_report.checked('hc2',row,e,raw=False)
  frames,pos,native,arrays,proof=payload('hc2',row)
  with np.load(OLD/e['path'])as z:assert all(np.array_equal(v,z[k])for k,v in arrays.items())
  op=e['input_binding'];ip=proof['actual_input_binding'];assert op['actual_native_RGB_sha256']==ip['actual_native_RGB_sha256']and op['actual_eight_RGB_sha256']==ip['actual_eight_RGB_sha256']and op['evidence_key']==ip['evidence_key']and op['original_teacher_payload_sha256']==proof['current_teacher']['payload_sha256']
  c=torch.load(OLD/e['support_path'],map_location='cpu',weights_only=False,mmap=True);assert np.array_equal(c['native_boxes'],native)and np.array_equal(c['positions8'],pos)
  files[row['id']]=dict(old_receipt_path=str(OLD/e['receipt_path']),old_receipt_sha256=e['receipt_sha256'],old_prediction_sha256=e['sha256'],old_support_sha256=e['support_sha256'],fresh_reference_arrays_and_same_eight_RGB_prompts_exact=True)
 rec=dict(status='passed_all16_existing_EdgeTAM_same_current_inputs_and_reference_arrays_support_exact',queries=16,files=files,worker=identity(),elapsed_wall_seconds=time.perf_counter()-start,new_model=0,new_tracking=0,new_Source=0,new_original_expert=0,new_GT=0,old_tracking_engineering_control_failure_costs_retained=True)
 atomic(N/'edge_inheritance_preflight_v1.json',rec);atomic(P/'inheritance_receipt_v1.json',{k:v for k,v in rec.items()if k!='files'})
if __name__=='__main__':run()
