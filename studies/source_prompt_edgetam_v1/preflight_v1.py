"""Four real Source-only prompt/readout contracts, no external model construction."""
from common import *
import numpy as np
def run():
 import torch
 validate();parent.gt_guard();torch.set_num_threads(1);start=time.perf_counter();checks=[]
 for ds in ['hc2','vidstg']:
  for row in rows(ds)[:2]:
   cap,origin=parent.selected.suite().load_parent(ds,row);allowed={k:cap[k]for k in ['Source','native_logits','records','frame_ids','positions8']};a=source_fields(allowed);b=source_fields(allowed);assert all(np.array_equal(x,y)for x,y in zip(a,b));frames,pos,prompts,arrays,proof=payload(ds,row);assert np.array_equal(a[0],pos)and np.array_equal(a[1],arrays['Source_interval'])and np.array_equal(a[2],prompts)and np.array_equal(prompts,arrays['Source'][pos]);assert not proof['ReferDINO_input_used']and not proof['current_method_event_used'];assert len(set(pos))==8 and frames.dtype==np.uint8
   checks.append(dict(dataset=ds,Source_capture_sha256=origin['capture_sha256'],Source_only_field_contract=True,Source_native_peroffset_MAP_hull_exact=True,Source_selected_eight_positions_exact=True,prompt_SHA=proof['Source_box_prompts_sha256'],native_RGB_SHA=proof['actual_native_RGB_sha256'],frames=len(frames),first_stage_external_models=0,no_teacher_mask_or_corrected_boxes_or_current_posterior_read=True))
 assert not torch.cuda.is_initialized();atomic(D/'preflight_v1.json',dict(status='passed_four_actual_Source_only_prompts_native_event_and_RGB_repeat_exact',cases=checks,new_Source=0,new_expert=0,new_model=0,new_GT_scoring=0,worker=identity(),elapsed_wall_seconds=time.perf_counter()-start));atomic(P/'preflight_receipt_v1.json',dict(status='passed_four_actual_Source_only_prompts_native_event_and_RGB_repeat_exact',cases=checks,new_model=0,new_tracking=0,new_GT_scoring=0))
if __name__=='__main__':run()
