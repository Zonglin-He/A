from mask_common import *
def run():
 import torch,numpy as np
 from adapter_v1 import rectangular_masks
 torch.set_num_threads(1);validate();parent.gt_guard();release();start=time.perf_counter();checks=[]
 for ds in ['hc2','vidstg']:
  for row in rows(ds)[:2]:
   frames,pos,native,arrays,proof=payload(ds,row);a=rectangular_masks(native,*frames.shape[1:3]);b=rectangular_masks(native,*frames.shape[1:3]);assert a.dtype==np.bool_ and a.shape==(8,*frames.shape[1:3])and np.array_equal(a,b);assert not proof['ReferDINO_input_used']and not proof['current_method_event_used'];old=R/'outputs/source_prompt_edgetam_v1';e=read(old/'pilot_v1'/ds/'sealed.json')['files'][row['id']];assert sha(old/e['path'])==e['sha256'];assert proof['Source_box_prompts_sha256']==e['input_binding']['Source_box_prompts_sha256']and proof['actual_native_RGB_sha256']==e['input_binding']['actual_native_RGB_sha256'];checks.append(dict(dataset=ds,actual_RGB_Source_native_time_and_eight_boxes_exact=True,rectangle_mask_SHA256=proof['Source_rectangle_mask_sha256'],pixel_counts=proof['rectangle_mask_pixels'],old_box_prompt_baseline_exact=True))
 # Synthetic explicit halfopen/pixel-center and fully off-image empty contracts.
 z=np.tile([.5,.5,1.,1.],(8,1));assert rectangular_masks(z,4,4).all();z[:,:2]=3;assert not rectangular_masks(z,4,4).any()
 assert not torch.cuda.is_initialized();rec=dict(status='passed_four_actual_Source_rectangle_mask_RGB_event_box_baseline_contracts',cases=checks,worker=identity(),elapsed_wall_seconds=time.perf_counter()-start,new_model=0,new_tracking=0,new_Source=0,new_expert=0,new_GT_scoring=0);atomic(D/'preflight_v1.json',rec);atomic(P/'preflight_v1.json',rec)
if __name__=='__main__':run()
