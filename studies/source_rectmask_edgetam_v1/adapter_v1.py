"""Only prompt API changes: Source boxes become native binary rectangles."""
from contextlib import contextmanager
import numpy as np,torch
from studies.review_strengthening_v1.sam2_input_adapter_v1 import lossless_rgb_loader,mask_box
from studies.edgetam_comparison_v1.adapter import keep_observed,scoped_rgb_loader

def rectangular_masks(native,height,width):
 b=np.asarray(native,dtype=np.float64);assert b.shape==(8,4)and np.isfinite(b).all()and(b[:,2:]>=0).all()
 lo=b[:,:2]-b[:,2:]/2;hi=b[:,:2]+b[:,2:]/2;xx=(np.arange(width)+.5)/width;yy=(np.arange(height)+.5)/height
 return ((xx[None,None,:]>=lo[:,0,None,None])&(xx[None,None,:]<hi[:,0,None,None])&(yy[None,:,None]>=lo[:,1,None,None])&(yy[None,:,None]<hi[:,1,None,None]))

def propagate_boxes(predictor,module,frames,positions,native,checkpoint_scores):
 positions=[int(p)for p in positions];assert len(set(positions))==8;masks=rectangular_masks(native,*frames.shape[1:3])
 with scoped_rgb_loader(module):state=predictor.init_state(frames,offload_video_to_cpu=True,offload_state_to_cpu=False,async_loading_frames=False)
 for j in sorted(range(8),key=lambda j:positions[j]):predictor.add_new_mask(state,frame_idx=positions[j],obj_id=1,mask=masks[j])
 assert set(state['temp_output_dict_per_obj'][0]['cond_frame_outputs'])==set(positions)
 boxes={};empty=[]
 for i,obj,scores in predictor.propagate_in_video(state,start_frame_idx=0,reverse=False):
  checkpoint_scores(i,scores)
  assert obj==[1]and scores.shape==(1,1,*frames.shape[1:3])and i not in boxes and torch.isfinite(scores).all()
  mask=(scores[0,0]>0).detach().cpu().numpy();boxes[i]=mask_box(mask)
  if not mask.any():empty.append(i)
 assert set(boxes)==set(range(len(frames)));raw=np.stack([boxes[i]for i in range(len(frames))])
 return keep_observed(raw,native,positions),dict(full_native_frame_coverage=True,all_eight_rectangular_mask_prompts=True,no_box_prompt_API_used=True,one_object=True,forward_start=0,reverse=False,observed_outputs_exact_input_boxes=True,empty_prompt_positions=[positions[j]for j,m in enumerate(masks)if not m.any()],rectangular_mask_pixels=[int(m.sum())for m in masks],empty_propagated_frames=empty,no_Source_fallback=True,no_our_DP=True,raw_mask_boxes=raw)
