"""Pure output conversion for official ReferFormer, no model construction."""
import hashlib
def selected_eight(outputs,input_shape):
 import torch
 logits=outputs['pred_logits'][0].detach();boxes=outputs['pred_boxes'][0].detach();masks=outputs['pred_masks'][0].detach()
 assert logits.ndim==3 and logits.shape[0]==8 and boxes.shape==(8,logits.shape[1],4)and masks.ndim==4 and masks.shape[:2]==logits.shape[:2]
 assert torch.isfinite(logits).all()and torch.isfinite(boxes).all()and torch.isfinite(masks).all();assert input_shape[:2]==[8,3]
 # Exact official inference_ytvos query selection, one index across all8.
 pred_scores=logits.sigmoid();pred_scores=pred_scores.mean(0);max_scores,_=pred_scores.max(-1);_,max_ind=max_scores.max(-1);j=int(max_ind);b=boxes[:,j].float().cpu();m=masks[:,j].float().cpu()
 assert bool((b[:,2:]>0).all())
 return dict(boxes=b,native_mask_logits=m,selected_query=j,native_query_scores=max_scores.float().cpu(),selected_frame_scores=logits[:,j].sigmoid().float().cpu(),input_shape=list(input_shape),nested_mask_all_false=True,native_same_track_all8=True,expert_model='ReferFormer-R50-RefYTVOS',expert_combination='one referring-video model, no extra detector/segmenter',GT_read=False)
def binary_mask(logits,height,width):
 import torch.nn.functional as F
 assert logits.ndim==3 and logits.shape[0]==8
 return(F.interpolate(logits[:,None].float(),size=(height,width),mode='bilinear',align_corners=False)[:,0].sigmoid()>.5)
