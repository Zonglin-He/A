"""R40 cached bbox extent association; no model inference or task labels."""
import sys,json,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'crossplatform/round39')]
import common39 as previous
from common39 import rows,prepared,arrays,read,dump,sha,seal,verify,load_npz
from association import rect_iou
BASE=ROOT/'crossplatform/round40';EXP=ROOT/'research/round40/experiment';PREVIOUS=ROOT/'crossplatform/round39/inference208_attempt002'
ARMS=previous.ARMS+['bbox_IoU_fullmap','bbox_IoU_oldsubset']

def check_plan():
 p=read(EXP/'PLAN_LOCK.json');q=read(EXP/'IMPLEMENTATION_LOCK.json');assert q['plan_sha256']==sha(EXP/'PLAN_LOCK.json')
 for f,h in dict(p['input_sha256'],**q['program_sha256']).items():assert sha(ROOT/f)==h,f
 return p

def iou_assignment(result,case,z):
 p=result['parsed'];grid=result['image_grid_thw'][0];W,H=case['image_size_wh'];Wp,Hp=int(grid[2])*14,int(grid[1])*14;assert result['model_resized_wh']==[Wp,Hp]
 detail={'status':p['status'],'mask_index':None,'bbox_resized_unchanged':p['value'],'bbox_original_xyxy':None,'mask_IoU':None,'top_IoU':None,'top2_margin':None,'exact_max_tie_n':None,'all_masks_including_unmappable':True}
 if p['status']!='parsed':return detail
 x1,y1,x2,y2=p['value'];assert 0<=x1<x2<=Wp and 0<=y1<y2<=Hp
 box=[x1*W/Wp,y1*H/Hp,x2*W/Wp,y2*H/Hp];detail['bbox_original_xyxy']=box
 # Fail the whole association for an incomplete inventory, never silently omit
 # an unknown detector or a mask that happens to lack a downstream 3D group.
 rects=z.get('mask_boxes');N=len(z['masks_bool'])
 if rects is None or np.asarray(rects).shape!=(N,4) or not np.isfinite(rects).all() or np.any(np.asarray(rects)[:,2:]<=np.asarray(rects)[:,:2]):detail['status']='missing_or_invalid_detector_box_inventory';return detail
 if N==0:detail.update(status='no_independent_masks',mask_IoU=[],exact_max_tie_n=0);return detail
 candidates={int(c['mask_index']):c for c in case['candidates']};assert set(candidates)==set(range(N))
 assert all(np.array_equal(np.asarray(candidates[m]['bbox_xyxy']),rects[m]) for m in range(N))
 vals=np.array([rect_iou(box,r) for r in rects],np.float64);assert np.isfinite(vals).all() and np.all((vals>=0)&(vals<=1));mx=float(vals.max());winners=np.flatnonzero(vals==mx);order=np.sort(vals);detail.update(mask_IoU=vals.tolist(),top_IoU=mx,top2_margin=float(mx-order[-2]) if N>1 else None,exact_max_tie_n=len(winners))
 if mx<=0:detail['status']='all_rectangle_IoU_zero'
 elif len(winners)!=1:detail['status']='exact_max_rectangle_IoU_tie'
 else:detail.update(status='unique_positive_rectangle_IoU',mask_index=int(winners[0]))
 return detail

def iou_readout(assignment,case,z,subset=False):
 q0=int(z['source_scores'].argmax());m=assignment['mask_index'];table=case['old_subset_groups'] if subset else case['full_groups'];g=table.get(str(m),[]) if m is not None else []
 common={'mask_index':m,'original_bbox_xyxy':assignment['bbox_original_xyxy'],'old_subset_contains_selected_mask':m is not None and str(m) in case['old_subset_groups'],'full_group_nonempty':m is not None and bool(case['full_groups'].get(str(m),[])),'assignment_status':assignment['status'],'no_second_best_if_3D_group_empty':True}
 if not g:return dict(common,query=q0,mapped=False,status=assignment['status'] if m is None else 'absent_from_old_subset' if subset and str(m) not in table else 'mask_without_3D_group',group_size=0,query_changed=False)
 q=max(g,key=lambda i:(float(z['source_scores'][i]),-i));return dict(common,query=int(q),mapped=True,status='mapped_original_query',group_size=len(g),query_changed=q!=q0)
