"""R47 cached geometry-domain diagnosis. No model, GT, clipping or unit changes."""
import sys,math,copy,datetime
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'crossplatform/round45'),str(ROOT/'crossplatform/round13')]
from common45 import rows,prepared,arrays,read,sha,verify,load_npz,atomic,atomic_npz,parse_bbox,bbox_readout,iou_readout,ARMS as STRICT_ARMS
from association import rect_iou
BASE=ROOT/'crossplatform/round47';EXP=ROOT/'research/round47/experiment';OUT=BASE/'predictions208_attempt001'
STRICT=ROOT/'crossplatform/round45/bboxA208_attempt001';P3=ROOT/'crossplatform/round39/inference208_attempt002';PRE=ROOT/'crossplatform/round37/prepared208_attempt001'
NEW_ARMS=[f'{cap}_bbox_domain_{mode}_{mapping}' for cap in ['3B','7B'] for mode in ['center','rectIoU'] for mapping in ['fullmap','oldsubset']]
ARMS=STRICT_ARMS+NEW_ARMS

def domain_parse(raw,wh):
 p=parse_bbox(raw,wh);s=copy.deepcopy(p);recover=p['status']=='invalid_box' and p.get('reason')=='outside_declared_input_range'
 if recover:s.update(status='parsed',reason=None)
 return s,{'strict_status':p['status'],'strict_reason':p.get('reason'),'outside_only_recovered':recover,'declared_range_compliant':p['status']=='parsed','original_values_unchanged':s['value']==p['value'],'not_original_prompt_compliance_certification':True}

def domain_assignment(result,case,z):
 p=result['parsed'];grid=result['image_grid_thw'][0];W,H=case['image_size_wh'];Wp,Hp=int(grid[2])*14,int(grid[1])*14;assert result['model_resized_wh']==[Wp,Hp]
 detail={'status':p['status'],'mask_index':None,'bbox_resized_unchanged':p['value'],'bbox_original_xyxy':None,'mask_IoU':None,'top_IoU':None,'top2_margin':None,'exact_max_tie_n':None,'all_masks_including_unmappable':True}
 if p['status']!='parsed':return detail
 x1,y1,x2,y2=p['value'];assert all(math.isfinite(v) for v in [x1,y1,x2,y2]) and x1<x2 and y1<y2
 box=[x1*W/Wp,y1*H/Hp,x2*W/Wp,y2*H/Hp];detail['bbox_original_xyxy']=box
 if not all(math.isfinite(v) for v in box):detail.update(status='nonfinite_geometry_unknown',bbox_original_xyxy=None);return detail
 rects=z.get('mask_boxes');N=len(z['masks_bool'])
 if rects is None or np.asarray(rects).shape!=(N,4) or not np.isfinite(rects).all() or np.any(np.asarray(rects)[:,2:]<=np.asarray(rects)[:,:2]):detail['status']='missing_or_invalid_detector_box_inventory';return detail
 if N==0:detail.update(status='no_independent_masks',mask_IoU=[],exact_max_tie_n=0);return detail
 candidates={int(c['mask_index']):c for c in case['candidates']};assert set(candidates)==set(range(N)) and all(np.array_equal(np.asarray(candidates[m]['bbox_xyxy']),rects[m]) for m in range(N))
 with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
  area=(box[2]-box[0])*(box[3]-box[1]);vals=np.array([rect_iou(box,r) for r in rects],np.float64)
 if not math.isfinite(area) or not np.isfinite(vals).all() or not np.all((vals>=0)&(vals<=1)):detail['status']='nonfinite_geometry_unknown';return detail
 mx=float(vals.max());winners=np.flatnonzero(vals==mx);order=np.sort(vals);detail.update(mask_IoU=vals.tolist(),top_IoU=mx,top2_margin=float(mx-order[-2]) if N>1 else None,exact_max_tie_n=len(winners))
 if mx<=0:detail['status']='all_rectangle_IoU_zero'
 elif len(winners)!=1:detail['status']='exact_max_rectangle_IoU_tie'
 else:detail.update(status='unique_positive_rectangle_IoU',mask_index=int(winners[0]))
 return detail

def readouts_for(cap,call,case,z):
 result=copy.deepcopy(call);p,flags=domain_parse(call['raw_answer_PRIVATE'] if 'raw_answer_PRIVATE' in call else call['raw'],call['model_resized_wh']);result['parsed']=p
 assignment=domain_assignment(result,case,z);reads={};center_result=result
 if p['status']=='parsed':
  v=p['value'];center=[(v[0]+v[2])/2,(v[1]+v[3])/2]
  if not all(math.isfinite(x) for x in center):center_result=copy.deepcopy(result);center_result['parsed']['status']='nonfinite_center_geometry_unknown'
 for mapping in ['fullmap','oldsubset']:
  reads[f'{cap}_bbox_domain_center_{mapping}']=bbox_readout(center_result,case,z,subset=mapping=='oldsubset')
  reads[f'{cap}_bbox_domain_rectIoU_{mapping}']=iou_readout(assignment,case,z,subset=mapping=='oldsubset')
 return reads,{'domain_parsed':p,'domain_flags':flags,'assignment':assignment}

def check_plan():
 p=read(EXP/'PLAN_LOCK.json');i=read(EXP/'IMPLEMENTATION_LOCK.json');a=read(EXP/'ROOT_EXECUTION_AUTHORIZATION.json')
 assert a['state']=='ROOT_AUTHORIZED_CACHED_BBOX_GEOMETRIC_DOMAIN_DIAGNOSTIC' and i['plan_sha256']==sha(EXP/'PLAN_LOCK.json')
 for f,h in {**p['input_sha256'],**i['program_sha256']}.items():assert sha(ROOT/f)==h,f
 assert read(EXP/'CPU_TECH_CHECKS.json')['state']=='PASSED'
 review=read(EXP/'ROOT_STATIC_REVIEW.json');assert review['state']=='PASS' and review['implementation_sha256']==sha(EXP/'IMPLEMENTATION_LOCK.json')
 return p
