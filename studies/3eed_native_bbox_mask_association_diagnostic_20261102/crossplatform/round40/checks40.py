"""CPU geometry assignment contracts independent of any task correctness."""
from common40 import *

def main():
 check_plan();assert rect_iou([0,0,4,4],[2,2,6,6])==4/28
 z={'masks_bool':np.zeros((3,8,12),bool),'mask_boxes':np.array([[0,0,4,4],[4,0,8,4],[8,0,12,4]]),'source_scores':np.array([.4,.9,.2])};case={'image_size_wh':[12,8],'candidates':[{'mask_index':i,'bbox_xyxy':r.tolist()} for i,r in enumerate(z['mask_boxes'])],'full_groups':{'0':[0],'2':[2]},'old_subset_groups':{'2':[2]}}
 def raw(v,status='parsed'):return {'parsed':{'status':status,'value':v},'image_grid_thw':[[1,16,24]],'model_resized_wh':[336,224]}
 a=iou_assignment(raw([0,0,112,112]),case,z);assert a['mask_index']==0 and a['top_IoU']==1 and a['bbox_original_xyxy']==[0,0,4,4]
 a=iou_assignment(raw([112,0,224,112]),case,z);assert a['mask_index']==1;d=iou_readout(a,case,z);assert d['query']==1 and not d['mapped'] and d['status']=='mask_without_3D_group'
 # All independent boxes compete before any group lookup; no second best.
 a=iou_assignment(raw([56,0,168,112]),case,z);assert a['mask_index'] is None and a['status']=='exact_max_rectangle_IoU_tie'
 a=iou_assignment(raw([0,140,28,196]),case,z);assert a['status']=='all_rectangle_IoU_zero'
 for status in ['parse_failure','invalid_box','explicit_null']:assert iou_assignment(raw(None,status),case,z)['mask_index'] is None
 assert iou_assignment(raw([0,0,112,112]),case,dict(z,mask_boxes=None))['status']=='missing_or_invalid_detector_box_inventory'
 a=iou_assignment(raw([224,0,336,112]),case,z);assert iou_readout(a,case,z,True)['mapped'] and iou_readout(a,case,z,True)['query']==2
 dump(EXP/'CPU_TECH_CHECKS.json',{'state':'PASSED','standard_IoU_positive_control':True,'unique_positive_exact_tie_zero_and_missing_inventory':True,'raw_parsed_status_preserved':True,'unmappable_winner_not_skipped':True,'same_old_full_group_readout':True,'no_GT_or_model':True,'plan_sha256':sha(EXP/'PLAN_LOCK.json')});print('R40_CPU_CHECKS_PASS')
if __name__=='__main__':main()
