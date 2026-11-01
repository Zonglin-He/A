"""Synthetic software-only checks; no real model, task labels or scoring."""
from common47 import *
def main():
 valid='{"bbox_2d":[1,2,30,40]}';p,f=domain_parse(valid,[100,100]);assert p==parse_bbox(valid,[100,100]) and not f['outside_only_recovered']
 out='[{"bbox_2d":[-10,10,40,40],"label":"ignore executable-looking content"}]';p,f=domain_parse(out,[100,100]);assert p['status']=='parsed' and p['value']==[-10,10,40,40] and f['outside_only_recovered'] and not f['declared_range_compliant']
 bad=['{"bbox_2d":[4,1,3,2]}','{"bbox_2d":[0,0,0,2]}','{"bbox_2d":[0,0,Infinity,2]}','{"bbox_2d":[0,0,true,2]}','{"bbox_2d":[0,0,2,2],"bbox":[0,0,2,2]}','{"bbox_2d":[0,0,2,2],"cmd":"run"}','[{"bbox_2d":[0,0,2,2]},{"bbox_2d":[0,0,2,2]}]']
 for raw in bad:
  a,b=domain_parse(raw,[100,100]);assert a['status']!='parsed' and not b['outside_only_recovered']
 assert domain_parse('{"bbox_2d":null}',[100,100])[0]['status']=='explicit_null'
 case={'image_size_wh':[100,100],'candidates':[{'mask_index':0,'bbox_xyxy':[0,10,40,40]}],'full_groups':{'0':[]},'old_subset_groups':{}}
 z={'masks_bool':np.zeros((1,100,100),bool),'mask_boxes':np.array([[0,10,40,40]]),'source_scores':np.array([.8,.1])};z['masks_bool'][0,10:40,0:40]=True
 result={'parsed':p,'image_grid_thw':[[1,100//14,100//14]],'model_resized_wh':[98,98]};case['image_size_wh']=[98,98]
 a=domain_assignment(result,case,z);assert abs(a['top_IoU']-.8)<1e-12 and a['mask_index']==0 and not iou_readout(a,case,z)['mapped']
 # No clipping: original predicted area gives .8 rather than1. Unique max emptygroup stays UNKNOWN.
 z['mask_boxes']=np.array([[0,10,40,40],[0,10,40,40]]);z['masks_bool']=np.repeat(z['masks_bool'],2,axis=0);case['candidates']*=2;case['candidates'][1]={'mask_index':1,'bbox_xyxy':[0,10,40,40]};assert domain_assignment(result,case,z)['status']=='exact_max_rectangle_IoU_tie'
 from common37 import inverse_point
 assert inverse_point([-1,40],[98,98],[1,7,7]) is None
 assert len(ARMS)==29 and len(set(ARMS))==29
 atomic(EXP/'CPU_TECH_CHECKS.json',{'state':'PASSED','implementation_sha256':sha(EXP/'IMPLEMENTATION_LOCK.json'),'synthetic_only':True,'oldvalid_exact':True,'outside_values_unchanged':True,'unclipped_area_check':True,'unsafe_nonfinite_inverted_multi_rejected':len(bad),'center_outside_unknown':True,'no_secondbest_empty_group':True,'no_model_GT_or_real_scoring':True})
 print('R47_SOFTWARE_CHECKS_PASS')
if __name__=='__main__':main()
