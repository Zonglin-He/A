"""Pure synthetic diagnostic checks, no actual metadata/GT/images/predictions."""
from pathlib import Path
import argparse,json,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import diagnose_association as d


def check():
 checks={};rng=np.random.default_rng(1302)
 for platform in ('drone','quad'):
  for _ in range(64):
   b=rng.normal(size=7);b[3:6]=rng.uniform(.1,5,3)
   # Arbitrary unused tail fields cannot affect original frozen official7D.
   wrapped=np.r_[b,rng.normal(size=2)]
   assert np.array_equal(d.transform_other7(b,platform),d.transform_gt(wrapped,platform))
 checks['explicit_other7_exact_R5_bridge_float64_JSON_geometry_128cases']=True
 raw=np.array([0.,0.,0.,2.,2.,2.,0.,.77,3.])
 same={'bbox_3d_other':raw[:7].tolist(),'class_other':'synthetic ignored'}
 far={'bbox_3d_other':[5.,0,0,2,2,2,0.]}
 target,others,schema=d.annotated_geometry(raw,[same,same,far,far],'quad')
 assert len(others)==1 and schema['target_exact_duplicate_annotation_n']==2 and schema['group_member_counts']==[3,2]
 assert schema['other_source_schema']=='other7_no_fabricated_tails'
 checks['exact7D_dedup_before_IoU_without_ID_or_class_usage']=True
 # Exact rule does not introduce an unregistered near-equality merge.
 almost=dict(same,bbox_3d_other=[1e-7,0,0,2,2,2,0])
 _,near,s2=d.annotated_geometry(raw,[almost],'quad');assert len(near)==1
 checks['near_but_not_exact_is_not_silently_merged']=True
 assert d.classify(.25,0.,'defined')=='unconfirmed_spatial_relation'
 assert d.classify(.5,.5,'defined')=='unconfirmed_spatial_relation'
 assert d.classify(.6,.2,'defined')=='target_spatially_dominant'
 assert d.classify(.2,.6,'defined')=='annotated_other_spatially_dominant'
 assert d.classify(.9,None,'defined')=='NA_no_distinct_annotated_other_geometry'
 assert d.classify(.9,.1,'NA_partial_other_schema')=='NA_partial_other_schema'
 checks['strict_thresholds_margin_and_unconfirmed_NA_not_wrong']=True
 assert d.mask_relation(2,2)=='same_defined_mask' and d.mask_relation(None,None)=='both_undefined'
 checks['same_detector_mask_separate_from_query_and_identity']=True
 classes=np.array([1,41,3]);a=d.predicted_category_diagnostic(1,classes,(1,));b=d.predicted_category_diagnostic(0,classes,(1,))
 assert a['firstnoun_family_compatible'] is False and b['firstnoun_family_compatible'] is True
 assert d.predicted_category_diagnostic(None,classes,(1,))['firstnoun_family_compatible'] is None
 checks['frozen_COCO_prediction_compatibility_not_GT']=True
 try:d.transform_other7(raw[:7],'waymo')
 except ValueError:checks['unknown_or_nonDQ_platform_rejected']=True
 else:raise AssertionError('Waymo silently accepted')
 # Loss of target IoU can happen from another proposal of the very same mask;
 # this fixture does not manufacture a separate semantic identity annotation.
 target7=d.transform_other7(raw[:7],'quad')
 values,_=d.candidate_ious(np.array([[0,0,0,2,2,2],[1,0,0,2,2,2]],float),target7)
 assert values[0]>values[1]
 checks['same_mask_different_proposal_can_change_target_overlap']=True
 return {'state':'passed_synthetic_postscore_diagnostic_checks_no_actual_numeric_GT',
  'checks':checks,'program_sha256':{str(p.relative_to(d.ROOT)):d.sha(p) for p in [Path(d.__file__),Path(__file__)]},
  'dependency_sha256':d.DEPENDENCIES,'actual_annotation_geometry_or_IoU_or_model_or_GPU_read':False}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=check()
 if a.output.exists():raise FileExistsError(a.output)
 d.dump(a.output,out);print(json.dumps({'state':out['state'],'n_checks':len(out['checks'])}))
