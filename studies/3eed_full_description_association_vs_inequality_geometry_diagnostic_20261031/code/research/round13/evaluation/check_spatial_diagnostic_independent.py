"""Synthetic other7 coordinate/dedup/category checks; never calls IoU or reads GT."""
from pathlib import Path
import importlib.util,json,hashlib
import numpy as np
R=Path(__file__).resolve().parents[1];f=R/'association_diagnostic/diagnose_association.py';s=importlib.util.spec_from_file_location('R13_diagnostic_adapter_under_review',f);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
rng=np.random.default_rng(713);n=0
for i in range(64):
 raw=np.r_[rng.normal(size=3)*20,rng.uniform(.1,8,3),rng.uniform(-np.pi,np.pi)].astype(np.float64);before=raw.copy()
 for p in ('drone','quad'):
  out=m.transform_other7(raw,p);expected=raw.copy()
  if p=='drone':expected[2]+=1.8
  assert out.dtype==np.float32 and np.array_equal(out,expected.astype(np.float32));ref=m.transform_gt(np.r_[raw,rng.normal(size=2)],p);assert np.array_equal(out,ref) and np.array_equal(raw,before);n+=1
checks={'independent128_synthetic_other7_R5_and_analytic_coordinate_bridges':True}
raw=np.array([1,2,3.2,2,2,2,.4],float);target=np.r_[raw,.3,1.];near=raw.copy();near[0]=float(np.nextafter(np.float32(1),np.float32(np.inf)))
t,other,d=m.annotated_geometry(target,[{'bbox_3d_other':raw.tolist()},{'bbox_3d_other':near.tolist()},{'bbox_3d_other':near.tolist()}],'drone');assert len(other)==1 and d['target_exact_duplicate_annotation_n']==1 and d['group_member_counts']==[2,2];checks['exact_f32_geometry_dedup_retains_one_ULP_distinct_geometry_not_physical_IDs']=True
_,_,bad=m.annotated_geometry(target,[{'bbox_3d_other':raw.tolist()},{'bbox_3d_other':[1,2]}],'quad');assert bad['schema']=='NA_partial_other_schema' and bad['malformed_other_n']==1 and m.classify(.8,.1,bad['schema'])=='NA_partial_other_schema';checks['malformed_other_preserves_NA_not_identity_wrong']=True
assert m.classify(.25,0,'defined')=='unconfirmed_spatial_relation' and m.classify(.5,.4,'defined')=='unconfirmed_spatial_relation' and m.classify(.6,.2,'defined')=='target_spatially_dominant' and m.classify(.1,.8,'defined')=='annotated_other_spatially_dominant' and m.classify(.8,None,'defined')=='NA_no_distinct_annotated_other_geometry';checks['strict_iou_margin_and_missing_NA_rules']=True
for rawbad,p,suf in [([1,2], 'drone',None),([0,0,0,-1,1,1,0],'quad',None),([0,0,0,1,1,1,0],'waymo',None),([0,0,0,1,1,1,0],'quad',0)]:
 try:m.transform_other7(rawbad,p,suf)
 except ValueError:pass
 else:raise AssertionError('Malformedadapter input accepted')
checks['non7_nonpositive_nonDQ_suffix_rejected']=True
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();out={'state':'PASS_R13_INDEPENDENT_SYNTHETIC_SPATIAL_ADAPTER_CHECKS','program_sha256':sha(f),'auditor_sha256':sha(Path(__file__)),'coordinate_bridge_comparisons':n,'checks':checks,'no_actual_GT_metadata_or_coordinates_index_payload_task_IoU_model_GPU':True};p=R/'evaluation/SPATIAL_ADAPTER_INDEPENDENT_CPU_FIXTURE.json';assert not p.exists();p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'state':out['state'],'bridge_n':n,'sha256':sha(p)}))
