"""Independent oracle-only coordinate capability audit on supplied sealed arrays.
No annotation/model file IO. Calling this is not unsupervised TTA evaluation.
"""
import numpy as np
MASKS={'C':np.array([1,1,1,0,0,0],bool),'S':np.array([0,0,0,1,1,1],bool),'CS':np.ones(6,bool)}

def errors_to_direct_target(actual,direct,mask):
 a=np.asarray(actual,float);d=np.asarray(direct,float);m=np.asarray(mask,bool)
 if a.shape!=d.shape or a.shape!=(6,) or m.shape!=(6,) or not np.isfinite(a).all() or not np.isfinite(d).all():raise ValueError('Finite six coordinates required')
 e=a-d
 return {'supervised_coordinate_Linf_m':float(abs(e[m]).max()),'supervised_coordinate_L2_m':float(np.linalg.norm(e[m])),'all6_coordinate_Linf_vs_direct_m':float(abs(e).max()),'center_L2_vs_direct_m':float(np.linalg.norm(e[:3])),'size_L2_vs_direct_m':float(np.linalg.norm(e[3:]))}

def review_capacity_case(source6,target6,boxes_by_arm,arm_map,trace_by_control=None,exactfit_tolerance=5e-5):
 """arm_map[C/S/CS] maps direct/param1/param3/exactfit to actual NPZ names.
 No argmax, candidate selection, target yaw or class is used here.
 """
 source=np.asarray(source6);target=np.asarray(target6)
 if source.shape!=(6,) or target.shape!=(6,) or not np.isfinite(source).all() or not np.isfinite(target).all() or (target[3:]<=0).any():raise ValueError('Source finite and GT target6 valid dimensions required')
 out={}
 for kind,m in MASKS.items():
  direct=np.where(m,target,source);names=arm_map[kind];supplied=np.asarray(boxes_by_arm[names['direct']]);assert np.array_equal(supplied,direct),'Direct GT must preserve all inactive raw coordinates'
  controls={}
  for k in ('direct','param1','param3','exactfit'):
   box=np.asarray(boxes_by_arm[names[k]]);assert box.shape==(6,) and np.array_equal(box[~m],source[~m]),'Inactive coordinates changed'
   controls[k]=errors_to_direct_target(box,direct,m)
  exact_error=controls['exactfit']['supervised_coordinate_Linf_m']
  # Preserve failure as data, never silently enlarge registered tolerance.
  out[kind]={'oracle_only_NOT_TTA':True,'mask':m.astype(int).tolist(),'source_error_to_direct_target':errors_to_direct_target(source,direct,m),'controls':controls,'exactfit_registered_tolerance_m':exactfit_tolerance,'exactfit_within_registered_tolerance':exact_error<=exactfit_tolerance,'source_raw_nonpositive_size_count':int((source[3:]<=0).sum())}
  if trace_by_control is not None:
   traces=trace_by_control[kind]
   for control in ('param1','param3','exactfit'):
    t=traces[control];out[kind]['controls'][control]['actual_parameter_update']=t
    # Grad/update need not be nonzero when source already equals target.
    nonzero_target=bool(np.any(source[m]!=target[m]));out[kind]['controls'][control]['source_differs_target']=nonzero_target
 return out
