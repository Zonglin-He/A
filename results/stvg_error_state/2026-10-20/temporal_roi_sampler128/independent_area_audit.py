"""Post-seal NumPy pixel/cell-intersection audit; no GT or model inference."""
from pathlib import Path
import sys,numpy as np
R=Path(__file__).resolve().parents[4];sys.path.insert(0,str(R))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load
D=R/'artifacts/stvg_temporal_roi_sampler128_v1';OLD=R/'artifacts/stvg_temporal_object_memory128_v1';PRE=R/'artifacts/stvg_referdino_condition_hc128_v1';FIN=R/'artifacts/stvg_referdino_finalize_v1';P=Path(__file__).resolve().parent
p=read(D/'PLAN_LOCK.json');checks=0;worst=0.;empty=0
for rec in p['records']:
 a=rec['alias'];z=load(D/'predictions'/f'{a}.pt');old=load(OLD/'temporal_inputs'/f'{a}.pt');inp=load(PRE/'inputs'/f'{a}.pt');sp=load(FIN/'Live/predictions'/f'{a}.pt');t=load(PRE/'teacher0'/f'{a}.pt')['native']['boxes'].numpy().astype(float);ids=np.asarray(inp['frame_ids']);xy=np.concatenate([t[:,:2]-t[:,2:]/2,t[:,:2]+t[:,2:]/2],1);interp=np.column_stack([np.interp(ids,ids[rec['positions8']],xy[:,j])for j in range(4)]);ref=np.concatenate([(interp[:,:2]+interp[:,2:])/2,interp[:,2:]-interp[:,:2]],1);shift=ref.copy();shift[:,0]=(shift[:,0]+.5)%1
 boxes=dict(E_ROI=np.asarray(inp['E']),**{f'Live{s}_ROI':np.asarray(sp['outputs'][f'T20_Live_{s}'])for s in p['seeds']},ReferROI=ref,ShiftedReferROI=shift)
 for off,g in enumerate(old['grids']):
  h,w=g['h'],g['w'];valid=g['valid'].numpy()
  for c,bs in boxes.items():
   got=z['pooling_audit'][c+str(off)]
   for k,(b,v)in enumerate(zip(bs[off::2],valid)):
    vm=v.reshape(h,w);vh=int(vm.any(1).sum());vw=int(vm.any(0).sum());mask=np.zeros((h,w));lo=b[:2]-b[2:]/2;hi=b[:2]+b[2:]/2
    # Direct intervals in normalized image coordinates, not Torch grid helper.
    for iy in range(vh):
     for ix in range(vw):mask[iy,ix]=max(0,min(hi[0],(ix+1)/vw)-max(lo[0],ix/vw))*max(0,min(hi[1],(iy+1)/vh)-max(lo[1],iy/vh))*vh*vw
    mask=(mask.flatten().astype(np.float32)*v);missing=mask.sum()<=0
    expected=v.astype(np.float32)/v.sum()if missing else mask/mask.sum();error=float(abs(expected-got['weights'][k].numpy()).max());assert error<2e-6 and missing==got['empty'][k];worst=max(worst,error);checks+=1;empty+=missing
put(P/'INDEPENDENT_AREA_AUDIT.json',dict(status='pass',direct_numpy_cell_area_checks=checks,maximum_weight_error=worst,empty_ROI_fallbacks=int(empty),reference_physical_interpolation_independently_rebuilt=True,active_weight_field_ValidGlobal='valid_weights; meta.weights is the unused ReferROI pool computed alongside it',sealed_predictions_unchanged=True,new_models=0,new_GT=0))
print(checks,worst,flush=True)
