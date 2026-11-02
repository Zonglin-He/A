"""Only old-A2 fresh-frame versus historical observation/prepare contracts."""
import sys
import numpy as np
from common46 import *
def main():
 a=args(__doc__);plan,rows,out=context(a);assert plan['execution_mode']=='engineering_oldA2' and plan['max_MaskRCNN_calls']+plan.get('prior_mask_calls',0)==2;verify(out/'frames')
 sys.path.insert(0,str(ROOT/'crossplatform/round24'));import common24
 from observations import load as old_load
 sys.path.insert(0,str(ROOT/'crossplatform/round17'));from pair_geometry import prepare
 records=[]
 for r in rows:
  old,os,ob=old_load(r);new,ns,nb=fresh_observation(r,out);mask_exact={k:bool(np.array_equal(old['instances'][k],new['instances'][k])) for k in ('boxes_xyxy','class_ids','scores','masks_bool','instance_indices')};assert all(mask_exact.values())
  camera_exact={k:bool(np.array_equal(getattr(old['camera'],k),getattr(new['camera'],k))) for k in ('intrinsic','extrinsic','distortion')};assert all(camera_exact.values()) and old['camera'].missing_distortion_policy==new['camera'].missing_distortion_policy
  assert np.array_equal(old['raw_points'],new['raw_points']) and np.array_equal(os['visible_points_model'],ns['visible_points_model']) and np.array_equal(os['depth_camera'],ns['depth_camera'])
  z=npz(ROOT/'crossplatform/round24/capture336_attempt001/samples_PRIVATE'/(r['id']+'.npz'));a0=prepare(os,z['boxes'],r['id']);a1=prepare(ns,z['boxes'],r['id']);fullprepare=serial(a0)==serial(a1);assert fullprepare
  records.append({'platform':r['platform'],'mask_arrays_exact':mask_exact,'camera_fields_exact':camera_exact,'raw_and_visible_points_depth_exact':True,'full_R17_prepare_exact':fullprepare,'all_original_G_downstream_inputs_unchanged':True})
 s=read(out/'frames/SEALED.json');assert s['model_calls']+s.get('prior_mask_calls',0)==2;result={'state':'PASS_FRESH_OLD_A2_MASK_CAMERA_AND_FULL_PREPARE_CONTRACTS','rows':records,'MaskRCNN_calls_total_including_prior_failed_io':2,'current_job_MaskRCNN_calls':s['model_calls'],'new_Qwen_DINO_optimizer_steps':0,'task_GT_FINAL_access':False,'frame_seal_sha256':sha(out/'frames/SEALED.json'),'old_full_controller_seal_reused_without_new_downstream_model_calls':sha(ROOT/'crossplatform/round46/engineering2_attempt001/SEALED.json')};dump(out/'FRESH_CONTRACT_AUDIT.json',result);print(result['state'],flush=True)
if __name__=='__main__':main()
