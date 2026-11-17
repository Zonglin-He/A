"""Prospective sensor-coordinate metadata check, separate from batch139.

No model, media decoding, point transformation or task annotation consumer.
This does not change any active experiment or prescribe a normalization.
"""
from common139 import *
import numpy as np
sys.path.insert(0,str(R/'crossplatform/round7'))
from camera_projection import CameraSpec,read_camera_allowlist,model_to_camera_matrix
def angle(A,B):return float(np.degrees(np.arccos(np.clip((np.trace(A@B.T)-1)/2,-1,1))))
def main():
 files={'waymo':R/'research/round76/input/SOURCE_VAL128_PRIVATE.json','drone':R/'research/round97/input/SOURCE_DRONE_VAL128_PRIVATE.json','quad':R/'research/round97/input/SOURCE_QUAD_VAL128_PRIVATE.json'};means={};summaries=[];private=[]
 for src,path in files.items():
  rows=rd(path);seen=set();Rs=[];assumed=0
  for row in rows:
   key=(row['metapath'],row.get('suffix',0))
   if key in seen:continue
   seen.add(key);cam=CameraSpec.from_camera_fields(src,read_camera_allowlist(row['metapath']),waymo_view_suffix=int(row.get('suffix') or 0),missing_distortion_policy='assume_zero_declared');assumed+=cam.validated()[3];T=model_to_camera_matrix(cam);Rs.append(T[:3,:3]);private.append({'source':src,'calibration_path_PRIVATE':row['metapath'],'T_model_to_camera_PRIVATE':T.tolist()})
  U,_,V=np.linalg.svd(np.stack(Rs).sum(0));D=np.eye(3);D[-1,-1]=np.linalg.det(U@V);M=U@D@V;means[src]=M;dev=[angle(Q,M) for Q in Rs];summaries.append({'source':src,'source_VAL_input_rows':len(rows),'unique_calibration_views':len(Rs),'median_rotation_deviation_deg':float(np.median(dev)),'p90_rotation_deviation_deg':float(np.quantile(dev,.9)),'max_rotation_deviation_deg':max(dev),'zero_distortion_assumptions':int(assumed)})
 between=[{'source':s,'target':t,'mean_camera_model_rotation_difference_deg':angle(means[s],means[t])} for s in files for t in files if s!=t];out={'prospective_metadata_only':True,'not_a_model_failure_or_normalization_proof':True,'batch139_unchanged':True,'source_only_calibration_metadata':True,'no_target_inputs_or_GT':True,'no_media_model_task_score':True,'within_source':summaries,'between_sources':between,'FINAL_access':False};dump(H/'CAMERA_COORDINATE_METADATA_ONLY.json',out);dump(H/'CAMERA_TRANSFORMS_PRIVATE.json',private);print(out,flush=True)
if __name__=='__main__':main()
