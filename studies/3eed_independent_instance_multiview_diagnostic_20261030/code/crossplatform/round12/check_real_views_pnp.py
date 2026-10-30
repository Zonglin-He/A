"""Input-only fixed five-pair sensor correspondence check. No source q0 or labels."""
from pathlib import Path
import argparse,json,hashlib,sys,time
import numpy as np
from PIL import Image
D=Path(__file__).resolve().parent;sys.path[:0]=[str(D),str(D.parent/'round7')]
from camera_projection import CameraSpec
from sensor_pnp import estimate_pair_registration

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def conv(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,dict):return {k:conv(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [conv(v) for v in x]
 return x

def main(a):
 import cv2
 cv2.setNumThreads(1);p=json.loads(a.plan.read_text());assert sha(a.plan)==a.plan_sha256 and p['program_sha256']==sha(__file__);assert p['temporal_module_sha256']==sha(D/'temporal_association.py');assert p['ICP_module_sha256']==sha(D/'sensor_icp.py');assert p['PnP_module_sha256']==sha(D/'sensor_pnp.py')
 index=json.loads(Path(p['frame_index']).read_text());assert sha(p['frame_index'])==p['frame_index_sha256'];md=Path(p['mask_dir']);assert sha(md/'SEALED.json')==p['mask_seal_sha256'];seal=json.loads((md/'SEALED.json').read_text())
 assert not a.output.exists();a.output.mkdir(parents=True);rows=[];start=time.perf_counter()
 def frame(alias):
  x=index[alias];assert sha(x['imagepath'])==x['image_sha256'] and sha(x['pointpath'])==x['point_sha256'];f=md/(alias+'.npz');assert sha(f)==seal['files'][f.name]
  with np.load(f,allow_pickle=False) as z:m=z['masks_bool'].copy()
  rgb=np.array(Image.open(x['imagepath']).convert('RGB'),copy=True);raw=np.fromfile(x['pointpath'],np.float32).reshape(-1,4)[:,:3]
  keys={k:x[k] for k in ['image_intrinsic','image_extrinsic']}
  if x['camera_distortion'] is not None:keys['camera_distortion']=x['camera_distortion']
  c=CameraSpec.from_camera_fields(x['platform'],keys,missing_distortion_policy='assume_zero_declared')
  return x,rgb,raw,c,m.any(0)
 for item in p['pairs_PRIVATE']:
  A,ra,pa,ca,ma=frame(item['anchor_frame_alias']);rr={}
  for role in ['helper_frame_alias','wrong_frame_alias']:
   B,rb,pb,cb,mb=frame(item[role]);assert (A['image_sha256'],A['point_sha256'])!=(B['image_sha256'],B['point_sha256'])
   out=estimate_pair_registration(rgb_A=ra,raw_points_A=pa,camera_A=ca,rgb_B=rb,raw_points_B=pb,camera_B=cb,excluded_mask_A=ma,excluded_mask_B=mb,seed=0)
   rr[role]=conv(out)
  rows.append({'case_alias':item['case_alias'],'platform':A['platform'],'pair_diagnostics':rr})
 out={'state':'completed_actual5sensor_pair_and_wrongview_input_only_probe','n_pairs':5,'program_sha256':sha(__file__),'plan_sha256':sha(a.plan),'mask_seal_sha256':p['mask_seal_sha256'],'rows':rows,'seconds':time.perf_counter()-start,'no_targetGT_or_sourceq0_or_neural_updates':True};(a.output/'RESULTS_PRIVATE.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');(a.output/'SEALED.json').write_text(json.dumps({'files':{'RESULTS_PRIVATE.json':sha(a.output/'RESULTS_PRIVATE.json')},'program_sha256':sha(__file__),'plan_sha256':sha(a.plan)},indent=2)+'\n');print(json.dumps({'pairs':5,'seconds':out['seconds'],'real_supported':sum(r['pair_diagnostics']['helper_frame_alias']['accepted'] for r in rows),'wrong_supported':sum(r['pair_diagnostics']['wrong_frame_alias']['accepted'] for r in rows)}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha256',required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
