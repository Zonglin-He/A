"""Same R144 fixed statistic, exact source sampledraw per prefix, no labels/models."""
from common145 import *
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0,str(R/'crossplatform/round7'))
from camera_projection import raw_to_model_points

def main():
 check();hs=rd(R/'research/round144/PREPARATION.json')['source_own_median_q10'];rows=[];seen={};mx=0.
 for e in rd(R/'research/round132/SOURCE_CALIBRATION_INPUTS_PRIVATE.json'):
  en=e['entry'];src,r=en['source'],en['row'];a=e['A'];assert a['eligible_for_extra_media'] and not a['protected_image_content_alias'];p=R/en['files']['prefix']['path'];assert sha(p)==en['files']['prefix']['sha256'];z=np.load(p);key=src+'|'+a['pointpath']+'|'+hashlib.sha256(z['point_indices'].tobytes()).hexdigest()+'|'+str(r.get('suffix'))
  if key not in seen:
   assert sha(a['pointpath'])==a['point_sha256'];f=Path(a['pointpath']);raw=np.load(f,allow_pickle=False) if f.suffix=='.npy' else np.fromfile(f,np.float32).reshape(-1,4);xyz=raw_to_model_points(raw[z['point_indices'],:3],src,int(r.get('suffix') or 0));assert xyz.shape==(16384,3) and np.isfinite(xyz).all();seen[key]=(float(np.quantile(xyz[:,2],.1)),xyz)
  q,xyz=seen[key];err=float(cKDTree(xyz).query(z['xyz'][0])[0].max());mx=max(mx,err);assert err<1e-4;old=rd(R/'crossplatform/round140/source'/src/(r['id']+'.json'));rows.append({'source':src,'id_PRIVATE':r['id'],'q10':q,'source_reference':hs[src],'delta':hs[src]-q if old['has_B'] else 0.,'has_B':old['has_B'],'seed_bridge_error':err})
 dump(H/'SOURCE_POSITION_ROWS_PRIVATE.json',rows);out={'n':len(rows),'sampledraws':len(seen),'max_seed_bridge_error':mx,'source_own_H_from_R144':hs,'no_source_or_target_GT_or_model_read':True,'noB_source_delta_zero':True};dump(H/'SOURCE_POSITION_PREPARATION.json',out);dump(H/'SOURCE_POSITION_SEAL.json',{'files':{str(p.relative_to(R)):sha(p) for p in [H/'SOURCE_POSITION_ROWS_PRIVATE.json',H/'SOURCE_POSITION_PREPARATION.json']}});print(out,flush=True)
if __name__=='__main__':main()
