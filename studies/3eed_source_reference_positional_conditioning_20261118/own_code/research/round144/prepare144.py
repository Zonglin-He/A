"""Source-only height reference; target input quantile, no model or task labels."""
from common144 import *
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0,str(R/'crossplatform/round7'))
from camera_projection import raw_to_model_points

def main():
 check();reg=rd(H/'INPUT_REGISTRY_PRIVATE.json');srcrows=rd(R/'research/round140/SOURCE_RAW_GEOMETRY_METADATA_ROWS_PRIVATE.json');source={s:float(np.median([x['z10'] for x in srcrows if x['source']==s])) for s in ['waymo','drone','quad']};rows=[];cache={};nearmax=0.;nr=0
 for e in reg:
  src,tgt,sid=e['source'],e['target'],e['row']['id'];name=src+'_to_'+tgt
  if not e['has_B']:rows.append({'source':src,'target':tgt,'id_PRIVATE':sid,'has_B':False,'delta':0.,'q10':None,'source_reference':source[src]});continue
  a=e['A_sensor'];assert a['eligible_for_extra_media'] and not a['protected_image_content_alias'];r=e['row'];z=np.load(R/'crossplatform/round131/formal'/name/'prefix_PRIVATE'/(sid+'.npz'));key=hashlib.sha256(z['point_indices'].tobytes()).hexdigest()+'|'+a['pointpath']+'|'+str(r.get('suffix'))+'|'+tgt
  if key not in cache:
   assert sha(a['pointpath'])==a['point_sha256'];p=Path(a['pointpath']);raw=np.load(p,allow_pickle=False) if p.suffix=='.npy' else np.fromfile(p,np.float32).reshape(-1,4);xyz=raw_to_model_points(raw[z['point_indices'],:3],tgt,int(r.get('suffix') or 0));assert xyz.shape==(16384,3) and np.isfinite(xyz).all();q=float(np.quantile(xyz[:,2],.1));cache[key]=(q,xyz);nr+=1
  q,xyz=cache[key];near=float(cKDTree(xyz).query(z['xyz'][0])[0].max());nearmax=max(nearmax,near);assert near<1e-4;rows.append({'source':src,'target':tgt,'id_PRIVATE':sid,'has_B':True,'q10':q,'delta':source[src]-q,'source_reference':source[src],'max_seed_input_coordinate_error':near})
 dump(H/'POSITION_ROWS_PRIVATE.json',rows);out={'n':len(reg),'eligible_B':sum(e['has_B'] for e in reg),'source_own_median_q10':source,'deduplicated_source_contents':len(srcrows),'target_input_sampledraws':nr,'max_seed_input_coordinate_error':nearmax,'GT_or_model_read':False,'FINAL_access':False};dump(H/'PREPARATION.json',out);dump(H/'POSITION_SEAL.json',{'files':{str(p.relative_to(R)):sha(p) for p in [H/'POSITION_ROWS_PRIVATE.json',H/'PREPARATION.json']}});print(out,flush=True)
if __name__=='__main__':main()
