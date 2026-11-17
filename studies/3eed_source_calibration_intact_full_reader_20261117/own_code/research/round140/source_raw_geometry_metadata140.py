"""Source-only physical-input versus selected-seed lower-envelope diagnostic."""
from common140 import *
from source_geometry_metadata140 import plane
import numpy as np
from scipy.spatial import cKDTree
sys.path[:0]=[str(R/'crossplatform/round7')]
from camera_projection import raw_to_model_points
def main():
 check();reg=rd(R/'research/round132/SOURCE_CALIBRATION_INPUTS_PRIVATE.json');seen=set();rows=[];maxnear=0.
 for e in reg:
  src=e['entry']['source'];a=e['A'];p=e['entry']['files']['prefix'];key=(src,a['content_group_PRIVATE'])
  if key in seen:continue
  seen.add(key);assert a['eligible_for_extra_media'] and not a['protected_image_content_alias'] and sha(R/p['path'])==p['sha256'];r=e['entry']['row'];assert sha(r['pointpath'])==a['point_sha256'];f=Path(r['pointpath']);raw=np.load(f,allow_pickle=False) if f.suffix=='.npy' else np.fromfile(f,np.float32).reshape(-1,4);z=np.load(R/p['path']);xyz=raw_to_model_points(raw[z['point_indices'],:3],src,int(r.get('suffix') or 0));near=float(cKDTree(xyz).query(z['xyz'][0])[0].max());maxnear=max(maxnear,near);assert near<1e-4;rows.append({'source':src,'content_group_PRIVATE':key[1],'input_points':len(xyz),'z10':float(np.quantile(xyz[:,2],.1)),'z50':float(np.median(xyz[:,2])),**plane(xyz)})
 summary=[]
 for src in ['waymo','drone','quad']:
  rr=[r for r in rows if r['source']==src];qq=[r for r in rr if r['qualified']];summary.append({'source':src,'distinct_content_groups':len(rr),'qualified_lower_envelope_plane':len(qq),'median_z10':float(np.median([r['z10'] for r in rr])),'median_z50':float(np.median([r['z50'] for r in rr])),'median_plane_origin_height':float(np.median([r['height_at_origin'] for r in qq])) if qq else None,'median_plane_tilt_degrees':float(np.median([r['tilt_degrees'] for r in qq])) if qq else None,'plane_height10_90':np.quantile([r['height_at_origin'] for r in qq],[.1,.9]).tolist() if qq else []})
 out={'status':'SOURCE_RAW_GEOMETRY_METADATA_ONLY','same_estimator_as_seed_diagnostic':True,'summary':summary,'max_selected_seed_to_original_sample_distance_m':maxnear,'source_only_16384sample_indices_exact':True,'semantic_ground_certified':False,'not_TTA_or_task_efficacy':True,'target_or_GT_read':False,'model_or_point_transform_started':False,'R140_predictor_unchanged':True,'FINAL_access':False};dump(H/'SOURCE_RAW_GEOMETRY_METADATA_ROWS_PRIVATE.json',rows);dump(H/'SOURCE_RAW_GEOMETRY_METADATA_ONLY.json',out);print(out,flush=True)
if __name__=='__main__':main()
