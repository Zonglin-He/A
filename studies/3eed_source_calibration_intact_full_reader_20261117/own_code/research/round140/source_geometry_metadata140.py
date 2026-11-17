"""Prospective source-coordinate diagnostic only; no model, target input or labels."""
from common140 import *
import numpy as np,collections
def plane(xyz):
 x=np.asarray(xyz,float);x=x[np.isfinite(x).all(1)];rr=np.linalg.norm(x[:,:2],axis=1);x=x[(rr>=5)&(rr<=40)];bins=np.floor(x[:,:2]/4).astype(int);cells=[]
 for xy in sorted(set(map(tuple,bins))):
  q=x[(bins==xy).all(1)]
  if len(q)>=3:
   z=np.quantile(q[:,2],.1);ids=np.argsort(abs(q[:,2]-z))[:max(1,len(q)//5)];cells.append(q[ids].mean(0))
 if len(cells)<12:return {'qualified':False,'cells':len(cells)}
 c=np.array(cells);A=np.c_[c[:,:2],np.ones(len(c))];coef=np.linalg.lstsq(A,c[:,2],rcond=None)[0]
 for _ in range(8):
  err=A@coef-c[:,2];med=np.median(err);scale=max(.05,float(np.median(abs(err-med)))*1.4826);w=np.minimum(1,1.345*scale/np.maximum(abs(err),1e-10));coef=np.linalg.lstsq(A*w[:,None]**.5,c[:,2]*w**.5,rcond=None)[0]
 residual=abs(A@coef-c[:,2]);tilt=float(np.degrees(np.arctan(np.linalg.norm(coef[:2]))));mad=float(np.median(residual));return {'qualified':bool(tilt<=30 and mad<=.35),'cells':len(cells),'height_at_origin':float(coef[2]),'tilt_degrees':tilt,'residual_median_m':mad,'residual90_m':float(np.quantile(residual,.9))}
def main():
 check();reg=rd(R/'research/round132/SOURCE_CALIBRATION_INPUTS_PRIVATE.json');seen=set();rows=[]
 for e in reg:
  src=e['entry']['source'];p=e['entry']['files']['prefix'];key=(src,e['A']['content_group_PRIVATE'])
  if key in seen:continue
  seen.add(key);assert e['A']['eligible_for_extra_media'] and not e['A']['protected_image_content_alias'] and sha(R/p['path'])==p['sha256'];x=np.load(R/p['path'])['xyz'][0];rows.append({'source':src,'content_group_PRIVATE':key[1],'z10':float(np.quantile(x[:,2],.1)),'z50':float(np.median(x[:,2])),'z90':float(np.quantile(x[:,2],.9)),**plane(x)})
 summary=[]
 for src in ['waymo','drone','quad']:
  rr=[r for r in rows if r['source']==src];qq=[r for r in rr if r['qualified']];summary.append({'source':src,'distinct_content_groups':len(rr),'qualified_lower_envelope_plane':len(qq),'median_z10':float(np.median([r['z10'] for r in rr])),'median_z50':float(np.median([r['z50'] for r in rr])),'median_plane_origin_height':float(np.median([r['height_at_origin'] for r in qq])) if qq else None,'median_plane_tilt_degrees':float(np.median([r['tilt_degrees'] for r in qq])) if qq else None,'plane_height10_90':np.quantile([r['height_at_origin'] for r in qq],[.1,.9]).tolist() if qq else []})
 dump(H/'SOURCE_GEOMETRY_METADATA_ROWS_PRIVATE.json',rows);out={'status':'SOURCE_GEOMETRY_METADATA_ONLY','summary':summary,'algorithm':'4m XYcells range5to40m >=3seeds lower10percent envelope, eightHuberweightedLS, qualified if>=12cells tilt<=30deg medianresidual<=.35m','source_own_cached_coordinates_only':True,'semantic_ground_identity_certified':False,'not_task_or_TTA_efficacy_evidence':True,'current_R140_predictor_unchanged':True,'target_inputs_or_GT_read':False,'source_GT_read':False,'new_model_or_point_transform':False,'FINAL_access':False};dump(H/'SOURCE_GEOMETRY_METADATA_ONLY.json',out);print(out,flush=True)
if __name__=='__main__':main()
