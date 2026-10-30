"""DEV post-seal descriptive region/visibleface diagnostics, no prediction generation."""
from pathlib import Path
import json,hashlib,sys,math
import numpy as np
R=Path(__file__).resolve().parent;B=R.parents[2];P=B/'crossplatform/round4';S=P/'source';D=P/'adaptation/dev_predictions'
sys.path.insert(0,str(R));from candidate_decomposition import rot
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def arr(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def main():
 out=R/'DEV_TEACHER_QUALITY_SUPPLEMENT_PRIVATE.json';assert not out.exists();assert sha(D/'SEALED.json')=='dbad6a7d85f5835bc0d114c234862059d8c094fd68acfdb4320e6239edf857d8';se=read(D/'SEALED.json')
 for n,h in se['files'].items():assert sha(D/n)==h
 for n,h in se['diagnostic_files'].items():assert sha(D/'diagnostics'/n)==h
 tr={x['id']:x for x in read(D/'trace.json')};cams={x['id']:x for x in read(S/'calibration/CORRECTION/dev_adapter_manifest.json')}
 for c in cams.values():assert sha(c['camera_path'])==c['camera_sha256']
 rows=[];axes=[]
 for s in read(S/'dev_scoring_index_PRIVATE.json'):
  sid=s['id'];t=tr[sid];d=arr(D/'diagnostics'/(sid+'_scores.npz'));ref=arr(S/'dev_source'/(sid+'_v0_full.npz'));v=arr(R/'dev_decomposition_v1'/(sid+'_GT_IOU_PRIVATE.npz'))['fixedPCA'];q=int(ref['semantic_scores'].argmax());best=int(v.argmax());base=ref['boxes_pca'][q];meta=read(s['meta_path']);gt=np.asarray(meta['ground_info'][s['object_index']]['bbox_3d'][:7],float)
  if s['platform']=='drone':gt[2]+=1.8
  if s['platform']=='waymo':a=math.radians([0,-45,45,-90,90][s['waymo_view_suffix']]);gt[:3]=rot(a)@gt[:3];gt[6]+=a
  camera=arr(cams[sid]['camera_path']);x=(camera['source_to_camera']@np.r_[gt[:3],1])[:3];k=camera['image_intrinsic'][:3,:3];dist=camera['camera_distortion'];assert not np.any(dist),'Supplement uses pinholeonly; stop if actualnonzeroD';uv=(k@x)[:2]/x[2];size=camera['image_size_wh'];valid=x[2]>.1 and (uv>=0).all() and (uv<size).all();region=t['caption_region'];inside=None if not region or not valid else bool(region['low']<=uv[0]/size[0]<=region['high']);local=(gt[:3]-base[:3])@rot(base[6]);viol=lambda z,j:max(d['geometry_interval_lower'][j]-z,z-d['geometry_interval_upper'][j],0)
  for a in t['geometry_teacher']['axes']:
   if a['accepted']:
    j=a['axis'];axes.append({'id':sid,'platform':s['platform'],'scene':s['scene'],'axis':j,'near_face_sign':a['camera_near_face_sign'],'normal_support':a['support'],'tangent_coverage':a['tangent_coverage'],'MAD_m':a['MAD_m'],'source_interval_distance_m':viol(0,j),'GTcenter_interval_distance_m':viol(local[j],j),'GTcenter_inside_interval':bool(viol(local[j],j)==0),'GTcenter_interval_confounded_by_selectedinstance_size_yaw':True})
  rows.append({'id':sid,'platform':s['platform'],'parsed':region is not None,'GTcentroid_projectable':bool(valid),'GTcentroid_inside_claimed_horizontal_region':inside,'source_selected_iou':float(v[q]),'best256_iou':float(v[best]),'rankrecoverable50':bool(v[q]<=.5 and v[best]>.5),'source_selected_spatial_penalty':float(d['spatial_penalty'][q]),'best256_spatial_penalty':float(d['spatial_penalty'][best]),'best256_penalty_lower_than_source':bool(d['spatial_penalty'][best]<d['spatial_penalty'][q]-1e-12),'best256_penalty_equal_source':bool(abs(d['spatial_penalty'][best]-d['spatial_penalty'][q])<=1e-12)})
 summary=[]
 for p in ['waymo','drone','quad']:
  rr=[x for x in rows if x['platform']==p];aa=[x for x in axes if x['platform']==p];recover=[x for x in rr if x['rankrecoverable50']];summary.append({'platform':p,'n':len(rr),'parsed':sum(x['parsed'] for x in rr),'GTprojectable':sum(x['GTcentroid_projectable'] for x in rr),'GTprojectable_and_parsed':sum(x['GTcentroid_inside_claimed_horizontal_region'] is not None for x in rr),'GTcentroid_inside_claimed_region':sum(x['GTcentroid_inside_claimed_horizontal_region'] is True for x in rr),'rankrecoverable50_n':len(recover),'rankrecoverable50_with_lower_best_penalty':sum(x['best256_penalty_lower_than_source'] for x in recover),'rankrecoverable50_with_equal_best_penalty':sum(x['best256_penalty_equal_source'] for x in recover),'accepted_normal_axes':len(aa),'accepted_local_horizontal_axes':sum(x['axis']!=2 for x in aa),'accepted_local_vertical_axes':sum(x['axis']==2 for x in aa),'GTcenter_inside_intervals':sum(x['GTcenter_inside_interval'] for x in aa)})
 out.write_text(json.dumps({'scope':'AlreadysealedDEVonly posthoc descriptive teacherdiagnosis, no methodchange/confirmGT','script_sha256':sha(__file__),'summaries':summary,'rows':rows,'accepted_axes':axes,'boundaries':['GT centroid inside broad region is constraint feasibility not ranking discrimination','candidatebestGT IoU doesnot prove exactidentity','faceintervalGTcenter mismatch can reflect crop contamination, wronginstance, frozen size/yaw, orface assumption','No new teacher/scoredprediction, GTcenter projection uses auditedcorrectedcamera withzeroD']},indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
