import sys,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'crossplatform/round25'),str(ROOT/'research/round5/audit')]
from common25 import *
from rgb25 import project
from association import box_corners,rect_iou
from camera_projection import project_model_points
from independent_core_v2 import candidate_ious
from PIL import Image
rr=rows('S3682')[:32];gt=read(EXP.parent/'input/SOURCE_GT_PRIVATE.json');records=[]
for r in rr:
 z=load_npz(BASE/'pilot32_source_attempt002/samples_PRIVATE'/(r['id']+'.npz'));rgb=load_npz(BASE/'pilot32_RGB_attempt002/samples_PRIVATE'/(r['id']+'.npz'));im=Image.open(r['imagepath']);_,_,_,sp=project(r,z,im.size);b=np.array(gt[r['id']],np.float32);R=np.array([[np.cos(b[6]),-np.sin(b[6]),0],[np.sin(b[6]),np.cos(b[6]),0],[0,0,1]]);corners=(box_corners(b[:6])-b[:3])@R.T+b[:3];p=project_model_points(corners,sp,im.size);m=read(r['metapath']);observed=m['ground_info'][r['object_index']]['bbox_2d_proj'];rect=np.r_[p['uv'].min(0),p['uv'].max(0)];v=candidate_ious(z['boxes'],b)[0];records.append({'alias':r['alias'],'projected_GT_vs_dataset_2D_iou':rect_iou(rect,observed),'candidate_oracle_iou':float(v.max()),'source_iou':float(v[z['scores'].argmax()]),'best_candidate_has_RGB':bool(rgb['valid'][v.argmax()]),'valid_RGB_candidates':int(rgb['valid'].sum())})
result={'n':32,'engineering_pilot_not_performance_selection':True,'median_GT_projection_2D_IoU':float(np.median([r['projected_GT_vs_dataset_2D_iou'] for r in records])),'min_GT_projection_2D_IoU':min(r['projected_GT_vs_dataset_2D_iou'] for r in records),'oracle_candidate_RGB_coverage':sum(r['best_candidate_has_RGB'] for r in records),'records':records,'point_feature_seconds':read(BASE/'pilot32_source_attempt002/SEALED.json')['seconds'],'RGB_feature_seconds':read(BASE/'pilot32_RGB_attempt002/SEALED.json')['seconds']};dump(EXP/'PILOT_REPAIRED_AUDIT.json',result);assert result['median_GT_projection_2D_IoU']>.9;print({k:v for k,v in result.items() if k!='records'})
