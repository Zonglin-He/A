"""Source-only offline projection audit, independent of learning and ROI masks."""
import sys,numpy as np,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'crossplatform/round25'),str(ROOT/'crossplatform/round13'),str(ROOT/'crossplatform/round7')]
from common25 import *
from camera_projection import *
from association import box_corners,rect_iou
from PIL import Image

def main():
 gt=read(EXP.parent/'input/SOURCE_GT_PRIVATE.json');records=[]
 for split in ['S3682','SV512']:
  for r in rows(split):
   m=read(r['metapath']);fields={k:m[k] for k in CAMERA_KEYS if k in m};sp=CameraSpec.from_camera_fields('waymo',fields,waymo_view_suffix=r['suffix'],missing_distortion_policy='assume_zero_declared');b=np.array(gt[r['id']],np.float32);R=z_rotation(np.rad2deg(b[6]));corners=(box_corners(b[:6])-b[:3])@R.T+b[:3];p=project_model_points(corners,sp,Image.open(r['imagepath']).size);okay=bool(np.isfinite(p['uv']).all() and (p['depth']>.1).all());iou=rect_iou(np.r_[p['uv'].min(0),p['uv'].max(0)],m['ground_info'][r['object_index']]['bbox_2d_proj']) if okay else None;records.append({'alias':r['alias'],'source_split':split,'view_suffix':r['suffix'],'projectable_GT':okay,'GT_projection_2D_IoU':iou})
 summary={}
 for view in range(5):
  r=[x for x in records if x['view_suffix']==view];v=[x['GT_projection_2D_IoU'] for x in r if x['projectable_GT']];summary[str(view)]={'n':len(r),'projectable':len(v),'median_IoU':float(np.median(v)) if v else None,'below05':sum(x<.5 for x in v)}
 dump(EXP/'SOURCE_CAMERA_AUDIT.json',{'n':len(records),'views':summary,'source_GT_used_only_for_offline_projection_check':True,'target_GT_read':False,'records':records});print(summary)
 assert all(x['median_IoU']>.9 for x in summary.values())
if __name__=='__main__':main()
