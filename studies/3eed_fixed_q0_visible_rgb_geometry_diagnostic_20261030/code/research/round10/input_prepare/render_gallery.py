"""Private fixed10 R10 RGB support gallery; execute only after root render release.
No model, GPU, labels, scorer, GT sidecars, or task-outcome-based selection.
"""
from pathlib import Path
import argparse,hashlib,json,sys,time
import numpy as np
from PIL import Image,ImageDraw,ImageFont
B=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(B/'crossplatform/round7'))
from camera_projection import read_camera_allowlist,CameraSpec,project_model_points
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
ARMS=('source','full_rgb','no_rgb_coordinate','rgb_gated_pointmask_ablation')
COLORS=((255,220,0),(0,255,80),(0,220,255),(255,80,210))
EDGES=tuple((i,i^(1<<bit)) for i in range(8) for bit in range(3) if not(i&(1<<bit)))
def corners(b):
 b=np.asarray(b,float);assert b.shape==(6,) and np.isfinite(b).all()
 return b[:3]+np.array([[(-1 if i&(1<<j)==0 else 1) for j in (2,1,0)] for i in range(8)])*b[3:]/2

def bind(p,digest):assert sha(p)==digest,(str(p),'Hash mismatch')
def verify_seal(directory,seal):
 for name,digest in seal['files'].items():
  path=Path(name);assert not path.is_absolute() and '..' not in path.parts
  bind(directory/path,digest)

def panel(rgb,label,color=(255,255,255)):
 out=Image.fromarray(rgb).convert('RGB');draw=ImageDraw.Draw(out);draw.rectangle((0,0,out.width,26),fill=(0,0,0));draw.text((5,5),label,fill=color);return out

def draw_box(image,box,camera,color):
 projection=project_model_points(corners(box),camera,image.size,min_depth=.1);uv=projection['uv'];depth=projection['depth'];draw=ImageDraw.Draw(image)
 for i,j in EDGES:
  if np.isfinite(uv[[i,j]]).all() and min(depth[i],depth[j])>.1:draw.line([tuple(uv[i]),tuple(uv[j])],fill=color,width=3)

def main(a):
 bind(a.release,a.release_sha256);release=read(a.release)
 for key in ('CPU_local_gallery_rendering_authorized','prediction_sealed_before_media_review','no_GT_or_scoring_access'):assert release.get(key) is True,key
 bind(__file__,release['renderer_sha256']);bind(a.plan,release['gallery_plan_sha256']);bind(a.predictions/'SEALED.json',release['prediction_seal_sha256']);bind(a.source/'SEALED.json',release['source_seal_sha256'])
 plan=read(a.plan);s=read(a.predictions/'SEALED.json');source=read(a.source/'SEALED.json')
 assert s['state']=='sealed_R10_fixedq0_direct_geometry_before_GT' and s['n_samples']==160 and s['n_arms']==4
 assert s['source_seal_sha256']==sha(a.source/'SEALED.json') and s['all_queries_fixed_original_q0'] and s['no_GT_or_parameter_update']
 manifest=a.predictions/'manifest_input.json';bind(manifest,plan['input_manifest_sha256']);rows={r['id']:r for r in read(manifest)}
 verify_seal(a.predictions,s);verify_seal(a.source,source)
 assert plan['n']==10 and len(plan['rows_PRIVATE'])==10 and len({x['case_alias'] for x in plan['rows_PRIVATE']})==10
 assert not a.output.exists();a.output.mkdir(parents=True);start=time.perf_counter();files={};records=[]
 for pick in plan['rows_PRIVATE']:
  row=rows[pick['id']];assert row['platform']==pick['platform'] and row['scene']==pick['scene']
  with np.load(a.predictions/(row['id']+'.npz'),allow_pickle=False) as z:
   boxes=z['arm_boxes_raw6'].copy();mask=z['RGB_mask2d_private'].copy();names=tuple(z['arm_names'].tolist());q=int(z['q_source']);changed=z['actual_changed_axes'].copy()
  assert names==ARMS and mask.dtype==bool and boxes.shape==(4,6)
  with Image.open(row['imagepath']) as im:rgb=np.asarray(im.convert('RGB')).copy()
  assert mask.shape==rgb.shape[:2]
  camera=CameraSpec.from_camera_fields(row['platform'],read_camera_allowlist(row['metapath']),waymo_view_suffix=0,missing_distortion_policy='assume_zero_declared')
  with np.load(a.source/'observations'/(row['id']+'.npz'),allow_pickle=False) as z:p=np.unique(z['sampled_points_model'],axis=0)
  pp=project_model_points(p,camera,(rgb.shape[1],rgb.shape[0]),min_depth=.1);visible=np.asarray(pp['in_image']);u=np.asarray(pp['uv'])[visible]
  p0=panel(rgb,'Original RGB: whole observation, no GT')
  blend=rgb.copy();blend[mask]=(blend[mask].astype(float)*.45+np.array([255,40,180])*.55).astype(np.uint8)
  p1=panel(blend,'GrabCut support mask (fallible q0 prior)');d=ImageDraw.Draw(p1)
  for xy in u[::max(1,len(u)//2500)]:
   x,y=np.floor(xy).astype(int);d.point((int(x),int(y)),fill=(0,255,255))
  panels=[p0,p1]
  for i,name in enumerate(ARMS):
   item=panel(rgb,f'{name}; q={q}; changed={int(changed[i].sum())}',COLORS[i]);draw_box(item,boxes[i],camera,COLORS[i]);panels.append(item)
  w,h=p0.size;footer=110;gallery=Image.new('RGB',(3*w,2*h+footer),(20,20,20))
  for j,item in enumerate(panels):gallery.paste(item,((j%3)*w,(j//3)*h))
  draw=ImageDraw.Draw(gallery);caption=row['caption'];wrapped=[]
  while caption:
   wrapped.append(caption[:max(40,int(3*w/7))]);caption=caption[max(40,int(3*w/7)):]
  draw.text((8,2*h+5),pick['case_alias']+' | '+row['platform']+' | private caption: '+'\n'.join(wrapped[:4]),fill=(255,255,255))
  draw.text((8,2*h+88),'Yellow=source; green=full RGB; cyan=no RGB coordinate; magenta=pointmask ablation. No GT identity inference.',fill=(220,220,220))
  name=pick['case_alias']+'.jpg';gallery.save(a.output/name,quality=90);files[name]=sha(a.output/name)
  records.append({'case_alias':pick['case_alias'],'id_PRIVATE':row['id'],'platform':row['platform'],'scene_PRIVATE':row['scene'],'q0':q,'source_observation_uniqueXYZ':len(p),'RGB_mask_pixels':int(mask.sum()),'changed_axis_counts':changed.sum(1).astype(int).tolist(),'image_sha256':sha(row['imagepath']),'prediction_sha256':sha(a.predictions/(row['id']+'.npz'))})
 receipt={'state':'private_R10_fixedhash10_gallery_rendered_no_GT_or_scoring','n':10,'gallery_plan_sha256':sha(a.plan),'renderer_sha256':sha(__file__),'release_sha256':sha(a.release),'prediction_seal_sha256':sha(a.predictions/'SEALED.json'),'source_seal_sha256':sha(a.source/'SEALED.json'),'media_caption_and_coordinate_panels_private_not_exportable':True,'no_method_change_or_quality_selection':True,'seconds':time.perf_counter()-start,'rows_PRIVATE':records}
 (a.output/'GALLERY_RECEIPT_PRIVATE.json').write_text(json.dumps(receipt,indent=2)+'\n');files['GALLERY_RECEIPT_PRIVATE.json']=sha(a.output/'GALLERY_RECEIPT_PRIVATE.json')
 (a.output/'SEALED.json').write_text(json.dumps({'state':'sealed_private_R10_gallery10_noGT','n':10,'files':files,'plan_sha256':sha(a.plan),'release_sha256':sha(a.release)},indent=2)+'\n')
 print(json.dumps({'n':10,'seal_sha256':sha(a.output/'SEALED.json'),'output':str(a.output)}))
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 for key in ('release','plan','predictions','source','output'):p.add_argument('--'+key,type=Path,required=True)
 p.add_argument('--release-sha256',required=True);main(p.parse_args())
