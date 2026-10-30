"""Private fixed5 R12 input-only mask/LiDAR gallery. Root explicit authorization.
No sourceq0/caption/taskscore/GT/registration estimator/model/GPU API.
"""
from pathlib import Path
import ast,hashlib,json,sys,time
import numpy as np
from PIL import Image,ImageDraw,ImageFont
B=Path(__file__).resolve().parents[3];A=B/'research/round12/data_audit';D=Path(__file__).resolve().parent
sys.path.insert(0,str(B/'crossplatform/round7'))
from camera_projection import CameraSpec,raw_to_model_points,project_model_points
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text());MASKS=B/'crossplatform/round12/masks501_attempt001'
COLORS=[(255,70,70),(60,180,255),(230,80,230),(255,200,40),(40,230,100),(150,100,255),(255,150,60),(30,220,220)]

def font(size):
 try:return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',size)
 except OSError:return ImageFont.load_default()

def overlay(rgb,masks,boxes,scores,labels,cats,title,points=None):
 x=rgb.astype(float).copy()
 for j,m in enumerate(masks):x[m]=.70*x[m]+.30*np.asarray(COLORS[j%len(COLORS)])
 im=Image.fromarray(np.clip(x,0,255).astype(np.uint8));draw=ImageDraw.Draw(im)
 for j,(box,score,category) in enumerate(zip(boxes,scores,labels)):
  c=COLORS[j%len(COLORS)];xy=[float(v) for v in box];draw.rectangle(xy,outline=c,width=2);label=f'{j}: {cats[int(category)]} {float(score):.3f} (pred)';anchor=(max(0,int(xy[0])),max(30,int(xy[1])-17));size=draw.textbbox(anchor,label,font=font(13));draw.rectangle(size,fill=(0,0,0));draw.text(anchor,label,fill=c,font=font(13))
 if points is not None:
  uv,inside=points
  for xy,yes in zip(uv,inside):
   x,y=map(int,xy);draw.ellipse((x-1,y-1,x+1,y+1),fill=(0,255,100) if yes else (255,125,0))
 draw.rectangle((0,0,im.width,29),fill=(0,0,0));draw.text((6,5),title,fill=(255,255,255),font=font(15));return im

def visible_projected_points(points,cam,wh):
 pr=project_model_points(points,cam,wh);u=pr['uv'];d=pr['depth'];valid=pr['in_image'];ids=np.flatnonzero(valid)
 pix=np.zeros((len(points),2),np.int64);pix[valid]=np.floor(u[valid]).astype(np.int64)
 key=pix[ids,1]*wh[0]+pix[ids,0];q=points[ids];order=np.lexsort((q[:,2],q[:,1],q[:,0],d[ids],key));ids=ids[order];key=key[order];take=ids[np.r_[True,key[1:]!=key[:-1]]] if len(ids) else ids
 return pix[take],{'raw_points':len(points),'positive_depth':int((d>.1).sum()),'continuous_FOV_points':int(valid.sum()),'nearest_depth_per_pixel_points':len(take),'projection_diagnostics':pr['diagnostics']}

def main():
 planpath=A/'INPUT_GALLERY5_PLAN_LOCK.json';assert sha(planpath)=='6a2694931d8bd577950e49b8ddd11879f848a00608089f6c217e49978e249cea';plan=read(planpath);assert plan['n']==5
 fi=read(A/'FRAME_INDEX_PRIVATE.json');assert sha(A/'FRAME_INDEX_PRIVATE.json')==plan['frame_index_sha256'];assert sha(A/'MASK_FRAME_INPUTS.json')==plan['mask_frame_inputs_sha256']
 sealpath=MASKS/'SEALED.json';assert sha(sealpath)=='faa8b9260f2dc1fa40cbaa9ca13adae48f444121ab88b2a73b1e4c019cfa8852';seal=read(sealpath);assert seal['state']=='sealed_R12_independent_whole_image_instance_masks_no_q0_caption_or_GT' and seal['frame_n']==501
 for name,digest in seal['files'].items():assert sha(MASKS/name)==digest
 source_plan=read(MASKS/'registered_plan.json');assert source_plan['score_threshold']==source_plan['mask_probability_threshold']==.5
 trace={x['frame_alias']:x for x in read(MASKS/'TRACE.json')}
 taxonomy=next(Path(p)/'torchvision/models/_meta.py' for p in sys.path if (Path(p)/'torchvision/models/_meta.py').is_file());tree=ast.parse(taxonomy.read_text());assign=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_COCO_CATEGORIES' for t in n.targets));cats=ast.literal_eval(assign.value);assert hashlib.sha256(json.dumps(cats,separators=(',',':')).encode()).hexdigest()==source_plan['COCO_categories_sha256']
 output=D/'gallery5_attempt001';assert not output.exists();output.mkdir();files={};rows=[];start=time.perf_counter()
 auth={'state':'root_explicit_local_render_authorization_materialized','basis':'Parent instruction authorizes actual frozen5 inputpairs, independent wholeimage predictions and LiDAR projection; noGT/score/model/newregistration.','gallery_plan_sha256':sha(planpath),'mask_seal_sha256':sha(sealpath),'renderer_sha256':sha(__file__),'no_registration_transform_available_or_used':True};(output/'AUTHORIZATION.json').write_text(json.dumps(auth,indent=2)+'\n');files['AUTHORIZATION.json']=sha(output/'AUTHORIZATION.json')
 for pick in plan['rows_PRIVATE']:
  assert pick['reference'] is not None;panels=[];observations=[]
  for role,alias in (('anchor',pick['anchor_frame_alias']),('reference',pick['reference']['frame_alias'])):
   frame=fi[alias];tr=trace[alias];image=Path(frame['imagepath']);point=Path(frame['pointpath']);assert sha(image)==frame['image_sha256']==tr['image_sha256'] and sha(point)==frame['point_sha256']
   with Image.open(image) as im:rgb=np.array(im.convert('RGB'))
   H,W=rgb.shape[:2]
   with np.load(MASKS/(alias+'.npz'),allow_pickle=False) as z:boxes=z['boxes_xyxy'].copy();scores=z['scores'].copy();labels=z['class_ids'].copy();masks=z['masks_bool'].copy();original=z['original_hw'].copy()
   assert masks.shape==(len(boxes),H,W) and masks.dtype==bool and np.array_equal(original,[H,W]) and [hashlib.sha256(m.tobytes()).hexdigest() for m in masks]==tr['mask_sha256']
   raw=np.fromfile(point,np.float32).reshape(-1,4)[:,:3];assert np.isfinite(raw).all();model=raw_to_model_points(raw,frame['platform']);fields={'image_intrinsic':frame['image_intrinsic'],'image_extrinsic':frame['image_extrinsic']}
   if frame['camera_distortion'] is not None:fields['camera_distortion']=frame['camera_distortion']
   cam=CameraSpec.from_camera_fields(frame['platform'],fields,missing_distortion_policy='assume_zero_declared');pixels,pdiag=visible_projected_points(model,cam,(W,H));union=np.any(masks,axis=0);inside=union[pixels[:,1],pixels[:,0]] if len(pixels) else np.zeros(0,bool)
   panels.append((overlay(rgb,masks,boxes,scores,labels,cats,f'{role}: all independent instances / predictions, not GT'),overlay(rgb,masks,boxes,scores,labels,cats,f'{role}: LiDAR in-any-mask GREEN / outside ORANGE',points=(pixels,inside))))
   observations.append({'role':role,'frame_alias_PRIVATE':alias,'timestamp_counter':frame['timestamp'],'image_sha256':sha(image),'point_sha256':sha(point),'mask_NPZ_sha256':sha(MASKS/(alias+'.npz')),'mask_instance_n':len(masks),'predicted_class_counts':dict(__import__('collections').Counter(cats[int(x)] for x in labels)),'projected_in_union_mask':int(inside.sum()),'projected_outside_union_mask':int((~inside).sum()),'pixel_mask_union_area':int(union.sum()),**pdiag})
  W,H=panels[0][0].size;assert panels[1][0].size==(W,H);footer=125;page=Image.new('RGB',(2*W,2*H+footer),(20,20,20))
  for col in (0,1):
   page.paste(panels[col][0],(col*W,0));page.paste(panels[col][1],(col*W,H))
  draw=ImageDraw.Draw(page);lines=[pick['pair_alias']+' | '+pick['platform']+f" | provided-counter gap={pick['reference']['absolute_counter_gap']} (units/sync unverified)",'No GT, caption truth, source q0, task score or new registration estimator is used. No inter-frame T applied.', 'Colors/labels are independent COCO predictions; point-in-mask means pixel support only, not correct target/identity.', 'Projection uses continuous FOV then floor + nearest depth per pixel; it is not complete visibility or synchronization proof.']
  for j,line in enumerate(lines):draw.text((8,2*H+8+25*j),line,fill=(240,240,240),font=font(15))
  name=pick['pair_alias']+'.jpg';page.save(output/name,quality=92);files[name]=sha(output/name);rows.append({'pair_alias':pick['pair_alias'],'platform':pick['platform'],'reference_role':pick['reference_role'],'counter_gap':pick['reference']['absolute_counter_gap'],'observations_PRIVATE':observations,'no_crossframe_transform_or_identity_proof':True})
 receipt={'state':'private_R12_fixed5_input_mask_LiDAR_gallery_rendered_no_GT_scores','n_pairs':5,'n_observations':10,'renderer_sha256':sha(__file__),'gallery_plan_sha256':sha(planpath),'mask_seal_sha256':sha(sealpath),'frame_index_sha256':sha(A/'FRAME_INDEX_PRIVATE.json'),'all_image_point_NPZ_and_perinstance_mask_hashes_verified':True,'no_caption_sourceq0_taskscore_GT_model_GPU_newregistration_used':True,'pose_and_clock_units_sync_not_verified_by_drawing':True,'rows_PRIVATE':rows,'seconds':time.perf_counter()-start};(output/'RECEIPT_PRIVATE.json').write_text(json.dumps(receipt,indent=2)+'\n');files['RECEIPT_PRIVATE.json']=sha(output/'RECEIPT_PRIVATE.json');(output/'SEALED.json').write_text(json.dumps({'state':'sealed_private_R12_fixed5_input_observation_gallery','n_pairs':5,'files':files,'renderer_sha256':sha(__file__),'plan_sha256':sha(planpath),'mask_seal_sha256':sha(sealpath)},indent=2)+'\n');print(json.dumps({'n':5,'output':str(output),'seal_sha256':sha(output/'SEALED.json')}))
if __name__=='__main__':main()
