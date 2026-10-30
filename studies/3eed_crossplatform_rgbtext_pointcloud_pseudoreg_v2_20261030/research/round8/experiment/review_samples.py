"""Private R8 unlabeled review rendering primitives (no model, GT or scoring).

No data files are opened on import. The actual sealed-driver output adapter is
bound only after its schema is supplied; this module does not guess NPZ fields or
search annotation files. Rendered figures are private evidence, not public assets.
"""
from pathlib import Path
import argparse,hashlib,json,math,sys,time
import numpy as np
from PIL import Image,ImageDraw,ImageFont

REVIEW_SALT='R8_unlabelled_visual_review_v1'
BOX_COLORS={'source':(255,80,60),'q0':(255,80,60),'q1':(255,210,30),'pseudo':(40,240,255)}
GROUP_COLORS=((70,235,100),(230,100,245),(255,170,40),(80,150,255),(255,100,160),(180,235,50))
EDGES=((0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7))

def fixed_hash_review_selection(rows,per_scene=1,salt=REVIEW_SALT):
 """Input-only hash selection: never accepts IoU/GT/success as a selector."""
 if per_scene<1:raise ValueError('per_scene must be positive')
 selected=[]
 for platform,scene in sorted({(r['platform'],r['scene']) for r in rows}):
  pool=[r for r in rows if (r['platform'],r['scene'])==(platform,scene)]
  key=lambda r:(hashlib.sha256((salt+'|'+r['id']).encode()).hexdigest(),r['id'])
  seen=set()
  for r in sorted(pool,key=key):
   if r['pointpath'] in seen:continue
   selected.append(r);seen.add(r['pointpath'])
   if len(seen)==per_scene:break
  if len(seen)!=per_scene:raise ValueError('Insufficient distinct frame paths')
 return selected

def _rgb(image):
 if not isinstance(image,Image.Image):raise TypeError('Renderer requires an explicitly decoded PIL image')
 return image.convert('RGB')

def _source_box_corners(box6):
 b=np.asarray(box6,dtype=np.float64)
 if b.shape!=(6,) or not np.isfinite(b).all() or np.any(b[3:]<=0):raise ValueError('Finite native yaw0 positive-size source-coordinate box6 required')
 signs=np.array([[-1 if i&4==0 else 1,-1 if i&2==0 else 1,-1 if i&1==0 else 1] for i in range(8)])
 return b[:3]+signs*b[3:]/2

def _project_box(box6,project):
 result=project(_source_box_corners(box6));uv=np.asarray(result['uv'],float);depth=np.asarray(result['depth'],float)
 if uv.shape!=(8,2) or depth.shape!=(8,):raise ValueError('Projection callback must return uv[8,2] and depth[8]')
 return uv,depth

def _point_overlay(image,uv,valid,max_points=2500):
 uv=np.asarray(uv,float);valid=np.asarray(valid,bool)
 if uv.ndim!=2 or uv.shape[1]!=2 or valid.shape!=(len(uv),):raise ValueError('Expected point uv/valid arrays')
 draw=ImageDraw.Draw(image);ids=np.flatnonzero(valid&np.isfinite(uv).all(1))
 # Deterministic visual subsampling only; it never changes model/G point support.
 if len(ids)>max_points:ids=ids[np.linspace(0,len(ids)-1,max_points,dtype=np.int64)]
 for i in ids:
  x,y=map(float,uv[i]);draw.ellipse((x-1,y-1,x+1,y+1),fill=(95,220,110))
 return len(ids)

def _draw_box(image,box6,project,label,color):
 uv,depth=_project_box(box6,project);draw=ImageDraw.Draw(image);front=(depth>.1)&np.isfinite(uv).all(1);edges=0
 for a,b in EDGES:
  if front[a] and front[b]:draw.line([tuple(uv[a]),tuple(uv[b])],fill=color,width=3);edges+=1
 inside=front&(uv[:,0]>=0)&(uv[:,0]<image.width)&(uv[:,1]>=0)&(uv[:,1]<image.height)
 if inside.any():
  p=uv[np.flatnonzero(inside)[0]];draw.text((int(p[0])+3,int(p[1])+3),label,fill=color)
 return {'front_corners':int(front.sum()),'corners_in_image':int(inside.sum()),'front_edge_segments_drawn':edges,'partial_box_if_crossing_camera_plane':not bool(front.all())}

def _rect(draw,box,color,width):
 a=np.asarray(box,dtype=float)
 if a.shape!=(4,) or not np.isfinite(a).all():raise ValueError('Invalid explicit pixelROI')
 if a[2]<=a[0] or a[3]<=a[1]:raise ValueError('Empty pixelROI')
 draw.rectangle(tuple(a),outline=color,width=width)

def render_review(image,*,point_projection,project_source_points,source_box6,q1_box6,pseudo_box6,groups,caption,case_label='private_case',status=None):
 """Return PILfigure + nonGT engineeringreceipt, no file IO.

 `project_source_points(Nx3)` uses the frozen camera bridge (Drone modelz undone,
 Quad rawXYZ), returns uv/depth in original RGBpixels. The caller supplies original
 points' projection; this function never invents a camera convention.
 `groups` is an explicitly adapted list: group_id/members/tight_xyxy/context_xyxy,
 plus optional semantic_tight/semantic_context/semantic_combined/support_uv.
 q0 is exactly source; q1 is a semantic-selected proposal, never declared correct.
 No annotation, GTbox, class label or taskmetric field is accepted or read.
 """
 image=_rgb(image);geometry=image.copy();rois=image.copy();draw=ImageDraw.Draw(rois)
 count=_point_overlay(geometry,point_projection['uv'],point_projection['in_image'])
 box_stats={}
 for name,b in [('source/q0',source_box6),('q1',q1_box6),('pseudo',pseudo_box6)]:
  if b is not None:box_stats[name]=_draw_box(geometry,b,project_source_points,name,BOX_COLORS['source' if name=='source/q0' else name])
 cropped=[];group_stats=[]
 for i,g in enumerate(groups):
  color=GROUP_COLORS[i%len(GROUP_COLORS)];gid=int(g['group_id']);tight=g['tight_xyxy'];context=g['context_xyxy']
  _rect(draw,tight,color,3);_rect(draw,context,color,1);draw.text((max(0,int(tight[0])),max(0,int(tight[1])-12)),f'G{gid}',fill=color)
  if 'support_uv' in g:
   for p in np.asarray(g['support_uv'],float):
    if np.isfinite(p).all() and 0<=p[0]<image.width and 0<=p[1]<image.height:draw.ellipse((p[0]-1,p[1]-1,p[0]+1,p[1]+1),fill=color)
  values={k:float(g[k]) for k in ('semantic_tight','semantic_context','semantic_combined') if k in g}
  if not all(math.isfinite(v) for v in values.values()):raise ValueError('Nonfinite semantic similarity')
  for role,bounds in [('tight',tight),('context',context)]:
   x0,y0,x1,y1=map(int,[np.floor(bounds[0]),np.floor(bounds[1]),np.ceil(bounds[2]),np.ceil(bounds[3])]);x0=max(0,x0);y0=max(0,y0);x1=min(image.width,x1);y1=min(image.height,y1)
   if x1<=x0 or y1<=y0:raise ValueError('ROIoutsideimage')
   tile=Image.new('RGB',(240,180),(25,25,25));crop=image.crop((x0,y0,x1,y1));crop.thumbnail((232,142));tile.paste(crop,((240-crop.width)//2,20+(142-crop.height)//2));td=ImageDraw.Draw(tile);td.text((4,3),f'G{gid} {role} {values.get("semantic_"+role,float("nan")):.4f}',fill=color);td.text((4,165),f'members={len(g["members"])}',fill=(235,235,235));cropped.append(tile)
  group_stats.append({'group_id':gid,'member_count':len(g['members']),**values})
 # Resizing below is for private figure layout only, never CLIP/model inputs.
 panel_w=640;panel_h=max(1,round(image.height*panel_w/image.width));geometry=geometry.resize((panel_w,panel_h));rois=rois.resize((panel_w,panel_h))
 cols=5;crop_rows=(len(cropped)+cols-1)//cols;header_h=100;figure=Image.new('RGB',(max(1280,cols*240),header_h+panel_h+crop_rows*180),(15,15,15));fd=ImageDraw.Draw(figure)
 fd.text((10,8),str(case_label)+' | UNLABELLED REVIEW: q1/pseudo are hypotheses, no GT or correctness labels',fill=(240,240,240))
 # Fullcaption is private only; wrap by characters so no sentence is silently dropped.
 caption=str(caption);lines=[caption[i:i+160] for i in range(0,len(caption),160)] or ['']
 for i,line in enumerate(lines[:3]):fd.text((10,27+i*18),line,fill=(225,225,225))
 if len(lines)>3:fd.text((10,81),'[Figure caption display truncated; full literal text retained in private trace]',fill=(255,190,60))
 figure.paste(geometry,(0,header_h));figure.paste(rois,(640,header_h))
 for i,tile in enumerate(cropped):figure.paste(tile,((i%cols)*240,header_h+panel_h+(i//cols)*180))
 return figure,{'state':'private_unlabelled_geometry_ROI_review_rendered_no_task_score','point_overlay_visual_count':count,'full_projected_point_count':len(point_projection['uv']),'box_projection_diagnostics':box_stats,'group_scalar_diagnostics':group_stats,'status':status,'no_GT_or_taskmetric_access':True,'figure_layout_resizing_not_encoder_preprocessing':True,'caption_display_truncated':len(lines)>3,'images_not_public_export_artifacts':True}

def file_sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()

def render_sealed(args):
 """Actual R8 schema adapter; requires root's hash-bound unlabeled render grant.
 Does not import the grounder/CLIP or scoring modules. Metadata is streamed only
 through the established camera allowlist; target annotations are never built.
 """
 B=Path(__file__).resolve().parents[3];r7=B/'crossplatform/round7';sys.path.insert(0,str(r7));sys.path.insert(0,str(r7/'experiment'))
 from camera_projection import read_camera_allowlist,CameraSpec,project_model_points
 from runner_helpers import sample_points,array_sha
 started=time.monotonic();read=lambda p:json.loads(Path(p).read_text())
 release=read(args.release);sealpath=args.predictions/'SEALED.json';seal=read(sealpath)
 assert release['unlabelled_render_authorized'] is True and release['no_GT_or_taskmetric_selection'] is True
 assert release['manifest_sha256']==file_sha(args.manifest)==seal['manifest_sha256']
 assert release['prediction_seal_sha256']==file_sha(sealpath)
 assert release['review_program_sha256']==file_sha(__file__) and release['per_scene']==args.per_scene and release['review_salt']==REVIEW_SALT
 assert seal['state']=='sealed_R8_strictFP32_crossdomain_predictions_no_target_GT_scoring' and seal['n_arms']==19
 assert seal['all_19arms_sealed_before_any_target_GT_scoring'] and seal['grounder_restored_and_CLIP_frozen']
 assert seal['program_sha256_base']=='workspace_root'
 for rel,digest in seal['program_sha256'].items():
  source=B/rel;assert source.resolve().is_relative_to(B) and file_sha(source)==digest
 rows=read(args.manifest);assert len(rows)==seal['n_samples'];chosen=fixed_hash_review_selection(rows,args.per_scene)
 tracepath=args.predictions/'trace.json';assert file_sha(tracepath)==seal['files']['trace.json'];traces={r['id']:r for r in read(tracepath)}
 assert not args.output.exists();prepared=[]
 for number,row in enumerate(chosen,1):
  tr=traces[row['id']];npzpath=args.predictions/(row['id']+'.npz');assert file_sha(npzpath)==seal['files'][npzpath.name]
  for key,tracekey in [('pointpath','raw_point_file_sha256'),('imagepath','image_file_sha256'),('metapath','camera_metadata_file_sha256')]:assert file_sha(row[key])==tr[tracekey]
  points,indices,raw_n=sample_points(row);assert array_sha(points)==tr['point_sample_sha256'] and array_sha(indices)==tr['raw_indices_sha256']
  with Image.open(row['imagepath']) as im:image=im.convert('RGB')
  fields=read_camera_allowlist(row['metapath']);camera=CameraSpec.from_camera_fields(row['platform'],fields,waymo_view_suffix=0,missing_distortion_policy='assume_zero_declared')
  project=lambda p:project_model_points(p,camera,image.size);projection=project(points)
  with np.load(npzpath,allow_pickle=False) as z:
   assert z['point_sample_sha256'].tobytes().hex()==array_sha(points) and z['image_file_sha256'].tobytes().hex()==tr['image_file_sha256']
   modes=z['modes'].tolist();mode_index=modes.index(args.mode);q0=int(z['q_source']);q1=int(z['q_association'][mode_index]);source=z['source_boxes_raw6'].copy();pseudo=z['pseudo_target6'][mode_index].copy()
   assert q0==tr['q_source0'] and q1==tr['G_modes'][args.mode]['associated_q1']
   assert source.shape==(256,6) and z['roi_xyxy'].shape==(len(z['roi_group_ids']),2,4)
   byid={int(g['group_id']):g for g in tr['groups']};groups=[]
   for i,gid in enumerate(z['roi_group_ids']):
    g=byid[int(gid)];c=z['semantic_cosines_tight_context'][i];groups.append({'group_id':int(gid),'members':g['members'],'tight_xyxy':z['roi_xyxy'][i,0].tolist(),'context_xyxy':z['roi_xyxy'][i,1].tolist(),'semantic_tight':float(c[0]),'semantic_context':float(c[1]),'semantic_combined':float((c[0]+c[1])*.5)})
   raw_boxes=[source[q0],source[q1],pseudo];display=[];clamps=[]
   for box in raw_boxes:
    b=np.asarray(box,float).copy();clamps.append(int(np.count_nonzero(b[3:]<=0)));b[3:]=np.maximum(b[3:],1e-6);display.append(b)
   figure,receipt=render_review(image,point_projection=projection,project_source_points=project,source_box6=display[0],q1_box6=display[1],pseudo_box6=display[2],groups=groups,caption=row['caption'],case_label='private_case_'+str(number).zfill(2),status=tr['G_modes'][args.mode]['G_diagnostics'])
   receipt.update({'id_PRIVATE':row['id'],'platform':row['platform'],'scene_PRIVATE':row['scene'],'q0':q0,'q1':q1,'mode':args.mode,'r_global_heuristic_not_probability':float(z['r_global'][mode_index]),'raw_nonpositive_dimensions_display_clamped_to1e_minus6':clamps,'camera_diagnostics':projection['diagnostics'],'CLIP_text_receipt':tr['CLIP'].get('text_receipt'),'source_prediction_npz_sha256':file_sha(npzpath)})
   prepared.append((number,figure,receipt))
 args.output.mkdir(parents=True);files={}
 for number,figure,receipt in prepared:
  filename='private_case_'+str(number).zfill(2)+'.png';figure.save(args.output/filename);files[filename]=file_sha(args.output/filename)
  name=Path(filename).with_suffix('.json').name;(args.output/name).write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n');files[name]=file_sha(args.output/name)
 result={'state':'sealed_private_unlabelled_visual_review_not_scored','n':len(prepared),'manifest_sha256':file_sha(args.manifest),'prediction_seal_sha256':file_sha(sealpath),'review_program_sha256':file_sha(__file__),'render_release_sha256':file_sha(args.release),'selection_salt':REVIEW_SALT,'per_scene':args.per_scene,'mode':args.mode,'files':files,'no_GT_or_taskmetric_read_or_selection':True,'no_model_CLIP_GPU_execution':True,'private_caption_images_and_coordinates_never_public_export':True,'seconds':time.monotonic()-started}
 (args.output/'SEALED.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'n':len(prepared),'review_seal_sha256':file_sha(args.output/'SEALED.json'),'output':str(args.output)},indent=2))


def synthetic_check():
 # Exercises only in-memory synthetic RGB and pinhole projection, no data files.
 image=Image.new('RGB',(320,240),(70,80,100))
 def project(p):
  p=np.asarray(p,float);z=p[:,2];uv=np.c_[160+100*p[:,0]/z,120+100*p[:,1]/z];return {'uv':uv,'depth':z,'in_image':(z>.1)&(uv[:,0]>=0)&(uv[:,0]<320)&(uv[:,1]>=0)&(uv[:,1]<240)}
 p=np.array([[0.,0.,5.],[.5,.5,5.]])
 group={'group_id':0,'members':[0,1],'tight_xyxy':[145,105,175,135],'context_xyxy':[135,95,185,145],'semantic_tight':.2,'semantic_context':.18,'semantic_combined':.19}
 figure,receipt=render_review(image,point_projection=project(p),project_source_points=project,source_box6=[0,0,5,1,1,1],q1_box6=[.1,0,5,1,1,1],pseudo_box6=[.05,0,5,1,1,1],groups=[group],caption='synthetic full description',case_label='synthetic')
 assert figure.width>=1280 and receipt['no_GT_or_taskmetric_access'] and len(receipt['box_projection_diagnostics'])==3
 return {'synthetic_in_memory_render_pass':True,'no_targetdata_or_GT_or_GPU_access':True,'actual_driver_output_adapter':'R8 RUNNER_INTERFACE explicit schema; requires real sealed19arms plus root hash-bound unlabeled render grant','review_program_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
if __name__=='__main__':
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--synthetic-check',action='store_true');ap.add_argument('--manifest',type=Path);ap.add_argument('--predictions',type=Path);ap.add_argument('--release',type=Path);ap.add_argument('--output',type=Path);ap.add_argument('--per-scene',type=int,default=1);ap.add_argument('--mode',choices=('rgb_fulltext','pc_only','semantic_shuffle'),default='rgb_fulltext');args=ap.parse_args()
 if args.synthetic_check:print(json.dumps(synthetic_check(),indent=2))
 else:
  for key in ('manifest','predictions','release','output'):
   if getattr(args,key) is None:ap.error('--'+key+' required for actual rendering')
  render_sealed(args)

