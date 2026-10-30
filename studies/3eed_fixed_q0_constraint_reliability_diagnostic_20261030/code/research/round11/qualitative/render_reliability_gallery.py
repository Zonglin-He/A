"""Private R11 reliability gallery with exact sealed A/B mask reconstruction.
Execute only with root media release; no GT/taskscore/model/GPU API.
"""
from pathlib import Path
import argparse,dataclasses,hashlib,json,sys,time
import numpy as np
from PIL import Image,ImageDraw
B=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(B/'crossplatform/round11'),str(B/'crossplatform/round10'),str(B/'crossplatform/round7')]
from reliability import physical_partitions,ReliabilityConfig
from rgb_support import make_rgb_support,RGBSupportConfig
from camera_projection import CameraSpec,read_camera_allowlist,project_model_points
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
masksha=lambda m:hashlib.sha256(np.ascontiguousarray(m).tobytes()).hexdigest()
EDGES=tuple((i,i^(1<<j)) for i in range(8) for j in range(3) if not i&(1<<j))
def corners(b):return b[:3]+np.array([[(-1 if i&(1<<j)==0 else 1) for j in (2,1,0)] for i in range(8)])*b[3:]/2

def panel(rgb,label):
 im=Image.fromarray(rgb).convert('RGB');d=ImageDraw.Draw(im);d.rectangle((0,0,im.width,28),fill=(0,0,0));d.text((5,5),label,fill=(255,255,255));return im

def overlay(rgb,mask,label):
 x=rgb.copy();x[mask]=(x[mask].astype(float)*.45+np.array([255,30,190])*.55).astype(np.uint8);return panel(x,label)

def project_box(rgb,box,camera,label,color):
 im=panel(rgb,label);pr=project_model_points(corners(np.asarray(box,float)),camera,im.size);uv=pr['uv'];z=pr['depth'];draw=ImageDraw.Draw(im)
 for i,j in EDGES:
  if min(z[i],z[j])>.1 and np.isfinite(uv[[i,j]]).all():draw.line([tuple(uv[i]),tuple(uv[j])],fill=color,width=3)
 return im

def main(a):
 assert sha(a.release)==a.release_sha256;grant=read(a.release)
 for key in ('CPU_local_gallery_rendering_authorized','no_GT_or_task_scores_read','prediction_allarms_sealed_before_rendering'):assert grant.get(key) is True,key
 for key,path in [('renderer_sha256',Path(__file__)),('gallery_plan_sha256',a.plan),('prediction_seal_sha256',a.predictions/'SEALED.json'),('source_seal_sha256',a.source/'SEALED.json')]:assert grant[key]==sha(path),key
 plan=read(a.plan);s=read(a.predictions/'SEALED.json');ss=read(a.source/'SEALED.json')
 assert s['state']=='sealed_R11_reliability11_and_atomic6_before_GT' and s['n_samples']==320 and s['n_arms']==11 and s['all_reliability_and_rank_weights_sealed_before_GT'] and s['source_seal_sha256']==sha(a.source/'SEALED.json')
 assert plan['prediction_seal_sha256']==sha(a.predictions/'SEALED.json') and plan['source_seal_sha256']==sha(a.source/'SEALED.json') and plan['renderer_sha256']==sha(__file__) and plan['n']==len(plan['rows_PRIVATE'])<=12
 assert len({r['id'] for r in plan['rows_PRIVATE']})==len({r['case_alias'] for r in plan['rows_PRIVATE']})==plan['n']
 for directory,seal in ((a.predictions,s),(a.source,ss)):
  for name,digest in seal['files'].items():
   relative=Path(name);assert not relative.is_absolute() and '..' not in relative.parts;assert sha(directory/name)==digest
 for name,digest in s['program_sha256'].items():assert sha(B/name)==digest,name
 cfg=read(a.predictions/'registered_plan.json');rc=RGBSupportConfig();rl=ReliabilityConfig();assert cfg['rgb_config']==dataclasses.asdict(rc) and cfg['reliability_config']==dataclasses.asdict(rl)
 manifest=a.predictions/'manifest_input.json';assert sha(manifest)==s['manifest_sha256']==plan['manifest_sha256'];rows={r['id']:r for r in read(manifest)};trace={r['id']:r for r in read(a.predictions/'trace.json')};source_trace={r['id']:r for r in read(a.source/'trace.json')}
 import cv2
 cv2.setNumThreads(1);assert cv2.__version__==s['runtime']['opencv']
 assert not a.output.exists();a.output.mkdir(parents=True);files={};records=[];start=time.perf_counter()
 for pick in plan['rows_PRIVATE']:
  sid=pick['id'];row=rows[sid];t=trace[sid];tr=t['reliability'];assert t['base_actual_change'] and row['platform']==pick['platform'] and row['scene']==pick['scene']
  assert sha(row['imagepath'])==source_trace[sid]['image_sha256'] and sha(row['metapath'])==source_trace[sid]['camera_container_sha256']
  with np.load(a.predictions/(sid+'.npz'),allow_pickle=False) as z:
   names=z['arm_names'].tolist();boxes=z['arm_boxes_raw6'].copy();base=z['RGB_mask2d_private'].copy();q=int(z['q_source']);rd=bool(z['case_defined']);rv=float(z['case_reliability']);status=str(z['association_status'])
  assert rd==pick['case_defined'] and (not rd or rv==pick['case_r'])
  if 'mask_sha256' in t['RGB']:assert masksha(base)==t['RGB']['mask_sha256']
  with np.load(a.source/'observations'/(sid+'.npz'),allow_pickle=False) as z:pc=z['sampled_points_model'].copy();raw0=z['raw_indices'].copy()
  p,first=np.unique(pc,axis=0,return_index=True);raw=raw0[first];p,raw,role,partition=physical_partitions(p,raw,sid,rl);assert partition==tr['partition_hashes']
  with Image.open(row['imagepath']) as im:rgb=np.array(im.convert('RGB'))
  assert base.shape==rgb.shape[:2] and base.dtype==bool
  camera=CameraSpec.from_camera_fields(row['platform'],read_camera_allowlist(row['metapath']),waymo_view_suffix=0,missing_distortion_policy='assume_zero_declared');camera.validated()
  expected=tr['features']['partition_segmentation'];assert len(expected)==2
  segments=[]
  for half in (0,1):
   seg=make_rgb_support(rgb=rgb,points_fit=p[role==half],source_box6=boxes[0],camera_spec=camera,config=rc)
   assert masksha(seg['mask2d'])==expected[half]['mask_sha256'] and seg['valid']==expected[half]['valid'],('Sealed partition mask mismatch',pick['case_alias'],half)
   segments.append(seg)
  labels=['Source q0 (yellow)','Original full G (green)','Reliability-weighted G (cyan)'];armkeys=['source','unconditional_full','weighted_constraints'];colors=[(255,220,0),(30,255,60),(0,220,255)]
  panels=[panel(rgb,'Original RGB; same image/q0 shared'),overlay(rgb,base,'Whole-observation base mask'),overlay(rgb,segments[0]['mask2d'],'A-mask; sealed SHA verified'),overlay(rgb,segments[1]['mask2d'],'B-mask; sealed SHA verified')]
  panels.extend(project_box(rgb,boxes[names.index(k)],camera,label,color) for k,label,color in zip(armkeys,labels,colors))
  H,W=rgb.shape[:2];info=Image.new('RGB',(W,H),(20,20,20));d=ImageDraw.Draw(info);ms=tr['features'].get('mask_stability',{});text=[pick['case_alias']+' | '+pick['selection_role'],'status='+status,'effective case r='+str(rv if rd else 'NA'),'raw case r='+str(tr.get('raw_case_reliability')),'A/B IoU='+str(ms.get('IoU')),'mask defined='+str(ms.get('defined')),'RGB r='+str(tr['RGB_weight'])+' defined='+str(tr['RGB_weight_defined']),'A/B physical points='+str(partition['partition_unique_XYZ_counts']),'A/B mask pixels='+str([int(x['mask2d'].sum()) for x in segments]),'Shared RGB, q0, ROI and source backbone.','Agreement is not identity correctness.','No GT/task score or benefit selection.']
  for j,line in enumerate(text):d.text((8,8+20*j),line,fill=(235,235,235))
  panels.append(info);footer=120;gallery=Image.new('RGB',(4*W,2*H+footer),(20,20,20))
  for j,im in enumerate(panels):gallery.paste(im,((j%4)*W,(j//4)*H))
  draw=ImageDraw.Draw(gallery);caption=row['caption'];width=max(60,int(4*W/7));wrapped=[caption[k:k+width] for k in range(0,len(caption),width)];draw.text((8,2*H+5),'PRIVATE caption: '+'\n'.join(wrapped[:4]),fill=(255,255,255));draw.text((8,2*H+90),'Postseal reliability-extreme/undefined diagnostic; shared source can be stably wrong. No semantic identity or task accuracy claim.',fill=(240,210,120))
  name=pick['case_alias']+'.jpg';gallery.save(a.output/name,quality=90);files[name]=sha(a.output/name);records.append({'case_alias':pick['case_alias'],'id_PRIVATE':sid,'selection_role':pick['selection_role'],'platform':row['platform'],'defined_r':rd,'effective_r':rv if rd else None,'association_status':status,'partition_mask_sha256':[masksha(x['mask2d']) for x in segments],'both_A_B_masks_equal_sealed_trace':True,'image_sha256':sha(row['imagepath'])})
 receipt={'state':'private_R11_gallery_exact_AB_masks_verified_no_GT_scores','n':len(records),'prediction_seal_sha256':sha(a.predictions/'SEALED.json'),'source_seal_sha256':sha(a.source/'SEALED.json'),'gallery_plan_sha256':sha(a.plan),'release_sha256':sha(a.release),'renderer_sha256':sha(__file__),'all_recomputed_partition_masks_SHA_bitexact_sealed':True,'no_GT_task_scores_model_GPU_or_methodchanges':True,'selection_counts':plan['by_platform_counts'],'rows_PRIVATE':records,'seconds':time.perf_counter()-start}
 (a.output/'RECEIPT_PRIVATE.json').write_text(json.dumps(receipt,indent=2)+'\n');files['RECEIPT_PRIVATE.json']=sha(a.output/'RECEIPT_PRIVATE.json');(a.output/'SEALED.json').write_text(json.dumps({'state':'sealed_private_R11_reliability_gallery','n':len(records),'files':files,'plan_sha256':sha(a.plan),'release_sha256':sha(a.release)},indent=2)+'\n');print(json.dumps({'n':len(records),'seal_sha256':sha(a.output/'SEALED.json')}))
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for k in ('release','plan','predictions','source','output'):p.add_argument('--'+k,type=Path,required=True)
 p.add_argument('--release-sha256',required=True);main(p.parse_args())
