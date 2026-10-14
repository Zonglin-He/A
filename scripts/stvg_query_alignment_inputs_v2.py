"""Fixed old-eight candidate custody, explicit legal mapping and sparse evidence.

No annotation access, candidate expansion, model inference or label-based choice.
Actual frames are decoded sequentially from the accepted canonical source video.
"""
import hashlib,json,time
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scripts.run_stvg_sparse_r1_v2 import ROOT,OLD,sha,read,write,inference_guard
from scripts.stvg_sparse_r1_prompt_v1 import load_source_case,_temporal_candidates,_spatial_candidates
from scripts.stvg_hierarchical_geometry_v1 import (map_candidate_geometry,map_and_deduplicate_candidates,
 canonical_sha256,build_verified_sparse_anchors,gather_verified_anchor_predictions,
 verified_anchors_for_student_core,action_transform_logits_cxcywh,diagonal_gaussian_log_prob_z,
 ANCHOR_RECEIPT_SCHEMA,run_cpu_contracts)
BASE=ROOT/'artifacts/stvg_hierarchical_evidence_guided_v1/revision001'
OLD_SPARSE=ROOT/'artifacts/stvg_sparse_stvg_r1_semantic_opd_v1'

def roster():return [r for r in read(OLD_SPARSE/'ROSTER_LOCK.json')['records'] if r['P0']]

def catalog(rec):
 case=load_source_case(ROOT/rec['source_case_path']);ids=list(map(int,case['frame_ids']))
 if sha(ROOT/rec['source_case_path'])!=rec['source_case_sha256']:raise ValueError('cold source changed')
 raw=_spatial_candidates(case,[0]);rows=[]
 for r in raw:
  rows.append(dict(candidate_id=r['canonical_id'],boxes_xyxy_normalized=r['tube'].tolist(),
    opaque_geometry_binding={'source_candidate_id':r['source_candidate_id'],'old_geometry_sha256':r['geometry_sha256']}))
 spatial=map_and_deduplicate_candidates(rows,ids,expected_source_case_sha256=rec['source_case_sha256'])
 native=next(r for r in spatial['candidates'] if r['opaque_geometry_binding']['source_candidate_id']=='student_native_baseline')
 temporal=_temporal_candidates(case,tuple(case['video_bounds']),float(case['input']['fps']))
 return case,dict(case_id=rec['case_id'],dataset=rec['dataset'],source_case_sha256=rec['source_case_sha256'],
  frame_ids=ids,spatial=spatial,temporal=temporal,native_mapped_candidate_id=native['candidate_id'],GT_read=False)

def positions(case,branch,left,right):
 ids=np.asarray(case['frame_ids']);n=len(ids);global4=np.rint(np.linspace(0,n-1,4)).astype(int).tolist()
 if branch=='temporal':
  boundary=[int(np.argmin(abs(ids-x))) for r in [left,right] for x in r['interval_frames']]
  wanted=global4+boundary
 else:wanted=list(map(int,case['observation_positions']))[:4]+global4
 unique=list(dict.fromkeys(wanted))
 for p in np.rint(np.linspace(0,n-1,n)).astype(int):
  if len(unique)>=8:break
  if int(p) not in unique:unique.append(int(p))
 return sorted(unique[:8]),global4

def decode(case,poses):
 import cv2
 req=read(OLD/'canonical_inputs'/case['case_id']/'REQUEST.json');video=Path(req['video_path'])
 if sha(video)!=req['canonical_video_sha256'] or req['source_video_sha256']!=case['input']['video_sha256']:
  raise ValueError('canonical source custody mismatch')
 fmap={r['physical_frame_id']:r['local_frame_index'] for r in req['frame_map']}
 wanted={fmap[int(case['frame_ids'][p])]:p for p in poses};images={};cap=cv2.VideoCapture(str(video));i=0
 try:
  while wanted:
   ok,bgr=cap.read()
   if not ok:raise ValueError('missing exact physical frame')
   if i in wanted:
    p=wanted.pop(i);rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);h=hashlib.sha256(rgb.tobytes()).hexdigest()
    prior=case['source_RGB'].get(int(case['frame_ids'][p]))
    if prior and prior['RGB_sha256']!=h:raise ValueError('saved source RGB mismatch')
    images[p]=(Image.fromarray(rgb),h)
   i+=1
 finally:cap.release()
 return images

def panel(rgb,box,identity,label):
 canvas=Image.new('RGB',(224,448),(18,20,24));d=ImageDraw.Draw(canvas)
 font=ImageFont.truetype('DejaVuSans.ttf',11)
 # No caption, clock, panel-position or same-target language inside teacher RGB.
 w,h=rgb.size;scale=min(220/w,180/h);rw,rh=round(w*scale),round(h*scale);ox=(224-rw)//2;oy=32+(180-rh)//2
 scene=rgb.resize((rw,rh),Image.Resampling.LANCZOS);canvas.paste(scene,(ox,oy))
 x1,y1,x2,y2=box;rect=(ox+x1*w*scale,oy+y1*h*scale,ox+x2*w*scale,oy+y2*h*scale)
 d.rectangle(rect,outline=(255,40,40),width=2);cx=(rect[0]+rect[2])/2;cy=(rect[1]+rect[3])/2
 d.text((cx,cy),str(identity),fill=(255,30,30),font=font,stroke_width=1,stroke_fill='white')
 # Candidate-specific context crop, precisely recorded and shared by B/C.
 bw=x2-x1;bh=y2-y1;crop=(max(0,x1-.25*bw),max(0,y1-.25*bh),min(1,x2+.25*bw),min(1,y2+.25*bh))
 coords=tuple(round(v*z) for v,z in zip(crop,[w,h,w,h]));detail=rgb.crop(coords)
 scale2=min(220/detail.width,180/detail.height);detail=detail.resize((max(1,round(detail.width*scale2)),max(1,round(detail.height*scale2))),Image.Resampling.LANCZOS)
 cxoff=(224-detail.width)//2;cyoff=250+(180-detail.height)//2;canvas.paste(detail,(cxoff,cyoff))
 # Cropped box uses the exact same legal geometry and crop affine transform.
 cropw=max(1,coords[2]-coords[0]);croph=max(1,coords[3]-coords[1])
 px=[cxoff+(x1*w-coords[0])*detail.width/cropw,cyoff+(y1*h-coords[1])*detail.height/croph,cxoff+(x2*w-coords[0])*detail.width/cropw,cyoff+(y2*h-coords[1])*detail.height/croph]
 d.rectangle(px,outline=(255,40,40),width=2);d.text(((px[0]+px[2])/2,(px[1]+px[3])/2),str(identity),fill=(255,30,30),font=font,stroke_width=1,stroke_fill='white')
 return canvas,dict(box_xyxy_normalized=box,crop_xyxy_normalized=list(crop),scene_scale=scale,
  scene_offset=[ox,oy],crop_pixel_rect=list(coords),candidate_visible_ID=identity)

def render(case,cat,branch,left,right,control,images,poses,global4):
 # Physical candidates stay fixed; visible IDs and panel order are independent controls.
 pair=[left,right];ids=[1,2] if control!='id_swap' else [2,1]
 order=[0,1] if control!='order_swap' else [1,0]
 native=next(r for r in cat['spatial']['candidates'] if r['candidate_id']==cat['native_mapped_candidate_id'])
 frames=[];details=[];fids=cat['frame_ids'];fps=float(case['input']['fps']);bounds=case['video_bounds']
 for p in poses:
  physical=fids[p];rgb,pixelsha=images[p];frame=Image.new('RGB',(448,448));parts=[]
  for column,j in enumerate(order):
   row=pair[j];box=(native if branch=='temporal' else row)['boxes_xyxy_normalized'][p]
   interval=row['interval_frames'] if branch=='temporal' else case['Projection_interval']
   inside=interval[0]<=physical<interval[1]
   # Boundary panels contain actual candidate-specific pixels, rather than an identical timeline-only cue.
   boundary_own=branch=='temporal' and p not in global4 and p in [int(np.argmin(abs(np.asarray(fids)-x))) for x in row['interval_frames']]
   label=f"ID {ids[j]} | {(physical-bounds[0])/fps:.2f}s | {'IN' if inside else 'context'}"
   view,meta=panel(rgb,box,ids[j],label)
   if branch=='temporal' and p not in global4 and not boundary_own:
    view=Image.new('RGB',(224,448),(18,20,24))
    meta={'candidate_visible_ID':ids[j],'masked_nonown_boundary_panel':True}
   frame.paste(view,(column*224,0));parts.append(dict(candidate_id=row['canonical_id'] if branch=='temporal' else row['candidate_id'],inside=bool(inside),**meta))
  frames.append(np.asarray(frame));details.append(dict(position=p,physical_frame_id=physical,pixel_sha256=pixelsha,
      timestamp_s=(physical-bounds[0])/fps,global_context=p in global4,panels=parts))
 meta=dict(branch=branch,left_candidate_id=left.get('canonical_id',left.get('candidate_id')),
  right_candidate_id=right.get('canonical_id',right.get('candidate_id')),visible_ID_by_canonical={str(j):ids[j] for j in [0,1]},
  display_order=order,control=control,frames=details,visual_budget_actual_frame_count=8,
  candidate_specific_boundary_panels=branch=='temporal',GT_read=False)
 # Four independent 2-frame clips: each actual physical gap has its own correct fps.
 videos=[np.stack(frames[i:i+2]).transpose(0,3,1,2) for i in range(0,8,2)]
 rates=[fps/(fids[poses[i+1]]-fids[poses[i]]) for i in range(0,8,2)]
 meta['pair_clip_fps']=rates;meta['marked_RGB_sha256']=hashlib.sha256(np.stack(frames).tobytes()).hexdigest()
 return videos,meta

def prepare():
 import torch
 torch.set_num_threads(2);rows=roster();write(BASE/'ROSTER_LOCK.json',dict(status='locked',records=rows,old_roster_sha256=sha(OLD_SPARSE/'ROSTER_LOCK.json'),historically_exposed=True,fresh_unseen_claim=False))
 fixture=run_cpu_contracts();write(BASE/'CPU_CONTRACTS.json',fixture)
 audits=[];jobs=[]
 for rec in rows:
  case,cat=catalog(rec);write(BASE/'catalogs'/f"{rec['case_id']}.json",cat)
  for m in cat['spatial']['candidates']:
   assert build_verified_sparse_anchors(m,cat['frame_ids'],None).count==0
   poses=list(map(int,case['observation_positions']))[:4];px=decode(case,poses)
   receipt=dict(schema=ANCHOR_RECEIPT_SCHEMA,status='verified',source_case_sha256=rec['source_case_sha256'],candidate_id=m['candidate_id'],
    original_geometry_sha256=m['original_geometry_sha256'],mapped_geometry_sha256=m['mapped_geometry_sha256'],
    opaque_geometry_binding_sha256=m['opaque_geometry_binding_sha256'],receipt_file_sha256='1'*64,GT_read=False,
    verified_observations=[dict(verification_status='verified',frame_position=p,physical_frame_id=cat['frame_ids'][p],
      box_xyxy_normalized=m['boxes_xyxy_normalized'][p],candidate_box_sha256=canonical_sha256(m['boxes_xyxy_normalized'][p]),
      pixel_sha256=px[p][1],observation_receipt_sha256='2'*64) for p in poses])
   # Explicitly a forced interface test, not actual teacher verification or training authority.
   anchor=build_verified_sparse_anchors(m,cat['frame_ids'],receipt);core=verified_anchors_for_student_core(anchor)
   pred=torch.tensor(m['boxes_xyxy_normalized'],dtype=torch.float64,requires_grad=True)
   gather=gather_verified_anchor_predictions(pred,anchor);gather['predictions'].sum().backward()
   assert pred.grad[[p for p in range(len(pred)) if p not in poses]].abs().sum()==0
   a=torch.tensor([x['box'].tolist() for x in core],dtype=torch.float64)
   z=torch.logit(a);mapped=action_transform_logits_cxcywh(z)
   assert torch.allclose(mapped,torch.tensor(anchor.boxes_xyxy_normalized,dtype=torch.float64),atol=2e-7,rtol=0)
   audits.append(dict(case_id=rec['case_id'],candidate_id=m['candidate_id'],clipped_box_count=m['clipped_box_count'],
    clipped_coordinate_count=m['clipped_coordinate_count'],mapped_geometry_sha256=m['mapped_geometry_sha256'],
    original_geometry_sha256=m['original_geometry_sha256'],K=anchor.count,unobserved_gradient_zero=True,
    forced_receipt_CPU_interface_only=True,actual_teacher_verified_anchor_count=0,actual_student_fit_count=0))
  native=next(r for r in cat['spatial']['candidates'] if r['candidate_id']==cat['native_mapped_candidate_id'])
  projection=next(r for r in cat['temporal'] if r['source_candidate_id']=='time_Projection')
  for branch,baseline,others in [('temporal',projection,[r for r in cat['temporal'] if r!=projection]),('spatial',native,[r for r in cat['spatial']['candidates'] if r!=native])]:
   for k,other in enumerate(others):
    jobs.append(dict(case_id=rec['case_id'],dataset=rec['dataset'],branch=branch,pair_index=k,left=baseline,right=other))
 write(BASE/'P0_A_ACTUAL_INTEGRATION.json',dict(status='pass',scope='actual_old32_geometry_sparse_CPU_interface_not_teacher_qualification',records=audits,
  candidate_count=len(audits),out_of_bounds_candidate_count=sum(r['clipped_box_count']>0 for r in audits),GT_read=False,CPU_only=True))
 write(BASE/'PAIR_ROSTER.json',dict(status='locked',records=jobs,pair_count=len(jobs),controls=['base','id_swap','order_swap'],
  maximum_model_calls=len(jobs)*11,actual_GT_before_seal=False))

if __name__=='__main__':
 import sys
 sys.addaudithook(inference_guard);prepare()
