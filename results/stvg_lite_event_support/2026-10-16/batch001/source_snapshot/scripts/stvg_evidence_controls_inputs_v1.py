"""Finite same-candidate representation controls; no labels, tracking or inference."""
import json,hashlib,time
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scripts.stvg_query_alignment_inputs_v3 import (ROOT,OLD,sha,read,write,roster,catalog,positions,decode,render as paired_render,inference_guard)
BASE=ROOT/'artifacts/stvg_sparse_r1_evidence_task_controls_v1'
CONTROLS=['base','id_swap','order_swap']

def pairs(rec):
 c,cat=catalog(rec);native=next(x for x in cat['spatial']['candidates'] if x['candidate_id']==cat['native_mapped_candidate_id'])
 projection=next(x for x in cat['temporal'] if x['source_candidate_id']=='time_Projection')
 nt=next(x for x in cat['temporal'] if x['source_candidate_id']=='time_Native')
 other=next(x for x in cat['spatial']['candidates'] if x['opaque_geometry_binding']['source_candidate_id']=='space_large_plus')
 trunc=dict(native);b=np.array(native['boxes_xyxy_normalized']);b[:,2]=b[:,0]+.45*(b[:,2]-b[:,0]);trunc['boxes_xyxy_normalized']=b.tolist();trunc['candidate_id']='coarse_native_right_truncated';trunc['mapped_geometry_sha256']=hashlib.sha256(b.astype('<f8').tobytes()).hexdigest()
 return c,cat,{'temporal':[projection,nt],'spatial':[native,other],'coarse':[native,trunc]}

def full_panel(rgb,box,ident):
 w,h=rgb.size;scale=min(448/w,448/h);rw,rh=round(w*scale),round(h*scale);ox=(448-rw)//2;oy=(448-rh)//2
 canvas=Image.new('RGB',(448,448),(18,20,24));canvas.paste(rgb.resize((rw,rh),Image.Resampling.LANCZOS),(ox,oy));d=ImageDraw.Draw(canvas)
 x1,y1,x2,y2=box;rect=(ox+x1*rw,oy+y1*rh,ox+x2*rw,oy+y2*rh)
 d.rectangle(rect,outline=(255,40,40),width=2);d.text(((rect[0]+rect[2])/2,(rect[1]+rect[3])/2),str(ident),font=ImageFont.truetype('DejaVuSans.ttf',18),fill='red',stroke_width=1,stroke_fill='white')
 return np.asarray(canvas),dict(scene_size=[rw,rh],offset=[ox,oy],box_xyxy_normalized=list(box))

def continuous_decode(case,pair):
 """Two fixed contiguous boundary windows, uniform stride four, eight frames each.
 All 32 frames in each window are decoded; 8 are supplied at source FPS / 4.
 No concatenated discontinuous windows are presented as one video.
 """
 import cv2
 req=read(OLD/'canonical_inputs'/case['case_id']/'REQUEST.json');video=Path(req['video_path']);assert sha(video)==req['canonical_video_sha256']
 assert req['source_video_sha256']==case['input']['video_sha256']
 fmap={x['physical_frame_id']:x['local_frame_index'] for x in req['frame_map']};physical=sorted(fmap);assert np.all(np.diff(physical)==1)
 mids=[round((pair[0]['interval_frames'][k]+pair[1]['interval_frames'][k])/2) for k in [0,1]]
 starts=[max(physical[0],min(physical[-1]-31,m-16)) for m in mids]
 windows=[list(range(s,s+32)) for s in starts];samples=[w[::4] for w in windows];want={f for w in windows for f in w};selected={f for w in samples for f in w};imgs={};hashes={};decoded=[]
 cap=cv2.VideoCapture(str(video));i=0;inv={v:k for k,v in fmap.items()}
 try:
  while want:
   ok,bgr=cap.read()
   if not ok:raise ValueError('missing canonical consecutive frame')
   f=inv.get(i)
   if f in want:
    want.remove(f);rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);h=hashlib.sha256(rgb.tobytes()).hexdigest();hashes[f]=h;decoded.append(f)
    prior=case['source_RGB'].get(f)
    if prior:assert prior['RGB_sha256']==h
    if f in selected:imgs[f]=Image.fromarray(rgb)
   i+=1
 finally:cap.release()
 return samples,imgs,dict(windows=windows,decoded_frame_ids=decoded,pixel_sha256_by_physical_frame={str(f):hashes[f] for f in sorted(hashes)},canonical_video_sha256=req['canonical_video_sha256'],uniform_physical_stride=4,clip_fps=float(case['input']['fps'])/4,all_window_frames_decoded=True)

def render(case,cat,branch,pair,representation,control):
 visible=[1,2] if control!='id_swap' else [2,1];order=[0,1] if control!='order_swap' else [1,0];actual_branch='spatial' if branch=='coarse' else branch
 canonical=[x.get('canonical_id',x.get('candidate_id')) for x in pair];mapping={str(visible[j]):canonical[j] for j in [0,1]}
 native=next(x for x in cat['spatial']['candidates'] if x['candidate_id']==cat['native_mapped_candidate_id'])
 intervals={str(visible[j]):x['interval_seconds'] for j,x in enumerate(pair)} if branch=='temporal' else None
 if representation=='paired_small':
  poses,global4=positions(case,actual_branch,*pair);imgs=decode(case,poses);videos,meta=paired_render(case,cat,actual_branch,*pair,control,imgs,poses,global4)
  rates=meta['pair_clip_fps'];clipnote=[{'frames':[meta['frames'][2*k]['physical_frame_id'],meta['frames'][2*k+1]['physical_frame_id']],'start_time_s':(meta['frames'][2*k]['physical_frame_id']-case['video_bounds'][0])/case['input']['fps'],'discontinuous_sparse_pair':True} for k in range(4)]
 else:
  videos=[];rates=[];clipnote=[];details=[]
  if branch=='temporal':
   samples,imgs,custody=continuous_decode(case,pair)
   for j in order:
    for wi,fids in enumerate(samples):
     frames=[]
     for f in fids:
      box=[float(np.interp(f,cat['frame_ids'],np.asarray(native['boxes_xyxy_normalized'])[:,k])) for k in range(4)]
      im,geom=full_panel(imgs[f],box,visible[j]);frames.append(im);details.append(dict(physical_frame_id=f,visible_ID=visible[j],**geom))
     videos.append(np.stack(frames).transpose(0,3,1,2));rates.append(custody['clip_fps']);clipnote.append(dict(visible_ID=visible[j],start_time_s=(fids[0]-case['video_bounds'][0])/case['input']['fps'],physical_frame_ids=fids,contiguous_source_window=custody['windows'][wi],uniform_stride=4))
  else:
   poses,global4=positions(case,'spatial',*pair);imgs=decode(case,poses);custody=dict(pixel_sha256_by_physical_frame={str(cat['frame_ids'][p]):imgs[p][1] for p in poses})
   # Same old eight physical frames; four separate true-gap pairs per candidate.
   for j in order:
    frames=[]
    for p in poses:
     im,geom=full_panel(imgs[p][0],pair[j]['boxes_xyxy_normalized'][p],visible[j]);frames.append(im);details.append(dict(physical_frame_id=cat['frame_ids'][p],visible_ID=visible[j],**geom))
    for k in range(0,8,2):
     fids=[cat['frame_ids'][poses[k]],cat['frame_ids'][poses[k+1]]];videos.append(np.stack(frames[k:k+2]).transpose(0,3,1,2));rates.append(case['input']['fps']/(fids[1]-fids[0]));clipnote.append(dict(visible_ID=visible[j],physical_frame_ids=fids,start_time_s=(fids[0]-case['video_bounds'][0])/case['input']['fps'],discontinuous_sparse_pair=True))
  meta=dict(custody=custody,frames=details,independent_full_scene_448=True)
 meta.update(branch=branch,representation=representation,control=control,visible_ID_to_candidate=mapping,candidate_intervals_seconds_by_visible_ID=intervals,clips=clipnote,clip_fps=rates,visual_frame_count=sum(len(x) for x in videos),marked_RGB_sha256=hashlib.sha256(b''.join(v.tobytes() for v in videos)).hexdigest(),GT_read=False)
 return videos,meta

def prepare():
 from scripts import stvg_evidence_controls_schema_v1 as s
 from scripts import stvg_evidence_controls_resource_guard_v1 as g
 rows=roster();jobs=[]
 for r in rows:
  c,cat,pp=pairs(r)
  for branch in ['temporal','spatial']:
   for rep in ['paired_small','independent_high']:
    for control in CONTROLS:
     jobs.append(dict(case_id=r['case_id'],branch=branch,representation=rep,control=control,tasks=['native_style','alignment'] if rep=='independent_high' else ['native_style']))
   jobs.append(dict(case_id=r['case_id'],branch=branch,representation='independent_high',control='wrong_query',tasks=['native_style']))
  for control in CONTROLS:jobs.append(dict(case_id=r['case_id'],branch='coarse',representation='independent_high',control=control,tasks=['native_style']))
 write(BASE/'ROSTER_LOCK.json',dict(status='locked',records=rows,old_roster_sha256=sha(ROOT/'artifacts/stvg_sparse_stvg_r1_semantic_opd_v1/ROSTER_LOCK.json'),historically_exposed=True,fresh_unseen_claim=False))
 write(BASE/'JOB_ROSTER.json',dict(status='locked',records=jobs,query_count=8,model_call_cap=sum(len(j['tasks'])+('alignment' in j['tasks']) for j in jobs),facts_calls_before_alignment=True))
 write(BASE/'CPU_CONTRACTS.json',dict(status='pass',schema=s.contracts(),resource=g.contracts(),GT_read=False,CPU_only=True))
 print(json.dumps({'jobs':len(jobs),'model_call_cap':read(BASE/'JOB_ROSTER.json')['model_call_cap']}))
