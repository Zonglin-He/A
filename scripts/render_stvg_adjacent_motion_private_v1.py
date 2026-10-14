"""Render sealed actual control inputs and student updates for private pixel review."""
import sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch,cv2
from PIL import Image,ImageDraw,ImageFont
from scripts.stvg_temporal_caption_query_v1 import read,write,sha,catalog
from scripts.review_stvg_sparse_r1_v2 import _truths_after_seal
from scripts.prepare_stvg_scene_object_id_v1 import _letterbox
B=ROOT/'artifacts/stvg_adjacent_motion_opd_v1';O=B/'private_views';O.mkdir(exist_ok=True);torch.set_num_threads(2)
bar=read(B/'METHOD_STUDENT_GLOBAL_BARRIER.json');assert bar['status']=='complete' and bar['count']==8 and read(B/'ROOT_POSTSEAL_REPORT.json')['status']=='pass'
for f in bar['files']:assert sha(ROOT/f['path'])==f['sha256']
roster=read(B/'ROSTER_LOCK.json')['records'];rows=read(B/'CANDIDATE_INPUT_BARRIER.json')['records'];cases={r['case_id']:catalog(r)[0] for r in roster};truths,_=_truths_after_seal(cases);font=ImageFont.truetype('DejaVuSans.ttf',14);allpaths=[];jobs=read(B/'TEACHER_JOB_LOCK.json')['records']
for row in rows:
 cid=row['case_id'];c=cases[cid];can=Image.new('RGB',(1280,6*348),'white');dd=ImageDraw.Draw(can)
 for j,job in enumerate([j for j in jobs if j['case_id']==cid]):
  v=job['videos'][0];assert sha(ROOT/v['path'])==v['sha256'];a=np.load(ROOT/v['path'],allow_pickle=False);dd.text((6,j*348+4),cid+' '+job['kind']+' | actual fixed badge input',fill='black',font=font)
  for k,im in enumerate(a):can.paste(Image.fromarray(im.transpose(1,2,0)).resize((320,320)),(k*320,j*348+25))
 p=O/(cid+'_controls.png');can.save(p);allpaths.append(p)
 req=read(ROOT/'artifacts/stvg_target_teacher_distillation_v1/canonical_inputs'/cid/'REQUEST.json');assert sha(Path(req['video_path']))==req['canonical_video_sha256'];fm={int(x['physical_frame_id']):int(x['local_frame_index']) for x in req['frame_map']};targets={fm[f]:f for f in row['physical_frame_ids']};imgs={};cap=cv2.VideoCapture(req['video_path']);i=0
 while targets:
  ok,bgr=cap.read();assert ok
  if i in targets:
   f=targets.pop(i);rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);assert hashlib.sha256(rgb.tobytes()).hexdigest()==row['source_RGB_sha256_by_frame'][str(f)];imgs[f]=Image.fromarray(rgb)
  i+=1
 cap.release();res=torch.load(B/'method/student'/f'{cid}.pt',map_location='cpu',weights_only=False);ref=next(x for x in read(B/'METHOD_REFERENCE_LOCK.json')['records'] if x['case_id']==cid)['references']['teacher'];selected=next((x for x in row['candidate_geometry'] if x['candidate_id']==ref['selected_candidate_id']),None)
 can=Image.new('RGB',(1280,4*348),'white');dd=ImageDraw.Draw(can)
 for ri,arm in enumerate(['reference','teacher_direct','teacher_iou','teacher_motion_iou']):
  dd.text((6,ri*348+4),cid+' '+arm+' | red Source; green known GT; blue actual decode; yellow motion reference',fill='black',font=font)
  for k,(pos,fid) in enumerate(zip(row['positions'],row['physical_frame_ids'])):
   im,lb=_letterbox(imgs[fid]);dr=ImageDraw.Draw(im);w,h=imgs[fid].size;s=min(448/w,448/h);rw,rh=round(w*s),round(h*s);ox,oy=(448-rw)//2,(448-rh)//2
   def box(b,color):
    x1,y1,x2,y2=b;dr.rectangle((ox+x1*rw,oy+y1*rh,ox+x2*rw,oy+y2*rh),outline=color,width=2)
   box(res['arms']['F']['boxes'][pos],(240,60,40))
   if arm!='reference':box(res['arms'][arm]['boxes'][pos],(20,150,255))
   elif selected and selected['geometry_boxes_xyxy'][k] is not None:box(selected['geometry_boxes_xyxy'][k],(255,210,20))
   else:dr.text((4,24),'REFERENCE MISSING',fill=(255,210,20),font=font)
   if fid in truths[cid]['truth']:box(np.array(truths[cid]['truth'][fid])/[w,h,w,h],(50,220,70))
   else:dr.text((4,42),'GT UNKNOWN',fill=(255,210,20),font=font)
   can.paste(im.resize((320,320)),(k*320,ri*348+25))
 p=O/f'{cid}_actual_updates.png';can.save(p);allpaths.append(p)
write(B/'PRIVATE_FIGURE_EXPORT.json',dict(status='exported_not_yet_viewed',contacts=len(allpaths),actual_control_slots=192,final_reference_and_update_slots=128,files=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in allpaths],GT_read=True))
print('exported',len(allpaths),'private contacts')
