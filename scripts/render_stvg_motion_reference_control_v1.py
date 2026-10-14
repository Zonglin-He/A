"""Private actual V03 fixed-reference control, after complete prediction seal."""
import sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch,cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
from scripts.stvg_temporal_caption_query_v1 import read,write,sha,catalog
from scripts.review_stvg_sparse_r1_v2 import _truths_after_seal
from scripts.prepare_stvg_scene_object_id_v1 import _letterbox
B=ROOT/'artifacts/stvg_motion_guided_instance_opd_v1';bar=read(B/'REFERENCE_CONTROL_STUDENT_GLOBAL_BARRIER.json');assert bar['status']=='complete'and bar['count']==1
for f in bar['files']:assert sha(ROOT/f['path'])==f['sha256']
cid='vidstg_002';row=next(r for r in read(B/'ROSTER_LOCK.json')['records']if r['case_id']==cid);c=catalog(row)[0];g,_=_truths_after_seal({cid:c});z=torch.load(B/'reference_control/student'/f'{cid}.pt',map_location='cpu',weights_only=False);req=read(ROOT/'artifacts/stvg_target_teacher_distillation_v1/canonical_inputs'/cid/'REQUEST.json');assert sha(Path(req['video_path']))==req['canonical_video_sha256'];pos=c['observation_positions'];fids=[c['frame_ids'][p]for p in pos];fm={x['physical_frame_id']:x['local_frame_index']for x in req['frame_map']};targets={fm[f]:f for f in fids};cap=cv2.VideoCapture(req['video_path']);i=0;ims={}
while targets:
 ok,bgr=cap.read();assert ok
 if i in targets:
  f=targets.pop(i);rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);assert hashlib.sha256(rgb.tobytes()).hexdigest()==c['source_RGB'][f]['RGB_sha256'];ims[f]=Image.fromarray(rgb)
 i+=1
cap.release();canvas=Image.new('RGB',(1280,1044),'white');d=ImageDraw.Draw(canvas);font=ImageFont.truetype('DejaVuSans.ttf',14)
for ri,role in enumerate(['Source','oldR1ID2','oldAlternativeID3']):
 d.text((6,348*ri+4),'V03 '+role+' IoU OPD | red Source; blue adapted; yellow reference; green known GT',fill='black',font=font)
 refs=z['supervision_anchors'][role]
 for k,(p,f)in enumerate(zip(pos,fids)):
  im,_=_letterbox(ims[f]);dd=ImageDraw.Draw(im);w,h=ims[f].size;s=min(448/w,448/h);rw,rh=round(w*s),round(h*s);ox,oy=(448-rw)//2,(448-rh)//2
  def box(b,color):
   x1,y1,x2,y2=b;dd.rectangle((ox+x1*rw,oy+y1*rh,ox+x2*rw,oy+y2*rh),outline=color,width=2)
  a=np.array(refs[k]['box_cxcywh']);bb=np.r_[a[:2]-a[2:]/2,a[:2]+a[2:]/2];box(bb,(255,210,20));box(z['arms']['F']['boxes'][p],(240,60,40));box(z['arms'][role+'_iou']['boxes'][p],(20,150,255))
  if f in g[cid]['truth']:box(np.array(g[cid]['truth'][f])/[w,h,w,h],(50,220,70))
  else:dd.text((4,25),'GT UNKNOWN',fill=(255,210,20),font=font)
  canvas.paste(im.resize((320,320)),(320*k,348*ri+25))
p=B/'private_views/V03_fixed_reference_control.png';canvas.save(p);print(p)
