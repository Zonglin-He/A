"""Actual fixed-frame private final-output overlays after full method/Oracle seals."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch,numpy as np
from PIL import Image,ImageDraw,ImageFont
from scripts.stvg_temporal_caption_query_v1 import read,write,sha,roster,catalog
from scripts.review_stvg_sparse_r1_v2 import _truths_after_seal
B=ROOT/'artifacts/stvg_video_conditioned_global_referent_v1'
for phase in ['METHOD','ORACLE']:
 b=read(B/f'{phase}_STUDENT_GLOBAL_BARRIER.json');assert b['count']==8
 for f in b['files']:assert sha(ROOT/f['path'])==f['sha256']
cases={r['case_id']:catalog(r)[0]for r in roster()};gt,_=_truths_after_seal(cases);inputs=read(B/'CANDIDATE_INPUT_BARRIER.json')['records'];files=[]
for idx,r in enumerate(inputs):
 cid=r['case_id'];c=cases[cid];m=torch.load(B/'method/student'/f'{cid}.pt',map_location='cpu',weights_only=False);o=torch.load(B/'oracle/student'/f'{cid}.pt',map_location='cpu',weights_only=False);source=np.load(B/'private_inputs'/cid/'candidate1_ID1.npy',allow_pickle=False);alias=('HC'if r['dataset']=='hc2'else'V')+f'{idx%4+1:02d}';canvas=Image.new('RGB',(960,560),'white');d=ImageDraw.Draw(canvas);font=ImageFont.truetype('DejaVuSans.ttf',13);d.text((8,5),alias+' | red Source; blue Teacher IoU; orange Offline GT IoU; green GT',fill='black',font=font)
 s=min(448/c['width'],448/c['height']);rw,rh=round(c['width']*s),round(c['height']*s);ox,oy=(448-rw)//2,(448-rh)//2
 for row,(label,arm,color)in enumerate([('Actual teacher reference / IoU-OPD',m['arms']['teacher_iou'],(30,170,255)),('Offline GT reference / IoU-OPD (privileged)',o['arms']['offline_oracle_iou'],(255,160,20))]):
  d.text((8,25+row*264),label,fill='black',font=font)
  for k,(p,f)in enumerate(zip(r['positions'],r['physical_frame_ids'])):
   im=Image.fromarray(source[k].transpose(1,2,0));draw=ImageDraw.Draw(im);x1,y1,x2,y2=arm['boxes'][p];draw.rectangle((ox+x1*rw,oy+y1*rh,ox+x2*rw,oy+y2*rh),outline=color,width=3)
   if f in gt[cid]['truth']:
    x1,y1,x2,y2=np.array(gt[cid]['truth'][f])/np.array([c['width'],c['height'],c['width'],c['height']]);draw.rectangle((ox+x1*rw,oy+y1*rh,ox+x2*rw,oy+y2*rh),outline=(50,220,70),width=2)
   else:draw.text((5,23),'GT unknown here',font=ImageFont.truetype('DejaVuSans.ttf',14),fill=(255,220,80))
   canvas.paste(im.resize((240,240),Image.Resampling.LANCZOS),(k*240,46+row*264))
 dst=B/'private_views'/f'{alias}_final_outputs.png';canvas.save(dst);files.append(dict(path=str(dst.relative_to(ROOT)),sha256=sha(dst),count=8))
write(B/'PRIVATE_FINAL_RENDER_BINDING.json',dict(status='rendered_pending_root_actual_view',count=8,files=files,GT_read=True,source_fixed_positions=True))
print('8 actual final private overlays generated')
