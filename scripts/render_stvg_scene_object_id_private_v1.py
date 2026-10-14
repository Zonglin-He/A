"""Private full-control scene and actual final readout contacts; no publication media."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from PIL import Image,ImageDraw,ImageFont
from scripts.stvg_temporal_caption_query_v1 import read,roster,catalog,sha,write
from scripts.review_stvg_sparse_r1_v2 import _truths_after_seal
B=ROOT/'artifacts/stvg_scene_object_id_referent_v1';bar=read(B/'METHOD_STUDENT_GLOBAL_BARRIER.json');assert bar['count']==8
for f in bar['files']:assert sha(ROOT/f['path'])==f['sha256']
rows=read(B/'CANDIDATE_INPUT_BARRIER.json')['records'];cases={r['case_id']:catalog(r)[0]for r in roster()};g,_=_truths_after_seal(cases);out=B/'private_views';out.mkdir(exist_ok=True)
for idx,row in enumerate(rows):
 cid=row['case_id'];alias=('HC'if row['dataset']=='hc2'else'V')+f'{idx%4+1:02d}';can=Image.new('RGB',(960,1315),'white');d=ImageDraw.Draw(can);font=ImageFont.truetype('DejaVuSans.ttf',13);d.text((8,5),alias+' | actual same-scene control pixels | all four original frames',fill='black',font=font)
 for ri,k in enumerate(['base','id_swap','wrong_query','Source_removed','unmarked_none']):
  a=np.load(B/'private_inputs'/cid/f'{k}.npy',allow_pickle=False);d.text((8,26+ri*256),k,fill='black',font=font)
  for ki,im in enumerate(a):can.paste(Image.fromarray(im.transpose(1,2,0)).resize((240,240),Image.Resampling.LANCZOS),(ki*240,44+ri*256))
 can.save(out/f'{alias}_all_controls.png');res=torch.load(B/'method/student'/f'{cid}.pt',map_location='cpu',weights_only=False);can=Image.new('RGB',(960,810),'white');d=ImageDraw.Draw(can);d.text((8,5),alias+' | red Source; green GT; blue adapted | actual complete model readouts',fill='black',font=font);c=cases[cid];rw,rh=round(c['width']*min(448/c['width'],448/c['height'])),round(c['height']*min(448/c['width'],448/c['height']));ox,oy=(448-rw)//2,(448-rh)//2;clean=np.load(B/'private_inputs'/cid/'unmarked_none.npy',allow_pickle=False)
 for ri,k in enumerate(['direct','iou','diou']):
  d.text((8,26+ri*256),k,fill='black',font=font)
  for ki,(p,fid)in enumerate(zip(row['positions'],row['physical_frame_ids'])):
   im=Image.fromarray(clean[ki].transpose(1,2,0));dd=ImageDraw.Draw(im)
   for boxes,color in [(res['arms']['F']['boxes'],(240,60,40)),(res['arms']['teacher_'+k]['boxes'],(20,150,255))]:
    x1,y1,x2,y2=boxes[p];dd.rectangle((ox+x1*rw,oy+y1*rh,ox+x2*rw,oy+y2*rh),outline=color,width=2)
   if fid in g[cid]['truth']:
    x1,y1,x2,y2=np.array(g[cid]['truth'][fid])/np.array([c['width'],c['height'],c['width'],c['height']]);dd.rectangle((ox+x1*rw,oy+y1*rh,ox+x2*rw,oy+y2*rh),outline=(50,220,70),width=2)
   else:dd.text((4,22),'GT unknown',font=font,fill=(255,220,90))
   can.paste(im.resize((240,240),Image.Resampling.LANCZOS),(ki*240,44+ri*256))
 can.save(out/f'{alias}_actual_updates.png')
print('16 actual private PNGs generated')
