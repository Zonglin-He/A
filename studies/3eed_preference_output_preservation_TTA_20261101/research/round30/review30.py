"""Available-evidence labels, independent of IoU and adapted outputs."""
import sys,hashlib,textwrap,numpy as np,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'crossplatform/round30'))
from common30 import *
from observations27 import load
from PIL import Image,ImageDraw,ImageFont
fb=read(ROOT/'crossplatform/round29/direct208_attempt001/FEEDBACK_PRIVATE.json');lookup={r['id']:r for r in rows()};key=[];out=EXP.parent/'evidence_views';out.mkdir(exist_ok=False);font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',19);big=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',24)
for i,x in enumerate(sorted(fb,key=lambda x:hashlib.sha256(('R30_evidence|'+x['id_PRIVATE']+x['pair_alias']).encode()).hexdigest())):
 case=f'E{i+1:02d}';r=lookup[x['id_PRIVATE']];frame,scene,_=load(r);ms=sorted([x['positive_mask_PRIVATE'],x['negative_mask_PRIVATE']],key=lambda m:hashlib.sha256((case+scene['objects'][m]['mask_sha256']).encode()).hexdigest());lab=dict(zip('XY',ms));rgb=Image.fromarray(frame['rgb']);W,H=rgb.size;rgb=rgb.resize((960,600));d=ImageDraw.Draw(rgb)
 for l,m in lab.items():
  b=list(map(int,np.array(scene['objects'][m]['detector_box_xyxy'])*np.array([960/W,600/H]*2)));d.rectangle(b,outline=(255,0,255),width=3);y=max(0,b[1]-28);d.rectangle((b[0],y,b[0]+30,y+28),fill='white');d.text((b[0]+3,y),l,font=big,fill='black')
 im=Image.new('RGB',(960,810),'white');im.paste(rgb,(0,35));d=ImageDraw.Draw(im);d.text((8,3),case+' '+r['platform']+' | evidence only, no scores/outcomes',font=big,fill='black')
 for j,line in enumerate(textwrap.wrap(r['caption'],width=86)):d.text((8,645+22*j),line,font=font,fill='black')
 im.save(out/(case+'.png'));key.append({'case':case,'alias':r['alias'],'id_PRIVATE':r['id'],'pair_alias':x['pair_alias'],'labels_to_mask_PRIVATE':lab,'positive_mask_PRIVATE':x['positive_mask_PRIVATE'],'platform':r['platform']})
for page in range(math.ceil(len(fb)/4)):
 im=Image.new('RGB',(1920,1620),'white')
 for j in range(4):
  if page*4+j<len(fb):im.paste(Image.open(out/f'E{page*4+j+1:02d}.png'),(960*(j%2),810*(j//2)))
 im.save(out/f'PAGE_{page+1:02d}.png')
dump(EXP.parent/'input/EVIDENCE22_KEY_PRIVATE.json',key);seal(out,'R30_22feedback_availableevidence_panels_before_review',n=22,no_task_adaptation_scores=True,directions_hidden=True);print('EVIDENCE22_READY')
