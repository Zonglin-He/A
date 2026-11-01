"""Audit every accepted B direction after fits; never filter training with review."""
import sys,hashlib,textwrap,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'crossplatform/round29'))
from common29 import *
from observations27 import load
from PIL import Image,ImageDraw,ImageFont
fb=[x for x in read(BASE/'direct208_attempt001/FEEDBACK_PRIVATE.json') if x['label']=='B'];lookup={r['id']:r for r in rows()[:208]};key=[];out=EXP.parent/'correction_views';out.mkdir(exist_ok=False);font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',19);big=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',24)
for i,x in enumerate(sorted(fb,key=lambda x:hashlib.sha256(('R29_corrections|'+x['id_PRIVATE']+x['pair_alias']).encode()).hexdigest())):
 case=f'D{i+1:02d}';r=lookup[x['id_PRIVATE']];frame,scene,_=load(r);ms=sorted([x['positive_mask_PRIVATE'],x['negative_mask_PRIVATE']],key=lambda m:hashlib.sha256((case+scene['objects'][m]['mask_sha256']).encode()).hexdigest());lab=dict(zip('XY',ms));rgb=Image.fromarray(frame['rgb']);W,H=rgb.size;rgb=rgb.resize((960,600));d=ImageDraw.Draw(rgb)
 for l,m in lab.items():
  b=list(map(int,np.array(scene['objects'][m]['detector_box_xyxy'])*np.array([960/W,600/H]*2)));d.rectangle(b,outline=(255,0,255),width=3);y=max(0,b[1]-28);d.rectangle((b[0],y,b[0]+30,y+28),fill='white');d.text((b[0]+3,y),l,fill='black',font=big)
 im=Image.new('RGB',(960,810),'white');im.paste(rgb,(0,35));d=ImageDraw.Draw(im);d.text((8,3),case+' '+r['platform']+' | scores and accepted direction hidden',font=big,fill='black')
 for j,line in enumerate(textwrap.wrap(r['caption'],width=86)):d.text((8,645+22*j),line,font=font,fill='black')
 im.save(out/(case+'.png'));key.append({'case':case,'alias':r['alias'],'id_PRIVATE':r['id'],'pair_alias':x['pair_alias'],'labels_to_mask_PRIVATE':lab,'positive_mask_PRIVATE':x['positive_mask_PRIVATE'],'platform':r['platform']})
dump(EXP.parent/'input/CORRECTION6_KEY_PRIVATE.json',key);seal(out,'R29_all6accepted_corrections_postfit_visual_audit',n=6,selection_all_B_not_GT=True,no_direction_scores_or_GT_shown=True,not_independent_method_confirmation=True);print('CORRECTION6_PANELS')
