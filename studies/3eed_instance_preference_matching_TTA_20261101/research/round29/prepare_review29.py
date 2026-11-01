import sys,hashlib,collections,textwrap,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'crossplatform/round29'))
from common29 import *
from observations27 import load
from PIL import Image,ImageDraw,ImageFont

def main():
 pools=collections.defaultdict(list)
 for r in rows()[:208]:
  for pair in old(r,'pairs')['pairs']:
   pools[(r['platform'],r['scene'])].append((hashlib.sha256(('R29_blind_pair|'+r['id']+'|'+pair['pair_alias']).encode()).hexdigest(),r,pair))
 selected=[]
 for pl in ['drone','quad']:
  gg=[sorted(v,key=lambda x:x[0]) for (p,s),v in sorted(pools.items()) if p==pl];seen=set();n=0
  while n<12:
   advanced=False
   for g in gg:
    while g and g[0][1]['id'] in seen:g.pop(0)
    if g and n<12:
     _,r,p=g.pop(0);seen.add(r['id']);selected.append((r,p));n+=1;advanced=True
   assert advanced
 out=EXP.parent/'private_views';font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',19);big=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',24);key=[];inputs=[]
 for i,(r,p) in enumerate(selected):
  case=f'C{i+1:02d}';frame,scene,bind=load(r);masks=sorted(p['mask_indices_PRIVATE'],key=lambda m:hashlib.sha256(('R29_label|'+case+'|'+scene['objects'][m]['mask_sha256']).encode()).hexdigest());lab=dict(zip('XY',masks));rgb=Image.fromarray(frame['rgb']);W,H=rgb.size;rgb=rgb.resize((960,600));d=ImageDraw.Draw(rgb)
  for l,m in lab.items():
   b=list(map(int,np.array(scene['objects'][m]['detector_box_xyxy'])*np.array([960/W,600/H]*2)));d.rectangle(b,outline=(255,0,255),width=3);y=max(0,b[1]-28);d.rectangle((b[0],y,b[0]+30,y+28),fill='white');d.text((b[0]+3,y),l,font=big,fill='black')
  panel=Image.new('RGB',(960,810),'white');panel.paste(rgb,(0,35));d=ImageDraw.Draw(panel);d.text((8,3),case+' '+r['platform']+' | X/Y: two candidate instances',fill='black',font=big)
  for j,line in enumerate(textwrap.wrap(r['caption'],width=86)):d.text((8,645+22*j),line,fill='black',font=font)
  panel.save(out/(case+'.png'));inputs.append({'case':case,'caption':r['caption'],'platform':r['platform']});key.append({'case':case,'id_PRIVATE':r['id'],'alias':r['alias'],'platform':r['platform'],'pair_alias':p['pair_alias'],'labels_to_mask_PRIVATE':lab,'mask_A_PRIVATE':p['mask_indices_PRIVATE'][0],'mask_B_PRIVATE':p['mask_indices_PRIVATE'][1]})
 for page in range(6):
  sheet=Image.new('RGB',(1920,1620),'white')
  for j in range(4):sheet.paste(Image.open(out/f'C{page*4+j+1:02d}.png'),(960*(j%2),810*(j//2)))
  sheet.save(out/f'PAGE_{page+1:02d}.png')
 dump(EXP.parent/'input/BLIND24_KEY_PRIVATE.json',key);dump(EXP.parent/'input/BLIND24_INPUTS_PRIVATE.json',inputs);seal(out,'R29_input_only_random_label_pair_review_before_annotations',n=24,task_and_method_scores_hidden=True);print('PANELS24_SEALED')
if __name__=='__main__':main()
