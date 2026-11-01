"""Input-only blind panel. No GT, source scores, VLM scores or outputs displayed."""
import sys,hashlib,collections,math,textwrap,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'crossplatform/round28'))
from common28 import *
from observations27 import load
from PIL import Image,ImageDraw,ImageFont
def main():
 allrows=rows();pools=collections.defaultdict(list);lookup={}
 for r in allrows:
  p=read(paths(r)['prep']/'cases_PRIVATE'/(r['id']+'.json'))
  if p['objects']>=2:pools[(r['platform'],r['scene'])].append((hashlib.sha256(('R28_blind_identity|'+r['id']).encode()).hexdigest(),r,p))
 selected=[]
 for platform in ['drone','quad']:
  groups=[sorted(v,key=lambda x:x[0]) for (p,s),v in sorted(pools.items()) if p==platform];n=0
  while n<12:
   for group in groups:
    if group and n<12:_,r,p=group.pop(0);selected.append((r,p));n+=1
 inp=[];priv=[];font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18);big=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',22);out=EXP.parent/'private_views'
 for i,(r,p) in enumerate(selected):
  case='B'+str(i+1).zfill(2);frame,scene,binding=load(r);pool=p['mask_pool_PRIVATE'];ordered=sorted(pool,key=lambda m:hashlib.sha256(('R28_maskorder|'+case+'|'+scene['objects'][m]['mask_sha256']).encode()).hexdigest());labels={chr(65+j):m for j,m in enumerate(ordered)};rgb=Image.fromarray(frame['rgb']);W,H=rgb.size;rgb=rgb.resize((960,600));d=ImageDraw.Draw(rgb)
  for label,m in labels.items():
   box=np.asarray(scene['objects'][m]['detector_box_xyxy'])*np.array([960/W,600/H]*2);b=list(map(int,box));d.rectangle(b,outline=(255,0,255),width=3);d.rectangle((b[0],max(0,b[1]-26),b[0]+28,max(0,b[1]-26)+26),fill='white');d.text((b[0]+3,max(0,b[1]-26)),label,fill='black',font=big)
  panel=Image.new('RGB',(960,850),'white');panel.paste(rgb,(0,36));draw=ImageDraw.Draw(panel);draw.text((8,5),case+' | '+r['platform']+' | no method scores / GT / model outputs',font=big,fill='black')
  lines=textwrap.wrap(r['caption'],width=91);assert len(lines)<=10,(case,len(lines))
  for j,line in enumerate(lines):draw.text((8,643+j*20),line,font=font,fill='black')
  panel.save(out/(case+'.png'));inp.append({'blind_case':case,'platform':r['platform'],'caption':r['caption'],'candidate_labels':list(labels),'image_path_PRIVATE':str(out/(case+'.png'))});priv.append({'blind_case':case,'alias':r['alias'],'id_PRIVATE':r['id'],'scene_PRIVATE':r['scene'],'platform':r['platform'],'labels_to_mask_PRIVATE':labels,'bindings':binding})
 for page in range(6):
  sheet=Image.new('RGB',(1920,1700),'white')
  for j in range(4):sheet.paste(Image.open(out/(f'B{4*page+j+1:02d}.png')),(960*(j%2),850*(j//2)))
  sheet.save(out/(f'PAGE_{page+1:02d}.png'))
 dump(EXP.parent/'input/BLIND_PANEL_INPUTS_PRIVATE.json',inp);dump(EXP.parent/'input/BLIND_KEY_PRIVATE.json',priv);seal(out,'R28_blind_identity_panels_before_visual_review',n=24,method_scores_GT_and_outputs_hidden=True)
 dump(EXP/'PLAN_LOCK.json',{'scope':'offline cache diagnosis,no new VLM/GPU/TTA','all528':'commonoffset stress,sourceprior,unmappedcompetition,readoutfactors andcenterlabel geometry diagnoses','blind24':'12each,inputhash+scene roundrobin,multiplepooledinstances>=2,no method/GT selection','identity':'assistant visual review of full literal caption and RGB independently of 3DIoU;unresolved retained;not expert-adjudicatedgold','annotation_based_3D_assignment':'unique predicted center inside annotated target/other OBB,independent of3DIoU;geometric proxy only','point_mapping_check':'exclusive visible LiDAR points of manuallyresolvedmask vs annotated objects,partial/unresolved kept','no_new_method_adoption':'GT postseal diagnostics and counterfactuals,not targetperformance confirmation'})
 print('BLIND24_READY')
if __name__=='__main__':main()
