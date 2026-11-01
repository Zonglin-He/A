"""Private inspection of all seven selected hybrid-vs-G ranking changes."""
import sys,textwrap,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'crossplatform/round24'),str(ROOT/'crossplatform/round13')]
from common24 import *
from observations import load
from association import box_corners
from camera_projection import project_model_points
from PIL import Image,ImageDraw,ImageFont

def main():
 changes=read(EXP/'MECHANISM_AUDIT.json')['all_changes'];rr={r['alias']:r for r in rows('M64')};ss={r['alias']:r for r in read(EXP/'M64_ROWS_PRIVATE.json')};out=EXP/'review_PRIVATE';out.mkdir(exist_ok=False);font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15);thumbs=[]
 for x in changes:
  r=rr[x['alias']];sc=ss[x['alias']];z=feature(r);frame,scene,binding=load(r);im=Image.fromarray(frame['rgb']);scale=1000/im.width;im=im.resize((1000,int(im.height*scale)));draw=ImageDraw.Draw(im)
  for arm,col in [('Qwen_G_direct','cyan'),('selected_TTA_plus_current_G','orange')]:
   box=z['boxes'][sc['q'][arm]];pr=project_model_points(box_corners(box),scene['camera_spec'],scene['image_size_wh']);valid=pr['depth']>.1
   if valid.any():
    p=pr['uv'][valid];a=p.min(0)*scale;b=p.max(0)*scale;clip=[max(-100,min(1100,a[0])),max(-100,min(im.height+100,a[1])),max(-100,min(1100,b[0])),max(-100,min(im.height+100,b[1]))];draw.rectangle(clip,outline=col,width=3)
  canvas=Image.new('RGB',(1000,im.height+180),'white');canvas.paste(im,(0,0));d=ImageDraw.Draw(canvas);lines=[x['alias']+' cyan=direct G, orange=adapted+same G',f"IoU {x['reference_iou']:.6f} -> {x['new_iou']:.6f}; same mapped mask={x['same_dominant_RGB_instance_mask']}",'Projection/association review is NOT identity gold.']+textwrap.wrap(r['caption'],110)[:5]
  for j,line in enumerate(lines):d.text((7,im.height+4+20*j),line,fill='black',font=font)
  canvas.save(out/(x['alias']+'.png'));thumb=canvas.copy();thumb.thumbnail((520,380));thumbs.append(thumb)
 sheet=Image.new('RGB',(1040,380*((len(thumbs)+1)//2)),(225,225,225))
 for i,im in enumerate(thumbs):sheet.paste(im,((i%2)*520,(i//2)*380))
 sheet.save(out/'ALL_SEVEN_CHANGES.png');dump(EXP/'REVIEW_MANIFEST.json',{'aliases':[x['alias'] for x in changes],'all_seven_selected_changes_including_harm_retained':True,'no_policy_or_hyperparameter_change_from_visual_review':True,'projection_not_identity_gold':True,'private_media_not_public_exported':True});print('REVIEW24',len(changes))
if __name__=='__main__':main()
