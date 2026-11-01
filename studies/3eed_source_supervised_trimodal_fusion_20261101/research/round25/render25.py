import sys,textwrap,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'crossplatform/round25'))
from common25 import *
from PIL import Image,ImageDraw,ImageFont

def main():
 rmap={r['alias']:r for r in rows('N64')};sc={r['alias']:r for r in read(EXP/'T400_ROWS_PRIVATE.json') if r['stage']=='N'};selected=[]
 for p in ['drone','quad']:
  rr=[r for r in sc.values() if r['platform']==p and r['q']['fusion_rgb_seed0']!=r['q']['source']]
  rr.sort(key=lambda r:r['ious']['fusion_rgb_seed0']-r['ious']['source']);selected += [r['alias'] for r in rr[:2]+rr[-2:]]
 selected=list(dict.fromkeys(selected));out=EXP/'review_PRIVATE';out.mkdir(exist_ok=False);font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17);thumbs=[]
 for alias in selected:
  r=rmap[alias];s=sc[alias];x=rgb_feature(r);im=Image.open(r['imagepath']).convert('RGB');scale=1000/im.width;im=im.resize((1000,int(im.height*scale)));d=ImageDraw.Draw(im)
  for arm,color in [('source','cyan'),('noRGB_source_train_seed0','gray'),('fusion_rgb_seed0','orange')]:
   q=s['q'][arm]
   if x['valid'][q]:d.rectangle((x['rects'][q]*scale).tolist(),outline=color,width=3)
  canvas=Image.new('RGB',(1000,im.height+180),'white');canvas.paste(im,(0,0));d=ImageDraw.Draw(canvas)
  lines=[alias+' | cyan Source, gray noRGB, orange RGB',f"IoU: Source {s['ious']['source']:.4f}, noRGB {s['ious']['noRGB_source_train_seed0']:.4f}, RGB {s['ious']['fusion_rgb_seed0']:.4f}",'Projected predicted boxes, not identity gold.']+textwrap.wrap(r['caption'],105)[:5]
  for j,line in enumerate(lines):d.text((5,im.height+3+21*j),line,fill='black',font=font)
  canvas.save(out/(alias+'.png'));canvas.thumbnail((520,390));thumbs.append(canvas)
 sheet=Image.new('RGB',(1040,390*((len(thumbs)+1)//2)),(225,225,225))
 for i,im in enumerate(thumbs):sheet.paste(im,((i%2)*520,(i//2)*390))
 sheet.save(out/'TAIL_REVIEW.png');dump(EXP/'REVIEW_MANIFEST.json',{'aliases':selected,'rule':'two largest gains and harms per platform among new-content primary-seed ranking changes','all_negative_comparison_records_preserved_elsewhere':True,'not_identity_gold':True,'no_retraining_or_selection_from_review':True});print('REVIEW25',selected)
if __name__=='__main__':main()
