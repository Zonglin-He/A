"""Private fixed center-slot contact sheets, not public case media."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_native_instance_common_v1 import read,write,sha
B=ROOT/'artifacts/stvg_weightfree_coordinate_v1/confirmation'
def main():
 from PIL import Image,ImageDraw
 rr=read(B.parent/'CONFIRMATION_ROSTER_LOCK.json')['records'];items=[];out=B/'private_visual';out.mkdir(exist_ok=True)
 for k in range(4):
  im=Image.new('RGB',(4*448,2*482),'white');d=ImageDraw.Draw(im)
  for j,r in enumerate(rr[k*8:(k+1)*8]):
   c=r['probe_alias'];p=B/'teacher_inputs'/c/'preview_2.png';x=(j%4)*448;y=(j//4)*482
   if p.exists():
    z=read(B/'teacher'/f'{c}_diverse8_2.json')['parsed'];pic=Image.open(p);im.paste(pic,(x,y+30));d.text((x+8,y+8),c+' slot2 selected '+str(z['selected_candidate_id']),fill='black')
   else:d.text((x+8,y+8),c+' no eligible four-window support',fill='black')
  p=out/f'contact{k+1}.png';im.save(p);items.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p)))
 write(B/'PRIVATE_VISUAL_EXPORT.json',dict(status='rendered_not_yet_viewed',files=items,scope='fixed center slot2 per source, at most32 of128 windows; no query/GT overlay; not all-frame review'))
if __name__=='__main__':main()
