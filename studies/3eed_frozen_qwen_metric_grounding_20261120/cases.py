"""Render local-only error examples from sealed scalar results and private RGBs."""
import csv
from pathlib import Path
import textwrap
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from paths import PRIV,OUT,read,write,digest
from geometry import camera,extent_corners,cook_rotation,sensor_pixels
from score import given_rect

def main():
    rows=list(csv.DictReader((OUT/'all_rows.csv').open()))
    panel={r['uid']:r for r in read(PRIV/'panel.json')};pred=read(PRIV/'predictions.json')
    chosen=[];groups=[]
    categories=[
        ('3D rescue at 0.50',lambda r:float(r['ray_depth'])>.50>=float(r['raw'])),
        ('3D harm at 0.25',lambda r:float(r['raw'])>.25>=float(r['ray_depth'])),
        ('2D target and missed 3D',lambda r:r['id2d']=='target' and float(r['ray_depth'])<=.25),
        ('2D distractor',lambda r:r['id2d']=='other'),
        ('2D identity unresolved',lambda r:r['id2d']=='unknown'),
        ('3D rescue at 0.25',lambda r:float(r['ray_depth'])>.25>=float(r['raw']))]
    for title,test in categories:
        options=[r for r in rows if r['uid'] not in chosen and test(r)]
        if options:
            r=min(options,key=lambda x:x['uid']);chosen.append(r['uid']);groups.append((title,r))
    out=PRIV/'case_pages';out.mkdir(exist_ok=True)
    cards=[]
    font_path='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font=ImageFont.truetype(font_path,15) if Path(font_path).exists() else ImageFont.load_default()
    for title,entry in groups:
        uid=entry['uid'];r=panel[uid];p=pred[uid];spec,size,mat=camera(r)
        with Image.open(r['rgb']) as source:
            im=source.convert('RGB')
        draw=ImageDraw.Draw(im); g=read(r['calibration'])['ground_info'][r['slot']]
        supplied=given_rect(g.get('bbox_2d_proj'),size)
        def box(rect,color,width=4):
            if rect is not None:draw.rectangle(tuple(map(float,rect)),outline=color,width=width)
        def wire(b,color):
            if b is None:return
            b=np.array(b)
            v=sensor_pixels(extent_corners(b[:3],b[3:6],cook_rotation(b)),spec)
            if v is None:return
            for i in range(8):
                for bit in (1,2,4):
                    if i<(i^bit):draw.line([tuple(v[i]),tuple(v[i^bit])],fill=color,width=3)
        wire(p['camera_boxes'].get('raw'),'#dd62f5')
        wire(p['camera_boxes'].get('ray_depth'),'#44bdff')
        if p['pixel'] is not None:box(np.array(p['pixel'])*np.tile(size,2)/1000,'#ffd54f',3)
        box(supplied,'#52ff89')
        im.thumbnail((560,322))
        card=Image.new('RGB',(580,452),'#16191e');d=ImageDraw.Draw(card)
        d.text((9,5),r['target'].upper()+'   '+title,font=font,fill='white')
        card.paste(im,((580-im.width)//2,28))
        label=' '.join(r['text'].split())
        d.multiline_text((9,355),'\n'.join(textwrap.wrap(label,71))[:250],font=font,fill='#eeeeee',spacing=2)
        d.text((9,426),f"raw IoU {float(entry['raw']):.3f}    corrected {float(entry['ray_depth']):.3f}",font=font,fill='white')
        cards.append(card)
    grid=Image.new('RGB',(1160,40+452*((len(cards)+1)//2)),'#16191e')
    draw=ImageDraw.Draw(grid)
    draw.text((8,8),'Green = annotated target   Yellow = Qwen 2D   Purple = native 3D   Blue = RGB ray + LiDAR 3D',font=font,fill='white')
    for i,im in enumerate(cards):grid.paste(im,((i%2)*580,40+452*(i//2)))
    grid.save(out/'case_sheet.jpg',quality=89)
    write(OUT/'example_rows.json',[dict(group=g,uid=r['uid'],target=r['target'],iou_raw=float(r['raw']),
          iou_corrected=float(r['ray_depth']),image_identity=r['id2d']) for g,r in groups])

if __name__=='__main__':main()
