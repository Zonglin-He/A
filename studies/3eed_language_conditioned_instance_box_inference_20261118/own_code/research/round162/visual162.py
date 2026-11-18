"""Local-only image and LiDAR panels for a fixed set of prior successes/failures."""
from collections import defaultdict
import hashlib
import textwrap

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from geometry162 import axes
from manifest162 import HERE, ORDER, WORK, atomic_json, read, target_rows
from mechanisms162 import annotation_inventory, read_evidence
from score162 import table, verified

SIGNS=np.array([(x,y,z) for x in (-1,1) for y in (-1,1) for z in (-1,1)])
LINES=[(i,j) for i,a in enumerate(SIGNS) for j,b in enumerate(SIGNS) if j>i and np.count_nonzero(a!=b)==1]


def corners(box):
    return (SIGNS * box[3:6]/2) @ axes(box[6]).T + box[:3]


def projection(ev, xyz):
    import cv2
    camera=xyz @ ev.model_to_camera[:3,:3].T + ev.model_to_camera[:3,3]
    ok=camera[:,2]>.1
    uv=np.full((len(xyz),2),np.nan)
    if ok.any():
        uv[ok]=cv2.projectPoints(camera[ok],np.zeros(3),np.zeros(3),
                                 ev.camera_intrinsic,ev.distortion)[0][:,0]
    uv /= [ev.width,ev.height]
    return uv


def picture(ev, points, target_box, old, new, support, caption, header, size=(1040,460)):
    full=Image.new('RGB',size,(250,249,248))
    outf=Image.open(points).convert('RGB')
    outf.thumbnail((630,305))
    left,top=(630-outf.width)//2,35+(305-outf.height)//2
    img=ImageDraw.Draw(outf)
    colors=[('#23de66',target_box),('#49b5ff',old),('#ff6654',new)]
    for color, box in colors:
        xy=projection(ev,corners(box))*[outf.width,outf.height]
        for i,j in LINES:
            if np.isfinite(xy[[i,j]]).all():
                img.line([tuple(xy[i]),tuple(xy[j])],fill=color,width=3)
    anchor=ev.seeds[ev.anchors[ev.old_winner]:ev.anchors[ev.old_winner]+1]
    for x,y in projection(ev,anchor)*[outf.width,outf.height]:
        if np.isfinite(x+y):
            img.ellipse([x-5,y-5,x+5,y+5],fill='#fce124',outline='#000000',width=1)
    sel=np.argsort(-support,kind='stable')[:24]
    sel=sel[support[sel]>0]
    for x,y in projection(ev,ev.seeds[sel])*[outf.width,outf.height]:
        if np.isfinite(x+y):
            img.ellipse([x-2,y-2,x+2,y+2],fill='#ff8670')
    full.paste(outf,(left,top))
    draw=ImageDraw.Draw(full)
    box_font=ImageFont.truetype('DejaVuSans.ttf',14)
    title_font=ImageFont.truetype('DejaVuSans.ttf',17)
    draw.text((14,10),header,font=title_font,fill='#191a20')
    canvas=(650,35,1020,340)
    draw.rectangle(canvas,fill='#182124')
    all_center=np.vstack((old[:2],new[:2],target_box[:2],anchor[0,:2]))
    median=(all_center.min(0)+all_center.max(0))/2
    radius=max(3.,1.25*np.max(abs(all_center-median))+max(old[3],target_box[3])/2)
    scale=140/radius
    def plot(x):
        p=(x-median)*[scale,-scale]+[835,187]
        return p
    sparse=ev.points[np.all(abs(ev.points[:,:2]-median)<radius,axis=1),:2]
    for x,y in plot(sparse):
        draw.point((x,y),fill='#7f939c')
    for color,box in colors:
        xy=plot(corners(box)[::2,:2])
        polygon=xy[[0,1,3,2,0]]
        draw.line([tuple(p) for p in polygon], fill=color,width=3)
    x,y=plot(anchor[0,:2])
    draw.ellipse([x-4,y-4,x+4,y+4],fill='#fce124')
    for x,y in plot(ev.seeds[sel,:2]):
        draw.ellipse([x-2,y-2,x+2,y+2],fill='#ff8670')
    draw.text((15,346),'Green label   Blue R158   Red LCIB   Yellow old query anchor   Coral LCIB seed support',
              font=box_font,fill='#192528')
    for j,line in enumerate(textwrap.wrap(caption,width=116)[:4]):
        draw.text((15,371+j*19),line,font=box_font,fill='#292d34')
    return full


def main():
    panel=read(WORK/'AUDIT_PANEL_PRIVATE.json')
    assert set(panel)<=set(verified('DEV'))
    by_ord={int(r['ordinal']):r for r in table(HERE/'results/DEV/rows.csv')}
    entries=target_rows()
    split=defaultdict(list)
    for i in panel:
        e=entries[i]
        correct=float(by_ord[e['ordinal']]['r158_full'])>.25
        split[e['source'],e['target'],correct].append(i)
    home=WORK/'visual_audit_PRIVATE'
    home.mkdir(exist_ok=True)
    index=[]
    for s,t in ORDER:
        page=Image.new('RGB',(1040,920))
        for k,status in enumerate((True,False)):
            i=min(split[s,t,status],key=lambda j:hashlib.sha256(f'lcib-visual|{j}'.encode()).digest())
            e=entries[i]
            r=e['r']
            row=by_ord[e['ordinal']]
            name=f'{s}_to_{t}'
            with np.load(WORK/'DEV'/f'{i:05d}.npz',allow_pickle=False) as pred:
                ev=read_evidence(pred)
                _,boxes=annotation_inventory(r['metapath'],t,int(r['suffix'] or 0),int(r['object_index']))
                selected=pred['boxes'][pred['arms'].tolist().index('lcib')]
                support=pred['lcib_support']
                support=support[0] if len(support) else np.zeros(len(ev.seeds))
                message=f'{s.title()} to {t.title()}   Case {e["ordinal"]}   IoU {float(row["r158_full"]):.3f} to {float(row["lcib"]):.3f}'
                page.paste(picture(ev,r['imagepath'],boxes[0],ev.old_pca,selected,support,r['caption'],message),(0,460*k))
            index.append(dict(direction=name,ordinal=e['ordinal'],old_success=status,page=name+'.jpg'))
        page.save(home/(name+'.jpg'),quality=92)
    atomic_json(home/'INDEX.json',index)
    print('Local-only case sheets', len(index), flush=True)


if __name__=='__main__':
    main()
