"""Create a score-blind recording-balanced review sheet from frozen-window RGB crops."""
from collections import defaultdict
import hashlib
from PIL import Image,ImageDraw
from common import HERE, OUT, PRIVATE, SPEC, crop_photo, digest, inputs, read, write

def key(*args):
    return hashlib.sha256(('/'.join(('visible-span-review-v1',*args))).encode()).hexdigest()

def sheets():
    panel, frozen=inputs()
    selected=[]
    for platform in ('waymo','drone','quad'):
        by=defaultdict(list)
        for row in panel:
            if row['target']==platform and frozen[row['uid']]['selected']:
                by[row['recording']].append(row)
        recs=sorted(by,key=lambda r:key(platform,r))
        for rs in by.values():
            rs.sort(key=lambda r:key(r['uid']))
        n=0
        while n<8:
            for rec in recs:
                if by[rec] and n<8:
                    selected.append(by[rec].pop(0))
                    n+=1
    selected.sort(key=lambda r:key('order',r['uid']))
    roster={}
    for start in range(0,len(selected),4):
        sheet=Image.new('RGB',(740,570),(44,44,44))
        canvas=ImageDraw.Draw(sheet)
        for j,row in enumerate(selected[start:start+4]):
            code=f'L{start+j+1:02d}';uid=row['uid']
            img,crop=crop_photo(row,frozen[uid])
            w,h=img.size
            scale=min(320/w,220/h)
            thumb=img.resize((max(1,round(w*scale)),max(1,round(h*scale))),Image.Resampling.BICUBIC)
            ox=370*(j%2)+25+(320-thumb.width)//2
            oy=285*(j//2)+32+(220-thumb.height)//2
            sheet.paste(thumb,(ox,oy))
            for step in range(5):
                frac=step/4
                tx=ox+round(frac*thumb.width);ty=oy+round(frac*thumb.height)
                canvas.line((tx,oy-4,tx,oy-1),fill='white')
                canvas.line((ox-4,ty,ox-1,ty),fill='white')
            canvas.text((j%2*370+15,j//2*285+8),f'{code}    {w} x {h} native',fill='white')
            roster[code]=dict(uid=uid,crop=crop,display=[ox,oy,thumb.width,thumb.height])
        sheet.save(PRIVATE/f'blind_{start//4+1}.png')
    write(PRIVATE/'review_roster.json',roster)

if __name__=='__main__':
    sheets()
