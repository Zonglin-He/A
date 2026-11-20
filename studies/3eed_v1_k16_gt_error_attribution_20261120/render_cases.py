"""Make private observation plates from calibrated RGB, source annotations and raw returns."""
import html
import itertools
import textwrap
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageOps

from cases import PRIV,read,raw_points,inside
from crossplatform.round7.camera_projection import CameraSpec,read_camera_allowlist,project_model_points

COL={'T':'#21d19f','G':'#559aff','B':'#ff8e3b','O':'#d97adb'}
INK='#20252b'
BG='#f6f5f2'
PANEL='#20262d'
SMALL=ImageFont.truetype('DejaVuSans.ttf',15)
FONT=ImageFont.truetype('DejaVuSans.ttf',19)
BOLD=ImageFont.truetype('DejaVuSans-Bold.ttf',22)
SIGNS=np.asarray(list(itertools.product((-1,1),repeat=3)))
EDGES=[(i,j) for i in range(8) for j in range(i) if np.sum(SIGNS[i]!=SIGNS[j])==1]


def corners(b):
    o=SIGNS*b[3:6]/2
    c,s=np.cos(b[6]),np.sin(b[6])
    return np.c_[c*o[:,0]-s*o[:,1],s*o[:,0]+c*o[:,1],o[:,2]]+b[:3]


def hull(points):
    p=sorted(set((round(float(a),3),round(float(b),3)) for a,b in points))
    if len(p)<3:return p
    def lower(p):
        h=[]
        for a in p:
            while len(h)>=2 and (h[-1][0]-h[-2][0])*(a[1]-h[-2][1])-(h[-1][1]-h[-2][1])*(a[0]-h[-2][0])<=0:
                h.pop()
            h.append(a)
        return h[:-1]
    return lower(p)+lower(reversed(p))


def region(uv,w,h):
    pts=uv[np.isfinite(uv).all(1)]
    if not len(pts):return (0,0,w,h)
    lo=pts.min(0); hi=pts.max(0)
    cen=(lo+hi)*.5
    wh=np.maximum(hi-lo,[40,35])*1.55
    wh=np.minimum(np.maximum(wh,[110,75]),[w,h])
    left=np.clip(cen-wh/2,[0,0],np.array([w,h])-wh)
    return tuple((np.r_[left,left+wh]).astype(int))


def fit(canvas,img,xy,wh,label):
    base=Image.new('RGB',wh,'#161b20')
    scaled=ImageOps.contain(img,wh,Image.Resampling.LANCZOS)
    base.paste(scaled,((wh[0]-scaled.width)//2,(wh[1]-scaled.height)//2))
    canvas.paste(base,xy)
    d=ImageDraw.Draw(canvas)
    d.rectangle([xy,(xy[0]+len(label)*9+14,xy[1]+22)],fill='#19232d')
    d.text((xy[0]+5,xy[1]+2),label,font=SMALL,fill='white')


def render(path):
    meta=read(path); row=meta['data']; platform=row['platform']
    gt_all=np.asarray(meta['annotations'])
    gt=gt_all[0]
    boxes=np.asarray(meta['raw_boxes'])
    observed=raw_points(row)
    g,b=boxes[[meta['good'],meta['bad']]]
    im=Image.open(row['imagepath']).convert('RGB')
    spec=CameraSpec.from_camera_fields(platform,read_camera_allowlist(row['metapath']),
             waymo_view_suffix=meta['view'],missing_distortion_policy='assume_zero_declared')
    def uv(b):
        pr=project_model_points(corners(b),spec,im.size)
        return pr['uv']
    outline=im.copy(); d=ImageDraw.Draw(outline)
    def draw_box(b,color,width):
        p=uv(b)
        for i,j in EDGES:
            if np.isfinite(p[[i,j]]).all() and abs(p[[i,j]]).max()<1e5:
                d.line([tuple(p[i]),tuple(p[j])],fill=color,width=width)
    # Show other source annotations in the same local 3D neighborhood only.
    r=max(4.,np.linalg.norm(g[:3]-b[:3])+np.linalg.norm(gt[3:6])*.7)
    other=[a for a in gt_all[1:] if min(np.linalg.norm(a[:3]-b[:3]),np.linalg.norm(a[:3]-gt[:3]))<r]
    for a in other:draw_box(a,COL['O'],1)
    for k,v in [('G',g),('T',gt),('B',b)]:draw_box(v,COL[k],3)
    d2=read(row['metapath'])['ground_info'][row['object_index']].get('bbox_2d_proj')
    if d2 is not None and len(d2)==4:
        a,c=np.asarray(d2).reshape(2,2)
        if (c>a).all():
            d.rectangle([tuple(a),tuple(c)],outline='#f4ff8a',width=1)
    focus=np.vstack((uv(gt),uv(g),uv(b)))
    crop=region(focus,*im.size)
    local_orig=im.crop(crop); local_mark=outline.crop(crop)
    near_bad=region(uv(b),*im.size)
    W,H=1500,1090
    page=Image.new('RGB',(W,H),BG);p=ImageDraw.Draw(page)
    p.text((12,8),meta['case_id']+'  |  '+platform.upper(),font=BOLD,fill=INK)
    p.text((225,12),'Green: target GT   |   Blue: highest-scored qualifying box   |   Orange: failed winner   |   Violet: nearby annotations',font=SMALL,fill=INK)
    for i,line in enumerate(textwrap.wrap(row['caption'],width=153)[:4]):
        p.text((12,39+i*19),line,font=SMALL,fill=INK)
    fit(page,im,(8,109),(735,278),'Full source RGB')
    fit(page,outline,(756,109),(735,278),'Calibrated projection')
    fit(page,local_orig,(8,398),(488,250),'Unmarked local region')
    fit(page,local_mark,(505,398),(488,250),'Matched local evidence')
    fit(page,im.crop(near_bad),(1002,398),(489,250),'Unmarked failed-box crop')
    center=gt[:3]
    c,s=np.cos(gt[6]),np.sin(gt[6])
    def local(x):
        v=x-center
        return np.c_[c*v[:,0]+s*v[:,1],-s*v[:,0]+c*v[:,1],v[:,2]]
    cc={k:local(corners(v)) for k,v in [('G',g),('B',b),('T',gt)]}
    xx=local(observed)
    target_pts=inside(observed,gt); bad_pts=inside(observed,b)
    cols=[('#86909b',~(target_pts|bad_pts)),('#299981',target_pts&~bad_pts),
          ('#f08033',bad_pts&~target_pts),('#f4e187',target_pts&bad_pts)]
    hulls=np.concatenate(list(cc.values()))
    bounds=np.vstack([hulls.min(0),hulls.max(0)])
    pad=np.maximum((bounds[1]-bounds[0])*.28,[1,.9,.5])
    bounds[0]-=pad;bounds[1]+=pad
    live=np.all((xx>=bounds[0])&(xx<=bounds[1]),axis=1)
    for n,(dims,title) in enumerate([([0,1],'Overhead'),([0,2],'Along target X / height'),([1,2],'Along target Y / height')]):
        size=(488,340); sub=Image.new('RGB',size,PANEL);ds=ImageDraw.Draw(sub)
        lo,hi=bounds[:,dims]; zoom=min((size[0]-38)/(hi[0]-lo[0]),(size[1]-40)/(hi[1]-lo[1]))
        def scale(z):
            q=(z[:,dims]-(hi+lo)/2)*zoom
            return np.c_[size[0]/2+q[:,0],size[1]/2-q[:,1]]
        pts=scale(xx)
        for color,mask in cols:
            for x,y in pts[mask&live]:ds.point((x,y),fill=color)
        for a in other:
            co=hull(scale(local(corners(a))))
            ds.line(co+co[:1],fill=COL['O'],width=1)
        for k in ('G','T','B'):
            co=hull(scale(cc[k]))
            ds.line(co+co[:1],fill=COL[k],width=3 if k=='T' else 2)
        ds.line((12,size[1]-14,12+zoom,size[1]-14),fill='white',width=2)
        ds.text((12,size[1]-35),'1 m',fill='white',font=SMALL)
        page.paste(sub,(8+497*n,661));ds=ImageDraw.Draw(page)
        ds.text((14+497*n,666),title,fill='white',font=SMALL)
    p=ImageDraw.Draw(page)
    p.text((14,1013),'Dots: grey surrounding LiDAR; turquoise target-only; orange failed-box-only; yellow shared. Box membership is geometric.',font=SMALL,fill=INK)
    p.text((14,1041),'GT label rectangles (when supplied) in pale yellow. Point-cloud views share a local target frame with the same metric scale per panel.',font=SMALL,fill=INK)
    directory=PRIV/'plates';directory.mkdir(exist_ok=True)
    page.save(directory/(meta['case_id']+'.jpg'),quality=94)
    # Compact read-only review tiles retain the local original, overlay and BEV at useful scale.
    tile=Image.new('RGB',(760,477),BG);dt=ImageDraw.Draw(tile)
    dt.text((7,4),meta['case_id']+'   '+platform,font=BOLD,fill=INK)
    fit(tile,local_orig,(3,33),(375,197),'RGB')
    fit(tile,local_mark,(382,33),(375,197),'RGB + boxes')
    fit(tile,page.crop((8,661,496,1001)),(3,235),(375,238),'Top')
    fit(tile,page.crop((505,661,993,1001)),(382,235),(375,238),'Side')
    return tile


def main():
    files=sorted(PRIV.glob('S[0-9][0-9][0-9].json'))
    assert len(files)==28
    tiles=[render(path) for path in files]
    for j in range(0,len(tiles),4):
        sheet=Image.new('RGB',(1520,954),BG)
        for k,img in enumerate(tiles[j:j+4]):
            sheet.paste(img,((k%2)*760,(k//2)*477))
        sheet.save(PRIV/f'panel_{j//4+1}.jpg',quality=94)
    page=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>K16 逐例图册</title><style>body{font:16px system-ui;max-width:1500px;margin:1.5rem auto;color:#20252b} img{max-width:100%;height:auto} nav{line-height:2.1} section{margin:3rem 0} a{color:#245789}</style>',
          '<h1>K16 源域逐例图册</h1><p>绿为主体真值，蓝为同一读出得分最高的达标框，橙为实际失败赢家，紫为附近其他已标注物体。每页含原图、局部图及三种点云投影。</p>',
          '<nav>'+' · '.join('<a href="#'+p.stem+'">'+p.stem+'</a>' for p in files)+'</nav>']
    for path in files:
        m=read(path)
        page+=['<section id="'+path.stem+'"><h2>'+path.stem+'</h2><p>'+html.escape(m['data']['caption'])+'</p>',
               '<a href="plates/'+path.stem+'.jpg"><img loading="lazy" src="plates/'+path.stem+'.jpg"></a></section>']
    (PRIV/'index.html').write_text(''.join(page)+'</html>')
    print('Wrote private observation plates',len(tiles))


if __name__=='__main__':
    main()
