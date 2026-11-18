"""Post-seal private identity atlas for the predeclared weighted draw."""
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
import textwrap

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "research/round156"
DATA = ROOT / "crossplatform/round156"
OUT = DATA / "atlas_PRIVATE"
sys.path.insert(0, str(ROOT / "research/round154"))
sys.path.insert(0, str(ROOT / "crossplatform/round7"))
from camera_projection import CameraSpec, project_model_points, raw_to_model_points, read_camera_allowlist
from score154 import ground_truth

COLORS = [(0, 193, 255), (255, 112, 5), (16, 200, 61)]
NAMES = ["A", "D", "GT"]
EDGES = [(a, b) for a in range(8) for b in range(a+1, 8) if (a ^ b).bit_count() == 1]
SIDES = np.array(list(itertools.product([-1.,1.], repeat=3)))
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT = ImageFont.truetype(FONT_PATH, 16)
SMALL = ImageFont.truetype(FONT_PATH, 14)


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def corner_points(box):
    co, si = np.cos(box[6]), np.sin(box[6])
    rot = np.array([[co, -si, 0.], [si, co, 0.], [0., 0., 1.]])
    return (SIDES*(box[3:6]/2.)) @ rot.T + box[:3]


def rect_uv(uv, width, height):
    finite = uv[np.isfinite(uv).all(axis=1)]
    if not len(finite):
        return None
    x0,y0 = np.maximum(finite.min(0), [0., 0.])
    x1,y1 = np.minimum(finite.max(0), [width, height])
    if x0 >= x1 or y0 >= y1:
        return None
    return np.array([x0,y0,x1,y1])


def crop_rect(rect, size, size_out, margin):
    w,h = size
    x0,y0,x1,y1 = rect
    width = max(100., (x1-x0)*margin)
    height = max(80., (y1-y0)*margin)
    ratio = size_out[0]/size_out[1]
    width,height = max(width,height*ratio), max(height,width/ratio)
    width,height = min(w,width), min(h,height)
    left = min(max(0.,(x0+x1-width)/2.),w-width)
    top = min(max(0.,(y0+y1-height)/2.),h-height)
    return tuple(map(round, [left, top, left+width, top+height]))


def rgb_panel(rgb, uv, indexes, region, size, linewidth=2):
    left,top,right,bottom = region
    ratio = min(size[0]/max(1,right-left),size[1]/max(1,bottom-top))
    shape = (max(1,round(ratio*(right-left))),max(1,round(ratio*(bottom-top))))
    small = rgb.crop(region).resize(shape, Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(small)
    for i in indexes:
        pix = (uv[i] - [left,top])*ratio
        for a,b in EDGES:
            v = pix[[a,b]]
            if np.isfinite(v).all() and np.abs(v).max() < 1.e6:
                draw.line([tuple(x) for x in v], fill=COLORS[i], width=linewidth)
    panel = Image.new("RGB", size, "white")
    panel.paste(small,((size[0]-shape[0])//2,(size[1]-shape[1])//2))
    return panel


def bird(points, boxes, corners, size, wide):
    image = Image.new("RGB",size,(245,246,247))
    c = np.concatenate([v[:,:2] for v in corners])
    if wide:
        lo,hi = np.quantile(points[:,:2],[.01,.99],axis=0)
        lo,hi = np.minimum(lo,c.min(0)-2),np.maximum(hi,c.max(0)+2)
    else:
        lo,hi = c.min(0)-1.5,c.max(0)+1.5
    center = (lo+hi)/2
    ratio = min((size[0]-20)/(hi[0]-lo[0]),(size[1]-20)/(hi[1]-lo[1]))
    def project(xy):
        return (xy-center)*[ratio,-ratio] + [size[0]/2,size[1]/2]
    draw = ImageDraw.Draw(image)
    in_view = np.all((points[:,:2]>=lo) & (points[:,:2]<=hi),axis=1)
    px = points[in_view]
    if len(px) > 60000:
        px = px[::int(np.ceil(len(px)/60000))]
    pq = np.rint(project(px[:,:2])).astype(int)
    # draw asphalt points lighter, elevated returns darker
    if len(px):
        bot = np.quantile(px[:,2], .25)
        for selector, color in ((px[:,2]<bot+.15, (196,200,202)), (px[:,2]>=bot+.15,(93,96,101))):
            draw.point(list(map(tuple,pq[selector])),fill=color)
    for i,v in enumerate(corners):
        ring = project(v[[0,2,6,4,0],:2])
        draw.line([tuple(p) for p in ring],fill=COLORS[i],width=2)
    return image


def sample(rows):
    rows = sorted(rows,key=lambda r:int(r["ordinal"]))
    assert [int(r["ordinal"]) for r in rows] == list(range(1,1799))
    d = np.array([float(r["D_pca"])-float(r["A_pca"]) for r in rows])
    w = np.array([float(r["sampling_weight"]) for r in rows])
    directions = [(r["source"],r["target"]) for r in rows]
    for key in set(directions):
        ix = np.array([k==key for k in directions])
        w[ix] /= 6*w[ix].sum()
    rng = np.random.default_rng(156)
    positive=np.maximum(d,0.)*w
    negative=np.maximum(-d,0.)*w
    pos=rng.choice(len(rows),36,replace=True,p=positive/positive.sum())
    neg=rng.choice(len(rows),6,replace=False,p=negative/negative.sum())
    keys=np.unique(np.r_[pos,neg])
    order=np.random.default_rng(15699).permutation(keys)
    return rows,pos,neg,order


def render(case, obs, boxes, output):
    row = obs["row"]
    assert not obs["protected_alias_fallback"] and obs["cohort"] == "EXT"
    sensor = obs["A_sensor"]
    assert sensor["eligible_for_extra_media"] and not sensor["protected_image_content_alias"]
    for key, sha in (("imagepath","image_sha256"),("pointpath","point_sha256")):
        assert digest(Path(row[key])) == sensor[sha]
    rgb = Image.open(row["imagepath"]).convert("RGB")
    quality = ground_truth(obs["target"],row["metapath"],row["object_index"],row["caption"])
    boxes=np.vstack([boxes,quality])
    corners = [corner_points(b) for b in boxes]
    suffix = row["suffix"] if obs["target"] == "waymo" else 0
    cam = CameraSpec.from_camera_fields(obs["target"],read_camera_allowlist(row["metapath"]),
             waymo_view_suffix=suffix,missing_distortion_policy="assume_zero_declared")
    uv = [project_model_points(pts,cam,rgb.size)["uv"] for pts in corners]
    rects = [rect_uv(u,*rgb.size) for u in uv]
    path=Path(row["pointpath"])
    raw=np.load(path,allow_pickle=False) if path.suffix == ".npy" else np.fromfile(path,np.float32).reshape(-1,4)
    cloud=raw_to_model_points(raw[:,:3],obs["target"],suffix)
    W,H=1220,815
    page=Image.new("RGB",(W,H),"white")
    draw=ImageDraw.Draw(page)
    draw.text((8,2),f"{case}   cyan A    orange D    green annotated referent   |  {obs['target']}",font=FONT,fill="black")
    for j,line in enumerate(textwrap.wrap(row["caption"],137)[:3]):
        draw.text((8,25+j*17),line,font=SMALL,fill="black")
    def put(pic,at,title):
        page.paste(pic,at)
        draw.text((at[0]+3,at[1]+pic.height+1),title,font=SMALL,fill="black")
    full=(0,0,*rgb.size)
    put(rgb_panel(rgb,uv,[0,1,2],full,(565,307)),(8,76),"Full RGB, 3D wireframes")
    put(bird(cloud,boxes,corners,(324,307),False),(577,76),"BEV local (x right, y up)")
    put(bird(cloud,boxes,corners,(307,307),True),(906,76),"BEV scene, raw LiDAR")
    for i in range(3):
        x=8+i*404
        if rects[i] is None:
            im=Image.new("RGB",(397,206),"#eeeeee")
            ImageDraw.Draw(im).text((4,90),"Not visible in calibrated image",font=FONT,fill="black")
            put(im,(x,404),NAMES[i]+" projection outside camera")
            continue
        crop=crop_rect(rects[i],rgb.size,(396,206),1.65)
        put(rgb_panel(rgb,uv,[i],crop,(396,206)),(x,404),NAMES[i]+" candidate crop" if i<2 else "Annotated referent crop")
        put(rgb_panel(rgb,uv,[],crop,(396,155)),(x,635),"Same crop without wireframe")
    page.save(output,"JPEG",quality=90)
    return [int(u is not None) for u in rects]


def main():
    audit=read(HERE/"results"/"PAIR_SCORE_AUDIT.json")
    assert audit["status"]=="PASS" and audit["n"]==1798
    assert audit["prediction_sha256"]==digest(DATA/"PAIR_PRIVATE.npz")
    registry=read(ROOT/"research/round145/INPUT_REGISTRY_PRIVATE.json")
    npz=DATA/"PAIR_PRIVATE.npz"
    with np.load(npz,allow_pickle=False) as v: boxes=np.array(v["boxes"])
    assert boxes.shape==(1798,2,3,7)
    table=HERE/"results/pair_rows.csv"
    with table.open() as f: rows,pos,neg,order=sample(list(csv.DictReader(f)))
    OUT.mkdir(exist_ok=True)
    mapping=[]
    for j,idx in enumerate(order,start=1):
        case=f"V{j:02d}"
        assert rows[idx]["source"]==registry[idx]["source"] and rows[idx]["target"]==registry[idx]["target"]
        proj=render(case, registry[idx],boxes[idx,:,1],OUT/(case+".jpg"))
        mapping.append({"case":case,"ordinal":int(idx+1),"positive_draws":int(np.sum(pos==idx)),
                      "negative_draws":int(np.sum(neg==idx)),"camera_visible":proj})
    for start in range(0,len(mapping),4):
        chosen=mapping[start:start+4]
        page=Image.new("RGB",(1900,1270),"white")
        for n,entry in enumerate(chosen):
            im=Image.open(OUT/(entry["case"]+".jpg"))
            page.paste(im.resize((950,635),Image.Resampling.LANCZOS),((n%2)*950,(n//2)*635))
        page.save(OUT/f"CONTACT{start//4+1:02d}.jpg","JPEG",quality=92)
    (OUT/"MAPPING_PRIVATE.json").write_text(json.dumps(mapping,indent=2)+"\n")
    print("PRIVATE_VISUAL_GALLERY_READY",len(mapping),OUT)

if __name__=="__main__":
    main()
