"""Project actual extremal returns and display the paired RGB evidence."""
from pathlib import Path
import json
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "3eed_qwen_completion_side_20261121"
sys.path.insert(0, str(PARENT))
from side import (BASE, QWEN, ETA, TOL, crop_photo, digest, hint, inputs, parse, read, write)

PRIVATE, OUT, SPEC = HERE/"private", HERE/"results", HERE/"PROTOCOL.json"
PANE = 448
MODES = ("span", "endpoint", "mismatch")
ENCODINGS = ("forward","reverse")
COLORS = ((0,235,255), (255,114,0))  # negative (cyan), positive (orange)
SWAP = {"positive":"negative", "negative":"positive", "balanced":"balanced"}


def ancestry():
    spec = read(SPEC)
    a = PARENT/"results/input_seal.json"
    b = PARENT/"results/prediction_seal.json"
    assert digest(a)==spec["parent_inputs"] and digest(b)==spec["parent_predictions"]
    assert digest(PARENT/"private/prepared.json")==read(a)["prepared"]
    assert digest(PARENT/"private/predictions.json")==read(b)["predictions"]
    assert not read(b)["target_labels_read"]
    panel, frozen = inputs()
    return panel, frozen, read(PARENT/"private/prepared.json"), read(PARENT/"private/predictions.json")


def fit_view(rect, pane):
    left,top,right,bottom = map(int,rect)
    scale = PANE/max(right-left,bottom-top)
    w,h = max(1,round((right-left)*scale)),max(1,round((bottom-top)*scale))
    return dict(rect=[left,top,right,bottom], wh=[w,h],
                corner=[pane*PANE+(PANE-w)//2,(PANE-h)//2])


def crop_rect(pixel, size):
    size=np.array(size)
    win=np.array(pixel)*np.tile(size,2)/1000
    pad=(win[2:]-win[:2])/8
    return np.r_[np.maximum(0,np.floor(win[:2]-pad)),np.minimum(size,np.ceil(win[2:]+pad))].astype(int)


def display_uv(uv, view):
    rect=np.array(view["rect"]); q=np.asarray(uv)
    if not ((q>=rect[:2]) & (q<rect[2:])).all():
        return None
    # PIL resize maps the centers of source pixels to centers of resized pixels.
    return (q-rect[:2]+.5)*np.array(view["wh"])/(rect[2:]-rect[:2])-.5+view["corner"]


def displayed_distance(pair, views):
    separated=[]
    for view in views:
        a,b=(display_uv(p,view) for p in pair)
        if a is not None and b is not None:
            separated.append(float(np.linalg.norm(a-b)))
    return max(separated, default=0.)


def misplaced(pair, size, crop):
    """Translate a pair within the same crop-membership tiles, preserving its vector."""
    p=np.asarray(pair)
    low=np.zeros_like(p)
    high=np.zeros_like(p)
    for j in range(2):
        edges=np.unique([0.,float(crop[j]),float(crop[j+2]),float(size[j])])
        idx=np.searchsorted(edges,p[:,j],side="right")-1
        low[:,j]=edges[idx]
        high[:,j]=np.nextafter(edges[idx+1],-np.inf)
    left=(low-p).max(0); right=(high-p).min(0)
    assert (left<=0).all() and (right>=0).all()
    limit=(np.array(crop[2:])-crop[:2])/2
    # Keep away from the exclusive upper edge, even after floating-point addition.
    delta=np.clip(left+right,-limit,limit)
    delta=np.clip(delta,np.minimum(0,left+1.e-7),np.maximum(0,right-1.e-7))
    wrong=p+delta
    np.testing.assert_allclose(wrong[1]-wrong[0],p[1]-p[0],atol=1e-9,rtol=0.)
    return wrong


def source_endpoints(row, entry, g, size, spec, transform):
    from scipy.spatial import cKDTree
    from geometry import (WAYMO_ANGLES, project_model_points, raw_to_model_points,
                          rigid_xyz, sensor_pixels, z_rotation)
    path=Path(row["points"])
    assert digest(path)==row["point_sha"]
    raw=(np.load(path,allow_pickle=False) if path.suffix==".npy" else np.fromfile(path,"<f4").reshape(-1,4))
    xyz=np.asarray(raw[:,:3],float)
    ids=np.random.default_rng(row["seed"]).choice(len(raw),16384,replace=len(raw)<16384)
    draws=xyz[ids].astype(np.float32)
    if row["target"]=="waymo":
        rot=z_rotation(WAYMO_ANGLES[row["suffix"]])
        draws=(draws@rot.T).astype(np.float32).astype(float)@rot
    elif row["target"]=="drone":
        draws[:,2]+=np.float32(1.8)
        draws=draws.astype(float)
        draws[:,2]-=1.8
    support=np.asarray(entry["support_raw"])
    err,indices=cKDTree(draws).query(support)
    assert err.max()<1.e-7
    seed_ids=ids[indices]
    station=(support-np.asarray(entry["boxes"]["H"][:3]))@g["axis"]
    levels=(float(station.min()),float(station.max()))
    assert abs(levels[1]-levels[0]-g["measured"][0])<TOL
    p=project_model_points(raw_to_model_points(xyz,row["target"],row["suffix"]),spec,size)
    eligible=np.flatnonzero(p["in_image"])
    w=int(size[0])
    pixels=np.floor(p["uv"][eligible]).astype(int)
    cell=pixels[:,0]+w*pixels[:,1]
    # Only store populated pixels. Exact image pixels retain the nearest observed depth.
    cells,group=np.unique(cell,return_inverse=True)
    zmin=np.full(len(cells),np.inf)
    np.minimum.at(zmin,group,p["depth"][eligible])
    ends=[]
    for endpoint in levels:
        ties=np.flatnonzero(abs(station-endpoint)<=TOL)
        candidates=sorted(set(int(seed_ids[i]) for i in ties))
        visible=[]
        best_status="behind_or_outside"
        for raw_id in candidates:
            if not p["in_image"][raw_id]:
                continue
            uv=p["uv"][raw_id]
            x,y=np.floor(uv).astype(int)
            pos=np.searchsorted(cells,x+w*y)
            assert cells[pos]==x+w*y
            gap=float(p["depth"][raw_id]-zmin[pos])
            if gap>.05:
                best_status="closer_return_same_pixel"
                continue
            independent=sensor_pixels(rigid_xyz(xyz[[raw_id]],transform),spec)[0]
            dif=float(np.linalg.norm(independent-uv))
            assert dif<1.e-6
            raw_station=float((xyz[raw_id]-np.array(entry["boxes"]["H"][:3]))@g["axis"])
            assert abs(raw_station-endpoint)<1.e-4
            visible.append(dict(status="ok",raw_index=raw_id,uv=uv.tolist(),
                                reprojection_error_px=dif,station_error_m=abs(raw_station-endpoint),
                                depth_above_nearest_m=gap))
        ends.append(visible[0] if visible else dict(status=best_status))
    return ends, float(err.max())


def canvas(image, views, pair=None, reverse=False):
    sheet=Image.new("RGB",(2*PANE,PANE),(60,60,60))
    for view in views:
        patch=image.crop(tuple(view["rect"])).resize(tuple(view["wh"]),Image.Resampling.BICUBIC)
        if pair is not None:
            pen=ImageDraw.Draw(patch)
            for j,uv in enumerate(pair):
                pos=display_uv(uv,view)
                if pos is None:
                    continue
                x,y=np.rint(pos-view["corner"]).astype(int)
                pen.ellipse((x-5,y-5,x+5,y+5),outline=(32,32,32),width=1)
                pen.ellipse((x-4,y-4,x+4,y+4),outline=COLORS[1-j if reverse else j],width=2)
        sheet.paste(patch, tuple(view["corner"]))
    return sheet


def message(row, saved, reverse):
    direction=saved["legend"]
    if reverse:
        direction=direction.replace("The positive direction runs","The negative direction runs").replace(
                                    "Negative is the opposite.","Positive is the opposite.")
    span=hint(saved["geometry"]["measured"],"Measured LiDAR span")
    return ("Left: whole RGB frame. Right: enlarged crop of the referred object. Both are the same photograph. "
            "The referring sentence (quoted data) is "+json.dumps(row["text"],ensure_ascii=False)+". "
            "A laser has measured a span of this object on the stated horizontal 3D axis. "+direction+" "+span+
            " Where endpoint rings are shown, cyan marks the negative end of the measured span and orange the positive end. "
            "Use the actual visible outline of the referred object in the photo, not the crop border. Look beyond EACH "
            "endpoint along its end of the axis. If the visible body continues farther beyond the positive end, choose "
            "positive; if farther beyond the negative end, choose negative. Choose balanced only with evidence for "
            "comparable missing length at both ends. If rings are absent, use the visible object and the span. When "
            "the points could be on a distractor, the outline is occluded, depth makes the comparison ambiguous, or the "
            "side cannot be determined, choose unknown. Return only JSON with one key side and a value of "
            "positive, negative, balanced, or unknown.")
