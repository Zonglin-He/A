"""Convert an observed silhouette and a calibrated 3D extension into a frozen center choice."""
import json
import re
from pathlib import Path

import cv2
import numpy as np

from endpoints import HERE, OUT, PRIVATE, ancestry, digest, read, write
from geometry import camera, rigid_xyz, sensor_pixels

ARMS=("outline","outline_wrong","outline_rect")
SIDES=("negative","balanced","positive")


def turn(a,b,c):
    u,v=b-a,c-a
    return u[0]*v[1]-u[1]*v[0]


def segment_contact(a,b,c,d):
    if np.any(np.maximum(np.minimum(a,b),np.minimum(c,d))>np.minimum(np.maximum(a,b),np.maximum(c,d))):
        return False
    o=[turn(a,b,c),turn(a,b,d),turn(c,d,a),turn(c,d,b)]
    return o[0]*o[1]<=0 and o[2]*o[3]<=0


def silhouette(answer, rect, mirror):
    clean=re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$","",answer.strip(),flags=re.I)
    try:
        doc=json.loads(clean)
        if not isinstance(doc,dict) or set(doc)!={"polygon"} or doc["polygon"] is None:
            return None
        points=np.asarray(doc["polygon"],dtype=float)
    except (ValueError,TypeError,OverflowError):
        return None
    if points.ndim!=2 or points.shape[1]!=2 or not np.isfinite(points).all() or np.any((points<0)|(points>1000)):
        return None
    if len(points):
        points=points[np.r_[True,(np.diff(points,axis=0)!=0).any(1)]]
    if len(points)>1 and np.array_equal(points[0],points[-1]):
        points=points[:-1]
    n=len(points)
    if n<3 or len(np.unique(points,axis=0))!=n:
        return None
    for i,(a,b) in enumerate(zip(points,np.roll(points,-1,axis=0))):
        c=points[(i+2)%n]
        if turn(a,b,c)==0 and np.dot(a-b,c-b)>0:
            return None
        for j in range(i+2,n):
            if (i!=0 or j!=n-1) and segment_contact(a,b,points[j],points[(j+1)%n]):
                return None
    if not cv2.contourArea(points.astype("float32"))>0:
        return None
    lo,hi=np.asarray(rect[:2]),np.asarray(rect[2:])-1
    if mirror:
        points[:,0]=1000-points[:,0]
    return (lo+points*(hi-lo)/1000).astype("float32")


def raw_paths(row, obs, g):
    path=Path(row["points"])
    assert digest(path)==row["point_sha"]
    cloud=(np.load(path,mmap_mode="r",allow_pickle=False) if path.suffix==".npy" else np.fromfile(path,"<f4").reshape(-1,4))
    xyz=cloud[[e["raw_index"] for e in obs["endpoints"]],:3].astype(float)
    spec,_,transform=camera(row)
    steps=np.linspace(0,g["slack_m"],65)
    paths=[]
    for i,end in enumerate(xyz):
        p=end+(2*i-1)*steps[:,None]*g["axis"]
        cam=rigid_xyz(p,transform)
        uv=np.full((len(cam),2),np.nan)
        front=cam[:,2]>.1
        if front.any():
            uv[front]=sensor_pixels(cam[front],spec)
        assert np.allclose(uv[0],obs["correct"][i],atol=1.e-6,rtol=0)
        paths.append(uv)
    return np.array(paths)


def gap_fraction(path, poly, rect):
    if not np.isfinite(path).all():
        return None,"behind_camera"
    if np.linalg.norm(path[-1]-path[0])<1.:
        return None,"subpixel_axis"
    lo,hi=np.array(rect[:2]),np.array(rect[2:])-1
    # The observed pixel centers define the viewport. Space beyond it supplies no outline evidence.
    last=None
    for i,p in enumerate(path):
        if np.any(p<lo) or np.any(p>hi):
            return None,"crop_censored"
        distance=cv2.pointPolygonTest(poly,(float(p[0]),float(p[1])),True)
        if distance<0:
            if i==0:
                return None,"start_outside"
            fraction=last/(last-distance)
            return (i-1+fraction)/(len(path)-1),"exit"
        last=distance
    return 1.,"reaches_limit"


def choice(paths,poly,rect):
    if poly is None:
        return "H","invalid_outline",None
    values=[gap_fraction(p,poly,rect) for p in paths]
    reasons=[status for x,status in values if x is None]
    if reasons:
        return "H","+".join(sorted(set(reasons))),None
    neg,pos=(x for x,_ in values)
    allocation=(1+pos-neg)/2
    distance=np.abs(np.array([0,.5,1])-allocation)
    options=np.flatnonzero(np.isclose(distance,distance.min(),rtol=0,atol=1.e-10))
    return (SIDES[options[0]] if len(options)==1 else "H",
            "observed" if len(options)==1 else "tie",[neg,pos])


def rectangle(poly):
    if poly is None:
        return None
    (x,y),(r,b)=poly.min(0), poly.max(0)
    return np.array([[x,y],[r,y],[r,b],[x,b]],dtype="float32")


def readout(pair):
    act=[x[0] for x in pair]
    if "H" in act:
        return "H","incomplete_views"
    if act[0]!=act[1]:
        return "H","mirror_conflict"
    return act[0],"agree"


def run():
    panel,_,prepared,old=ancestry()
    seal=read(OUT/"outline_raw_seal.json")
    assert seal["pins"]["protocol"]==digest(HERE/"OUTLINE_PROTOCOL.json")
    assert seal["pins"]["code"]==digest(HERE/"read_outline.py") and not seal["target_labels_read"]
    assert digest(OUT/"input_seal.json")==seal["pins"]["original_inputs"]
    assert read(OUT/"input_seal.json")["observations"]==digest(PRIVATE/"observations.json")
    observed=read(PRIVATE/"observations.json")
    manifest=read(PRIVATE/"outline_hashes.json")
    assert digest(PRIVATE/"outline_hashes.json")==seal["replies"]
    preds={}
    for row in panel:
        uid=row["uid"]; obs=observed[uid]
        fallback=(obs["status"] if obs["status"]!="eligible" else "endpoint_outside_zoom")
        preds[uid]={arm:dict(action="H",status=fallback,box=old[uid]["H"],views=[]) for arm in ARMS}
        if uid not in manifest:
            continue
        path=PRIVATE/"outline_replies"/(uid+".json")
        assert digest(path)==manifest[uid]
        raw=read(path)
        assert raw["uid"]==uid and raw["pins"]==seal["pins"]
        rect=obs["views"][1]["rect"]
        polys=[silhouette(ans,rect,i==1) for i,ans in enumerate(raw["answers"])]
        correct=raw_paths(row,obs,prepared[uid]["geometry"])
        displacement=np.subtract(obs["wrong"],obs["correct"])
        np.testing.assert_allclose(displacement[0],displacement[1],rtol=0,atol=1.e-9)
        for arm in ARMS:
            curves=correct+displacement[0] if arm=="outline_wrong" else correct
            pair=[choice(curves,rectangle(p) if arm=="outline_rect" else p,rect) for p in polys]
            label,status=readout(pair)
            box=old[uid]["H"] if label=="H" else old[uid]["discrete_boxes"][label]
            preds[uid][arm]=dict(action=label,status=status,box=box,
                                 views=[dict(action=x,reason=s,gap_fractions=v) for x,s,v in pair])
    assert len(preds)==416 and len(manifest)==239
    write(PRIVATE/"outline_predictions.json",preds)
    write(OUT/"outline_prediction_seal.json",dict(frames=len(preds),eligible=len(manifest),
           source=digest(OUT/"outline_raw_seal.json"),code=digest(__file__),
           predictions=digest(PRIVATE/"outline_predictions.json"),target_labels_read=False,FINAL_access=False))
    print("Sealed contour-based box choices on",len(preds),"frames.")


if __name__=="__main__":
    run()
