"""Independent yaw-box IoU: 2D polygon clipping times exact z overlap.

The task scorer uses general 3D convex intersection. This audit uses a separate
algorithm and coordinate transformation implementation after prediction seals.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

def sha(p):return hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()
def cross(a,b):return float(a[0]*b[1]-a[1]*b[0])
def corners(b):
    half=b[3:5]/2;p=np.array([[-half[0],-half[1]],[half[0],-half[1]],[half[0],half[1]],[-half[0],half[1]]])
    c,s=np.cos(b[6]),np.sin(b[6]);return p@np.array([[c,-s],[s,c]]).T+b[:2]
def clip(poly,other):
    for a,b in zip(other,np.roll(other,-1,axis=0)):
        if not len(poly):break
        output=[];previous=poly[-1];edge=b-a
        before=cross(edge,previous-a)
        for current in poly:
            after=cross(edge,current-a);pin=before>=-1e-10;cin=after>=-1e-10
            if pin!=cin:
                den=before-after
                if abs(den)>1e-20:output.append(previous+(current-previous)*(before/den))
            if cin:output.append(current)
            previous=current;before=after
        poly=np.asarray(output,dtype=float).reshape(-1,2)
    return poly
def area(p):
    if len(p)<3:return 0.
    q=p-p.mean(0);return abs(sum(cross(a,b) for a,b in zip(q,np.roll(q,-1,axis=0))))/2
def iou(a,b):
    if not np.isfinite(a).all() or np.any(a[3:6]<=0):return 0.
    assert np.isfinite(b).all() and np.all(b[3:6]>0)
    xy=area(clip(corners(a),corners(b)));z=max(0.,min(a[2]+a[5]/2,b[2]+b[5]/2)-max(a[2]-a[5]/2,b[2]-b[5]/2))
    intersection=xy*z;return float(intersection/(np.prod(a[3:6])+np.prod(b[3:6])-intersection))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--predictions',type=Path,required=True);ap.add_argument('--scores',type=Path,required=True);ap.add_argument('--index',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    # Exact analytic fixtures, independent of any task labels.
    unit=np.array([0.,0,0,2,2,2,0]);assert abs(iou(unit,unit)-1)<1e-12
    shifted=unit.copy();shifted[0]=1;assert abs(iou(shifted,unit)-1/3)<1e-12
    shifted[0]=3;assert iou(shifted,unit)==0
    seal=json.loads((a.predictions/'sealed.json').read_text());scores=json.loads(a.scores.read_text())
    assert sha(a.predictions/'sealed.json')==scores['prediction_seal_sha256']
    assert sha(a.predictions/'registered_plan.json')==seal['plan_sha256']
    samples=json.loads(a.index.read_text());assert len(samples)==seal['n']==scores['n']==len(seal['files'])
    for file,digest in seal['files'].items():assert sha(a.predictions/file)==digest
    given={(r['id'],r['arm']):r['iou'] for r in scores['rows']};errors=[];rows=[]
    for sample in samples:
        assert sample['object_index']==0
        metadata=json.loads(Path(sample['meta_path']).read_text());gt=np.array(metadata['ground_info'][0]['bbox_3d'][:7],float)
        if sample['platform']=='drone':gt[2]+=1.8
        if sample['platform']=='waymo':
            degrees=[0,-45,45,-90,90][sample['Waymo_view_suffix']];gt[:3]=Rotation.from_euler('z',degrees,degrees=True).as_matrix()@gt[:3];gt[6]+=np.deg2rad(degrees)
        with np.load(a.predictions/(sample['id']+'.npz'),allow_pickle=False) as data:
            for arm in seal['arms']:
                box=data[arm].astype(float);assert box.shape==(7,)
                value=iou(box,gt);errors.append(abs(value-given[sample['id'],arm]));rows.append({'id':sample['id'],'arm':arm,'iou':value,'acc25':value>.25,'acc50':value>.5})
    assert max(errors,default=0)<1e-6,max(errors)
    summaries=[{'arm':arm,'n':len(samples),'mean_iou':float(np.mean([r['iou'] for r in rows if r['arm']==arm])),'acc25':sum(r['acc25'] for r in rows if r['arm']==arm),'acc50':sum(r['acc50'] for r in rows if r['arm']==arm)} for arm in seal['arms']]
    payload={'all_prediction_seals_checked_before_numeric_targets':True,'independent_2D_clipping_times_height':True,'analytic_fixtures_passed':True,'max_error_vs_3D_convex_scorer':max(errors),'n':len(samples),'rows':rows,'summaries':summaries}
    with a.output.open('x') as f:json.dump(payload,f,indent=2)
    print(json.dumps({k:v for k,v in payload.items() if k!='rows'},indent=2))

if __name__=='__main__':main()
