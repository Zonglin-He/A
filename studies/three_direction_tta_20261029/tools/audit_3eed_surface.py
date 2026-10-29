"""Independent point-to-box-boundary candidate selection and OBB scoring."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from obb_iou import oriented_iou


def sha(p):return hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()


def points_for(s,index):
    p=s['lidar_path']
    raw=np.load(p) if p.endswith('.npy') else np.fromfile(p,np.float32).reshape(-1,4)
    idx=np.random.default_rng(20261029+index).choice(len(raw),16384,replace=len(raw)<16384)
    xyz=raw[idx,:3].copy()
    if s['platform']=='drone':xyz[:,2]+=1.8
    if s['platform']=='waymo':
        view=int(Path(p).parent.name.split('_')[-1]);q=Rotation.from_euler('z',[0,-45,45,-90,90][view],degrees=True).as_matrix()
        xyz=xyz@q.T
    return np.unique(xyz.astype(np.float32),axis=0)


def distance(points,box):
    local=(points-box[:3])@Rotation.from_euler('z',box[6]).as_matrix()
    half=box[3:6]/2
    inside=(abs(local)<=half).all(1)
    result=np.linalg.norm(local-np.clip(local,-half,half),axis=1)
    result[inside]=np.min(half-abs(local[inside]),axis=1)
    return float(result.mean()/min(box[3],box[4]))


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('output',type=Path);p.add_argument('--new36',action='store_true');a=p.parse_args()
    cohort=a.root/('replication36_surface' if a.new36 else 'surface_dev45')
    pred=cohort/('surface' if a.new36 else 'predictions')
    seal=json.loads((pred/'sealed.json').read_text())
    assert sha(cohort/'PLAN_LOCK.json')==seal['plan_sha256']
    for f,h in seal['files'].items():assert sha(pred/f)==h
    if a.new36:
        entries=[(i,s) for i,s in enumerate(json.loads((cohort/'manifest.json').read_text()))]
    else:
        entries=[]
        for path in [a.root/'manifest.json',a.root/'replication36/manifest.json']:
            entries.extend(enumerate(json.loads(path.read_text())))
    given={r['id']:r for r in json.loads((cohort/'scored.json').read_text())['rows']}
    trace={r['id']:r for r in json.loads((pred/'trace.json').read_text())}
    rows=[];errors=[];loss_errors=[]
    for index,s in entries:
        with np.load(pred/f"{s['id']}.npz",allow_pickle=False) as d:
            source,pca,surface=(d[k].copy() for k in ['source7d','pca7d','surface7d'])
        assert np.array_equal(source[:6],pca[:6]) and np.array_equal(source[:6],surface[:6])
        points=points_for(s,index);crop=points[(abs(points-source[:3])<=source[3:6]/2+1e-7).all(1)]
        expected=source.copy();choice=0
        if len(crop)>=8 and source[3]/source[4]>=1.5:
            centered=crop[:,:2].astype(float)-crop[:,:2].astype(float).mean(0)
            _,singular,axes=np.linalg.svd(centered,full_matrices=False)
            if singular[0]**2/(len(crop)-1)>1e-12:
                yaw=np.arctan2(axes[0,1],axes[0,0]);yaw=((yaw+np.pi/2)%np.pi)-np.pi/2
                assert abs(((float(pca[6])-yaw+np.pi/2)%np.pi)-np.pi/2)<2e-6
                candidates=[source.copy(),pca.copy(),pca.copy()]
                candidates[2][6]=((float(pca[6])+np.pi)%np.pi)-np.pi/2
                losses=[distance(crop.astype(float),b.astype(float)) for b in candidates]
                choice=next(i for i,l in enumerate(losses) if l<=min(losses)+1e-12)
                expected=candidates[choice]
                saved=trace[s['id']]['candidate_losses']
                loss_errors.extend(abs(x-y) for x,y in zip(losses,saved))
        assert np.array_equal(surface,expected),(s['id'],choice)
        meta=json.loads(Path(s['meta_path']).read_text());gt=np.array(meta['ground_info'][s['obj_index']]['bbox_3d'][:7],float)
        if s['platform']=='drone':gt[2]+=1.8
        if s['platform']=='waymo':
            view=int(Path(s['lidar_path']).parent.name.split('_')[-1]);angle=[0,-45,45,-90,90][view]
            gt[:3]=Rotation.from_euler('z',angle,degrees=True).as_matrix()@gt[:3];gt[6]+=np.deg2rad(angle)
        values={}
        for name,b in [('source',source),('pca',pca),('surface',surface)]:
            v=oriented_iou(gt[:3],gt[3:6],Rotation.from_euler('z',gt[6]).as_matrix(),b[:3],b[3:6],Rotation.from_euler('z',float(b[6])).as_matrix())
            errors.append(abs(v-given[s['id']][name+'_iou']));values[name+'_iou']=v
        rows.append({'id':s['id'],'platform':s['platform'],'selected_candidate':choice,**values})
    assert max(errors)<1e-6 and max(loss_errors,default=0)<1e-6
    result={'n':len(rows),'seals_verified':True,'center_size_bitexact':True,
            'independent_boundary_distance_selection_matches':True,
            'loss_max_error':max(loss_errors,default=0),'iou_max_error':max(errors),'rows':rows,
            'scope':'Output-space geometry candidate selection; not network parameter adaptation. Development45 and new36 are separate.'}
    with a.output.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))


if __name__=='__main__':main()
