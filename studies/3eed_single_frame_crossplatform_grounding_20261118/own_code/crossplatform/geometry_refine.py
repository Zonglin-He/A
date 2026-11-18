"""Fixed CPU point-PCA latent yaw refinement. Inputs do not include GT."""
from pathlib import Path
import json,hashlib,time,sys
import numpy as np
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R/'upstream'))
from utils.transform_waymo import transform_to_front_view

def reconstruct_points(s,seed):
 p=s['lidar_path'];raw=np.load(p) if p.endswith('.npy') else np.fromfile(p,np.float32).reshape(-1,4)
 rng=np.random.default_rng(seed);idx=rng.choice(len(raw),16384,replace=len(raw)<16384);xyz=raw[idx,:3].copy()
 if s['platform']=='drone':xyz[:,2]+=1.8
 if s['platform']=='waymo':
  view=['F','FL','FR','SL','SR'][int(Path(p).parent.name.split('_')[-1])];xyz,_=transform_to_front_view(xyz,np.zeros((1,7)),view)
 return xyz.astype(np.float32)
def main():
 lock=json.loads((R/'PLAN_GEOMETRY_PCA.json').read_text())
 for name,h in lock['source_sha256'].items():assert hashlib.sha256((R/name).read_bytes()).hexdigest()==h
 assert hashlib.sha256((R/'manifest.json').read_bytes()).hexdigest()==lock['manifest_sha256']
 source=R/'runs/pilot_v2_scale';seal=json.loads((source/'sealed.json').read_text());traces=json.loads((source/'trace.json').read_text());trace={x['id']:x for x in traces};manifest=json.loads((R/'manifest.json').read_text())
 out=R/'runs/geometry_pca_v1';out.mkdir(exist_ok=False)
 (out/'registered_plan.json').write_bytes((R/'PLAN_GEOMETRY_PCA.json').read_bytes());records=[]
 for i,s in enumerate(manifest):
  start=time.perf_counter();name=s['id']+'_baseline.npz';assert hashlib.sha256((source/name).read_bytes()).hexdigest()==seal['files'][name]
  top=trace[s['id']]['diagnostics']['entropy']['initial_top'];src=np.load(source/name)['boxes'][top].copy();source7=np.r_[src,np.float32(0)].astype(np.float32);refined=source7.copy();points=reconstruct_points(s,20261029+i);points=np.unique(points,axis=0);selected=np.all(abs(points-src[:3])<=src[3:6]/2+1e-7,axis=1);crop=points[selected];reason='updated';eigenvalues=None
  if src[0:6].shape!=(6,) or not np.isfinite(src).all() or np.any(src[3:6]<=0):raise RuntimeError('Invalid sourcebox')
  if len(crop)<8:reason='fewer_than8_crop_points'
  elif src[3]/src[4]<1.5:reason='predicted_L_over_W_below1.5'
  else:
   xy=crop[:,:2].astype(float);cov=np.cov(xy,rowvar=False);values,vectors=np.linalg.eigh(cov);eigenvalues=values.tolist()
   if values[-1]<=1e-12:reason='degenerate_XY_covariance'
   else:
    direction=vectors[:,-1];yaw=np.arctan2(direction[1],direction[0]);refined[6]=((yaw+np.pi/2)%np.pi)-np.pi/2
  assert np.array_equal(source7[:6],refined[:6]),'Center/size changed'
  dest=out/(s['id']+'.npz');np.savez_compressed(dest,source7d=source7,refined7d=refined)
  rec={'id':s['id'],'platform':s['platform'],'source_query_index':int(top),'unique_visible_points':len(points),'predicted_crop_points':len(crop),'predicted_L_over_W':float(src[3]/src[4]),'eigenvalues_XY':eigenvalues,'yaw_radians':float(refined[6]),'reason':reason,'center_and_size_bitexact':True,'seconds':time.perf_counter()-start};records.append(rec);print(json.dumps(rec),flush=True)
 (out/'trace.json').write_text(json.dumps(records,indent=2));hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*.npz')};(out/'sealed.json').write_text(json.dumps({'files':hashes,'n':len(records),'plan_sha256':hashlib.sha256((R/'PLAN_GEOMETRY_PCA.json').read_bytes()).hexdigest(),'label_access':False},indent=2));print('SEALED',len(records),flush=True)
if __name__=='__main__':main()
