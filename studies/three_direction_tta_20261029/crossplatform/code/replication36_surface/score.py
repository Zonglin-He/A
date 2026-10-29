from pathlib import Path
import sys,json,hashlib,numpy as np,importlib.util
R=Path(__file__).resolve().parent;P=R.parent;sys.path[:0]=[str(P),str(P/'upstream')];spec=importlib.util.spec_from_file_location('exact_3eed_score',P/'score.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);exact_iou=module.exact_iou
from utils.transform_waymo import transform_to_front_view

def rz(x):
 c,s=np.cos(x),np.sin(x);return np.array([[c,-s,0],[s,c,0],[0,0,1]])
def iou(g,p):
 q=g.copy();q[:3]=(g[:3]-p[:3])@rz(p[6]);q[6]=g[6]-p[6];return exact_iou(q,np.r_[np.zeros(3),p[3:6]])
seals={k:json.loads((R/k/'sealed.json').read_text()) for k in ['source','pca','surface']};assert all(z['n']==36 for z in seals.values());manifest=json.loads((R/'manifest.json').read_text());rows=[]
for s in manifest:
 name=s['id']+'.npz'
 for arm,seal in seals.items():assert hashlib.sha256((R/arm/name).read_bytes()).hexdigest()==seal['files'][name]
 meta=json.loads(Path(s['meta_path']).read_text());gt=np.asarray(meta['ground_info'][s['obj_index']]['bbox_3d'][:7],float)
 if s['platform']=='drone':gt[2]+=1.8
 if s['platform']=='waymo':
  view=['F','FL','FR','SL','SR'][int(Path(s['lidar_path']).parent.name.split('_')[-1])];_,g=transform_to_front_view(np.zeros((1,3)),gt[None],view);gt=g[0]
 d=np.load(R/'surface'/name);r={'id':s['id'],'platform':s['platform'],'scene':s['scene']}
 for arm in ['source','pca','surface']:r[arm+'_iou']=iou(gt,d[arm+'7d'].astype(float))
 rows.append(r)
def aggregate(rr):
 ret={'n':len(rr)}
 for arm in ['source','pca','surface']:ret[arm]={'mean_iou':float(np.mean([r[arm+'_iou'] for r in rr])),'acc25_count':sum(r[arm+'_iou']>.25 for r in rr),'acc50_count':sum(r[arm+'_iou']>.5 for r in rr)}
 for a,b in [('source','pca'),('source','surface'),('pca','surface')]:
  d=np.array([r[b+'_iou']-r[a+'_iou'] for r in rr]);ret[b+'_minus_'+a]={'mean_delta_iou':float(d.mean()),'min_delta_iou':float(d.min()),'max_delta_iou':float(d.max()),'rescue25':sum(r[a+'_iou']<=.25<r[b+'_iou'] for r in rr),'harm25':sum(r[b+'_iou']<=.25<r[a+'_iou'] for r in rr),'rescue50':sum(r[a+'_iou']<=.5<r[b+'_iou'] for r in rr),'harm50':sum(r[b+'_iou']<=.5<r[a+'_iou'] for r in rr)}
 return ret
summary=[{'platform':k,**aggregate([r for r in rows if k=='all' or r['platform']==k])} for k in ['all','waymo','drone','quad']];scenes=[{'platform':p,'scene':s,**aggregate([r for r in rows if r['platform']==p and r['scene']==s])} for p,s in sorted(set((r['platform'],r['scene']) for r in rows))];out={'scope':'Sole extra36unexposedframes after45developmentselection, no furthertuning; explicit6D→7D pose-refinement, not networkTTA','metric':'official first7DGT andfullOBBintersection','rows':rows,'summary':summary,'scene_clusters':scenes,'surface_seal_sha256':hashlib.sha256((R/'surface/sealed.json').read_bytes()).hexdigest()};(R/'scored.json').write_text(json.dumps(out,indent=2));print(json.dumps(summary,indent=2))
