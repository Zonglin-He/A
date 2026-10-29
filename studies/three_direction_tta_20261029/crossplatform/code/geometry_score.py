"""Post-seal7D OBB scoring for geometric refinement; GT only here."""
from pathlib import Path
import sys,json,hashlib,numpy as np
from score import exact_iou
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R/'upstream'))
from utils.transform_waymo import transform_to_front_view

def rz(x):
 c,s=np.cos(x),np.sin(x);return np.array([[c,-s,0],[s,c,0],[0,0,1]])
def iou_obb(g,p):
 q=g.copy();q[:3]=(g[:3]-p[:3])@rz(p[6]);q[6]=g[6]-p[6];return exact_iou(q,np.r_[np.zeros(3),p[3:6]])
run=R/'runs/geometry_pca_v1';seal=json.loads((run/'sealed.json').read_text());manifest={s['id']:s for s in json.loads((R/'manifest.json').read_text())};rows=[]
for name,h in seal['files'].items():
 assert hashlib.sha256((run/name).read_bytes()).hexdigest()==h;s=manifest[name[:-4]];meta=json.loads(Path(s['meta_path']).read_text());gt=np.asarray(meta['ground_info'][s['obj_index']]['bbox_3d'][:7],float)
 if s['platform']=='drone':gt[2]+=1.8
 if s['platform']=='waymo':
  view=['F','FL','FR','SL','SR'][int(Path(s['lidar_path']).parent.name.split('_')[-1])];_,g=transform_to_front_view(np.zeros((1,3)),gt[None],view);gt=g[0]
 p=np.load(run/name);base=iou_obb(gt,p['source7d'].astype(float));adapt=iou_obb(gt,p['refined7d'].astype(float));rows.append({'id':s['id'],'platform':s['platform'],'source_iou':base,'refined_iou':adapt,'delta_iou':adapt-base,'source_acc25':base>.25,'refined_acc25':adapt>.25,'source_acc50':base>.5,'refined_acc50':adapt>.5})
summary=[]
for platform in ['all','waymo','drone','quad']:
 rr=[r for r in rows if platform=='all' or r['platform']==platform]
 summary.append({'platform':platform,'n':len(rr),'source_mean_iou':float(np.mean([r['source_iou'] for r in rr])),'refined_mean_iou':float(np.mean([r['refined_iou'] for r in rr])),'source_acc25':float(np.mean([r['source_acc25'] for r in rr])),'refined_acc25':float(np.mean([r['refined_acc25'] for r in rr])),'source_acc50':float(np.mean([r['source_acc50'] for r in rr])),'refined_acc50':float(np.mean([r['refined_acc50'] for r in rr])),'rescued25':sum(not r['source_acc25'] and r['refined_acc25'] for r in rr),'harmed25':sum(r['source_acc25'] and not r['refined_acc25'] for r in rr),'rescued50':sum(not r['source_acc50'] and r['refined_acc50'] for r in rr),'harmed50':sum(r['source_acc50'] and not r['refined_acc50'] for r in rr)})
result={'measurement':'valid_official7d_GT_with_explicit_oriented7d_output_extension','claim':'exploratory_test_time_geometric_refinement_not_network_TTA','n':len(rows),'seal_sha256':hashlib.sha256((run/'sealed.json').read_bytes()).hexdigest(),'rows':rows,'summary':summary};(run/'scored.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
