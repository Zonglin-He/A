from pathlib import Path
import sys,json,hashlib,numpy as np
R=Path(__file__).resolve().parent;P=R.parent;sys.path[:0]=[str(P),str(P/'upstream')]
# Load exact convex intersection from parent score under a nonconflicting name.
import importlib.util
spec=importlib.util.spec_from_file_location('exact_3eed_score',P/'score.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);exact_iou=module.exact_iou
from utils.transform_waymo import transform_to_front_view

def rz(x):
 c,s=np.cos(x),np.sin(x);return np.array([[c,-s,0],[s,c,0],[0,0,1]])
def iou(g,p):
 q=g.copy();q[:3]=(g[:3]-p[:3])@rz(p[6]);q[6]=g[6]-p[6];return exact_iou(q,np.r_[np.zeros(3),p[3:6]])
source=R/'source';out=R/'pca';ss=json.loads((source/'sealed.json').read_text());ps=json.loads((out/'sealed.json').read_text());assert ss['n']==ps['n']==36;manifest=json.loads((R/'manifest.json').read_text());rows=[]
for s in manifest:
 name=s['id']+'.npz';assert hashlib.sha256((source/name).read_bytes()).hexdigest()==ss['files'][name];assert hashlib.sha256((out/name).read_bytes()).hexdigest()==ps['files'][name]
 meta=json.loads(Path(s['meta_path']).read_text());gt=np.asarray(meta['ground_info'][s['obj_index']]['bbox_3d'][:7],float)
 if s['platform']=='drone':gt[2]+=1.8
 if s['platform']=='waymo':
  view=['F','FL','FR','SL','SR'][int(Path(s['lidar_path']).parent.name.split('_')[-1])];_,g=transform_to_front_view(np.zeros((1,3)),gt[None],view);gt=g[0]
 d=np.load(out/name);a=iou(gt,d['source7d'].astype(float));b=iou(gt,d['refined7d'].astype(float));rows.append({'id':s['id'],'platform':s['platform'],'scene':s['scene'],'source_iou':a,'refined_iou':b,'delta_iou':b-a,'source_acc25':a>.25,'refined_acc25':b>.25,'source_acc50':a>.5,'refined_acc50':b>.5})
def aggregate(rr):
 return {'n':len(rr),'source_mean_iou':float(np.mean([r['source_iou'] for r in rr])),'refined_mean_iou':float(np.mean([r['refined_iou'] for r in rr])),'mean_delta_iou':float(np.mean([r['delta_iou'] for r in rr])),'source_acc25':float(np.mean([r['source_acc25'] for r in rr])),'refined_acc25':float(np.mean([r['refined_acc25'] for r in rr])),'source_acc50':float(np.mean([r['source_acc50'] for r in rr])),'refined_acc50':float(np.mean([r['refined_acc50'] for r in rr])),'min_delta_iou':min(r['delta_iou'] for r in rr),'max_delta_iou':max(r['delta_iou'] for r in rr),'rescued25':sum(not r['source_acc25'] and r['refined_acc25'] for r in rr),'harmed25':sum(r['source_acc25'] and not r['refined_acc25'] for r in rr),'rescued50':sum(not r['source_acc50'] and r['refined_acc50'] for r in rr),'harmed50':sum(r['source_acc50'] and not r['refined_acc50'] for r in rr)}
summary=[{'platform':k,**aggregate([r for r in rows if k=='all' or r['platform']==k])} for k in ['all','waymo','drone','quad']];scenes=[{'platform':p,'scene':s,**aggregate([r for r in rows if r['platform']==p and r['scene']==s])} for p,s in sorted(set((r['platform'],r['scene']) for r in rows))];scene_macro={'n_scene_clusters':len(scenes),'source_mean_iou':float(np.mean([s['source_mean_iou'] for s in scenes])),'refined_mean_iou':float(np.mean([s['refined_mean_iou'] for s in scenes])),'mean_delta_iou':float(np.mean([s['mean_delta_iou'] for s in scenes])),'note':'scene clusters overlap original development cohort; frame and scene macro differ; no independent unseen-scene claim'};result={'measurement':'official_first7D_GT_vs_explicit7D_oriented_prediction','claim':'fixed_rule_unexposed_new_frame_replication_with_possible_scene_overlap_not_networkTTA','n':36,'source_seal_sha256':hashlib.sha256((source/'sealed.json').read_bytes()).hexdigest(),'pca_seal_sha256':hashlib.sha256((out/'sealed.json').read_bytes()).hexdigest(),'rows':rows,'summary':summary,'scene_clusters':scenes,'scene_macro':scene_macro};(R/'scored.json').write_text(json.dumps(result,indent=2));print(json.dumps({'summary':summary,'scene_macro':scene_macro},indent=2))
