from pathlib import Path
import json,numpy as np,hashlib
R=Path(__file__).resolve().parent;lock=json.loads((R/'ANALYSIS_GROUP_LOCK.json').read_text());score=json.loads((R/'scored.json').read_text());groups=lock['groups'];out=[]
for group,ids in [('all',[r['id'] for r in score['rows']]),*groups.items()]:
 rr=[r for r in score['rows'] if r['id'] in ids];keys=sorted(set((r['platform'],r['scene']) for r in rr));clusters=[]
 for p,s in keys:
  z=[r for r in rr if r['platform']==p and r['scene']==s];clusters.append({'platform':p,'scene':s,'n':len(z),**{arm+'_mean_iou':float(np.mean([r[arm+'_iou'] for r in z])) for arm in ['source','pca','surface']}})
 metrics={}
 for a,b in [('source','pca'),('source','surface'),('pca','surface')]:
  rng=np.random.default_rng(lock['bootstrap_seed']);means=np.array([c[b+'_mean_iou']-c[a+'_mean_iou'] for c in clusters]);draw=means[rng.integers(len(means),size=(10000,len(means)))].mean(1);by=[]
  for p in sorted(set(c['platform'] for c in clusters)):
   values=np.array([c[b+'_mean_iou']-c[a+'_mean_iou'] for c in clusters if c['platform']==p]);by.append(values[rng.integers(len(values),size=(10000,len(values)))].mean(1))
  balanced=np.array(by).mean(0);metrics[b+'_minus_'+a]={'frame_mean_delta_iou':float(np.mean([r[b+'_iou']-r[a+'_iou'] for r in rr])),'scene_macro_delta':float(means.mean()),'scene_cluster_macro_ci95':np.quantile(draw,[.025,.975]).tolist(),'platform_balanced_scene_macro_delta':float(balanced.mean()),'platform_stratified_cluster_ci95':np.quantile(balanced,[.025,.975]).tolist()}
 out.append({'group':group,'n':len(rr),'n_scene_clusters':len(clusters),'metrics':metrics,'scene_clusters':clusters,'limits':'new/seen groups partiallyplatformconfounded; fewrobotclusters; descriptivebootstrap not universalpopulationeffect'})
(R/'grouped_scored.json').write_text(json.dumps({'analysis_lock_sha256':hashlib.sha256((R/'ANALYSIS_GROUP_LOCK.json').read_bytes()).hexdigest(),'groups':out},indent=2));print(json.dumps([{k:v for k,v in x.items() if k!='scene_clusters'} for x in out],indent=2))
