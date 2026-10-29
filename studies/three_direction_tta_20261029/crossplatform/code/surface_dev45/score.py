from pathlib import Path
import sys,json,hashlib,numpy as np,importlib.util
R=Path(__file__).resolve().parent;P=R.parent;sys.path[:0]=[str(P),str(P/'upstream')];spec=importlib.util.spec_from_file_location('exact_3eed_score',P/'score.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);exact_iou=module.exact_iou
from utils.transform_waymo import transform_to_front_view
from refine import rotation
out=R/'predictions';seal=json.loads((out/'sealed.json').read_text());assert seal['n']==45;rows=[]
for cohort,mp in [('development9',P/'manifest.json'),('replication36',P/'replication36/manifest.json')]:
 for s in json.loads(mp.read_text()):
  name=s['id']+'.npz';assert hashlib.sha256((out/name).read_bytes()).hexdigest()==seal['files'][name];m=json.loads(Path(s['meta_path']).read_text());gt=np.array(m['ground_info'][s['obj_index']]['bbox_3d'][:7],float)
  if s['platform']=='drone':gt[2]+=1.8
  if s['platform']=='waymo':
   view=['F','FL','FR','SL','SR'][int(Path(s['lidar_path']).parent.name.split('_')[-1])];_,g=transform_to_front_view(np.zeros((1,3)),gt[None],view);gt=g[0]
  d=np.load(out/name);record={'id':s['id'],'cohort':cohort,'platform':s['platform']}
  for arm in ['source','pca','surface']:
   b=d[arm+'7d'].astype(float);g=gt.copy();g[:3]=(gt[:3]-b[:3])@rotation(b[6]);g[6]=gt[6]-b[6];record[arm+'_iou']=exact_iou(g,np.r_[np.zeros(3),b[3:6]])
  rows.append(record)
summary=[]
for cohort in ['all45','development9','replication36']:
 rr=[r for r in rows if cohort=='all45' or r['cohort']==cohort];summary.append({'cohort':cohort,'n':len(rr),**{arm:{'mean_iou':float(np.mean([r[arm+'_iou'] for r in rr])),'acc25_count':sum(r[arm+'_iou']>.25 for r in rr),'acc50_count':sum(r[arm+'_iou']>.5 for r in rr)} for arm in ['source','pca','surface']}})
severe=next(r for r in rows if r['id']=='01384dc3febeb0f9');all45=summary[0];passed=severe['surface_iou']>.5 and all45['surface']['mean_iou']>all45['pca']['mean_iou'] and all45['surface']['acc50_count']>all45['pca']['acc50_count'];tails=sorted(rows,key=lambda r:r['surface_iou']-r['pca_iou']);result={'scope':'EXPOSED45development geometry-selection, not heldout validation ornetworkTTA','rows':rows,'summary':summary,'severe_negative_case':severe,'neighbor_case':next(r for r in rows if r['id']=='0077e5b4aba8ab60'),'surface_vs_pca_tails':tails,'continuation_rule':'severeSurfaceIoU>.5 AND overall45meanSurface>meanPCA AND overall45Acc50countSurface>countPCA','continue_fixed_new36':passed,'seal_sha256':hashlib.sha256((out/'sealed.json').read_bytes()).hexdigest()};(R/'scored.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ['rows','surface_vs_pca_tails']},indent=2))
