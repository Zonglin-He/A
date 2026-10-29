"""Independent CPU-only scoring after sealed prediction verification."""
import argparse,json,hashlib,itertools,re,sys,os
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import ConvexHull,QhullError
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parent
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path.insert(0,str(ROOT/'upstream'))
from transformers import RobertaTokenizerFast
from tta import text_span
from utils.transform_waymo import transform_to_front_view
from utils.eval_det import iou3d_rotated_vs_aligned

def exact_iou(gt,pred):
 gt=np.asarray(gt,float);pred=np.asarray(pred,float)
 if np.any(gt[3:6]<=0) or np.any(pred[3:6]<=0) or not np.isfinite(pred).all():raise ValueError('Invalid box')
 angles=list(gt[6:])+[0,0];yaw,pitch,roll=angles[:3]
 rz=Rotation.from_euler('z',yaw).as_matrix();ry=Rotation.from_euler('y',pitch).as_matrix();rx=Rotation.from_euler('x',roll).as_matrix();rot=rz@ry@rx
 # Inside-oriented-box constraints A*x <= b. Rows of rot.T project to local axes.
 a=np.concatenate([rot.T,-rot.T,np.eye(3),-np.eye(3)])
 b=np.concatenate([gt[3:6]/2+rot.T@gt[:3],gt[3:6]/2-rot.T@gt[:3],pred[3:6]/2+pred[:3],pred[3:6]/2-pred[:3]])
 vertices=[]
 for ids in itertools.combinations(range(12),3):
  aa=a[list(ids)];bb=b[list(ids)]
  if abs(np.linalg.det(aa))<1e-10:continue
  x=np.linalg.solve(aa,bb)
  if np.all(a@x<=b+1e-8):vertices.append(x)
 if len(vertices)<4:inter=0.
 else:
  points=np.unique(np.round(vertices,9),axis=0)
  try:inter=ConvexHull(points).volume
  except QhullError:
   if np.linalg.matrix_rank(points-points.mean(0),tol=1e-7)<3:inter=0.
   else:raise
 union=np.prod(gt[3:6])+np.prod(pred[3:6])-inter
 return float(inter/union)
def main():
 p=argparse.ArgumentParser();p.add_argument('run');a=p.parse_args();run=Path(a.run).resolve();seal=json.loads((run/'sealed.json').read_text())
 for name,h in seal['files'].items():
  if hashlib.sha256((run/name).read_bytes()).hexdigest()!=h:raise RuntimeError('Seal mismatch: '+name)
 if hashlib.sha256((run/'plan.json').read_bytes()).hexdigest()!=seal['plan_sha256']:raise RuntimeError('Plan mismatch')
 tok=RobertaTokenizerFast.from_pretrained(ROOT/'upstream/data/roberta_base',local_files_only=True)
 manifest={s['id']:s for s in json.loads((ROOT/'manifest.json').read_text())};rows=[]
 # Separate target-class-assisted readout is diagnostic only. Adaptation saw text-only mask.
 sys.path.insert(0,str(ROOT/'upstream'))
 from src.joint_det_dataset import WAYMO_SYNONYMS,M3ED_SYNONYMS
 for name in seal['files']:
  sid,arm=name[:-4].split('_',1);s=manifest[sid];meta=json.loads(Path(s['meta_path']).read_text());obj=meta['ground_info'][s['obj_index']];gt=np.array(obj['bbox_3d'],float)
  if s['platform']=='drone':gt[2]+=1.8
  if s['platform']=='waymo':
   view=['F','FL','FR','SL','SR'][int(Path(s['lidar_path']).parent.name.split('_')[-1])];_,b=transform_to_front_view(np.zeros((1,3)),gt[None],view);gt=b[0]
  data=np.load(run/name);b=data['boxes'];ts=data['token_scores'];mask,_=text_span(tok,s['text']);masks={'text_only':mask.numpy()}
  words=(WAYMO_SYNONYMS if s['platform']=='waymo' else M3ED_SYNONYMS).get(obj['class'].lower(),[obj['class'].lower()])
  match=None
  for word in words:
   match=re.search(r'\b'+re.escape(word)+r'\b',s['text'],re.I)
   if match:break
  if match:
   offsets=np.asarray(tok(s['text'],return_offsets_mapping=True)['offset_mapping']);masks['target_class_assisted']=(offsets[:,1]>match.start())&(offsets[:,0]<match.end())
  for readout,mask in masks.items():
   score=(ts*mask[:ts.shape[1]][None]).sum(1);top=np.argsort(-score,kind='stable');idx=int(top[0]);full=exact_iou(gt,b[idx]);yawonly=exact_iou(gt[:7],b[idx]);native=float(iou3d_rotated_vs_aligned(torch.tensor(gt[:7][None],dtype=torch.float32),torch.tensor(b[idx:idx+1]))[0][0,0]);best=max(exact_iou(gt,x) for x in b[top[:10]])
   rows.append({'id':sid,'platform':s['platform'],'arm':arm,'readout':readout,'iou_full':full,'iou_yaw_only':yawonly,'iou_native':native,'top10_best_iou_full':best,'top_query':idx,'score':float(score[idx])})
 summary=[]
 for readout in sorted(set(r['readout'] for r in rows)):
  for platform in ['all','waymo','drone','quad']:
   for arm in sorted(set(r['arm'] for r in rows)):
    r=[x for x in rows if x['readout']==readout and x['arm']==arm and (platform=='all' or x['platform']==platform)]
    if not r:continue
    summary.append({'readout':readout,'platform':platform,'arm':arm,'n':len(r),'mean_iou_full':float(np.mean([x['iou_full'] for x in r])),'acc25':float(np.mean([x['iou_full']>.25 for x in r])),'acc50':float(np.mean([x['iou_full']>.5 for x in r]))})
 (run/'scored.json').write_text(json.dumps({'measurement':'valid_for_fixed_pilot_only','prediction_seal_sha256':hashlib.sha256((run/'sealed.json').read_bytes()).hexdigest(),'rows':rows,'summary':summary},indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
