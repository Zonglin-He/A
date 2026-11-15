"""Postseal target scoring and independent reduction; no selection of individual outputs."""
from pathlib import Path
import json,hashlib,sys,numpy as np
R=Path(__file__).resolve().parents[2];H=R/'research/round99';B=R/'crossplatform/round99';O=H/'scores_attempt001';rd=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();sys.path.insert(0,str(R/'research/round6/source_validation_scoring'));from batch_yaw_iou import many_iou
sys.path.insert(0,str(R/'tools'));from obb_iou import oriented_iou
def dump(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def met(rows,arm):
 a=np.array([r['arms'][arm]['IoU'] for r in rows]);return {'n':len(a),'mIoU_percent':float(a.mean()*100),'Acc25_percent':float((a>.25).mean()*100),'Acc50_percent':float((a>.5).mean()*100)}
def main():
 assert rd(H/'NATIVE_AUDIT.json')['state'].startswith('PASS_');matrix=rd(B/'MATRIX_SEALED.json');assert matrix['n']==768;O.mkdir(exist_ok=False);summary=[];paired=[];allrows=[];scenes=[];error=0.
 for name,h in matrix['directions'].items():
  src,_,tgt=name.split('_');root=B/name;assert sha(root/'SEALED.json')==h;rows=rd(H/f'input/TARGET_{tgt.upper()}128_PRIVATE.json');gtrows=rd(R/f'research/round96/experiment/OFFICIAL_{tgt.upper()}_VAL_ORDER_PRIVATE.json');oldrows={r['id_PRIVATE']:r for r in rd(R/'research/round98/scores_attempt001'/(name+'_ALL_ROWS_PRIVATE.json'))};rr=[]
  for row in rows:
   gold=gtrows[row['anno_index']];z=np.load(root/'samples_PRIVATE'/(row['id']+'.npz'));boxes=z['source_boxes'].astype(np.float64);boxes[:,3:]=np.maximum(boxes[:,3:],1e-6);boxes=np.c_[boxes,np.zeros(256)];gt=np.array(gold['source_GT7_float32']);ys=many_iou(boxes,gt);arms={}
   for arm,p in zip(z['arm_names'],z['scores']):
    q=int(p.argmax());arms[str(arm)]={'IoU':float(ys[q]),'query_PRIVATE':q,'oracle_IoU_diagnostic':float(ys.max())}
    if arm=='joint3_FC':
     c,s=np.cos(gt[6]),np.sin(gt[6]);rot=np.array([[c,-s,0],[s,c,0],[0,0,1]]);v=oriented_iou(boxes[q,:3],boxes[q,3:6],np.eye(3),gt[:3],gt[3:6],rot);error=max(error,abs(v-ys[q]));assert error<1e-6
   for arm in oldrows[row['id']]['arms']:
    if arm in arms:assert abs(arms[arm]['IoU']-oldrows[row['id']]['arms'][arm]['IoU'])<1e-12
   rec=rd(root/'records_PRIVATE'/(row['id']+'.json'));rr.append({'source':src,'target':tgt,'id_PRIVATE':row['id'],'scene_PRIVATE':row['scene'],'arms':arms,'fits':{n:{k:v for k,v in x.items() if k!='trials'} for n,x in rec.items()}})
  for arm in matrix['arms']:summary.append({'source':src,'target':tgt,'arm':arm,**met(rr,arm)})
  for sc in sorted({r['scene_PRIVATE'] for r in rr}):
   for arm in matrix['arms']:scenes.append({'source':src,'target':tgt,'scene_PRIVATE':sc,'arm':arm,**met([r for r in rr if r['scene_PRIVATE']==sc],arm)})
  for method,control in [('joint3_FC','prepared_frozen'),('joint3_FC','lm_FC'),('joint3_FC','direct_FC'),('joint3_FC','direct_four'),('joint3_FC','joint3_shuffle_FC'),('joint3_C','lm_C'),('joint3_C','frozen_C'),('joint3_F','lm_F'),('joint3_F','binding_direct')]:
   a=np.array([r['arms'][method]['IoU'] for r in rr]);b=np.array([r['arms'][control]['IoU'] for r in rr]);d=a-b;paired.append({'source':src,'target':tgt,'method':method,'control':control,'gain_pp':float(d.mean()*100),'better':int((d>1e-12).sum()),'worse':int((d< -1e-12).sum()),'same':int((abs(d)<=1e-12).sum())})
  dump(O/(name+'_ALL_ROWS_PRIVATE.json'),rr);allrows+=rr
 agg=[]
 for group in ['all_six_equal','waymo','drone','quad']:
  for arm in matrix['arms']:
   ss=[x for x in summary if x['arm']==arm and (group=='all_six_equal' or x['target']==group)];agg.append({'group':group,'arm':arm,**{k:float(np.mean([x[k] for x in ss])) for k in ['mIoU_percent','Acc25_percent','Acc50_percent']}})
 # A separate reduction from per-input values checks every published metric.
 for x in summary:
  rr=[r for r in allrows if r['source']==x['source'] and r['target']==x['target']];m=met(rr,x['arm']);assert all(abs(x[k]-v)<1e-10 for k,v in m.items())
 dump(O/'ALL_METHOD_ROWS.json',summary);dump(O/'ALL_PAIRED_ROWS.json',paired);dump(O/'AGGREGATE_ROWS.json',agg);dump(O/'ALL_SCENE_ROWS_PRIVATE.json',scenes);dump(H/'SCALAR_AUDIT.json',{'state':'PASS_R99_ALL_768_INPUTS_15360_READOUTS_REDUCED_AND_NATIVE_GEOMETRY_CHECKED','n':768,'readouts':len(allrows)*20,'max_geometry_error':error,'FINAL_access':False});dump(O/'SEALED.json',{'files':{p.name:sha(p) for p in O.glob('*.json')}});print('SCORE99_PASS',len(allrows),len(allrows)*20)
if __name__=='__main__':main()
