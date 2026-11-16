"""Task labels are opened only after all prediction seals and CPU replay audit."""
from pathlib import Path
import json,hashlib,sys,numpy as np
R=Path(__file__).resolve().parents[2];H=R/'research/round105';B=R/'crossplatform/round105';O=H/'scores_attempt001';rd=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();sys.path.insert(0,str(R/'research/round6/source_validation_scoring'));from batch_yaw_iou import many_iou
sys.path.insert(0,str(R/'tools'));from obb_iou import oriented_iou
def dump(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def met(rows,arm):
 a=np.array([r['arms'][arm]['IoU'] for r in rows]);return {'n':len(a),'mIoU_percent':float(a.mean()*100),'Acc25_percent':float((a>.25).mean()*100),'Acc50_percent':float((a>.5).mean()*100)}
def main():
 assert rd(H/'NATIVE_AUDIT.json')['state'].startswith('PASS_');matrix=rd(B/'MATRIX_SEALED.json');assert matrix['n']==768;O.mkdir(exist_ok=False);summary=[];paired=[];allrows=[];scenes=[];error=0.
 controls=[('TTA_jointABC',k) for k in ['prepared_frozen','frozen_jointABC','frozen_ensemble3','teacher_shuffle_jointABC','wrong_TTA_jointABC','joint3_FC']]+[('frozen_jointABC','prepared_frozen'),('frozen_jointABC','wrong_frozen_jointABC'),('TTA_AC','frozen_AC'),('TTA_ensemble3','frozen_ensemble3')]
 for name,h in matrix['directions'].items():
  src,_,tgt=name.split('_');root=B/name;assert sha(root/'SEALED.json')==h;rows=rd(R/f'research/round99/input/TARGET_{tgt.upper()}128_PRIVATE.json');gtrows=rd(R/f'research/round96/experiment/OFFICIAL_{tgt.upper()}_VAL_ORDER_PRIVATE.json');rr=[]
  for row in rows:
   z=np.load(root/'samples_PRIVATE'/(row['id']+'.npz'));boxes=z['source_boxes'].astype(np.float64);boxes[:,3:]=np.maximum(boxes[:,3:],1e-6);boxes=np.c_[boxes,np.zeros(256)];gt=np.array(gtrows[row['anno_index']]['source_GT7_float32']);ys=many_iou(boxes,gt);arms={}
   for arm,p in zip(z['arm_names'],z['scores']):
    q=int(p.argmax());arms[str(arm)]={'IoU':float(ys[q]),'query_PRIVATE':q,'oracle_IoU_diagnostic':float(ys.max())}
    if arm=='TTA_jointABC':
     c,s=np.cos(gt[6]),np.sin(gt[6]);rot=np.array([[c,-s,0],[s,c,0],[0,0,1]]);v=oriented_iou(boxes[q,:3],boxes[q,3:6],np.eye(3),gt[:3],gt[3:6],rot);error=max(error,abs(v-ys[q]));assert abs(v-ys[q])<1e-6
   rec=rd(root/'records_PRIVATE'/(row['id']+'.json'));fits={k:{a:b for a,b in x.items() if a not in ['trials','target']} for k,x in rec['fits'].items()};rr.append({'source':src,'target':tgt,'id_PRIVATE':row['id'],'scene_PRIVATE':row['scene'],'arms':arms,'status':rec['status'],'fits':fits})
  dump(O/(name+'_ALL_ROWS_PRIVATE.json'),rr);allrows+=rr
  for arm in matrix['arms']:
   summary.append({'source':src,'target':tgt,'arm':arm,**met(rr,arm)})
   for sc in sorted({r['scene_PRIVATE'] for r in rr}):scenes.append({'source':src,'target':tgt,'scene_PRIVATE':sc,'arm':arm,**met([r for r in rr if r['scene_PRIVATE']==sc],arm)})
  for arm,control in controls:
   d=np.array([r['arms'][arm]['IoU']-r['arms'][control]['IoU'] for r in rr]);paired.append({'source':src,'target':tgt,'method':arm,'control':control,'n':len(rr),'gain_pp':float(d.mean()*100),'better':int((d>1e-12).sum()),'worse':int((d< -1e-12).sum()),'same':int((abs(d)<=1e-12).sum()),'largest_single_positive_pp':float(d.max()*100/len(rr)),'largest_single_negative_pp':float(d.min()*100/len(rr))})
 agg=[]
 for group in ['all_six_equal','waymo','drone','quad']:
  for arm in matrix['arms']:
   ss=[x for x in summary if x['arm']==arm and (group=='all_six_equal' or x['target']==group)];agg.append({'group':group,'arm':arm,**{k:float(np.mean([x[k] for x in ss])) for k in ['mIoU_percent','Acc25_percent','Acc50_percent']}})
 for x in summary:
  vals=[r['arms'][x['arm']]['IoU'] for r in allrows if r['source']==x['source'] and r['target']==x['target']];assert abs(x['mIoU_percent']-sum(vals)*100/len(vals))<1e-10
 dump(O/'ALL_METHOD_ROWS.json',summary);dump(O/'ALL_PAIRED_ROWS.json',paired);dump(O/'AGGREGATE_ROWS.json',agg);dump(O/'ALL_SCENE_ROWS_PRIVATE.json',scenes);dump(H/'SCALAR_AUDIT.json',{'state':'PASS_R105_ALL_FIXED_INPUTS_READOUTS_GEOMETRY_AND_INDEPENDENT_REDUCTION','n':len(allrows),'arms':len(matrix['arms']),'readouts':len(allrows)*len(matrix['arms']),'max_geometry_error':error,'FINAL_access':False});dump(O/'SEALED.json',{'files':{p.name:sha(p) for p in O.glob('*.json')}});print('SCORE105_PASS',len(allrows),len(matrix['arms']))
if __name__=='__main__':main()
