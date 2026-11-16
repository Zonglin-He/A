"""Postseal new17 arms only scored; already audited R110 scalar results reused exactly."""
from pathlib import Path
import sys,json,hashlib,numpy as np
R=Path(__file__).resolve().parents[2];H=R/'research/round111';B=R/'crossplatform/round111';O=H/'scores_attempt001';rd=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();sys.path.insert(0,str(R/'research/round6/source_validation_scoring'));from batch_yaw_iou import many_iou
sys.path.insert(0,str(R/'tools'));from obb_iou import oriented_iou
PRIMARY='native_real_fusedABC';CONTROLS=['original_source','original_source_matchedcoverage','prepared_frozen','native_real_ensemble','native_randomA_fusedABC','native_wrong_fusedABC','real_frozen_fusedABC','real_frozen_ensemble','real_TTA_fusedABC','direct_three_geometric'];rd=lambda p:json.loads(Path(p).read_text())
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def met(rr,k):
 a=np.array([r['arms'][k]['IoU'] for r in rr]);return {'n':len(a),'mIoU_percent':float(a.mean()*100),'Acc25_percent':float((a>.25).mean()*100),'Acc50_percent':float((a>.5).mean()*100)}
def main():
 assert rd(H/'NATIVE_AUDIT.json')['n']==768;matrix=rd(B/'MATRIX_SEALED.json');O.mkdir(exist_ok=False);methods=[];allrows=[];scenes=[];paired=[];err=0.;controls=[(PRIMARY,k) for k in CONTROLS]+[('preparedA_native_head_base','preparedA_prepared_head_base'),('preparedF_native_head_base','preparedF_prepared_head_base'),('sourceA_prepared_head','original_source_matchedcoverage')]
 for name,h in matrix['directions'].items():
  src,_,tgt=name.split('_');D=B/name;assert sha(D/'SEALED.json')==h;rows=rd(R/f'research/round99/input/TARGET_{tgt.upper()}128_PRIVATE.json');golds=rd(R/f'research/round96/experiment/OFFICIAL_{tgt.upper()}_VAL_ORDER_PRIVATE.json');prior={r['id_PRIVATE']:r for r in rd(R/'research/round110/scores_attempt001'/(name+'_ALL_ROWS_PRIVATE.json'))};rr=[]
  for row in rows:
   z=np.load(D/'samples_PRIVATE'/(row['id']+'.npz'));boxes=z['source_boxes'].astype(float);boxes[:,3:]=np.maximum(boxes[:,3:],1e-6);boxes=np.c_[boxes,np.zeros(256)];gt=np.array(golds[row['anno_index']]['source_GT7_float32']);ys=many_iou(boxes,gt);arms=prior[row['id']]['arms'].copy()
   for arm,p in zip(z['arm_names'],z['scores']):
    if arm in arms:assert arms[arm]['query_PRIVATE']==int(p.argmax());continue
    q=int(p.argmax());arms[str(arm)]={'IoU':float(ys[q]),'query_PRIVATE':q,'oracle_IoU_diagnostic':float(ys.max())}
    if arm==PRIMARY:
     c,s=np.cos(gt[6]),np.sin(gt[6]);rot=np.array([[c,-s,0],[s,c,0],[0,0,1]]);y=oriented_iou(boxes[q,:3],boxes[q,3:6],np.eye(3),gt[:3],gt[3:6],rot);err=max(err,abs(y-ys[q]));assert abs(y-ys[q])<1e-6
   rec=rd(D/'records_PRIVATE'/(row['id']+'.json'));rr.append({'source':src,'target':tgt,'id_PRIVATE':row['id'],'scene_PRIVATE':row['scene'],'status':rec,'fits':{},'arms':arms})
  dump(O/(name+'_ALL_ROWS_PRIVATE.json'),rr);allrows+=rr
  for k in matrix['arms']:
   methods.append({'source':src,'target':tgt,'arm':k,**met(rr,k)})
   for sc in sorted({x['scene_PRIVATE'] for x in rr}):scenes.append({'source':src,'target':tgt,'scene_PRIVATE':sc,'arm':k,**met([r for r in rr if r['scene_PRIVATE']==sc],k)})
  for k,c in controls:
   a=np.array([r['arms'][k]['IoU']-r['arms'][c]['IoU'] for r in rr]);paired.append({'source':src,'target':tgt,'method':k,'control':c,'n':len(a),'gain_pp':float(a.mean()*100),'better':int((a>1e-12).sum()),'worse':int((a< -1e-12).sum()),'same':int((abs(a)<=1e-12).sum())})
 agg=[]
 for group in ['all_six_equal','waymo','drone','quad']:
  for arm in matrix['arms']:
   ss=[x for x in methods if x['arm']==arm and (group=='all_six_equal' or x['target']==group)];agg.append({'group':group,'arm':arm,**{k:float(np.mean([x[k] for x in ss])) for k in ['mIoU_percent','Acc25_percent','Acc50_percent']}})
 for x in methods:
  vv=met([r for r in allrows if r['source']==x['source'] and r['target']==x['target']],x['arm']);assert all(abs(x[k]-v)<1e-10 for k,v in vv.items())
 for name,x in [('ALL_METHOD_ROWS',methods),('ALL_SCENE_ROWS_PRIVATE',scenes),('ALL_PAIRED_ROWS',paired),('AGGREGATE_ROWS',agg)]:dump(O/(name+'.json'),x)
 dump(H/'SCALAR_AUDIT.json',{'state':'PASS_R111_NEW17ARMS_IOU_PLUS_REUSED_AUDITED58ARMS_ALL_AGGREGATION','n':len(allrows),'readouts':len(allrows)*len(matrix['arms']),'new_readouts':len(allrows)*17,'reused_readouts':len(allrows)*58,'arms':len(matrix['arms']),'max_geometry_error':err,'FINAL_access':False});dump(O/'SEALED.json',{'files':{f.name:sha(f) for f in O.glob('*.json')}});print('SCORE111_PASS',len(allrows),len(matrix['arms']))
if __name__=='__main__':main()
