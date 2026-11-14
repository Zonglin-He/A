"""Post-seal complete-method task metrics with exact same R97 cohorts and frozen boxes."""
from pathlib import Path
import sys,json,hashlib,numpy as np,time
R=Path(__file__).resolve().parents[2];H=R/'research/round98';B=R/'crossplatform/round98';S=H/'scores_attempt001';sys.path.insert(0,str(R/'research/round6/source_validation_scoring'));from batch_yaw_iou import many_iou
sys.path.insert(0,str(R/'tools'));from obb_iou import oriented_iou
rd=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def met(rr,arm):
 x=np.array([r['arms'][arm]['IoU'] for r in rr]);return {'n':len(x),'mIoU_percent':float(x.mean()*100),'Acc25_percent':float((x>.25).mean()*100),'Acc50_percent':float((x>.5).mean()*100)}
def main():
 assert rd(H/'NATIVE_AUDIT.json')['state'].startswith('PASS_');matrix=rd(B/'MATRIX_SEALED.json');assert matrix['n']==18620;S.mkdir(exist_ok=False);allrows=[];summary=[];scenes=[];paired=[];maxerr=0.;checks=0;start=time.perf_counter()
 controls=['prepared_frozen','frozen_FC','frozen_four','direct_FC','direct_four','mixture_FC','lm_mixture_FC','lm_shuffle_FC','R97_physical_bt','lm_F','lm_C','binding_direct','R97_direct_mean3','R97_direct_feature3']
 for direction,hashv in matrix['directions'].items():
  src,_,tgt=direction.split('_');root=B/direction;assert sha(root/'SEALED.json')==hashv;se=rd(root/'SEALED.json');idx=rd(R/f'research/round96/experiment/OFFICIAL_{tgt.upper()}_VAL_ORDER_PRIVATE.json');rows=rd(R/f'research/round97/input/TARGET_{tgt.upper()}_PRIVATE.json');rr=[]
  for n,(r,gold) in enumerate(zip(rows,idx)):
   assert n==r['anno_index']==gold['anno_index'];f=root/'samples_PRIVATE'/(r['id']+'.npz');assert sha(f)==se['files'][str(f.relative_to(root))];z=np.load(f,allow_pickle=False);boxes=z['source_boxes'].astype(np.float64);boxes[:,3:]=np.maximum(boxes[:,3:],1e-6);pred7=np.c_[boxes,np.zeros(256)];gt=np.array(gold['source_GT7_float32'],np.float64);ys=many_iou(pred7,gt);c,s=np.cos(gt[6]),np.sin(gt[6]);rot=np.array([[c,-s,0],[s,c,0],[0,0,1.]]);arms={}
   for arm,pp in zip(z['arm_names'],z['scores']):
    order=np.argsort(-pp,kind='stable');q=int(order[0]);iou=float(ys[q]);arms[str(arm)]={'IoU':iou,'top5_IoU':float(ys[order[:5]].max()),'top10_IoU':float(ys[order[:10]].max()),'oracle_IoU_diagnostic':float(ys.max()),'query_PRIVATE':q}
    if arm=='lm_FC':
     v=oriented_iou(pred7[q,:3],pred7[q,3:6],np.eye(3),gt[:3],gt[3:6],rot);maxerr=max(maxerr,abs(v-iou));assert abs(v-iou)<1e-6;checks+=1
   rec=rd(root/'records_PRIVATE'/(r['id']+'.json'));fits={k:{a:b for a,b in v.items() if a!='trials'} for k,v in rec['fits'].items() if k.startswith('lm')};rr.append({'row':'ROW'+str(n+1).zfill(6),'source':src,'target':tgt,'id_PRIVATE':r['id'],'scene_PRIVATE':r['scene'],'arms':arms,'fits':fits,'status':rec['readout_status']})
  assert len(rr)==se['n']
  for arm in se['arms']:summary.append({'source':src,'target':tgt,'arm':arm,**met(rr,arm)})
  for sc in sorted({r['scene_PRIVATE'] for r in rr}):
   part=[r for r in rr if r['scene_PRIVATE']==sc]
   for arm in se['arms']:scenes.append({'source':src,'target':tgt,'scene_PRIVATE':sc,'arm':arm,**met(part,arm)})
  for control in controls:
   a=np.array([r['arms']['lm_FC']['IoU'] for r in rr]);b=np.array([r['arms'][control]['IoU'] for r in rr]);d=a-b;paired.append({'source':src,'target':tgt,'method':'lm_FC','control':control,'n':len(rr),'gain_pp':float(d.mean()*100),'better':int((d>1e-12).sum()),'worse':int((d< -1e-12).sum()),'same':int((abs(d)<=1e-12).sum()),'Acc25_rescue':int(((a>.25)&(b<=.25)).sum()),'Acc25_break':int(((a<=.25)&(b>.25)).sum()),'largest_positive_contribution_pp':float(max(0,d.max())/len(rr)*100),'largest_negative_contribution_pp':float(min(0,d.min())/len(rr)*100)})
  dump(S/(direction+'_ALL_ROWS_PRIVATE.json'),rr);allrows+=rr;print('SCORE98',direction,len(rr),flush=True)
 aggregate=[]
 for target in ['all_six_equal','waymo','drone','quad']:
  for arm in matrix['arms']:
   ss=[x for x in summary if x['arm']==arm and (target=='all_six_equal' or x['target']==target)];aggregate.append({'target_group':target,'arm':arm,'directions':len(ss),'direction_task_n':sum(x['n'] for x in ss),**{k:float(np.mean([x[k] for x in ss])) for k in ['mIoU_percent','Acc25_percent','Acc50_percent']}})
 dump(S/'ALL_METHOD_ROWS.json',summary);dump(S/'ALL_SCENE_ROWS_PRIVATE.json',scenes);dump(S/'ALL_PAIRED_ROWS.json',paired);dump(S/'AGGREGATE_ROWS.json',aggregate);dump(S/'SCORING_RECEIPT.json',{'state':'R98_COMPLETE_R95_SIX_DIRECTIONS_SCORED_ONLY_AFTER_SEAL_AUDIT','n':len(allrows),'readouts':len(allrows)*29,'primary':'lm_FC','independent_general_3D_checks':checks,'max_geometry_error':maxerr,'FINAL_access':False,'no_target_tuning':True,'seconds':time.perf_counter()-start});dump(S/'SEALED.json',{'files':{p.name:sha(p) for p in S.glob('*.json')}});print('SCORE98_COMPLETE',len(allrows))
if __name__=='__main__':main()
