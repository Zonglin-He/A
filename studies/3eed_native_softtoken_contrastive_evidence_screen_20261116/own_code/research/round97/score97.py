"""All post-seal task scores, equal-direction and incoming-target means; no model."""
from pathlib import Path
import sys,json,hashlib,numpy as np,time
R=Path(__file__).resolve().parents[2];H=R/'research/round97';B=R/'crossplatform/round97';sys.path.insert(0,str(R/'research/round6/source_validation_scoring'));from batch_yaw_iou import many_iou
sys.path.insert(0,str(R/'tools'));from obb_iou import oriented_iou
rd=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def metrics(rr,arm):
 x=np.array([r['arms'][arm]['IoU'] for r in rr]);return {'n':len(x),'mIoU_percent':float(x.mean()*100),'Acc25_percent':float((x>.25).mean()*100),'Acc50_percent':float((x>.5).mean()*100)}
def main():
 assert rd(H/'NATIVE_AUDIT.json')['state'].startswith('PASS_');matrix=rd(B/'MATRIX_SEALED.json');assert matrix['n']==18620;out=H/'scores_attempt001';out.mkdir(exist_ok=False);allrows=[];summ=[];scenes=[];paired=[];maxerr=0.;checks=0;start=time.perf_counter()
 for name,sh in matrix['seals_sha256'].items():
  src,_,tgt=name.split('_');root=B/name;assert sha(root/'SEALED.json')==sh;se=rd(root/'SEALED.json');index=rd(R/f'research/round96/experiment/OFFICIAL_{tgt.upper()}_VAL_ORDER_PRIVATE.json');inputs=rd(H/f'input/TARGET_{tgt.upper()}_PRIVATE.json');assert len(index)==len(inputs)==se['n'];rr=[]
  for i,(r,x) in enumerate(zip(inputs,index)):
   assert r['anno_index']==x['anno_index']==i;f=root/'samples_PRIVATE'/(r['id']+'.npz');assert sha(f)==se['files'][str(f.relative_to(root))];z=np.load(f,allow_pickle=False);boxes=z['source_boxes'].astype(np.float64);boxes[:,3:]=np.maximum(boxes[:,3:],1e-6);pred7=np.c_[boxes,np.zeros(256)];gt=np.array(x['source_GT7_float32'],np.float64);ys=many_iou(pred7,gt);c,s=np.cos(gt[6]),np.sin(gt[6]);rot=np.array([[c,-s,0],[s,c,0],[0,0,1]]);arms={}
   for a,scores in zip(z['arm_names'],z['scores']):
    order=np.argsort(-scores,kind='stable');q=int(order[0]);v=float(ys[q]);arms[str(a)]={'IoU':v,'top5_IoU':float(ys[order[:5]].max()),'top10_IoU':float(ys[order[:10]].max()),'oracle_IoU_diagnostic':float(ys.max()),'query_PRIVATE':q}
    if a=='physical_bt':
     ck=oriented_iou(pred7[q,:3],pred7[q,3:6],np.eye(3),gt[:3],gt[3:6],rot);maxerr=max(maxerr,abs(v-ck));assert abs(v-ck)<1e-6;checks+=1
   rec=rd(root/'records_PRIVATE'/(r['id']+'.json'));rr.append({'row':'ROW'+str(i+1).zfill(6),'source':src,'target':tgt,'scene_PRIVATE':r['scene'],'id_PRIVATE':r['id'],'arms':arms,'updates':rec['updates'],'status':rec['readout_status']})
  for arm in se['arms']:summ.append({'source':src,'target':tgt,'arm':arm,**metrics(rr,arm)})
  for sc in sorted({r['scene_PRIVATE'] for r in rr}):
   rs=[r for r in rr if r['scene_PRIVATE']==sc]
   for arm in se['arms']:scenes.append({'source':src,'target':tgt,'scene_PRIVATE':sc,'arm':arm,**metrics(rs,arm)})
  for comp in ['original_source','prepared_frozen','direct_mean3','direct_feature3','wrong_physical_bt']:
   a=np.array([r['arms']['physical_bt']['IoU'] for r in rr]);b=np.array([r['arms'][comp]['IoU'] for r in rr]);delta=a-b;paired.append({'source':src,'target':tgt,'method':'physical_bt','control':comp,'n':len(rr),'gain_pp':float(delta.mean()*100),'better':int((delta>1e-12).sum()),'worse':int((delta< -1e-12).sum()),'same':int((abs(delta)<=1e-12).sum()),'Acc25_rescue':int(((a>.25)&(b<=.25)).sum()),'Acc25_break':int(((a<=.25)&(b>.25)).sum()),'largest_positive_contribution_pp':float(max(0,delta.max())/len(rr)*100),'largest_negative_contribution_pp':float(min(0,delta.min())/len(rr)*100)})
  dump(out/(name+'_ALL_ROWS_PRIVATE.json'),rr);allrows+=rr;print('SCORE97',name,len(rr),flush=True)
 aggregate=[];arms=list(allrows[0]['arms'])
 for target in ['all_six_equal','waymo','drone','quad']:
  for arm in arms:
   ss=[s for s in summ if s['arm']==arm and (target=='all_six_equal' or s['target']==target)];aggregate.append({'target_group':target,'arm':arm,'directions':len(ss),'direction_task_n':sum(s['n'] for s in ss),**{k:float(np.mean([s[k] for s in ss])) for k in ['mIoU_percent','Acc25_percent','Acc50_percent']}})
 dump(out/'ALL_METHOD_ROWS.json',summ);dump(out/'ALL_SCENE_ROWS_PRIVATE.json',scenes);dump(out/'ALL_PAIRED_ROWS.json',paired);dump(out/'AGGREGATE_ROWS.json',aggregate);dump(out/'SCORING_RECEIPT.json',{'state':'R97_POSTSEAL_ALL_ROWS_SCORED','n':len(allrows),'readouts':len(allrows)*8,'independent_selected_3D_checks':checks,'max_geometry_error':maxerr,'no_target_selection':True,'FINAL_access':False,'seconds':time.perf_counter()-start});dump(out/'SEALED.json',{'files':{f.name:sha(f) for f in out.glob('*.json')}});print('SCORE97_COMPLETE',len(allrows))
if __name__=='__main__':main()
