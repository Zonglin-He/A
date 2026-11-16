"""Sealed 24 new fixed readouts only, old 12 task scalar results reused exactly."""
from pathlib import Path
import sys,json,numpy as np,hashlib
R=Path(__file__).resolve().parents[2];H=R/'research/round116';B=R/'crossplatform/round116';O=H/'scores_attempt001';rd=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
sys.path.insert(0,str(R/'research/round6/source_validation_scoring'));from batch_yaw_iou import many_iou
sys.path.insert(0,str(R/'tools'));from obb_iou import oriented_iou

def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def metrics(rr,k):
 y=np.array([r['arms'][k]['IoU'] for r in rr]);return {'n':len(y),'mIoU_percent':float(y.mean()*100),'Acc25_percent':float((y>.25).mean()*100),'Acc50_percent':float((y>.5).mean()*100)}
def main():
 assert rd(H/'NATIVE_AUDIT.json')['n']==768;mat=rd(B/'MATRIX_SEALED.json');O.mkdir(exist_ok=False);rows=[];methods=[];scene=[];paired=[];newn=0;mx=0.;primary=mat['primary'];controls=['neither_correct_fusedABC','native_real_fusedABC','direct_three_geometric','native_gate_only_fusedABC','wrong_depth_trust_raw_fusedABC','random_GN_trust_raw_fusedABC'];compare=[(primary,k) for k in controls]+[('depth_natural_raw_fusedABC','neither_correct_fusedABC'),('depth_trust_bt_fusedABC','neither_correct_fusedABC')]
 for name in mat['directions']:
  src,_,tgt=name.split('_');prior={r['id_PRIVATE']:r for r in rd(R/'research/round114/scores_attempt001'/(name+'_ALL_ROWS_PRIVATE.json'))};gtrows=rd(R/f'research/round96/experiment/OFFICIAL_{tgt.upper()}_VAL_ORDER_PRIVATE.json');rr=[];D=B/'formal'/name
  for r in rd(R/f'research/round99/input/TARGET_{tgt.upper()}128_PRIVATE.json'):
   a=np.load(D/'samples_PRIVATE'/(r['id']+'.npz'));boxes=np.c_[a['source_boxes'],np.zeros(256)].astype(float);boxes[:,3:6]=np.maximum(boxes[:,3:6],1e-6);gt=np.array(gtrows[r['anno_index']]['source_GT7_float32']);ys=many_iou(boxes,gt);arms={};rec=rd(D/'records_PRIVATE'/(r['id']+'.json'))
   for k,p in zip(a['arm_names'],a['scores']):
    q=int(p.argmax())
    if k not in mat['new_arms']:arms[str(k)]=prior[r['id']]['arms'][str(k)];assert arms[str(k)]['query_PRIVATE']==q
    else:
     newn+=1;arms[str(k)]={'IoU':float(ys[q]),'query_PRIVATE':q,'oracle_IoU_diagnostic':float(ys.max())};c,s=np.cos(gt[6]),np.sin(gt[6]);rot=np.array([[c,-s,0],[s,c,0],[0,0,1]]);ind=oriented_iou(boxes[q,:3],boxes[q,3:6],np.eye(3),gt[:3],gt[3:6],rot);er=abs(ind-ys[q]);mx=max(mx,er);assert er<1e-6
   rr.append({'source':src,'target':tgt,'id_PRIVATE':r['id'],'scene_PRIVATE':r['scene'],'protected_alias_fallback':rec['protected_alias_fallback'],'C_available':rec.get('evaluation',{}).get('C_available',False),'seconds':rec['seconds'],'fits':rec.get('fits',{}),'physical_holdout':rec.get('evaluation',{}).get('physical_holdout_loss',{}),'arms':arms})
  dump(O/(name+'_ALL_ROWS_PRIVATE.json'),rr);rows+=rr
  for k in mat['arms']:
   methods.append({'source':src,'target':tgt,'arm':k,**metrics(rr,k)})
   for sc in sorted({r['scene_PRIVATE'] for r in rr}):scene.append({'source':src,'target':tgt,'scene_PRIVATE':sc,'arm':k,**metrics([r for r in rr if r['scene_PRIVATE']==sc],k)})
  for k,c in compare:
   ds=np.array([r['arms'][k]['IoU']-r['arms'][c]['IoU'] for r in rr]);paired.append({'source':src,'target':tgt,'method':k,'control':c,'n':len(ds),'gain_pp':float(ds.mean()*100),'better':int((ds>1e-12).sum()),'worse':int((ds< -1e-12).sum()),'same':int((abs(ds)<=1e-12).sum())})
 agg=[{'group':g,'arm':k,**{m:float(np.mean([x[m] for x in methods if x['arm']==k and (g=='all_six_equal' or x['target']==g)])) for m in ['mIoU_percent','Acc25_percent','Acc50_percent']}} for g in ['all_six_equal','waymo','drone','quad'] for k in mat['arms']];aa={r['arm']:r['mIoU_percent'] for r in agg if r['group']=='all_six_equal'}
 for n,x in [('ALL_METHOD_ROWS',methods),('ALL_SCENE_ROWS_PRIVATE',scene),('ALL_PAIRED_ROWS',paired),('AGGREGATE_ROWS',agg)]:dump(O/(n+'.json'),x)
 assert newn==16128;dec={'primary':primary,'mIoU_percent':aa[primary],'gains_pp':{c:aa[primary]-aa[c] for c in controls},'passes_all_prelocked_controls':all(aa[primary]>aa[c] for c in controls),'adoption':'fixed_expansion_only_if_passes_else_stop_depth_GN_recipe_no_layer_grid','no_posthoc_primary_switch':True,'FINAL_access':False};dump(H/'DECISION.json',dec);dump(H/'GEOMETRY_AUDIT.json',{'n':len(rows),'new_readouts':newn,'reused_old_readouts':7680,'independent_geometry_max_error':mx});print(json.dumps(dec,indent=2),flush=True)
if __name__=='__main__':main()
