"""All cohort predictions sealed before taskGT; all negatives/aliases preserved."""
from core118 import R,H,B,rd,dump,rows,REFMAP
import numpy as np,sys
sys.path.insert(0,str(R/'research/round6/source_validation_scoring'));from batch_yaw_iou import many_iou
sys.path.insert(0,str(R/'tools'));from obb_iou import oriented_iou
O=H/'scores_attempt001'
def metric(rr,k):
 y=np.array([r['arms'][k]['IoU'] for r in rr]);return {'n':len(y),'mIoU_percent':float(y.mean()*100),'Acc25_percent':float((y>.25).mean()*100),'Acc50_percent':float((y>.5).mean()*100)}
def main():
 assert rd(H/'NATIVE_AUDIT.json')['n']==2304;mat=rd(B/'MATRIX_SEALED.json');O.mkdir(exist_ok=False);allrows=[];methods=[];scenes=[];pairs=[];newn=0;mx=0.;p=mat['primary'];controls=['native_F','prepared_F','repaired_F','lateOnly_F','ABC_native','ABC_lSA_rSA','ABC_lSA_rzero','F_lSA_rSAshuffled','F_lSAshuffled_rSAshuffled'];comparisons=[(p,c) for c in controls]+[('F_lSA_rzero','F_lNF_rzero'),('F_lSA_rSA','F_lSA_rzero'),('F_lSA_rSA','F_lSF_rSF')]
 for combo in mat['directions']:
  co,name=combo.split('/');src,_,tgt=name.split('_');prior={r['id_PRIVATE']:r for r in rd(R/f'research/round{113 if co=="DEV" else 117}/scores_attempt001/{name}_ALL_ROWS_PRIVATE.json')};gtrows=rd(R/f'research/round96/experiment/OFFICIAL_{tgt.upper()}_VAL_ORDER_PRIVATE.json');rr=[];D=B/'formal'/co/name
  for r in rows(co,tgt):
   sid=r['id'];a=np.load(D/'samples_PRIVATE'/(sid+'.npz'));boxes=np.c_[a['source_boxes'],np.zeros(256)].astype(float);boxes[:,3:6]=np.maximum(boxes[:,3:6],1e-6);gt=np.array(gtrows[r['anno_index']]['source_GT7_float32']);ys=many_iou(boxes,gt);ar={};meta=rd(D/'records_PRIVATE'/(sid+'.json'));c,s=np.cos(gt[6]),np.sin(gt[6]);rot=np.array([[c,-s,0],[s,c,0],[0,0,1]])
   for k,prob in zip(a['arm_names'],a['scores']):
    k=str(k);q=int(prob.argmax())
    if k not in mat['new_arms']:ar[k]=prior[sid]['arms'][REFMAP[co][k]];assert ar[k]['query_PRIVATE']==q
    else:
     newn+=1;ar[k]={'IoU':float(ys[q]),'query_PRIVATE':q,'oracle_IoU_diagnostic':float(ys.max())};ind=oriented_iou(boxes[q,:3],boxes[q,3:6],np.eye(3),gt[:3],gt[3:6],rot);err=abs(ind-ys[q]);mx=max(mx,err);assert err<1e-6
   rr.append({'cohort':co,'source':src,'target':tgt,'id_PRIVATE':sid,'scene_PRIVATE':r['scene'],'protected_alias_fallback':meta['protected_alias_fallback'],'C_available':meta['C_available'],'seconds':meta['seconds'],'arms':ar})
  dump(O/(co+'_'+name+'_ALL_ROWS_PRIVATE.json'),rr);allrows+=rr
  for k in mat['arms']:
   methods.append({'cohort':co,'source':src,'target':tgt,'arm':k,**metric(rr,k)})
   for sc in sorted({x['scene_PRIVATE'] for x in rr}):scenes.append({'cohort':co,'source':src,'target':tgt,'scene_PRIVATE':sc,'arm':k,**metric([x for x in rr if x['scene_PRIVATE']==sc],k)})
  for k,c in comparisons:
   dd=np.array([x['arms'][k]['IoU']-x['arms'][c]['IoU'] for x in rr]);pairs.append({'cohort':co,'source':src,'target':tgt,'method':k,'control':c,'n':len(dd),'gain_pp':float(dd.mean()*100),'better':int((dd>1e-12).sum()),'worse':int((dd< -1e-12).sum()),'same':int((abs(dd)<=1e-12).sum())})
 agg=[{'cohort':co,'group':g,'arm':k,**{m:float(np.mean([x[m] for x in methods if x['cohort']==co and x['arm']==k and (g=='all_six_equal' or x['target']==g)])) for m in ['mIoU_percent','Acc25_percent','Acc50_percent']}} for co in ['DEV','EXP'] for g in ['all_six_equal','waymo','drone','quad'] for k in mat['arms']]
 for n,x in [('ALL_METHOD_ROWS',methods),('ALL_SCENE_ROWS_PRIVATE',scenes),('ALL_PAIRED_ROWS',pairs),('AGGREGATE_ROWS',agg)]:dump(O/(n+'.json'),x)
 assert newn==43776
 dec={'primary':p,'cohorts':{},'TTA_adopted':False,'FINAL_access':False,'posthoc_primary_switch':False}
 for co in ['DEV','EXP']:
  aa={r['arm']:r['mIoU_percent'] for r in agg if r['cohort']==co and r['group']=='all_six_equal'};dec['cohorts'][co]={'mIoU_percent':aa[p],'gains_pp':{c:aa[p]-aa[c] for c in controls},'passes_all_prelocked_controls':all(aa[p]>aa[c] for c in controls)}
 dec['static_primary_gate']=all(x['passes_all_prelocked_controls'] for x in dec['cohorts'].values());dump(H/'DECISION.json',dec);dump(H/'GEOMETRY_AUDIT.json',{'n':2304,'new_readouts':newn,'reused_readouts':13824,'independent_geometry_max_error':mx});print(__import__('json').dumps(dec,indent=2),flush=True)
if __name__=='__main__':main()
