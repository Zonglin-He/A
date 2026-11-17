"""Postseal same-information controls and all-negative task scoring."""
from pathlib import Path
import json,sys,numpy as np
R=Path(__file__).resolve().parents[2];H=R/'research/round127';B=R/'crossplatform/round127';O=H/'scores_attempt001';rd=lambda p:json.loads(Path(p).read_text())
sys.path.insert(0,str(R/'research/round118'));from core118 import rows
sys.path.insert(0,str(R/'research/round6/source_validation_scoring'));from batch_yaw_iou import many_iou
sys.path.insert(0,str(R/'tools'));from obb_iou import oriented_iou
REF={'original_source':'original_source','prepared_A':'prepared_A','legacy_ABC_context':'mean_context'}
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def met(rr,k):
 y=np.array([r['arms'][k]['IoU'] for r in rr]);return {'n':len(y),'mIoU_percent':float(y.mean()*100),'Acc25_percent':float((y>.25).mean()*100),'Acc50_percent':float((y>.5).mean()*100)}
def main():
 assert rd(H/'NATIVE_AUDIT.json')['n']==2304;mat=rd(B/'MATRIX_SEALED.json');O.mkdir(exist_ok=False);allrows=[];methods=[];scenes=[];paired=[];newn=0;mx=0.;p=mat['primary'];controls=['zero_F','direct_full_F','direct_native_bridge_F','wrong_fab_full_F','random_F'];comparisons=[(p,c) for c in controls]+[(k,'zero_F') for k in ['ab_native_F','ab_full_F','fab_native_F']]+[('ab_full_F','ab_native_F'),('fab_native_F','ab_native_F'),('fab_full_F','fab_native_F'),('fab_full_F','ab_full_F')]+[(k+'_C','zero_C') for k in ['ab_native','ab_full','fab_native','fab_full','wrong_fab_full','random']]
 for combo in mat['directions']:
  co,name=combo.split('/');src,_,tgt=name.split('_');D=B/'formal'/co/name;prior={r['id_PRIVATE']:r for r in rd(R/f'research/round125/scores_attempt001/{co}_{name}_ALL_ROWS_PRIVATE.json')};gtrows=rd(R/f'research/round96/experiment/OFFICIAL_{tgt.upper()}_VAL_ORDER_PRIVATE.json');priorC={r['id_PRIVATE']:r for r in rd(R/f'research/round120/scores_attempt001/{co}_{name}_ALL_ROWS_PRIVATE.json')};rr=[]
  for r in rows(co,tgt):
   sid=r['id'];a=np.load(D/'samples_PRIVATE'/(sid+'.npz'));boxes=np.c_[a['source_boxes'],np.zeros(256)].astype(float);boxes[:,3:6]=np.maximum(boxes[:,3:6],1e-6);rec=rd(D/'records_PRIVATE'/(sid+'.json'));ar={}
   if not rec['protected_alias_fallback']:
    gt=np.array(gtrows[r['anno_index']]['source_GT7_float32']);ys=many_iou(boxes,gt);cs,sn=np.cos(gt[6]),np.sin(gt[6]);rot=np.array([[cs,-sn,0],[sn,cs,0],[0,0,1]])
   for k,prob in zip(a['arm_names'],a['scores']):
    k=str(k);q=int(prob.argmax())
    if k in REF:ar[k]=prior[sid]['arms'][REF[k]];assert ar[k]['query_PRIVATE']==q
    else:
     newn+=1
     if rec['protected_alias_fallback']:
      ar[k]=dict(prior[sid]['arms']['prepared_A']);assert ar[k]['query_PRIVATE']==q
     else:
      ar[k]={'IoU':float(ys[q]),'query_PRIVATE':q,'oracle_IoU_diagnostic':float(ys.max())};ind=oriented_iou(boxes[q,:3],boxes[q,3:6],np.eye(3),gt[:3],gt[3:6],rot);er=abs(ind-ys[q]);mx=max(mx,er);assert er<1e-6
   rr.append({'cohort':co,'source':src,'target':tgt,'id_PRIVATE':sid,'scene_PRIVATE':r['scene'],'protected_alias_fallback':rec['protected_alias_fallback'],'C_available':rec.get('C_available',False),'B_active':rec.get('fit',{}).get('active',False),'seconds':rec['seconds'],'arms':ar})
  dump(O/(co+'_'+name+'_ALL_ROWS_PRIVATE.json'),rr);allrows+=rr
  for k in mat['arms']:
   methods.append({'cohort':co,'source':src,'target':tgt,'arm':k,**met(rr,k)})
   for sc in sorted({r['scene_PRIVATE'] for r in rr}):scenes.append({'cohort':co,'source':src,'target':tgt,'scene_PRIVATE':sc,'arm':k,**met([r for r in rr if r['scene_PRIVATE']==sc],k)})
  for k,c in comparisons:
   dd=np.array([r['arms'][k]['IoU']-r['arms'][c]['IoU'] for r in rr]);paired.append({'cohort':co,'source':src,'target':tgt,'method':k,'control':c,'n':len(dd),'gain_pp':float(dd.mean()*100),'better':int((dd>1e-12).sum()),'worse':int((dd< -1e-12).sum()),'same':int((abs(dd)<=1e-12).sum())})
 agg=[{'cohort':co,'group':g,'arm':k,**{m:float(np.mean([x[m] for x in methods if x['cohort']==co and x['arm']==k and (g=='all_six_equal' or x['target']==g)])) for m in ['mIoU_percent','Acc25_percent','Acc50_percent']}} for co in ['DEV','EXP'] for g in ['all_six_equal','waymo','drone','quad'] for k in mat['arms']]
 # Full-cohort scene counts only, no label-dependent weights. MissingEXPscenes common-support marked.
 post=[]
 for co in ['DEV','EXP']:
  for name in sorted({r['source']+'_to_'+r['target'] for r in allrows}):
   src,_,tgt=name.split('_');full=rd(R/'research/round119/analysis_attempt001'/(name+'_ALL_ROWS_PRIVATE.json'));cr=[r for r in allrows if r['cohort']==co and r['source']==src and r['target']==tgt];scs={r['scene_PRIVATE'] for r in cr};support=[r for r in full if r['scene_PRIVATE'] in scs];mass=len(support)/len(full)
   for k in mat['arms']:
    values={m:0. for m in ['mIoU_percent','Acc25_percent','Acc50_percent']}
    for sc in scs:
     w=sum(r['scene_PRIVATE']==sc for r in support)/len(support);metrics=met([r for r in cr if r['scene_PRIVATE']==sc],k)
     for m in values:values[m]+=w*metrics[m]
    post.append({'cohort':co,'source':src,'target':tgt,'arm':k,'covered_full_mass':mass,**values})
 postagg=[{'cohort':co,'group':g,'arm':k,'min_covered_full_mass':min(x['covered_full_mass'] for x in post if x['cohort']==co and x['arm']==k and (g=='all_six_equal' or x['target']==g)),**{m:float(np.mean([x[m] for x in post if x['cohort']==co and x['arm']==k and (g=='all_six_equal' or x['target']==g)])) for m in ['mIoU_percent','Acc25_percent','Acc50_percent']}} for co in ['DEV','EXP'] for g in ['all_six_equal','waymo','drone','quad'] for k in mat['arms']]
 for n,x in [('ALL_METHOD_ROWS',methods),('ALL_SCENE_ROWS_PRIVATE',scenes),('ALL_PAIRED_ROWS',paired),('AGGREGATE_ROWS',agg),('POSTSTRATIFIED_METHOD_ROWS',post),('POSTSTRATIFIED_AGGREGATE_ROWS',postagg)]:dump(O/(n+'.json'),x)
 assert newn==41472;decision={'primary':p,'cohorts':{},'FINAL_access':False,'posthoc_primary_switch':False}
 for co in ['DEV','EXP']:
  aa={x['arm']:x['mIoU_percent'] for x in agg if x['cohort']==co and x['group']=='all_six_equal'};pa={x['arm']:x['mIoU_percent'] for x in postagg if x['cohort']==co and x['group']=='all_six_equal'};decision['cohorts'][co]={'mIoU_percent':aa[p],'gains_pp':{c:aa[p]-aa[c] for c in controls},'poststratified_mIoU_percent':pa[p],'poststratified_gains_pp':{c:pa[p]-pa[c] for c in controls},'passes_all_prelocked_controls':all(aa[p]>aa[c] and pa[p]>pa[c] for c in controls)}
 held=[]
 for co in ['DEV','EXP']:
  gains=[]
  for name in sorted({r['source']+'_to_'+r['target'] for r in allrows}):
   src,_,tgt=name.split('_');cr=[r for r in allrows if r['cohort']==co and r['source']==src and r['target']==tgt and r['C_available'] and r['B_active']];d=[r['arms']['fab_full_C']['IoU']-r['arms']['zero_C']['IoU'] for r in cr];gain=float(np.mean(d)*100) if d else None;gains.append(gain);held.append({'cohort':co,'source':src,'target':tgt,'n':len(d),'gain_pp':gain,'better':sum(v>1e-12 for v in d),'worse':sum(v< -1e-12 for v in d)})
  decision['cohorts'][co]['unused_C_available_macro_gain_pp']=float(np.mean(gains)) if all(g is not None for g in gains) else None;decision['cohorts'][co]['unused_C_gate']=bool(all(g is not None for g in gains) and np.mean(gains)>0)
 dump(H/'HELDOUT_C_TASK_ROWS.json',held)
 decision['development_gate_pass']=all(x['passes_all_prelocked_controls'] for x in decision['cohorts'].values());decision['C_is_descriptive_not_configuration_gate']=True;decision['TTA_robustly_adopted']=False;dump(H/'DECISION.json',decision);dump(H/'GEOMETRY_AUDIT.json',{'n':2304,'new_readouts':newn,'reused_readouts':6912,'geometry_max_error':mx});print(json.dumps(decision,indent=2),flush=True)
if __name__=='__main__':main()
