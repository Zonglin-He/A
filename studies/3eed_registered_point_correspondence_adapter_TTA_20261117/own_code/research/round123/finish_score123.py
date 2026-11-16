"""Recover only serializer terminal state; preserve every previously scored value."""
from score123 import *
def main():
 mat=rd(B/'MATRIX_SEALED.json');agg=rd(O/'AGGREGATE_ROWS.json');post=rd(O/'POSTSTRATIFIED_AGGREGATE_ROWS.json');held=rd(H/'HELDOUT_C_TASK_ROWS.json');controls=['mean_context','headmean_context','robust_context','free3_F','wrong3_F','random3_F'];p=mat['primary'];decision={'primary':p,'cohorts':{},'FINAL_access':False,'posthoc_primary_switch':False}
 for co in ['DEV','EXP']:
  aa={r['arm']:r['mIoU_percent'] for r in agg if r['cohort']==co and r['group']=='all_six_equal'};pa={r['arm']:r['mIoU_percent'] for r in post if r['cohort']==co and r['group']=='all_six_equal'};g=[r['gain_pp'] for r in held if r['cohort']==co];decision['cohorts'][co]={'mIoU_percent':aa[p],'gains_pp':{c:aa[p]-aa[c] for c in controls},'poststratified_mIoU_percent':pa[p],'poststratified_gains_pp':{c:pa[p]-pa[c] for c in controls},'passes_all_prelocked_controls':all(aa[p]>aa[c] and pa[p]>pa[c] for c in controls),'unused_C_available_macro_gain_pp':float(np.mean(g)) if all(v is not None for v in g) else None,'unused_C_gate':bool(all(v is not None for v in g) and np.mean(g)>0)}
 decision['development_gate_pass']=all(r['passes_all_prelocked_controls'] and r['unused_C_gate'] for r in decision['cohorts'].values());decision['TTA_robustly_adopted']=False;dump(H/'DECISION.json',decision)
 # Repeat only geometry verification because its terminal maximum was not serialized.
 n=0;mx=0.
 for f in sorted(O.glob('*_ALL_ROWS_PRIVATE.json')):
  rr=rd(f);co=rr[0]['cohort'];src=rr[0]['source'];tgt=rr[0]['target'];gtrows=rd(R/f'research/round96/experiment/OFFICIAL_{tgt.upper()}_VAL_ORDER_PRIVATE.json');reg={r['id']:r for r in rows(co,tgt)}
  for r in rr:
   sample=np.load(B/'formal'/co/(src+'_to_'+tgt)/'samples_PRIVATE'/(r['id_PRIVATE']+'.npz'));boxes=np.c_[sample['source_boxes'],np.zeros(256)].astype(float);boxes[:,3:6]=np.maximum(boxes[:,3:6],1e-6);gt=np.array(gtrows[reg[r['id_PRIVATE']]['anno_index']]['source_GT7_float32']);y=many_iou(boxes,gt);cs,sn=np.cos(gt[6]),np.sin(gt[6]);rot=np.array([[cs,-sn,0],[sn,cs,0],[0,0,1]])
   for arm in mat['new_arms']:
    q=r['arms'][arm]['query_PRIVATE'];assert y[q]==r['arms'][arm]['IoU'];err=abs(oriented_iou(boxes[q,:3],boxes[q,3:6],np.eye(3),gt[:3],gt[3:6],rot)-y[q]);assert err<1e-6;mx=max(mx,err);n+=1
 assert n==32256;dump(H/'GEOMETRY_AUDIT.json',{'n':2304,'new_readouts':n,'reused_readouts':25344,'geometry_max_error':mx,'serialization_recovery_only':True});print(json.dumps(decision,indent=2),flush=True)
if __name__=='__main__':main()
