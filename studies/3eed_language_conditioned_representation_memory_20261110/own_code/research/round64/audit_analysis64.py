"""Recompute every derived table directly from saved task/process rows."""
from common64 import *
p=read(ROOT/'research/round64/experiment/FORMAL_AUDIT_AND_SCORE_PLAN_LOCK.json');H=ROOT/'research/round64';tasks=read(H/'score_dev336_attempt001/ALL_TASK_ROWS_PRIVATE.json');table=read(H/'ALL_ORDER_METHOD_ROWS.json');scenes=read(H/'ALL_ORDER_SCENE_ROWS.json');means=read(H/'MEAN_ACROSS_FIXED_ORDERS.json');process=read(H/'ALL_STREAM_PROCESS_SCALARS.json');orders=read(ROOT/p['orders']);pred=ROOT/p['prediction_root'];err=0.
def close(a,b):
 global err
 err=max(err,abs(float(a)-float(b)));assert abs(float(a)-float(b))<1e-11
for r in table:
 arm=r['order']+'__'+r['method'];ref=r['order']+'__r59_direct_all';v=np.array([x['IoU'][arm] for x in tasks]);base=np.array([x['IoU'][ref] for x in tasks]);close(r['balanced_mIoU'],v.mean()*100);close(r['gain_vs_R59_pp'],(v-base).mean()*100);close(r['Acc25'],(v>.25).mean()*100);close(r['Acc50'],(v>.5).mean()*100);assert r['improved_vs_R59']==int(((v-base)>1e-10).sum()) and r['worsened_vs_R59']==int(((v-base)<-1e-10).sum()) and r['query_changes_vs_R59']==sum(x['queries'][arm]!=x['queries'][ref] for x in tasks)
 for pl in ['drone','quad']:close(r[pl.capitalize()+'_mIoU'],np.mean([x['IoU'][arm] for x in tasks if x['platform']==pl])*100)
for r in means:
 rows=[x for x in table if x['method']==r['method']]
 for k in ['Drone_mIoU','Quad_mIoU','balanced_mIoU','Acc25','Acc50','gain_vs_R59_pp']:close(r[k],np.mean([x[k] for x in rows]))
 close(r['order_min_mIoU'],min(x['balanced_mIoU'] for x in rows));close(r['order_max_mIoU'],max(x['balanced_mIoU'] for x in rows))
for r in scenes:
 pl=r['platform'];scene=sorted({x['scene_PRIVATE'] for x in tasks if x['platform']==pl})[int(r['scene_alias'][-2:])-1];rows=[x for x in tasks if x['platform']==pl and x['scene_PRIVATE']==scene];assert r['n']==len(rows);arm=r['order']+'__'+r['method'];ref=r['order']+'__r59_direct_all';close(r['mIoU'],np.mean([x['IoU'][arm] for x in rows])*100);close(r['gain_vs_R59_pp'],np.mean([x['IoU'][arm]-x['IoU'][ref] for x in rows])*100)
assert len(table)==33 and len(scenes)==165 and len(process)==1008
historical_ABC_probability_error=0.
oldpred=ROOT/p['r59_root'];oldseal=read(oldpred/'SEALED.json');newseal=read(pred/'SEALED.json')
for r in tasks:
 rel='samples_PRIVATE/'+r['id_PRIVATE']+'.npz';assert sha(oldpred/rel)==oldseal['files'][rel] and sha(pred/rel)==newseal['files'][rel];previous=load(oldpred/rel);current=load(pred/rel);oa=previous['arm_names'].tolist();na=current['arm_names'].tolist()
 for o in p['order_names']:
  for a,b in [('source','source'),('frozen_ABC','frozen_ABC'),('r59_direct_all','direct_all')]:
   x=current['scores'][na.index(o+'__'+a)];y=previous['scores'][oa.index(b)]
   if a=='frozen_ABC':
    historical_ABC_probability_error=max(historical_ABC_probability_error,float(abs(x-y).max()));assert x.argmax()==y.argmax() and abs(x.sum()-1)<1e-12 and np.isfinite(x).all()
   else:assert np.array_equal(x,y)

for r in process:
 sid=orders[r['order']][r['platform']][r['stream_position']];a=read(pred/'cases_PRIVATE'/(sid+'.json'))[r['order']]
 for k in ['pool_before','pool_after','write_proposals','replacements','evictions']:assert r[k]==a[k]
 for kind in ['memory','shuffled_values','no_explicit_text']:
  assert r[kind+'_matched_candidates']==sum(a['memory'][kind][v]['matched_candidates'] for v in 'ABC');assert r[kind+'_matched_views']==sum(a['memory'][kind][v]['matched_candidates']>0 for v in 'ABC');close(r[kind+'_median_correction_norm'],np.mean([a['memory'][kind][v]['median_correction_norm'] for v in 'ABC']))
dec=read(H/'DECISION.json');passed=all(next(r for r in table if r['order']==o and r['method']=='memory')['balanced_mIoU']>next(r for r in table if r['order']==o and r['method']==ref)['balanced_mIoU']+1e-10 for o in p['order_names'] for ref in ['r59_direct_all','shuffled_values']);assert dec['locked_development_criterion_passed']==passed and not dec['new_TTA_adopted'];out=H/'independent_derived_audit_attempt001';assert not out.exists();out.mkdir();r={'state':'PASS_R64_ALL_ORDER_METHOD_SCENE_PROCESS_MEANS_AND_DECISION_AUDIT','method_rows':len(table),'scene_rows':len(scenes),'stream_process_rows':len(process),'mean_rows':len(means),'max_scalar_error':err,'orders_not_independent_data':True,'all_order_source_R59_score_vectors_bitexact_historical_references':True,'historical_ABC_all_Top1_identical':True,'historical_ABC_max_probability_drift_reported':historical_ABC_probability_error,'R64_independent_replay_tolerance_unchanged':True,'program_sha256':sha(__file__)};dump(out/'AUDIT_RESULT.json',r);print(r)
