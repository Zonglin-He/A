"""Read-only independent public scalar/CI/denominator and source-selection audit."""
from pathlib import Path
import json,sys
import numpy as np
from scipy.stats import t
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import BASE,read,put,sha
P=ROOT/'results/stvg_joint_event_critic/2026-10-17/pilot32'
def audit():
 x=read(P/'RESULTS.json');old=read(BASE/'POSTSEAL_RESULTS.json');rows=x['rows'];assert len(rows)==32;assert len({r['source_alias']for r in rows})==32;n=0
 original={r['alias']:r for r in old['rows']}
 for r in rows:
  o=original[r['source_alias']];assert r['dataset']==o['dataset'];assert r['all_hypothesis_metrics']==o['all_hypothesis_metrics'];n+=18
  for arm,metrics in r['values'].items():
   for k in ['v','t','s']:assert metrics[k]==o['values'][arm][k];n+=1
   for threshold,key in[(.3,'R_at_0p3'),(.5,'R_at_0p5')]:assert metrics[key]==float(metrics['v']>threshold);n+=1
  s=np.array(r['critic_scores']);j=list(np.unravel_index(s.argmax(),s.shape));assert j==r['joint'];assert r['values']['Joint']['v']==r['all_hypothesis_metrics'][j[0]][j[1]]['v'];n+=1
 for ds,g in x['groups'].items():
  rr=[r for r in rows if r['dataset']==ds];assert len(rr)==16
  for arm,metrics in g['metrics'].items():
   for key,value in metrics.items():assert abs(np.mean([r['values'][arm][key]for r in rr])-value)<1e-14;n+=1
  for arm,z in g['deltas_vs_Direct_native'].items():
   d=np.array([r['values'][arm]['v']-r['values']['Direct_native']['v']for r in rr]);mean=d.mean();ci=t.interval(.95,df=15,loc=mean,scale=d.std(ddof=1)/4);assert abs(z['mean_pp']-mean*100)<1e-12
   assert np.allclose(z['t95_pp'],np.array(ci)*100,atol=1e-12,rtol=0);assert z['positive']==int((d>1e-8).sum());assert z['negative']==int((d< -1e-8).sum());assert z['neutral']==int((abs(d)<=1e-8).sum());n+=6
  total=0;right=0.
  for r in rr:
   scores=np.array(r['critic_scores']).flatten();quality=np.array([[a['v']for a in q]for q in r['all_hypothesis_metrics']]).flatten();mask=(quality[:,None]-quality[None,:])>1e-8;d=(scores[:,None]-scores[None,:])[mask];total+=len(d);right+=float(((d>0)+.5*(d==0)).sum())
  assert total==g['strict_pairs']and abs(right/total-g['strict_pair_accuracy'])<1e-14;n+=2
 src=read(P/'SOURCE_CANDIDATE_QUALITY.json');private={r['alias']:r for r in read(BASE/'SOURCE_LABELS.json')['rows']};assert len(src['rows'])==len(private)==96
 for r in src['rows']:
  o=private[r['source_alias']];assert r['dataset']==o['dataset']and r['split']==o['split']and r['six_candidate_metrics']==o['values'];n+=18
 training=read(P/'SOURCE_TRAINING.json')
 for ds,c in training['critics'].items():
  assert c==read(BASE/'critics'/(ds+'.json'));curve=c['history'];assert len(curve)==80
  best=min(curve,key=lambda r:float('inf')if r['validation_pair_loss']is None else r['validation_pair_loss']);assert c['selected_epoch']==best['epoch'];assert c['selected_validation_pair_accuracy']==best['validation_pair_accuracy'];n+=len(curve)*3+2
 lock=read(BASE/'RUNTIME_LOCK003.json');assert sha(BASE/'PLAN_LOCK.json')==lock['plan_sha256'];assert sha(BASE/'SOURCE_LABELS.json')==lock['source_labels_sha256']
 assert not x['advance_matched_OPD']and read(P/'DECISION.json')['status']=='stop_finite_pilot'
 result=dict(status='pass',count=32,source_candidates=96,checked_scalar_equalities=n,all_signed_neutral_unknown_rows_preserved=True,all_group_means_R_and_parent_t95_rebuilt=True,all_source_validation_curves_and_earliest_minimum_selection_exact=True,all_target_scores_and_joint_choices_exact=True,no_further_model_or_GT_reader=True)
 put(BASE/'ROOT_PUBLIC_SCALAR_READBACK.json',result);put(P/'ROOT_PUBLIC_SCALAR_READBACK.json',result);print(json.dumps(result))
if __name__=='__main__':audit()
