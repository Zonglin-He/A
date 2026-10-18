"""Independent published aggregate and all unchanged-reference/raw score controls."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load
from scripts.run_stvg_spatial_readout32_v1 import BASE,PREV
PUB=ROOT/'results/stvg_spatial_readout/2026-10-18/matched32'
def run():
 import numpy as np
 public=read(PUB/'RESULTS.json');old=read(ROOT/'results/stvg_identity_trajectory/2026-10-18/matched32/RESULTS.json');by={r['source_alias']:r for r in old['rows']};checks=0;noncontext=0
 for r in public['rows']:
  z=load(BASE/'predictions'/(r['source_alias']+'.pt'))
  for arm,prior in [('Source','Source'),('original_OPD','top1_frame')]:
   for m in ['v','t','s']:
    assert r['values'][arm][m]==by[r['source_alias']]['values'][prior]['Native'][m];checks+=1
  for _,raw in z['raw'].items():
   if raw['context']:continue
   d=raw['detection'];p=raw['old_probe']
   assert d['margin']==p['margin'];checks+=1
   if len(p['target_scores']):assert d['score']==p['target_scores'][0];checks+=1
   noncontext+=1
 for d,g in public['groups'].items():
  rr=[r for r in public['rows']if r['dataset']==d];assert len(rr)==16
  for a,metrics in g.items():
   for m,value in metrics.items():assert abs(value-sum(r['values'][a][m]for r in rr)/16)<1e-12;checks+=1
  for a,p in public['paired'][d].items():
   delta=np.array([r['values'][a]['v']-r['values']['original_OPD']['v']for r in rr])*100;rg=np.random.default_rng(20261018);ci=np.quantile(delta[rg.integers(0,len(rr),(10000,len(rr)))].mean(1),[.025,.975]);assert np.max(abs(ci-p['CI95']))<1e-12;assert abs(delta.mean()-p['mean_pp'])<1e-12
   assert p['positive']==int((delta>1e-9).sum())and p['negative']==int((delta< -1e-9).sum())and p['neutral']==int((abs(delta)<=1e-9).sum());checks+=5
 refs=read(PUB/'CANDIDATE_QUALITY.json');assert len(refs['observations'])==128;assert sum(r['DINO_called']for r in refs['observations'])==124;assert all(r['identity']=='unknown'for r in refs['observations']);checks+=3
 for ds,g in refs['groups'].items():
  rr=[r for r in refs['observations']if r['dataset']==ds];known=[r for r in rr if r['GT_available']];assert len(known)==g['exact_GT'];assert len(rr)==64
  for a in ['old','new']:
   assert abs(g[a+'_all_GT_mean_IoU']-sum(r[a+'_all_available_GT_IoU']for r in known)/len(known))<1e-12;assert g[a+'_success_at_0p5']==sum(r[a+'_all_available_GT_IoU']>=.5 for r in known);checks+=2
 put(BASE/'ROOT_PUBLIC_SCALAR_READBACK.json',dict(status='pass',count=32,checks=checks,noncontext_raw_scores_exact_calls=noncontext,all_historical_Source_and_OPD_metrics_exact=True,all128_planned_slots_retained=True,actual124_DINO_calls=True));print('independent public controls',checks)
if __name__=='__main__':run()
