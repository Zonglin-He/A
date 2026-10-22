"""Read-only provenance audit for existing TA-STVG adaptation references."""
import json,sys
from pathlib import Path
from tastvg_paper_full import ROOT,PROVIDER,RUN,PUBLIC,read,sha,atomic

def main():
 import numpy as np
 old=read(PROVIDER/'outputs/boxfixed_full_v1/plan.json');result=dict(third_party_generic_TTA='No verified complete TA-STVG third-party generic-TTA block located; do not fill a main-table number from summaries or a different backbone/stream.',historical_project_OPD={},new_OPD_runs=0,new_GT_scoring=0)
 for ds in ('hc2','vidstg'):
  base=PROVIDER/'outputs/formal/tastvg'/ds
  if not(base/'sealed.json').exists():
   result['historical_project_OPD'][ds]=dict(eligible=False,reason='Incomplete historical stream; no truncation/subset result accepted',predictions=len(list((base/'predictions').glob('*.npz'))));continue
  seal=read(base/'sealed.json');rows=[json.loads(s)for s in(PROVIDER/old['jobs'][ds]['manifest']).read_text().splitlines()];proof=dict(queries=len(rows),Source_exact=0,ERef_exact=0,positions_exact=0,native_time_exact=0)
  assert seal['queries']==len(rows)
  for r in rows:
   f=base/'predictions'/(r['id']+'.npz');rec=read(f.with_suffix('.json'));assert sha(f)==rec['prediction_sha256'];assert rec['labels_read']is False
   with np.load(f,allow_pickle=False)as a,np.load(PROVIDER/'outputs/boxfixed_full_v1'/ds/'predictions'/(r['id']+'.npz'),allow_pickle=False)as b:
    for field,x,y in [('Source_exact','Source','Source'),('ERef_exact','E_Ref','E_Ref'),('positions_exact','positions8','positions8'),('native_time_exact','Native_interval','Source_interval')]:assert np.array_equal(a[x],b[y]),(ds,r['id'],field);proof[field]+=1
  result['historical_project_OPD'][ds]=dict(eligible=True,role='historical project OPD-TTA reference, NOT a third-party generic TTA baseline',proof=proof,seal_sha256=sha(base/'sealed.json'),metrics_sha256=sha(base/'metrics.json'),labels_sha256=old['jobs'][ds]['labels_sha256'],original_temporal='Strong tworesolution original fixed method, not our Global6',state='Original per-query temporary content adaptation; Source/ERef/native observations exact. No new run, no online-stream subset.',private_prediction_root=str(base))
 atomic(RUN/'tta_audit.json',result);public=json.loads(json.dumps(result))
 for x in public['historical_project_OPD'].values():x.pop('private_prediction_root',None)
 atomic(PUBLIC/'tta_audit.json',public);print(json.dumps(public))

if __name__=='__main__':main()
