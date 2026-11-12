"""Existing R80 evidence association only; not a new method or new GT score."""
from pathlib import Path
import json,collections
ROOT=Path(__file__).resolve().parents[2];H=ROOT/'research/round82'
def read(p):return json.loads(p.read_text())
rows=[]
for part,n in [('DEV336',336),('VAL1280',1280),('TRAIN1024',1024)]:
 rr=read(ROOT/('research/round80/score_'+part.lower()+'_attempt001/ALL_TASK_ROWS_PRIVATE.json'));assert len(rr)==n;den=collections.Counter(v['platform'] for v in rr)
 for arm in ['uniform_sgd1','uniform_wrong_sgd1']:
  for descent in [True,False]:
   picked=[]
   for r in rr:
    rec=read(ROOT/'crossplatform/round80/prediction2640_attempt001/records_PRIVATE'/(r['id_PRIVATE']+'.json'))[arm]
    if 'auxiliary_before' not in rec:
     native=read(ROOT/'crossplatform/round79/prediction464_attempt001/records_PRIVATE'/(r['id_PRIVATE']+'.json'))['steps']['uniform'];rec={**native,**rec}
    if (rec['auxiliary_after']-rec['auxiliary_before']<-1e-10)==descent:picked.append(r)
   ds=[r['IoU'][arm]-r['IoU']['frozen_bi'] for r in picked];rows.append({'cohort':part,'arm':arm,'auxiliary_descent':descent,'n':len(picked),'improved':sum(v>1e-10 for v in ds),'worsened':sum(v< -1e-10 for v in ds),'unchanged':sum(abs(v)<=1e-10 for v in ds),'balanced_contribution_pp':sum(50*d/den[r['platform']] for d,r in zip(ds,picked))})
(H/'R80_SAVED_LOSS_TASK_ASSOCIATION.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows,indent=2))
