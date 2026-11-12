"""Descriptive complete-cohort subject-head strata, never an adaptation gate."""
from core87 import *
from collections import defaultdict
rows=read(H87/'experiment/DEV336_PRIVATE.json');tasks={r['id_PRIVATE']:r for r in read(H87/'score_dev336_attempt001/ALL_TASK_ROWS_PRIVATE.json')};buckets=defaultdict(list)
for r in rows:
 g=read(B86/'evidence366_attempt002/cases_PRIVATE'/(r['id']+'.json'))['graph'];head=g['entities'][0]['head'].casefold() if g['entities'] else 'UNKNOWN';buckets[head].append(tasks[r['id']])
out=[]
for head,rs in sorted(buckets.items()):
 for arm in ['source','frozen_bi','full_direct','full_tta1','no_subject_tta1','wrong_tta1']:
  delta=np.array([r['IoU'][arm]-r['IoU']['frozen_bi'] for r in rs]);out.append({'subject_head_class':head,'n':len(rs),'drone_n':sum(r['platform']=='drone' for r in rs),'quad_n':sum(r['platform']=='quad' for r in rs),'arm':arm,'mIoU_percent':float(np.mean([r['IoU'][arm] for r in rs])*100),'gain_vs_frozen_pp':float(delta.mean()*100),'improved_n':int((delta>1e-10).sum()),'harmed_n':int((delta< -1e-10).sum()),'descriptive_not_selection_or_independent_evidence':True})
assert sum(len(v) for v in buckets.values())==336;dump(H87/'SUBJECT_HEAD_STRATA_ROWS.json',out)
for r in out:
 if r['arm']=='full_tta1':print(r)
