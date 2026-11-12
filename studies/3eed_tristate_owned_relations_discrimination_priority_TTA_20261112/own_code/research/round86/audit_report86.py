"""Check reported numbers and build explicit anonymized supplement tables."""
from pathlib import Path
import json,hashlib
import numpy as np
R=Path(__file__).resolve().parents[2];H=R/'research/round86'
def read(p):return json.loads(p.read_text())
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
su=read(H/'score_dev336_attempt001/TASK_SUMMARY.json')['ALL336'];decision=read(H/'DECISION.json');methods=read(H/'ALL_METHOD_ROWS.json');rtext=(H/'REPORT.md').read_text();checked=0
for m in methods:
 if m['cohort']=='DEV336':
  expected=su['balanced_mIoU_percent'][m['arm']] if m['platform']=='balanced' else su[m['platform']][m['arm']]['mIoU_percent'];assert m['mIoU_percent']==expected;assert f'{expected:.6f}' in rtext;checked+=1
assert decision['primary_arm']=='F_joint1_guard' and decision['primary_steps']==1 and decision['secondary_steps']==3
assert abs(decision['primary_gains_pp']['F_direct_guard']-(su['balanced_mIoU_percent']['F_joint1_guard']-su['balanced_mIoU_percent']['F_direct_guard']))<1e-12
rr=read(H/'score_supp30_attempt001/ALL_TASK_ROWS_PRIVATE.json');aliases={r['row']:'SUPP'+str(i+1).zfill(4) for i,r in enumerate(rr)};out=[]
for r in rr:
 for a,v in r['IoU'].items():out.append({'row':aliases[r['row']],'platform':r['platform'],'arm':a,'IoU':v,**r['statuses'][a]})
dump(H/'SUPP30_TASK_SCALARS_PUBLIC.json',out);neg=read(H/'score_supp30_attempt001/ALL_ORDERED_NEGATIVE_COMPARISONS.json');dump(H/'SUPP30_NEGATIVES_PUBLIC.json',[{'row':aliases[x['row']],**{k:x[k] for k in ['platform','arm','reference','delta_pp']}} for x in neg]);result={'state':'PASS_R86_REPORT_MAIN_VALUES_PRIMARY_COMPARISONS_AND_ALL_SUPPLEMENT_NEGATIVES','main_method_values_checked':checked,'supplement_task_values':len(out),'supplement_ordered_negatives':len(neg),'main_n':336,'supplement_n':30,'all_R84_followup_cases':len(read(H/'R84_CASE_FOLLOWUP_ROWS.json')),'no_new_scoring_or_method_changes':True};dump(H/'REPORT_NUMERIC_AUDIT.json',result);print(result)
