"""Independent whole-cohort report/decision/count verification from saved results."""
from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parents[2];H=R/'research/round95';rd=lambda p:json.loads(p.read_text());su=rd(H/'score_dev336_attempt001/TASK_SUMMARY.json')['ALL336'];T=rd(H/'score_dev336_attempt001/ALL_TASK_ROWS_PRIVATE.json');text=(H/'REPORT.md').read_text();M=rd(H/'ALL_METHOD_ROWS.json');E=rd(H/'COMPONENT_EFFECT_ROWS.json')
for x in M:
 tt=T if x['platform']=='balanced' else [t for t in T if t['platform']==x['platform']];v=np.array([t['IoU'][x['arm']] for t in tt]);assert abs(x['mIoU_percent']-float(v.mean()*100))<1e-12 and x['Acc25_percent']==float((v>.25).mean()*100) and x['Acc50_percent']==float((v>.5).mean()*100) and f'{x["mIoU_percent"]:.6f}' in text
for x in E:
 tt=T if x['platform']=='balanced' else [t for t in T if t['platform']==x['platform']];v=np.array([t['IoU'][x['arm']]-t['IoU'][x['reference']] for t in tt]);assert abs(x['net_pp']-float(v.mean()*100))<1e-12 and x['better_n']==int((v>1e-10).sum()) and x['worse_n']==int((v<-1e-10).sum()) and f'{x["net_pp"]:+.6f}' in text
D=rd(H/'DECISION.json');assert D['primary']=='lm_FC' and D['primary_mIoU_percent']==su['balanced_mIoU_percent']['lm_FC']
for k,v in D['gains_pp'].items():assert v==D['primary_mIoU_percent']-su['balanced_mIoU_percent'][k]
assert D['passes_fixed_development_increment_checks']==all(su['balanced_mIoU_percent']['lm_FC']>su['balanced_mIoU_percent'][b] for b in ['R91_frozen','direct_FC','direct_four','lm_shuffle_FC'])
rows=rd(H/'ALL_EXECUTION_ROWS.json');main=rd(H/'ALL_UPDATE_ROWS.json');assert len(rows)==336 and len(main)==1344 and all(x['C_not_used_for_update_acceptance_or_stopping'] for x in rows)
O={'state':'PASS_R95_ALL_REPORT_METHOD_COMPONENT_PRIMARY_AND_EXECUTION_VALUES','method_rows':len(M),'component_rows':len(E),'new_GT_scoring':False,'FINAL_access':False};(H/'REPORT_NUMERIC_AUDIT.json').write_text(json.dumps(O,indent=2)+'\n');print(O)
