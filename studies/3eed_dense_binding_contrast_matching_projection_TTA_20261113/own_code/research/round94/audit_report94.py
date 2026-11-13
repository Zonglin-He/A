"""Saved numerical report validation independent of report generation."""
from pathlib import Path
import json,numpy as np
H=Path(__file__).resolve().parent;read=lambda p:json.loads(p.read_text());su=read(H/'score_dev336_attempt001/TASK_SUMMARY.json')['ALL336'];tasks=read(H/'score_dev336_attempt001/ALL_TASK_ROWS_PRIVATE.json');text=(H/'REPORT.md').read_text();methods=read(H/'ALL_METHOD_ROWS.json');components=read(H/'COMPONENT_EFFECT_ROWS.json')
for r in methods:
 x=su['balanced_mIoU_percent'][r['arm']] if r['platform']=='balanced' else su[r['platform']][r['arm']]['mIoU_percent'];assert x==r['mIoU_percent'] and f'{x:.6f}' in text
for r in components:
 tt=tasks if r['platform']=='balanced' else [t for t in tasks if t['platform']==r['platform']];delta=float(np.mean([t['IoU'][r['arm']]-t['IoU'][r['reference']] for t in tt])*100);assert abs(delta-r['net_pp'])<1e-12 and f'{delta:+.6f}' in text
d=read(H/'DECISION.json');assert d['mainline']=='R91_R76_full_disc' and d['primary']=='binding_bt1' and not d['FINAL_access']
for k,v in d['gains_pp'].items():assert v==su['balanced_mIoU_percent'][d['primary']]-su['balanced_mIoU_percent'][k]
solver=read(H/'score_solver336_attempt001/TASK_SUMMARY.json')['ALL336'];st=read(H/'score_solver336_attempt001/ALL_TASK_ROWS_PRIVATE.json')
for r in read(H/'SOLVER_METHOD_ROWS.json'):
 value=solver['balanced_mIoU_percent'][r['arm']] if r['platform']=='balanced' else solver[r['platform']][r['arm']]['mIoU_percent'];assert value==r['mIoU_percent'] and f'{value:.6f}' in text
for r in read(H/'SOLVER_EFFECT_ROWS.json'):
 tt=st if r['platform']=='balanced' else [v for v in st if v['platform']==r['platform']];dd=float(np.mean([v['IoU'][r['arm']]-v['IoU'][r['reference']] for v in tt])*100);assert abs(dd-r['net_pp'])<1e-12 and f'{dd:+.6f}' in text
assert d['secondary_solver']['mIoU_percent']==solver['balanced_mIoU_percent']['gn_bt'] and d['secondary_solver']['not_promoted_to_original_primary']
x={'state':'PASS_R94_REPORT_ALL_METHOD_COMPONENT_AND_FIXED_PRIMARY_VALUES','method_rows':len(methods),'component_rows':len(components),'new_GT_scoring':False};(H/'REPORT_NUMERIC_AUDIT.json').write_text(json.dumps(x,indent=2)+'\n');print(x)
