"""Independent saved-only numerical report consistency after all narrative additions."""
from pathlib import Path
import json,numpy as np
H=Path(__file__).resolve().parent;read=lambda p:json.loads(p.read_text());su=read(H/'score_dev336_attempt001/TASK_SUMMARY.json')['ALL336'];tasks=read(H/'score_dev336_attempt001/ALL_TASK_ROWS_PRIVATE.json');text=(H/'REPORT.md').read_text();methods=read(H/'ALL_METHOD_ROWS.json');components=read(H/'COMPONENT_EFFECT_ROWS.json');d=read(H/'DECISION.json')
for r in methods:
 expected=su['balanced_mIoU_percent'][r['arm']] if r['platform']=='balanced' else su[r['platform']][r['arm']]['mIoU_percent'];assert r['mIoU_percent']==expected and f'{expected:.6f}' in text
for r in components:
 tt=tasks if r['platform']=='balanced' else [t for t in tasks if t['platform']==r['platform']];delta=float(np.mean([t['IoU'][r['arm']]-t['IoU'][r['reference']] for t in tt])*100);assert abs(delta-r['net_pp'])<1e-12 and f'{delta:+.6f}' in text
for r in read(H/'SOURCE_AUX_SUMMARY_ROWS.json'):assert f'{r["mIoU_percent"]:.6f}' in text
for k,v in d['gains_pp'].items():assert v==su['balanced_mIoU_percent'][d['primary']]-su['balanced_mIoU_percent'][k]
assert d['primary']=='R76_full_disc_physical_bt' and not d['FINAL_access'];out={'state':'PASS_R91_REPORT_ALL_SOURCE_TARGET_COMPONENTS_AND_FIXED_PRIMARY','main_n':336,'method_rows_checked':len(methods),'component_rows_checked':len(components),'new_GT_scoring':False};(H/'REPORT_NUMERIC_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
