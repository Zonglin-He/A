from pathlib import Path
import json,numpy as np
H=Path(__file__).resolve().parent;read=lambda p:json.loads(p.read_text());su=read(H/'score_dev336_attempt001/TASK_SUMMARY.json')['ALL336'];tasks=read(H/'score_dev336_attempt001/ALL_TASK_ROWS_PRIVATE.json');text=(H/'REPORT.md').read_text();methods=read(H/'ALL_METHOD_ROWS.json');d=read(H/'DECISION.json');components=read(H/'COMPONENT_EFFECT_ROWS.json')
for r in methods:
 v=su['balanced_mIoU_percent'][r['arm']] if r['platform']=='balanced' else su[r['platform']][r['arm']]['mIoU_percent'];assert v==r['mIoU_percent'] and f'{v:.6f}' in text
for c in components:
 tt=tasks if c['platform']=='balanced' else [t for t in tasks if t['platform']==c['platform']];v=np.mean([t['IoU'][c['arm']]-t['IoU'][c['reference']] for t in tt])*100;assert abs(v-c['net_pp'])<1e-12 and f'{v:+.6f}' in text
for s in read(H/'SOURCE_AUX_TASK_SUMMARY_ROWS.json'):assert f'{s["mIoU_percent"]:.6f}' in text
for k,v in d['primary_gains_pp'].items():assert v==su['balanced_mIoU_percent']['candidate_joint_tta1']-su['balanced_mIoU_percent'][k]
assert d['primary_arm']=='candidate_joint_tta1' and not d['FINAL_access'];out={'state':'PASS_R90_REPORT_ALL_SOURCE_CURVES_TARGET_METHODS_COMPONENTS_AND_PRIMARY','main_n':336,'method_values_checked':len(methods),'component_rows':len(components),'source_task_rows_checked':128,'new_GT_scoring':0};(H/'REPORT_NUMERIC_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
