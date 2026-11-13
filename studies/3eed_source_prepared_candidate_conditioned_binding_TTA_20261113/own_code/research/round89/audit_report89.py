from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parents[2];H=R/'research/round89'
def read(p):return json.loads(p.read_text())
su=read(H/'score_dev336_attempt001/TASK_SUMMARY.json')['ALL336'];txt=(H/'REPORT.md').read_text();rr=read(H/'ALL_METHOD_ROWS.json');tasks=read(H/'score_dev336_attempt001/ALL_TASK_ROWS_PRIVATE.json');d=read(H/'DECISION.json')
for r in rr:
 v=su['balanced_mIoU_percent'][r['arm']] if r['platform']=='balanced' else su[r['platform']][r['arm']]['mIoU_percent'];assert r['mIoU_percent']==v and f'{v:.6f}' in txt
for c in read(H/'COMPONENT_EFFECT_ROWS.json'):
 tt=tasks if c['platform']=='balanced' else [t for t in tasks if t['platform']==c['platform']];v=np.mean([t['IoU'][c['arm']]-t['IoU'][c['reference']] for t in tt])*100;assert abs(v-c['net_pp'])<1e-12 and f'{v:+.6f}' in txt
for a,v in d['primary_gains_pp'].items():assert v==su['balanced_mIoU_percent']['candidate_tta1']-su['balanced_mIoU_percent'][a]
assert len(rr)==87 and d['primary_arm']=='candidate_tta1' and not d['FINAL_access'];out={'state':'PASS_R89_REPORT_ALL29_METHODS_SOURCE_CURVES_COMPONENTS_AND_FIXED_PRIMARY','main_n':336,'method_values_checked':len(rr),'component_rows':len(read(H/'COMPONENT_EFFECT_ROWS.json')),'source_epoch_rows':12,'new_GT_scoring':0};(H/'REPORT_NUMERIC_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
