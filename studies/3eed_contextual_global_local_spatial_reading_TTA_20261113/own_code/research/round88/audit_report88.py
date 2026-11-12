"""Saved-only numeric audit of all final report methods/ablations/decision."""
from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parents[2];H=R/'research/round88'
def read(p):return json.loads(p.read_text())
su=read(H/'score_dev336_attempt001/TASK_SUMMARY.json')['ALL336'];txt=(H/'REPORT.md').read_text();rr=read(H/'ALL_METHOD_ROWS.json');tasks=read(H/'score_dev336_attempt001/ALL_TASK_ROWS_PRIVATE.json');d=read(H/'DECISION.json')
for row in rr:
 v=su['balanced_mIoU_percent'][row['arm']] if row['platform']=='balanced' else su[row['platform']][row['arm']]['mIoU_percent'];assert row['mIoU_percent']==v and f'{v:.6f}' in txt
for c in read(H/'COMPONENT_EFFECT_ROWS.json'):
 key=c['component_removed'];b=key+'_'+c['readout'] if key in ['no_relation','no_local','wrong_space','role_swap'] else key;a='full_'+c['readout'];v=np.mean([t['IoU'][a]-t['IoU'][b] for t in tasks])*100;assert abs(v-c['component_net_pp'])<1e-12 and f'{v:+.6f}' in txt
for ref,gain in d['primary_gains_pp'].items():assert gain==su['balanced_mIoU_percent']['full_tta1']-su['balanced_mIoU_percent'][ref]
assert len(rr)==66 and d['primary_arm']=='full_tta1' and d['FINAL_access'] is False
out={'state':'PASS_R88_REPORT_ALL22_METHODS_COMPONENTS_AND_FIXED_PRIMARY_NUMBERS','main_n':336,'method_values_checked':len(rr),'component_rows':len(read(H/'COMPONENT_EFFECT_ROWS.json')),'fixed_old_cases':len(read(H/'FIXED_R87_CASE_ROWS.json')),'new_GT_scoring':0};(H/'REPORT_NUMERIC_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
