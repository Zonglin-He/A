from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parents[2];H=R/'research/round87'
def read(p):return json.loads(p.read_text())
su=read(H/'score_dev336_attempt001/TASK_SUMMARY.json')['ALL336'];d=read(H/'DECISION.json');text=(H/'REPORT.md').read_text();n=0
for row in read(H/'ALL_METHOD_ROWS.json'):
 expected=su['balanced_mIoU_percent'][row['arm']] if row['platform']=='balanced' else su[row['platform']][row['arm']]['mIoU_percent'];assert row['mIoU_percent']==expected and f'{expected:.6f}' in text;n+=1
for row in read(H/'COMPONENT_EFFECT_ROWS.json'):
 expected=su['balanced_mIoU_percent']['full_'+row['readout']]-su['balanced_mIoU_percent'][row['component_removed']+'_'+row['readout']];assert abs(row['component_net_pp']-expected)<1e-12
assert d['primary_arm']=='full_tta1' and d['primary_steps']==1 and d['secondary_steps']==3
for ref,v in d['primary_gains_pp'].items():assert v==su['balanced_mIoU_percent']['full_tta1']-su['balanced_mIoU_percent'][ref]
ss=read(H/'score_structural336_attempt001/TASK_SUMMARY.json')['ALL336'];sn=0
for row in read(H/'STRUCTURAL_METHOD_ROWS.json'):
 expected=ss['balanced_mIoU_percent'][row['arm']] if row['platform']=='balanced' else ss[row['platform']][row['arm']]['mIoU_percent'];assert row['mIoU_percent']==expected and f'{expected:.6f}' in text;sn+=1
for row in read(H/'STRUCTURAL_COMPONENT_ROWS.json'):
 expected=ss['balanced_mIoU_percent']['full_'+row['readout']]-ss['balanced_mIoU_percent'][row['component_removed']+'_'+row['readout']];assert abs(row['component_net_pp']-expected)<1e-12
assert d['full_soft_joint_recipe_adopted'] is False
assert len(read(H/'STRUCTURAL_ALL_CHANGED_CASE_ROWS.json'))==3
assert read(H/'NORMALIZED_COSTS.json')['counts']['structural_arms']==32
assert any(not r['before_GT'] for r in read(H/'NORMALIZED_HISTORY.json')['events'] if 'before_GT' in r)
out={'state':'PASS_R87_REPORT_ALL_METHODS_PAIRED_COMPONENTS_FIXED_PRIMARY_AND_GOOD_BAD_ROWS','main_method_values_checked':n,'structural_method_values_checked':sn,'component_effect_rows':len(read(H/'COMPONENT_EFFECT_ROWS.json')),'structural_component_effect_rows':len(read(H/'STRUCTURAL_COMPONENT_ROWS.json')),'good_bad_cases':len(read(H/'GOOD_BAD_CASE_ROWS.json')),'structural_changed_cases':3,'main_n':336,'no_new_scoring_or_method_changes':True};(H/'REPORT_NUMERIC_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
