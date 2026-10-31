"""Root-prescribed fixed regression/fresh engineering4, metadata-only addendum."""
from pathlib import Path
import json,hashlib
D=Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text())
plan=D/'INPUT_PLAN_LOCK.json';rows=read(D/'FORMAL204_GPU_INPUTS.json');cases=read(D/'FORMAL_CASE204_PRIVATE.json');byid={r['id']:r for r in rows};byalias={r['case_alias']:r for r in cases};aliases=['reg_drone_00033','reg_quad_00056','fresh_quad_00001','fresh_drone_00001'];selected=[byalias[a] for a in aliases]
def write(name,x):
 p=D/name;assert not p.exists();p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n');return p
m=write('ENGINEERING4_GPU_INPUTS.json',[byid[c['id']] for c in selected]);c=write('ENGINEERING_CASE4_PRIVATE.json',selected)
x={'state':'FIXED_ENGINEERING4_ADDENDUM_NO_MODEL_OR_GT','program_sha256':sha(__file__),'input_plan_sha256':sha(plan),'engineering_manifest_sha256':sha(m),'engineering_cases_sha256':sha(c),'fixed_aliases':aliases,'selection':'Root predetermined oldR14 Drone33/Quad56 exposed failures and firsthashorderedfreshQuad/Drone; no replacements or signal-based reranking.','old2_are_exposed_regression_not_blind_validation':True,'engineering_cases_in_formal204_exact_once':True,'new_VLM_successful_calls_expected':12,'max_extra_format_failure_engineering_successful_calls':12,'formal204_new_VLM_successful_calls_expected':612,'root_total_successful_new_VLM_budget_cap':720,'input_plan_not_modified':True,'source_model_mask_model_GT_or_new_IoU_execution':False}
p=write('ENGINEERING4_SELECTION_LOCK.json',x);print(json.dumps({'engineering4_lock_sha256':sha(p),'input_plan_sha256':sha(plan)}))
