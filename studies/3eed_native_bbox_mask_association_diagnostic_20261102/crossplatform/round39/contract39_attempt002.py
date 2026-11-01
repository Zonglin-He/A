"""Technical serialization fix only, retains failed generation and original lock."""
from common39 import *
def check_fix():
 check_plan();fix=read(EXP/'ATTEMPT002_IMPLEMENTATION_LOCK.json')
 assert fix['original_plan_sha256']==sha(EXP/'PLAN_LOCK.json')
 assert fix['root_technical_budget_authorization_sha256']==sha(EXP/'TECHNICAL_FIX_BUDGET_AUTHORIZATION.json')
 assert len((EXP/'GENERATION_BUDGET.jsonl').read_text().splitlines())==fix['failed_actual_generations_before_fix']==1
 for f,h in fix['program_sha256'].items():assert sha(ROOT/f)==h,f
 return fix
