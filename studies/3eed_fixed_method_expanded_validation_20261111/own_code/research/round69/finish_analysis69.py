"""Run pinned saved-only analysis, independent audit and report sequentially."""
from common69 import *
import subprocess,sys
H=ROOT/'research/round69';lock=read(H/'experiment/DERIVED_SUMMARIES_PROGRAM_LOCK.json');pin(lock);assert read(H/'independent_scalar_audit_attempt001/AUDIT_RESULT.json')['state'].startswith('PASS_')
for name in ['analyze69.py','audit_derived69.py','write_report69.py']:
 subprocess.run([sys.executable,str(H/name)],cwd=ROOT,check=True)
dump(H/'ANALYSIS_RECEIPT.json',{'state':'R69_FIXED_METHOD_EXPANDED_VALIDATION_ANALYZED_NO_RESELECT','program_lock_sha256':sha(H/'experiment/DERIVED_SUMMARIES_PROGRAM_LOCK.json'),'report_sha256':sha(H/'REPORT.md'),'decision_sha256':sha(H/'DECISION.json'),'independent_derived_sha256':sha(H/'independent_derived_audit_attempt001/AUDIT_RESULT.json')})
