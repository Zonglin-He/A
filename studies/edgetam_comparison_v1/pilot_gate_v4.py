from pilot_execution_v4 import *
from pilot_gate_v3 import gate as previous_gate
def gate():
 validate_execution_v4();v=previous_gate()
 if v is None:return None
 assert read(D/'dtype_positive_control_v1/completion.json')['all_original_raw_mask_bytes_SHA_exact'];assert read(D/'existing_nine_cold_audit_before_dtype_recovery_v1.json')['queries']==9
 return dict(previous_gate=v,existing_nine_paid_queries_preserved=True,original_successful_case_all65float_SHA_exact_after_dtype_compatibility=True,failed_partial_tenth_preserved=True)
