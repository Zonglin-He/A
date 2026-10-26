from pilot_execution_v3 import *
from pilot_gate_v2 import gate as previous_gate
def gate():
 validate_execution_v3();r=previous_gate()
 if r is None:return None
 assert read(D/'first_raw_restore_v1/completion.json')['all_original_raw_mask_bytes_SHA_exact'];assert read(D/'pilot_v1/hc2/root_raw_restore_bridge_receipt_v2.json')['status']=='original_full_float_SHA_exact_root_bridge_passed'
 return dict(previous_user_pause_gate=r,original_two_commits_preserved=True,extra_engineering_tracking_queries=1,original_full_raw_SHA_exact_restored=True)
