"""Original numeric run unchanged; only current small-comparison release gate."""
from eg_execution_common_v5 import *
from eg_pilot_gate_v5 import gate
import eg_pilot_v1 as original
def release():
 v=gate();assert v is not None;return v
if __name__=='__main__':
 p=validate_execution_v5();atomic(D/'pilot_runtime_execution_receipt_v5.json',dict(execution_fingerprint=p['fingerprint'],original_science_plan_sha256=sha(D/'pilot_plan_v1.json'),worker=identity(),new_Source=0,full_extension_deferred=True));original.require_release=release;original.run()
