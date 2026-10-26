"""Unchanged original tracking run, independently versioned pause-aware gate."""
from pilot_execution_v2 import *
from pilot_gate_v2 import gate
import pilot_gate_v1 as old_gate
import pilot_worker_v1 as original
if __name__=='__main__':
 p=validate_execution();old_gate.gate=gate;atomic(D/'pilot_execution_receipt_v2.json',dict(execution_fingerprint=p['fingerprint'],original_science_plan_sha256=sha(D/'pilot_plan_v1.json'),worker=identity(),numeric_entry_unchanged=True));original.run()
