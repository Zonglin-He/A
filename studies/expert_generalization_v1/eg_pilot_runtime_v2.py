"""Original frozen pilot numerical function, only latest scheduling gate changed."""
from eg_execution_common_v2 import *
import eg_pilot_v1 as original
from eg_pilot_gate_v2 import gate

def require_release():
 value=gate();assert value is not None,'Wait for actual active predecessor completion/root release';return value

if __name__=='__main__':
 p=validate_execution();atomic(D/'pilot_runtime_execution_receipt_v2.json',dict(execution_fingerprint=p['fingerprint'],original_pilot_plan_sha256=sha(D/'pilot_plan_v1.json'),worker=identity(),change='Only dependency gate; original numerical run code object unchanged, gamma paused not completed'))
 original.require_release=require_release;original.run()
