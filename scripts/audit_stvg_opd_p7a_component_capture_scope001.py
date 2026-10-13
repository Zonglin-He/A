"""Original independent anonymous checker plus two exact presentation sources.

No statistics, runtime pins, actual captures or accepted fits are changed.
One full frozen_forward invocation comprises two original native offset model
calls; the presentation revision makes that counter's scope explicit.
"""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import audit_stvg_opd_p7a_component_public_v1 as original
original.REPORTING=original.REPORTING+[
 'scripts/report_stvg_opd_p7a_component_capture_scope001.py',
 'scripts/audit_stvg_opd_p7a_component_capture_scope001.py']
def run(checkout):
 result=original.run(checkout)
 receipt=original.read(Path(checkout)/original.OUT_REL/'receipts/CAPTURE_CALL_SCOPE_READBACK.json')
 original.privacy(receipt['projection'],'capture_call_scope')
 p=receipt['projection'];assert p['status']=='pass' and p['CPU_only'] and not p['GT_read']
 for k,n in {'qualification_full_capture_invocations':16,'formal_full_capture_invocations':128,
  'qualification_native_offset_model_calls':32,'formal_native_offset_model_calls':256,
  'native_offset_calls_per_full_capture':2}.items():assert p[k]==n
 assert p['extra_decoder_replay_calls_included'] is False
 result['capture_call_scope_verified']=True
 return result

if __name__=='__main__':print(json.dumps(run(sys.argv[1]),allow_nan=False))
