"""CPU promotion adapter for raw recheck captures lacking native_interval.

The frozen full worker ALREADY reconstructs this field by original native logits.
This adapter applies exactly that existing field reconstruction to regression-only
raw captures; no prediction, posterior, geometry, label or inference code changes.
Original failed regression/source/plan remain preserved; no fingerprint waiver.
"""
import sys,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
import full_regression as original
from full import RUN,PUBLIC,atomic,sha
from stvg.temporal import probabilities,mass_table,final_mbr
frozen=original.derive

def raw_capture_adapter(cap):
 if 'native_interval'not in cap:
  ids=cap['frame_ids'];U,q=mass_table(probabilities(cap['records'],[x[-1]for x in cap['logits']]))
  assert U==ids;cap={**cap,'native_interval':final_mbr(ids,q)['interval']}
 return frozen(cap)

if __name__=='__main__':
 before=sha(RUN/'plan.json');original.derive=raw_capture_adapter;original.main();assert sha(RUN/'plan.json')==before
 out=json.loads((RUN/'regression.json').read_text());out.update(regression_adapter=dict(path='tools/promotion_adapter.py',sha256=sha(Path(__file__)),reason='Raw recheck Vid captures lack native_interval. Reconstruct metadata by EXACT existing production native-logit fallback, no time/space score change; failed original regression preserved.',inference_plan_unchanged=True,original_source_hash_preserved=True))
 atomic(RUN/'regression.json',out);atomic(PUBLIC/'REGRESSION.json',out)
