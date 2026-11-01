"""Pre-dataset measurement gate reassessment, preserves original two controls."""
from common37 import ROOT,EXP,sha,read
from local_engine37 import check_amendment

def check_attempt003():
 check_amendment();a=read(EXP/'ATTEMPT003_IMPLEMENTATION_LOCK.json');assert a['root_gate_reassessment_sha256']==sha(EXP/'MEASUREMENT_GATE_REASSESSMENT_BEFORE_DATASET_LOCK.json');assert a['original_plan_sha256']==sha(EXP/'PLAN_LOCK.json')
 for p,h in a['program_sha256'].items():assert sha(ROOT/p)==h,p
 return a

def check_plan_inputs003():
 p=read(EXP/"PLAN_LOCK.json")
 for n,h in p["program_sha256"].items():assert sha(ROOT/n)==h,n
 for n,h in p["input_sha256"].items():
  actual=EXP/"ORIGINAL_REVIEW_SOURCES_SNAPSHOT.json" if n=="research/NIGHT_20261101/reviews/native_grounding_G/SOURCES.json" else ROOT/n
  assert sha(actual)==h,n
 return p
