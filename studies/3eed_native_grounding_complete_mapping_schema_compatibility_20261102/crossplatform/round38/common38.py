"""R38 cached-format engineering correction. No model, image, or GT access."""
import sys,json,re,math,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'crossplatform/round37'))
import common37 as strict
from common37 import rows,prepared,arrays,read,dump,sha,seal,verify,load_npz,direct_readout
BASE=ROOT/'crossplatform/round38';EXP=ROOT/'research/round38/experiment';R37=ROOT/'crossplatform/round37/inference208_attempt003'
ARMS=['source','R29_direct','legacy_G','point_strict_fullmap','point_strict_oldsubset','som_strict_fullmap','som_strict_oldsubset','point_compatible_fullmap','point_compatible_oldsubset','som_compatible_fullmap','som_compatible_oldsubset']
def check_plan():
 p=read(EXP/'PLAN_LOCK.json')
 for f,h in p['program_sha256'].items():assert sha(ROOT/f)==h,f
 for f,h in p['input_sha256'].items():assert sha(ROOT/f)==h,f
 return p

def parse_compatible(raw,interface):
 t=str(raw).strip();wrappers=[]
 m=re.fullmatch(r'```(?:json)?\s*(.*?)\s*```',t,re.S|re.I)
 if m:t=m.group(1).strip();wrappers.append('markdown_json_fence')
 m=re.fullmatch(r'<answer>\s*(.*?)\s*</answer>',t,re.S|re.I)
 if m:t=m.group(1).strip();wrappers.append('answer_tag')
 def fail(reason):return {'status':'parse_failure','reason':reason,'value':None,'wrappers':wrappers,'singleton_unwrapped':False,'optional_label_ignored':False,'value_equal_raw_unique_position':False}
 def pairs(ps):
  d={}
  for k,v in ps:
   if k in d:raise ValueError('duplicate JSON key')
   d[k]=v
  return d
 try:o=json.loads(t,object_pairs_hook=pairs,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('nonfinite JSON constant')))
 except Exception as e:return fail(type(e).__name__)
 singleton=False
 if isinstance(o,list):
  if len(o)!=1 or not isinstance(o[0],dict):return fail('exact_single_object_list_required')
  o=o[0];singleton=True
 if not isinstance(o,dict):return fail('object_or_singleton_object_list_required')
 allowed={'point':{'point','point2d','point_2d'},'som':{'target_id','id','candidate_id'}}[interface];keys=set(o);loc=keys&allowed
 if len(loc)!=1:return fail('exactly_one_recognized_location_key_required')
 key=next(iter(loc))
 if not keys<=allowed|{'label'}:return fail('unknown_extra_key')
 if 'label' in o and not isinstance(o['label'],str):return fail('optional_label_must_be_inert_string')
 # Label content is never interpreted, executed, or used to select a location.
 # Another location KEY (even with the same coordinates) is always rejected.
 if len(keys-{'label'})!=1:return fail('multiple_location_fields')
 v=o[key]
 if v is not None:
  if interface=='point':okay=isinstance(v,list) and len(v)==2 and all(isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) for x in v)
  else:okay=isinstance(v,int) and not isinstance(v,bool)
  if not okay:return fail('invalid_value_type_or_shape_or_nonfinite')
 return {'status':'explicit_null' if v is None else 'parsed','reason':None,'value':v,'wrappers':wrappers,'canonical_key':'point' if interface=='point' else 'target_id','input_key':key,'singleton_unwrapped':singleton,'optional_label_ignored':'label' in o,'ignored_label_PRIVATE':o.get('label'),'value_equal_raw_unique_position':v==o[key],'raw_unique_position':o[key]}
