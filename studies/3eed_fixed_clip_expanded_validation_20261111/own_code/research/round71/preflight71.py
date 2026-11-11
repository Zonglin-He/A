from common71 import *
from prepare71 import prepare
import sys
from scipy.special import softmax
sys.path.insert(0,str(ROOT/'research/round70'));from models70 import evidence
selected=[];rr=devrows()
for pl in ['drone','quad']:
 selected+=sorted([r for r in rr if r['platform']==pl],key=lambda r:hashlib.sha256(('R71devgeometry|'+r['id']).encode()).hexdigest())[:2]
for r in selected:
 rel='samples_PRIVATE/'+r['id']+'.npz';a=load(ROOT/'crossplatform/round70/prepared336_attempt001'/rel);z={k:a[k] for k in ['boxes','scores']};got,rec=prepare(r,z)
 for k,v in got.items():assert np.array_equal(a[k],v),k
 c=load(ROOT/'crossplatform/round70/clip336_attempt001'/rel);assert np.array_equal(control_evidence(a,c),evidence(a,c,reliable=False));assert rec['subject_PRIVATE']==read(ROOT/'crossplatform/round70/prepared336_attempt001/cases_PRIVATE'/(r['id']+'.json'))['subject_PRIVATE']
for r in rr:
 rel='samples_PRIVATE/'+r['id']+'.npz';a=load(ROOT/'crossplatform/round70/prepared336_attempt001'/rel);c=load(ROOT/'crossplatform/round70/clip336_attempt001'/rel);v=load(ROOT/'crossplatform/round70/predicted336_attempt001'/rel,['arm_names','scores']);p=softmax(np.log(np.maximum(a['scores'].astype(float),1e-300))+control_evidence(a,c));expected=v['scores'][list(v['arm_names']).index('A_projectonly_joint')];assert np.array_equal(p,expected)
dump(H/'PREFLIGHT_RESULT.json',{'state':'PASS_R71_EXACT_R70_DIRECT_MATH_ALL336_AND_RAW_GEOMETRY_SUBSET','all336_direct_readouts_bit_exact':True,'raw_geometry_subset':4,'no_task_labels_or_new_models':True,'mismatch_valid_groups_only_preserves_unknown_pattern':True});print(read(H/'PREFLIGHT_RESULT.json'))
