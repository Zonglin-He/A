from pathlib import Path
import json, numpy as np
from capacity_metrics import MASKS,review_capacity_case
from score_gt_controlled import ARMS
source=np.array([1,2,3,-1,2,3],np.float32);target=np.array([0,1,2,4,5,6],np.float32);boxes={'source':source};mapping={}
for g,m in MASKS.items():
 expected=np.where(m,target,source);mapping[g]={'direct':'GT_'+g+'__direct_target','param1':'GT_'+g+'__param1','param3':'GT_'+g+'__param3','exactfit':'GT_'+g+'__exactfit'}
 for key,name in mapping[g].items():boxes[name]=expected.copy() if key in ('direct','exactfit') else np.where(m,source+.1*(target-source),source)
assert set(boxes)==set(ARMS)
r=review_capacity_case(source,target,boxes,mapping);assert all(v['exactfit_within_registered_tolerance'] for v in r.values());assert r['C']['source_raw_nonpositive_size_count']==1
bad={n:v.copy() for n,v in boxes.items()};bad['GT_C__param1'][3]=1e-6
try:review_capacity_case(source,target,bad,mapping)
except AssertionError:pass
else:raise AssertionError('Inactive source size silently clamped')
p=Path(__file__).with_name('CPU_CAPACITY_ARRAY_CONTRACT_FIXTURE.json');assert not p.exists();p.write_text(json.dumps({'state':'synthetic_CPU_capacity13_array_contract_pass','actual13_names':True,'inactive_negative_source_size_preserved':True,'inactive_clamp_rejected':True,'registered5e5_comparison':True,'no_model_GPU_data_or_GT_annotation_read':True},indent=2)+'\n')
print('synthetic capacity13 contract passed')
