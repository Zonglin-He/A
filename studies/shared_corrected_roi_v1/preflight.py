"""Real cached-feature and raw-grid/live-entry bridges, zero model/labels."""
from shared_common import *
from core import derive
import torch,numpy as np
from stvg.runtime import guard
from stvg.current_shared_roi import finish
from stvg.spatial import SpatialGeometry
validate();sys.addaudithook(guard);torch.set_num_threads(1);proof=[]
for ds in COUNTS:
 lookup={r['id']:r for r in records(ds)}
 for key,e in list(fixed(PREV/'raw_reuse_index.json')[ds].items())[:2]:
  item=dict(row=lookup[key],key=key);a,meta=derive('components',ds,item);b,meta2=derive('components',ds,item);assert meta==meta2 and all(np.array_equal(a[k],b[k])for k in a)
  assert sha(e['path'])==e['sha256'];raw=torch.load(e['path'],map_location='cpu',weights_only=False,mmap=True);c,par=suite().load_parent(ds,lookup[key]);cache=dict(frame_ids=c['frame_ids'],posterior={'Native':c['native_prior']},geometry=SpatialGeometry.from_cached_geometry(c['geometry']),E=c['E'],native_local_grid=raw['native_local_grid'],planned_positions=c['positions8'])
  torch.set_num_threads(4)
  try:x=finish(cache);y=finish(cache)
  finally:torch.set_num_threads(1)
  assert np.array_equal(x['boxes'],a['FullMAP']) and np.array_equal(x['boxes'],y['boxes']) and x['interval']==a['FullMAP_interval'].tolist()==y['interval']
  cp=fixed(PREV/'spatial'/f'{ds}_sealed.json')['files'][key];compact=torch.load(PREV/'spatial'/cp['compact_path'],map_location='cpu',weights_only=False,mmap=True);assert np.array_equal(x['content_features'],compact['features']['AuxE_ROI'])
  proof.append(dict(dataset=ds,parent_capture_sha256=par['capture_sha256'],raw_grid_sha256=e['sha256'],all_arrays_repeat_exact=True,live_finish_actual_E_ROI_pool_and_audited_AuxE_ROI_boxes_interval_exact=True,first_auxiliary_DP_executed=False,final_DP_retained=True))
assert not torch.cuda.is_initialized();receipt=dict(status='passed',cases=proof,GT_read=False,new_model=0,new_Source=0,new_expert=0,new_optimizer=0,code_sha256=sha(Path(__file__)),plan_sha256=sha(D/'plan.json'));atomic(D/'preflight_receipt.json',receipt);atomic(P/'preflight_receipt.json',receipt);print('four real cached/direct-ROI live bridges passed')
