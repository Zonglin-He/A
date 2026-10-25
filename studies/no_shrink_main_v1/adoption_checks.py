"""Additional read-only authority checks and live-finish raw-grid bridge."""
from ns_common import *
import argparse

def check_TTA():
 import numpy as np
 audit=read(PREV/'published_TTA/reporting_receipt.json');assert audit['score_sha256']==sha(PREV/'published_TTA/score_receipt.json');validate()
 for ds in COUNTS:
  rr=read(PREV/'published_TTA'/ds/'metrics.json')['private_rows'];ns={r['id']:r for r in read(PREV/'temporal_v2_v2'/ds/'metrics.json')['private_rows']};report=read(P/'TTA'/ds/'report.json')
  for arm,mm in report['means'].items():
   for field,value in mm.items():
    vv=[(ns[x['id']]['values']['NoShrink']if arm=='FullMAP'else x['values'][arm])[field]for x in rr];vv=[x for x in vv if x is not None];assert(value is None and not vv)or abs(value-float(np.mean(vv)))<1e-12
  rec=dict(status='verified_read_only_comparison_update',queries=128,old_TTA_audit_sha256=sha(PREV/'published_TTA/reporting_receipt.json'),old_TTA_score_sha256=sha(PREV/'published_TTA/score_receipt.json'),old_TTA_metrics_sha256=sha(PREV/'published_TTA'/ds/'metrics.json'),NoShrink_audit_sha256=sha(PREV/f'temporal_v2_{ds}_reporting_receipt.json'),new_report_sha256=sha(P/'TTA'/ds/'report.json'),all_means_recomputed=True,optimizer_repeated=False,new_GT_scoring=0,new_Source=0,new_expert=0,adapter_scope='Pinned official Tent/SAR temporal-decoder LayerNorm affine,5episodic updates; unchanged native/joint readouts, noncanonicalBN/fullnetwork. Ours8expert/0updates vsTTA0expert/5updates.',code_sha256=sha(Path(__file__)));atomic(P/'TTA'/ds/'adoption_receipt.json',rec)
def bridge():
 import torch,numpy as np
 from stvg.current import finish
 from stvg.spatial import SpatialGeometry
 from stvg.runtime import guard
 validate();sys.addaudithook(guard);torch.set_num_threads(4);proof=[]
 for ds in COUNTS:
  lookup={x['id']:x for x in records(ds)}
  for key,e in list(fixed(PREV/'raw_reuse_index.json')[ds].items())[:2]:
   assert sha(e['path'])==e['sha256'];raw=torch.load(e['path'],map_location='cpu',weights_only=False,mmap=True);c,parent=suite().load_parent(ds,lookup[key]);cache=dict(frame_ids=c['frame_ids'],posterior={'Native':c['native_prior']},geometry=SpatialGeometry.from_cached_geometry(c['geometry']),native_local_grid=raw['native_local_grid'],planned_positions=c['positions8']);a=finish(cache);b=finish(cache);assert np.array_equal(a['boxes'],b['boxes'])and a['interval']==b['interval']and np.array_equal(a['content_features'],c['content_features']);verify_main(ds,key,dict(FullMAP=a['boxes'],FullMAP_interval=a['interval']));proof.append(dict(dataset=ds,parent_capture_sha256=parent['capture_sha256'],raw_grid_sha256=e['sha256'],current_finish_pool_repeat_and_NoShrink_arrays_exact=True))
 atomic(D/'current_live_finish_bridge.json',dict(status='passed',cases=proof,code_sha256=sha(R/'stvg/current.py'),GT_read=False,new_Source=0,new_expert=0,new_GPU=0));atomic(P/'current_live_finish_bridge.json',read(D/'current_live_finish_bridge.json'))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('command',choices=['bridge','check_TTA']);z=p.parse_args();globals()[z.command]()
