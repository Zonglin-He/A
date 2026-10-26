"""Recompose simple tubes and reuse whole-audited exact native-time spatial DP."""
from common import *
import numpy as np
from stvg.paper_panel import interpolation
from core_runtime_v3 import arrsha,checked_np

def derive(ds,row):
 key=row['id'];pe=fixed(selected.RUN/ds/'sealed.json')['files'][key];ce=fixed(selected.D/f'components_{ds}_sealed.json')['files'][key];se=fixed(STRUCT/f'spatial_{ds}_sealed.json')['files'][key]
 with checked_np(selected.RUN/ds,pe)as p,checked_np(selected.D,ce)as cur,checked_np(STRUCT,se)as st:
  ids=p['frame_ids'].copy();pos=p['positions8'].copy();native=cur['Source_interval'].copy();Source=cur['Source'].copy();E=p['SharedInit'].copy();boxes=p['native_boxes'].copy()
  assert np.array_equal(ids,cur['frame_ids'])and np.array_equal(ids,st['frame_ids'])and np.array_equal(pos,cur['positions8'])and np.array_equal(pos,st['positions8'])
  assert np.array_equal(Source,p['Source'])and np.array_equal(Source,st['Source'])and np.array_equal(native,st['Source_interval'])and np.array_equal(native,st['NoTemporalInference_interval'])
  assert np.array_equal(E,cur['SharedCorrection'])and np.array_equal(cur['FullMAP'],st['FullMAP'])and np.array_equal(cur['FullMAP_interval'],st['FullMAP_interval'])
  replacement=Source.copy();replacement[pos]=boxes;inter=interpolation(E,ids,pos,boxes);assert np.array_equal(inter,p['Interpolation'])
  a=dict(Source=Source,EightReplacementNative=replacement,SharedCorrectionNative=E,ExpertInterpolationNative=inter,NoTemporalRefinement=st['NoTemporalInference'].copy(),FullMAP=cur['FullMAP'].copy(),frame_ids=ids,positions8=pos,expert_boxes=boxes)
  for name in ARMS:a[name+'_interval']=cur['FullMAP_interval'].copy()if name=='FullMAP'else native.copy()
 proof=dict(parent_prediction_sha256=pe['sha256'],parent_receipt_sha256=pe['receipt_sha256'],parent_capture_sha256=pe['capture_sha256'],current_prediction_sha256=ce['sha256'],current_receipt_sha256=ce['receipt_sha256'],structural_prediction_sha256=se['sha256'],structural_receipt_sha256=se['receipt_sha256'],shared_expert_observations_sha256=arrsha(boxes),native_interval_sha256=arrsha(native),NoTemporal_matches_audited_native_NoTemporalInference=True,legacy_parent_Source_interval_is_not_used=True,legacy_component_metrics_reused_for_simple_fusion=False,GT_read=False,new_Source=0,new_expert=0,new_optimizer=0,NoTemporal_spatial_DP_reexecuted=False,inherited_whole_domain_native_spatial_DP_audit_sha256=sha(STRUCT/f'spatial_{ds}_reporting_receipt.json'),inherited_current_Full_audit_sha256=sha(selected.D/f'components_{ds}_reporting_receipt.json'),simple_tubes_freshly_recomposed=True)
 assert all(np.isfinite(v).all()for v in a.values());return a,proof
