"""Only native Source time is used for no-refinement and simple-fusion controls."""
from common import *
import numpy as np,torch
from stvg.spatial import SpatialGeometry,evidence_dp
from stvg.prototype import one_step
from stvg.current import map_interval
from stvg.paper_panel import interpolation

def arrsha(v):
 a=np.asarray(v);assert a.dtype.kind!='O';return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()
def checked_np(base,e):
 assert sha(base/e['path'])==e['sha256']and sha(base/e['receipt_path'])==e['receipt_sha256'];return np.load(base/e['path'],allow_pickle=False)

def derive(ds,row):
 key=row['id'];root=selected.PREV/'spatial';e=fixed(root/f'{ds}_sealed.json')['files'][key];c,pr=selected.checked_payload(root,e)
 pe=fixed(selected.RUN/ds/'sealed.json')['files'][key];ce=fixed(selected.D/f'components_{ds}_sealed.json')['files'][key];se=fixed(STRUCT/f'spatial_{ds}_sealed.json')['files'][key]
 assert c['parent_capture_sha256']==pe['capture_sha256']and e['input_binding']==selected.binding(row,fixed(selected.RUN/'plan.json')['provider_runtime_fingerprint'])
 ids=np.asarray(c['ids']);pos=np.asarray(c['pos']);g=SpatialGeometry.from_cached_geometry(c['geometries']['main']);native=np.asarray(c['original_interval']);q,_=one_step(c['features']['AuxE_ROI'],c['q0'],kappa=0,beta=1);iv=np.asarray(map_interval(ids,q))
 fixed_arrays={k:arrsha(getattr(g,k).numpy())for k in ['E','native_boxes','boxes','features','valid','u','link']}
 full=evidence_dp(g.for_interval(iv))['final'].numpy();disabled=evidence_dp(g.for_interval(native))['final'].numpy()
 with checked_np(selected.RUN/ds,pe)as p,checked_np(selected.D,ce)as cur,checked_np(STRUCT,se)as st:
  assert np.array_equal(ids,p['frame_ids'])and np.array_equal(pos,p['positions8'])and np.array_equal(native,cur['Source_interval'])and np.array_equal(native,st['Source_interval'])and np.array_equal(c['Source'],p['Source'])
  assert np.array_equal(c['E'],p['SharedInit'])and np.array_equal(c['E'],g.E.numpy())and np.array_equal(g.native_boxes.numpy(),p['native_boxes'])
  replace=p['Source'].copy();replace[pos]=p['native_boxes'];inter=interpolation(c['E'],ids,pos,p['native_boxes']);assert np.array_equal(inter,p['Interpolation'])
  assert np.array_equal(full,cur['FullMAP'])and np.array_equal(iv,cur['FullMAP_interval'])
  assert np.array_equal(disabled,st['NoTemporalInference'])and np.array_equal(native,st['NoTemporalInference_interval'])
  a=dict(Source=p['Source'].copy(),EightReplacementNative=replace,SharedCorrectionNative=np.asarray(c['E']).copy(),ExpertInterpolationNative=inter,NoTemporalRefinement=disabled,FullMAP=full,frame_ids=ids,positions8=pos,expert_boxes=p['native_boxes'].copy())
  for name in ARMS:a[name+'_interval']=iv.copy()if name=='FullMAP'else native.copy()
  # Prior component rows may use joint-MAP time or event-gated replacements;
  # their scalar values are never transplanted into these full-tube rows.
  pr.update(parent_prediction_sha256=pe['sha256'],parent_receipt_sha256=pe['receipt_sha256'],parent_capture_sha256=pe['capture_sha256'],current_prediction_sha256=ce['sha256'],current_receipt_sha256=ce['receipt_sha256'],structural_prediction_sha256=se['sha256'],structural_receipt_sha256=se['receipt_sha256'],shared_expert_observations_sha256=arrsha(p['native_boxes']),same_spatial_geometry_sha256=digest(fixed_arrays),native_interval_sha256=arrsha(native),NoTemporal_matches_audited_native_NoTemporalInference=True,legacy_parent_Source_interval_is_not_used=True,legacy_component_metrics_reused_for_simple_fusion=False)
 assert all(arrsha(getattr(g,k).numpy())==v for k,v in fixed_arrays.items())
 assert all(np.isfinite(v).all()for v in a.values())and np.array_equal(a['EightReplacementNative'][pos],a['expert_boxes'])
 pr.update(GT_read=False,new_Source=0,new_expert=0,new_optimizer=0,no_temporal_branch_executed_for_disabled_arm=True,NoTemporal_spatial_DP_reexecuted=True)
 return a,pr
