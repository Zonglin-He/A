"""Actual new posterior/MAP/final-DP from exact saved shared-box ROI features."""
from shared_common import *
import numpy as np,torch
from stvg.spatial import SpatialGeometry,evidence_dp
from stvg.prototype import one_step
from stvg.current import map_interval
from supports import greedy

def derive(task,ds,item):
 key=item['key'];root=PREV/'spatial';e=fixed(root/f'{ds}_sealed.json')['files'][key];c,proof=checked_payload(root,e)
 assert not c['GT_read'];assert e['input_binding']==binding(item['row'],fixed(RUN/'plan.json')['provider_runtime_fingerprint'])
 assert c['parent_capture_sha256']==fixed(RUN/ds/'sealed.json')['files'][key]['capture_sha256']
 g=SpatialGeometry.from_cached_geometry(c['geometries']['main']);q,_=one_step(c['features']['AuxE_ROI'],c['q0'],kappa=0,beta=1);iv=map_interval(c['ids'],q);boxes=evidence_dp(g.for_interval(iv))['final'].numpy()
 old=fixed(parent.D/f'spatial_{ds}_sealed.json')['files'][key];assert sha(parent.D/old['path'])==old['sha256']and sha(parent.D/old['receipt_path'])==old['receipt_sha256']
 with np.load(parent.D/old['path'])as z:
  assert np.array_equal(boxes,z['AuxE_ROI'])and np.array_equal(np.asarray(iv),z['AuxE_ROI_interval'])
  historical=z['FullMAP'].copy();historical_iv=z['FullMAP_interval'].copy()
 comp=fixed(parent.D/f'components_{ds}_sealed.json')['files'][key];assert sha(parent.D/comp['path'])==comp['sha256']and sha(parent.D/comp['receipt_path'])==comp['receipt_sha256']
 with np.load(parent.D/comp['path'])as z:a={k:z[k].copy()for k in z.files if k in [v for arm in ['Source','SourceJointMAP','EightReplacement','SharedCorrection','SpatialInference']for v in [arm,arm+'_interval']]}
 a.update(HistoricalAuxDP=historical,HistoricalAuxDP_interval=historical_iv,FullMAP=boxes,FullMAP_interval=np.asarray(iv),FinalIndependent=greedy(g.for_interval(iv)),FinalIndependent_interval=np.asarray(iv),frame_ids=np.asarray(c['ids']),positions8=np.asarray(c['pos']))
 assert all(np.isfinite(v).all()for v in a.values())
 proof.update(audited_structural_prediction_sha256=old['sha256'],audited_structural_receipt_sha256=old['receipt_sha256'],unchanged_components_prediction_sha256=comp['sha256'],unchanged_components_receipt_sha256=comp['receipt_sha256'],ROI='exact raw-grid pooled shared-corrected-box features',first_auxiliary_DP_executed=False,posterior_newly_computed=True,final_interval_DP_executed=True)
 return a,proof
