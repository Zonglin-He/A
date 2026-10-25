"""Actual CG raw-grid pooling in corrected boxes and new posterior/final-DP."""
from cg_common import *
import numpy as np,torch
from stvg.spatial import SpatialGeometry
from stvg.prototype import one_step
from stvg.current import map_interval
def checked_parent(ds,key):
 e=fixed(CGOLD/f'{ds}_sealed.json')['files'][key]
 for k,h in [('prediction_path','prediction_sha256'),('compact_path','compact_sha256'),('receipt_path','receipt_sha256')]:assert sha(CGOLD/e[k])==e[h]
 c=torch.load(CGOLD/e['compact_path'],map_location='cpu',weights_only=False,mmap=True);assert not c['GT_read'];return c,e
def replay(c):
 legacy();from studies.cgstvg_full_v1.core import solve
 from cgstvg_runtime_v9 import clip_native_boxes
 g=SpatialGeometry.from_cached_geometry(c['geometry']);q,_=one_step(c['E_ROI_features'],c['q0'],kappa=0,beta=c['profile']['beta']);iv=map_interval(c['frame_ids'],q)
 a=dict(Source=c['Source'].copy(),Source_interval=np.asarray(c['native_map']),SourceJointMAP=c['Source'].copy(),SourceJointMAP_interval=np.asarray(map_interval(c['frame_ids'],c['q0'])),HistoricalTwoDP=c['historical_boxes'].copy(),HistoricalTwoDP_interval=c['historical_interval'].copy(),FullMAP=clip_native_boxes(solve(g,iv,c['profile'])),FullMAP_interval=np.asarray(iv),frame_ids=np.asarray(c['frame_ids']),positions8=np.asarray(c['positions8']))
 if c['dataset']=='hc2':
  u=np.ones(len(c['frame_ids']),bool);u[c['positions8']]=False;assert np.array_equal(a['FullMAP'][u],c['Source'][u])
 assert all(np.isfinite(x).all()for x in a.values());return a
def build(cap,native,ds,key):
 old=legacy();from studies.cgstvg_full_v1.core import feedback
 from cgstvg_interfaces_v9 import pool_single_grid
 c,e=checked_parent(ds,key);assert cap['frame_ids']==c['frame_ids']and cap['positions8']==c['positions8']
 assert np.array_equal(cap['Source'],c['Source'])and np.array_equal(cap['q0'],c['q0'])and torch.equal(cap['logits'],c['logits'])and cap['native_map']==c['native_map']
 # Compatibility path retains its own cost. It is not the selected inference.
 tick=time.perf_counter();before,compat=feedback(cap,native,ds,old.PROFILES[ds])
 with np.load(CGOLD/e['prediction_path'])as z:assert all(np.array_equal(before[k],z[k])for k in before)
 assert np.array_equal(compat['pooled_features'],c['pooled_features']);compatibility_seconds=time.perf_counter()-tick
 tick=time.perf_counter();profile={**c['profile'],'kappa':0};g=SpatialGeometry.from_cached_geometry(c['geometry']);pool_boxes=g.E.numpy().copy();pool_boxes[:,2:]*=profile['context']/1.5
 f=pool_single_grid(cap['grids']['projected'],cap['grids']['valid'],pool_boxes);hs=fixed(parent.D/f'CG_{ds}_sealed.json')['files'][key]
 assert sha(parent.D/hs['path'])==hs['sha256']and sha(parent.D/hs['receipt_path'])==hs['receipt_sha256']
 with np.load(parent.D/hs['path'])as z:historical=z['FullMAP'].copy();historical_iv=z['FullMAP_interval'].copy();assert np.array_equal(z['Source'],c['Source'])
 compact=dict(dataset=ds,Source=c['Source'].copy(),native_map=c['native_map'],q0=c['q0'],frame_ids=c['frame_ids'],positions8=c['positions8'],geometry=c['geometry'],profile=profile,E_ROI_features=f,historical_boxes=historical,historical_interval=historical_iv,GT_read=False,first_aux_DP_in_selected_path=False,raw_grids_retained=False)
 arrays=replay(compact);again=replay(compact);assert all(np.array_equal(arrays[k],again[k])for k in arrays)
 proof=dict(original_CG_prediction_sha256=e['prediction_sha256'],original_CG_compact_sha256=e['compact_sha256'],original_CG_receipt_sha256=e['receipt_sha256'],historical_kappa0_prediction_sha256=hs['sha256'],historical_kappa0_receipt_sha256=hs['receipt_sha256'],Source_and_native_logits_and_q0_exact=True,old_profile_all_arrays_and_pool_exact=True,new_E_ROI_actual_raw_grid_pool=True,first_aux_DP_in_selected_path=False,final_DP_retained=True,profile=profile)
 return arrays,compact,proof,dict(compatibility_seconds=compatibility_seconds,selected_finish_seconds=time.perf_counter()-tick)
