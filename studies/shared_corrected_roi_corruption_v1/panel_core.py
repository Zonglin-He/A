"""New actual E-ROI pool, kappa0 posterior and new-interval final DP."""
from panel_common import *
import numpy as np,torch
from stvg.spatial import SpatialGeometry,evidence_dp
from stvg.representation import extract
from stvg.prototype import one_step
from stvg.current import map_interval
def replay(c):
 q,_=one_step(c['E_ROI_features'],c['q0'],kappa=0,beta=1);iv=map_interval(c['frame_ids'],q);g=SpatialGeometry.from_cached_geometry(c['geometry']);return dict(Source=c['Source'].copy(),Source_interval=np.asarray(c['Source_interval']),FullMAP=evidence_dp(g.for_interval(iv))['final'].numpy(),FullMAP_interval=np.asarray(iv),HistoricalTwoDP=c['historical_boxes'].copy(),HistoricalTwoDP_interval=c['historical_interval'].copy(),frame_ids=np.asarray(c['frame_ids']),positions8=np.asarray(c['positions8']))
def build(ds,key,view,raw):
 c,e=checked_parent(ds,key,view);assert raw['frame_ids']==c['frame_ids']and not raw['GT_read'];g=SpatialGeometry.from_cached_geometry(c['geometry']);ids=c['frame_ids'];tick=time.perf_counter();aux=evidence_dp(g.for_interval([ids[0],ids[-1]+1]))['final'].numpy();oldf,_=extract(raw,aux);assert np.array_equal(oldf,c['content_features']),'Exact original auxiliary-ROI pool required; no prefix fallback';qold,_=one_step(oldf,c['native_prior'],kappa=64,beta=1);old_iv=map_interval(ids,qold)
 with np.load(PREV/e['path'])as z:assert np.array_equal(z['Source'],c['Source'])and np.array_equal(z['Source_interval'],c['Source_interval'])and np.array_equal(z['FullMAP_interval'],old_iv)and np.array_equal(z['FullMAP'],evidence_dp(g.for_interval(old_iv))['final'].numpy())
 compatibility=time.perf_counter()-tick;tick=time.perf_counter();f,_=extract(raw,g.E.numpy());hs=fixed(parent.D/f'corruption_{ds}_sealed.json')['files'][key+'.'+view];assert sha(parent.D/hs['path'])==hs['sha256']and sha(parent.D/hs['receipt_path'])==hs['receipt_sha256']
 with np.load(parent.D/hs['path'])as z:historical=z['FullMAP'].copy();historical_iv=z['FullMAP_interval'].copy()
 compact=dict(Source=np.asarray(c['Source']).copy(),Source_interval=c['Source_interval'],frame_ids=ids,positions8=c['positions8'],geometry=c['geometry'],q0=c['native_prior'],E_ROI_features=f,historical_boxes=historical,historical_interval=historical_iv,GT_read=False,first_aux_DP_in_selected_path=False,raw_grid_retained=False);a=replay(compact);proof=dict(old_prediction_sha256=e['sha256'],old_compact_sha256=e['compact_sha256'],old_receipt_sha256=e['receipt_sha256'],old_aux_ROI_pool_and_MAP_arrays_exact=True,new_E_ROI_actual_pool=True,first_aux_DP_in_selected_path=False,final_new_interval_DP_retained=True,historical_kappa0_prediction_sha256=hs['sha256'],historical_kappa0_receipt_sha256=hs['receipt_sha256']);return a,compact,proof,dict(compatibility_seconds=compatibility,selected_finish_seconds=time.perf_counter()-tick)
