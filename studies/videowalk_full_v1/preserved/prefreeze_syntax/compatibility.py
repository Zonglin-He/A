"""Strict old Source/current checks, never input E or DP to Matching."""
from common import *
from stvg.spatial import SpatialGeometry,evidence_dp
from stvg.evidence import SourceEvidence,extract_visual_grids
from stvg.representation import extract
from stvg.prior import native_prior
from stvg.readout import prediction
from stvg.positions import observation_positions
from stvg.prototype import one_step
from stvg.current_shared_roi import map_interval
def current_exact(cap,raw,control):
 assert np.array_equal(raw['Source'],cap['Source'])and raw['records']==cap['records']and len(raw['native_logits'])==len(cap['native_logits'])==2
 assert all(torch.equal(a,b)for a,b in zip(raw['native_logits'],cap['native_logits']))
 ids=cap['frame_ids'];q0=native_prior(raw['records'],raw['native_logits'],ids);assert np.array_equal(q0,cap['native_prior']);pr=prediction(raw['native_logits'],torch.as_tensor(raw['Source']),raw['records'],ids);assert observation_positions(ids,pr['indices'])==cap['positions8']
 
 if 'spatial_inputs'in raw:features,valid,ii,shape,_=extract_visual_grids(raw['spatial_inputs'],ids);assert ii==ids
 else:features,valid,shape=raw['spatial_features'],raw['spatial_valid'],raw['spatial_grid_shape']
 g0=SpatialGeometry.from_cached_geometry(cap['geometry']);ev=SourceEvidence.from_arrays(features,valid,ids,cap['positions8'],g0.side_maps[cap['positions8'],0].clone());g=SpatialGeometry(ev,np.asarray(cap['E']),np.asarray(g0.native_boxes))
 checked=0
 for k,v in cap['geometry'].items():
  if torch.is_tensor(v):assert k in g.__dict__and torch.equal(v,g.__dict__[k]),k;checked+=1
 aux=evidence_dp(g.for_interval([ids[0],ids[-1]+1]))['final'].numpy();f_aux,_=extract(raw,aux);assert np.array_equal(f_aux,cap['content_features'])
 f,_=extract(raw,np.asarray(cap['E']));q,_=one_step(f,q0,kappa=0,beta=1);iv=np.asarray(map_interval(ids,q));assert np.array_equal(iv,control['Current_interval']);path=evidence_dp(g.for_interval(iv))['final'].numpy();assert np.array_equal(path,control['Current'])
 return features,valid,shape,dict(original_Source_records_logits_q0_observation_positions_exact=True,original_current_geometry_tensors_exact=checked,original_pretext_aux_pool_exact=True,current_Epool_posterior_interval_finalDP_exact=True)
