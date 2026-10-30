"""Independentcamera/first-target region constraints; no learnedGTselector.
Camera matrices musthave passed the external GT-free physicalaudit first.
"""
import numpy as np
from caption_regions import parse_target_horizontal_region

def interval_violation(u,low,high):
    u=np.asarray(u,dtype=np.float64);return np.maximum(np.maximum(low-u,u-high),0.)

def spatial_posterior(prior,candidate_uv,projection_valid,image_size_wh,region,source_valid=None,sigma=.25,source_floor=.1):
    """Use the parent-frozen .9 normalized(prior*exp(-E))+.1prior.
    Unsupported candidates receive no cost, not a fabricatedsemanticpenalty;
    their normalizedmass can stillchange and is reported as a limitation.
    Allunsupported/noinformativepenalty returns exactcopy of the originalprior.
    """
    p=np.asarray(prior,dtype=np.float64);uv=np.asarray(candidate_uv,dtype=np.float64);valid=np.asarray(projection_valid,bool).copy();size=np.asarray(image_size_wh,dtype=float)
    if p.ndim!=1 or uv.shape!=(len(p),2) or valid.shape!=p.shape:raise ValueError('teacher shape mismatch')
    if not np.isfinite(p).all() or np.any(p<0) or p.sum()<=0:raise ValueError('invalid semanticprior')
    p=p/p.sum();penalty=np.zeros(len(p));supported=valid&np.isfinite(uv).all(-1)
    if source_valid is not None:supported&=np.asarray(source_valid,bool)
    if sigma<=0 or not 0<=source_floor<=1:raise ValueError('invalid fixedteacherconstants')
    if region is None:return p.copy(),{'reason':'unsupported_language','supported_candidates':0,'penalty':penalty,'informative':False}
    supported&=(uv[:,0]>=0)&(uv[:,0]<size[0])&(uv[:,1]>=0)&(uv[:,1]<size[1])
    if not supported.any():return p.copy(),{'reason':'no_supported_projection','supported_candidates':0,'penalty':penalty,'informative':False}
    normalized_u=uv[:,0]/size[0];penalty[supported]=(interval_violation(normalized_u[supported],region.low,region.high)/sigma)**2
    mass=float(p[supported].sum());e=penalty
    if mass<=0 or np.ptp(e)<=1e-12:return p.copy(),{'reason':'no_relative_spatial_information','supported_candidates':int(supported.sum()),'penalty':penalty,'informative':False}
    weights=p*np.exp(-(e-e.min()));weights=weights/weights.sum()
    out=(1-source_floor)*weights+source_floor*p
    assert np.allclose(out.sum(),1.,atol=1e-14)
    return out,{'reason':'calibrated_region_constraint','supported_candidates':int(supported.sum()),'penalty':penalty,'informative':True,'region':region.__dict__,'sigma':sigma,'source_floor':source_floor,'unsupported_penalty_zero':True,'unsupported_posterior_mass_can_change_by_normalization':True,'calibrated_projection_is_not_semantic_correctness':True}
