"""Strict graph deletions for the fixed TA main method, without replacement models."""
import sys
from pathlib import Path
R=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(R),str(R/'tools')]
import numpy as np
import torch
from stvg.spatial import SpatialGeometry,evidence_dp
from stvg.evidence import SourceEvidence
from stvg.representation import extract
from stvg.prototype import one_step
from stvg.temporal import final_mbr
from stvg.readout import prediction

ARMS=['Full','NoSharedCorrection','NoSpatialTrajectory','NoTemporalInference']

def solve(geometry,interval):
    return evidence_dp(geometry.for_interval(interval))['final'].numpy()

def native_interval(cap):
    return prediction(cap['native_logits'],torch.as_tensor(cap['Source']),
                      cap['records'],cap['frame_ids'])['physical_interval']

def derive(cap,raw):
    ids=cap['frame_ids'];pos=cap['positions8'];g=SpatialGeometry.from_cached_geometry(cap['geometry'])
    q0=cap['native_prior'];q1,_=one_step(cap['content_features'],q0)
    full_iv=final_mbr(ids,q1)['interval'];original_iv=native_interval(cap)
    full=solve(g,full_iv)
    # Removing the whole time branch removes BOTH posterior revision and MBR.
    no_time=solve(g,original_iv)
    # Recompute all source-dependent geometry after bypassing shared correction.
    seeds=g.side_maps[pos,0].clone()
    evidence=SourceEvidence.from_arrays(raw['spatial_features'],raw['spatial_valid'],ids,pos,seeds)
    uncorrected=SpatialGeometry(evidence,np.asarray(cap['Source']),g.native_boxes.numpy())
    aux=solve(uncorrected,[ids[0],ids[-1]+1])
    f_uncorrected,_=extract(dict(frame_ids=ids,native_local_grid=raw['native_local_grid']),aux)
    q_uncorrected,_=one_step(f_uncorrected,q0);iv_uncorrected=final_mbr(ids,q_uncorrected)['interval']
    no_shared=solve(uncorrected,iv_uncorrected)
    # Bypass the entire spatial graph. E is an already existing upstream output.
    # No observed-frame expert overwrite, interpolation or greedy replacement.
    E=np.asarray(cap['E'])
    f_E,_=extract(dict(frame_ids=ids,native_local_grid=raw['native_local_grid']),E)
    q_E,_=one_step(f_E,q0);iv_E=final_mbr(ids,q_E)['interval']
    arrays=dict(Full=full,Full_interval=np.asarray(full_iv),
        NoSharedCorrection=no_shared,NoSharedCorrection_interval=np.asarray(iv_uncorrected),
        NoSpatialTrajectory=E.copy(),NoSpatialTrajectory_interval=np.asarray(iv_E),
        NoTemporalInference=no_time,NoTemporalInference_interval=np.asarray(original_iv),
        Source=np.asarray(cap['Source']),Source_interval=np.asarray(original_iv),
        SourcePlusMBR=np.asarray(cap['Source']),SourcePlusMBR_interval=np.asarray(cap['native_interval']),
        frame_ids=np.asarray(ids),positions8=np.asarray(pos))
    assert np.array_equal(arrays['NoSpatialTrajectory'],E)
    assert arrays['NoTemporalInference_interval'].tolist()==original_iv
    assert torch.equal(uncorrected.E,torch.as_tensor(cap['Source'],dtype=torch.float64))
    compact=dict(frame_ids=ids,positions8=pos,Source=np.asarray(cap['Source']),E=E,
        q0=q0,geometry=g.state_dict(),uncorrected_geometry=uncorrected.state_dict(),
        full_features=cap['content_features'],uncorrected_features=f_uncorrected,E_features=f_E,
        original_interval=original_iv,native_mbr=cap['native_interval'],GT_read=False)
    return arrays,compact

def replay(c):
    ids=c['frame_ids'];g=SpatialGeometry.from_cached_geometry(c['geometry'])
    u=SpatialGeometry.from_cached_geometry(c['uncorrected_geometry'])
    def iv(f):return final_mbr(ids,one_step(f,c['q0'])[0])['interval']
    a,b,e=iv(c['full_features']),iv(c['uncorrected_features']),iv(c['E_features'])
    return dict(Full=solve(g,a),Full_interval=np.asarray(a),
        NoSharedCorrection=solve(u,b),NoSharedCorrection_interval=np.asarray(b),
        NoSpatialTrajectory=c['E'].copy(),NoSpatialTrajectory_interval=np.asarray(e),
        NoTemporalInference=solve(g,c['original_interval']),NoTemporalInference_interval=np.asarray(c['original_interval']),
        Source=c['Source'],Source_interval=np.asarray(c['original_interval']),
        SourcePlusMBR=c['Source'],SourcePlusMBR_interval=np.asarray(c['native_mbr']),
        frame_ids=np.asarray(ids),positions8=np.asarray(c['positions8']))
