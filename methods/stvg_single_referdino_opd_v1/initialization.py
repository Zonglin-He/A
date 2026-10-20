"""ReferDINO-only initialization; the teacher evidence is shared with OPD."""
import hashlib
import numpy as np
from methods.stvg_referdino_opd_v1.initialization import encode,estimate,apply

def evidence_hash(boxes):
    b=np.asarray(boxes,dtype=np.float64);assert b.shape==(8,4)and np.isfinite(b).all()and(b[:,2:]>0).all()
    return hashlib.sha256(b.tobytes()).hexdigest()

def initialize(source,positions,reference,calibration):
    src=np.asarray(source,dtype=np.float64);ref=np.asarray(reference,dtype=np.float64);positions=list(positions)
    assert len(positions)==len(set(positions))==8 and len(src)>=8
    digest=evidence_hash(ref)
    # Native ReferDINO chooses one track, so every frame contributes a singleton.
    # Relative candidate-score cost is identically zero for a singleton. No
    # calibrated reliability claim or hidden absolute-confidence gate is made.
    pool=[encode(ref[i:i+1])-encode(src)[pos]for i,pos in enumerate(positions)]
    est=estimate(pool,[np.ones(1)for _ in positions],calibration['scale'],calibration['truncation'],1/16,diagnostics=False)
    E=apply(src,est['delta'])
    return E,dict(evidence_sha256=digest,reference_positions=positions,
        selected_reference_indices=est['selected'],support_frames=est['support_frames'],
        rejected_frames=est['rejected_frames'],delta=est['delta'].tolist(),
        objective=est['objective'],evidence='one native ReferDINO selected track at eight positions',
        singleton_relative_score_cost_zero=True,GT_read=False)
