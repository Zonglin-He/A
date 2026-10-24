"""Frozen domain-specific CG profiles, with lossless sufficient-state replay."""
import sys
from pathlib import Path
R = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(R), str(R/'tools'), str(R/'studies/paper_followup_v1'),
               str(R/'studies/paper_review_suite_v1')]
import numpy as np
import torch
from full import read
from stvg.initialization import initialize
from stvg.predictor import output_writeback
from stvg.evidence import SourceEvidence
from stvg.spatial import SpatialGeometry, evidence_dp
from stvg.prototype import one_step
from stvg.temporal import final_mbr
from cgstvg_interfaces_v9 import pool_single_grid
from cgstvg_runtime_v9 import clip_native_boxes

PROFILES = {
    'hc2': dict(calibration='Transferred', gamma=0., unobserved=0., link=0.,
                beta=.5, kappa=64, context=1.5),
    'vidstg': dict(calibration='Transferred', gamma=.5, unobserved=.5, link=.25,
                   beta=1., kappa=64, context=1.),
}
ARMS = ['SourceNative', 'Source', 'SharedCorrection', 'Space', 'Full']

def solve(geo, interval, profile):
    bank = geo.for_interval(interval)
    u = profile['unobserved']
    if u != .5:
        observed = bank.event & bank.observed
        unobserved = bank.event & ~bank.observed
        ow = (1-u)*observed.double()/max(1, int(observed.sum()))
        uw = u*bank.supported.double()*unobserved/max(1, int(unobserved.sum()))
        bank.score_unary = (ow+uw)[:, None]*bank.u
    if profile['link'] != .25:
        bank.score_pair *= profile['link']/.25
    return evidence_dp(bank)['final'].numpy()

def replay(compact, profile):
    """All spatial and posterior decisions replay exactly, no raw grids needed."""
    geo = SpatialGeometry.from_cached_geometry(compact['geometry'])
    ids = compact['frame_ids']
    features = compact['pooled_features']
    q1, _ = one_step(features, compact['q0'], kappa=profile['kappa'], beta=profile['beta'])
    iv = final_mbr(ids, q1)['interval']
    return dict(SourceNative=compact['Source'].copy(), Source=compact['Source'].copy(),
        SourceNative_interval=np.asarray(compact['native_map']),
        Source_interval=np.asarray(compact['native_mbr']),
        SharedCorrection=clip_native_boxes(geo.E.numpy()),
        SharedCorrection_interval=np.asarray(compact['native_mbr']),
        Space=clip_native_boxes(solve(geo, compact['native_mbr'], profile)),
        Space_interval=np.asarray(compact['native_mbr']),
        Full=clip_native_boxes(solve(geo, iv, profile)), Full_interval=np.asarray(iv),
        frame_ids=np.asarray(ids), positions8=np.asarray(compact['positions8']))

def feedback(cap, teacher, ds, profile):
    source = cap['Source']; pos = cap['positions8']; ids = cap['frame_ids']
    ref = np.asarray(teacher['boxes'], np.float64)
    calibration = read(R/'outputs/paper_followup_v1/cg_calibrations.json')[ds][profile['calibration']]
    raw, init = initialize(source, pos, ref, calibration)
    E = output_writeback(source, raw, profile['gamma'])
    g = cap['grids']
    ev = SourceEvidence.from_native_masks(g['spatial'], g['spatial_valid'], ids, pos,
        teacher['native_mask_logits'], teacher['input_shape'][-2:], g['shape'],
        selected_query=teacher['selected_query'])
    geo = SpatialGeometry(ev, E, ref)
    aux = solve(geo, [ids[0], ids[-1]+1], profile)
    poolboxes = aux.copy()
    poolboxes[:, 2:] *= profile['context']/1.5
    features = pool_single_grid(g['projected'], g['valid'], poolboxes)
    compact = dict(schema='cg_full_sufficient_state_v1', Source=source.copy(),
        native_map=cap['native_map'], native_mbr=cap['native_mbr'],
        frame_ids=ids, positions8=pos, q0=cap['q0'], logits=cap['logits'],
        geometry=geo.state_dict(), aux=aux, pooled_features=features,
        initialization=init, profile=profile, GT_read=False,
        raw_grids_retained=False, exact_replay_includes='geometry/aux/pooled prefix/q0/posterior/MBR/finalDP')
    arrays = replay(compact, profile)
    assert np.array_equal(aux, solve(SpatialGeometry.from_cached_geometry(compact['geometry']),
                                    [ids[0], ids[-1]+1], profile))
    if ds == 'hc2':
        observed = np.zeros(len(ids), bool); observed[pos] = True
        assert np.array_equal(arrays['Full'][~observed], source[~observed])
    return arrays, compact
