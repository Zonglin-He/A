"""Strict PTD record-to-state boundary for the prepared conditional modules.

The generic compressor accepts small synthetic vocabularies for testing. This
runtime boundary requires the actual PTD152775 vocabulary and generated IDs.
It is not installed into the running audit; no model forward happens here.
"""
import torch
from vg_tta.desta3d_v3_gap_candidates import compressed_native_state
from vg_tta.desta3d_v3_actuation_support import spatial_positions

def native_state_from_records(prediction,trace,evidence_boxes,evidence_known):
    frames=len(prediction['frame_ids']);branches=trace['branches']
    time_logits=branches[0]['logits'].get('time') if branches else None
    coordinate=branches[1]['logits'].get('coordinate') if len(branches)>1 else None
    positions=list(prediction['positions']);boxes=prediction['boxes_cxcywh'].detach().float()
    validity=prediction['geometry_valid'].detach().bool()
    if coordinate is not None:
        if coordinate.shape[-1]!=152775:raise ValueError('PTD state probabilities require complete152775 vocabulary')
        if spatial_positions(trace)!=positions:raise ValueError('Native geometry and cached action supports differ; preserve failure, do not GT-fill')
        blocks=prediction['readout']['raw_blocks']
        if blocks.shape!=(len(positions),6):raise ValueError('Native block support differs')
        ids=blocks[:,1:5].detach().long()
    else:
        if positions:raise ValueError('Missing policy on declared geometry support')
        ids=None
    xyxy=torch.cat((boxes[:,:2]-.5*boxes[:,2:],boxes[:,:2]+.5*boxes[:,2:]),-1)
    return compressed_native_state(time_logits=time_logits,interval=prediction['interval'],positions=positions,
        boxes_xyxy=xyxy,geometry_valid=validity,coordinate_logits=coordinate,coordinate_token_ids=ids,
        evidence_boxes=evidence_boxes,evidence_known=evidence_known,frames=frames)
