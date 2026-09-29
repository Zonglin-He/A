"""Detached cached expert reward and explicit pair/tie semantics."""
import numpy as np
from vg_tta.box_stability_diagnostics_v1 import overlap

ALL_PAIRS=[(i,j) for i in range(9) for j in range(i+1,9)]
ANTITHETIC=[(1+2*m,2+2*m) for m in range(4)]

def rewards(candidate_boxes,expert_boxes,valid):
    valid=np.asarray(valid,bool)
    if not valid.any():return None
    return np.array([overlap(np.asarray(b)[valid],np.asarray(expert_boxes)[valid]).mean() for b in candidate_boxes])

def pair_record(reward_difference,gt_difference,epsilon=1e-12):
    es=int(reward_difference>epsilon)-int(reward_difference< -epsilon);gs=int(gt_difference>epsilon)-int(gt_difference< -epsilon)
    return dict(reward_difference=float(reward_difference),gt_difference=float(gt_difference),margin=float(abs(reward_difference)),expert_sign=es,GT_sign=gs,GT_tie=gs==0,expert_tie=es==0,sign_agreement=float(es==gs),accuracy=None if gs==0 else (.5 if es==0 else float(es==gs)),decisive_accuracy=float(es==gs) if gs!=0 and es!=0 else None)
