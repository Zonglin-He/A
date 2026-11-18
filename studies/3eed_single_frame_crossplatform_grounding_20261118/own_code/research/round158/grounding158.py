"""Text-to-box selection for the frozen scene-grounding pipeline.

Pass the full sentence prefix, anchor geometry and role route from the original
3EED front end. When a physical B exists, pass its registered image graph and
the registered joint point prefix on the original A anchors.
"""
import sys
from pathlib import Path
import numpy as np
from reader158 import overlaps, probability


def select_grounding_box(boxes, sentence_probability):
    boxes=np.asarray(boxes,dtype=np.float64)
    p=np.asarray(sentence_probability,dtype=np.float64)
    if boxes.shape!=(256,6) or p.shape!=(256,) or not np.isfinite(boxes).all() or not np.isfinite(p).all() or (p<0).any() or not np.isclose(p.sum(),1.,rtol=1e-6):
        raise ValueError('expected 256 physical A boxes and a normalized full-sentence probability')
    kernel=overlaps(boxes)
    adjusted=probability(np.log(np.maximum(p,1e-300))-np.log((kernel*kernel).sum(1)))
    utility=kernel@adjusted
    return int(np.argmax(utility)),utility


def ground_frame(base,matcher,boxes,anchor_prefix,graph_A,role_route,observation,*,
                 registered_prefix=None,graph_B=None):
    """Return selected A index, original box, decision scores and sentence probability."""
    root=Path(__file__).resolve().parents[2]
    sys.path.insert(0,str(root/'research/round141'))
    from core141 import read_bank141
    import torch
    if (registered_prefix is None)!=(graph_B is None):
        raise ValueError('registered geometry and camera evidence must arrive together')
    z=anchor_prefix; graphs=[graph_A]
    if graph_B is not None:
        for key in ('xyz','query_points_sample_inds'):
            if not torch.equal(anchor_prefix[key],registered_prefix[key]):
                raise ValueError('B evidence is not aligned to A')
        z=registered_prefix; graphs.append(graph_B)
    with torch.inference_mode():
        likelihood,_,_=read_bank141(base,matcher,z,graphs,role_route,observation)
    idx,utility=select_grounding_box(boxes,likelihood)
    return idx,np.asarray(boxes)[idx].copy(),utility,likelihood
