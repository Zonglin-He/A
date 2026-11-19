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
