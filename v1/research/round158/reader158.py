import numpy as np
from scipy.special import logsumexp

def overlaps(boxes):
    side=np.maximum(0,boxes[:,3:].astype(np.float64))
    lo=boxes[:,:3]-side/2
    hi=boxes[:,:3]+side/2
    inter=np.prod(np.maximum(0,np.minimum(hi[:,None],hi[None])-np.maximum(lo[:,None],lo[None])),axis=2)
    union=side.prod(1)[:,None]+side.prod(1)[None]-inter
    similarity=np.divide(inter,union,out=np.zeros_like(inter),where=union>0)
    # A degenerate candidate is a distinct proposal, with no support from other boxes.
    np.fill_diagonal(similarity,1)
    return similarity

def probability(scores):
    return np.exp(scores-logsumexp(scores))
