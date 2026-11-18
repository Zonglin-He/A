"""Replacement decision using two full-sentence likelihoods over the original A slots."""
import numpy as np
from reader158 import derangement,location_intervals,normalized_rows,overlaps,probability,unary_layout

NAMES=('context_overlap','context_A_only','context_no_layout','context_wrong_layout',
       'context_competition_only','context_native_only','context_wrong_B','context_argmax','uniform_overlap')


def select_contextual_winner(boxes, caption, target_span, rect, visible, full_sentence_competition,
                             target_logits_A, target_logits_B=None, coverage_B=None, *,
                             diagnostics=False, competition_A=None, competition_wrong_B=None):
    """Choose A256 index from frozen distributions; a missing B contributes no evidence."""
    p=np.asarray(full_sentence_competition,dtype=np.float64)
    boxes=np.asarray(boxes)
    if p.shape!=(256,) or not np.isfinite(p).all() or np.any(p<0) or abs(p.sum()-1)>1e-5:
        raise ValueError('expected a normalized probability on A256')
    a=normalized_rows(np.asarray(target_logits_A)[None,:])[0]
    if a.shape!=(256,) or not np.isfinite(a).all():
        raise ValueError('invalid A logits')
    b=a; w=np.zeros(256)
    if target_logits_B is not None:
        b=normalized_rows(np.asarray(target_logits_B)[None,:])[0]
        support=np.asarray(coverage_B,dtype=float)
        if b.shape!=a.shape or not np.isfinite(b).all() or support.shape!=a.shape or not np.isfinite(support).all() or np.any((support<0)|(support>1)):
            raise ValueError('invalid registered support')
        w=support/2
    trained=np.log(np.maximum(p,1e-300)); native=a+(b-a)*w
    geom=overlaps(boxes); density=np.log((geom*geom).sum(1))
    bands=location_intervals(caption,target_span)
    loc=unary_layout(rect,visible,bands)
    def choose(sentence,layout):
        posterior=probability(sentence+layout-density)
        return int(np.argmax(geom@posterior))
    joint=(trained+native)/2
    decision=choose(joint,loc)
    if not diagnostics:
        return decision
    if competition_A is None or competition_wrong_B is None:
        raise ValueError('diagnostics require their matched frozen readouts')
    trained_A=np.log(np.maximum(competition_A,1e-300))
    wrong_trained=np.log(np.maximum(competition_wrong_B,1e-300))
    bs=np.flatnonzero(w>0);wrong=b.copy()
    if len(bs)>1:
        import hashlib
        key=hashlib.sha256(boxes.tobytes()).hexdigest()
        wrong[bs]=b[bs[derangement(key+'bank',len(bs))]]
    wrong_loc=unary_layout(rect,visible,[(s,1-hi,1-lo) for s,lo,hi in bands])
    return dict(context_overlap=decision,context_A_only=choose((trained_A+a)/2,loc),
                context_no_layout=choose(joint,0),context_wrong_layout=choose(joint,wrong_loc),
                context_competition_only=choose(trained,loc),context_native_only=choose(native,loc),
                context_wrong_B=choose((wrong_trained+a+(wrong-a)*w)/2,loc),
                context_argmax=int(np.argmax(joint+loc)),uniform_overlap=choose(np.zeros(256),0))
