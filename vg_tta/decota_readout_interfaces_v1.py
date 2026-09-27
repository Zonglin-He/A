"""Label-free experimental readouts. No data/label loader or production mutation."""
import copy
import numpy as np
from scipy.special import logit


def signed_select(candidates, probability, w, c=0.):
    if not candidates:
        raise ValueError('Caller must provide its original default B')
    fallback={**candidates[0], 'selected':0, 'scores':None, 'reason':'invalid_keep_B'}
    try:
        p=np.asarray(probability,float); w=np.asarray(w,float)
        if p.ndim!=1 or p.shape!=w.shape or not len(p): return fallback
        if not np.isfinite(p).all() or not np.isfinite(w).all() or not np.isfinite(c): return fallback
        if (p<0).any() or (p>1).any() or (w<=0).any(): return fallback
        masks=[]
        for candidate in candidates:
            a,b=candidate['indices']
            if not (int(a)==a and int(b)==b and 0<=a<=b<len(p)): return fallback
            mask=np.zeros(len(p),float);mask[int(a):int(b)+1]=1;masks.append(mask)
        evidence=w*(logit(np.clip(p,1e-6,1-1e-6))+float(c))
        scores=[float((m-masks[0])@evidence) for m in masks]
        if not np.isfinite(scores).all(): return fallback
        chosen=0
        for j in range(1,len(scores)):
            if scores[j]>scores[chosen]+1e-12:chosen=j
        return {**candidates[chosen], 'selected':chosen, 'scores':scores,
                'reason':'signed_improved' if chosen else 'keep_B'}
    except (TypeError,ValueError,KeyError,OverflowError):
        return fallback


def target_tokens(text, offsets, span):
    """Explicit character occurrence, not a bag of all equal noun tokens."""
    a,b=span
    if not 0<=a<b<=len(text):raise ValueError('Invalid character span')
    selected=[i for i,(lo,hi) in enumerate(offsets) if hi>lo and lo<b and hi>a]
    coverage={j for i in selected for j in range(max(a,offsets[i][0]),min(b,offsets[i][1]))}
    if not selected or any(j not in coverage for j in range(a,b) if not text[j].isspace()):
        raise ValueError('Target span truncated or uncovered')
    return selected


def target_gate(probe, scores, threshold_only=False):
    """Change score source, never candidate order/reference/ROI/NMS or thresholds."""
    z=copy.copy(probe);s=np.asarray(scores,float)
    if s.shape!=(len(probe['boxes']),) or not np.isfinite(s).all():
        raise ValueError('Invalid target candidate scores')
    margin=float(probe['margin']) if threshold_only else float(s[0]-(s[1] if len(s)>1 else 0.)) if len(s) else 0.
    if not len(s): reason='no_candidates'
    elif s[0]<.35:reason='low_target_score'
    elif margin<.05:reason='ambiguous_distinct_instances'
    else:reason='accepted'
    z.update(accepted=reason=='accepted',reason=reason,margin=margin,target_scores=s)
    return z
