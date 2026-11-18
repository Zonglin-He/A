"""Joint inference over all registered candidates; reusable without task labels."""
import hashlib
import re

import numpy as np
from scipy.special import logsumexp

ARMS=('frozen_A','joint_zero','R145','role_risk','same_posterior_MAP','old_scores_risk',
      'new_A_only','without_reference','without_layout','without_phrase','misbound_reference',
      'misbound_B','inverted_layout')


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


def location_intervals(caption, target_span):
    text=caption.lower()
    field=re.search(r'\b(?:position information|target (?:position|location))\s*:\s*([^;.]+)',text)
    if field:
        clause=field[1]
    elif target_span:
        # Only a camera-relative phrase explicitly attached to the current subject.
        tail=text[target_span[1]:]
        found=re.match(r'^[\s,]*(?:(?:is|was|stands?|standing|sits?|sitting|located|positioned|appears?|visible|parked)\s+){0,3}(?:at|near|on|in|to)\s+(?:the\s+)?([^;.]{1,65}?)\s+(?:of|in)\s+(?:the\s+)?(?:image|picture|frame|view|screen|photo)\b',tail)
        if not found:
            return []
        clause=found[1]
    else:
        return []
    # Orientations of a car/person and relations to named objects are separate questions.
    clause=re.split(r'\b(?:facing|heading|moving|oriented|turned|next to|beside|behind|in front of)\b|,|;',clause)[0]
    clause=re.sub(r'\b(?:left|right) (?:side|wheel|door|shoulder) (?:is|of (?!the (?:view|image|frame|picture)))\b.*','',clause)
    if re.search(r'\b(?:left|right|top|bottom) of (?!the (?:image|view|frame|picture|photo|screen|center|centre))',clause):
        return []
    if not clause.strip():
        return []
    out=[]
    for axis,low,high,center in [(0,'left','right','horizontally'),(1,'upper|top','lower|bottom','vertically')]:
        l=bool(re.search(r'\b(?:'+low+r')\b',clause)); h=bool(re.search(r'\b(?:'+high+r')\b',clause))
        if l!=h:
            slight='slightly' in clause or 'just' in clause
            out.append((axis, (1/3 if slight else 0) if l else .5, .5 if l else (2/3 if slight else 1)))
        elif not l and (re.search(r'\b(?:center|centre|middle)\b',clause) and
                       (axis==0 or center in clause or not re.search(r'\b(?:left|right)\b',clause))):
            out.append((axis,1/3,2/3))
    return out


def unary_layout(rect,visible,bands):
    score=np.zeros(len(rect))
    for axis,low,high in bands:
        l,r=rect[:,0,axis],rect[:,1,axis]
        w=r-l
        frac=np.divide(np.maximum(0,np.minimum(high,r)-np.maximum(low,l)),w,
                       out=((l>=low)&(l<=high)).astype(float),where=w>1e-8)
        score+=np.where(visible,np.log(.5+.5*frac/(high-low)),0.)
    return score


def order_effect(rect,seen,overlap):
    """P(u_target < u_reference) for independent uniform horizontal extents."""
    l=rect[:,0,0];r=rect[:,1,0];width=np.maximum(r-l,1e-8)
    # Analytic integral of the uniform target CDF over the reference extent.
    def integral(t):
        return (np.maximum(0,t-l[:,None])**2-np.maximum(0,t-r[:,None])**2)/(2*width[:,None])
    prob=np.clip((integral(r[None,:])-integral(l[None,:]))/width[None,:],0,1)
    usable=seen & ((r-l)>1e-8)
    return (2*prob-1)*usable[:,None]*usable[None,:]*(1-overlap)


def probability(scores):
    return np.exp(scores-logsumexp(scores))


def normalized_rows(x):
    return x-logsumexp(x,axis=1,keepdims=True)


def derangement(key,n):
    rng=np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'little'))
    order=rng.permutation(n); perm=np.arange(n)
    if n>1:
        perm[order]=np.roll(order,1)
    return perm


def select_from_description(evidence,caption,target_span,old_scores=None):
    """All outputs are slot indices in the original A256. Historical controls are isolated."""
    a,b=normalized_rows(evidence['A']),normalized_rows(evidence['B'])
    cov=evidence['coverage']
    if a.shape!=b.shape or a.shape[1]!=256 or np.any((cov<0)|(cov>1)):
        raise ValueError('candidate or reliability mismatch')
    boxes=evidence['boxes']; geom=overlaps(boxes)
    density=np.sum(geom*geom,axis=1)
    correction=np.log(density)
    bands=location_intervals(caption,target_span)
    rect,vis=evidence['rect'],evidence['seen']
    layout=unary_layout(rect,vis,bands)
    flipped=unary_layout(rect,vis,[(axis,1-hi,1-lo) for axis,lo,hi in bands])
    signs=evidence['signs']
    order=order_effect(rect,vis,geom) if len(signs) else None
    # The two controls relocate evidence inside the current observation, independent of GT.
    key=str(evidence['box_sha'])
    wrong_ref=derangement(key+'ref',256)
    bslots=np.flatnonzero(cov>0)
    bwrong=b.copy()
    if len(bslots)>1:
        bwrong[:,bslots]=b[:,bslots[derangement(key+'bank',len(bslots))]]
    def posterior(view,layout_score,use_ref=True,use_phrase=True,permute_ref=False):
        target=view[0] if len(view)<2 or not use_phrase else .5*(view[0]+view[1])
        score=target-correction+layout_score
        if use_ref and len(signs):
            evidence_terms=[]
            for sign,role in zip(signs,view[2:]):
                prior=probability((role[wrong_ref] if permute_ref else role)-correction)
                effect=order@prior
                evidence_terms.append(np.log1p(.5*effect*(-sign)))
            score=score+np.mean(evidence_terms,axis=0)
        return probability(score)
    def merge(other):
        return a+(other-a)*(.5*cov)[None,:]
    def pick(post):
        return int(np.argmax(geom@post))
    v=merge(b); post=posterior(v,layout)
    if old_scores is None:
        return pick(post)
    pred={k:int(np.argmax(old_scores[k])) for k in ('frozen_A','joint_zero','R145')}
    pred.update(role_risk=pick(post),same_posterior_MAP=int(np.argmax(post)),
                old_scores_risk=pick(probability(np.log(np.maximum(old_scores['joint_zero'],1e-300))-correction)),
                new_A_only=pick(posterior(a,layout)),without_reference=pick(posterior(v,layout,use_ref=False)),
                without_layout=pick(posterior(v,layout*0)),without_phrase=pick(posterior(v,layout,use_phrase=False)),
                misbound_reference=pick(posterior(v,layout,permute_ref=True)),
                misbound_B=pick(posterior(merge(bwrong),layout)),inverted_layout=pick(posterior(v,flipped)))
    if not cov.any() and not (pred['role_risk']==pred['new_A_only']==pred['misbound_B']):
        raise ValueError('zero B evidence changed reader')
    return pred,{'subject_phrase':len(a)>1,'reference_edges':len(signs),'layout_axes':len(bands),
                 'visible_boxes':int(vis.sum()),'covered_B_boxes':int((cov>0).sum()),
                 'current_old_new_overlap':float(geom[pred['joint_zero'],pred['role_risk']])}
