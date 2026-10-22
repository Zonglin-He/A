"""Frozen, output-only event readout; no labels, fitted parameters or model imports."""
import numpy as np

def event_words(tokens):
    """Fixed dependency rule. Lexical verbs plus their nonsubject arguments.

    All lexical verbs seed the group (including descriptive participles). If
    absent, accept a copular predicate. No empirical action-word whitelist.
    """
    by={int(t['id']):t for t in tokens}
    seeds={i for i,t in by.items() if t['upos']=='VERB'}
    if not seeds:
        seeds={int(t['head']) for t in tokens if t['deprel']=='cop' and int(t['head']) in by}
    selected=set(seeds);blocked={'nsubj','csubj','punct','discourse','vocative'}
    # Full nonsubject predicate subtrees preserve interacting objects, negation,
    # prepositions, auxiliaries and multiword relations; subject appearance is
    # not added merely because it is the described target.
    change=True
    while change:
        change=False
        for i,t in by.items():
            if int(t['head']) in selected and t['deprel'].split(':')[0] not in blocked and i not in selected:
                selected.add(i);change=True
    return [by[i] for i in sorted(selected) if by[i]['upos'] not in {'PUNCT','SYM'}]

def word_tokens(caption,tokens,offsets):
    """Map cached lowercase original character positions to normalized input."""
    source=caption.strip().lower();chars=[];mapping={};space=False
    for i,c in enumerate(source):
        if c.isspace():
            if not space and chars:chars.append(' ')
            space=True
        else:
            mapping[i]=len(chars);chars.append(c);space=False
    normalized=''.join(chars).strip();assert normalized==' '.join(caption.lower().split())
    groups=[]
    for t in tokens:
        lo,hi=int(t['start']),int(t['end'])
        if source[lo:hi]!=t['text'].lower():return [],'cached_character_span_mismatch'
        expected=[mapping[j] for j in range(lo,hi) if j in mapping]
        if not expected:continue
        a,b=min(expected),max(expected)+1
        ids=[i for i,(x,y) in enumerate(offsets) if y>x and x<b and y>a]
        covered={j for i in ids for j in range(max(a,offsets[i][0]),min(b,offsets[i][1]))}
        if not ids or any(j not in covered for j in expected):return [],'event_word_truncated_or_uncovered'
        groups.append(ids)
    return groups,'valid' if groups else 'no_event_predicate'

def semantic_curve(premax,groups,query):
    x=np.asarray(premax,float)
    return np.stack([x[:,query,g].mean(-1) for g in groups],-1).mean(-1) if groups else None

def associate(student,teacher):
    s=np.asarray(student,float)[:,None,:];b=np.asarray(teacher,float)
    lo=np.maximum(s[...,:2]-s[...,2:]/2,b[...,:2]-b[...,2:]/2)
    hi=np.minimum(s[...,:2]+s[...,2:]/2,b[...,:2]+b[...,2:]/2)
    inter=np.maximum(hi-lo,0).prod(-1);union=s[...,2:].prod(-1)+b[...,2:].prod(-1)-inter
    iou=inter/np.maximum(union,1e-300);avg=iou.mean(0);j=int(np.argmax(avg))
    return j,dict(mean_IoU=float(avg[j]),frame_IoU=iou[:,j].tolist(),all_query_mean_IoU=avg.tolist(),no_geometric_overlap=bool(avg.max()==0))

def candidates(ids,positions,frame_count,initial):
    times=np.asarray(ids,float)[positions]+.5
    edges=np.array([0]+[int(np.floor((a+b)/2+.5)) for a,b in zip(times[:-1],times[1:])]+[int(frame_count)])
    assert np.all(np.diff(edges)>0) and len(edges)==9
    cs={tuple(map(int,initial)),(0,int(frame_count))}
    for i in range(8):
        for j in range(i+1,9):cs.add((int(edges[i]),int(edges[j])))
    return edges,[list(v)for v in sorted(cs)]

def interval_scores(curve,edges,cs):
    if curve is None:return [None]*len(cs)
    x=np.asarray(curve,float);assert x.shape==(8,) and np.isfinite(x).all()
    widths=np.diff(edges);total=float(widths.sum());out=[]
    for a,b in cs:
        w=np.maximum(0,np.minimum(edges[1:],b)-np.maximum(edges[:-1],a));inside=w.sum();outside=total-inside
        out.append(None if inside<=0 or outside<=0 else float(np.dot(w,x)/inside-np.dot(widths-w,x)/outside))
    return out

def choose(curve,edges,cs,initial):
    scores=interval_scores(curve,edges,cs);valid=[i for i,x in enumerate(scores)if x is not None]
    if not valid or np.ptp(curve)<=1e-12:return list(initial),scores,'no_event_or_constant_response'
    best=max(scores[i]for i in valid);ties=[i for i in valid if abs(scores[i]-best)<=1e-12]
    idx=min(ties,key=lambda i:(abs(cs[i][0]-initial[0])+abs(cs[i][1]-initial[1]),cs[i][0],cs[i][1]))
    return cs[idx],scores,'contrast_argmax'

