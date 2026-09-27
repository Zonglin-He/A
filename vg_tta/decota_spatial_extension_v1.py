"""F21 label-free query interface; exact cached-score feasibility helpers."""
from fractions import Fraction
import math
import numpy as np


def parse_context(parser, query):
    q=query.strip().lower()
    doc=parser.nlp(q)
    audit=[dict(id=w.id,text=w.text,lemma=w.lemma,upos=w.upos,head=w.head,
                deprel=w.deprel,start=w.start_char,end=w.end_char)
           for s in doc.sentences for w in s.words]
    fallback=lambda reason: dict(eligible=False,reason=reason,context=None,span=None,tokens=audit)
    if len(doc.sentences)!=1:return fallback('multiple_sentences')
    words=doc.sentences[0].words
    if not words:return fallback('empty_query')
    by={w.id:w for w in words};root=next((w for w in words if w.head==0),None)
    if root is None:return fallback('no_root')
    first=words[0];wh=first.lemma.lower() in {'who','whom','which','what'}
    target=None;rule=None
    if wh:
        if first.lemma.lower() in {'what','which'} and first.deprel=='det':
            target=by.get(first.head);rule='explicit_wh_nominal'
            if target is None or target.upos not in {'NOUN','PROPN'}:return fallback('unresolved_wh_category')
        elif first.lemma.lower() in {'who','whom'}:
            subjects=[w for w in words if w.head==root.id and w.deprel.startswith('nsubj')]
            if first not in subjects or len(subjects)!=1:return fallback('unresolved_wh_role_keep_S1')
            q='person who'+q[first.end_char:];rule='wh_subject_to_person_who'
            context=q.rstrip('.?!').rstrip()+'.'
            return dict(eligible=True,reason=rule,context=context,span=[0,6],entity='person',tokens=audit)
        else:return fallback('unresolved_wh_category')
    else:
        subjects=[w for w in words if w.head==root.id and w.deprel.startswith('nsubj') and w.upos in {'NOUN','PROPN'}]
        if len(subjects)==1:target=subjects[0];rule='unique_root_nominal_subject'
        elif root.upos in {'NOUN','PROPN'} and first.lemma.lower() in {'a','an','the'} and first.deprel=='det' and first.head==root.id:
            if any(w.deprel=='cc' and w.head in {v.id for v in words if v.deprel=='conj' and v.head==root.id} for w in words):
                return fallback('coordinated_nominal_fragment')
            target=root;rule='initial_determined_nominal_root'
        else:return fallback('unresolved_declarative')
    # Ambiguous explicitly coordinated subjects are not assigned to just one noun.
    if any(w.deprel=='conj' and w.head==target.id and any(v.deprel=='cc' and v.head==w.id for v in words) for w in words):
        return fallback('coordinated_target')
    context=q.rstrip('.?!').rstrip()+'.';span=[target.start_char,target.end_char]
    assert context[span[0]:span[1]]==target.text.lower()
    return dict(eligible=True,reason=rule,context=context,span=span,entity=context[slice(*span)],tokens=audit)


def pool_gate(probe):
    s=np.asarray(probe['target_scores'],float)
    if not np.isfinite(s).all() or len(s)!=len(probe['boxes']):raise ValueError('Invalid scores')
    order=np.argsort(-s,kind='stable');best=int(order[0]) if len(s) else None
    margin=float(s[best]-(s[order[1]] if len(s)>1 else 0)) if len(s) else 0.
    reason='no_candidates' if not len(s) else 'low_target_score' if s[best]<.35 else 'ambiguous_distinct_instances' if margin<.05 else 'accepted'
    return dict(**{k:v for k,v in probe.items() if k not in {'accepted','reason','margin','unary'}},
        unary=probe['target_scores'],accepted=reason=='accepted',reason=reason,margin=margin,admission_winner=best)


def path_score(probes,path,weight):
    unary=sum(math.log(max(float(probes[i]['unary'][j]),1e-30)/.35) for i,j in path)
    edges=[]
    for (i,j),(ii,jj) in zip(path[:-1],path[1:]):
        f=np.asarray(probes[i]['roi'][j],float);g=np.asarray(probes[ii]['roi'][jj],float)
        cos=float(np.clip(f@g/max(np.linalg.norm(f)*np.linalg.norm(g),1e-24),-1,1))
        edges.append(math.log(max((1+cos)/2,1e-12)))
    return dict(unary=float(unary),edges=edges,edge_sum=float(sum(edges)),total=float(unary+weight*sum(edges)))


# Intervals use rational endpoints so threshold ties are not rounded away.
# None denotes unbounded; tuple(lo,hi,left_closed,right_closed).
def intersect(a,b):
    lo=a[0];lc=a[2]
    if lo is None or b[0] is not None and b[0]>lo:lo,lc=b[0],b[2]
    elif b[0] is not None and b[0]==lo:lc=lc and b[2]
    hi=a[1];hc=a[3]
    if hi is None or b[1] is not None and b[1]<hi:hi,hc=b[1],b[3]
    elif b[1] is not None and b[1]==hi:hc=hc and b[3]
    if lo is not None and hi is not None and (lo>hi or lo==hi and not (lc and hc)):return None
    return (lo,hi,lc,hc)


def affine_winning_sets(intercepts,slopes,tolerance=1e-12):
    """Exact decision tree for sequential >tol replacement, not naive argmax.

    Returns a possibly disconnected winning set for every candidate. Zero tolerance
    is the mathematical upper envelope with first-index ties. Positive tolerance
    implements the actual B-first production comparator, including nontransitivity.
    """
    A=list(map(lambda x:Fraction(float(x)),intercepts));L=list(map(lambda x:Fraction(float(x)),slopes));eps=Fraction(float(tolerance))
    if len(A)!=len(L) or not A:raise ValueError('Empty or inconsistent lines')
    states=[(0,(None,None,False,False))]
    for j in range(1,len(A)):
        new=[]
        for k,iv in states:
            da=A[j]-A[k]-eps;ds=L[j]-L[k]
            if ds==0:new.append((j if da>0 else k,iv));continue
            boundary=-da/ds
            yes=(boundary,None,False,False) if ds>0 else (None,boundary,False,False)
            no=(None,boundary,False,True) if ds>0 else (boundary,None,True,False)
            for win,part in [(j,yes),(k,no)]:
                hit=intersect(iv,part)
                if hit is not None:new.append((win,hit))
        states=new
    return {k:[iv for j,iv in states if j==k] for k in range(len(A))}


def contains(iv,c):
    c=Fraction(float(c)) if not isinstance(c,Fraction) else c
    a,b,lc,hc=iv
    return (a is None or c>a or c==a and lc) and (b is None or c<b or c==b and hc)


def intersect_unions(xs,ys):
    return [z for x in xs for y in ys if (z:=intersect(x,y)) is not None]


def interval_json(iv):
    a,b,lc,hc=iv
    return dict(lo=float(a) if a is not None else None,hi=float(b) if b is not None else None,
        lo_exact=str(a) if a is not None else None,hi_exact=str(b) if b is not None else None,
        lo_closed=lc,hi_closed=hc)


def witness(iv):
    a,b,lc,hc=iv
    if a is None and b is None:return Fraction(0)
    if a is None:return b-max(Fraction(1),abs(b))
    if b is None:return a+max(Fraction(1),abs(a))
    return (a+b)/2
