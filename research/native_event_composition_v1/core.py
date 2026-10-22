"""Query-only same-subject conjunctions and exact independent interval envelope."""
import re
import numpy as np


def from_words(words,caption):
    by={w['id']:w for w in words};children={i:[]for i in by}
    for w in words:
        if w['head']in children:children[w['head']].append(w['id'])
    roots=[w for w in words if w['head']==0 and w['upos']=='VERB']
    if len(roots)!=1:return dict(supported=False,reason='no_single_main_verb',clauses=[caption])
    root=roots[0]['id'];verbs={root}
    for _ in words:
        more={w['id']for w in words if w['head']in verbs and w['rel']=='conj'and w['upos']=='VERB'}
        if more<=verbs:break
        verbs|=more
    if len(verbs)<2:return dict(supported=False,reason='no_main_verbal_conjunction',clauses=[caption])
    # This bounded branch handles plain AND composition only, not logical alternatives
    # or a main event modified by before/after/while. No quality/GT gate is used.
    if any(w['lemma'].lower()in {'before','after','until','while','unless','when','or','but'}for w in words):return dict(supported=False,reason='unsupported_temporal_or_logical_relation',clauses=[caption])
    subjects=[i for i in children[root]if by[i]['rel']=='nsubj']
    if len(subjects)!=1:return dict(supported=False,reason='missing_unique_subject',clauses=[caption])
    subject=subjects[0]
    for v in verbs-{root}:
        own=[i for i in children[v]if by[i]['rel']=='nsubj']
        if own:return dict(supported=False,reason='explicit_new_subject_not_resolved',clauses=[caption])
    def subtree(i,exclude):
        if i in exclude:return set()
        out={i}
        for ch in children[i]:out|=subtree(ch,exclude)
        return out
    subj=subtree(subject,verbs);clauses=[]
    for v in sorted(verbs):
        keep=subtree(v,verbs-{v})|subj
        keep={i for i in keep if by[i]['rel']not in {'cc','punct'}and not(by[i]['rel']=='nsubj'and i!=subject)}
        text=' '.join(by[i]['text']for i in sorted(keep));text=re.sub(r"\s+([,.;:!?])",r'\1',text).strip(' ,.;:!?')+'.'
        clauses.append(text)
    return dict(supported=True,reason='same_subject_main_AND',clauses=clauses,first=clauses[0],last=clauses[-1],main_verbs=sorted(verbs))


def decompose(caption,parser):
    doc=parser.nlp(caption)
    if len(doc.sentences)!=1:return dict(supported=False,reason='multiple_sentences',clauses=[caption])
    words=[dict(id=w.id,head=w.head,upos=w.upos,rel=w.deprel.split(':')[0],lemma=w.lemma,text=w.text)for w in doc.sentences[0].words]
    return from_words(words,caption)


def envelope(q,r):
    """Distribution of [min(start1,start2),max(end1,end2)] in O(n²).

    C(a,b)=P(start>=a,end<=b); hull contained in[a,b] iff BOTH intervals
    contained. Invert the two-dimensional containment CDF; clipping only roundoff.
    """
    q=np.asarray(q,dtype=np.float64);r=np.asarray(r,dtype=np.float64);assert q.shape==r.shape and q.ndim==2 and q.shape[0]==q.shape[1]
    n=len(q);legal=np.triu(np.ones((n,n),bool),1)
    for x in [q,r]:assert (x>=0).all()and not x[~legal].any()and abs(x.sum()-1)<1e-10
    c=(q[::-1].cumsum(0)[::-1].cumsum(1))*(r[::-1].cumsum(0)[::-1].cumsum(1));pad=np.pad(c,((0,1),(1,0)));out=pad[:n,1:]-pad[1:,1:]-pad[:n,:n]+pad[1:,:n]
    assert out.min()>-1e-12 and abs(out.sum()-1)<1e-10 and abs(out[~legal]).max()<1e-12
    out[~legal]=0;out=np.maximum(out,0);out/=out.sum();return out
