"""Equal-length real textual interventions and fixed posterior feedback."""
import numpy as np

def masked_inputs(input_ids,positions,mask_id):
    result=input_ids.clone();result[0,list(positions)]=int(mask_id);return result

def normalize(q,score):
    take=q>0;v=np.log(q[take])+np.asarray(score)[take];v=np.exp(v-v.max());v/=v.sum();out=np.zeros_like(q);out[take]=v;return out

def posteriors(q,event,control,identity):
    take=q>0
    def log(x):
        assert x.shape==q.shape and np.array_equal(x>0,take);v=np.zeros_like(q);v[take]=np.log(x[take]);return v
    l=log(q);e=log(event);c=log(control);i=log(identity)
    score=c-e # Difference of full/event-mask and full/control-mask responses.
    reverse=score[::-1,::-1].T.copy() # Reverse temporal placement, same index duration.
    # All have the same original posterior and MBR. No GT, acceptance gate or sweep.
    return dict(NativeG6=q,Selective=normalize(q,score),EventOnly=normalize(q,l-e),ControlOnly=normalize(q,l-c),IdentityContrast=normalize(q,l-i),Sharpen=normalize(q,l),ReverseSelective=normalize(q,reverse)),score

def intervention(caption,subject,parser,tokenizer):
    doc=parser.nlp(caption);words=[]
    for sentence in doc.sentences:
        for token in sentence.tokens:
            for w in token.words:words.append(dict(id=w.id,head=w.head,upos=w.upos,rel=w.deprel.split(':')[0],start=token.start_char,end=token.end_char))
    predicate={w['id']for w in words if w['head']==0 and w['upos']=='VERB'}
    for _ in words:
        more={w['id']for w in words if w['upos']=='VERB'and w['head']in predicate and w['rel']in {'conj','xcomp','ccomp','advcl'}}
        if more<=predicate:break
        predicate|=more
    nouns=[w for w in words if w['rel']=='nsubj'and w['upos']in {'NOUN','PROPN'}];noun=nouns[0]if nouns else next((w for w in words if w['upos']in {'NOUN','PROPN'}),None)
    protect={noun['id']}if noun else set()
    for _ in words:
        more={w['id']for w in words if w['head']in protect and w['id']not in predicate}
        if more<=protect:break
        protect|=more
    text=subject+' '+caption;shift=len(subject)+1;tok=tokenizer([text],return_offsets_mapping=True,return_tensors='pt');offset=tok['offset_mapping'][0].tolist()
    def indices(wanted):
        spans=[(w['start']+shift,w['end']+shift)for w in words if w['id']in wanted]
        return sorted({j for j,(a,b)in enumerate(offset)if a<b and any(a<end and b>start for start,end in spans)})
    event=indices(predicate);protected=set(indices(protect))|{j for j,(a,b)in enumerate(offset)if a<b and a<shift}
    function={w['id']for w in words if w['upos']in {'DET','ADP','CCONJ','SCONJ','PART','PUNCT','AUX'}and w['id']not in protect and w['id']not in predicate}
    control=[j for j in indices(function)if j not in protected and j not in event][:len(event)]
    valid=bool(event and len(control)==len(event)and not protected.intersection(event))
    spans=[(w['start'],w['end'])for w in words if w['id']in protect]
    if spans:lo=min(a for a,b in spans);hi=max(b for a,b in spans);identity=caption[lo:hi].strip(' .')+'.'
    else:identity=caption
    return dict(event_positions=event,control_positions=control,protected_positions=sorted(protected),supported=valid,identity_caption=identity,full_input_ids=tok['input_ids'],word_count=len(words),event_words=len(predicate),subject=subject,caption=caption)
