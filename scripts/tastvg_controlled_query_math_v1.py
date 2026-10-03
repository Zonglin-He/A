"""Text-only natural-query matching; fixed probe readouts, no fitting."""
import hashlib
import numpy as np
from scripts.tastvg_information_atlas_math_v1 import VIEWS, predict


def norm(s):
    return ' '.join(s.casefold().split())


def text_hash(s):
    return hashlib.sha256(s.encode()).hexdigest()


def query_form(s):
    return 'question' if '?' in s or norm(s).split()[0] in {'who','whom','what','which','where','when','why','how'} else 'declarative'


def native_subject(words):
    """Same query-only selection as QuerySubjectParser, including unknown what."""
    for w in words:
        if w['lemma'] in {'which','what'} and w['deprel']=='det':
            head=next((v for v in words if v['local_id']==w['local_head']),None)
            if head:return head['lemma']
    if any(w['lemma'] in {'who','whom'} for w in words):return 'person'
    if any(w['lemma']=='what' for w in words):return ''
    w=next((w for w in words if w['deprel'].startswith('nsubj') and w['upos'] in {'NOUN','PROPN'}),None)
    if w is None:w=next((w for w in words if w['upos'] in {'NOUN','PROPN'}),None)
    return w['lemma'] if w else ''


def linguistic_signature(words):
    subject=native_subject(words)
    nominal=next((w for w in words if w['deprel'].startswith('nsubj') and w['upos'] in {'NOUN','PROPN'}),None)
    descriptors=[]
    if nominal:
        kept={nominal['id']}
        # Only nominal modifiers, never actions/relative clauses or their objects.
        for _ in words:
            added={w['id'] for w in words if w['head'] in kept and
                   w['deprel'].split(':')[0] in {'amod','nmod','compound','case','det','nummod'} and
                   w['upos'] not in {'VERB','AUX'} and w['lemma'] not in {'a','the','an','this','that'}}
            if added<=kept:break
            kept|=added
        descriptors=[w['lemma'] for w in words if w['id'] in kept and w['lemma'] not in {'a','the','an'}]
    roots=[w for w in words if w['deprel']=='root' and w['upos']=='VERB']
    main_ids={w['id'] for w in roots}
    for _ in words:
        added={w['id'] for w in words if w['head'] in main_ids and w['deprel']=='conj' and w['upos']=='VERB'}
        if added<=main_ids:break
        main_ids|=added
    actions=sorted({w['lemma'] for w in words if w['id'] in main_ids})
    root=roots[0]['lemma'] if roots else ''
    verb_ids={w['id'] for w in words if w['upos']=='VERB'}
    atoms=[]
    for w in words:
        if w['upos']=='VERB':atoms.append(('verb',w['lemma']))
        if w['head'] in verb_ids and w['deprel'] in {'obj','iobj','compound:prt'}:
            lemma='person' if w['lemma'] in {'man','woman','boy','girl','adult','person','child','baby','he','she','him','her','who','whom'} else w['lemma']
            atoms.append((w['deprel'],lemma))
        if w['head'] in verb_ids and w['deprel'] in {'nsubj:pass','aux:pass'}:atoms.append(('voice','passive'))
    identity_signature=tuple(sorted(set(atoms)))
    return dict(subject=subject,subject_signature=tuple(descriptors) if descriptors else (subject,),
                root_action=root,actions=tuple(actions),lexical_verbs=tuple(sorted({w['lemma'] for w in words if w['upos']=='VERB'})),
                identity_signature=identity_signature)


def pair_kind(recipient, donor, arm):
    """Return a prespecified quality tier; None means unavailable, never imputed."""
    if norm(recipient['caption'])==norm(donor['caption']) or recipient['form']!=donor['form']:return None
    a,b=recipient['signature'],donor['signature']
    if not a['root_action'] or not b['root_action']:return None
    same_media=recipient['video_sha256']==donor['video_sha256']
    same_segment=recipient.get('segment')==donor.get('segment')
    same_ref=(same_media and recipient.get('target_id') is not None and donor.get('target_id')==recipient['target_id'])
    diff_ref=(same_media and recipient.get('target_id') is not None and donor.get('target_id') is not None and donor['target_id']!=recipient['target_id'])
    disjoint=not(set(a['lexical_verbs'])&set(b['lexical_verbs']))
    if arm=='event':
        if not disjoint:return None
        if same_ref:return (0 if same_segment else 1,'same_video_same_referent_disjoint_actions')
        if same_media:return None
        if a['subject'] and a['subject']==b['subject']:
            exact=bool(a['subject_signature']) and tuple(a['subject_signature'])==tuple(b['subject_signature'])
            return (2 if exact else 3,'cross_video_subject_phrase_match' if exact else 'cross_video_subject_class_match')
    elif arm=='subject':
        exact=(tuple(tuple(v) if isinstance(v,list) else v for v in a.get('identity_signature',a['actions']))==
               tuple(tuple(v) if isinstance(v,list) else v for v in b.get('identity_signature',b['actions'])))
        if diff_ref and exact:return (0 if same_segment else 1,'same_video_different_referent_same_action_signature')
        if same_media:return None
        different=bool(a['subject'] and b['subject']) and a['subject']!=b['subject']
        if different and a['root_action']==b['root_action']:
            return (2 if exact else 3,'cross_video_same_action_signature' if exact else 'cross_video_same_root_action')
    else:raise ValueError(arm)
    return None


def choose_donor(recipient, donors, arm, dataset):
    valid=[]
    for donor in donors:
        kind=pair_kind(recipient,donor,arm)
        if kind is None:continue
        # Text length is a nuisance match, not a prediction/GT-dependent selector.
        length=abs(len(recipient['caption'].split())-len(donor['caption'].split()))
        key=text_hash(f'controlled-query-v1|{dataset}|{recipient["key"]}|{arm}|{donor["key"]}')
        valid.append(((kind[0],length,key),donor,kind))
    if not valid:return None
    _,donor,kind=min(valid,key=lambda x:x[0])
    return dict(donor=donor,tier=kind[0],kind=kind[1],eligible_donors=len(valid))


def frozen_readout(feature, models):
    out={}
    for view in ['Full','Contrast','Geometry']:
        for task in ['precision','recall','tiou']:
            name=f'candidate/{view}/{task}/real'
            x=feature['geometry'] if view=='Geometry' else feature['x'][:,VIEWS[view]]
            out[name]=predict(models[name],x)
    name='frame/Hidden/event/real';m=models[name];h=feature['hidden'].double().numpy()
    out[name]=predict(m,h);out[name+'/logit']=(h-m['mean'])/m['std']@m['weight']+m['bias']
    return out


def multi_difference(rows, terms, field='r2', draws=10000, seed=20261003):
    """A single paired resampling for a difference of specificity differences."""
    from scripts.tastvg_information_atlas_math_v1 import source_moments,metric_from_moments
    vectors=[source_moments(rows,name) for name,weight in terms]
    ids=sorted(set.intersection(*(set(x) for x in vectors)))
    if not ids:return dict(sources=0,mean=None,ci95=None,bootstrap_defined=0)
    n=len(ids);rng=np.random.default_rng(seed);w=rng.multinomial(n,np.ones(n)/n,size=draws)/n
    def ave(x,weights):
        mask=np.isfinite(x);den=weights@mask
        return np.divide(weights@np.nan_to_num(x),den,out=np.full(den.shape,np.nan),where=den>0)
    bs=np.zeros(draws);point=0.
    for vector,(_,coefficient) in zip(vectors,terms):
        a=np.asarray([vector[i] for i in ids]);bs+=coefficient*metric_from_moments(ave(a,w),field)
        point+=coefficient*float(metric_from_moments(ave(a,np.ones((1,n))/n),field)[0])
    good=np.isfinite(bs)
    return dict(sources=n,mean=point if np.isfinite(point) else None,
                ci95=np.quantile(bs[good],[.025,.975]).tolist() if good.any() else None,bootstrap_defined=int(good.sum()))
