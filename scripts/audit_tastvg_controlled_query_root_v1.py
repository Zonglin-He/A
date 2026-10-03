"""Private immutable-binding, linguistic control and independent readout audit."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='4'
os.environ['OMP_NUM_THREADS']='4'
import sys,time,json,hashlib,collections
from pathlib import Path
import numpy as np
import torch
from scipy.special import expit
from sklearn.metrics import roc_auc_score,average_precision_score
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_tastvg_controlled_query_v1 import BASE,PUB,OLD,ATLAS,PLAN,POOL,read,write,sha,checked,verify,prefix
from scripts import audit_tastvg_query_swap_root_v1 as independent
from methods.decota_final_simplified_v1.tensors import state_hash


def normalize(s):return ' '.join(s.lower().split())
def tuple_sig(s):return tuple(tuple(x) for x in s)
def kind(a,b,arm):
    if normalize(a['caption'])==normalize(b['caption']) or a['form']!=b['form']:return None
    x,y=a['signature'],b['signature']
    if not x['root_action'] or not y['root_action']:return None
    media=a['video_sha256']==b['video_sha256'];segment=a['segment']==b['segment']
    same_id=media and a['target_id'] is not None and a['target_id']==b['target_id']
    diff_id=media and a['target_id'] is not None and b['target_id'] is not None and a['target_id']!=b['target_id']
    exact=tuple_sig(x['identity_signature'])==tuple_sig(y['identity_signature'])
    if arm=='event':
        if set(x['lexical_verbs'])&set(y['lexical_verbs']):return None
        if same_id:return (int(not segment),'same_video_same_referent_disjoint_actions')
        if media:return None
        if x['subject'] and x['subject']==y['subject']:
            phrase=bool(x['subject_signature']) and x['subject_signature']==y['subject_signature']
            return (2 if phrase else 3,'cross_video_subject_phrase_match' if phrase else 'cross_video_subject_class_match')
    else:
        if diff_id and exact:return (int(not segment),'same_video_different_referent_same_action_signature')
        if media:return None
        if x['subject'] and y['subject'] and x['subject']!=y['subject'] and x['root_action']==y['root_action']:
            return (2 if exact else 3,'cross_video_same_action_signature' if exact else 'cross_video_same_root_action')
    return None


def pairing_audit():
    pool=read(BASE/'PRIVATE_DONOR_POOL.json');matches=read(BASE/'PRIVATE_MATCHES.json');nlp=read(BASE/'PRIVATE_NLP_TEXT.json')
    people={'man','woman','boy','girl','adult','person','child','baby','he','she','him','her','who','whom'}
    for text,z in nlp.items():
        words=z['words'];sig=z['signature'];verbs={w['id'] for w in words if w['upos']=='VERB'}
        assert sig['lexical_verbs']==sorted({w['lemma'] for w in words if w['upos']=='VERB'})
        atoms=set()
        for w in words:
            if w['upos']=='VERB':atoms.add(('verb',w['lemma']))
            if w['head'] in verbs and w['deprel'] in {'obj','iobj','compound:prt'}:
                atoms.add((w['deprel'],'person' if w['lemma'] in people else w['lemma']))
            if w['head'] in verbs and w['deprel'] in {'nsubj:pass','aux:pass'}:atoms.add(('voice','passive'))
        assert tuple_sig(sig['identity_signature'])==tuple(sorted(atoms))
    coverage={};reviews={}
    for ds in ['vidstg','hc2']:
        assert len(pool[ds])==({'vidstg':561,'hc2':3482}[ds])
        for d in pool[ds]:assert d['signature']==nlp[normalize(d['caption'])]['signature']
        bymedia=collections.defaultdict(set)
        for d in pool[ds]:bymedia[d['video_sha256']].add(normalize(d['caption']))
        if ds=='hc2':assert max(map(len,bymedia.values()))==1
        counts=collections.Counter();private=[]
        for i,m in matches[ds].items():
            a=m['recipient'];assert a['signature']==nlp[normalize(a['caption'])]['signature']
            for arm,p in m['pairs'].items():
                choices=[]
                for b in pool[ds]:
                    k=kind(a,b,arm)
                    if k is not None:
                        score=(k[0],abs(len(a['caption'].split())-len(b['caption'].split())),
                            hashlib.sha256(f'controlled-query-v1|{ds}|{a["key"]}|{arm}|{b["key"]}'.encode()).hexdigest())
                        choices.append((score,b,k))
                if not choices:assert p is None;continue
                _,b,k=min(choices,key=lambda v:v[0]);assert b==p['donor'] and k==(p['tier'],p['kind'])
                assert len(choices)==p['eligible_donors']
                counts[arm+'/'+str(k[0])]+=1
                private.append(dict(source_index=int(i),arm=arm,tier=k[0],recipient=a['caption'],donor=b['caption'],kind=k[1]))
        coverage[ds]=dict(counts);reviews[ds]=private
    p=BASE/'PRIVATE_PAIR_READBACK.json'
    if p.exists():assert read(p)==reviews,'Existing immutable text readback differs'
    else:write(p,reviews)
    return dict(parsed_texts=len(nlp),donor_pools={d:len(p) for d,p in pool.items()},
        deterministic_min_tier_length_SHA_reproduced=True,strong_action_atoms_reconstructed=True,
        exact_media_single_caption_HC_verified=True,coverage=coverage,
        limitation='operational text signatures, not complete event equivalence or cross-video identity')


def audit():
    verify(True);tick=time.time();torch.set_num_threads(2)
    cfg=read(PUB/'CONFIG.json');co=read(BASE/'COHORT.json');seal=read(BASE/'GLOBAL_READOUT_SEAL.json')
    join=read(PUB/'LABEL_JOIN.json');matches=read(BASE/'PRIVATE_MATCHES.json')
    pairing=pairing_audit();assert seal['controlled_cells']==324 and not seal['GT_read'] and seal['time']<join['time']
    review=read(BASE/'TEXT_CONTROL_REVIEW.json')
    assert review['time']<seal['time'] and review['pairing_sha256']==sha(BASE/'PRIVATE_MATCHES.json')
    assert not review['GT_time_or_metrics_read'] and review['fixed_mapping_unchanged']
    accepted=read(BASE/'SMOKE_ROOT_ACCEPTANCE.json');assert accepted['status']=='passed' and not accepted['GT_read']
    assert sha(ROOT/'methods/CURRENT_METHOD.json')==cfg['production_method_sha256']
    count=controlled=changed=probe_count=0;features={}
    for ds in ['vidstg','hc2']:
        rows=read(PLAN/ds/'PLAN.json')['rows'];old=checked(ATLAS/ds/'FROZEN_ATLAS.pt');probe_count+=len(old['models'])
        assert sha(ATLAS/ds/'FROZEN_ATLAS.pt')==cfg['probe_hashes'][ds]
        assert sha(BASE/ds/'SMOKE.json')==accepted['receipts'][ds]
        receipt=read(BASE/ds/'CAPTURE_BARRIER.json')
        assert accepted['time']<receipt['time']<seal['time'] and receipt['model_restored'] and not receipt['GT_read']
        for p,h in receipt['files'].items():assert sha(BASE/p)==h
        assert sha(BASE/ds/'SEALED_READOUT.pt')==seal['files'][ds]
        outputs={r['cell']:r['arms'] for r in checked(BASE/ds/'SEALED_READOUT.pt')}
        scored={r['cell']:r for r in read(PUB/ds/'ROWS.json')}
        predecessor={r['cell']:r for r in read(ROOT/f'results/tastvg_query_swap_specificity/2026-10-03/{ds}/ROWS.json')}
        gt={s:read(POOL/ds/f'GT_LABELS_{s}.json') for s in ['search','confirm']};changes=[]
        for c in co['cells']:
            if c['dataset']!=ds:continue
            tr=checked(ROOT/c['feature']);state=torch.load(ROOT/c['state_payload'],map_location='cpu',weights_only=False)
            assert sha(ROOT/c['state_payload'])==read(Path(c['state_payload']).with_suffix('.json'))['sha256']
            assert state_hash(state['pre_state'])==tr['A_state_pre_sha256']==c['pre_sha']
            assert state_hash(state['post_state'])==tr['A_state_post_sha256']==c['post_sha']
            arms={'true':tr,'generic':checked(OLD/ds/'swap_features'/f'{prefix(c)}.pt')}
            for a,p in matches[ds][str(c['parent'])]['pairs'].items():
                if p is not None:arms[a]=checked(BASE/ds/'features'/a/f'{prefix(c)}.pt')
            assert set(arms)==set(outputs[tr['cell_key']])==set(scored[tr['cell_key']]['available_arms'])
            for arm,z in arms.items():
                for k in ['cell_key','frame_ids','candidate_indices','anchor_index','pixel_sha256','A_state_pre_sha256','A_state_post_sha256']:
                    assert z[k]==tr[k],(arm,k)
                if arm in ['event','subject']:
                    p=matches[ds][str(c['parent'])]['pairs'][arm]
                    assert z['tier']==p['tier'] and z['kind']==p['kind'] and z['arm']==arm
                    assert z['donor_caption_sha256']==hashlib.sha256(p['donor']['caption'].encode()).hexdigest()
                    assert not z['GT_read'] and not z['candidate_support_changed'] and z['parameter_updates']==0
                    delta=(z['hidden'].double()-tr['hidden'].double()).numpy();controlled+=1
                    changed+=int(np.any(delta!=0));changes.append(dict(cell=z['cell_key'],arm=arm,source_index=c['parent'],
                        hidden_RMS_change=float(np.sqrt(np.mean(delta**2))),hidden_max_change=float(np.max(abs(delta)))))
                h=z['hidden'].double().numpy();assert np.isfinite(h).all()
                x,g,context=independent.feature_values(h,z['frame_ids'],z['candidate_indices'],rows[c['parent']]['input']['fps'])
                independent.compare(x,z['x']);independent.compare(g,z['geometry']);assert context==z['context']
                ids=np.array(z['frame_ids']);intervals=np.array([[ids[i],ids[j]+1] for i,j in z['candidate_indices']]);s,e=gt[c['split']][str(c['parent'])]['span']
                overlap=np.maximum(0,np.minimum(intervals[:,1],e)-np.maximum(intervals[:,0],s));length=np.diff(intervals,axis=1)[:,0]
                labels=dict(event=((ids>=s)&(ids<e)).astype(float),precision=overlap/length,
                    recall=overlap/(e-s),tiou=overlap/(length+e-s-overlap))
                for name,p in outputs[tr['cell_key']][arm].items():
                    if name.endswith('/logit'):continue
                    family,view,task,_=name.split('/');model=old['models'][name]
                    xx=h if family=='frame' else (g if view=='Geometry' else x[:,1280:1792] if view=='Contrast' else x)
                    linear=((xx-np.array(model['mean']))/np.array(model['std']))@np.array(model['weight'])+model['bias']
                    pred=expit(linear) if model['model']=='logistic' else linear
                    independent.compare(pred,p);m=scored[tr['cell_key']]['metrics'][arm+'/'+name]
                    y=labels[task];error=pred-y
                    independent.compare([m['y'],m['y2'],m['prediction'],m['mse'],m['mae']],
                        [y.mean(),(y*y).mean(),pred.mean(),(error*error).mean(),abs(error).mean()])
                    if np.var(y)>1e-12:independent.compare(m['within_r2'],1-np.mean(error**2)/np.var(y))
                    else:assert m['within_r2'] is None
                    if task=='event':
                        independent.compare(m['logloss'],np.mean(np.logaddexp(0,linear)-y*linear))
                        independent.compare(linear,outputs[tr['cell_key']][arm][name+'/logit'])
                        if len(set(y))==2:independent.compare([m['auc'],m['ap']],[roc_auc_score(y,pred),average_precision_score(y,pred)])
                        else:assert m['auc'] is m['ap'] is None
                    if arm in ['true','generic']:
                        key=('true' if arm=='true' else 'swap')+'/'+name
                        for f,v in m.items():
                            ref=predecessor[tr['cell_key']]['metrics'][key][f]
                            if v is None:assert ref is None
                            else:independent.compare(v,ref,tol=0)
            count+=1
        features[ds]=changes
    from scripts.audit_tastvg_controlled_query_public_v1 import audit as public
    anonymous=public(PUB);assert anonymous['status']=='passed'
    assert count==288 and controlled==324 and 0<=changed<=controlled and probe_count==136 and not torch.cuda.is_initialized()
    receipt=dict(status='passed',time=time.time(),original_cells=count,controlled_arrival_arms=controlled,
        changed_controlled_hidden=changed,unchanged_frozen_source_models=probe_count,
        scalar_checks=independent.checks,maximum_numeric_error=independent.maximum,
        original_generic_metric_bitwise_parity=True,fixed_pixels_candidates_A_state=True,
        original_GT_only_after_global_score_seal=True,production_method_unchanged=True,
        pre_score_text_caveats_preserved=True,
        linguistic_matching_readback=pairing,independent_public_audit=anonymous,CPU_wall_seconds=time.time()-tick)
    write(BASE/'FINAL_ROOT_AUDIT.json',receipt);write(PUB/'ROOT_AUDIT.json',receipt)
    write(PUB/'PUBLIC_AUDIT.json',anonymous);write(PUB/'FEATURE_CHANGE.json',features)
    print(json.dumps(receipt,indent=2,allow_nan=False))


if __name__=='__main__':audit()
