"""Confirm that the positive local ranking signal survives winner rejection and a paired seed."""
from argparse import ArgumentParser
from collections import Counter, defaultdict
from pathlib import Path
import sys

import numpy as np

from catalog import HOME, OVERNIGHT, PRIVATE, RESULTS, atomic_json, earlier_rank, read, sha, write_table
from compare import HEAD_FIELDS, fit_augmented, fit_head, head_scores, load_features, load_labels
from compare import rows_for, summarize, supervision_groups
from verify_added_labels import summary as verification_summary, test_outcome, train
from verify_winner import CandidateReadout, inner_partitions


LOCK = HOME/'verifier_protocol.json'


def loaded():
    sys.path.insert(0,str(OVERNIGHT))
    data,labels,truth_scope=load_labels()
    feature,feature_pin=load_features(data)
    representation=read(PRIVATE/'HEAD_SEAL.json')
    scores=PRIVATE/'ALL_SCORES.json'
    fits=PRIVATE/'SCALAR_240_FITS.json'
    scalar_pin=read(PRIVATE/'SCALAR_240_SEAL.json')
    assert representation['features']==feature_pin and representation['scores']==sha(scores)
    assert representation['labels']==truth_scope['label_seal'] and not representation['protected_data_access']
    assert scalar_pin['folds']==sha(fits)
    return data,labels,feature,read(scores),read(fits),dict(inputs=representation['labels'],protocol=sha(LOCK),
                 fixed_rank_seal=sha(PRIVATE/'HEAD_SEAL.json'),feature_seal=feature_pin)


def current(x, values):
    scores=np.asarray(values)
    slot=int(scores.argmax())
    return CandidateReadout(x,'pool_centered',scores,slot,earlier_rank.winner_status(x,slot))


def verify_candidate():
    data,labels,feature,scores,folds,provenance=loaded()
    by_key={x.alias:x for x in data}
    rows,details=[],[]
    for i,f in enumerate(folds,1):
        if f['platform']!='waymo':
            continue
        outer=[by_key[key] for key in f['fit']]
        assert all(x.rec!=f['held'] for x in outer)
        proof=[]
        inner_d, inner_new=[],[]
        for level,(fit,held) in enumerate(inner_partitions(outer),1):
            assert not {a.rec for a in fit}&{a.rec for a in held}
            w, _=fit_augmented(fit,labels)
            net,diag=fit_head(list(supervision_groups(fit,labels)),feature,'extra',w,331119+100*i+level)
            proof.append(dict(fit=[x.alias for x in fit],held=[x.alias for x in held],
                              head_training=diag))
            for x in held:
                base=x.example.unary
                z=head_scores(net,feature[x.alias],'extra',base+x.centered@w)
                inner_d.append(current(x,base))
                inner_new.append(current(x,z))
        assert Counter(c.example.alias for c in inner_new)==dict.fromkeys(f['fit'],1)
        foldsweep=dict(held=f['held'],fit=f['fit'],cases=f['cases'],inner=proof,verifier={})
        for rule,arm,examples in (('actual_raw_D','raw_D',inner_d),('actual_new_D','D_new_evidence',inner_new)):
            risk,fit_log=train(examples,labels,augmented=True)
            foldsweep['verifier'][rule]=fit_log
            for key in f['cases']:
                x=by_key[key]
                candidate=current(x,scores[key][arm])
                p=risk(candidate)
                rows.append(dict(case=x.alias,recording=x.rec,cohort=x.cohort,platform='waymo',rule=rule,arm=arm,
                                 outcome=test_outcome(candidate,labels),reject=p>=.5))
                rows.append(dict(case=x.alias,recording=x.rec,cohort=x.cohort,platform='waymo',rule=rule+'_keep',arm=arm,
                                 outcome=test_outcome(candidate,labels),reject=False))
        details.append(foldsweep)
        if len(details)%12==0:
            print('Nested candidate verification folds',len(details),flush=True)
    table={}
    variants=sorted({r['rule'] for r in rows})
    for group in ('initial','enriched','pooled'):
        table[group]={rule:verification_summary([r for r in rows if r['rule']==rule and (group=='pooled' or r['cohort']==group)])
                      for rule in variants}
    method=table['pooled']['actual_new_D']
    existing=table['pooled']['actual_raw_D']
    direct=table['pooled']['actual_raw_D_keep']
    check=dict(correct_acceptance_no_drop_vs_original=method['correct_accepted']>=direct['correct_accepted'],
               correct_acceptance_no_drop_vs_matched=method['correct_accepted']>=existing['correct_accepted'])
    write_table(RESULTS/'feature_winner_verification.csv',rows)
    atomic_json(PRIVATE/'FEATURE_WINNER_FOLDS.json',details)
    atomic_json(RESULTS/'FEATURE_WINNER_VERIFICATION.json',dict(metrics=table,criteria=check,
              folds=len(details),provenance=provenance,protected_data_access=False))
    print('Actual candidate acceptance',check,flush=True)


def repeat():
    data,labels,feature,prior,folds,provenance=loaded()
    by_key={x.alias:x for x in data}
    out={k:{arm:q[arm] for arm in ('raw_D','A_known_only','B_verified') if arm in q}
         for k,q in prior.items()}
    for i,f in enumerate(folds,1):
        if f['B'] is None:
            continue
        b=np.array(f['B']['weights'])
        registered=list(supervision_groups([by_key[k] for k in f['fit']],labels))
        for arm,field in HEAD_FIELDS.items():
            head,_=fit_head(registered,feature,field,b,311119+i)
            for key in f['cases']:
                out[key][arm]=head_scores(head,feature[key],field,np.asarray(out[key]['B_verified']))
    scorefile=PRIVATE/'ALTERNATE_SEED_SCORES.json'
    atomic_json(scorefile,out)
    rows=rows_for(data,out,labels)
    write_table(RESULTS/'seed_repeat_rows.csv',rows)
    stats=summarize(rows)
    atomic_json(RESULTS/'SEED_REPEAT.json',dict(metrics=stats,frames=len(data),seed_formula='311119+fold',
                             pred=sha(scorefile),provenance=provenance,protected_data_access=False))
    print('Paired seed repeat completed',flush=True)


if __name__=='__main__':
    p=ArgumentParser(description=__doc__)
    p.add_argument('step',choices=('verifier','seed'))
    verify_candidate() if p.parse_args().step=='verifier' else repeat()
