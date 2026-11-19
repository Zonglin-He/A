"""Isolate availability of actual-winner labels on an unchanged scalar ranking."""
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from catalog import HOME, PRIVATE, PROTOCOL, RESULTS, atomic_json, earlier, earlier_rank, read, sha, write_table
from compare import load_labels
from verify_winner import inner_partitions, observation, readout


ARMS=('old_owner_supervision','verified_expression_supervision')


def label(entry, labels, augmented):
    x = entry.example
    if x.example.status=='natural_miss':
        return 1
    if x.example.status!='covered':
        return None
    owner=x.owners[entry.index]
    if owner>=0:
        return int(owner!=x.example.gold)
    if augmented and labels[x.alias][entry.index]==0:
        return 1
    return None


def train(rows, labels, augmented):
    sources=[]
    for row in rows:
        y=label(row,labels,augmented)
        if y is not None:
            sources.append((observation(row,'actual'),y))
    y=np.asarray([v for _,v in sources],dtype=float)
    positive=int(y.sum())
    negative=len(y)-positive
    if not positive or not negative:
        return lambda _:float(bool(positive)),dict(correct=negative,incorrect=positive,fitted=False)
    x=np.stack([v for v,_ in sources])
    avg,scale=x.mean(0),x.std(0)
    scale[scale<=1e-6]=1.
    z=np.c_[np.ones(len(x)),(x-avg)/scale]
    regularization=np.r_[0.,np.full(z.shape[1]-1,.15)]
    w=np.where(y>0,.5/positive,.5/negative)

    def objective(v):
        a=z@v
        reg=v*regularization
        loss=w@(np.logaddexp(0.,a)-y*a)+.5*(v@reg)
        grad=z.T@(w*(expit(a)-y))+reg
        return loss,grad

    result=minimize(objective,np.zeros(z.shape[1]),jac=True,method='L-BFGS-B')
    assert result.success and np.isfinite(result.x).all()

    def predict(e):
        return float(expit(result.x[0]+result.x[1:]@((observation(e,'actual')-avg)/scale)))

    return predict,dict(correct=negative,incorrect=positive,fitted=True)


def summary(rows):
    bins=defaultdict(list)
    for r in rows:
        bins[r['outcome']].append(r)
    def kept(key):
        return sum(not r['reject'] for r in bins[key])
    def rejected(key):
        return len(bins[key])-kept(key)
    return dict(n=len(rows),correct=len(bins['correct']),correct_accepted=kept('correct'),
                confirmed_wrong=len(bins['confirmed_wrong']),confirmed_wrong_rejected=rejected('confirmed_wrong'),
                newly_certified_wrong=len(bins['newly_certified_wrong']),
                newly_certified_wrong_rejected=rejected('newly_certified_wrong'),
                natural_miss=len(bins['natural_miss']),natural_miss_rejected=rejected('natural_miss'),
                undetermined=len(bins['undetermined']),undetermined_rejected=rejected('undetermined'))


def test_outcome(entry, labels):
    x=entry.example
    if x.example.status=='natural_miss':
        return 'natural_miss'
    if x.example.status!='covered':
        return 'undetermined'
    q=labels[x.alias][entry.index]
    if q==1:
        return 'correct'
    if q==0:
        return 'newly_certified_wrong' if x.owners[entry.index]<0 else 'confirmed_wrong'
    return 'undetermined'


def run():
    data,labels,scope=load_labels()
    scalars=read(PRIVATE/'SCALAR_240_FITS.json')
    data_by={x.alias:x for x in data}
    fold=[t for t in scalars if t['fit']]
    all_rows, details=[],[]
    for f in fold:
        fit=[data_by[key] for key in f['fit']]
        assert f['held'] not in {x.rec for x in fit}
        sub=[]; proof=[]
        for train_rows,hold_rows in inner_partitions(fit):
            assert not {x.rec for x in train_rows}&{x.rec for x in hold_rows}
            w=earlier_rank.fit_rank(train_rows,'pool_centered')
            sub.extend(readout(x,'pool_centered',w) for x in hold_rows)
            proof.append(dict(fit=[x.alias for x in train_rows],eval=[x.alias for x in hold_rows]))
        assert Counter(x.example.alias for x in sub)==dict.fromkeys(f['fit'],1)
        base=earlier_rank.Fit(np.array(f['A']['weights']),0,0,0,'archive')
        post=[readout(data_by[key],'pool_centered',base) for key in f['cases']]
        trace=dict(held=f['held'],cases=f['cases'],fit=f['fit'],nested=proof,counts={})
        for armed in ARMS:
            predictor,evidence=train(sub,labels,armed==ARMS[1])
            trace['counts'][armed]=evidence
            for entry in post:
                x=entry.example
                p=predictor(entry)
                all_rows.append(dict(case=x.alias,recording=x.rec,platform=x.example.platform,
                                     cohort=x.cohort,rule=armed,outcome=test_outcome(entry,labels),
                                     reject=p>=.5))
        details.append(trace)
    metrics=defaultdict(dict)
    for cohort,platform in sorted({(r['cohort'],r['platform']) for r in all_rows}):
        for arm in ARMS:
            relevant=[r for r in all_rows if (r['cohort'],r['platform'],r['rule'])==(cohort,platform,arm)]
            metrics[cohort+'_'+platform][arm]=summary(relevant)
    write_table(RESULTS/'winner_supervision.csv',all_rows)
    atomic_json(PRIVATE/'VERIFIER_SPLITS.json',details)
    atomic_json(RESULTS/'WINNER_SUPERVISION.json',dict(rank_arm='A_known_only',fixed_threshold=.5,
               folds=len(details),metrics=metrics,outer_seal=sha(PRIVATE/'SCALAR_240_SEAL.json'),
               labels=scope['label_seal'],producer=sha(Path(__file__)),protected_data_access=False))
    print('Fixed actual-winner supervision comparison complete',flush=True)


if __name__=='__main__':
    run()
