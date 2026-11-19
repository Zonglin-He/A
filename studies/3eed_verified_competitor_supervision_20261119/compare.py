"""Source-held-recording scalar and fixed-capacity feature comparisons."""
from argparse import ArgumentParser
from collections import Counter, defaultdict
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import minimize

from catalog import (FULL, HOME, OVERNIGHT, PRIVATE, RESULTS, PROTOCOL, atomic_json, csv_rows,
                     earlier, earlier_rank, label_truths, read, sha, write_table)


SCALAR = ('raw_D','A_known_only','B_verified')
HEAD_FIELDS = {'C_old_head':'old','D_new_evidence':'extra','D_rgb_off':'rgb_off','D_rgb_shuffled':'rgb_shuffled'}


def load_labels():
    data, truth, previous = label_truths()
    datafile=PRIVATE/'CANDIDATE_LABELS.json'
    labels=read(datafile)
    lock=read(PRIVATE/'LABEL_SEAL.json')
    assert lock['table']==sha(datafile) and lock['truth_unchanged'] and not lock['protected_data_access']
    assert len({r['alias'] for r in labels})==len(data)==144
    by_key={r['alias']:r for r in labels}
    out={}
    for x in data:
        line=by_key[x.alias]
        assert line['owner']==x.owners.tolist()
        relation=np.array(line['reference_match'])
        valid=np.array(line['instance_validity'])
        assert np.isin(relation,(-1,0,1)).all() and np.isin(valid,(-1,0,1)).all()
        label=x.example
        inherited=np.where(x.owners<0,-1,np.where(x.owners==label.gold,1,0))
        np.testing.assert_array_equal(relation[x.owners>=0],inherited[x.owners>=0])
        assert (relation[x.owners==-1] <= 0).all()
        for j in np.flatnonzero((x.owners==-1)&(relation==0)):
            key=str(j)
            assert j in line['shortlisted'] and line['geometry'][key]['verified_exclusion']
        out[x.alias]=relation
    return data,out,dict(label_seal=sha(PRIVATE/'LABEL_SEAL.json'),original_inputs=previous)


def supervision_groups(training, labels):
    """Use known physical identities and singleton certified negatives."""
    for x in training:
        e=x.example
        if e.status!='covered':
            continue
        groups=[np.flatnonzero(x.owners==j) for j in (e.gold,*sorted(set(x.owners)-{-1,e.gold}))]
        new=np.flatnonzero((x.owners==-1)&(labels[x.alias]==0))
        groups.extend(np.array([j]) for j in new)
        if len(groups)>1:
            assert len(groups[0]) and all(len(q) for q in groups)
            yield x,groups


def fit_augmented(train, labels, limit=240):
    examples=[(x,groups,x.centered) for x,groups in supervision_groups(train,labels)]
    n_new=sum(int(((x.owners==-1)&(labels[x.alias]==0)).sum()) for x,_,_ in examples)
    if not examples:
        return np.zeros(3),dict(n_new=n_new,training_frames=0,optimization='empty',train_improvement=0.)
    objective=earlier_rank.rank_loss(examples,pair=False)
    starts=(np.zeros(3),np.array([.4,0.,0.]),np.array([-.4,0.,0.]))
    weights=min(starts,key=objective)
    value=objective(weights)
    any_converged=False
    for init in starts:
        result=minimize(objective,init,method='Powell',bounds=[(-2.,2.)]*3,
                        options={'xtol':1.e-5,'ftol':1.e-8,'maxfev':limit})
        assert np.isfinite(result.fun) and np.isfinite(result.x).all()
        any_converged |= bool(result.success)
        if result.fun<value:
            value,weights=result.fun,result.x
    return weights,dict(n_new=n_new,training_frames=len(examples),
                        optimization='powell' if any_converged else 'powell_budget',
                        train_improvement=objective(np.zeros(3))-float(value))


def scalar_scores(data, labels, budget):
    if budget not in (240,1200):
        raise ValueError('Use locked matched solver budgets')
    if budget==1200:
        numeric=read(FULL/'results/NUMERICAL_CHECK.json')
        archive=FULL/'private/numeric_budget/FIT_DETAILS.json'
        assert numeric['original_completed_summary']==sha(FULL/'results/SUMMARY.json')
        assert numeric['larger_budget_summary']==sha(archive.with_name('SUMMARY.json'))
        archived=read(archive)
    else:
        archived=read(FULL/'private/FIT_DETAILS.json')
    checked={(f['held'],f['cohort'],f['platform']):f for f in archived}
    predictions, ledger = {}, []
    for train,test in earlier.outer_folds(data):
        case=test[0]
        key=(case.rec,case.cohort,case.example.platform)
        a=checked[key]
        assert a['cases']==[x.alias for x in test] and a['calibration']==[x.alias for x in train]
        detail=dict(held=case.rec,cohort=case.cohort,platform=case.example.platform,cases=a['cases'],
                    fit=a['calibration'],B=None)
        if train:
            w_a=np.array(a['rank']['pool_centered']['weights'])
            w_b,info=fit_augmented(train,labels,budget)
            if info['n_new']==0:
                np.testing.assert_allclose(w_b,w_a,atol=1.e-12,rtol=0.)
            detail['A']=dict(weights=w_a.tolist(),training_frames=a['rank']['pool_centered']['frames'])
            detail['B']=dict(weights=w_b.tolist(),**info)
        ledger.append(detail)
        for x in test:
            base=x.example.unary
            pack={'raw_D':base.tolist()}
            if train:
                pack.update(A_known_only=(base+x.centered @ w_a).tolist(),
                            B_verified=(base+x.centered @ w_b).tolist())
            predictions[x.alias]=pack
    return predictions,ledger


def rows_for(data, scores, labels):
    rows=[]
    for x in data:
        label=labels[x.alias]
        for arm,score in scores[x.alias].items():
            score=np.asarray(score)
            assert score.shape==label.shape and np.isfinite(score).all()
            row=earlier_rank.rank_row(x,arm,score)
            top=int(score.argmax())
            previous_slot=int(np.argmax(scores[x.alias]['A_known_only'])) if 'A_known_only' in scores[x.alias] else 0
            correct=row['winner_status']=='correct'
            a_correct=x.owners[previous_slot]==x.example.gold and x.example.gold>=0
            row.update(confirmed_wrong=bool(label[top]==0),
                       newly_verified_negative_winner=bool(label[top]==0 and x.owners[top]==-1),
                       uncertified_winner=bool(label[top]==-1),
                       rescued_vs_A=bool(correct and not a_correct),harmed_vs_A=bool(a_correct and not correct),
                       original_correct_kept=bool(correct and x.owners[0]==x.example.gold))
            rows.append(row)
    return rows


def summarize(rows):
    panels=defaultdict(list)
    for r in rows:
        panels[r['cohort']+'_'+r['platform']].append(r)
        if r['platform']=='waymo':
            panels['waymo_pooled'].append(r)
    out={}
    for name,subset in sorted(panels.items()):
        by_arm=defaultdict(list)
        for r in subset:
            by_arm[r['arm']].append(r)
        out[name]={}
        for arm,t in sorted(by_arm.items()):
            stats=earlier_rank.rank_summary(t)
            covered=[r for r in t if r['status']=='covered']
            head=[r['hard_concordance'] for r in covered if r['hard_concordance'] is not None]
            stats.update(confirmed_wrong_winner=sum(r['confirmed_wrong'] for r in covered),
                         certified_negative_winner=sum(r['newly_verified_negative_winner'] for r in covered),
                         uncertified_winner=sum(r['uncertified_winner'] for r in covered),
                         rescued_vs_A=sum(r['rescued_vs_A'] for r in covered),harmed_vs_A=sum(r['harmed_vs_A'] for r in covered),
                         original_correct_kept=sum(r['original_correct_kept'] for r in covered),
                         paired_accuracy=None if not head else float(np.mean(head)))
            out[name][arm]=stats
    return out


def verify_scalar_replay(rows, budget):
    origin = FULL/'results/rank_rows.csv' if budget==240 else FULL/'private/numeric_budget/rank_rows.csv'
    baseline={(x['case'],x['arm']):x for x in csv_rows(origin)}
    for r in rows:
        if r['arm'] not in ('raw_D','A_known_only'):
            continue
        a='D' if r['arm']=='raw_D' else 'pool_centered'
        old=baseline[r['case'],a]
        assert old['winner_status']==r['winner_status']
        assert (int(old['box_rank']) if old['box_rank'] else None)==r['box_rank']
        assert (int(old['instance_rank']) if old['instance_rank'] else None)==r['instance_rank']


def save_scalar(budget):
    assert budget in (240,1200)
    data,labels,lock=load_labels()
    scores,folds=scalar_scores(data,labels,budget)
    scorefile=PRIVATE/f'SCALAR_{budget}_SCORES.json'
    assert not scorefile.exists()
    atomic_json(scorefile,scores)
    fitfile=PRIVATE/f'SCALAR_{budget}_FITS.json'
    atomic_json(fitfile,folds)
    atomic_json(PRIVATE/f'SCALAR_{budget}_SEAL.json',dict(scores=sha(scorefile),folds=sha(fitfile),**lock,
        program=sha(Path(__file__)),protocol=sha(PROTOCOL),protected_data_access=False))
    rows=rows_for(data,scores,labels)
    verify_scalar_replay(rows,budget)
    write_table(RESULTS/f'scalar_{budget}_rows.csv',rows)
    metrics=summarize(rows)
    atomic_json(RESULTS/f'SCALAR_{budget}.json',dict(metrics=metrics,frames=len(data),
          fits=digest_fits(folds),budget=budget,exact_D_and_A_replay=True,protected_data_access=False))
    print('Scalar run',budget,'Waymo Top-1',{k:v['correct'] for k,v in metrics['waymo_pooled'].items()},flush=True)


def digest_fits(rows):
    fitted=[r for r in rows if r['B'] is not None]
    return dict(folds=len(rows),updated=len(fitted),status=dict(Counter(r['B']['optimization'] for r in fitted)),
                augmented_folds=sum(r['B']['n_new']>0 for r in fitted))


def load_features(data):
    from predict171 import read_npz

    root=PRIVATE/'features'
    meta=read(root/'SEAL.json')
    lock=read(root/'LOCK.json')
    assert not meta['protected_data_access'] and not meta['labels_loaded'] and meta['source_weights_unchanged']
    assert meta['lock']==sha(root/'LOCK.json') and lock['protocol']==sha(PROTOCOL)
    assert lock['producer']==sha(HOME/'capture_features.py')
    case={x.alias:x for x in data}
    assert set(meta['files'])==set(case)
    out={}
    for key,checksum in meta['files'].items():
        path=root/(key+'.npz')
        assert sha(path)==checksum
        p=read_npz(path)
        for var in HEAD_FIELDS.values():
            assert p[var].shape==(len(case[key].owners),352) and np.isfinite(p[var]).all()
        assert p['lock'].item()==meta['lock'] and p['frozen_forward_path'].item()
        assert p['usable'].shape==(len(case[key].owners),) and p['phrase'].shape==(288,)
        out[key]=p
    return out,sha(root/'SEAL.json')


def head_batch(rows, features, field, b):
    import torch

    n=len(rows)
    k=max(len(r.owners) for r,_ in rows)
    data=np.zeros((n,k,352),np.float32)
    base=np.zeros((n,k),np.float32)
    usable=np.zeros((n,k),bool)
    mask=np.zeros((n,k,k),bool)
    phrases=np.stack([features[x.alias]['phrase'] for x,_ in rows])
    for i,(x,groups) in enumerate(rows):
        pp=features[x.alias]
        m=len(x.owners)
        data[i,:m]=pp[field]
        base[i,:m]=x.example.unary + x.centered @ b
        usable[i,:m]=pp['usable']
        for j,group in enumerate(groups):
            mask[i,j,group]=True
    return tuple(torch.from_numpy(a) for a in (phrases,data,base,usable,mask))


def fit_head(rows, features, field, b, seed):
    import torch
    from appearance_fit import Readout

    torch.manual_seed(seed)
    model=Readout().cpu()
    assert sum(p.numel() for p in model.parameters())==10240
    if not rows:
        return None,None
    torch.set_num_threads(1)
    phrase,data,base,usable,mask=head_batch(rows,features,field,b)

    def objective(ix):
        residual=.5*model(phrase[ix],data[ix])
        slots=base[ix]+residual*usable[ix]
        groups=slots[:,None,:].masked_fill(~mask[ix],-1.e4).amax(-1)
        return (groups.logsumexp(1)-groups[:,0]).mean()

    opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.01)
    rng=torch.Generator().manual_seed(seed+1)
    with torch.no_grad():
        before=float(objective(torch.arange(len(rows))))
    model.train()
    for _ in range(16):
        for ix in torch.randperm(len(rows),generator=rng).split(24):
            opt.zero_grad(set_to_none=True)
            loss=objective(ix)
            assert torch.isfinite(loss)
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        after=float(objective(torch.arange(len(rows))))
    return model,dict(train_loss_before=before,train_loss_after=after,
                      training_frames=len(rows),visible_training_frames=sum(
                          bool(features[x.alias]['usable'].any()) for x,_ in rows))


def head_scores(model, entry, field, base):
    import torch

    if model is None:
        return np.asarray(base).tolist()
    with torch.inference_mode():
        residual=.5*model(torch.from_numpy(entry['phrase'][None]),torch.from_numpy(entry[field][None])).numpy()[0]
    return (base+residual*entry['usable']).tolist()


def run_heads():
    sys.path.insert(0,str(OVERNIGHT))
    data,labels,enabled=load_labels()
    feat,source_seal=load_features(data)
    start=read(PRIVATE/'SCALAR_240_SEAL.json')
    fitfile=PRIVATE/'SCALAR_240_FITS.json'
    scorefile=PRIVATE/'SCALAR_240_SCORES.json'
    assert start['scores']==sha(scorefile) and start['folds']==sha(fitfile)
    assert start['label_seal']==enabled['label_seal'] and start['program']==sha(Path(__file__))
    out=read(scorefile)
    by_case={x.alias:x for x in data}
    ledger=[]
    for i,f in enumerate(read(fitfile),1):
        if f['B'] is None:
            continue
        w=np.array(f['B']['weights'])
        train=[by_case[k] for k in f['fit']]
        assert f['held'] not in {r.rec for r in train}
        cells=list(supervision_groups(train,labels))
        row=dict(held=f['held'],training=f['fit'],testing=f['cases'],heads={})
        for arm,field in HEAD_FIELDS.items():
            estimator,diag=fit_head(cells,feat,field,w,251119+i)
            row['heads'][arm]=diag
            for key in f['cases']:
                out[key][arm]=head_scores(estimator,feat[key],field,np.asarray(out[key]['B_verified']))
        ledger.append(row)
        if i%12==0:
            print('Completed representation folds',i,flush=True)
    scores=PRIVATE/'ALL_SCORES.json'
    assert not scores.exists()
    atomic_json(scores,out)
    fit=PRIVATE/'HEAD_FITS.json'
    atomic_json(fit,ledger)
    atomic_json(PRIVATE/'HEAD_SEAL.json',dict(features=source_seal,scalar=sha(PRIVATE/'SCALAR_240_SEAL.json'),
               scores=sha(scores),fit=sha(fit),old_head_definition=sha(OVERNIGHT/'appearance_fit.py'),
               labels=enabled['label_seal'],protected_data_access=False,source_weights_unchanged=True))
    rows=rows_for(data,out,labels)
    verify_scalar_replay(rows,240)
    write_table(RESULTS/'all_rows.csv',rows)
    out=summarize(rows)
    primary=out['waymo_pooled']
    a=primary['D_new_evidence']
    rank_gate=dict(
        top1=a['correct']>max(primary[k]['correct'] for k in ('raw_D','B_verified','C_old_head','D_rgb_shuffled')),
        mrr=a['instance_mrr']>max(primary[k]['instance_mrr'] for k in ('raw_D','B_verified','C_old_head','D_rgb_shuffled')),
        no_top1_panel_loss=all(out[k]['D_new_evidence']['correct']>=out[k]['raw_D']['correct']
                               for k in ('initial_waymo','enriched_waymo')),
        keeps_baseline_winners=a['original_correct_kept']>=primary['raw_D']['original_correct_kept'],
    )
    measure=Counter(bool(f['usable'].any()) for f in feat.values())
    changed=sum(not np.array_equal(f['extra'],f['rgb_shuffled']) for f in feat.values())
    atomic_json(RESULTS/'REPRESENTATION.json',dict(metrics=out,head_parameter_count=10240,features=source_seal,
            observation_frame_support={'one_or_more':measure[True],'none':measure[False],
                                       'rgb_changed_when_shuffled':changed},
            rank_gate=rank_gate,joint_eligible=all(rank_gate.values()),protected_data_access=False))
    print('Matched feature comparison done',rank_gate,flush=True)


if __name__=='__main__':
    p=ArgumentParser(description=__doc__)
    p.add_argument('step',choices=('scalar','numerical','heads'))
    step=p.parse_args().step
    if step=='heads':
        run_heads()
    else:
        save_scalar(240 if step=='scalar' else 1200)
