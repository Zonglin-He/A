"""Check positive protection, separate label roles, matched budgets and source-only scoring."""
from collections import Counter
from pathlib import Path

import numpy as np

from catalog import HOME, PRIVATE, PROTOCOL, RESULTS, atomic_json, csv_rows, earlier, label_truths, read, sha, source_roster
from compare import HEAD_FIELDS, load_features, load_labels, supervision_groups


def check_new_labels(data, labelbook):
    from mechanisms162 import annotation_inventory, seed_owners
    from predict171 import read_npz
    from score import points_in
    from score154 import many_iou

    _, sources, _=label_truths()
    roster,_=source_roster()
    official={x['alias']:x for x in read(PRIVATE/'CANDIDATE_LABELS.json')}
    counts=Counter()
    for row in data:
        a=row.alias
        mask=labelbook[a]
        registered=official[a]
        assert registered['owner']==row.owners.tolist()
        known=row.owners>=0
        assert np.array_equal(mask[known]==1,row.owners[known]==row.example.gold)
        negatives=np.flatnonzero(~known & (mask==0))
        assert not (~known & (mask==1)).any()
        counts['certified_negatives']+=len(negatives)
        counts['shortlisted']+=len(registered['shortlisted'])
        counts['still_unknown']+=sum(int(mask[j]==-1) for j in registered['shortlisted'])
        if not len(negatives):
            continue
        true=sources[a]
        r=true['row']
        _, inv=annotation_inventory(r['metapath'],r['platform'],int(r['suffix'] or 0),r['object_index'])
        p=read_npz(roster[a][1])
        owners=seed_owners(p['points'],inv)
        ref=inv[row.example.gold].copy()
        ref[3:6]=ref[3:6]*1.2+.4
        assert np.all(many_iou(p['boxes'][negatives],ref)<1.e-12)
        assert (owners==row.example.gold).sum()>=6
        inclusion=points_in(p['boxes'][negatives],p['points'])
        assert (inclusion.sum(1)>=3).all()
        assert not (inclusion & (owners==row.example.gold)).any()
        for j in negatives:
            stamp=registered['geometry'][str(j)]
            assert j in registered['shortlisted'] and stamp['gap']>.02
    for x,groups in supervision_groups(data,labelbook):
        pos,*rest=groups
        assert np.all(labelbook[x.alias][pos]==1)
        assert np.all(labelbook[x.alias][np.concatenate(rest)]==0)
    assert counts['certified_negatives']+counts['still_unknown']==counts['shortlisted']
    return dict(counts)


def validate_fold_sets(data):
    keyed={x.alias:x for x in data}
    scal=read(PRIVATE/'SCALAR_240_FITS.json')
    his={(f['held'],f['cohort'],f['platform']):f for f in scal}
    for fit,test in earlier.outer_folds(data):
        s=test[0]
        r=his[(s.rec,s.cohort,s.example.platform)]
        assert r['fit']==[x.alias for x in fit] and r['cases']==[x.alias for x in test]
        assert set(r['fit']).isdisjoint(r['cases'])
    seen={}
    for rec in read(PRIVATE/'HEAD_FITS.json'):
        f=next(q for q in scal if q['held']==rec['held'])
        assert rec['testing']==f['cases'] and rec['training']==f['fit']
        n={r['training_frames'] if r else 0 for r in rec['heads'].values()}
        assert len(n)==1 and set(rec['heads'])==set(HEAD_FIELDS)
        seen[rec['held']]={arm:bool(r and r['train_loss_after']<r['train_loss_before'])
                          for arm,r in rec['heads'].items()}
    def inner(history, inner_key):
        total=0
        for row in history:
            ending=Counter()
            exclude={keyed[k].rec for k in row['cases']}
            for i in row[inner_key]:
                take=set(i['fit'])
                test=set(i.get('held',i.get('eval')))
                rf={keyed[k].rec for k in take}
                re={keyed[k].rec for k in test}
                assert rf.isdisjoint(re) and (rf|re).isdisjoint(exclude)
                assert take|test==set(row['fit'])
                ending.update(test)
                total+=1
            assert ending==dict.fromkeys(row['fit'],1)
        return total
    old=inner(read(PRIVATE/'VERIFIER_SPLITS.json'),'nested')
    new=inner(read(PRIVATE/'FEATURE_WINNER_FOLDS.json'),'inner')
    return dict(outer_folds=len(his),heads=len(seen),matched_winner_inner_folds=old,
                rich_winner_inner_folds=new,loss_reduced_by_head={name:sum(x[name] for x in seen.values())
                                                              for name in HEAD_FIELDS})


def verify_feature_readout(data):
    features, _=load_features(data)
    from common160 import array_sha
    from predict171 import read_npz

    roster,_=source_roster()
    for x in data:
        r=features[x.alias]
        orig=read_npz(roster[x.alias][1])
        assert array_sha(orig['raw'])==str(r['original_pool_sha'])
        assert str(r['sample_sha'])==str(orig['sample_sha'])
        assert not r['rgb_off'][:,288:344].any()
        for n in ('rgb_shuffled','rgb_off'):
            np.testing.assert_array_equal(r['extra'][:,:288],r[n][:,:288])
            np.testing.assert_array_equal(r['extra'][:,344:],r[n][:,344:])
        avail=r['usable']
        assert Counter(map(tuple,r['extra'][avail,288:344]))==Counter(map(tuple,r['rgb_shuffled'][avail,288:344]))
        np.testing.assert_array_equal(r['extra'][~avail],r['rgb_shuffled'][~avail])
    return len(features)


def independent_rows(data,scores_file,csv_file,labels):
    keyed={x.alias:x for x in data}
    reference=read(scores_file)
    rows=csv_rows(csv_file)
    keys={(r['case'],r['arm']) for r in rows}
    assert len(keys)==len(rows)==sum(map(len,reference.values()))
    assert {r['case'] for r in rows}==set(keyed)
    for r in rows:
        id=r['case']
        a=r['arm']
        e=keyed[id].example
        z=np.asarray(reference[id][a],dtype=float)
        assert len(z)==len(e.unary) and np.isfinite(z).all()
        i=int(np.argsort(-z,kind='stable')[0])
        assert (r['winner_status']=='correct')==(e.gold>=0 and int(e.owners[i])==e.gold)
        assert r['confirmed_wrong']==str(bool(labels[id][i]==0))
        if e.status=='covered':
            p=max(z[e.owners==e.gold])
            represented=set(t for t in e.owners if t>=0 and t!=e.gold)
            groups=[float(max(z[e.owners==j])) for j in represented]
            groups.extend(z[e.owners==-1].tolist())
            assert int(r['instance_rank'])==1+sum(q>p for q in groups)
        else:
            assert not r['instance_rank']
    return len(rows)


def audit_predicted_rejection():
    pairs=(
       ('WINNER_SUPERVISION.json','winner_supervision.csv',('initial_waymo','enriched_waymo','initial_drone')),
       ('FEATURE_WINNER_VERIFICATION.json','feature_winner_verification.csv',('initial','enriched','pooled')),
    )
    total=0
    for jf,cf,panels in pairs:
        result=read(RESULTS/jf)
        rows=csv_rows(RESULTS/cf)
        for p in panels:
            metrics=result['metrics'][p]
            for ar,m in metrics.items():
                selected=[r for r in rows if r['rule']==ar and (p=='pooled' or p==r['cohort'] or
                                             p==r['cohort']+'_'+r['platform'])]
                assert len(selected)==m['n']
                assert sum(r['outcome']=='correct' and r['reject']=='False' for r in selected)==m['correct_accepted']
                for key in ('confirmed_wrong','newly_certified_wrong','undetermined','natural_miss'):
                    assert sum(r['outcome']==key and r['reject']=='True' for r in selected)==m[key+'_rejected']
        total+=len(rows)
    return total


def run():
    d,labels,scope=load_labels()
    labels_checked=check_new_labels(d,labels)
    n=verify_feature_readout(d)
    splits=validate_fold_sets(d)
    stats={}
    for s,r in (('SCALAR_240_SCORES.json','scalar_240_rows.csv'),('SCALAR_1200_SCORES.json','scalar_1200_rows.csv'),
                ('ALL_SCORES.json','all_rows.csv'),('ALTERNATE_SEED_SCORES.json','seed_repeat_rows.csv')):
        stats[r]=independent_rows(d,PRIVATE/s,RESULTS/r,labels)
    for budget in (240,1200):
        seal=read(PRIVATE/f'SCALAR_{budget}_SEAL.json')
        assert seal['label_seal']==scope['label_seal'] and seal['protocol']==sha(PROTOCOL)
        assert seal['program']==sha(HOME/'compare.py')
        assert seal['scores']==sha(PRIVATE/f'SCALAR_{budget}_SCORES.json')
    seal=read(PRIVATE/'HEAD_SEAL.json')
    assert seal['scores']==sha(PRIVATE/'ALL_SCORES.json') and seal['source_weights_unchanged']
    assert seal['fit']==sha(PRIVATE/'HEAD_FITS.json')
    assert not seal['protected_data_access']
    stats['rejection_rows']=audit_predicted_rejection()
    files=set(r.name for r in RESULTS.iterdir() if r.suffix in ('.json','.csv'))-{'AUDIT.json'}
    atomic_json(RESULTS/'AUDIT.json',dict(status='PASS',n_feature_frames=n,labels=labels_checked,splits=splits,rows=stats,
                 input_scope=scope,source_checkpoint_update=False,protected_data_access=False,
                 files={f:sha(RESULTS/f) for f in sorted(files)}))
    print('Independent overlap, feature, split, ranking and rejection audit passed',flush=True)


if __name__=='__main__':
    run()
