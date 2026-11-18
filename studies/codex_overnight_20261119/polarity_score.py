"""Inspect physical endpoint discrimination after sealing all predicate interventions."""
from collections import Counter
from pathlib import Path

import numpy as np

from polarity import DATA, HERE, PLATFORMS, R171, SPEC, WORK, read, sha, atomic_json
from edge_score import csv_atomic


def ranked(unary, owners, gold):
    order = np.argsort(-unary,kind='stable')
    hit = np.flatnonzero(owners[order]==gold)
    rank = int(hit[0])+1 if len(hit) else 0
    known = np.unique(owners[owners>=0])
    other = [float(unary[owners==k].max()) for k in known if k!=gold]
    if np.any(owners==gold) and other:
        diff = float(unary[owners==gold].max())-np.asarray(other)
        score = float((np.sum(diff>1e-10)+.5*np.sum(np.abs(diff)<=1e-10))/len(diff))
    else:
        score = ''
    return rank,score


def score():
    from labels166 import occupied_owners,physical_labels
    from predict171 import read_npz
    from score170 import label
    from score171 import select_manifest

    p = read(DATA/'SEAL.json')
    assert not p['target_labels_read'] and not p['protected_data_access']
    lock = read(DATA/'LOCK.json')
    assert lock['protocol']==sha(SPEC) and p['lock']==sha(DATA/'LOCK.json')
    old = read(R171/'PREDICTION_SEAL.json')
    assert lock['parent']==sha(R171/'PREDICTION_SEAL.json')
    assert p['files'].keys()==old['files'].keys()
    assert all(sha(DATA/key)==value for key,value in p['files'].items())
    out = HERE/'results/polarity'
    out.mkdir(parents=True,exist_ok=True)
    assert not (out/'AUDIT.json').exists()
    atomic_json(out/'SCORE_LOCK.json',dict(code=sha(Path(__file__)),predictions=sha(DATA/'SEAL.json')))
    rows=[]
    full, scoped = 0,0
    controlled = Counter()
    for c, v in select_manifest().items():
        for entry, blind, key in v:
            s_id,r_id = label(key,blind['subject']),label(key,blind['reference'])
            for source in PLATFORMS:
                if source==entry['target']:
                    continue
                full += 1
                index = f"{c}/{entry['case']:02d}_{source}.npz"
                a = read_npz(DATA/index)
                assert a['pin']==p['lock']
                if min(s_id,r_id)<0:
                    continue
                scoped += 1
                b = read_npz(WORK/'predictions'/index)
                assert sha(WORK/'predictions'/index)==old['files'][index]
                names,gt,cloud,ids = physical_labels(entry['row'],b['seeds'])
                for role,bclass,gold in [('s','subject_boxes',s_id),('r','D_natural_boxes',r_id)]:
                    regions=b[bclass]
                    owners = occupied_owners(regions,cloud,ids,len(names)) if len(regions) else np.empty(0,int)
                    base = a['base_'+role]
                    expected = b['subject_unary'] if role=='s' else b['D_native_unary'][:-1]
                    np.testing.assert_array_equal(base,expected)
                    diff, wrong = a['delta_'+role],a['wrong_'+role]
                    good = a['supported_'+role]
                    np.testing.assert_array_equal(np.sort(diff),np.sort(wrong))
                    assert (diff[~good]==0.).all() and (np.abs(diff)<=1.).all()
                    controlled[role] += int(not np.array_equal(diff,wrong))
                    meta = dict(cohort=c,frame=entry['case'],source=source,target=entry['target'],role=role,
                                known_correct_candidate=int((owners==gold).any()),
                                supported_correct_candidate=int((owners[good]==gold).any()),
                                num_supported=int(good.sum()),effective_mismatch=int(not np.array_equal(diff,wrong)),
                                seconds=float(a['seconds']))
                    for method,z in [('r171_unary',base),('polarity',base+diff),('mismatched_polarity',base+wrong)]:
                        rank, concordance = ranked(z,owners,gold)
                        rows.append(dict(**meta,method=method,rank=rank,reciprocal=1./rank if rank else 0.,
                                         is_correct=int(rank==1),identity_concordance=concordance))
    assert full==168 and scoped==60
    summary=[]
    for cohort in ('replay170','new_frames','combined'):
        for role in ('s','r'):
            r = [t for t in rows if t['role']==role and (cohort=='combined' or t['cohort']==cohort)]
            for m in ('r171_unary','polarity','mismatched_polarity'):
                v = [t for t in r if t['method']==m]
                k = [t['identity_concordance'] for t in v if t['identity_concordance']!='']
                summary.append(dict(cohort=cohort,role=role,method=m,n=len(v),
                        correct=sum(t['is_correct'] for t in v),
                        base_covered=sum(t['known_correct_candidate'] for t in v),
                        corrected_common_support=sum(t['supported_correct_candidate'] for t in v),
                        mean_reciprocal=float(np.mean([t['reciprocal'] for t in v])),
                        comparative_identity=float(np.mean(k)) if k else '',comparative_rows=len(k)))
    csv_atomic(out/'rows.csv',rows)
    csv_atomic(out/'summary.csv',summary)
    atomic_json(out/'AUDIT.json',dict(ordinary_rows=full,dual_verified=scoped,mismatches_effective=dict(controlled),
             files={f.name:sha(f) for f in out.glob('*.csv')},protected_data_access=False))
    print('Polarity endpoint discrimination',summary,flush=True)


if __name__=='__main__':
    score()
