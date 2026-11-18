"""Localize the effect of the role-exchange term using frozen paired predictions."""
import csv
from collections import Counter
from pathlib import Path

import numpy as np

from binding_collect import ARMS, HERE, PLATFORMS, PRIVATE, WORK, read, sha, atomic_json
from binding_score import checked
from edge_score import csv_atomic


def describe_owners(old, new):
    if old < 0 or new < 0:
        return 'unresolved'
    return 'same' if old==new else 'different'


def main():
    from labels166 import occupied_owners,physical_labels
    from score170 import label
    from score171 import select_manifest

    path,lock,npz = checked()
    cert = HERE/'results/binding_target/SCORE_SEAL.json'
    proof = read(cert)
    assert not proof['protected_data_access'] and proof['hashes']['rows.csv']==sha(cert.parent/'rows.csv')
    with (cert.parent/'rows.csv').open() as f:
        stats={(r['cohort'],int(r['frame']),r['source'],r['arm']):r for r in csv.DictReader(f)}
    compare,arm = 'additive_geometric','fixed_exchange_joint'
    old_index,new_index = ARMS.index(compare),ARMS.index(arm)
    hold=[]
    old_gold_count,exchange_gold_count=0,0
    for cohort, roster in select_manifest().items():
        for entry,blind,key in roster:
            pair = [label(key,blind[t]) for t in ('subject','reference')]
            for src in PLATFORMS:
                if src==entry['target']:
                    continue
                meta=(cohort,entry['case'],src)
                before=float(stats[(*meta,compare)]['official_iou'])
                after=float(stats[(*meta,arm)]['official_iou'])
                if abs(before-after)<1.e-10 and min(pair)<0:
                    continue
                filename=f"{cohort}/{entry['case']:02d}_{src}.npz"
                saved=npz(path/filename)
                old=npz(WORK/'predictions'/filename)
                ft=npz(PRIVATE/'target'/filename)
                assert old['sample_sha'].item() and saved['f_hash']==sha(PRIVATE/'target'/filename)
                names,_,cloud,owners = physical_labels(entry['row'],old['seeds'])
                assert len(names)==key['inventory_size']
                old_box,new_box=saved['boxes'][[old_index,new_index]]
                token=occupied_owners(np.stack((old_box,new_box)),cloud,owners,len(names))
                if min(pair)>=0:
                    old_gold_count+=int((token[0]==pair).all())
                    exchange_gold_count+=int((token[1]==pair).all())
                if abs(before-after)<1.e-10:
                    continue
                a,b=(tuple(x) for x in saved['choices'][[old_index,new_index]])
                same=ft['baseline']+ft['offset'].sum(0)
                whole=ft['edge'].sum(0)
                marginal=float(same[b]-same[a])
                interaction=float(whole[b]-whole[a])
                assert marginal<=1.e-10 and marginal+interaction>=-1.e-10
                hold.append(dict(cohort=cohort,frame=entry['case'],source=src,target=entry['target'],
                    old_iou=before,new_iou=after,delta_iou=after-before,
                    subject_change=describe_owners(*token[:,0]),reference_change=describe_owners(*token[:,1]),
                    prior_subject_right=int(token[0,0]==pair[0]) if pair[0]>=0 else '',
                    exchange_subject_right=int(token[1,0]==pair[0]) if pair[0]>=0 else '',
                    prior_reference_right=int(token[0,1]==pair[1]) if pair[1]>=0 else '',
                    exchange_reference_right=int(token[1,1]==pair[1]) if pair[1]>=0 else '',
                    same_evidence_delta=marginal,interaction_delta=interaction,combined_delta=marginal+interaction))
    assert old_gold_count==exchange_gold_count==3
    assert len(hold)==7 and sum(r['delta_iou']>0 for r in hold)==5
    out=cert.parent/'changed_cases.csv'
    csv_atomic(out,hold)
    atomic_json(cert.parent/'ATTRIBUTION_AUDIT.json',dict(nonconstant=7,gained=5,worsened=2,
                subject_identity_transition_counts=dict(Counter(r['subject_change'] for r in hold)),
                reference_identity_transition_counts=dict(Counter(r['reference_change'] for r in hold)),
                both_correct_same_evidence=old_gold_count,both_correct_exchange=exchange_gold_count,
                max_wrong_pair_dominated_by_interaction=max(r['interaction_delta'] for r in hold),
                rows_sha256=sha(out),protected_data_access=False))
    print('Role exchange attribution, all seven changed-IoU cases',flush=True)


if __name__=='__main__':
    main()
