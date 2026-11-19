"""Post-seal diagnostic using the existing R174 reference-identity panel."""
from collections import Counter
import sys

import numpy as np

from scope178 import (ARMS,ORIGINAL,ORDER,PRIVATE,PRIOR,RESULTS,json_atomic,
                      ordinary_rows,read,sha)
from scope174 import FACTOR
from predict178 import OUTPUT
from score174 import write_csv


def reference_panel():
    sys.path.append(str(FACTOR))
    from reference_truth import assigned_owners,library,role_phrase,truth_pins,verified_refs

    receipt = read(PRIOR.parent/'results/target_reference_intervention_receipt.json')
    assert not receipt['FINAL_access'] and receipt['truth_pins']==truth_pins()
    assert receipt['prediction']==sha(ORIGINAL/'PREDICTION_SEAL.json')
    _,inventory = library()
    verified = verified_refs()
    results = []
    unmatched = 0
    for source,target in ORDER:
        route = f'{source}_to_{target}'
        old = ORIGINAL/route
        new = OUTPUT/route
        before = read(old/'SEAL.json')
        after = read(new/'SEAL.json')
        for (platform,ordinal),options in sorted(verified.items()):
            if platform!=target:
                continue
            row = ordinary_rows(target)[ordinal]
            phrase = role_phrase(row)
            candidates = {(ref,n) for label,ref,n,_ in options if label and label==phrase}
            if len(candidates)!=1:
                unmatched += 1
                continue
            ref,count = candidates.pop()
            names,boxes = inventory(row['metapath'],target,int(row['suffix'] or 0),row['object_index'])
            assert 0 < ref < count and len(names)==count
            shard = sorted(after['files'])[ordinal//128]
            assert sha(new/shard)==after['files'][shard] and sha(old/shard)==before['files'][shard]
            with np.load(new/shard,allow_pickle=False) as a,np.load(old/shard,allow_pickle=False) as b:
                j = int(np.searchsorted(a['ordinals'],ordinal))
                assert a['ordinals'][j]==b['ordinals'][j]==ordinal
                choices = a['choices'][j]
                subjects = b['subject_slots'][j,choices[:,0]]
                references = b['reference_slots'][j,choices[:,1]]
                owner = assigned_owners(b['seeds'][j],np.concatenate((subjects,references)),boxes)
                s,r = owner.reshape(2,len(ARMS))
                relation = bool(b['direct_left_right'][j])
            for i,arm in enumerate(ARMS):
                results.append(dict(direction=route,ordinal=ordinal,variant=arm,
                                    direct=relation,subject_correct=bool(s[i]==0),
                                    subject_unknown=bool(s[i]<0),reference_correct=bool(r[i]==ref),
                                    reference_unknown=bool(r[i]<0),both_correct=bool(s[i]==0 and r[i]==ref)))
    json_atomic(PRIVATE/'reviewed_identity_rows.json',results)
    table = []
    for scope,test in (('verified',lambda r:True),('verified_direct_left_right',lambda r:r['direct'])):
        for arm in ARMS:
            found = [r for r in results if r['variant']==arm and test(r)]
            count = Counter(k for r in found for k in ('subject_correct','subject_unknown',
                            'reference_correct','reference_unknown','both_correct') if r[k])
            table.append(dict(subset=scope,variant=arm,n=len(found),
                              subject_correct=count['subject_correct'],
                              subject_wrong=len(found)-count['subject_correct']-count['subject_unknown'],
                              subject_unknown=count['subject_unknown'],
                              reference_correct=count['reference_correct'],
                              reference_wrong=len(found)-count['reference_correct']-count['reference_unknown'],
                              reference_unknown=count['reference_unknown'],
                              both_correct=count['both_correct'],
                              role_join_unknown=unmatched if scope=='verified' else ''))
    write_csv(RESULTS/'reviewed_reference_identity.csv',table)
    return dict(reviewed_direction_records=len(verified)*2,
                exact_phrase_direction_records=len(results)//len(ARMS),unmatched=unmatched,
                prior_truth_receipt=sha(PRIOR.parent/'results/target_reference_intervention_receipt.json'),
                private_outcomes=sha(PRIVATE/'reviewed_identity_rows.json'))
