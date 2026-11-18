"""Apply the same coverage definitions to the independently reviewed reference labels."""
from collections import Counter
import numpy as np

from review166 import PANEL,TRUTH,evidence,open_block,verified_prediction_inputs
from scope166 import BLOCK,HERE,ORDER,W165,atomic_json,read,sha
from labels166 import mask_owners,occupied_owners,physical_labels


def main():
    rows,_=verified_prediction_inputs()
    panel=PANEL/'REGISTRY_PRIVATE.json'
    cases,labels=read(panel),read(TRUTH)
    assert sha(panel)==read(HERE/'REVIEW_PANEL_SEAL.json')['digest']
    counts={k:Counter() for k in ('a256','r88','h0_pool','h0_eligible','h0_beam')}
    n=0
    for case,review in zip(cases,labels,strict=True):
        assert case['case']==review['case']
        if review['status']!='verified_boxes':
            continue
        target,ii,edge=(case[x] for x in ('target','ordinal','edge_number'))
        ref=review['reference']
        for source,t in ORDER:
            if t!=target:
                continue
            row=rows[t][ii]
            b,_,(role,side)=evidence(source,t,ii,edge)
            _,objects,seeds,truth=physical_labels(row,b['seeds'])
            data=open_block(source,t,ii//BLOCK)
            j=ii%BLOCK
            native=W165/'ordinary'/f'{source}_to_{t}'/f'{ii//BLOCK:04d}.npz'
            assert sha(native)==str(data['parent_block_sha256'])
            with np.load(native,allow_pickle=False) as raw:
                boxes=raw['raw'][j]
            axis=(np.abs(seeds[None]-boxes[:,None,:3])<=boxes[:,None,3:]/2).all(-1)
            a=mask_owners(axis,truth,len(objects))
            h=occupied_owners(data['starts'][j],seeds,truth,len(objects))
            routes=data['routes'][j]
            eu,beam=data['eligibility'][j],data['in_beam'][j]
            stage=dict(a256=(a,a),r88=tuple(a[routes[x,routes[x]>=0]] for x in (0,role)),
                       h0_pool=(h[0],h[role]),h0_eligible=tuple(h[x,eu[x]] for x in (0,role)),
                       h0_beam=tuple(h[x,beam[x]] for x in (0,role)))
            for step,(first,second) in stage.items():
                a1,a2=int((first==0).any()),int((second==ref).any())
                counts[step].update(subject=a1,reference=a2,both=a1*a2)
            n+=1
    assert n==read(HERE/'VISUAL_REVIEW_AUDIT.json')['verified_direction_edges']
    atomic_json(HERE/'VISUAL_COVERAGE_AUDIT.json',dict(visible_identity_edges=n,
                coverage={k:dict(v) for k,v in counts.items()},post_prediction_blind_labels=True,FINAL_access=False))


if __name__=='__main__':
    main()
