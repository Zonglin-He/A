"""After sealing, compare the selected physical subjects for every beyond-A256 rescue."""
import csv
import gzip
import numpy as np
from paths_tree import ARMS, OUT, digest, read
from full_roster import FULL
from shared_paths import anonymous
from scope183 import ordinary, frame
from terrain_followup import boxes_in_model
from common_lex import sample
from endpoint_score import arrays_to_owners, write_gz
from score_roles import inventory, csv_file


def owner_state(owners, arm):
    indices=[int(owners[rule][0][arm]) for rule in ('unique_iou','seed_ownership')]
    if indices[0]==indices[1] and indices[0]>=0:
        return 'target' if indices[0]==0 else 'distractor'
    return 'unknown'


def trace():
    sealed=FULL/'predictions'
    assert digest(sealed/'SEALED.json')==read(OUT/'full_audit.json')['predictions']
    with gzip.open(OUT/'full_geometry_rows.csv.gz','rt') as f:
        changed=[r for r in csv.DictReader(f) if r['arm']=='shared_cut' and int(r['verified_a256_rescue'])]
    roster={(target,anonymous((target,i))):(i,v) for target,rows in ordinary().items() for i,v in enumerate(rows)}
    results=[]
    for r in changed:
        src,tgt=r['source'],r['target']
        i,row=roster[tgt,r['case']]
        folder=sealed/(src+'_to_'+tgt)
        path=folder/(str(i)+'.npz')
        assert read(folder/'SEALED.json')['files'][str(i)]==digest(path)
        with np.load(path) as z:
            boxes=[boxes_in_model(z[k],tgt,row['suffix']) for k in ('completed','raw')]
        truth,_=inventory(row['metapath'],tgt,row['suffix'],row['object_index'])
        points=np.unique(sample(frame(row)).astype(float),axis=0)
        owners=arrays_to_owners(boxes,truth,points)
        verdicts={arm:owner_state(owners,j) for j,arm in enumerate(ARMS)}
        assert verdicts['shared_cut']=='target'
        result={k:r[k] for k in ('case','source','target','recording','frame','original','completed',
                            'raw_a256_and_pca','phrase_pools','best_measured_physical','verified_all_native_rescue')}
        results.append(dict(**result,**verdicts))
    write_gz(OUT/'full_beyond_pool_identity_rows.csv.gz',results)
    summary=[]
    for scope,rows in (('all_a256',results),('all_native',[x for x in results if int(x['verified_all_native_rescue'])])):
        for arm in ('v1','independent','wrong_reference_binding','image_center','separate_occupancy','pair_relation'):
            summary.append(dict(ceiling=scope,condition=arm,n=len(rows),recordings=len({x['recording'] for x in rows}),
                **{k:sum(x[arm]==k for x in rows) for k in ('target','distractor','unknown')},mixed=0))
        arms=('independent','wrong_reference_binding')
        summary.append(dict(ceiling=scope,condition='both_independent_and_mismatch',n=len(rows),
            recordings=len({x['recording'] for x in rows}),target=sum(all(x[a]=='target' for a in arms) for x in rows),
            distractor=sum(all(x[a]=='distractor' for a in arms) for x in rows),
            unknown=sum(any(x[a]=='unknown' for a in arms) for x in rows),
            mixed=sum({x[a] for a in arms}=={'target','distractor'} for x in rows)))
    csv_file(OUT/'full_beyond_pool_identity.csv',summary)


if __name__=='__main__':
    trace()
