"""Post-seal physical upper bounds, including whether both distinct utterances are reachable."""
import numpy as np
import pandas as pd

from token_paths import ARMS, OUT, WORK, digest, prior, read, roster, row_id
from endpoint_score import arrays_to_owners
from role_cut import layout,shared,structure


def run():
    receipt=read(WORK/'SEALED.json')
    assert not receipt['labels']
    labels=read(prior.previous.EXTRA/'labels.json')
    rows,_=roster()
    original=read(prior.WORK/'PLANS.json')['files']
    records=[]
    for row in rows:
        fid=row['frame'];e=labels[row['uid']];ts=e['subject'];tr=e['reference']
        path=prior.WORK/'plans'/(fid+'.npz')
        assert digest(path)==original[fid]
        with np.load(path) as f:plan=dict(f)
        pred=WORK/'predictions'/(row_id(row)+'.npz')
        assert digest(pred)==receipt['files'][row_id(row)]
        with np.load(pred) as z:choices=z['nodes']
        with np.load(row['prior']) as z:seeds=z['seeds']
        with np.load(row['role']) as z:
            side,depth,near=int(z['side']),int(z['depth']),bool(z['immediate'])
        a,b=row['words']
        n=len(plan['p'])
        _,K=structure(plan['parent'],plan['stop'],plan['size'])
        _,_,vis,gap=layout(plan['view'],side,'physical')
        allowed=(K==0)&(gap>0)&vis[:,None]&vis[None,:]
        j=np.flatnonzero(plan['row']==row_id(row)).item()
        if n:
            joint,_=shared(plan['parent'],K==0,plan['stop'],plan['p'],(a,b),plan['view'],side,depth,near,'physical')
            supported=joint*plan['us'][j,:,None]*plan['ur'][j,None,:]>0
        else:supported=np.empty((0,0),bool)
        assert not (supported&~allowed).any()
        coords=[plan['boxes'],plan['completed'][:,a],plan['completed'][:,b]]
        ow=arrays_to_owners(coords,np.array(e['truth']),seeds) if n else {r:[np.array([],int)]*3 for r in ('seed_ownership','unique_iou')}
        ow['both_tests']=[np.where(x==y,x,-1) for x,y in zip(ow['unique_iou'],ow['seed_ownership'])]
        for rule,v in ow.items():
            for geo,(sids,rids) in [('raw',(v[0],v[0])),('completed',(v[1],v[2]))]:
                sh=(sids==ts)&vis;rh=(rids==tr)&vis
                record=dict(source=row['source'],pair=e['pair'],half=e['half'],frame=e['frame'],recording=e['recording'],
                            geometry=geo,rule=rule,any_subject=int(sh.any()),any_reference=int(rh.any()),
                            both_present=int(sh.any() and rh.any()),both_legal=int((allowed&sh[:,None]&rh).any()),
                            both_nonzero=int((supported&sh[:,None]&rh).any()),
                            subject_legal=int((allowed&sh[:,None]).any()))
                for arm in ('tree','token_interaction','wrong_support'):
                    s,r=choices[ARMS.index(arm)]
                    record[arm+'_subject']=int(s>=0 and sh[s])
                    record[arm+'_pair']=int(s>=0 and r>=0 and sh[s] and rh[r])
                records.append(record)
    data=pd.DataFrame(records)
    data.to_csv(OUT/'coverage_rows.csv.gz',index=False)
    cols=data.columns[7:]
    g=data.groupby(['geometry','rule','source','pair'])
    assert g.size().eq(2).all()
    pair=g[list(cols)].min().groupby(['geometry','rule']).sum()
    assert (pair.both_nonzero<=pair.both_legal).all() and (pair.both_legal<=pair.both_present).all()
    pair['pairs']=64
    pair.to_csv(OUT/'coverage_pairs.csv')
    print(pair.loc['completed'].to_string())


if __name__=='__main__':
    run()
