"""Independent scalar count of threshold crossings and paired physical identities."""
import csv
import gzip
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from token_paths import ARMS, OUT, WORK, digest, read, roster, row_id, save


def read_rows(name):
    with gzip.open(OUT/(name+'_rows.csv.gz'),'rt') as f:
        return list(csv.DictReader(f))


def paired(rows,key,ours,peer):
    table=defaultdict(dict)
    for r in rows:
        table[tuple(r[k] for k in key)][r['arm']]=r
    assert all(ours in v and peer in v for v in table.values())
    return table


def summarize():
    rows,_=roster()
    seal=read(WORK/'SEALED.json')
    assert len(rows)==len(seal['files'])==128
    for r in rows:
        rid=row_id(r)
        assert digest(WORK/'predictions'/(rid+'.npz'))==seal['files'][rid]
    pairkey=('source','pair','frame','recording')
    tasks=read_rows('task');inventory=read_rows('inventory');physical=read_rows('physical_identity')
    coverage=read_rows('coverage')
    output=[]
    for panel,table in [('measured',physical),('known_boxes',inventory)]:
        if panel=='measured':
            table=[r for r in table if r['geometry']=='completed' and r['rule']=='unique_iou']
        for peer in ('tree','wrong_support','contrastive_interaction','independent','wrong_reference'):
            groups=defaultdict(list)
            for key,vs in paired(table,(*pairkey,'half'),'token_interaction',peer).items():
                groups[key[:-1]].append(vs)
            assert len(groups)==64 and all(len(v)==2 for v in groups.values())
            for field in ('subject','reference','both'):
                change=[]
                for k,v in groups.items():
                    new=all(int(w['token_interaction'][field]) for w in v)
                    base=all(int(w[peer][field]) for w in v)
                    change.append((k[-1],int(new)-int(base),int(new),int(base)))
                totals=defaultdict(lambda:[0,0])
                for rec,d,_,_ in change:
                    totals[rec][0]+=d;totals[rec][1]+=1
                clusters=np.array([totals[k] for k in sorted(totals)])
                draw=np.random.default_rng(20535).integers(len(clusters),size=(5000,len(clusters)))
                sums=clusters[draw].sum(1);interval=np.quantile(100*sums[:,0]/sums[:,1],[.025,.975])
                output.append(dict(panel=panel,endpoint=field,peer=peer,pairs=64,recordings=len(clusters),
                      main=sum(x[2] for x in change),control=sum(x[3] for x in change),
                      rescue=sum(x[1]==1 for x in change),harm=sum(x[1]==-1 for x in change),
                      ci_low=interval[0],ci_high=interval[1]))
    pd.DataFrame(output).to_csv(OUT/'paired_identity.csv',index=False)
    threshold_rows=[]
    for geo in ('raw','completed'):
        tab=[r for r in tasks if r['geometry']==geo]
        for peer in ARMS:
            for src in sorted({r['source'] for r in tab}):
                table=paired([r for r in tab if r['source']==src],(*pairkey,'half'),'token_interaction',peer)
                for cut in (.25,.5):
                    counts=Counter()
                    for cell in table.values():
                        a,b=map(lambda s:float(cell[s]['iou'])>cut,('token_interaction',peer))
                        counts[a,b]+=1
                    threshold_rows.append(dict(geometry=geo,source=src,threshold=cut,peer=peer,n=sum(counts.values()),
                        rescue=counts[True,False],harm=counts[False,True],unchanged=counts[True,True]+counts[False,False]))
    pd.DataFrame(threshold_rows).to_csv(OUT/'paired_task.csv',index=False)
    saved=pd.read_csv(OUT/'coverage_pairs.csv').set_index(['geometry','rule'])
    for key,part in pd.DataFrame(coverage).groupby(['geometry','rule']):
        for field in saved.columns.drop('pairs'):
            val=part.groupby(['source','pair'])[field].agg(lambda x:all(int(v) for v in x)).sum()
            assert int(val)==saved.loc[key,field]
    old=pd.read_csv(OUT/'physical_pairs.csv').set_index(['arm','geometry','rule'])
    inv=pd.read_csv(OUT/'inventory_pairs.csv').set_index('arm')
    for v in output:
        if v['panel']=='known_boxes':
            expected=inv.loc[v['peer'],v['endpoint']]
        else:
            expected=old.loc[(v['peer'],'completed','unique_iou'),v['endpoint']]
        assert expected==v['control'] and v['rescue']-v['harm']==v['main']-v['control']
    split=[]
    for (a,same),part in pd.DataFrame(inventory).groupby(['arm','same_noun']):
        for role in ('subject','reference'):
            scores=part[role+'_error'].value_counts()
            split.append(dict(arm=a,same_role_word=int(same),role=role,n=len(part),
                **{k:int(scores.get(k,0)) for k in ('correct','same_class_other_object','wrong_class','no_choice')}))
    pd.DataFrame(split).to_csv(OUT/'competitors_by_word.csv',index=False)
    with np.errstate(over='raise',invalid='raise'):
        rng=np.random.default_rng(1781);scores=rng.normal(size=(32,6))
        object_bias=rng.normal(size=(32,1));noun_bias=rng.normal(size=(1,6))
        from decide_token import double_center
        np.testing.assert_allclose(double_center(scores+object_bias+noun_bias),double_center(scores),atol=1e-12)
        np.testing.assert_allclose(double_center(object_bias+noun_bias),0,atol=1e-12)
    save(OUT/'audit.json',dict(seal=digest(WORK/'SEALED.json'),directions_in_panel=2,
        captures=64,utterances=128,physical_ids_checked=len(physical),
        paired_role_contrasts=len(output),threshold_contrasts=len(threshold_rows),
        coverage_cells=int(saved.shape[0]*(saved.shape[1]-1)),
        additive_nuisance_test=True,final_access=False,source_training=0,target_updates=0))


if __name__=='__main__':
    summarize()
