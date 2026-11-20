"""Account for every frozen output and paired change, including each literal fallback."""
import numpy as np
import pandas as pd
from partition_paths import ARMS, GEOMS, OUT, SOURCES, read, save


def frames(kind):
    return pd.concat([pd.read_csv(f,keep_default_na=False) for f in sorted(OUT.glob('*_'+kind+'_rows.csv.gz'))],ignore_index=True)


def clustered(rows):
    """Resample each target's recordings together for its two foreign models."""
    arr=rows.groupby(['target','recording','source']).agg(n=('case','size'),change=('change','sum'))
    rng=np.random.default_rng(90173)
    values=[]
    for _,frame in arr.groupby(level='target'):
        n=frame.n.unstack('source').to_numpy()
        change=frame.change.unstack('source').to_numpy()
        assert np.isfinite(n).all()
        ids=rng.integers(len(n),size=(4000,len(n)))
        values.append((change[ids].sum(1)/n[ids].sum(1)).mean(1))
    return list(np.quantile(100*np.mean(values,axis=0),[.025,.975]))


def run():
    t=frames('task');fallback=t[t.arm=='all_fallback']
    t=pd.concat([t[t.arm!='all_fallback'],*(fallback.assign(arm=a) for a in ARMS)],ignore_index=True)
    keys=['panel','source','target','case']
    assert not t.duplicated(keys+['arm']).any()
    assert (t.groupby(keys).size()==len(ARMS)).all()
    full=t[t.panel=='full'];assert len(full)==len(ARMS)*18620
    scores=t.melt(id_vars=keys+['arm','eligible','emitted'],value_vars=GEOMS,var_name='geometry',value_name='iou')
    scores['acc25']=100*(scores.iou>.25);scores['acc50']=100*(scores.iou>.5);scores['miou']=100*scores.iou
    groups=scores.groupby(['panel','source','target','arm','geometry'])
    d=groups[['acc25','acc50','miou']].mean();d['records']=groups.size();d['emitted']=groups.emitted.sum()
    d.reset_index().to_csv(OUT/'direction.csv',index=False)
    group=d.groupby(level=['panel','arm','geometry'])
    macro=group[['acc25','acc50','miou']].mean();macro['records']=group.records.sum();macro['emitted']=group.emitted.sum()
    macro.to_csv(OUT/'macro.csv')
    indices=['source','target','case','recording']
    main=full[full.arm=='word_cut'].set_index(indices)
    pair=[]
    for arm in ARMS:
        control=full[full.arm==arm].set_index(indices).reindex(main.index)
        geometries={'word_cut':GEOMS,'v1':('measured','envelope')}.get(arm,('envelope',))
        for g in geometries:
            for threshold in (.25,.50):
                good=main.envelope>threshold;other=control[g]>threshold
                plan=pd.DataFrame(dict(change=good.astype(int)-other.astype(int),gain=good & ~other,loss=other & ~good)).reset_index()
                for (src,tgt),ss in plan.groupby(['source','target']):
                    pair.append(dict(arm=arm,geometry=g,threshold=threshold,source=src,target=tgt,n=len(ss),
                                     rescue=int(ss.gain.sum()),harm=int(ss.loss.sum()),diff_pp=100*ss.change.mean(),lo='',hi=''))
                lo,hi=clustered(plan)
                pair.append(dict(arm=arm,geometry=g,threshold=threshold,source='six',target='equal',n=len(plan),
                         rescue=int(plan.gain.sum()),harm=int(plan.loss.sum()),
                         diff_pp=100*plan.groupby(['source','target']).change.mean().mean(),lo=lo,hi=hi))
    pd.DataFrame(pair).to_csv(OUT/'paired.csv',index=False)
    ids=frames('identity')
    per=ids.groupby(['panel','geometry','arm','rule'])[['subject','reference','both']].agg('sum')
    per['known']=ids.groupby(['panel','geometry','arm','rule']).size();per.to_csv(OUT/'identity_summary.csv')
    d=ids[ids.panel.isin(['noun','fresh'])]
    grouping=['panel','source','pair','arm','geometry','rule']
    g=d.groupby(grouping);assert g.size().eq(2).all()
    p=g[['subject','reference','both']].min().reset_index()
    g=p.groupby(['panel','arm','geometry','rule'])
    endpoints=g[['subject','reference','both']].sum();endpoints['pairs']=g.size();endpoints.to_csv(OUT/'identity_pairs.csv')
    pivot=p.pivot(index=['panel','source','pair','geometry','rule'],columns='arm',values='both')
    swaps=[]
    for arm in ARMS:
        dif=pivot['word_cut']-pivot[arm]
        for (panel,geometry,rule),v in dif.groupby(level=['panel','geometry','rule']):
            swaps.append(dict(panel=panel,geometry=geometry,rule=rule,control=arm,rescue=int((v>0).sum()),harm=int((v<0).sum()),pairs=len(v)))
    pd.DataFrame(swaps).to_csv(OUT/'paired_identities.csv',index=False)
    c=frames('coverage');keys=['source','pair','arm','geometry'];g=c.groupby(keys);assert g.size().eq(2).all()
    v=g[['subject','reference','both','admissible']].min().reset_index();g=v.groupby(['arm','geometry'])
    covered=g[['subject','reference','both','admissible']].sum();covered['pairs']=g.size();covered.to_csv(OUT/'coverage_pairs.csv')
    changes=frames('geometry')
    changes['stable_rescue']=changes.stable*changes.rescue;changes['stable_harm']=changes.stable*changes.harm
    for a in ('independent','wrong_reference','wrong_marks','same_count','union','previous_tree','v1'):
        changes['beyond_vs_'+a]=changes.outside_a256*changes[a+'_wrong_instance']
    changes['both_controls_and_ref']=changes.beyond_vs_independent*changes.beyond_vs_wrong_reference*changes.ref_verified
    cols=['rescue','harm','stable_rescue','stable_harm','outside_a256','both_controls_and_ref']+[v for v in changes if v.startswith('beyond_vs_')]
    changes.groupby(['panel','arm','geometry','control'])[cols].sum().to_csv(OUT/'geometry_summary.csv')
    basic=t[t.arm=='word_cut']
    group=basic.groupby('panel')
    pop=group.agg(records=('case','size'),active=('eligible','sum'),emitted=('emitted','sum'),frames=('frame','nunique'),recordings=('recording','nunique'))
    pop.to_csv(OUT/'population.csv')
    save(OUT/'evaluation.json',dict(all_directions=6,ordinary=18620,
        max_score_chain_error=max(read(OUT/(s+'_score.json'))['iou_chain_error'] for s in SOURCES),
        max_prior_replay_error=max(read(OUT/(s+'_score.json'))['baseline_replay_error'] for s in SOURCES),target_updates=0,source_training=0,FINAL_access=False))


if __name__=='__main__':run()
