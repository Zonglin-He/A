"""Retrospective within-description rank diagnostic on all sealed four-slot observations."""
from collections import defaultdict
from pathlib import Path
import csv
import gzip
import numpy as np
from paths_shape import HERE, OUT, read, save, digest
from score_roles import csv_file

MARGIN=HERE.parent/'3eed_v1_subject_margin_attribution_20261120'
W=MARGIN/'run_PRIVATE'


def run():
    seal=read(W/'SEALED.json');pin=digest(W/'SEALED.json')
    assert seal['original_four_and_winners_checked']==18620 and not seal['FINAL_access']
    records=[]; summary=[];scores=[]; reconstruction=0.
    for direction,ch in seal['directions'].items():
        folder=W/direction
        assert digest(folder/'SEALED.json')==ch
        path=W/('scored_'+direction+'.npz')
        proof=read(path.with_suffix('.json'))
        assert proof['file_sha256']==digest(path) and proof['prediction_seal']==pin
        with np.load(path) as z:pool=z['pool_iou'].copy()
        counters=defaultdict(list);stats=defaultdict(list);collected=[]
        seen=set()
        for fname,h in read(folder/'SEALED.json')['files'].items():
            file=folder/fname;assert digest(file)==h
            with np.load(file) as z:
                a=z['prior'].astype(float);u=z['unary'].astype(float);r=z['rel'].astype(float)
                score=z['final'].astype(float);which=z['choices'][:,0];act=z['active'];ids=z['ordinals']
                assert not (set(ids)&seen);seen.update(ids)
                np.testing.assert_equal(score.argmax(1),which)
                reconstruction=max(reconstruction,float(abs(score-a-u-r).max()))
                iq=pool[ids[:,None],z['slots']]
                mass=z['support'][:,1,:4,9].astype(float)
                centered=-z['support'][:,1,:4,10].astype(float)
                feats=dict(prior=a,unary=u,relation=r,visible_mass=mass,small_centroid_offset=centered)
                for t in (.25,.5):
                    good=iq>t
                    rows=np.arange(len(a))
                    hit=good[rows,which];prior_hit=good[rows,a.argmax(1)]
                    valid=good.any(1)&~good.all(1)
                    best=np.max(np.where(good,a,-np.inf),axis=1)
                    locked=~hit & good.any(1) & (a.max(1)-best >= (1.5+act.astype(float)))
                    masks=np.stack((hit,~(pool[ids]>t).any(1),(pool[ids]>t).any(1)&~good.any(1),locked,
                                    ~hit&good.any(1)&~locked))
                    assert np.all(masks.sum(0)==1)
                    collected.append((t,masks.sum(1)))
                    for nm,f in feats.items():
                        # An equal score contributes half credit. Each eligible sentence contributes equally.
                        diff=f[:,:,None]-f[:,None,:]
                        pairs=good[:,:,None]&~good[:,None,:]
                        wins=((diff>0)+.5*(diff==0))*pairs
                        auc=np.divide(wins.sum((1,2)),pairs.sum((1,2)),out=np.zeros(len(a)),where=valid)
                        for stratum,mask in [('all_mixed',valid),('prior_correct',valid&prior_hit),('prior_wrong',valid&~prior_hit)]:
                            counters[t,nm,stratum].extend(auc[mask].tolist())
                        bests=f==f.max(1,keepdims=True)
                        # Tied extrema default to the best existing prior then slot order.
                        pick=np.where(bests,a,-np.inf).argmax(1)
                        selected=good[rows,pick]
                        stats[t,nm].append((len(selected),int(selected.sum()),int((selected&~hit).sum()),int((~selected&hit).sum())))
        assert len(seen)==len(pool)
        for t in (.25,.5):
            v=np.sum([a for tt,a in collected if tt==t],axis=0)
            records.append(dict(direction=direction,threshold=t,n=len(pool),**dict(zip(
                ['hit','none_in_A256','only_outside_four','blocked','reachable_miss'],map(int,v)))))
        for (t,nm,cohort),u in counters.items():
            summary.append(dict(direction=direction,threshold=t,measure=nm,stratum=cohort,eligible=len(u),mean_auc=float(np.mean(u))))
        for (t,nm),ls in stats.items():
            n,h,p,q=map(int,np.sum(ls,axis=0))
            scores.append(dict(direction=direction,threshold=t,measure=nm,n=n,hits=h,pct=100*h/n,rescue=p,harm=q))
    with (MARGIN/'results/ordinary_attribution.csv').open() as fh:
        stored={(r['direction'],float(r['threshold'])):r for r in csv.DictReader(fh)}
    for v in records:
        raw=stored[v['direction'],v['threshold']]
        assert all(v[k]==int(raw[k]) for k in v if k not in ('direction','threshold'))
    assert reconstruction<1.e-6
    csv_file(OUT/'margin_counts.csv',records);csv_file(OUT/'support_rank.csv',summary);csv_file(OUT/'support_direct.csv',scores)
    save(OUT/'margin_verification.json',dict(n=18620,slots=74480,directions=6,sealed_predictions=pin,
            max_score_reconstruction_difference=reconstruction,all_gaps_and_denominators_reproduced=True,
            matching_archived_counts=len(records),source_training=0,target_updates=0,FINAL_access=False))

if __name__=='__main__':
    run()
