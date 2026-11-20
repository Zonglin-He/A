"""Read only same-platform GT after the complete prediction seal and audit every threshold crossing."""
from collections import defaultdict
import csv
import gzip
import hashlib
import sys

import numpy as np

from native_pool import HERE,MASKS,OUT,PLATFORMS,WORK,atomic,digest,legacy,pinned_rows,read
sys.path.insert(0,str(HERE.parent/'3eed_v1_camera_word_replacement_20261120'))
from calibrate import csv_file


def code(s):
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def interval(groups):
    rng=np.random.default_rng(69357)
    diffs=[]
    for src in PLATFORMS:
        rows=groups[src]
        by=defaultdict(list)
        for r in rows:
            by[r['recording']].append(int(r['iou']>.5)-int(r['frozen_iou']>.5))
        v=np.array([[sum(x),len(x)] for x in by.values()])
        sampled=v[rng.integers(len(v),size=(3000,len(v)))].sum(1)
        diffs.append(sampled[:,0]/sampled[:,1])
    return list(map(float,np.percentile(np.mean(diffs,axis=0)*100,[2.5,97.5])))


def summarize(rows):
    iou=np.array([r['iou'] for r in rows])
    base=np.array([r['frozen_iou'] for r in rows])
    ok=np.array([r['opportunity50'] for r in rows],bool)
    margin=[r['margin50']-r['frozen_margin50'] for r in rows if r['opportunity50']]
    changed=sum(r['switch'] for r in rows)
    return dict(n=len(rows),acc25=float(100*np.mean(iou>.25)),acc50=float(100*np.mean(iou>.50)),miou=float(100*iou.mean()),
        changed=changed,rescue25=int(((iou>.25)&(base<=.25)).sum()),harm25=int(((iou<=.25)&(base>.25)).sum()),
        rescue50=int(((iou>.50)&(base<=.50)).sum()),harm50=int(((iou<=.50)&(base>.50)).sum()),
        from_pass50_to_zero=int(((base>.50)&(iou==0)).sum()),misses_rescuable50=int(ok.sum()),
        improved_iou=int((iou>base+1e-12).sum()),worse_iou=int((iou<base-1e-12).sum()),
        delta_margin50=float(np.mean(margin)) if margin else None)


def main():
    dest=WORK/'deletion'
    seal=read(dest/'SEALED.json')
    assert not any(seal[k] for k in ('grounding_labels_requested','target_data','FINAL'))
    assert len(seal['files'])==384 and seal['lock']==digest(dest/'LOCK.json')
    assert all(digest(HERE/f)==h for f,h in read(dest/'LOCK.json')['code'].items())
    old=read(legacy.PRIVATE/'source/LOCK.json')['roster']
    records=[]
    histogram=[]
    for src in PLATFORMS:
        _,rows=pinned_rows(src)
        bank=legacy.sealed_arrays(legacy.R181,'SEALED.json','source_scores',src,['ious','chosen']) 
        old_i=set(old[src])
        assert len(old_i)==64
        for i,(r,gt) in enumerate(zip(rows,bank['ious'],strict=True)):
            path=dest/f'{src}_{i:04d}.npz'
            assert digest(path)==seal['files'][path.name]
            with np.load(path,allow_pickle=False) as f:
                scores,hist,loss_mse=f['scores'],f['hist'],f['loss_sum']
                inc,sizes,timing=int(f['v1']),f['scope_sizes'],f['seconds']
            keys=dict(source=src,frame=code(src+'|'+r['point_sha256']),recording=code(src+'|'+r['recording']),
                      exposed192=int(i in old_i))
            for j,scope in enumerate(('scene','K16','winner')):
                for n in range(1,17):
                    histogram.append(dict(**keys,region=scope,n=n,groups=int(hist[j,0,n]),center_present=int(hist[j,1,n])))
            base=int(scores[0].argmax())
            assert base==int(bank['chosen'][i,0])
            for j,arm in enumerate(('v1',*MASKS)):
                mode=max(j-1,0)
                sc=scores[mode]
                k=inc if arm=='v1' else int(sc.argmax())
                row=dict(**keys,arm=arm,iou=float(gt[k]),frozen_iou=float(gt[base]),best=float(gt.max()),
                         switch=int(k!=base),opportunity50=int(gt[base]<=.5<gt.max()),winner_centered_full=int(hist[2,1,16]),
                         all_centered_full=int(hist[0,1,16]),all_sparse=int(hist[0,1,2:16].sum()),native_slate_points=sizes[1],
                         max_abs_score_change=float(abs(sc-scores[0]).max()),score_change_span=float(np.ptp(sc-scores[0])),
                         original_top2_margin=float(scores[0,base]-np.partition(scores[0],-2)[-2]),
                         seconds=float(timing[mode]))
                for t in (25,50):
                    good=gt>t/100
                    row['margin'+str(t)]=float(sc[good].max()-sc[base]) if good.any() else None
                    row['frozen_margin'+str(t)]=float(scores[0,good].max()-scores[0,base]) if good.any() else None
                for l,scope in enumerate(('scene','K16','winner')):
                    row[scope+'_sse']=float(loss_mse[mode,l])
                    row[scope+'_groups']=int(sizes[l])
                    row[scope+'_eligible']=int(hist[l,1,16])
                records.append(row)
    for name,table in (('deletion_all_rows',records),('source_neighborhoods',histogram)):
        with gzip.open(OUT/(name+'.csv.gz'),'wt',newline='') as fp:
            w=csv.DictWriter(fp,fieldnames=table[0].keys())
            w.writeheader()
            w.writerows(table)
    metrics=[]
    decisions={}
    for arm in ('v1',*MASKS):
        g={src:[r for r in records if r['source']==src and r['arm']==arm] for src in PLATFORMS}
        sums=[summarize(group) for group in g.values()]
        glob=summarize([r for rr in g.values() for r in rr])
        for m in ('acc25','acc50','miou'):
            glob[m]=float(np.mean([q[m] for q in sums]))
        lo,hi=interval(g)
        for src,m in [*zip(PLATFORMS,sums),('macro',glob)]:
            metrics.append(dict(source=src,arm=arm,**m,lower95_acc50_pp=lo if src=='macro' else None,
                                upper95_acc50_pp=hi if src=='macro' else None))
        decisions[arm]=glob
    csv_file(OUT/'deletion_metrics.csv',metrics)
    scopes=[]
    for src in PLATFORMS:
        for group in ('scene','K16','winner'):
            g=[v for v in histogram if v['source']==src and v['region']==group]
            scopes.append(dict(source=src,region=group,total=sum(v['groups'] for v in g),
                saturated_16=sum(v['groups'] for v in g if v['n']==16),
                saturated_16_with_center=sum(v['center_present'] for v in g if v['n']==16),
                native_eligible_2_to_15=sum(v['center_present'] for v in g if 2<=v['n']<=15),
                singleton=sum(v['groups'] for v in g if v['n']==1),
                center_absent=sum(v['groups']-v['center_present'] for v in g)))
    csv_file(OUT/'coverage.csv',scopes)
    advance=all(decisions[arm]['harm50']>0 for arm in ('n4_211','n4_947'))
    atomic(OUT/'deletion_decision.json',dict(advance_to_source_calibration=advance,
        rule='at least one pass50-to-miss in each of the two prespecified four-return deletions',
        counts=seal['counts'],exposed192_opportunities=sum(r['opportunity50'] for r in records if r['arm']=='native' and r['exposed192']),
        results=decisions,unchanged_native_verified=True,physical_duplicates_invariant=True,
        seconds=seal['seconds'],seal=digest(dest/'SEALED.json'),target_data=False,FINAL=False))
    print('advance to source calibration',advance,[(a,decisions[a]['rescue50'],decisions[a]['harm50']) for a in MASKS],flush=True)


if __name__=='__main__':
    main()
