"""Recompute complete macro results directly from exported case rows, without the summarizer."""
import csv
import gzip
from collections import defaultdict
import numpy as np
from vote_paths import *

def stream(path):
    with (gzip.open(path,'rt') if path.suffix=='.gz' else path.open()) as f:
        yield from csv.DictReader(f)

def main():
    seal=read(WORK/'SEALED.json'); assert not seal['labels_used']
    per=defaultdict(dict);rows=defaultdict(dict);excluded=0
    for src in SOURCES:
        file=WORK/'votes'/src/'SEALED.json';v=read(file)
        assert v['original_query_max_error']<1e-4 and v['extent_max_error']<1e-4
        source=read(WORK/(src+'_out.json'));assert source['capture']==digest(file)
        assert all(source['code'][f]==digest(HERE/f) for f in source['code'])
        for k,h in v['files'].items():assert digest(file.parent/(k+'.npz'))==h
        for x in stream(OUT/(src+'_task_rows.csv.gz')):
            if x['panel']!='full':continue
            case=x['case'];fallback=x['arm']=='all_fallback'
            excluded+=int(fallback)
            scores=[float(x[g]) for g in ('raw','anchored','fit','null_fit')]
            for arm in ARMS if fallback else [x['arm']]:
                u=per[src,x['target']].setdefault(case,{})
                assert arm not in u;u[arm]=scores
    assert len(per)==6 and sum(map(len,per.values()))==18620 and excluded==17326
    bydir={}
    for (s,t),items in per.items():
        assert all(set(x)==set(ARMS) for x in items.values())
        mat=np.array([[x[a] for a in ARMS] for x in items.values()])
        bydir[s+'_to_'+t]=np.stack([100*f for f in ((mat>.25).mean(0),(mat>.5).mean(0),mat.mean(0))],axis=-1)
    bydir['macro']=np.mean(list(bydir.values()),axis=0)
    peak=0.;checks=0
    for row in stream(OUT/'metrics.csv'):
        if row['panel']!='full':continue
        g=('raw','anchored','fit','null_fit').index(row['geometry'])
        measured=bydir[row['direction']][ARMS.index(row['arm']),g]
        error=abs(measured-np.array([float(row[q]) for q in ('acc25','acc50','miou')]))
        peak=max(peak,float(error.max()));checks+=len(error)
    assert checks==7*len(ARMS)*4*3 and peak<1e-9
    raw=np.vstack([np.array([[x[a] for a in ARMS] for x in items.values()]) for items in per.values()])
    allpairs=0
    for row in stream(OUT/'paired.csv'):
        if row['panel']!='full' or row['direction']!='macro':continue
        g=('raw','anchored','fit','null_fit')
        a=raw[:,ARMS.index(row['arm']),g.index(row['geometry'])]
        b=raw[:,ARMS.index(row['control']),g.index(row['control_geometry'])]
        for t in (25,50):
            assert ((a>t/100)&(b<=t/100)).sum()==int(row['rescue'+str(t)])
            assert ((b>t/100)&(a<=t/100)).sum()==int(row['harm'+str(t)])
        allpairs+=1
    prev=[r for r in stream(old.OUT/'full_metrics.csv') if r['subset']=='all' and r['direction']=='macro' and r['geometry']=='completed']
    for arm,geom,foreign in [('tree',1,'shared_cut'),('v1',0,'v1_raw'),('v1',1,'v1')]:
        zz=next(r for r in prev if r['arm']==foreign)
        np.testing.assert_allclose(bydir['macro'][ARMS.index(arm),geom],[float(zz[k]) for k in ('acc25','acc50','miou')],atol=1e-8,rtol=0)
    save(OUT/'independent_audit.json',dict(full_records=18620,same_untouched_fallback=excluded,
         source_observation_frames=sum(len(read(WORK/'votes'/s/'SEALED.json')['files']) for s in SOURCES),
         direction_macro_and_arm_scalars_checked=checks,paired_contrasts_checked=allpairs,largest_difference=peak,
         verified_predecessor_full_aggregate=True,prediction_implementation=seal['code'],
         frozen_registry=pins()['frozen_registry'],FINAL_access=False))
if __name__=='__main__':main()

