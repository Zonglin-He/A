"""Independently recompute ordinary macro scores from anonymous rows, and check all receipts."""
from collections import defaultdict,Counter
import csv
import gzip
import numpy as np
import pandas as pd
from local_paths import *
from method_local import ARMS,GEOMS
from predict_local import code
from fresh_inference import FRESH,pins

def raw(path):
    with (gzip.open(path,'rt') if path.suffix=='.gz' else path.open()) as handle:
        return list(csv.DictReader(handle))
def main():
    seal=read(WORK/'SEALED.json');assert seal['code']==code() and seal['upstream']==base() and not seal['labels_used']
    cap=read(WORK/'PLANS.json');frame_seen=set();objects=0;counts=Counter()
    for src in SOURCES:
        obs=read(WORK/'observed'/src/'SEALED.json')
        decision=read(WORK/(src+'_decisions.json'));assert not decision['labels_used'] and not obs['labels_used']
        assert not(frame_seen&obs['files'].keys());frame_seen.update(obs['files'])
        for fid,hh in obs['files'].items():
            observed=WORK/'observed'/src/(fid+'.npz')
            plan=WORK/'plans'/(fid+'.npz')
            assert digest(observed)==hh and digest(plan)==cap['files'][fid]
            with np.load(observed) as f,np.load(plan) as g:
                np.testing.assert_equal(f['ids'],g['crop_nodes']);n=len(f['ids']);objects+=n
                assert np.array_equal(f['support'][:,0],g['size'][f['ids']])
                equal=f['support'][:,2]==0
                counts['unchanged_null']+=int(equal.sum())
                counts['subsampled_larger_than_2048']+=int((f['support'][:,0]>2048).sum())
                observed_values=f['logit'];assert np.isfinite(observed_values).all()
                assert np.array_equal(observed_values[equal,0],observed_values[equal,1])
        for case,h in decision['files'].items():assert h==seal['files'][case] and h==digest(WORK/'predictions'/(case+'.npz'))
    assert objects==sum(cap['counts'].values()) and frame_seen==set(cap['files'])
    reference={(r['direction'],r['arm'],r['geometry']):r for r in raw(old.OUT/'full_metrics.csv') if r['subset']=='all'}
    got={(r['direction'],r['arm'],r['geometry']):r for r in raw(OUT/'metrics.csv') if r['panel']=='full'}
    sums=defaultdict(lambda:np.zeros(4));fallback=0;keys=set();eligible=set();panels=set();old_error=0.;max_error=0.
    for s in SOURCES:
        for r in raw(OUT/(s+'_task_rows.csv.gz')):
            tag=(s,r['target'],r['case'],r['panel']);panels.add(tag)
            if r['panel']!='full':continue
            direction=s+'_to_'+r['target']
            if r['arm']=='all_fallback':fallback+=1;arms=ARMS
            else:arms=(r['arm'],)
            keys.add(tag)
            if r['eligible']=='1':eligible.add(tag)
            for a in arms:
                for geo in GEOMS:
                    x=float(r[geo]);sums[direction,a,geo]+=[1,int(x>.25),int(x>.5),x]
    assert (fallback,len(eligible),len(keys),len(panels),len(seal['files']))==(17326,1294,18620,19952,2626)
    macros=defaultdict(list)
    for (direct,a,g),val in sums.items():
        score=100*val[1:]/val[0];macros[a,g].append(score)
        v=got[direct,a,g];other=[float(v[c]) for c in ('acc25','acc50','miou')]
        assert float(v['n'])==val[0];max_error=max(max_error,float(np.max(np.abs(score-other))))
        if a in ('tree','v1') and g in ('raw','anchored'):
            prev=reference[direct,'v1' if a=='v1' else 'shared_cut', 'measured' if g=='raw' else 'completed']
            old_error=max(old_error,float(np.max(np.abs(score-[float(prev[c]) for c in ('acc25','acc50','miou')]))))
    for (a,g),val in macros.items():
        assert len(val)==6;v=got['macro',a,g];max_error=max(max_error,float(np.max(np.abs(np.mean(val,axis=0)-[float(v[c]) for c in ('acc25','acc50','miou')]))))
    assert max_error<1e-8 and old_error<1e-8
    fseal=read(FRESH/'SEALED.json');assert fseal['pins']==pins() and fseal['n']==36 and not fseal['labels_read']
    for s in SOURCES:
        p=FRESH/s/'SEALED.json';assert digest(p)==fseal['sources'][s];v=read(p)
        for k,h in v['files'].items():assert h==digest(FRESH/s/(k+'.npz'))
    errs=[read(OUT/(s+'_score.json')) for s in SOURCES]
    save(OUT/'audit.json',dict(registry=digest(ROOT/'crossplatform/CURRENT_METHOD.json'),protocol=digest(SPEC),seal=digest(WORK/'SEALED.json'),
        ordinary=len(keys),eligible=len(eligible),fallback=fallback,known_or_additional_small=len(panels)-len(keys),
        captures=len(frame_seen),object_reads=objects,extra_new_frames=9,extra_new_direction_records=36,real_null_counts=dict(counts),
        independent_report_error=max_error,original_public_replay_error=old_error,independent_iou_error=max(v['max_iou_crosscheck'] for v in errs),
        independent_fresh_iou_error=read(OUT/'fresh_check.json')['worst_iou_error'],all_negatives_retained=True,
        source_optimizer_steps=0,target_optimizer_steps=0,FINAL_access=False))
    print('independent full check',len(panels),objects,old_error,max_error,flush=True)

if __name__=='__main__':main()
