"""Independent count check and a post-seal decomposition of unequal switching frequencies."""
import gzip
import csv
from collections import Counter
import numpy as np
import pandas as pd
from capture import HERE, ORIGINAL, WORK, OUT, PERMS, NAMES, bound_components, read, sha, save

def main():
    seal=read(WORK/'SEALED.json');assert seal['code']==sha(HERE/'capture.py') and not seal['labels_read']
    counts=Counter();samples={};contrasts=[];groups=[]
    for d,z in bound_components():
        samples.setdefault(d,[]).append(z)
    for direction,parts in samples.items():
        z={k:np.concatenate([p[k] for p in parts]) for k in parts[0]}
        with np.load(WORK/(direction+'.npz')) as P:picks=P['picks'].astype(int)
        assert len(picks)==len(z['final'])
        gtpath=ORIGINAL/('scored_'+direction+'.npz')
        assert sha(gtpath)==read(gtpath.with_suffix('.json'))['file_sha256']
        with np.load(gtpath) as G:all_iou=G['pool_iou']
        slot_iou=all_iou[np.arange(len(picks))[:,None],z['slots'].astype(int)]
        score=slot_iou[np.arange(len(picks))[:,None],picks];v1=picks[:,0];a=z['prior']
        for j,perm in enumerate(PERMS,1):
            # Independent scalar version checks all 18620 x 10 choices, including ties.
            for k,original in enumerate(v1):
                obs=z['support'][k,1,list(perm),9:11]
                candidates=[q for q in range(4) if obs[q,0]>obs[original,0] and obs[q,1]<obs[original,1]]
                other=max(candidates,key=lambda v:z['final'][k,v]) if candidates else original
                assert other==picks[k,j]
        for t in (.25,.5):
            good=slot_iou>t
            missed=~(score[:,0]>t)&good.any(1)
            hard=missed & (a.max(1)-np.where(good,a,-np.inf).max(1)>=1.5+z['active'])
            counts[t,'hard']+=hard.sum();counts[t,'missed']+=missed.sum()
            for j,name in enumerate(NAMES):
                sel=score[:,j]>t;old=score[:,0]>t
                groups.append(dict(direction=direction,threshold=t,arm=name,records=len(picks),hits=int(sel.sum()),
                    rescues=int((sel&~old).sum()),harms=int((~sel&old).sum()),different_slots=int((picks[:,j]!=v1).sum())))
            actual=picks[:,1]!=v1
            for j,name in enumerate(NAMES[2:],2):
                null=picks[:,j]!=v1
                x,y=score[:,1]>t,score[:,j]>t
                for status,mm in [('both_switch',actual&null),('only_true',actual&~null),('only_mismatch',~actual&null),('neither',~actual&~null)]:
                    up=int((x&~y&mm).sum());down=int((~x&y&mm).sum())
                    contrasts.append(dict(direction=direction,threshold=t,control=name,status=status,records=int(mm.sum()),
                        different_winner=int((mm&(picks[:,1]!=picks[:,j])).sum()),rescue_vs_control=up,harm_vs_control=down,
                        contribution_full_direction_pp=100*(up-down)/len(x)))
    original=pd.read_csv(OUT/'direction.csv').query('scope=="all"');idx=['direction','threshold','arm']
    cols=['records','hits','rescues','harms','different_slots']
    pd.testing.assert_frame_equal(original.set_index(idx)[cols].sort_index(),pd.DataFrame(groups).set_index(idx)[cols].sort_index())
    out=pd.DataFrame(contrasts);out.to_csv(OUT/'switch_comparison.csv',index=False)
    grouped=out.groupby(['threshold','control','status'])
    total=grouped[['records','different_winner','rescue_vs_control','harm_vs_control']].sum()
    total['macro_contribution_pp']=grouped.contribution_full_direction_pp.mean()
    total.reset_index().groupby(['threshold','status'])[total.columns].mean().to_csv(OUT/'switch_decomposition.csv')
    with gzip.open(OUT/'all_rows.csv.gz','rt') as f:
        rr=list(csv.DictReader(f))
    assert len({(r['case'],r['threshold']) for r in rr})==37240
    for t,b,m in [(.25,83,3273),(.5,36,1252)]:
        assert counts[t,'hard']==b and counts[t,'missed']==m
        sums=out[out.threshold==t].groupby('control').contribution_full_direction_pp.sum()/6
        official=pd.read_csv(OUT/'paired.csv').query('threshold==@t and contrast=="true_minus_mean_derangements"').iloc[0].delta_macro_pp
        assert abs(sums.mean()-official)<1e-10
    save(OUT/'verification.json',dict(seal=sha(WORK/'SEALED.json'),all_slots_scalar_decisions=18620*10,
         blocked25=83,blocked50=36,groups_checked=len(groups),decision_inputs_frozen=True,
         masks='Descriptive decomposition added after scoring; disjoint masks add to full-cohort difference.',
         FINAL_access=False))
if __name__=='__main__':main()
