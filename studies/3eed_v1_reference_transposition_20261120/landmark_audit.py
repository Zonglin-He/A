"""After sealing, test whether the transplant actually moves the named physical landmark."""
import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.special import expit

from registry_swap import ROOT, OUT, OBS, PANELS, WORK, key, read, save
from posterior_swap import agreement
from physical_swap import Exchange
from roles_joint import affinity
from endpoint_score import arrays_to_owners
from score_swap import seals
from score_roles import inventory, csv_file, anon
from diagnose_roles import COHORTS, historical_records, historical_endpoints


def run(repo):
    seals()
    history=historical_records(repo)
    panels={name:read(path) for name,path in PANELS.items()}
    desired={}
    rows=[]
    for rec in read(WORK/'roster.json'):
        panel,idx,src=rec['panel'],rec['case'],rec['source']
        if panel not in panels:
            continue
        entry=panels[panel][idx]
        row=entry['row']
        target=row['platform']
        truth,_=inventory(row['metapath'],target,int(row['suffix'] or 0),int(row['object_index']))
        seed=None
        if panel in COHORTS:
            if (panel,idx) not in desired:
                desired[panel,idx],_=historical_endpoints(COHORTS[panel],entry,truth,history)
            subject,reference=desired[panel,idx]
            tag=f"{COHORTS[panel]}/{entry['case']:02d}_{src}.npz"
            path=ROOT/'crossplatform/round171/predictions'/tag
            with np.load(path) as old:
                seed=old['seeds']
            group='natural'
        else:
            subject,reference=0,int(entry['ref_id'])
            group='source_proxy' if src==target else 'cross_proxy'
        if subject is None or reference is None:
            continue
        with np.load(OBS/key(rec)) as z:
            if z['active']:
                a,d,lm=z['a'],z['d'],z['landmarks']
                owners=arrays_to_owners([a,d,lm],truth,seed)
                stationary=Exchange(z['cells'],z['half']).stationary(a)
                u=np.stack([affinity(a,bank)@mass for bank,mass in zip(z['views'],z['masses'])])
                u/=u.sum(1,keepdims=True)
                valid=(u>0).all(0)
                log=np.log(u,where=u>0,out=np.zeros_like(u))
                l0,ls=z['looks']
                h=np.divide(l0,l0+ls,out=np.full_like(l0,.5),where=l0+ls>0)
                allowed=stationary[:,None]&valid[:,None]&z['available']&(l0+ls>0)
                c=agreement(expit(log[0]-log[1]-log[2]+log[3]),h,allowed)
                useful=allowed & (abs(c-1)>1e-8)
            for rule in (('unique_iou','seed_ownership') if seed is not None else ('unique_iou',)):
                info=dict(group=group,rule=rule,source=src,target=target,case=anon(f'{panel}|{target}|{idx}'),
                    active=int(z['active']),full_D_ref=0,swapped_named=0,stationary_named_subject=0,
                    correct_pair_has_contrast=0,identifiable_comparison=0,margin_up=0,margin_down=0,
                    crossing_rescue=0,crossing_harm=0)
                if z['active']:
                    sid,did,lid=owners[rule]
                    named=(lid==reference)
                    good=(sid==subject)&stationary
                    same_pair=useful & good[:,None] & named[None,:]
                    info.update(full_D_ref=int((did==reference).any()),swapped_named=int(named.any()),
                         stationary_named_subject=int(good.any()),correct_pair_has_contrast=int(same_pair.any()))
                    bad=(sid>=0)&(sid!=subject)&stationary
                    if same_pair.any() and bad.any():
                        # Compare the same best labelled-good vs best labelled-wrong in each state;
                        # the named landmark is used only in this post-hoc conditional margin.
                        factor=c[:,named].mean(1)
                        score=u[0]*factor
                        prior=u[0,good].max()-u[0,bad].max()
                        delta=score[good].max()-score[bad].max()
                        info.update(identifiable_comparison=1,margin_up=int(delta>prior+1e-12),
                            margin_down=int(delta<prior-1e-12),crossing_rescue=int(prior<=0<delta),
                            crossing_harm=int(delta<=0<prior))
                rows.append(info)
    fields=list(rows[0])[5:]
    buckets=defaultdict(list)
    for row in rows:
        buckets[row['group'],row['rule']].append(row)
    table=[]
    for (group,rule),items in buckets.items():
        table.append(dict(group=group,rule=rule,n=len(items),**{k:sum(r[k] for r in items) for k in fields}))
    csv_file(OUT/'landmark_identity.csv',table)
    csv_file(OUT/'landmark_identity_rows.csv',rows)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path)
    run(p.parse_args().archive)
