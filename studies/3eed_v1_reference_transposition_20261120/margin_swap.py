"""Compare evidence reaching real candidate margins with its within-pool placebo."""
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.special import expit

from registry_swap import OBS, OUT, PRED, WORK, key, read
from physical_swap import Exchange
from roles_joint import affinity
from posterior_swap import ARMS, agreement
from score_swap import seals
from score_roles import anon, csv_file


def diagnose(rec):
    name=key(rec)
    panel,src,target=rec['panel'],rec['source'],rec['frame']['target']
    group=('source_proxy' if src==target else 'cross_proxy') if panel=='source_proxy' else ('natural' if panel.startswith('natural') else panel)
    with np.load(OBS/name) as z,np.load(PRED/name) as dec:
        factual,raw,wrong,interval,uncond=[ARMS.index(k) for k in (
            'native_prior','crossed_raw','wrong_landmark','crossed_interval','interval_same_winner')]
        ref_changed=int(not np.array_equal(dec['refs'][factual],dec['refs'][raw],equal_nan=True))
        info=dict(group=group,source=src,target=target,case=anon(f"{panel}|{target}|{rec['case']}"),
            applied=int(z['active']),pool_has_contrast=int(dec['observed_changes']>0),selected_could_use=0,
            selected_votes_differ=0,selected_inside_crop=0,max_abs_factor_delta=0.,selected_vote_delta=0.,
            raw_changed=int(not np.array_equal(dec['raw'][factual],dec['raw'][raw])),
            ref_changed=ref_changed,placebo_raw_changed=int(not np.array_equal(dec['raw'][wrong],dec['raw'][raw])),
            interval_delta=float(np.max(abs(dec['boxes'][interval]-dec['boxes'][uncond]))))
        if not z['active']:
            return info
        a=z['a']
        u=np.stack([affinity(a,bs)@m for bs,m in zip(z['views'],z['masses'])])
        u/=u.sum(1,keepdims=True)
        log=np.log(u,where=u>0,out=np.zeros_like(u))
        l0,ls=z['looks']
        h=np.divide(l0,l0+ls,out=np.full_like(l0,.5),where=l0+ls>0)
        stable=Exchange(z['cells'],z['half']).stationary(a)
        eligible=z['available'] & stable[:,None] & (u>0).all(0)[:,None] & (l0+ls>0)
        c=agreement(expit(log[0]-log[1]-log[2]+log[3]),h,eligible)
        effect=np.max(abs(c-1),axis=1)
        i=u[0].argmax()
        info.update(selected_could_use=int(eligible[i].any()),selected_votes_differ=int(effect[i]>1e-8),
            selected_inside_crop=int(not stable[i]),max_abs_factor_delta=float(effect.max()),
            selected_vote_delta=float(abs(((c[i]-1)*z['q']).sum())))
        return info


def run():
    seals()
    with ProcessPoolExecutor(max_workers=5) as workers:
        rows=list(workers.map(diagnose,read(WORK/'roster.json'),chunksize=5))
    from endpoint_score import write_gz
    write_gz(OUT/'margin_rows.csv.gz',rows)
    groups=defaultdict(list)
    for r in rows:
        groups[r['group']].append(r)
    table=[]
    fields=('applied','pool_has_contrast','selected_could_use','selected_votes_differ','selected_inside_crop','raw_changed','ref_changed','placebo_raw_changed')
    for name,items in groups.items():
        active=[r for r in items if r['applied']]
        table.append(dict(group=name,n=len(items),**{k:sum(r[k] for r in items) for k in fields},
            largest_conditioning_shift=max(r['interval_delta'] for r in items),
            median_max_abs_delta=float(np.median([r['max_abs_factor_delta'] for r in active])),
            median_selected_vote_delta=float(np.median([r['selected_vote_delta'] for r in active]))))
    csv_file(OUT/'margin_transfer.csv',table)


if __name__=='__main__':
    run()
