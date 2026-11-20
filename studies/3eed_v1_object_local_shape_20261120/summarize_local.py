"""Direction-balanced cross-platform scores, transitions and role identity controls."""
import numpy as np
import pandas as pd

from local_paths import *
from method_local import ARMS,GEOMS
from summarize_delivery import numbers,paired
from score_roles import record_intervals

def data(kind):
    paths=[OUT/(s+'_'+kind+'_rows.csv.gz') for s in SOURCES]
    return pd.concat([pd.read_csv(p,dtype={'case':str,'pair':str,'frame':str,'recording':str}) for p in paths if p.exists()],ignore_index=True)

def task_scores():
    observed=data('task');fallback=observed[observed.arm.eq('all_fallback')]
    assert len(fallback)==17326
    tasks=pd.concat([observed[observed.arm.ne('all_fallback')],*[fallback.assign(arm=a) for a in ARMS]],ignore_index=True)
    assert len(tasks)==(18620+1204+128)*len(ARMS) and not tasks.duplicated(['panel','source','case','arm']).any()
    scopes=dict(list(tasks.groupby('panel')))
    full=scopes['full'];scopes['full_eligible']=full[full.eligible.eq(1)]
    scopes['full_outside_small']=full[full.old_panel.eq(0)]
    out=[];pairs=[];columns=[(arm,g) for arm in ARMS for g in GEOMS]
    contrasts=set()
    for g in GEOMS:
        for a in ARMS:contrasts.add((('shape_difference',g),(a,g)))
        for z in [('v1','raw'),('v1','anchored'),('tree','anchored')]:contrasts.add((('shape_difference',g),z))
    for a in ARMS:
        for j,k in [('anchored','raw'),('fit','raw'),('fit','anchored'),('direct_fit','anchored'),('fit','direct_fit')]:
            contrasts.add(((a,j),(a,k)))
    for scope,sub in scopes.items():
        matrix={};rec={};vals={}
        for (s,t),part in sub.groupby(['source','target']):
            d=part.pivot(index=['case','recording'],columns='arm',values=list(GEOMS))
            x=d.loc[:,[(g,a) for a,g in columns]].to_numpy()
            assert np.isfinite(x).all()
            matrix[s,t]=x;rec[s,t]=dict(recording=d.index.get_level_values('recording').to_numpy())
            vals[s+'_to_'+t]=(x,numbers(x))
        joined=np.vstack([a for a,_ in vals.values()]);macro=np.mean([v for _,v in vals.values()],axis=0)
        vals['macro']=joined,macro
        for direction,(x,values) in vals.items():
            for (a,g),v in zip(columns,values):
                out.append(dict(panel=scope,direction=direction,n=len(x),arm=a,geometry=g,acc25=v[0],acc50=v[1],miou=v[2]))
        for aa,bb in sorted(contrasts):
            if aa==bb:continue
            ia,ib=columns.index(aa),columns.index(bb)
            main=scope=='full' and aa[0]=='shape_difference' and aa[1] in ('anchored','fit')
            ci=record_intervals(rec,matrix,ia,ib) if main else ('','')
            for direction,(x,v) in vals.items():
                d=v[ia]-v[ib];lo,hi=ci if direction=='macro' else ('','')
                pairs.append(dict(panel=scope,direction=direction,n=len(x),arm=aa[0],geometry=aa[1],control=bb[0],
                    control_geometry=bb[1],acc25_pp=d[0],acc50_pp=d[1],miou_pp=d[2],lo50=lo,hi50=hi,**paired(x[:,ia],x[:,ib])))
    pd.DataFrame(out).to_csv(OUT/'metrics.csv',index=False)
    pd.DataFrame(pairs).to_csv(OUT/'paired.csv',index=False)

def mechanisms():
    d=data('identity');groups=['panel','arm','geometry','rule'];keys=['subject','reference','both']
    w=d.groupby(groups);tot=w[keys].sum();tot['n']=w.size();tot.to_csv(OUT/'identity.csv')
    p=d[d.half.ge(0)].groupby(groups+['source','pair'])
    assert p.size().eq(2).all() and p.half.nunique().eq(2).all()
    q=p[keys].min().reset_index().groupby(groups)
    both=q[keys].sum();both['pairs']=q.size();both.to_csv(OUT/'pair_identity.csv')
    f=data('rank');grp=['panel','arm','role'];f['beats']=f.margin.gt(0);q=f.groupby(grp)
    r=q[['cover','role_nomination','either_nomination','comparable','beats']].sum()
    r['rows']=q.size();r['mean_margin']=q.margin.mean();r.to_csv(OUT/'ranking.csv')
    g=data('geometry');grp=['panel','arm','geometry','control']
    for k in ['rescue','harm']:g['stable_'+k]=g.stable*g[k]
    for k in ['independent','misbound_reference','tree','v1','misbound_crop']:
        g['outside_wrong_'+k]=g.outside_a256*g[k+'_wrong_instance']
    g.groupby(grp)[['rescue','harm','stable_rescue','stable_harm','outside_a256','both_known']+
       [c for c in g if c.startswith('outside_wrong_')]].sum().to_csv(OUT/'geometry.csv')
    rows=[]
    for s in SOURCES:
        obs=read(WORK/'observed'/s/'SEALED.json');pred=read(WORK/(s+'_decisions.json'));score=read(OUT/(s+'_score.json'))
        rows.append(dict(source=s,frames=len(obs['files']),object_crops=obs['objects'],native_batches=obs['batches'],capture_seconds=obs['seconds'],
            predicted=len(pred['files']),prediction_seconds=pred['seconds'],expression_reads=pred['exposures'],prior_box_error=pred['error'],
            original_iou_error=score['predecessor_error'],iou_crosscheck=score['max_iou_crosscheck']))
    pd.DataFrame(rows).to_csv(OUT/'runtime.csv',index=False)

if __name__=='__main__':task_scores();mechanisms()
