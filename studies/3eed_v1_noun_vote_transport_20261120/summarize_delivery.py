"""Aggregate every trial with equal direction weighting and paired threshold transitions."""
import numpy as np
import pandas as pd
from vote_paths import *
from score_roles import record_intervals
from score_delivery import GEOMS

def load_rows(kind):
    paths=sorted(OUT.glob('*_'+kind+'_rows.csv.gz'))
    assert len(paths)==3
    return pd.concat([pd.read_csv(p,dtype={'case':str,'frame':str,'recording':str,'pair':str}) for p in paths],ignore_index=True)

def write_table(filename,rows):
    pd.DataFrame(rows).to_csv(OUT/filename,index=False)

def numbers(x):
    return 100*np.stack(((x>.25).mean(axis=0),(x>.5).mean(axis=0),x.mean(axis=0)),axis=-1)

def paired(a,b):
    return {**{f'{action}{t}':int(mask.sum()) for t in (25,50)
               for action,mask in [('rescue',(a>t/100)&(b<=t/100)),('harm',(a<=t/100)&(b>t/100))]},
            'iou_up':int((a>b+1.e-9).sum()),'iou_down':int((a<b-1.e-9).sum())}

def summarize(tasks):
    stats=[];paired_rows=[]
    groups={name:g for name,g in tasks.groupby('panel')}
    full=groups['full']
    groups.update(full_eligible=full[full.eligible.eq(1)],full_fallback=full[full.eligible.eq(0)],
                  full_outside_small=full[full.old_panel.eq(0)])
    for panel,frm in groups.items():
        data={};recordings={};allvals=[]
        columns=[(arm,g) for arm in ARMS for g in GEOMS]
        for (s,t),part in frm.groupby(['source','target']):
            arms=part.pivot(index=['case','recording'],columns='arm',values=list(GEOMS))
            x=arms.loc[:,[(g,arm) for arm,g in columns]].to_numpy();assert np.isfinite(x).all()
            rec=arms.index.get_level_values('recording').to_numpy();data[s,t]=x
            recordings[s,t]={'recording':rec}
            val=numbers(x);allvals.append(val)
            for (arm,geo),(a,b,c) in zip(columns,val):
                stats.append(dict(panel=panel,direction=s+'_to_'+t,n=len(x),arm=arm,geometry=geo,acc25=a,acc50=b,miou=c))
        merged=np.vstack(list(data.values()));macro=np.mean(allvals,axis=0)
        for (arm,geo),v in zip(columns,macro):
            stats.append(dict(panel=panel,direction='macro',n=len(merged),arm=arm,geometry=geo,acc25=v[0],acc50=v[1],miou=v[2]))
        contrasts=set()
        for geo in GEOMS:
            for control in ARMS:
                if control!='association':contrasts.add((('association',geo),(control,geo)))
            for base in [('tree','anchored'),('v1','raw'),('v1','anchored')]:
                contrasts.add((('association',geo),base))
        for arm in ARMS:
            for g,c in [('fit','raw'),('anchored','raw'),('fit','anchored'),('fit','null_fit'),('null_fit','anchored')]:
                contrasts.add(((arm,g),(arm,c)))
        for aa,bb in sorted(contrasts):
            ia,ib=columns.index(aa),columns.index(bb)
            main=(panel=='full' and aa[0]=='association' and aa[1] in ('anchored','fit'))
            ci=record_intervals(recordings,data,ia,ib) if main else ('','')
            for direction,x in [(s+'_to_'+t,m) for (s,t),m in data.items()]+[('macro',merged)]:
                v=(macro if direction=='macro' else numbers(x))
                d=v[ia]-v[ib]
                paired_rows.append(dict(panel=panel,direction=direction,arm=aa[0],geometry=aa[1],control=bb[0],
                       control_geometry=bb[1],n=len(x),delta25=d[0],delta50=d[1],delta_miou=d[2],**paired(x[:,ia],x[:,ib]),
                       rec_bootstrap50_lower=ci[0] if direction=='macro' else '',
                       rec_bootstrap50_upper=ci[1] if direction=='macro' else ''))
    write_table('metrics.csv',stats);write_table('paired.csv',paired_rows)

def identities():
    ids=load_rows('identity');measure=['subject','reference','both']
    g=ids.groupby(['panel','arm','rule','geometry'])
    sums=g[measure].sum();sums['known']=g.size()
    sums.to_csv(OUT/'endpoint_metrics.csv')
    pairs=ids[ids.half.ge(0)].copy()
    index=['panel','arm','geometry','rule','source','pair']
    halves=pairs.groupby(index)
    assert halves.size().eq(2).all() and halves.half.nunique().eq(2).all()
    a=halves[measure].min().rename(columns={'subject':'two_subjects','reference':'two_references','both':'four_endpoints'})
    a['distinct_identified_subjects']=(halves.subject_owner.min().ge(0)&halves.subject_owner.nunique().eq(2)).astype(int)
    a=a.reset_index();g=a.groupby(['panel','arm','rule','geometry']);v=g[['two_subjects','two_references','four_endpoints','distinct_identified_subjects']].sum()
    v['pairs']=g.size();v.to_csv(OUT/'paired_identity.csv')
    rank=load_rows('ranking')
    g=rank.groupby(['panel','arm','ownership'])
    a=g[['margin','pair_accuracy']].mean()
    a['beats_strongest']=g.beats_strongest.sum();a['eligible']=g.size();a.to_csv(OUT/'ranking.csv')
    geo=load_rows('geometry')
    groups=['panel','arm','geometry','control'];g=geo.groupby(groups)
    a=g[['rescue','harm','outside_a256','outside_all_native','both_endpoints']].sum()
    for key in ('harm','rescue'):
        a['verified_'+key]=(geo[key]*geo.subject_kept).groupby([geo[c] for c in groups]).sum()
        a['both_'+key]=(geo[key]*geo.both_endpoints).groupby([geo[c] for c in groups]).sum()
    a.to_csv(OUT/'geometry.csv')

def main():
    OUT.mkdir(exist_ok=True)
    rows=load_rows('task')
    fallback=rows.loc[rows.arm.eq('all_fallback')];assert len(fallback)==17326
    tasks=pd.concat([rows[~rows.arm.eq('all_fallback')]]+[fallback.assign(arm=a) for a in ARMS],ignore_index=True)
    assert not tasks.duplicated(['panel','source','case','arm']).any()
    assert len(tasks)==(18620+1204+128)*len(ARMS)
    summarize(tasks);identities()
    sources=[]
    for s in SOURCES:
        meta=read(WORK/(s+'_out.json'));obs=read(WORK/'votes'/s/'SEALED.json');score=read(OUT/(s+'_scoring.json'))
        fields=pd.DataFrame(read(WORK/(s+'_activity.json')))
        sources.append(dict(source=s,records=len(meta['files']),frame_sources=len(fields),center_rms_median=fields.word_center_rms.median(),
             association_tv_median=fields.association_tv.median(),fraction_box_votes_on_returns=fields.supported.sum()/fields.votes.sum(),
             small_archive_replay_error=meta['prior_reproduction_error'],full_scalar_replay_error=score['prior_error'],
             iou_independent_error=score['ground_error'],source_capture_seconds=obs.get('seconds'),prediction_seconds=meta['seconds']))
    write_table('runtime.csv',sources)
    save(OUT/'audit.json',dict(seal=digest(WORK/'SEALED.json'),protocol=digest(SPEC),cases=len(tasks)//len(ARMS),
        ordinary=18620,ordinary_eligible=1294,ordinary_fallback=17326,independent_frame_sources=sum(r['frame_sources'] for r in sources),
        frozen_checkpoint_sources=len(SOURCES),threshold='strict >',geometry_policies=GEOMS,arms=ARMS,
        scores_after_seal=True,prior_replay_max=max(s['small_archive_replay_error'] for s in sources),
        ordinary_error=max(s['full_scalar_replay_error'] for s in sources),
        independent_iou_error=max(s['iou_independent_error'] for s in sources),
        source_optimizer_steps=0,target_optimizer_steps=0,FINAL_access=False))
if __name__=='__main__':main()

