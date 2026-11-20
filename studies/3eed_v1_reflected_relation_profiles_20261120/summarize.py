"""Six-direction paired task outcomes, observed physical owners and conditional geometry."""
import numpy as np
import pandas as pd

from observe import RESULTS, digest, read
from profiles import ARMS
from score_roles import csv_file, measure, record_intervals


def population(rows,group):
    return rows[rows['group'].isin(('natural170','natural171'))] if group=='natural_all' else rows[rows['group']==group]


def summarize():
    scored=read(RESULTS/'scoring.json')
    rows=pd.read_csv(RESULTS/'all_rows.csv.gz',keep_default_na=False)
    assert len(rows)==scored['rows']
    metrics,identity,geometry,paired,intervals=[],[],[],[],[]
    for group in ('natural_all','natural170','natural171','own_proxy','cross_proxy'):
        tables={}
        mats={}
        for key,batch in population(rows,group).groupby(['source','target'],sort=True):
            arms={a:batch[batch.arm==a].sort_values(['panel','case']).reset_index(drop=True) for a in ARMS}
            base=arms['v1'][['panel','case']]
            assert all(base.equals(d[['panel','case']]) for d in arms.values())
            tables[key]=arms
            mats[key]=np.column_stack([arms[a].iou for a in ARMS])
        combined={a:pd.concat([arms[a] for arms in tables.values()],ignore_index=True) for a in ARMS}
        macro=np.mean([measure(m) for m in mats.values()],axis=0)
        jobs=[('_to_'.join(k),tabs,measure(mats[k])) for k,tabs in tables.items()]
        jobs.append(('macro',combined,macro))
        for direction,tabs,values in jobs:
            for i,arm in enumerate(ARMS):
                d=tabs[arm]
                head=dict(group=group,direction=direction,arm=arm,n=len(d))
                metrics.append(dict(**head,acc25=values[i,0],acc50=values[i,1],miou=values[i,2],active=int(d.active.sum()),
                                    box_changed=int(d.shape_modified.sum()),raw_switched=int(d.switched.sum()),
                                    recordings=int(d.recording.nunique())))
                subject_known=d.human_raw!='NA'
                ref_known=d.reference!='NA'
                known=subject_known&ref_known
                both_raw=(d.human_raw=='T')&(d.reference=='T')
                both_final=(d.human_subject=='T')&(d.reference=='T')
                cover_s=d.subject_available.astype(str)=='1'
                cover_r=d.ref_available.astype(str)=='1'
                states={f'{old}_to_{new}':int(((d.prior_subject==old)&(d.raw_subject==new)).sum())
                        for old in 'TWU' for new in 'TWU'}
                identity.append(dict(**head,**states,official_prior_correct=int((d.prior_subject=='T').sum()),
                     official_raw_correct=int((d.raw_subject=='T').sum()),official_final_correct=int((d.subject=='T').sum()),
                     physically_different_resolved=int(d.new_physical_owner.sum()),
                     known_subject=int(subject_known.sum()),known_reference=int(ref_known.sum()),known_both=int(known.sum()),
                     confirmed_raw_subject=int((d.human_raw=='T').sum()),confirmed_final_subject=int((d.human_subject=='T').sum()),
                     correct_reference=int((d.reference=='T').sum()),both_raw_correct=int(both_raw.sum()),both_final_correct=int(both_final.sum()),
                     both_available=int((known&cover_s&cover_r).sum()),active_verified_both=int((known&(d.active==1)).sum()),
                     reference_available=int((cover_r&ref_known).sum())))
                for denom,ok in [('own_raw_correct',(d.raw_subject=='T')),('prior_and_raw_correct',(d.raw_subject=='T')&(d.prior_subject=='T')),
                                 ('independent_raw_correct',d.human_raw=='T')]:
                    valid=(d.human_subject=='T') if denom=='independent_raw_correct' else (d.subject=='T')
                    ok=ok.to_numpy()
                    hit=(d.iou>.50).to_numpy()
                    raw_hit=(d.raw_iou>.50).to_numpy()
                    corrected=ok&hit&valid.to_numpy()
                    original_pool=(d.best_original_union<=.50).to_numpy()
                    extended_pool=(d.best_all_union<=.50).to_numpy()
                    geometry.append(dict(**head,condition=denom,eligible=int(ok.sum()),
                        modified=int((ok&(d.shape_modified==1)).sum()),
                        rescue50=int((ok&hit&~raw_hit).sum()),harm50=int((ok&~hit&raw_hit).sum()),
                        lost_owner=int((ok&~valid.to_numpy()).sum()),correct_raw_original_pool_miss=int((ok&original_pool).sum()),
                        verified_rescue_outside_original=int((corrected&original_pool).sum()),
                        correct_raw_four_view_pool_miss=int((ok&extended_pool).sum()),
                        verified_rescue_outside_all=int((corrected&extended_pool).sum())))
                comparisons=(('v1',) if arm!='joint_profile' else tuple(a for a in ARMS if a!=arm))
                for control in comparisons:
                    c=tabs[control]
                    j=ARMS.index(control)
                    flags={}
                    for level in (25,50):
                        hit=d.iou>level/100
                        old=c.iou>level/100
                        flags[f'rescue{level}']=int((hit&~old).sum())
                        flags[f'harm{level}']=int((~hit&old).sum())
                    positive=(d.human_raw=='T')&(d.reference=='T')
                    prior=(c.human_raw=='T')&(c.reference=='T')
                    paired.append(dict(**head,control=control,delta_acc25=values[i,0]-values[j,0],
                        delta_acc50=values[i,1]-values[j,1],delta_miou=values[i,2]-values[j,2],**flags,
                        better_iou=int((d.iou-c.iou>1.e-9).sum()),worse_iou=int((c.iou-d.iou>1.e-9).sum()),
                        both_gain=int((positive&~prior).sum()),both_loss=int((prior&~positive).sum()),
                        wrong_to_correct=int(((c.raw_subject=='W')&(d.raw_subject=='T')).sum()),
                        correct_to_wrong=int(((c.raw_subject=='T')&(d.raw_subject=='W')).sum())))
        recordings={k:{'recording':d['v1'].recording.to_numpy()} for k,d in tables.items()}
        for j,control in enumerate(ARMS[:-1]):
            lo,hi=record_intervals(recordings,mats,len(ARMS)-1,j)
            intervals.append(dict(group=group,control=control,lower_acc50_pp=lo,upper_acc50_pp=hi,
                                  target_recordings=int(combined['v1'].recording.nunique())))
    for name,data in [('metrics',metrics),('paired',paired),('identity',identity),('geometry',geometry),('intervals',intervals)]:
        csv_file(RESULTS/(name+'.csv'),data)


if __name__=='__main__':
    summarize()

