"""All paired threshold transitions, verified identities and geometry under fixed input budgets."""
import numpy as np
import pandas as pd
from collect import OUT,read
from joint_coverage import ARMS
from score_roles import measure,record_intervals,csv_file


def scope(table,group):
    if group=='natural_all':
        return table[table.group.isin(('natural170','natural171'))]
    return table[table.group==group]


def keep(table,n):
    return table[n].astype(str).isin(('1','1.0','True'))


def arms_by_direction(rows):
    blocks={}
    for key,block in rows.groupby(['source','target'],sort=True):
        arms={a:d.sort_values(['group','case']).reset_index(drop=True) for a,d in block.groupby('arm')}
        assert set(arms)==set(ARMS)
        order=arms['v1'][['group','case']]
        assert all(arm[['group','case']].equals(order) for arm in arms.values())
        blocks[key]=arms
    return blocks


def run():
    scored=read(OUT/'scoring.json')
    rows=pd.read_csv(OUT/'all_rows.csv.gz',keep_default_na=False)
    id_rows=pd.read_csv(OUT/'identity_rows.csv.gz',keep_default_na=False)
    assert len(rows)==scored['arm_rows'] and len(id_rows)==scored['identity_rows']
    metrics=[];pairs=[];identities=[];geometry=[];ci=[]
    for g in ('natural_all','natural170','natural171','own_proxy','cross_proxy'):
        obs=scope(rows,g); judgments=scope(id_rows,g)
        dirs=arms_by_direction(obs)
        mats={key:np.column_stack([ds[a].iou for a in ARMS]) for key,ds in dirs.items()}
        values={key:measure(mat) for key,mat in mats.items()}
        all_arms={a:pd.concat([ds[a] for ds in dirs.values()],ignore_index=True) for a in ARMS}
        jobs=[('_to_'.join(key),ds,values[key]) for key,ds in dirs.items()]
        jobs.append(('macro',all_arms,np.mean(list(values.values()),axis=0)))
        for dir_label,arms,table in jobs:
            for i,arm in enumerate(ARMS):
                r=arms[arm];applied=r.applied==1
                meta=dict(group=g,direction=dir_label,arm=arm,n=len(r))
                cov=pd.to_numeric(r.loc[applied,'expected_coverage'],errors='coerce').dropna()
                metrics.append(dict(**meta,acc25=table[i,0],acc50=table[i,1],miou=table[i,2],
                    active=int(applied.sum()),different_raw=int(r.raw_changed.sum()),shape_changed=int(r.shape_changed.sum()),
                    known_correct_raw=int((r.official_raw=='T').sum()),correct_final=int((r.official_final=='T').sum()),
                    slate_hit25=int((r.slate_best>.25).sum()),slate_hit50=int((r.slate_best>.50).sum()),
                    factual_mass_covered=(float(cov.mean()) if len(cov) else ''),
                    same_subject_as_main=int(r.same_raw_as_main.sum()),same_reference_as_main=int(r.same_ref_as_main.sum()),
                    same_output_as_main=int(r.same_box_as_main.sum()),same_subject_slate=int(r.same_subject_slate_as_main.sum()),
                    same_ref_slate=int(r.same_ref_slate_as_main.sum()),recordings=int(r.recording.nunique())))
                for control in (('v1',) if arm!='joint_nomination' else tuple(x for x in ARMS if x!=arm)):
                    c=arms[control];j=ARMS.index(control)
                    counts={}
                    for level in (25,50):
                        a=r.iou>level/100;old=c.iou>level/100
                        counts[f'rescue{level}']=int((a&~old).sum());counts[f'harm{level}']=int((~a&old).sum())
                    pairs.append(dict(**meta,control=control,delta25=table[i,0]-table[j,0],delta50=table[i,1]-table[j,1],
                         delta_miou=table[i,2]-table[j,2],better_iou=int((r.iou-c.iou>1e-9).sum()),worse_iou=int((c.iou-r.iou>1e-9).sum()),
                         official_W_to_T=int(((c.official_raw=='W')&(r.official_raw=='T')).sum()),
                         official_T_to_W=int(((c.official_raw=='T')&(r.official_raw=='W')).sum()),**counts))
                hit=r.iou>.50;raw=r.raw_iou>.50
                for label,eligible in [('official_raw_correct',r.official_raw=='T'),
                                      ('prior_and_raw_correct',(r.official_raw=='T')&(r.official_prior=='T'))]:
                    geometry.append(dict(**meta,condition=label,eligible=int(eligible.sum()),
                        modified=int((eligible&(r.shape_changed==1)).sum()),lost_owner=int((eligible&(r.official_final!='T')).sum()),
                        rescue50=int((eligible&hit&~raw).sum()),harm50=int((eligible&~hit&raw).sum()),
                        opportunity_a=int((eligible&(r.best_original<=.5)).sum()),
                        beyond_a=int((eligible&hit&(r.official_final=='T')&(r.best_original<=.5)).sum()),
                        opportunity_all=int((eligible&(r.best_all<=.5)).sum()),
                        beyond_all=int((eligible&hit&(r.official_final=='T')&(r.best_all<=.5)).sum())))
        recording={key:{'recording':r['v1'].recording.to_numpy()} for key,r in dirs.items()}
        for i,control in enumerate(ARMS[:-1]):
            low,high=record_intervals(recording,mats,len(ARMS)-1,i)
            ci.append(dict(group=g,control=control,lower50=low,upper50=high,recordings=all_arms['v1'].recording.nunique()))
        for rule,js in judgments.groupby('rule'):
            blocks=arms_by_direction(js)
            views=[('_to_'.join(k),v) for k,v in blocks.items()]
            views.append(('pooled',{a:pd.concat([v[a] for v in blocks.values()],ignore_index=True) for a in ARMS}))
            for direction,arms in views:
                for a,r in arms.items():
                    c=arms['v1'];known=r.both_known==1
                    dual=(r.raw=='T')&(r.ref=='T');old=(c.raw=='T')&(c.ref=='T')
                    qualified=known&keep(r,'cover_s')&keep(r,'cover_r')
                    possible=known&keep(r,'pool_s')&keep(r,'pool_r')
                    actual=(r.raw=='T');hit=r.iou>.50;prior=r.raw_iou>.50
                    subset=r.applied==1
                    identities.append(dict(group=g,direction=direction,rule=rule,arm=a,n=len(r),
                        known_both=int(known.sum()),known_subject=int((r.raw!='NA').sum()),known_ref=int((r.ref!='NA').sum()),
                        active_known=int((known&subset).sum()),full_both=int(possible.sum()),slate_both=int(qualified.sum()),
                        only_subject=int((known&keep(r,'cover_s')&~keep(r,'cover_r')).sum()),
                        only_reference=int((known&~keep(r,'cover_s')&keep(r,'cover_r')).sum()),
                        active_full_both=int((subset&possible).sum()),active_slate_both=int((subset&qualified).sum()),
                        active_dual=int((subset&dual).sum()),dual=int(dual.sum()),final_dual=int(((r['final']=='T')&(r.ref=='T')).sum()),
                        dual_gain=int((dual&~old).sum()),dual_loss=int((old&~dual).sum()),
                        correct_reference=int((r.ref=='T').sum()),raw_correct=int(actual.sum()),final_correct=int((r['final']=='T').sum()),
                        W_to_T=int(((r.prior=='W')&actual).sum()),T_to_W=int(((r.prior=='T')&(r.raw=='W')).sum()),
                        T_to_U=int(((r.prior=='T')&(r.raw=='U')).sum()),U_to_T=int(((r.prior=='U')&actual).sum()),
                        cond_rescue50=int((actual&hit&~prior).sum()),cond_harm50=int((actual&~hit&prior).sum()),
                        beyond_all=int((actual&hit&(r['final']=='T')&(r.best_all<=.5)).sum()),
                        cond_lost_owner=int((actual&(r['final']!='T')).sum())))
    for name,objs in [('metrics',metrics),('paired',pairs),('geometry',geometry),('identity',identities),('intervals',ci)]:
        csv_file(OUT/(name+'.csv'),objs)


if __name__=='__main__':
    run()

