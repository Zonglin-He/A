"""Summarize sealed attribution and paired metrics without exporting annotations."""
import argparse

import numpy as np

from scope176 import (N, ORDER, PLATFORMS, PRIOR, PRIVATE, RESULTS, json_atomic, read, sha)
from diagnose176 import ARMS, DUAL_CUTS, STAGES, SUBJECT_CUTS
from score175 import write_csv
from score176 import diagnosis
from retrieve176 import OUTPUT

METHODS = ('r158_pca', 'original_pair', 'overlap_retrieval')
METRICS = ('Acc25', 'Acc50', 'mIoU')
SUBJECT_STAGES = ('A256_no_subject', 'four_slots_no_subject', 'output_miss', 'success')


def make_gate():
    seal = diagnosis()
    rows = [r for s in PLATFORMS for r in read(PRIVATE/f'dual_source_{s}.json')]
    counts = np.bincount([x['stage'][0][0] for x in rows],minlength=4)
    value = dict(n=len(rows),rule='largest mutually exclusive failure count on source held-out dual IoU > .10',
                 counts=dict(zip(STAGES,counts.tolist(),strict=True)),
                 retrieval_is_largest=bool(counts[1]>max(counts[0],counts[2])),
                 diagnosis=seal,FINAL_access=False)
    p=PRIVATE/'RETRIEVAL_GATE.json'
    if p.exists():
        assert read(p)==value
    else:
        json_atomic(p,value)
    return value


def dual_summary(rows, arms, group, cohort):
    available = [r for r in rows if r['status']=='verified']
    for c,(s,r) in enumerate(DUAL_CUTS):
        eligible = np.asarray([q['good_roles'][c] for q in available],dtype=bool).reshape(-1,2,3)
        fs,fr,fp = eligible[:,0].T
        ss,sr,sp = eligible[:,1].T
        gs = np.asarray([q['selected_ious'] for q in available],float).reshape(-1,len(arms),2)
        for i,arm in enumerate(arms):
            stage = np.array([q['stage'][c][i] for q in available],int)
            count = np.bincount(stage,minlength=4)
            assert len(stage) == sum(count)
            yield dict(panel=group,cohort=cohort,arm=arm,subject_cut=s,reference_cut=r,n=len(rows),
                       verified_n=len(available),unmatched=len(rows)-len(available),
                       recordings=len({q['recording'] for q in available}),
                       physical_frames=len({q['frame'] for q in available}),
                       **dict(zip(STAGES,count.tolist(),strict=True)),
                       A256_subject_covered=int(fs.sum()),A256_reference_covered=int(fr.sum()),
                       four_subject_covered=int(ss.sum()),four_reference_covered=int(sr.sum()),
                       A256_pair_collision=int((fs & fr & ~fp).sum()),
                       four_pair_collision=int((ss & sr & ~sp & fp).sum()),
                       lost_subject_only=int((fp & ~ss & sr).sum()),
                       lost_reference_only=int((fp & ss & ~sr).sum()),
                       lost_both_roles=int((fp & ~ss & ~sr).sum()),
                       selected_subject=int((gs[:,i,0]>s).sum()),selected_reference=int((gs[:,i,1]>r).sum()))


def dual_tables():
    sources = [r for s in PLATFORMS for r in read(PRIVATE/f'dual_source_{s}.json')]
    target_by_arm = [
        [r for s,t in ORDER for r in read(PRIVATE/f'{prefix}_{s}_to_{t}.json')]
        for prefix in ('dual_target','new_dual')]
    dual, row_archive = [], []
    properties = [('verified','all',lambda r:True),('relation','active',lambda r:r['relation']!=0),
                  ('relation','inactive',lambda r:r['relation']==0)]
    for group, rows, arms in (('source_held',sources,ARMS),
                              ('target_review',target_by_arm[0],ARMS),
                              ('target_review',target_by_arm[1],('overlap_retrieval',))):
        strata = properties + [('platform',p,lambda r,p=p:r['target']==p) for p in PLATFORMS]
        if group=='source_held':
            strata += [('origin',o,lambda r,o=o:r['origin']==o)
                       for o in ('natural_unique_inventory','synthetic_labelled_pair')]
        for _,name,test in strata:
            dual.extend(dual_summary([r for r in rows if test(r)],arms,group,name))
        perm=np.random.default_rng(1761).permutation(len(rows))
        for j,k in enumerate(perm):
            rec=rows[k]
            for c,(s,r) in enumerate(DUAL_CUTS):
                for i,arm in enumerate(arms):
                    valid=rec['status']=='verified'
                    stage=STAGES[rec['stage'][c][i]] if valid else rec['status']
                    row_archive.append(dict(panel=group,direction=rec['source']+'_to_'+rec['target'],
                               record=f'{group}_{j+1:04}',arm=arm,
                               relation_active=int(rec['relation']!=0),origin=rec['origin'],
                               subject_cut=s,reference_cut=r,stage=stage,
                               selected_subject_iou=rec['selected_ious'][i][0] if valid else '',
                               selected_reference_iou=rec['selected_ious'][i][1] if valid else ''))
    assert len(sources)==336 and all(len(x)==50 for x in target_by_arm)
    return dual,row_archive


def metric_values(iou):
    return np.stack((iou>.25,iou>.5,iou),axis=-1).astype(float)


def direction_data(s,t):
    route=f'{s}_to_{t}'
    with np.load(PRIVATE/f'subject_{route}.npz',allow_pickle=False) as z:
        old={k:z[k] for k in ('stage','r158_iou','pair_iou','full_iou','relation','ordinal')}
    with np.load(PRIVATE/f'new_subject_{route}.npz',allow_pickle=False) as z:
        np.testing.assert_array_equal(z['ordinal'],old['ordinal'])
        new={k:z[k] for k in ('stage','iou','chosen_changed')}
    with np.load(PRIOR/'score'/f'{route}.npz',allow_pickle=False) as z:
        np.testing.assert_array_equal(z['ordinal'],old['ordinal'])
        rec=z['recording']
    return (np.column_stack((old['r158_iou'],old['pair_iou'],new['iou'])),
            np.concatenate((old['stage'],new['stage'][...,None]),axis=-1),
            old['relation']!=0,new['chosen_changed'],rec)


def direction_tables():
    task, attribution, paired, transitions = [], [], [], []
    endpoints, strata = {}, {}
    pairs=(('overlap_retrieval','original_pair'),('overlap_retrieval','r158_pca'),
           ('original_pair','r158_pca'))
    for s,t in ORDER:
        route=f'{s}_to_{t}'
        iou, stages, related, changed, recordings = direction_data(s,t)
        endpoints[route]=dict(iou=iou,recording=recordings)
        n=len(iou)
        perm=np.random.default_rng(17680+len(endpoints)).permutation(n)
        rows=[]
        for i,k in enumerate(perm):
            rows.append(dict(record=f'{i+1:04}',relation_active=int(related[k]),
                             changed_subject_query=int(changed[k]),
                             **{name:float(iou[k,j]) for j,name in enumerate(METHODS)},
                             original_stage25=int(stages[k,0,0]),original_stage50=int(stages[k,1,0]),
                             new_stage25=int(stages[k,0,2]),new_stage50=int(stages[k,1,2])))
        write_csv(RESULTS/f'paired_{route}.csv',rows)
        for label,mask in (('all',np.ones(n,bool)),('relation_active',related),('relation_inactive',~related)):
            chosen=iou[mask]
            if not len(chosen):
                continue
            stats=100.*metric_values(chosen).mean(0)
            strata[route,label]=stats
            for j,method in enumerate(METHODS):
                task.append(dict(direction=route,subset=label,method=method,n=len(chosen),
                                 **dict(zip(METRICS,stats[j].tolist(),strict=True))))
            for a,b in pairs:
                ai,bi=METHODS.index(a),METHODS.index(b)
                delta=chosen[:,ai]-chosen[:,bi]
                improve=[int(((chosen[:,ai]>cut)&(chosen[:,bi]<=cut)).sum()) for cut in SUBJECT_CUTS]
                harm=[int(((chosen[:,ai]<=cut)&(chosen[:,bi]>cut)).sum()) for cut in SUBJECT_CUTS]
                paired.append(dict(direction=route,subset=label,method=a,control=b,n=len(chosen),
                                   **dict(zip(METRICS,(stats[ai]-stats[bi]).tolist(),strict=True)),
                                   iou_improve=int((delta>1e-12).sum()),iou_harm=int((delta < -1e-12).sum()),
                                   iou_same=int((abs(delta)<=1e-12).sum()),
                                   acc25_gain=improve[0],acc25_harm=harm[0],
                                   acc50_gain=improve[1],acc50_harm=harm[1]))
            for c,cut in enumerate(SUBJECT_CUTS):
                trial=stages[mask,c]
                for k,method in enumerate(('original_pair','original_full','overlap_retrieval')):
                    count=np.bincount(trial[:,k],minlength=4)
                    assert count.sum()==len(chosen)
                    attribution.append(dict(direction=route,subset=label,method=method,cut=cut,n=len(chosen),
                                            **dict(zip(SUBJECT_STAGES,count.tolist(),strict=True))))
                for orig in range(4):
                    for novel in range(4):
                        transitions.append(dict(direction=route,subset=label,cut=cut,original=SUBJECT_STAGES[orig],
                                              overlap=SUBJECT_STAGES[novel],
                                              n=int(((trial[:,0]==orig)&(trial[:,2]==novel)).sum())))
    for label in ('all','relation_active','relation_inactive'):
        parts=[stats for (route,scope),stats in strata.items() if scope==label]
        stats=np.mean(parts,0)
        for j,method in enumerate(METHODS):
            n=sum(r['n'] for r in task if r['method']==method and r['subset']==label)
            task.append(dict(direction='macro_six',subset=label,method=method,n=n,
                             **dict(zip(METRICS,stats[j].tolist(),strict=True))))
        for a,b in pairs:
            ai,bi=METHODS.index(a),METHODS.index(b)
            entries=[r for r in paired if r['method']==a and r['control']==b and r['subset']==label]
            totals={k:sum(r[k] for r in entries) for k in
                    ('n','iou_improve','iou_harm','iou_same','acc25_gain','acc25_harm','acc50_gain','acc50_harm')}
            paired.append(dict(direction='macro_six',subset=label,method=a,control=b,
                               **dict(zip(METRICS,(stats[ai]-stats[bi]).tolist(),strict=True)),**totals))
    return task,attribution,paired,transitions,endpoints


def recording_intervals(data):
    rng=np.random.default_rng(17680)
    reps=4000
    draws=[]
    rec_counts={}
    for target in PLATFORMS:
        variants=[data[f'{s}_to_{target}'] for s in PLATFORMS if s!=target]
        n,c=np.unique(variants[0]['recording'],return_inverse=True)
        np.testing.assert_array_equal(variants[0]['recording'],variants[1]['recording'])
        rec_counts[target]=len(n)
        weight=rng.multinomial(len(n),np.full(len(n),1/len(n)),size=reps)
        denominator=(weight@np.bincount(c))[:,None,None]
        for v in variants:
            group=np.zeros((len(n),len(METHODS),len(METRICS)))
            np.add.at(group,c,metric_values(v['iou']))
            draws.append(100.*np.einsum('ij,jad->iad',weight,group)/denominator)
    score=np.mean(draws,0)
    differences={}
    for i in (0,1):
        delta=score[:,2]-score[:,i]
        differences[METHODS[i]]=dict(zip(METRICS,np.quantile(delta,[.025,.975],axis=0).T.tolist(),strict=True))
    return dict(unit='target recording; shared draw for its two source directions',
                recordings=rec_counts,repeats=reps,descriptive_intervals_pp=differences)


def main():
    d=diagnosis()
    gate=make_gate()
    model=read(PRIVATE/'RETRIEVAL_SCORE_SEAL.json')
    seal=read(OUTPUT/'PREDICTION_SEAL.json')
    audit=read(RESULTS/'independent_audit.json')
    assert not any(x['FINAL_access'] for x in (model,seal,audit))
    assert model['diagnosis']==d and model['predictions']==sha(OUTPUT/'PREDICTION_SEAL.json')
    assert audit['score_seal']==sha(PRIVATE/'RETRIEVAL_SCORE_SEAL.json')
    assert model['n']==seal['n']==audit['n']==N
    assert all(sha(PRIVATE/p)==h for p,h in model['files'].items())
    dual, cases=dual_tables()
    task, attribution, paired, transitions, outputs=direction_tables()
    for name, rows in [('dual_coverage',dual),('dual_anonymous',cases),('task',task),
                       ('subject_attribution',attribution),('task_deltas',paired),
                       ('subject_stage_transition',transitions)]:
        write_csv(RESULTS/f'{name}.csv',rows)
    json_atomic(RESULTS/'recording_intervals.json',recording_intervals(outputs))
    condition=[row for row in paired if row['direction']=='macro_six' and row['subset']=='all'
               and row['method']=='overlap_retrieval']
    assert len(condition)==2 and all(row['n']==N for row in condition)
    success=all(row['Acc25']>0 and row['Acc50']>0 for row in condition)
    delta={row['control']:{m:row[m] for m in METRICS} for row in condition}
    final=dict(n=N, FINAL_access=False, gate=gate['counts'], retrieval_run_completed=True,
               eligible_to_replace_main=success, primary_deltas_pp=delta,
               audit_sha256=sha(RESULTS/'independent_audit.json'),
               target_updates=0, source_backbone_updates=0)
    json_atomic(RESULTS/'DECISION.json',final)
    json_atomic(PRIVATE/'RESULTS_SEAL.json',dict(n=N, result_hashes={p.name:sha(p) for p in RESULTS.iterdir()},
                                                diagnosis=d, FINAL_access=False))
    print('R176_COMPLETE', final, flush=True)


if __name__ == '__main__':
    args=argparse.ArgumentParser()
    args.add_argument('--gate',action='store_true')
    if args.parse_args().gate:
        print(make_gate())
    else:
        main()
