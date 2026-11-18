"""Audit the fixed predictions against official geometry and blind physical identities."""
from collections import Counter
from math import fsum

import numpy as np

from scope170 import ARMS, HERE, INPUT, ORDER, OUT, PLATFORMS, VARIANTS, WORK, atomic_json, lock, read, sha, status
from review170 import ADJUDICATION, LIST, verify_predictions
from labels166 import MISSING_TRUTH, NO_OUTPUT, RIGHT, occupied_owners, physical_labels, truth_status
from score165 import many_iou, sensor_values, write_table

COMPARISONS=(('D_joint','r158_pca'),('D_joint','D_independent'),('D_joint','C_joint'),
             ('C_joint','C_independent'))
METRICS=('mIoU','Acc25','Acc50')


def predictions(entry, source, receipt):
    key=f"{entry['case']:02d}_{source}.npz"
    path=WORK/'predictions'/key
    assert sha(path)==receipt['files'][key]
    with np.load(path,allow_pickle=False) as p:
        assert str(p['protocol'])==receipt['prediction_lock'] and tuple(p['arms'])==ARMS
        return dict(p)


def label(lookup,decision):
    if decision['status']!='verified':
        return MISSING_TRUTH
    return lookup['display_to_annotation'][decision['box']]


def status_for(gold,owner,provided=True):
    if gold<0:
        return MISSING_TRUTH
    return truth_status(owner,gold) if provided else NO_OUTPUT


def hits_at(values,gold):
    if gold<0:
        return ''
    hit=np.flatnonzero(values==gold)
    return int(hit[0]+1) if len(hit) else 0


def attribution(d,j,meta,gt_id,manual_subject,seeds,ids,n_object):
    n,pool,linked=int(d['native_n'][j]),int(d['common_n']),int(d['mapped_n'][j])
    native=occupied_owners(d['natural'][j,:n],seeds,ids,n_object) if n else np.empty(0,dtype=int)
    common=occupied_owners(d['common'][:pool],seeds,ids,n_object) if pool else np.empty(0,dtype=int)
    links=d['links'][j,:n]
    found=(links>=0)
    assert (links[found]<pool).all()
    assert (d['witnesses'][j,:n][found]>=.25).all() and (d['witnesses'][j,:n][~found]==0).all()
    destination=np.full(n,NO_OUTPUT)
    destination[found]=common[links[found]]
    evidence=d['bank_scores'][j,:pool]
    available=np.flatnonzero(np.isfinite(evidence))
    assert len(available)==linked and set(available)==set(links[found])
    ranked=available[np.argsort(-evidence[available],kind='stable')]
    mapped=common[ranked]
    expected_ref=(-1 if not linked else mapped[0])
    if linked:
        # Confirm the decision actually emits this top mapped physical endpoint.
        chosen=occupied_owners(d['boxes'][j,0,1:2],seeds,ids,n_object)[0]
        assert expected_ref==chosen and d['choices'][j,0,1]>=0
    else:
        assert d['choices'][j,0,1]<0
    natural_rank=hits_at(native,gt_id)
    mapped_rank=hits_at(mapped,gt_id)
    true=native==gt_id if gt_id>=0 else np.zeros(n,bool)
    in_pool=int((common==gt_id).any()) if gt_id>=0 else ''
    preserved=int((destination[true]==gt_id).sum()) if gt_id>=0 else ''
    if gt_id<0:
        fate='truth_unresolved'
    elif not d['executed'][j]:
        fate='rewrite_fallback'
    elif not true.any():
        fate='native_reference_missing' if not mapped_rank else 'mapped_correct_with_native_owner_unknown_or_other'
    elif mapped_rank:
        fate='reference_represented_after_mapping'
    elif not in_pool:
        fate='correct_identity_absent_from_fixed_bank'
    elif not found[true].any():
        fate='all_correct_donors_unlinked'
    else:
        fate='correct_donors_reach_only_unknown_or_other_owners'
    if gt_id>=0 and n and native[0]==gt_id:
        t=destination[0]
        top_link=('unlinked' if t==NO_OUTPUT else ('owner_unknown' if t<0 else ('correct' if t==gt_id else 'wrong')))
        if mapped_rank==1:
            top_fate='retained_correct_top1'
        elif mapped_rank:
            top_fate='correct_reference_mapped_but_reranked'
        else:
            top_fate=fate
    else:
        top_link=''
        top_fate='new_mapped_top1' if mapped_rank==1 else ''
    own_link=int(n>0 and native[0]==manual_subject) if manual_subject>=0 else ''
    own_final=int(linked>0 and mapped[0]==manual_subject) if manual_subject>=0 else ''
    return dict(**meta,variant=VARIANTS[j],native_candidates=n,mapped_candidates=linked,physical_pool=pool,
                 executed=int(d['executed'][j]),active_left=int(d['active_relation'][j]),
                 native_rank=natural_rank,mapped_rank=mapped_rank,correct_in_fixed_pool=in_pool,
                 natural_top=status_for(gt_id,native[0] if n else -1,n>0),
                 mapped_top=status_for(gt_id,expected_ref,linked>0),
                 reference_coverage_outcome=fate,top1_outcome=top_fate,original_top1_link=top_link,
                 correct_natural_donors=int(true.sum()) if gt_id>=0 else '',
                 correct_natural_donors_that_map_to_correct_box=preserved,
                 correct_donors_unlinked=int((~found&true).sum()) if gt_id>=0 else '',
                 correct_donors_link_to_unknown=int((true&(destination==-1)).sum()) if gt_id>=0 else '',
                 correct_donors_link_to_wrong=int((true&(destination>=0)&(destination!=gt_id)).sum()) if gt_id>=0 else '',
                 natural_top_is_original_subject=own_link,mapped_top_is_original_subject=own_final,
                 same_slot_seeds=int(d['same_slot_seeds'][j]),forward_seconds=float(d['forward_seconds'][j]),
                 association_seconds=float(d['association_seconds'][j]))


def score_case(entry, blind, alias, receipt):
    row=entry['row']
    target=entry['target']
    gold_subject=label(alias,blind['subject'])
    gold_ref=label(alias,blind['reference'])
    assert min(gold_subject,gold_ref)<0 or gold_subject!=gold_ref
    report,mapping=[],[]
    error=0.
    for source in PLATFORMS:
        if source==target:
            continue
        pred=predictions(entry,source,receipt)
        names,gt,seeds,seed_ids=physical_labels(row,pred['seeds'])
        assert len(names)==alias['inventory_size'] and max(gold_subject,gold_ref)<len(names)
        method=pred['method_boxes']
        assert np.array_equal(method[0,0],method[1,0]) and np.array_equal(method[0,0],method[3,0])
        assert len(method)==len(ARMS) and np.isfinite(method[:,0]).all()
        iou=many_iou(method[:,0],gt[0])
        sensor=sensor_values(method[:,0],gt[0],target,int(row['suffix'] or 0))
        error=max(error,float(np.abs(iou-sensor).max()))
        assert error<1.e-7 and all(np.array_equal(iou>t,sensor>t) for t in (.25,.50))
        if gold_subject>=0:
            audited_iou=many_iou(method[:,0],gt[gold_subject])
            audited_sensor=sensor_values(method[:,0],gt[gold_subject],target,int(row['suffix'] or 0))
            error=max(error,float(np.abs(audited_iou-audited_sensor).max()))
            assert error<1.e-7 and all(np.array_equal(audited_iou>t,audited_sensor>t) for t in (.25,.50))
        else:
            audited_iou=['']*len(method)
        owners=occupied_owners(method,seeds,seed_ids,len(names))
        meta=dict(case=entry['case'],source=source,target=target,recording=entry['recording'],
                  rewrite=entry['rewrite']['status'],subject_label=blind['subject']['status'],
                  reference_label=blind['reference']['status'],verified_pair=int(min(gold_subject,gold_ref)>=0),
                  subject_label_agrees_with_official=int(gold_subject==0) if gold_subject>=0 else '')
        record=meta.copy()
        for k,arm in enumerate(ARMS):
            i=(k-1)//2
            is_fallback=k>0 and pred['mapped_n'][i]==0
            if is_fallback:
                assert np.array_equal(method[k,0],method[0,0])
                assert not np.isfinite(method[k,1]).any()
            have_ref=k>0 and np.isfinite(method[k,1]).all()
            sub=status_for(gold_subject,owners[k,0])
            ref=status_for(gold_ref,owners[k,1],have_ref)
            dual=int(sub==ref==RIGHT) if k and meta['verified_pair'] else ''
            record.update({arm+'_iou':float(iou[k]),arm+'_verified_subject_iou':audited_iou[k],
                           arm+'_subject_identity':int(sub),arm+'_reference_identity':int(ref),arm+'_dual_correct':dual,
                           arm+'_has_reference':int(have_ref),arm+'_fallback':int(is_fallback),
                           arm+'_active_left':int(k>0 and k%2==0 and pred['active_relation'][i]),
                           arm+'_subject_changed':int(not np.array_equal(method[0,0],method[k,0]))})
        report.append(record)
        for j in range(len(VARIANTS)):
            mapping.append(attribution(pred,j,meta,gold_ref,gold_subject,seeds,seed_ids,len(names)))
    return report,mapping,error


def group(rows):
    found={(s,t):[x for x in rows if (x['source'],x['target'])==(s,t)] for s,t in ORDER}
    direction={f'{s}_to_{t}':[x] for (s,t),x in found.items() if x}
    targets={t:[r for (s,v),r in found.items() if v==t and r] for t in PLATFORMS}
    return dict(six_equal=[x for x in found.values() if x],pooled=[rows],**{k:v for k,v in targets.items() if v},**direction)


def parts(rows,arm):
    y=np.array([float(x[arm+'_iou']) for x in rows])
    return np.c_[y,y>.25,y>.50]


def summarize(rows):
    summary=[]
    paired=[]
    for scope,cohort in (('all',rows),('rewrite_supported',[x for x in rows if x['rewrite']=='supported']),
                          ('blind_pair',[x for x in rows if x['verified_pair']])):
        for key,value in group(cohort).items():
            x=[r for p in value for r in p]
            if not x:
                continue
            for arm in ARMS:
                means=100*np.mean([parts(p,arm).mean(0) for p in value],axis=0)
                s=[float(r[arm+'_verified_subject_iou']) for r in x if r[arm+'_verified_subject_iou']!='']
                summary.append(dict(scope=scope,group=key,arm=arm,n=len(x),directions=len({(r['source'],r['target']) for r in x}),
                                    frames=len({r['case'] for r in x}),recordings=len({r['recording'] for r in x}),
                                    **dict(zip(METRICS,means,strict=True)),subject_verified_n=len(s),
                                    verified_subject_Acc25_count=sum(v>.25 for v in s),verified_subject_Acc50_count=sum(v>.5 for v in s),
                                    blind_subject_correct=sum(r[arm+'_subject_identity']==RIGHT for r in x),
                                    blind_reference_n=sum(r['reference_label']=='verified' for r in x),
                                    blind_reference_correct=sum(r[arm+'_reference_identity']==RIGHT for r in x),
                                    both_labels_n=sum(r['verified_pair'] for r in x),
                                    dual_correct=sum(r[arm+'_dual_correct']==1 for r in x) if arm!='r158_pca' else '',
                                    physical_reference_outputs=sum(r[arm+'_has_reference'] for r in x),
                                    unknown_emitted_references=sum(r[arm+'_reference_identity']==-1 for r in x),
                                    fallbacks=sum(r[arm+'_fallback'] for r in x),active_left=sum(r[arm+'_active_left'] for r in x),
                                    subject_box_changes=sum(r[arm+'_subject_changed'] for r in x)))
            for method,control in COMPARISONS:
                diffs=[parts(p,method)-parts(p,control) for p in value]
                mean=100*np.mean([z.mean(0) for z in diffs],axis=0)
                z=np.concatenate(diffs)
                item=dict(scope=scope,group=key,arm=method,control=control,n=len(x))
                for j,metric in enumerate(METRICS):
                    rescue=[r for r,d in zip(x,z[:,j],strict=True) if d>1.e-10]
                    harm=[r for r,d in zip(x,z[:,j],strict=True) if d< -1.e-10]
                    item.update({metric+'_pp':float(mean[j]),metric+'_rescues':len(rescue),metric+'_harms':len(harm),
                                 metric+'_rescue_frames':len({r['case'] for r in rescue}),metric+'_harm_frames':len({r['case'] for r in harm}),
                                 metric+'_rescue_recordings':len({r['recording'] for r in rescue}),metric+'_harm_recordings':len({r['recording'] for r in harm}),
                                 metric+'_rescues_with_dual_correct':sum(r[method+'_dual_correct']==1 for r in rescue)})
                pair=[r for r in x if r[method+'_dual_correct']!='' and r[control+'_dual_correct']!='']
                item.update(dual_evaluable=len(pair),dual_rescue=sum(r[method+'_dual_correct']>r[control+'_dual_correct'] for r in pair),
                            dual_harm=sum(r[method+'_dual_correct']<r[control+'_dual_correct'] for r in pair))
                paired.append(item)
    return summary,paired


def clusters(rows):
    records=[]
    for key in sorted({x['recording'] for x in rows}):
        excluded=[x for x in rows if x['recording']==key]
        kept=group([x for x in rows if x['recording']!=key])['six_equal']
        for arm,control in COMPARISONS:
            delta=100*np.mean([(parts(x,arm)-parts(x,control)).mean(0) for x in kept],axis=0) if len(kept)==6 else ('',)*3
            records.append(dict(excluded_recording=key,target=excluded[0]['target'],n_excluded=len(excluded),
                    n=len(rows)-len(excluded),coverage_directions=len(kept),arm=arm,control=control,
                    **{name+'_pp':v for name,v in zip(METRICS,delta,strict=True)}))
    return records


def map_summary(rows):
    sizes=[]
    fates=[]
    for v in VARIANTS:
        for target in ('pooled',*PLATFORMS):
            a=[r for r in rows if r['variant']==v and (target=='pooled' or r['target']==target)]
            valid=[r for r in a if r['reference_label']=='verified']
            n=len(valid)
            item=dict(variant=v,target=target,n=len(a),verified_reference_directions=n,
                      verified_reference_frames=len({r['case'] for r in valid}),
                      executed=sum(r['executed'] for r in a),no_link_after_rewrite=sum(r['executed'] and r['mapped_candidates']==0 for r in a),
                      covered_by_fixed_bank=sum(r['correct_in_fixed_pool']==1 for r in valid),
                      native_top1_correct=sum(r['native_rank']==1 for r in valid),mapped_top1_correct=sum(r['mapped_rank']==1 for r in valid),
                      native_coverage=sum(r['native_rank']>0 for r in valid),mapped_coverage=sum(r['mapped_rank']>0 for r in valid),
                      natural_mrr=fsum(1/r['native_rank'] for r in valid if r['native_rank'])/n if n else '',
                      mapped_mrr=fsum(1/r['mapped_rank'] for r in valid if r['mapped_rank'])/n if n else '',
                      correct_native_top_lost=sum(r['native_rank']==1 and r['mapped_rank']!=1 for r in valid),
                      wrong_native_top_recovered=sum(r['native_rank']!=1 and r['mapped_rank']==1 for r in valid))
            sizes.append(item)
            for view,keep in (('coverage',valid),('natural_top1_correct',[r for r in valid if r['native_rank']==1])):
                counts=Counter((r['reference_coverage_outcome'] if view=='coverage' else r['top1_outcome']) for r in keep)
                for cause,count in sorted(counts.items()):
                    fates.append(dict(variant=v,target=target,view=view,cause=cause,n=count))
    return sizes,fates


def score():
    design=lock('scoring',('score170.py',))
    proof=verify_predictions()
    label_seal=read(HERE/'VISUAL_TRUTH_SEAL.json')
    assert label_seal['adjudication']==sha(ADJUDICATION)
    assert label_seal['certificate']==sha(HERE/'REVIEW_PANEL_SEAL.json')
    assert read(HERE/'REVIEW_PANEL_SEAL.json')['registry']==sha(LIST)
    assert not label_seal['FINAL_access'] and label_seal['n']*2==proof['n']
    status('scoring',physical_frames=proof['physical_frames'],direction_records=proof['n'])
    rows=[]
    maps=[]
    max_error=0.
    for entry,blind,p in zip(read(INPUT),read(ADJUDICATION),read(LIST),strict=True):
        assert entry['case']==blind['case']==p['case']
        a,b,err=score_case(entry,blind,p,proof)
        rows.extend(a)
        maps.extend(b)
        max_error=max(max_error,err)
    assert len(rows)==proof['n'] and len(maps)==2*proof['n']
    assert len({(r['case'],r['source']) for r in rows})==len(rows)
    assert all((r['r158_pca_iou']==r['D_independent_iou']==r['C_independent_iou']) for r in rows)
    metrics,deltas=summarize(rows)
    coverage,fates=map_summary(maps)
    OUT.mkdir(exist_ok=True)
    files={}
    for name,x in (('directions_and_cases',rows),('mapping_cases',maps),('metrics',metrics),('paired',deltas),
                   ('mapping_summary',coverage),('mapping_loss_fates',fates),('recording_deletion',clusters(rows))):
        path=OUT/(name+'.csv')
        write_table(path,x)
        files[path.name]=sha(path)
    atomic_json(HERE/'EVALUATION_AUDIT.json',dict(n=len(rows),mapping_rows=len(maps),scoring_lock=design,
                 max_sensor_iou_error=max_error,source_prediction_seal=sha(HERE/'PREDICTION_SEAL.json'),
                 blind_truth_seal=sha(HERE/'VISUAL_TRUTH_SEAL.json'),files=files,FINAL_access=False))
    print('R170 scoring done',len(rows),len(maps),max_error,flush=True)


if __name__=='__main__':
    score()
