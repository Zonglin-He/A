"""Judge reference identity before opening the joint subject endpoints."""
from collections import Counter

import numpy as np

from scope169 import HERE, OUT, PANEL, PANEL_N, PLATFORMS, QUERY_ARMS, R168, atomic_json, lock, read, sha, status, roster
from cache169 import known_reference
from panel169 import lookup
from requery169 import tag
from labels166 import NO_OUTPUT, RIGHT, occupied_owners, physical_labels, truth_status
from score165 import many_iou, sensor_values, table, write_table


def sealed():
    z = read(HERE/'QUERY_PREDICTION_SEAL.json')
    assert z['design_lock']==lock('panel_forward',('panel169.py','cache169.py','requery169.py'))
    p = PANEL/'INPUT_PRIVATE.json'
    assert z['panel_sha256']==sha(p)==read(HERE/'PANEL_INPUT_SEAL.json')['roster']
    assert z['n']==PANEL_N and not z['target_identity_read'] and not z['FINAL_access']
    for i,case in enumerate(read(p),1):
        file=tag(i,'requery')+'.npz'
        path=PANEL/'predictions'/file
        assert sha(path)==z['files'][file]
        with np.load(path,allow_pickle=False) as value:
            assert tuple(value['arms'])==QUERY_ARMS
            d=dict(value)
        yield i,case,d


def identity_data(case, pred):
    row=roster()[case['target']][case['ordinal']]
    k,_,_,earlier=lookup(case)
    names,gt,seeds,ids=physical_labels(row,pred['seeds'])
    found=known_reference(row,earlier,k,case['ordinal'],case['target'],case['edge'],
                          case['role'],case['sign'],names,gt)[2]
    assert 0 < found < len(names)
    return row, names, gt, seeds, ids, found


def hit_rank(labels, truth):
    indices=np.flatnonzero(labels==truth)
    return int(indices[0]+1) if len(indices) else 0


def rows_for(case_id, case, d, eval_data, previous):
    row,names,gt,seeds,ids,truth=eval_data
    budget=int(d['common_n'])
    common=occupied_owners(d['common'][:budget],seeds,ids,len(names)) if budget else np.zeros(0,dtype=int)
    supported=case['status']=='supported'
    cohort=dict(panel=case_id,source=case['source'],target=case['target'],frame=case['frame'],recording=case['recording'],
                expression=case['expression'],edge=case['edge'],rewrite_status=case['status'],
                potential_same_class_distractor=int(names.count(names[truth])>1))
    for arm, label in enumerate(QUERY_ARMS):
        n=int(d['native_n'][arm])
        pca=d['natural'][arm,:n]
        natural=occupied_owners(pca,seeds,ids,len(names)) if n else np.zeros(0,dtype=int)
        ranked=np.flatnonzero(np.isfinite(d['bank_score'][arm,:budget]))
        ranked=ranked[np.argsort(-d['bank_score'][arm,ranked],kind='stable')]
        mapped=common[ranked]
        got=int(d['chosen'][arm,0,1])
        assert (len(ranked)>0)==(got>=0)
        natural_ref=truth_status(int(natural[0]),truth) if n else NO_OUTPUT
        mapped_ref=truth_status(int(mapped[0]),truth) if len(mapped) else NO_OUTPUT
        if len(mapped):
            seen=occupied_owners(d['boxes'][arm,0,1:2],seeds,ids,len(names))[0]
            assert seen==mapped[0]
        is_a=arm==0
        if is_a:
            prev=previous[case['source'],case['target'],case['expression'],case['edge']]
            # R168 encoded absent and unowned as one unknown; R169 separates them.
            assert (mapped_ref if mapped_ref!=NO_OUTPUT else -1)==int(prev['independent_visual_ref'])
        else:
            assert bool(d['executed'][arm])==supported
        links=d['links'][arm,:n]
        linked=links[links>=0]
        assert (linked<budget).all()
        stored=d['overlaps'][arm,:n]
        assert ((stored[links>=0]>=.25).all() and (stored[links<0]==0).all())
        transfer_target=np.full(n,NO_OUTPUT)
        transfer_target[links>=0]=common[linked]
        both_diverge=(natural>=0)&(transfer_target>=0)&(natural!=transfer_target)
        assigned_top=(int(transfer_target[0]) if n else NO_OUTPUT)
        top_map_status=(truth_status(assigned_top,truth) if assigned_top!=NO_OUTPUT else NO_OUTPUT)
        native_subject_iou=''
        if arm in (0,1) and d['executed'][arm]:
            box=d['native_subject'][arm:arm+1]
            value=many_iou(box,gt[0])[0]
            assert abs(value-sensor_values(box,gt[0],row['platform'],int(row['suffix'] or 0))[0])<1.e-7
            native_subject_iou=float(value)
        yield dict(**cohort,arm=label,executed=int(d['executed'][arm]),
              token_count=int(d['token_count'][arm]),shared_native_seeds=int(d['query_seeds_shared'][arm]),
              identical_query_slots=int(d['query_slots_equal'][arm]),natural_candidates=n,mapped_candidates=len(ranked),
              original_reference_pool_size=budget, original_reference_pool_covers=int((common==truth).any()),
              natural_top=int(natural_ref), mapped_top=int(mapped_ref), natural_rank=hit_rank(natural,truth),
              mapped_rank=hit_rank(mapped,truth), native_iou_original_subject=native_subject_iou,
              original_subject_as_natural_reference=int(n>0 and natural[0]==0),
              original_subject_as_mapped_reference=int(len(mapped)>0 and mapped[0]==0),
              native_top_donor_mapped_status='' if is_a else top_map_status,
              mapped_donors='' if is_a else len(linked),links_between_different_known_objects='' if is_a else int(both_diverge.sum()),
              true_donor_mapped_to_other_object='' if is_a else int(((natural==truth)&both_diverge).sum()),
              true_donor_mapped_to_unknown='' if is_a else int(((natural==truth)&(transfer_target==-1)).sum()),
              true_donor_unlinked='' if is_a else int(((natural==truth)&(links<0)).sum()),
              full_forward_seconds=float(d['seconds'][arm]), scoring_and_association_seconds=float(d['readout_seconds'][arm]))


def describe(rows):
    output=[]
    for scope, filtered in (('all_attempts',rows),('paired_supported',[x for x in rows if x['rewrite_status']=='supported'])):
        for group in ('pooled',*PLATFORMS):
            choose=[x for x in filtered if group=='pooled' or x['target']==group]
            for label in QUERY_ARMS:
                r=[x for x in choose if x['arm']==label]
                if not r:
                    continue
                item=dict(scope=scope,target=group,arm=label,n=len(r),frames=len({x['frame'] for x in r}),
                          recordings=len({x['recording'] for x in r}),executed=sum(x['executed'] for x in r),
                          original_reference_pool_covers=sum(x['original_reference_pool_covers'] for x in r),
                          natural_own_subject=sum(x['original_subject_as_natural_reference'] for x in r),
                          mapped_own_subject=sum(x['original_subject_as_mapped_reference'] for x in r),
                          median_full_forward_seconds=float(np.median([x['full_forward_seconds'] for x in r if x['executed']])) if any(x['executed'] for x in r) else '',
                          median_association_and_scoring_seconds=float(np.median([x['scoring_and_association_seconds'] for x in r if x['executed']])) if any(x['executed'] for x in r) else '',
                          total_full_forward_seconds=sum(x['full_forward_seconds'] for x in r))
                for kind in ('natural','mapped'):
                    rank=[x[kind+'_rank'] for x in r]
                    item.update({kind+'_correct':sum(x[kind+'_top']==RIGHT for x in r),
                                 kind+'_unknown_identity':sum(x[kind+'_top']==-1 for x in r),
                                 kind+'_no_output':sum(x[kind+'_top']==NO_OUTPUT for x in r),
                                 kind+'_cover':sum(bool(v) for v in rank),kind+'_mrr':sum(1/v for v in rank if v)/len(r)})
                native_correct=[x for x in r if x['natural_top']==RIGHT and x['native_top_donor_mapped_status']!='']
                link_categories=Counter(x['native_top_donor_mapped_status'] for x in native_correct)
                item.update(correct_natural_top_direct_link_correct=link_categories[1],
                            correct_natural_top_direct_link_wrong=link_categories[0],
                            correct_natural_top_direct_link_owner_unknown=link_categories[-1],
                            correct_natural_top_unlinked=link_categories[NO_OUTPUT],
                            true_donor_mapped_to_other_object=sum(x['true_donor_mapped_to_other_object'] or 0 for x in r),
                            links_between_different_known_objects=sum(x['links_between_different_known_objects'] or 0 for x in r))
                output.append(item)
    return output


def paired_identity(rows):
    selected=[r for r in rows if r['rewrite_status']=='supported']
    chosen={a:{r['panel']:r for r in selected if r['arm']==a} for a in QUERY_ARMS}
    out=[]
    for arm,control in ((1,0),(2,0),(2,1),(3,0),(3,1),(3,2)):
        method,baseline=QUERY_ARMS[arm],QUERY_ARMS[control]
        assert set(chosen[method])==set(chosen[baseline])
        record=dict(arm=method,control=baseline,n=len(chosen[method]))
        for label in ('natural','mapped'):
            better,worse=[],[]
            for i,x in chosen[method].items():
                a=x[label+'_top']==RIGHT
                b=chosen[baseline][i][label+'_top']==RIGHT
                if a and not b:
                    better.append(x)
                elif b and not a:
                    worse.append(x)
            record.update({label+'_rescue':len(better), label+'_harm':len(worse),
                           label+'_rescue_frames':len({x['frame'] for x in better}),
                           label+'_harm_frames':len({x['frame'] for x in worse})})
        out.append(record)
    return out


def score_identity():
    design=lock('panel_score',('score169.py',))
    assert (HERE/'QUERY_PREDICTION_SEAL.json').exists()
    parent=R168/'results/relation_edges.csv'
    assert sha(parent)==read(R168/'EVALUATION_AUDIT.json')['files'][parent.name]
    previous={(v['source'],v['target'],v['expression'],int(v['edge'])):v for v in table(parent) if v['independently_verified']=='1'}
    result=[]
    delta=0.
    for n,case,d in sealed():
        result.extend(rows_for(n,case,d,identity_data(case,d),previous))
        delta=max(delta,float(d['replay_diff']))
    assert len(result)==PANEL_N*len(QUERY_ARMS)
    stats=describe(result)
    paired=paired_identity(result)
    checked={r['arm']:r for r in stats if r['scope']=='paired_supported' and r['target']=='pooled'}
    left,right=(checked[a] for a in QUERY_ARMS[:2])
    eligible=[a for a in QUERY_ARMS[2:] if checked[a]['natural_correct']>right['natural_correct'] and
              checked[a]['mapped_correct']>max(left['mapped_correct'],right['mapped_correct'])]
    assert checked[QUERY_ARMS[0]]['n']==sum(r['executed'] for r in result if r['arm']==QUERY_ARMS[3])
    formats=[]
    for a in QUERY_ARMS[:2]:
        v=[float(r['native_iou_original_subject']) for r in result if r['arm']==a and r['rewrite_status']=='supported']
        formats.append(dict(arm=a,n=len(v),mIoU=float(np.mean(v)*100),Acc25_count=sum(x>.25 for x in v),Acc50_count=sum(x>.5 for x in v)))
    files={}
    for key,table_rows in (('query_identity_rows',result),('query_identity_summary',stats),('query_identity_paired',paired),('canonicalization_control',formats)):
        p=OUT/(key+'.csv')
        write_table(p,table_rows)
        files[p.name]=sha(p)
    atomic_json(HERE/'QUERY_IDENTITY_AUDIT.json',dict(n=PANEL_N,paired_n=int(left['n']),physical_frames=int(left['frames']),
                eligible_for_joint=eligible, same_point_sample=True, max_A_reference_score_error=delta,
                design_lock=design, query_seal=sha(HERE/'QUERY_PREDICTION_SEAL.json'), files=files, FINAL_access=False))
    print('R169 reference identity gate', eligible, flush=True)
    return eligible


def score_joint(eligible):
    """Use the already sealed joint decisions only for variants passing the identity screen."""
    design=lock('panel_score',('score169.py',))
    identity=read(HERE/'QUERY_IDENTITY_AUDIT.json')
    assert identity['design_lock']==design and identity['eligible_for_joint']==eligible
    if not eligible:
        atomic_json(HERE/'QUERY_JOINT_AUDIT.json',dict(state='stopped_at_identity',eligible=[],FINAL_access=False))
        return
    entered={QUERY_ARMS.index(a):a for a in eligible}
    report=[]
    max_error=0.
    for n,case,d in sealed():
        if case['status']!='supported':
            continue
        row,names,gt,seeds,ids,truth=identity_data(case,d)
        for idx,arm in entered.items():
            pred=d['boxes'][idx]
            iou=many_iou(pred[:,0],gt[0])
            proof=sensor_values(pred[:,0],gt[0],row['platform'],int(row['suffix'] or 0))
            max_error=max(max_error,float(np.abs(iou-proof).max()))
            assert max_error<1.e-7 and all(np.array_equal(iou>t,proof>t) for t in (.25,.50))
            owners=occupied_owners(pred,seeds,ids,len(names))
            data=dict(panel=n,arm=arm,source=case['source'],target=case['target'],frame=case['frame'],
                      recording=case['recording'],subject_switch=int(d['chosen'][idx,0,0]!=d['chosen'][idx,1,0]))
            for j,mode in enumerate(('independent','joint')):
                ref = NO_OUTPUT if d['chosen'][idx,j,1]<0 else truth_status(owners[j,1],truth)
                data.update({mode+'_iou':float(iou[j]),mode+'_ref':int(ref),
                             mode+'_subject':truth_status(owners[j,0],0),
                             mode+'_both':int(owners[j,0]==0 and owners[j,1]==truth)})
            report.append(data)
    aggregates=[]
    for arm in eligible:
        for group in ('pooled',*PLATFORMS):
            r=[x for x in report if x['arm']==arm and (group=='pooled' or group==x['target'])]
            n=len(r)
            u,v=[np.asarray([x[k+'_iou'] for x in r]) for k in ('independent','joint')]
            item=dict(arm=arm,target=group,n=n,frames=len({x['frame'] for x in r}),
                      ref_no_output=sum(x['independent_ref']==NO_OUTPUT for x in r),
                      subject_switches=sum(x['subject_switch'] for x in r))
            if group=='pooled':
                assert n==identity['paired_n']
            for label,values in (('independent',u),('joint',v)):
                item.update({label+'_mIoU':float(np.mean(values)*100), label+'_Acc25':100*float(np.mean(values>.25)),
                             label+'_Acc50':100*float(np.mean(values>.5)),label+'_Acc25_count':int((values>.25).sum()),
                             label+'_Acc50_count':int((values>.5).sum()),
                             label+'_subject_correct':sum(x[label+'_subject']==RIGHT for x in r),
                             label+'_ref_correct':sum(x[label+'_ref']==RIGHT for x in r),
                             label+'_both_correct':sum(x[label+'_both'] for x in r)})
            for name,c,d in (('IoU',v,u),('Acc25',v>.25,u>.25),('Acc50',v>.5,u>.5)):
                delta=c.astype(float)-d.astype(float)
                item[name+'_rescues']=int((delta>1.e-10).sum())
                item[name+'_harms']=int((delta< -1.e-10).sum())
            aggregates.append(item)
    details,summary=(OUT/(n+'.csv') for n in ('query_joint_rows','query_joint_summary'))
    write_table(details,report)
    write_table(summary,aggregates)
    atomic_json(HERE/'QUERY_JOINT_AUDIT.json',dict(state='scored_same_information',eligible=eligible,paired_rows=len(report),
                max_sensor_error=max_error,files={p.name:sha(p) for p in (details,summary)},FINAL_access=False))
    print('R169 panel same-information comparison',len(report),flush=True)


if __name__=='__main__':
    status('panel_scoring')
    score_joint(score_identity())
