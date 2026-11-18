"""Score the sealed h0 ablations and locate missing annotated role candidates."""
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from multiprocessing import get_context
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scope166 import ARMS,HERE,ORDER,PROTOCOL,W165,WORK,atomic_json,inputs,lock,read,sha,status
from score165 import anonymizer,many_iou,sensor_values,table,write_table
from mechanisms162 import box_owner
from labels166 import (MISSING_TRUTH,NO_OUTPUT,RIGHT,UNKNOWN,WRONG,box_masks,joint_status,key_for,
                       label_pair,mask_owners,occupied_owners,parser_record,physical_labels,truth_status)

OUT=HERE/'results'
PAIR_ARMS=(*ARMS,'r158_r88_reference')
METRICS=('mIoU','Acc25','Acc50','subject_identity','subject_box_proxy')
IDENTITY=('subject','reference','both')
VARIANTS=('all','parsed_direct','camera_horizontal','active_factor')
TRANSFER=('a256','r88','h0_pool','h0_eligible','h0_beam')


@lru_cache(maxsize=1)
def verified_prediction_inputs():
    lock()
    rows,_=inputs()
    seal=read(HERE/'PREDICTION_SEAL.json')
    assert seal['n']==18620 and not seal['FINAL_access'] and not seal['target_label_read']
    assert seal['protocol_sha256']==sha(PROTOCOL)
    return rows,seal


def row_scope(row, edges, active):
    graph=parser_record(row['id'])['graph']
    any_parse=any((e['subject']=='T')!=(e['reference']=='T') for e in graph['relations'])
    direct=any((a==0)!=(b==0) for a,b,_ in edges)
    assert not direct or any_parse
    assert not active or direct
    return dict(parsed_direct=int(any_parse),camera_horizontal=int(direct),active_factor=int(active))


def score_edges(row, bank, raw, all_edges, objects, names, seeds, seed_ids, case, owners):
    direct=list(dict.fromkeys((int(b if a==0 else a),int(k if a==0 else -k))
                               for a,b,k in all_edges if (a==0)!=(b==0)))
    if not direct:
        return []
    aux=occupied_owners(bank['diagnostic'],seeds,seed_ids,len(objects))
    candidate=occupied_owners(bank['starts'],seeds,seed_ids,len(objects))
    # Native A256/R88 evidence uses axis-box seed support, as in build88.
    axis_masks=(np.abs(seeds[None]-raw[:,None,:3])<=raw[:,None,3:6]/2).all(-1)
    axis_owners=mask_owners(axis_masks,seed_ids,len(objects))
    camera=SimpleNamespace(edges=all_edges,role_spans=bank['spans'],camera_intrinsic=bank['camera_intrinsic'],
                           model_to_camera=bank['model_to_camera'],distortion=bank['distortion'],width=int(bank['width']))
    result=[]
    for serial,(r,side) in enumerate(direct,1):
        truth,ref=label_pair(row,camera,0,r,side,names,objects)
        item=dict(source=case['source'],target=case['target'],recording=case['recording'],frame=case['frame'],
                  expression=case['expression'],edge_number=serial,truth=truth,visible_relation=case['visible_relation'],
                  active_factor=case['active_factor'])
        for a,who in zip(PAIR_ARMS,(*owners,aux)):
            s=truth_status(who[0],0)
            t=(NO_OUTPUT if a=='r158_pca' else MISSING_TRUTH) if ref<0 else (
                NO_OUTPUT if a=='r158_pca' else truth_status(who[r],ref))
            joint=t if t in (MISSING_TRUTH,NO_OUTPUT) else joint_status(s,t)
            item.update({f'{a}_subject':s,f'{a}_reference':t,f'{a}_both':joint})
        for metric in TRANSFER:
            subject=reference=MISSING_TRUTH
            if ref>=0:
                stage={
                    'a256':(axis_owners,axis_owners),
                    'r88':tuple(axis_owners[q[q>=0]] for q in bank['routes'][[0,r]]),
                    'h0_pool':(candidate[0],candidate[r]),
                    'h0_eligible':tuple(candidate[e,bank['eligibility'][e]] for e in (0,r)),
                    'h0_beam':tuple(candidate[e,bank['in_beam'][e]] for e in (0,r)),
                }[metric]
                subject=int((stage[0]==0).any())
                reference=int((stage[1]==ref).any())
            item.update({metric+'_subject':subject,metric+'_reference':reference,
                         metric+'_both':MISSING_TRUTH if ref<0 else int(subject and reference)})
        result.append(item)
    return result


def score_direction(direction):
    rows,seal=verified_prediction_inputs()
    source,target=direction
    name=f'{source}_to_{target}'
    folder=WORK/'ordinary'/name
    parent=W165/'ordinary'/name
    outcome_dir=WORK/'scalars'/name
    outcome_dir.mkdir(exist_ok=True,parents=True)
    prediction=read(folder/'SEAL.json')
    assert sha(folder/'SEAL.json')==seal['directions'][name]['seal']
    assert not prediction['target_label_read'] and not prediction['FINAL_access']
    prior=read(parent/'SEAL.json')
    assert prediction['parent_forward_sha256']==sha(parent/'SEAL.json')
    older=parent/'IOU_AUDIT_PRIVATE.npz'
    with np.load(older,allow_pickle=False) as f:
        previous=f['values'][:,-1]
        assert f['ordinals'].tolist()==list(range(len(previous)))
    alias=anonymizer()
    all_rows,all_relations=[],[]
    max_error,base_error=0.,0.
    changed_uncorrected_angle=0
    for key,digest in sorted(prediction['files'].items()):
        f=folder/key
        assert sha(f)==digest and sha(parent/key)==prior['files'][key]
        with np.load(f,allow_pickle=False) as p,np.load(parent/key,allow_pickle=False) as r:
            assert tuple(p['arms'])==ARMS and str(p['protocol'])==sha(PROTOCOL)
            assert str(p['parent_block_sha256'])==prior['files'][key]
            pred=dict(p)
            raw=r['raw']
            np.testing.assert_array_equal(p['ordinals'],r['ordinals'])
            np.testing.assert_array_equal(p['boxes'][:,0,0],r['boxes'][:,-1])
        for i,j in enumerate(pred['ordinals']):
            row=rows[target][int(j)]
            op={k:pred[k][i] for k in pred if k not in ('edges','edge_offsets','parent_block_sha256','protocol','arms')}
            edges=pred['edges'][pred['edge_offsets'][i]:pred['edge_offsets'][i+1]]
            names,boxes,seeds,seed_ids=physical_labels(row,op['seeds'])
            subject=op['boxes'][:,0]
            scores=many_iou(subject,boxes[0])
            independently=sensor_values(subject,boxes[0],target,int(row['suffix'] or 0))
            error=float(np.abs(independently-scores).max())
            if error>1e-7 or any(not np.array_equal(independently>cut,scores>cut) for cut in (.25,.50)):
                raise ValueError(f'Sensor coordinate check failed on {name} ordinal {j}')
            max_error=max(max_error,error)
            base_error=max(base_error,abs(float(scores[0])-float(previous[j])))
            owners=occupied_owners(op['boxes'],seeds,seed_ids,len(names))
            head=[truth_status(q,0) for q in owners[:,0]]
            secondary=[truth_status(box_owner(b,boxes),0) for b in subject]
            _,path,slot=key_for(row)
            image=Path(path)
            result=dict(source=source,target=target,recording=alias('rec',image.parent.parent.as_posix()),
                        frame=alias('frame',image),expression=alias('obs',(str(image),slot)),
                        **row_scope(row,edges,bool(op['active_relation'])),
                        visible_relation=int(op['visible_relation']))
            result.update({f'{a}_iou':float(v) for a,v in zip(ARMS,scores)})
            result.update({f'{a}_subject':v for a,v in zip(ARMS,head)})
            result.update({f'{a}_subject_box':v for a,v in zip(ARMS,secondary)})
            result.update({f'{a}_switch':int(not np.array_equal(b,subject[0])) for a,b in zip(ARMS,subject)})
            result['full_new_subject']=int(op['origins'][1,0]==-1)
            result['full_fallback_subject']=int(op['origins'][1,0]==-2)
            result['subject_bank_slots']=int(op['budget'][0,0])
            result['subject_original_slots']=int(op['budget'][0,1])
            result['any_bank_deficit']=int((op['budget'][:,1]<op['budget'][:,0]).any())
            result['any_new_hypotheses']=int(op['budget'][:,2].any())
            result['new_hypotheses_total']=int(op['budget'][:,2].sum())
            result['native_h0_iou']=float(many_iou(op['native_full'][:1],boxes[0])[0])
            changed_uncorrected_angle+=int(op['native_full'][0,6]!=subject[1,6])
            result['legacy_ref_slot_deficit']=int(np.sum(op['budget'][1:,0]-op['budget'][1:,1]))
            all_rows.append(result)
            all_relations.extend(score_edges(row,op,raw[i],edges,boxes,names,seeds,seed_ids,result,owners))
        if len(all_rows)%1024==0 or len(all_rows)==len(previous):
            print('R166 scored',name,len(all_rows),'/',len(previous),flush=True)
    assert len(all_rows)==len(previous)==prediction['n'] and base_error<1e-11
    write_table(outcome_dir/'rows.csv',all_rows)
    write_table(outcome_dir/'edges.csv',all_relations)
    return dict(n=len(all_rows),edges=len(all_relations),geometry_error=max_error,baseline_error=base_error,
                changed_native_h0_angle=changed_uncorrected_angle,
                rows_sha256=sha(outcome_dir/'rows.csv'),edges_sha256=sha(outcome_dir/'edges.csv'))


def metric_array(rows,arm):
    iou=np.asarray([float(row[arm+'_iou']) for row in rows])
    return np.c_[iou,iou>.25,iou>.5,
                 [int(row[arm+'_subject'])==RIGHT for row in rows],
                 [int(row[arm+'_subject_box'])==RIGHT for row in rows]]


def paired_rows(rows,subset,group,weighted):
    arm_metrics={a:[] for a in ARMS}
    if weighted:
        for key in ORDER:
            g=[r for r in rows if (r['source'],r['target'])==key]
            if g:
                for arm in ARMS:
                    arm_metrics[arm].append(metric_array(g,arm))
    else:
        for arm in ARMS:
            arm_metrics[arm].append(metric_array(rows,arm))
    stats={a:np.mean([v.mean(0) for v in mats],axis=0)*100 for a,mats in arm_metrics.items()}
    base=dict(group=group,subset=subset,n=len(rows),directions=len(arm_metrics[ARMS[0]]))
    overview=[dict(**base,arm=a,**dict(zip(METRICS,val)),
                   subject_wrong=sum(int(r[a+'_subject'])==WRONG for r in rows),
                   subject_unknown=sum(int(r[a+'_subject'])==UNKNOWN for r in rows)) for a,val in stats.items()]
    comparisons=[]
    for control in ARMS:
        if control=='h0_full':
            continue
        main='h0_full'
        d=np.concatenate([a-b for a,b in zip(arm_metrics[main],arm_metrics[control])])
        result=dict(**base,main=main,control=control)
        for k,m in enumerate(METRICS):
            result.update({m+'_pp':stats[main][k]-stats[control][k],m+'_up':int((d[:,k]>1e-10).sum()),
                           m+'_down':int((d[:,k]<-1e-10).sum())})
        comparisons.append(result)
    return overview,comparisons


def arm_counts(edges,arm):
    out={}
    for m in IDENTITY:
        c=Counter(int(r[f'{arm}_{m}']) for r in edges)
        out.update({m+'_n':sum(c.values())-c[NO_OUTPUT]-c[MISSING_TRUTH],m+'_correct':c[RIGHT],
                    m+'_wrong':c[WRONG],m+'_unknown':c[UNKNOWN],m+'_no_output':c[NO_OUTPUT]})
    return out


def analyze_edges(all_edges):
    tables,coverage=[] ,[]
    for group in ('all',*(f'{a}_to_{b}' for a,b in ORDER),*(f'target_{t}' for t in ('waymo','drone','quad'))):
        scope=[r for r in all_edges if group=='all' or group==f"{r['source']}_to_{r['target']}" or group==f"target_{r['target']}"]
        for truth in ('all_supported','annotation_proxy','visual_verified'):
            selection=[r for r in scope if r['truth']==truth or (truth=='all_supported' and r['truth'] in ('annotation_proxy','visual_verified'))]
            if not selection:
                continue
            base=dict(group=group,truth=truth,n=len(selection),total_parser_edges=len(scope))
            for arm in PAIR_ARMS:
                counts=arm_counts(selection,arm)
                values={m:100.*counts[m+'_correct']/counts[m+'_n'] if counts[m+'_n'] else '' for m in IDENTITY}
                tables.append(dict(**base,arm=arm,**values,**counts))
            for step in TRANSFER:
                reach={m:sum(int(r[step+'_'+m])==1 for r in selection) for m in IDENTITY}
                coverage.append(dict(**base,stage=step,**reach,
                   covered_but_failed_both=sum(int(r[step+'_both'])==1 and int(r['h0_full_both'])!=RIGHT for r in selection)))
    return tables,coverage


def assemble(by_direction):
    all_rows,edges=[],[]
    for s,t in ORDER:
        here=WORK/'scalars'/f'{s}_to_{t}'
        a,b=here/'rows.csv',here/'edges.csv'
        audit=by_direction[f'{s}_to_{t}']
        assert sha(a)==audit['rows_sha256'] and sha(b)==audit['edges_sha256']
        all_rows+=table(a)
        edges+=table(b)
    assert len(all_rows)==18620
    all_rows.sort(key=lambda r:(r['source'],r['target'],r['expression']))
    edges.sort(key=lambda r:(r['source'],r['target'],r['expression'],int(r['edge_number'])))
    metrics,pairs,scenes=[],[],[]
    for scope in VARIANTS:
        cohort=[r for r in all_rows if scope=='all' or int(r[scope])]
        if not cohort:
            continue
        for group in ('six_equal','all_pooled',*(f'{s}_to_{t}' for s,t in ORDER),*(f'target_{t}' for t in ('waymo','drone','quad'))):
            chosen=[r for r in cohort if group in ('six_equal','all_pooled',f"{r['source']}_to_{r['target']}",f"target_{r['target']}")]
            if not chosen:
                continue
            u,v=paired_rows(chosen,scope,group,group=='six_equal')
            metrics.extend(u)
            pairs.extend(v)
    by_recording=defaultdict(list)
    for row in all_rows:
        by_recording[row['source'],row['target'],row['recording']].append(row)
    for (s,t,rec),grp in sorted(by_recording.items()):
        for scope in ('all','camera_horizontal'):
            chosen=[r for r in grp if scope=='all' or int(r[scope])]
            if chosen:
                a,_=paired_rows(chosen,scope,rec,False)
                scenes.extend(dict(source=s,target=t,**r) for r in a)
    paired_identity,cover=analyze_edges(edges)
    for name,values in (('all_rows.csv',all_rows),('relation_edges.csv',edges),('metrics.csv',metrics),
                        ('paired.csv',pairs),('identity.csv',paired_identity),('coverage.csv',cover),('recordings.csv',scenes)):
        write_table(OUT/name,values)
    totals=Counter(r['truth'] for r in edges)
    subject={k:len({(r['target'],r['expression'],r['edge_number']) for r in edges if r['truth']==k}) for k in totals}
    result=dict(n=len(all_rows),per_direction=by_direction,
        segmentation={x:sum(int(r[x]) for r in all_rows) for x in VARIANTS[1:]},
        edge_truth_counts=dict(totals),edge_truth_physical_expression_counts=subject,
        geometry_endpoints=len(all_rows)*len(ARMS),
        exact_h0_candidate_frozen_center_and_size=True,
        all_seed_identity_metrics_use_common_voting=True,
        max_absolute_sensor_iou_error=max(q['geometry_error'] for q in by_direction.values()),
        max_r165_baseline_iou_difference=max(q['baseline_error'] for q in by_direction.values()),
        arm_budgets=dict(original_bank_subject_deficit=sum(int(r['subject_bank_slots'])>int(r['subject_original_slots']) for r in all_rows),
                         any_role_deficit=sum(int(r['any_bank_deficit']) for r in all_rows),
                         new_subject_selected=sum(int(r['full_new_subject']) for r in all_rows),
                         subject_fallback=sum(int(r['full_fallback_subject']) for r in all_rows)),
        files={f.name:sha(f) for f in sorted(OUT.glob('*.csv'))},
        prediction_seal=sha(HERE/'PREDICTION_SEAL.json'),FINAL_access=False)
    atomic_json(HERE/'EVALUATION_AUDIT.json',result)
    print('R166 task and identity scoring sealed',len(all_rows),dict(totals),flush=True)


if __name__=='__main__':
    verified_prediction_inputs()
    status('running',stage='post_seal_evaluation',protocol='round166/experiment/PROTOCOL.json')
    with ProcessPoolExecutor(max_workers=6,mp_context=get_context('spawn')) as pool:
        values=list(pool.map(score_direction,ORDER))
    assemble(dict(zip((f'{s}_to_{t}' for s,t in ORDER),values)))
