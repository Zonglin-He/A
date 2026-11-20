"""Subject scores, and an oracle-identity test of the physical displacement itself."""
from collections import Counter
import csv
import gzip

import numpy as np

from capture import ARMS, HERE, OLD, OUT, PRIVATE, PROXY, SOURCES, digest, read, save
from score_roles import anon, csv_file, inventory, owners, score_boxes


def annotated_tuple(z,labels,reference):
    banks=[z[name] for name in ('s','r','rs','rr')]
    correct=[np.array(owners(bank,labels))==truth for bank,truth in zip(banks,(0,reference,0,reference),strict=True)]
    eligible=correct[0][:,None,None,None]&correct[1][None,:,None,None]&correct[2][None,None,:,None]&correct[3][None,None,None,:]
    support=[int(c.any()) for c in correct]
    if not eligible.any():
        return support,None
    prior=z['p'][:,None,None,None]*z['conditional'][:,:,None,None]*z['reverse'][None,None,:,:]
    indices=np.unravel_index(np.argmax(np.where(eligible,prior,-1)),prior.shape)
    return support,[bank[i] for bank,i in zip(banks,indices,strict=True)]


def translation_diagnostic(tuple_boxes,labels):
    sf,rf,sd,rd=tuple_boxes
    h=rf[:3]-rd[:3]
    raw=sd.copy()
    shifted=sd.copy();shifted[:3]+=h
    opposite=sd.copy();opposite[:3]-=h
    shapes=np.stack((raw,shifted,opposite))
    ious,err=score_boxes(shapes,labels[0])
    truth=labels[0,:3]
    errors=((shapes[:,:3]-truth)**2).sum(1)
    same_read=((shapes[:,:3]-sf[:3])**2).sum(1)
    observation_gain=2*(sf[:3]-sd[:3])@h - h@h
    task_gain=2*(truth-sd[:3])@h - h@h
    assert abs(errors[0]-errors[1]-task_gain)<1e-9 and abs(same_read[0]-same_read[1]-observation_gain)<1e-9
    return dict(ref_displacement_m=float(np.linalg.norm(h)),raw_center_m2=errors[0],shift_center_m2=errors[1],
                opposite_center_m2=errors[2],raw_chart_m2=same_read[0],shift_chart_m2=same_read[1],
                true_center_gain_m2=task_gain,consistency_gain_m2=observation_gain,
                raw_iou=ious[0],shift_iou=ious[1],opposite_iou=ious[2],shift_still_same=int(owners(shapes[1:2],labels)[0]==0)),err


def write_compressed(filename,rows):
    with gzip.open(OUT/filename,'wt',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys())
        writer.writeheader();writer.writerows(rows)


def summarize(predictions,pairs):
    task=[]
    physical=[]
    for group in (*SOURCES,'macro'):
        tt=[r for r in predictions if group=='macro' or r['source']==group]
        for arm in ARMS:
            rows=[r for r in tt if r['arm']==arm]
            measure={}
            for name,fn in [('acc25',lambda x:x>.25),('acc50',lambda x:x>.5),('miou',lambda x:x)]:
                means=[np.mean([fn(r['iou']) for r in rows if r['source']==s]) for s in sorted({r['source'] for r in rows})]
                measure[name]=100*float(np.mean(means))
            task.append(dict(source=group,arm=arm,n=len(rows),**measure,
                rescue25=sum(r['baseline_iou']<=.25<r['iou'] for r in rows),harm25=sum(r['iou']<=.25<r['baseline_iou'] for r in rows),
                rescue50=sum(r['baseline_iou']<=.5<r['iou'] for r in rows),harm50=sum(r['iou']<=.5<r['baseline_iou'] for r in rows),
                slot_switch=sum(r['switched'] for r in rows),wrong_to_target=sum(r['old_owner']=='W' and r['owner']=='T' for r in rows),
                target_to_wrong=sum(r['old_owner']=='T' and r['owner']=='W' for r in rows),
                correct_subject_outside_pool=sum(r['old_owner']==r['owner']=='T' and r['all_best']<=.5<r['iou'] for r in rows)))
        rows=[r for r in pairs if group=='macro' or r['source']==group]
        seen=[r for r in rows if r['eligible']]
        def mean(key):
            return float(np.mean([r[key] for r in seen])) if seen else None
        fields=dict(n=len(rows),eligible=len(seen),recordings=len({r['recording'] for r in seen}),
            **{k:sum(r[k] for r in rows) for k in ('subject_f','reference_f','subject_d','reference_d')},
            nonzero_shifts=sum(r['ref_displacement_m']>1e-6 for r in seen),
            center_improved=sum(r['true_center_gain_m2']>1e-8 for r in seen),center_worse=sum(r['true_center_gain_m2']< -1e-8 for r in seen),
            mean_raw_center_m2=mean('raw_center_m2'),mean_shift_center_m2=mean('shift_center_m2'),
            mean_opposite_center_m2=mean('opposite_center_m2'),mean_raw_between_reads_m2=mean('raw_chart_m2'),
            mean_shift_between_reads_m2=mean('shift_chart_m2'),mean_raw_iou=mean('raw_iou'),mean_shift_iou=mean('shift_iou'),
            mean_opposite_iou=mean('opposite_iou'),rescue50=sum(r['raw_iou']<=.5<r['shift_iou'] for r in seen),
            harm50=sum(r['shift_iou']<=.5<r['raw_iou'] for r in seen),opposite_rescue50=sum(r['raw_iou']<=.5<r['opposite_iou'] for r in seen),
            opposite_harm50=sum(r['opposite_iou']<=.5<r['raw_iou'] for r in seen),lost_target=sum(not r['shift_still_same'] for r in seen))
        physical.append(dict(source=group if group!='macro' else 'pooled',**fields))
    csv_file(OUT/'task.csv',task)
    csv_file(OUT/'physical.csv',physical)


def run():
    seal=read(PRIVATE/'SEAL.json')
    assert seal['n']==111 and seal['only_source_own'] and not seal['labels_used_for_outputs'] and not seal['FINAL_access']
    assert seal['lock']==digest(PRIVATE/'LOCK.json')
    cfg=read(PRIVATE/'LOCK.json')
    assert cfg['design']==digest(HERE/'PROTOCOL.json') and cfg['roster']==digest(PROXY)
    assert all(digest(HERE/f)==sig for f,sig in cfg['code'].items())
    with gzip.open(OLD/'results/all_rows.csv.gz','rt') as f:
        previous={(r['case'],r['source']):r for r in csv.DictReader(f) if r['group']=='own_proxy' and r['arm']=='v1'}
    predictions,pairs=[],[]
    err=0.
    for case,entry in enumerate(read(PROXY)):
        row=entry['row']
        source=row['platform']
        case_id=anon(f'source_proxy|{case}')
        old=previous[case_id,source]
        path=PRIVATE/f'source_proxy_{case:03d}_{source}.npz'
        assert digest(path)==seal['files'][path.name]
        labels,_=inventory(row['metapath'],source,int(row['suffix'] or 0),int(row['object_index']))
        with np.load(path,allow_pickle=False) as z:
            assert str(z['lock'])==seal['lock']
            coverage,matched=annotated_tuple(z,labels,int(entry['ref_id']))
            ious,error=score_boxes(z['boxes'],labels[0])
            err=max(err,error)
            assert abs(ious[0]-float(old['iou']))<1e-8
            physical_id=owners(z['s'],labels)
            states=['U' if i<0 else 'T' if i==0 else 'W' for i in physical_id]
            start=int(z['choices'][0])
            meta=dict(source=source,case=case_id,recording=old['recording'])
            for idx,arm in enumerate(ARMS):
                pick=int(z['choices'][idx])
                predictions.append(dict(**meta,arm=arm,baseline_iou=float(ious[0]),iou=float(ious[idx]),all_best=float(old['best_all_axis_and_pca']),
                    old_owner=states[start],owner=states[pick],switched=int(start!=pick)))
            ds=dict(**meta,**dict(zip(('subject_f','reference_f','subject_d','reference_d'),coverage,strict=True)),eligible=int(matched is not None))
            if matched is not None:
                values,error=translation_diagnostic(matched,labels)
                ds.update(values);err=max(err,error)
            pairs.append(ds)
    assert len(pairs)==111 and len(predictions)==111*len(ARMS)
    write_compressed('source_all_rows.csv.gz',predictions)
    fields=list(dict.fromkeys(k for r in pairs for k in r))
    write_compressed('identity_conditioned_rows.csv.gz',[{k:r.get(k) for k in fields} for r in pairs])
    summarize(predictions,pairs)
    save(OUT/'audit.json',dict(n=seal['n'],sealed=digest(PRIVATE/'SEAL.json'),seconds=seal['seconds'],max_independent_iou_error=err,
        eligible=sum(r['eligible'] for r in pairs),source_counts=dict(Counter(r['source'] for r in pairs)),target_access=False,FINAL_access=False))


if __name__=='__main__':
    run()
