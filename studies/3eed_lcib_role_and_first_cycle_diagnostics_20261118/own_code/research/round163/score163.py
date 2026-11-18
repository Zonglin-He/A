"""Independently score the sealed source replay, including signed per-axis changes."""
from collections import defaultdict
import sys

import numpy as np

from scope163 import HERE, PLATFORMS, PROTOCOL, SETTINGS, WORK, atomic_json, read, sha, source_scope
from cycle163 import AXES, MASKS, STAGES
from audit160 import raw_annotations, sensor_frame, transform_gt
from audit162 import sensor_iou
from geometry162 import in_box
from manifest162 import canonical
from mechanisms162 import annotation_inventory, seed_owners, support_owner
from score154 import many_iou
from score162 import put_table

CONTRASTS = (
    ('r158_to_initial','r158','initial'),
    ('support_and_read','initial','support_only'),
    ('fit_fixed_support_winner','support_only','fit_keep_support_winner'),
    ('fit_fixed_initial_winner','initial','fit_keep_initial_winner'),
    ('reselect','fit_keep_support_winner','reselect'),
    ('first_cycle','initial','reselect'),
    ('fit_and_read_after_support','support_only','reselect'),
    ('three_cycles_from_initial','initial','r162_final'),
    ('r158_to_final','r158','r162_final'),
    *((x,'initial',x) for x in MASKS),
    ('drop_only_source_prior','all7','all7_instance_prior_only'),
)


def paired_metrics(old,new):
    fields={}
    for label,d in [('mIoU',new-old),
                    ('Acc25',(new>.25).astype(int)-(old>.25)),
                    ('Acc50',(new>.50).astype(int)-(old>.50))]:
        fields.update({label+'_pp':float(100*np.mean(d)),
                      label+'_gain':int((d>1e-12).sum()),label+'_loss':int((d< -1e-12).sum())})
    return fields


def summarize(rows, parameters):
    arms, contrasts, shifts=[],[],[]
    for source in (*PLATFORMS, 'all'):
        active=[r for r in rows if r['source']==source or source=='all']
        pts=[r for r in parameters if r['source']==source or source=='all']
        values={arm:np.array([r[arm] for r in active]) for arm in (*STAGES,*MASKS)}
        for arm,x in values.items():
            arms.append(dict(source=source,arm=arm,n=len(x),mIoU=100*x.mean(),Acc25=100*(x>.25).mean(),
                             Acc50=100*(x>.50).mean(),count25=int((x>.25).sum()),count50=int((x>.50).sum())))
        for name,a,b in CONTRASTS:
            contrasts.append(dict(source=source,contrast=name,before=a,after=b,n=len(active),
                                  **paired_metrics(values[a],values[b])))
        for arm,dimensions in MASKS.items():
            for axis in (AXES[k] for k in dimensions):
                sel=[r for r in pts if r['arm']==arm and r['axis']==axis and r['fit_attempted']]
                x=np.asarray([r['signed_change'] for r in sel])
                shifts.append(dict(source=source,arm=arm,axis=axis,n_total=len(active),n_optimized=len(sel),
                    mean=float(np.mean(x)) if len(x) else '',median=float(np.median(x)) if len(x) else '',
                    p10=float(np.quantile(x,.1)) if len(x) else '',p90=float(np.quantile(x,.9)) if len(x) else '',
                    increases=int((x>1e-7).sum()),decreases=int((x< -1e-7).sum()),
                    low_hits=sum(r['bound_hit']==-1 for r in sel),high_hits=sum(r['bound_hit']==1 for r in sel),
                    hit_percent=100*sum(r['bound_hit']!=0 for r in sel)/len(sel) if sel else ''))
    return arms,contrasts,shifts


def main():
    source=source_scope()
    seal=read(HERE/'SOURCE_SEAL.json')
    index=WORK/'SOURCE_FILES_PRIVATE.json'
    assert seal['n']==len(source) and seal['index_sha256']==sha(index) and seal['protocol_sha256']==sha(PROTOCOL)
    assert seal['source_config_sha256']==sha(SETTINGS)
    files=read(index)
    for name,digest in files.items():
        assert sha(WORK/'source_predictions'/name)==digest
    rows,per_axis=[],[]
    worst=0.
    output=HERE/'results'
    for i,r in enumerate(source):
        view=int(r['suffix'] or 0)
        typ=r['source']
        gt=np.asarray(r['val_GT'],dtype=float)
        pos=int(r['object_index'])
        caption,annotated=raw_annotations(r['metapath'])[pos]
        assert canonical(caption)==r['caption']
        audited_gt=transform_gt(annotated,typ,view).astype(float)
        np.testing.assert_allclose(audited_gt,gt,atol=1e-7,rtol=0.)
        coords=sensor_frame(gt,typ,view)
        _,labels=annotation_inventory(r['metapath'],typ,view,pos)
        with np.load(WORK/'source_predictions'/f'{i:05d}.npz',allow_pickle=False) as f:
            boxes=f['boxes']
            assert f['arms'].tolist()==[*STAGES,*MASKS]
            ious=many_iou(boxes,gt)
            independent=np.array([sensor_iou(b,coords,typ,view) for b in boxes])
            assert np.array_equal(independent[:,None]>.25*np.array([1,2]),ious[:,None]>.25*np.array([1,2]))
            worst=max(worst,float(np.max(np.abs(independent-ious))))
            before,post_support,postfit=map(int,(f['h0'],f['hS'],f['hF']))
            energy=f['energy']
            seed_label=seed_owners(f['seeds'],labels)
            support=f['supports']
            known=[support_owner(w,seed_label,len(labels)) for w in support]
            row=dict(ordinal=i+1,source=typ,**{a:float(x) for a,x in zip((*STAGES,*MASKS),ious)},
                 switched_after_support=before!=post_support,switched_after_fit=post_support!=postfit,
                 h0_matches_frozen_R158=bool(f['initial_matches_old_query']),
                 initial_unknown=before<0,fitted_fixed_initial=bool(f['fit_evals'][0]),
                 old_subject_support_in_target_initial=known[0]==0,old_subject_support_in_target_updated=known[1]==0,
                 support_initial_to_updated_both_known=min(known[:2])>=0,
                 support_initial_to_updated_changed_object=min(known[:2])>=0 and known[0]!=known[1],
                 support_initial_to_updated_l1=float(abs(support[1]-support[0]).sum()),
                 new_selected_support_assigned=known[2]>=0,new_selected_support_in_target=known[2]==0,
                 energy_initial=float(energy[0]),energy_support=float(energy[1]),energy_fit=float(energy[2]),
                 effective_fixed_support=float(f['fit_effective_points']))
            rows.append(row)
            # Each saved displacement is in the canonical model axes for the
            # translated center, and axial yaw for rotation.
            for m,(arm,active) in enumerate(MASKS.items()):
                moves=f['drift'][m].copy()
                moves[6]=np.degrees(moves[6])
                for k,axis in enumerate(AXES):
                    per_axis.append(dict(ordinal=i+1,source=typ,arm=arm,axis=axis,variable_enabled=k in active,
                        fit_attempted=int(f['fit_evals'][m])>0,signed_change=float(moves[k]),
                        unit='m' if k<3 else 'log ratio' if k<6 else 'degree',bound_hit=int(f['hits'][m,k])))
                if len(active)<7:
                    disabled=sorted(set(range(7))-set(active))
                    assert np.array_equal(boxes[len(STAGES)+m,disabled],boxes[1,disabled])
    assert len(rows)==192 and worst<1e-7
    arms,contrasts,shifts=summarize(rows,per_axis)
    put_table(output/'source_rows.csv',rows)
    put_table(output/'source_metrics.csv',arms)
    put_table(output/'paired_contrasts.csv',contrasts)
    put_table(output/'signed_per_example.csv',per_axis)
    put_table(output/'signed_axis_summary.csv',shifts)
    stage_summary={}
    for s in (*PLATFORMS,'all'):
        g={r['contrast']:r for r in contrasts if r['source']==s}
        for metric in ('mIoU','Acc25','Acc50'):
            name=metric+'_pp'
            parts=sum(g[x][name] for x in ('support_and_read','fit_fixed_support_winner','reselect'))
            assert abs(parts-g['first_cycle'][name])<1e-10
        scoped=[(r,m) for r,m in zip(rows,source) if s in (r['source'],'all')]
        selection=[r for r,_ in scoped]
        keys=('switched_after_support','switched_after_fit','initial_unknown','fitted_fixed_initial',
              'old_subject_support_in_target_initial','old_subject_support_in_target_updated',
              'support_initial_to_updated_both_known','support_initial_to_updated_changed_object')
        stage_summary[s]=dict(records=len(scoped),
                          point_files=len({(m['source'],m['pointpath']) for _,m in scoped}),
                          recording_groups=len({(m['source'],m['recording']) for _,m in scoped}),
                          support_weights_changed=sum(r['support_initial_to_updated_l1']>1e-9 for r in selection),
                          energy_decreased_during_support=sum(r['energy_support']<r['energy_initial']-1e-12 for r in selection),
                          energy_decreased_during_fit=sum(r['energy_fit']<r['energy_support']-1e-12 for r in selection),
                          target_support_became_unassigned=sum(r['old_subject_support_in_target_initial']
                              and not r['old_subject_support_in_target_updated']
                              and not r['support_initial_to_updated_both_known'] for r in selection),
                          **{k:sum(r[k] for r in selection) for k in keys})
    atomic_json(output/'SOURCE_STAGE_SUMMARY.json',stage_summary)
    atomic_json(HERE/'SOURCE_AUDIT.json',dict(rows=192,sensor_frame_endpoint_checks=192*(len(STAGES)+len(MASKS)),
                max_abs_iou_error=worst,original_model_and_three_cycle_outputs_reproduced=True,
                original_full_axes_fit_reproduced=True,telescoping_contrasts_verified=True,
                source_seal_sha256=sha(HERE/'SOURCE_SEAL.json'),result_rows_sha256=sha(output/'source_rows.csv'),FINAL_access=False))
    print('Source audit complete',len(rows),worst,flush=True)


if __name__=='__main__':
    main()
