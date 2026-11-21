"""Task and endpoint scoring from sealed predictions with full frame denominators."""
from collections import Counter
import csv
import sys
import numpy as np
from common import BASE,HERE,OUT,PRIVATE,SPEC,digest,inputs,read,write
from predict_span import ARMS
from span import TOL,axes
from score import canonical_box,many_iou,score_boxes,truth_for_slot,rotation
from score_confirmation import cluster_interval,paired_stats
from score_region import save

def groups(rows):
    for name,g in [('all',rows),('validation',[r for r in rows if r['panel']=='validation']),
                   ('recording_separated',[r for r in rows if r['separation']=='new_recording']),
                   ('recording_extension',[r for r in rows if r['panel']!='validation'])]:
        if g:
            yield (name,'macro'),g
        for t in sorted({r['target'] for r in g}):
            yield (name,t),[r for r in g if r['target']==t]

def macro(group, vals):
    platforms=np.array([r['target'] for r in group]);vals=np.asarray(vals)
    return float(np.mean([vals[platforms==t].mean() for t in np.unique(platforms)]))

def metric(group,arm):
    x=np.array([r[arm] for r in group])
    centers=[r[arm+'_center_m'] for r in group if arm+'_center_m' in r]
    return dict(acc25=100*macro(group,x>.25),acc50=100*macro(group,x>.5),miou=100*macro(group,x),
                center_n=len(centers),center_median_m=float(np.median(centers)) if centers else None)

def scores(box,normal,offsets,gt):
    rows=np.repeat(box[None],len(offsets),axis=0)
    rows[:,:3]+=np.array(list(offsets.values()))[:,None]*normal
    ious,gap=score_boxes(rows,gt)
    vals={}
    for k,coord,iou in zip(offsets,rows,ious,strict=True):
        vals[k]=float(iou)
        vals[k+'_center_m']=float(np.linalg.norm(coord[:3]-gt[:3]))
        vals[k+'_lateral_err_m']=abs(float((coord[:3]-gt[:3])@normal))
    return vals,gap

def contained(x,interval):
    return interval is not None and interval[0]-TOL<=x<=interval[1]+TOL

def inside_gt(x,gt):
    return bool((np.abs((np.array(x)-gt[:3])@rotation(gt[6]))<=gt[3:6]/2+.05).all())

def run():
    panel,frozen=inputs()
    seal=read(OUT/'prediction_seal.json')
    assert seal['private_sha']==digest(PRIVATE/'predictions.json') and seal['protocol']==digest(SPEC)
    assert all(digest(HERE/k)==v for k,v in seal['implementation'].items())
    assert not seal['target_labels_read'] and not seal['FINAL_access']
    assert read(OUT/'oracle_seal.json')['private_sha']==digest(PRIVATE/'oracle.json')
    estimates=read(PRIVATE/'predictions.json');oracle=read(PRIVATE/'oracle.json')
    rows=[];max_gap=parity=0.
    for f in panel:
        uid=f['uid'];e=estimates[uid];o=oracle[uid];old=frozen[uid]
        dx=e['offsets']
        row={k:f[k] for k in ('uid','target','panel','recording','separation')}
        row.update(selected=e['selected'],status=e['status'],read_status=e['read_status'],
                   part=e.get('part',''),oracle=o['oracle'],recoverable50=o['recoverable50'],
                   extended=e.get('new_span_m',0.)>TOL,new_span_m=e.get('new_span_m',0.),
                   matched_anchor='physical_anchor' in e,feasible=e['status']=='applied_or_already_inside',
                   common_wrong='wrong_read' in e,wrong_read_status=e.get('wrong_read',{}).get('status',''),
                   pair_moved_vs_anchor=abs(dx['pair']-dx['anchor'])>TOL,
                   real_span_m=e.get('real_span_m'),width_m=e.get('width_m'),anchor_px=e.get('anchor_px'))
        row.update({arm+'_shift_m':v for arm,v in dx.items()})
        for k in ('reprojection_px','sin_angle','nominal_s'):
            row[k]=e.get('pair_read',{}).get(k)
        row.update(dict.fromkeys(ARMS,0.))
        row['span_oracle']=0.
        if old['boxes']['H'] is not None:
            box=np.asarray(old['boxes']['H']);n,width=axes(box)
            gt=canonical_box(truth_for_slot(f,read(f['calibration'])['ground_info']))
            vals,gap=scores(box,n,dx,gt);row.update(vals); max_gap=max(max_gap,gap)
            truth_station=float((gt[:3]-box[:3])@n)
            row['span_oracle']=row['frozen']
            if e.get('real_interval') is not None:
                row['true_center_in_real_interval']=contained(truth_station,e['real_interval'])
                canonical_n,_=axes(canonical_box(box))
                ds=o['delta']*np.sign(n@canonical_n)
                # A unimodal translational overlap can be maximized on a
                # constrained subinterval by clamping an unconstrained maximizer.
                ds=float(np.clip(ds,*e['real_interval']))
                changed=box.copy();changed[:3]+=n*ds
                row['span_oracle']=float(max(row['frozen'],many_iou(changed[None],gt)[0]))
            if 'interval' in e:
                row['true_center_in_pair_interval']=contained(truth_station,e['interval'])
            if 'physical_anchor' in e:
                row['anchor_inside_gt']=inside_gt(e['physical_anchor'],gt)
            wit=e.get('witness')
            if wit:
                row.update(witness=True,witness_unsampled=wit['not_in_frozen_tree_draws'],
                           witness_node=wit['in_selected_region'],witness_inside_gt=inside_gt(wit['xyz'],gt))
                ray=e['pair_read']
                row['copy_witness_abs_m']=abs(wit['z']-ray['copy_z'])
                if 'nominal_z' in ray:
                    row['pair_witness_abs_m']=abs(wit['z']-ray['nominal_z'])
            if 'independent' in e:
                row['blind24_status']=e['independent']['status']
                vm,_=scores(box,n,{'blind24':e['independent']['offsets']['pair']},gt);row.update(vm)
        parity=max(parity,abs(row['frozen']-o['baseline']))
        assert row['span_oracle']<=o['oracle']+1.e-8
        rows.append(row)
    assert max_gap<1.e-8 and parity<1.e-8 and len(rows)==416
    metrics=[];paired=[];funnel=[];intervals={};record=[]
    comparisons=[('real','frozen'),('anchor','real'),('pair','frozen'),('pair','anchor'),
                 ('copy','anchor'),('pair','copy'),('pair','wrong'),('point','anchor')]
    for (scope,t),g in groups(rows):
        tag=dict(panel=scope,platform=t,n=len(g),recordings=len({r['recording'] for r in g}))
        for arm in (*ARMS,'oracle','span_oracle'):
            metrics.append(dict(**tag,arm=arm,**metric(g,arm)))
        for a,b in comparisons:
            paired.append(dict(**tag,arm=a,reference=b,**paired_stats(g,a,b)))
            if t=='macro' and a=='pair':
                intervals[f'{scope}/{a}-{b}']=cluster_interval(g,a,b)
        paired_wrong=[r for r in g if r['common_wrong']]
        if paired_wrong:
            paired.append(dict(**(tag|dict(panel=scope+'_matched_wrong',n=len(paired_wrong),recordings=len({r['recording'] for r in paired_wrong}))),
                               arm='pair',reference='wrong',**paired_stats(paired_wrong,'pair','wrong')))
        for subgroup,gg in [('all',g),('oracle_failures', [r for r in g if r['recoverable50']])]:
            funnel.append(dict(**tag,scope=subgroup,denominator=len(gg),
                    selected=sum(r['selected'] for r in gg),rgb_pair=sum(r['read_status']=='pair' for r in gg),
                    anchor=sum(r['matched_anchor'] for r in gg),pair_compatible=sum(r['feasible'] for r in gg),
                    new_span=sum(r['extended'] for r in gg),center_changed_vs_anchor=sum(r['pair_moved_vs_anchor'] for r in gg),
                    rescue50=sum(r['pair']>.5 and r['frozen']<=.5 for r in gg),
                    harm50=sum(r['pair']<=.5 and r['frozen']>.5 for r in gg),
                    feasible_oracle_rescue=sum(r['span_oracle']>.5 and r['frozen']<=.5 for r in gg)))
    for k in sorted({(r['target'],r['recording']) for r in rows}):
        g=[r for r in rows if (r['target'],r['recording'])==k]
        record.append(dict(target=k[0],recording=k[1],n=len(g),**paired_stats(g,'pair','frozen')))
    audit=[]
    for name,rule in [('held_cell',lambda r:True),('unsampled_held_cell',lambda r:r.get('witness_unsampled')),
                      ('unsampled_both_in_padded_gt',lambda r:r.get('witness_unsampled') and r.get('anchor_inside_gt') and r.get('witness_inside_gt'))]:
        for feasible in (False,True):
            g=[r for r in rows if r.get('witness') and rule(r) and (not feasible or r['feasible'])]
            common=[r for r in g if r.get('pair_witness_abs_m') is not None]
            audit.append(dict(scope=name,accepted_only=feasible,n=len(g),both_predicted=len(common),
                    pair_mae_m=float(np.mean([r['pair_witness_abs_m'] for r in common])) if common else None,
                    copy_mae_m=float(np.mean([r['copy_witness_abs_m'] for r in common])) if common else None,
                    pair_median_m=float(np.median([r['pair_witness_abs_m'] for r in common])) if common else None,
                    copy_median_m=float(np.median([r['copy_witness_abs_m'] for r in common])) if common else None))
    files=dict(rows=rows,metrics=metrics,paired=paired,funnel=funnel,recordings=record,witnesses=audit)
    for k,v in files.items():save(OUT/f'{k}.csv',v)
    write(OUT/'evaluation_receipt.json',dict(primary_seal=digest(OUT/'prediction_seal.json'),
        oracle_seal=digest(OUT/'oracle_seal.json'),frames=len(rows),maximum_scorer_gap=max_gap,
        previous_metric_gap=parity,code=digest(__file__),files={k:digest(OUT/f'{k}.csv') for k in files},
        intervals=intervals,ground_truth_used_for_scoring_only=True,FINAL_access=False))
    print('Scored',len(rows),'frames; max parity gap',parity,flush=True)

if __name__=='__main__':
    run()
