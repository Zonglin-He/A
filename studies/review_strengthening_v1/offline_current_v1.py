"""Offline current-prediction expert quality and physical-distance diagnostics.

Only complete, already-scored/audited selected predictions are read. Actual GT
is used for descriptive evaluation, never for prediction or online selection.
"""
from review_common import *
import argparse, csv

QGROUPS=['IoU_0_25','IoU_25_50','IoU_50_75','IoU_75_100','unscorable_no_GT_anchor']
EDGES=[0,.5,1,2,4,float('inf')]
ARMS=['Source','SharedCorrection','Interpolation','Ours','FinalIndependent']
PAIRS=[('Ours','Source'),('Ours','SharedCorrection'),('Ours','Interpolation'),
       ('Ours','FinalIndependent'),('Interpolation','SharedCorrection')]
PLAN=D/'offline_current_plan_v1.json'


def prepare():
    assert not PLAN.exists();selected.validate()
    inputs=[D/'intent.json']
    for ds in COUNTS:
        inputs += [RUN/ds/'sealed.json',selected.D/f'components_{ds}_sealed.json',
                   selected.D/f'components_{ds}_reporting_receipt.json',
                   selected.D/'components'/ds/'metrics.json',
                   R/'outputs/paper_review_suite_v1/offline'/ds/'private.json']
        job=read(RUN/'plan.json')['jobs'][ds]
        inputs += [PROVIDER/job['labels'],PROVIDER/job['manifest']]
    p=dict(schema='current_selected_offline_quality_distance_v1',
           scope='Both complete selected domains, 12205 queries. No new predictions/model/Source/expert/optimizer. Separate offline derive and independent full rederive audit before root analysis/publication.',
           code_sha256={str(f.relative_to(R)):sha(f) for f in [Path(__file__),S/'review_common.py',R/'stvg/metrics.py',R/'stvg/metrics_base.py',R/'tools/full_score.py']},
           input_sha256={str(f):sha(f) for f in inputs},
           quality='Exactly original mean actual-GT-supported expert anchor box IoU; four fixed quarter bins plus unscorable group. Also report annotated anchor coverage and anchor histogram. Check every quality value against old offline definition, not old method scores. Unannotated/out-of-event frames are never fake zero quality.',
           distance='Only actually GT-annotated native sampled UNOBSERVED frames, same support for every arm, nearest physical expert observation distance divided by fps; fixed [0,.5),[.5,1),[1,2),[2,4),[4,inf). No additional annotation or dense interpolated GT. Query-equal within-bin mean with parent-cluster CI; frame-weighted descriptive means and exact coverage separately.',
           methods='Current Ours/FinalIndependent/Source/E and original fixed center-logsize interpolation; coordinate-only fixed-support sIoU ignores each temporal interval. Quality groups use already-audited main-table t/v scalar values; interpolation main scores are not relabeled current-time. No GT-based online gate.',
           inference_GT=False,offline_GT=True,full_target_exposure=True,default_unchanged=True,
           limits=read(D/'intent.json')['limits'])
    p['fingerprint']=digest(p);atomic(PLAN,p);atomic(P/'offline_current_protocol_v1.json',p)


def validate_plan():
    p=read(PLAN);assert p['fingerprint']==digest({k:v for k,v in p.items() if k!='fingerprint'})
    for field in ['code_sha256','input_sha256']:
        for f,h in p[field].items():assert sha(R/f)==h,f
    selected.validate();return p


def one(ds,row,truth,metric):
    import numpy as np
    from stvg.metrics import box_iou
    from stvg.metrics_base import xyxy
    pe,ce=paths(ds,row['id']);m=row['metadata'];gt={int(k):v for k,v in truth['boxes'].items()}
    with np.load(RUN/ds/pe['path']) as parent,np.load(selected.D/ce['path']) as cur:
        ids=parent['frame_ids'];pos=parent['positions8'];assert np.array_equal(ids,cur['frame_ids']) and np.array_equal(pos,cur['positions8'])
        native=xyxy(parent['native_boxes'],m['width'],m['height'])
        if ds=='hc2':native=np.maximum(native,0)
        observed=ids[pos];scores=[float(box_iou(native[j],gt[int(fid)])) for j,fid in enumerate(observed) if int(fid) in gt]
        quality=float(np.mean(scores)) if scores else None
        group=QGROUPS[4] if quality is None else QGROUPS[min(int(quality*4),3)]
        boxes={'Source':cur['Source'],'SharedCorrection':cur['SharedCorrection'],
               'Interpolation':parent['Interpolation'],'Ours':cur['FullMAP'],
               'FinalIndependent':cur['FinalIndependent']}
        px={a:xyxy(v,m['width'],m['height']) for a,v in boxes.items()}
        if ds=='hc2':px={a:np.maximum(v,0) for a,v in px.items()}
        lookup={int(f):i for i,f in enumerate(ids)}
        supported=[(int(f),i) for i,f in enumerate(ids) if int(f) in gt and int(f) not in set(observed)]
        binned=[[] for _ in range(5)]
        for fid,i in supported:
            seconds=float(np.min(abs(observed-fid))/m['fps'])
            bi=min(int(np.searchsorted(EDGES,seconds,side='right')-1),4)
            binned[bi].append(dict(frame=fid,seconds=seconds,values={a:float(box_iou(v[i],gt[fid])) for a,v in px.items()}))
    main={'Source':metric['values']['Source'],'SharedCorrection':metric['values']['SharedCorrection'],
          'Ours':metric['values']['FullMAP'],'FinalIndependent':metric['values']['FinalIndependent']}
    return dict(id=row['id'],parent=str(truth['parent']),quality=quality,quality_group=group,
                scorable_anchors=len(scores),anchor_ious=scores,main=main,
                distance=binned,native_sampled_frames=len(ids),observations=8,
                supported_unobserved_frames=len(supported),
                parent_prediction_sha256=pe['sha256'],current_prediction_sha256=ce['sha256'],
                parent_receipt_sha256=pe['receipt_sha256'],current_receipt_sha256=ce['receipt_sha256'])


def derive(ds):
    import numpy as np
    p=validate_plan();who=identity();dest=D/'offline_current'/ds
    assert not(dest/'worker.json').exists();atomic(dest/'worker.json',who)
    audit=read(selected.D/f'components_{ds}_reporting_receipt.json')
    assert audit['status']=='verified_whole_domain' and audit['all_arrays_cold_rederived_exact']
    truth=selected.labels(ds);metrics={x['id']:x for x in read(selected.D/'components'/ds/'metrics.json')['private_rows']}
    old={x['id']:x for x in read(R/'outputs/paper_review_suite_v1/offline'/ds/'private.json')['anchor_quality']}
    rows=[];started=time.perf_counter()
    for i,row in enumerate(records(ds),1):
        assert time.perf_counter()-started<259200
        x=one(ds,row,truth[row['id']],metrics[row['id']]);o=old[row['id']]
        assert x['quality']==o['mean_anchor_iou'] and x['scorable_anchors']==o['scorable_anchors'] and x['quality_group']==o['group']
        rows.append(x)
        if i%100==0:atomic(dest/'progress.json',dict(completed=i,total=COUNTS[ds],offline_GT=True,no_new_prediction=True))
    assert len(rows)==COUNTS[ds]
    atomic(dest/'derived.json',dict(rows=rows))
    atomic(dest/'sealed.json',dict(status='offline_derived',queries=len(rows),worker=who,
                                  seconds=time.perf_counter()-started,derived_sha256=sha(dest/'derived.json'),
                                  plan_fingerprint=p['fingerprint'],new_prediction=0,new_model=0))


def audit(ds):
    p=validate_plan();dest=D/'offline_current'/ds;seal=read(dest/'sealed.json')
    assert seal['queries']==COUNTS[ds] and seal['derived_sha256']==sha(dest/'derived.json')
    rows=read(dest/'derived.json')['rows'];truth=selected.labels(ds)
    metrics={x['id']:x for x in read(selected.D/'components'/ds/'metrics.json')['private_rows']}
    for row,x in zip(records(ds),rows):
        assert one(ds,row,truth[row['id']],metrics[row['id']])==x
    receipt=dict(status='independent_full_offline_rederive_exact',queries=len(rows),
                 parent_current_prediction_receipt_SHA_checks=4*len(rows),
                 all_GT_supported_anchors_distances_IoUs_exact=True,
                 derived_sha256=seal['derived_sha256'],seal_sha256=sha(dest/'sealed.json'),
                 worker=identity(),new_prediction=0,new_model=0,new_Source=0,new_expert=0)
    atomic(dest/'audit_receipt.json',receipt)


def export(ds):
    import numpy as np
    from full_score import cluster
    validate_plan();dest=D/'offline_current'/ds;folder=P/'offline_current'/ds
    audit=read(dest/'audit_receipt.json');assert audit['status']=='independent_full_offline_rederive_exact'
    assert audit['derived_sha256']==sha(dest/'derived.json') and audit['seal_sha256']==sha(dest/'sealed.json')
    rows=read(dest/'derived.json')['rows'];assert len(rows)==COUNTS[ds]
    groups={}
    for group in QGROUPS:
        rr=[dict(parent=x['parent'],values=x['main']) for x in rows if x['quality_group']==group]
        groups[group]=dict(queries=len(rr),parents=len({x['parent'] for x in rr}),
                          annotated_anchors=sum(x['scorable_anchors'] for x in rows if x['quality_group']==group),
                          means={a:{k:float(np.mean([x['values'][a][k] for x in rr])) for k in ['v','t']} for a in rr[0]['values']} if rr else {},
                          paired={a+'-'+b:cluster(rr,a,b) for a,b in [('Ours','Source'),('Ours','SharedCorrection'),('Ours','FinalIndependent')]} if rr else {})
    curves=[]
    for bi in range(5):
        rr=[dict(parent=x['parent'],values={a:{'s':float(np.mean([f['values'][a] for f in x['distance'][bi]]))} for a in ARMS}) for x in rows if x['distance'][bi]]
        ff=[f for x in rows for f in x['distance'][bi]]
        curves.append(dict(seconds=[EDGES[bi],None if bi==4 else EDGES[bi+1]],
                           annotated_unobserved_frames=len(ff),queries=len(rr),parents=len({x['parent'] for x in rr}),
                           query_equal_means={a:float(np.mean([x['values'][a]['s'] for x in rr])) for a in ARMS} if rr else {},
                           frame_weighted_means={a:float(np.mean([f['values'][a] for f in ff])) for a in ARMS} if ff else {},
                           paired={a+'-'+b:cluster(rr,a,b,[('s',100)])['s'] for a,b in PAIRS} if rr else {}))
    flat=[q for x in rows for q in x['anchor_ious']]
    report=dict(dataset=ds,queries=len(rows),parents=len({x['parent'] for x in rows}),quality_groups=groups,
                coverage=dict(total_expert_observations=8*len(rows),GT_annotated_observations=len(flat),
                              unsupported_observations=8*len(rows)-len(flat),
                              queries_no_GT_anchor=sum(x['quality'] is None for x in rows),
                              annotated_anchor_IoU_histogram=np.histogram(flat,bins=[0,.25,.5,.75,1])[0].tolist(),
                              GT_supported_unobserved_frames=sum(x['supported_unobserved_frames'] for x in rows)),
                distance_curve=curves,quality_definition_exactly_matches_old_offline=True,
                scope=read(PLAN)['methods'],exposure='Historical full target sets already exposed; descriptive offline subgroup analysis, no causal quality claim or online gate.',
                audit_sha256=sha(dest/'audit_receipt.json'))
    atomic(folder/'report.json',report)
    with(folder/'all_query_quality_distance.tsv').open('w') as f:
        w=csv.writer(f,delimiter='\t',lineterminator='\n');w.writerow(['query','quality_bin','mean_annotated_anchor_IoU','annotated_anchors','distance_bin','supported_unobserved_frames',*[a+'_fixed_support_sIoU' for a in ARMS]])
        for i,x in enumerate(rows,1):
            for bi,frames in enumerate(x['distance']):
                w.writerow([f'Q{i:05d}',x['quality_group'],x['quality'],x['scorable_anchors'],bi,len(frames),*[float(np.mean([v['values'][a] for v in frames])) if frames else None for a in ARMS]])
    text=f'# Current selected method: offline expert quality and distance ({ds})\n\n{len(rows)} complete queries, {report["parents"]} parents. Independent full rederivation of all annotated anchor IoUs, physical distances, fixed-support frame IoUs and input SHA checks passed. No model, Source, expert, optimizer or new prediction.\n\nQuality is exactly the historical mean annotated anchor-IoU definition, now joined to current already-audited main t/v outcomes. Unsupported anchors are missing, never zero. Quality is evaluated offline only; associations do not prove a causal expert-quality effect.\n\n|Quality bin|Queries|Parents|Annotated anchors|Ours minus Source v pp [95% CI]|\n|---|---:|---:|---:|---|\n'
    for group,g in groups.items():
        v=g['paired'].get('Ours-Source',{}).get('v')
        text+=f'|{group}|{g["queries"]}|{g["parents"]}|{g["annotated_anchors"]}|{v}|\n'
    text+='\nDistance analysis uses identical actually annotated native sampled unobserved-frame support for every arm, independent of temporal interval. Physical nearest-observation distance/fps; no interpolated GT or extra frames. Query-equal means and parent-cluster10000 bootstrap95% CIs are primary; frame-weighted means are separate descriptive summaries. Pointwise intervals, not familywise. Interpolation keeps its original path and receives no DP.\n\n|Seconds|Frames|Queries|Ours minus E fixed-support sIoU pp [95% CI]|\n|---|---:|---:|---|\n'
    for g in curves:text+=f'|{g["seconds"]}|{g["annotated_unobserved_frames"]}|{g["queries"]}|{g["paired"].get("Ours-SharedCorrection")}|\n'
    text+='\nAll signed query/bin means and all positive/negative subgroup comparisons are retained. This descriptive sensitivity/quality analysis does not erase prior target-domain development selection or establish independent unseen validation. Inherited Source/expert acquisition cost is nonzero; this CPU/offline work is not deployment timing.\n'
    (folder/'ANALYSIS.md').write_text(text)
    rec=dict(status='root_offline_quality_distance_complete',queries=len(rows),
             code_sha256=sha(Path(__file__)),audit_sha256=sha(dest/'audit_receipt.json'),
             report_sha256=sha(folder/'report.json'),analysis_sha256=sha(folder/'ANALYSIS.md'),
             offline_GT_only=True,all_signed_coverage_parent_CI=True,new_model=0,new_prediction=0)
    atomic(dest/'root_export_receipt.json',rec);atomic(folder/'root_export_receipt.json',rec)
    print(json.dumps(dict(dataset=ds,status=rec['status'],coverage=report['coverage'])))


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('command',choices=['prepare','derive','audit','export']);a.add_argument('--dataset',choices=COUNTS)
    args=a.parse_args();globals()[args.command](*([] if args.command=='prepare' else [args.dataset]))
