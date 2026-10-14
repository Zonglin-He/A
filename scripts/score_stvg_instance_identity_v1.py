"""Post-seal matched-grid evaluation. No annotation is opened before custody checks."""
import sys,math,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_instance_identity_common_v1 import BASE,OLD,read,write,sha,verify

def main():
    import torch,numpy as np
    from scripts.stvg_sparse_r1_prompt_v1 import load_source_case
    from scripts import score_stvg_native_instance_nativeclock005 as metric
    from scripts import review_stvg_sparse_r1_v2 as scorer
    verify();seal=read(BASE/'GLOBAL_PREDICTION_BARRIER.json');assert seal['status']=='complete' and seal['GT_read'] is False
    for f in seal['files']:assert sha(ROOT/f['path'])==f['sha256']
    assert seal['scorer_sha256']==sha(__file__)
    roster=read(OLD/'ROSTER_LOCK.json')['records'];refs={r['case_id']:r for r in read(BASE/'REFERENCE_LOCK.json')['records']};geo={r['case_id']:r for r in read(BASE/'TEACHER_GEOMETRY_ONLY_READOUT.json')['records']}
    cases={};outputs={}
    for r in roster:
        cid=r['case_id'];p=BASE/'student'/(cid+'.pt');rc=read(p.with_suffix('.receipt.json'))
        assert sha(p)==rc['sha256'] and p.stat().st_size==rc['bytes']
        assert sha(ROOT/r['source_case_path'])==r['source_case_sha256']
        c=load_source_case(ROOT/r['source_case_path']);d=torch.load(p,map_location='cpu',weights_only=False)
        assert not d['GT_read'] and d['source_restored'] and d['source_case_sha256']==r['source_case_sha256']
        assert d['reference_sha256']==sha(BASE/'REFERENCE_LOCK.json')
        assert list(c['frame_ids'])==d['frame_ids'];assert len(d['arms'])==12
        for a in d['arms'].values():assert np.asarray(a['boxes']).shape==(len(d['frame_ids']),4) and np.isfinite(a['boxes']).all()
        for name,g in [('T_geometry_strict',geo[cid]['strict_original']),('T_geometry_raw',geo[cid]['raw_decoupled'])]:
            assert g['frame_ids_physical']==d['frame_ids'] and g['interval_frame_bounds_half_open']==d['arms']['T']['interval']
            d['arms'][name]=dict(boxes=g['boxes_xyxy_normalized'],interval=g['interval_frame_bounds_half_open'])
            for i,fallback in enumerate(g['fallback_used']):
                if fallback:assert np.array_equal(np.asarray(g['boxes_xyxy_normalized'][i]),np.asarray(d['arms']['F']['boxes'][i]))
        cases[cid]=c;outputs[cid]=d
    assert len(cases)==8
    # First annotation read occurs only after every sealed method prediction has passed custody validation.
    targets,provenance=scorer._truths_after_seal(cases);official,independent,contract=scorer._metric_functions()
    rows=[]
    for cid,d in outputs.items():
        # Same raw-decoupled support across comparisons; actual arm support also disclosed separately.
        support={int(a['position']) for a in refs[cid]['raw_decoupled']['anchors']}
        for arm,value in d['arms'].items():
            m,audit=metric._score_student_arm(value,cases[cid],targets[cid],support,official,independent)
            rows.append(dict(case_id=cid,dataset=d['dataset'],arm=arm,metrics=m,support=audit['support'],metric_audit=audit['metric_audit'],status='scored'))
    names=list(next(iter(outputs.values()))['arms']);summaries=[];contrasts=[]
    pairs=[(n,'T') for n in names if n.startswith('T_')]+[('T_raw_iou','T_strict_iou'),('T_supported_iou','T_raw_iou'),('T_loss_weighted_iou','T_supported_iou'),('T_validated_loss_weighted_iou','T_loss_weighted_iou'),('T_validated_loss_weighted_iou','T_geometry_raw')]
    for ds in ['hc2','vidstg']:
        selected=[r for r in rows if r['dataset']==ds]
        for n in names:
            rr=[r for r in selected if r['arm']==n]
            for k in rr[0]['metrics']:
                values=[r['metrics'][k] for r in rr if r['metrics'][k] is not None]
                summaries.append(dict(dataset=ds,arm=n,metric=k,mean=None if not values else math.fsum(values)/len(values),count=len(values),unknown_count=4-len(values)))
        lookup={(r['case_id'],r['arm']):r for r in selected}
        for left,right in pairs:
            for k in selected[0]['metrics']:
                diffs=[]
                for cid in [r['case_id'] for r in selected if r['arm']==left]:
                    a=lookup[cid,left]['metrics'][k];b=lookup[cid,right]['metrics'][k]
                    if a is not None and b is not None:diffs.append(a-b)
                contrasts.append(dict(dataset=ds,left=left,right=right,metric=k,count=len(diffs),unknown_count=4-len(diffs),mean_delta_pp=None if not diffs else 100*math.fsum(diffs)/len(diffs),positive=sum(x>1e-12 for x in diffs),negative=sum(x< -1e-12 for x in diffs),neutral=sum(abs(x)<=1e-12 for x in diffs),paired_bootstrap_95_pp=metric._paired_bootstrap(diffs,ds,left+'-'+right,k)))
    write(BASE/'POSTSEAL_ROWS.json',dict(status='complete',GT_read=True,prediction_barrier_sha256=sha(BASE/'GLOBAL_PREDICTION_BARRIER.json'),rows=rows))
    write(BASE/'POSTSEAL_SUMMARY.json',dict(status='complete',GT_read=True,historically_exposed_development=True,summary=summaries,contrasts=contrasts,aggregation='equal source mean; four per direction; no development selection for production'))
    write(BASE/'CPU_COMPLETION.json',dict(status='pending_independent_root_view_public_archive',GT_read=True,count=8,row_count=len(rows),summary_count=len(summaries),contrast_count=len(contrasts),rows_sha256=sha(BASE/'POSTSEAL_ROWS.json'),summary_sha256=sha(BASE/'POSTSEAL_SUMMARY.json')))
    print(json.dumps(dict(status='scored',count=len(rows))))
if __name__=='__main__':main()
