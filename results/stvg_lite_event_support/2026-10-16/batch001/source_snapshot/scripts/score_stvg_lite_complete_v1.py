"""Score five final arms only after BOTH full and Native-clock pipelines seal."""
from pathlib import Path
import math
import statistics
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.stvg_lite_event_common_v1 import BASE,read,write,sha,verify,load_case
ARMS=('Frozen','Temporal','Spatial_native_clock','Full','Matched_original_OPD')
METRICS=('m_vIoU','m_tIoU','m_sIoU_dense_GT','m_sIoU_supervised_GT','m_sIoU_unsupervised_GT')


def aggregate(rows,per_domain):
    summary=[];paired=[];index={(r['case_id'],r['arm']):r for r in rows};assert len(index)==len(rows)
    for ds in ('hc2','vidstg'):
        ids=sorted({r['case_id']for r in rows if r['dataset']==ds});assert len(ids)==per_domain
        for arm in ARMS:
            for metric in METRICS:
                vals={cid:index[cid,arm]['metrics'][metric]for cid in ids};known=[float(v)for v in vals.values()if v is not None]
                assert all(math.isfinite(v)for v in known)
                summary.append(dict(dataset=ds,arm=arm,metric=metric,count=per_domain,n_valid=len(known),unknown_case_ids=[cid for cid,v in vals.items()if v is None],mean=math.fsum(known)/len(known)if known else None,median=statistics.median(known)if known else None,min=min(known)if known else None,max=max(known)if known else None))
            vals=[index[cid,arm]['metrics']['m_vIoU']for cid in ids]
            for threshold in (.3,.5):
                known=[v for v in vals if v is not None]
                summary.append(dict(dataset=ds,arm=arm,metric=f'R@vIoU>={threshold:.1f}',mean=sum(v>=threshold for v in known)/len(known)if known else None,n_valid=len(known),count=per_domain,unknown_case_ids=[cid for cid,v in zip(ids,vals)if v is None]))
        for left,right in [('Temporal','Frozen'),('Spatial_native_clock','Frozen'),('Full','Frozen'),('Full','Temporal'),('Full','Spatial_native_clock'),('Full','Matched_original_OPD')]:
            for metric in METRICS:
                values={};unknown=[]
                for cid in ids:
                    a=index[cid,left]['metrics'][metric];b=index[cid,right]['metrics'][metric]
                    if a is None or b is None:unknown.append(cid)
                    else:values[cid]=100*(float(a)-float(b))
                vals=list(values.values());tol=1e-6
                paired.append(dict(dataset=ds,contrast=left+'_minus_'+right,metric=metric,per_source_delta_pp=values,n_pairs=len(vals),unknown_case_ids=unknown,positive_case_ids=[c for c,v in values.items()if v>tol],negative_case_ids=[c for c,v in values.items()if v< -tol],neutral_case_ids=[c for c,v in values.items()if abs(v)<=tol],neutral_tolerance_pp=tol,mean_delta_pp=math.fsum(vals)/len(vals)if vals else None,median_delta_pp=statistics.median(vals)if vals else None,worst_delta_pp=min(vals)if vals else None))
    return summary,paired


def main():
    import torch
    from scripts import review_stvg_sparse_r1_v2 as truth
    from scripts import score_stvg_native_instance_nativeclock005 as metric
    verify();cohort=sys.argv[1];assert cohort in ('development','confirmation')
    scope=BASE if cohort=='development'else BASE/'confirmation';count=128 if cohort=='development'else 64
    out=scope/'complete';out.mkdir(parents=True,exist_ok=True);assert not(out/'CPU_COMPLETION.json').exists()
    root_roster=ROOT/'artifacts/stvg_soft_ep_coordinate128_v1/ROSTER_LOCK.json'if cohort=='development'else scope/'ROSTER_LOCK.json'
    rr=read(root_roster)['records'];assert len(rr)==count
    plan=read(scope/'COMPLETE_PIPELINE_LOCK.json');assert plan['status']=='locked' and plan['recipe_sha256']==sha(BASE/'FINAL_RECIPE_LOCK.json')
    assert plan['count']==count and plan['cohort']==cohort and plan['no_further_method_selection']is True
    assert set(plan['trials'])=={'full','native'} and plan['trials']['full']!=plan['trials']['native']
    recipe=read(BASE/'FINAL_RECIPE_LOCK.json');assert recipe['status']=='locked'
    payloads={};refs={};bindings={};allfiles=[]
    for kind in ('full','native'):
        b=scope/'trials'/plan['trials'][kind];root=read(b/'ROOT_READBACK.json');bar=read(b/'STUDENT_GLOBAL_BARRIER.json');pred=read(b/'GLOBAL_PREDICTION_BARRIER.json')
        design=read(b/'DESIGN_LOCK.json')
        assert design['cohort']==cohort and design['reference_clock']==kind and design['recipe_sha256']==sha(BASE/'FINAL_RECIPE_LOCK.json')
        assert design['recipe']==recipe['selected_spatial_recipe'] and design['response_scope']==recipe['selected_response_scope']
        for ds in ('hc2','vidstg'):assert design['configs'][ds]['M0']==design['configs'][ds]['M2']==recipe['configs'][ds]
        assert root['status']=='pass' and root['count']==count and bar['status']==pred['status']=='complete' and bar['count']==pred['count']==count and not pred['GT_read']
        assert pred['root_readback_sha256']==sha(b/'ROOT_READBACK.json') and pred['student_barrier_sha256']==sha(b/'STUDENT_GLOBAL_BARRIER.json') and pred['scorer_sha256']==sha(Path(__file__))
        for f in pred['files']:assert sha(ROOT/f['path'])==f['sha256']
        payloads[kind]={}
        for f in bar['files']:
            d=torch.load(ROOT/f['path'],map_location='cpu',weights_only=False);assert not d['GT_read'];payloads[kind][d['case_id']]=d
        assert set(payloads[kind])=={r['probe_alias']for r in rr}
        refs[kind]={r['case_id']:r for r in read(b/'REFERENCE_LOCK.json')['records']}
        for name in ('ROOT_READBACK.json','STUDENT_GLOBAL_BARRIER.json','GLOBAL_PREDICTION_BARRIER.json','RUNTIME.json','DESIGN_LOCK.json','REFERENCE_LOCK.json','TEMPORAL_LOCK.json'):
            bindings[str((b/name).relative_to(ROOT))]=sha(b/name)
        allfiles.extend(pred['files'])
    bindings[str(root_roster.relative_to(ROOT))]=sha(root_roster);bindings[str((BASE/'FINAL_RECIPE_LOCK.json').relative_to(ROOT))]=sha(BASE/'FINAL_RECIPE_LOCK.json')
    bindings[str((scope/'COMPLETE_PIPELINE_LOCK.json').relative_to(ROOT))]=sha(scope/'COMPLETE_PIPELINE_LOCK.json')
    cases={r['probe_alias']:load_case(r)[0]for r in rr}
    for rel,h in bindings.items():assert sha(ROOT/rel)==h
    write(out/'GLOBAL_PREDICTION_BARRIER.json',dict(status='complete',count=count,arm_count=5,files=allfiles,bindings=bindings,GT_read=False))
    write(out/'POSTSEAL_GT_EXPOSURE.json',dict(status='after_all_full_and_native_predictions_and_independent_root',count=count,bindings=bindings,GT_read=True,time=time.time()))
    targets,_=truth._truths_after_seal(cases);official,independent,_=truth._metric_functions();rows=[]
    for rec in rr:
        cid=rec['probe_alias'];case=cases[cid];full=payloads['full'][cid];native=payloads['native'][cid]
        assert native['temporal_record']['selected_interval']==case['Native_interval'] and native['arms']['T']['interval']==case['Native_interval']
        assert native['reference_records']['source_case_sha256']==full['reference_records']['source_case_sha256']==case['source_case_sha256']
        mapping={'Frozen':full['arms']['F'],'Temporal':full['arms']['T'],'Spatial_native_clock':native['arms']['S_M2'],'Full':full['arms']['TS_M2'],'Matched_original_OPD':full['arms']['TS_M0']}
        for arm,prediction in mapping.items():
            kind='native'if arm=='Spatial_native_clock'else'full'
            support={a['position']for a in refs[kind][cid]['anchors']if a['weight']>0}
            scores,audit=metric._score_student_arm(prediction,case,targets[cid],support,official,independent)
            rows.append(dict(case_id=cid,dataset=rec['dataset'],parent_sha256=rec['parent_sha256'],arm=arm,status='scored',metrics=scores,support=audit['support'],support_clock=kind,metric_audit=audit['metric_audit']))
    summary,paired=aggregate(rows,count//2)
    common=dict(status='complete',count=count,row_count=count*5,bindings=bindings,GT_read=True,bootstrap_executed=False,new_selection_using_these_complete_outputs=False,cohort_role='exposed_development_closed_loop'if cohort=='development'else'disjoint_from_current128_confirmation',CURRENT_promoted=False,matched_original_OPD_definition='original validated OPD algorithm with exactly the SAME final parameters and teacher information as M2; not its separately retuned optimum')
    write(out/'POSTSEAL_ROWS.json',dict(common,rows=rows));write(out/'POSTSEAL_SUMMARY.json',dict(common,summary=summary));write(out/'POSTSEAL_PAIRED.json',dict(common,paired_contrasts=paired))
    files={n:sha(out/n)for n in ('POSTSEAL_ROWS.json','POSTSEAL_SUMMARY.json','POSTSEAL_PAIRED.json','POSTSEAL_GT_EXPOSURE.json','GLOBAL_PREDICTION_BARRIER.json')}
    write(out/'CPU_COMPLETION.json',dict(common,CPU_only=True,scope='five final arms; supervised-frame supports differ across reference clocks and are not directly mixed',output_sha256=files))
    print('COMPLETE_FIVE_ARMS_SCORED',cohort,count,flush=True)


if __name__=='__main__':main()
