"""Same official metrics, only after all 128 predictions and root seal exist."""
import json
import math
import sys
import statistics
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.stvg_lite_event_common_v1 import BASE,read,write,sha,load_case,roster,verify


def main():
    import torch
    from scripts import review_stvg_sparse_r1_v2 as truth
    from scripts import score_stvg_native_instance_nativeclock005 as metric
    from scripts.measure_stvg_component_trial005 import _aggregate,METRICS
    verify();b=BASE/'trials'/sys.argv[1];assert not (b/'CPU_COMPLETION.json').exists()
    bar=read(b/'STUDENT_GLOBAL_BARRIER.json');root=read(b/'ROOT_READBACK.json');pred=read(b/'GLOBAL_PREDICTION_BARRIER.json')
    assert root['status']=='pass' and bar['status']==pred['status']=='complete' and root['count']==bar['count']==pred['count']==128
    assert pred['student_barrier_sha256']==sha(b/'STUDENT_GLOBAL_BARRIER.json') and pred['root_readback_sha256']==sha(b/'ROOT_READBACK.json') and pred['scorer_sha256']==sha(Path(__file__))
    for row in pred['files']:assert sha(ROOT/row['path'])==row['sha256']
    cases={};payloads={};refs={r['case_id']:r for r in read(b/'REFERENCE_LOCK.json')['records']}
    for rec,item in zip(roster(),bar['files']):
        cid=rec['probe_alias'];case,sp=load_case(rec);cases[cid]=case
        assert item['case_id']==cid
        z=torch.load(ROOT/item['path'],map_location='cpu',weights_only=False)
        assert z['source_case_sha256']==case['source_case_sha256'] and z['source_restored'] and not z['GT_read'];payloads[cid]=z
    bindings=dict(student_barrier_sha256=sha(b/'STUDENT_GLOBAL_BARRIER.json'),root_readback_sha256=sha(b/'ROOT_READBACK.json'),prediction_barrier_sha256=sha(b/'GLOBAL_PREDICTION_BARRIER.json'),runtime_sha256=sha(b/'RUNTIME.json'))
    write(b/'POSTSEAL_GT_EXPOSURE.json',dict(status='after_full_128_seal_and_root',count=128,bindings=bindings,GT_read=True,time=time.time()))
    targets,_=truth._truths_after_seal(cases);official,independent,_=truth._metric_functions();rows=[];cost=[]
    for rec in roster():
        cid=rec['probe_alias'];case=cases[cid];z=payloads[cid]
        support={int(a['position'])for a in refs[cid]['anchors']if a['weight']>0}
        for arm in ('F','T','S_M2','TS_M2'):
            score,audit=metric._score_student_arm(z['arms'][arm],case,targets[cid],support,official,independent)
            rows.append(dict(case_id=cid,dataset=rec['dataset'],arm=arm,status='scored',metrics=score,support=audit['support'],metric_audit=audit['metric_audit']))
        fit=z['fits']['M2']
        cost.append(dict(case_id=cid,dataset=rec['dataset'],has_reference=fit is not None,fit_cost=None if fit is None else fit['matched_efficiency'],full_forward_seconds=None if fit is None else z['readouts']['M2']['matched_complete_forward_wall_seconds'],accepted=0 if fit is None else fit['optimizer_steps'],proposed=0 if fit is None else fit['proposed_optimizer_steps']))
    summary,paired=_aggregate(rows,('F','T','S_M2','TS_M2'))
    common=dict(status='complete',count=128,row_count=512,bindings=bindings,GT_read=True,bootstrap_executed=False)
    write(b/'POSTSEAL_ROWS.json',dict(common,rows=rows));write(b/'POSTSEAL_SUMMARY.json',dict(common,summary=summary));write(b/'POSTSEAL_PAIRED.json',dict(common,paired_contrasts=paired))
    write(b/'MATCHED_EFFICIENCY_READBACK.json',dict(status='complete',count=128,records=cost,scope='same input/settings, CUDA-synchronized fit only; excludes Source/model/Teacher and full-final-forward latency',GT_read=False))
    write(b/'POSTSEAL_SCORE_BARRIER.json',dict(status='complete',count=512,case_count=128,rows_sha256=sha(b/'POSTSEAL_ROWS.json'),summary_sha256=sha(b/'POSTSEAL_SUMMARY.json'),paired_sha256=sha(b/'POSTSEAL_PAIRED.json'),gt_exposure_sha256=sha(b/'POSTSEAL_GT_EXPOSURE.json'),bindings=bindings,GT_read=True))
    write(b/'CPU_COMPLETION.json',dict(status='complete',count=128,row_count=512,score_barrier_sha256=sha(b/'POSTSEAL_SCORE_BARRIER.json'),CPU_only=True,GT_read=True,bootstrap_executed=False,time=time.time()))
    print('LITE_SCORE_COMPLETE',b.name,128,512,flush=True)


if __name__=='__main__':main()
