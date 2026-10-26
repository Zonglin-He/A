"""Whole-domain official scoring and a separate sufficient-state/input auditor."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
from gamma_full_common import *
import argparse,subprocess,traceback
def score(ds):
    import torch,numpy as np
    from full_score import metric
    torch.set_num_threads(1);validate();seal=read(D/f'{ds}_sealed.json')
    assert seal['queries']==COUNTS[ds]and not seal['GT_read'];truth=selected.labels(ds);rr=[];checks=0;maxerr=0.
    for row in records(ds):
        key=row['id'];e=committed(ds,key);assert e==seal['files'][key];g=truth[key];values={}
        with np.load(D/e['path'])as z:
            inp={**row,'metadata':{**row['metadata'],'frame_ids':z['frame_ids'].tolist()}}
            for arm in ARMS:
                values[arm],err=metric(z[arm],inp,{int(k):v for k,v in g['boxes'].items()},g['interval'],z[arm+'_interval'].tolist(),z['positions8'])
                assert err<2e-10;maxerr=max(maxerr,err);checks+=1
        rr.append(dict(id=key,parent=str(g['parent']),duration_group='short_le5s'if(g['interval'][1]-g['interval'][0])/row['metadata']['fps']<=5 else'long_gt5s',values=values))
    folder=P/ds;report=selected.suite().parent_report(rr,ARMS,pairs(ds),folder);selected.suite().signed_tails(rr,pairs(ds),folder)
    atomic(D/ds/'metrics.json',dict(private_rows=rr,report=report))
    rec=dict(status='scored',queries=len(rr),checks=checks,max_error=maxerr,seal_sha256=sha(D/f'{ds}_sealed.json'),whole_domain_sealed_before_GT=True)
    atomic(D/f'{ds}_score_receipt.json',rec);atomic(folder/'score_receipt.json',rec)
def audit(ds):
    import torch,numpy as np
    torch.set_num_threads(1);validate();seal=read(D/f'{ds}_sealed.json');sp=D/f'{ds}_score_receipt.json'
    assert read(sp)['seal_sha256']==sha(D/f'{ds}_sealed.json');checks=0;segments={}
    for row in records(ds):
        key=row['id'];e=committed(ds,key);assert e==seal['files'][key]and e['input_binding']==binding(row,read(RUN/'plan.json')['provider_runtime_fingerprint'])
        current,ce=checked_current(ds,key);assert e['current_prediction_sha256']==ce['sha256']
        for f,h in [('capture','capture_sha256'),('prediction','prediction_sha256'),('receipt','receipt_sha256')]:assert sha(e['parent'][f])==e['parent'][h];checks+=1
        compact=torch.load(D/e['compact_path'],map_location='cpu',weights_only=False,mmap=True);assert not compact['GT_read'];a=complete_arrays(compact,current)
        with np.load(D/e['path'])as z:assert set(z.files)==set(a)and all(np.array_equal(z[k],a[k])for k in a)
        da=default_arm(ds);assert np.array_equal(a[da],current['FullMAP'])and np.array_equal(a[da+'_interval'],current['FullMAP_interval'])
        if e['teacher']:
            te=e['teacher'];assert sha(te['payload_path'])==te['payload_sha256']and sha(te['receipt_path'])==te['receipt_sha256'];checks+=2
        checks+=3;w=e['worker'];ident=f"{w['pid']}:{w['startticks']}"
        s=segments.setdefault(ident,dict(queries=0,Source_seconds=0.,query_elapsed_wall_seconds=0.,modes={}))
        s['queries']+=1;s['Source_seconds']+=e['Source_seconds'];s['query_elapsed_wall_seconds']+=e['query_elapsed_wall_seconds'];s['modes'][e['mode']]=s['modes'].get(e['mode'],0)+1
    rr=read(D/ds/'metrics.json')['private_rows'];report=read(P/ds/'report.json')
    for arm,means in report['means'].items():
        for k,v in means.items():
            x=[r['values'][arm][k]for r in rr if r['values'][arm][k]is not None]
            assert(v is None and not x)or abs(v-float(np.mean(x)))<1e-12
    rec=dict(status='verified_whole_domain',queries=len(rr),SHA_checks=checks,all_five_gamma_arrays_cold_rederived_exact=True,default_current_all_queries_bitwise=True,all_means_recomputed=True,score_sha256=sha(sp),worker_segments=segments,model_load=seal['model_load'],actual_counts=seal['counts'],new_GT_scoring=0,new_expert=0,inherited_acquisition_cost_nonzero=True,scope='Full exposed-target descriptive sensitivity, not default selection or unseen validation. Source/model/teacher/compatibility/build/query/whole clocks are nested or separate as actually recorded, never added as deployment latency.')
    atomic(D/f'{ds}_reporting_receipt.json',rec);atomic(P/ds/'reporting_receipt.json',rec)
def domain(ds):
    validate();atomic(D/f'{ds}_CPU_worker.json',identity())
    try:
        for action in ['score','audit']:
            assert not(D/f'{ds}_{action}_exit.json').exists();atomic(D/f'{ds}_CPU_state.json',dict(status=action))
            with(D/f'{ds}_{action}.log').open('ab')as log:
                c=subprocess.Popen([sys.executable,'-B',str(S/'report.py'),action,'--dataset',ds],stdout=log,stderr=subprocess.STDOUT)
                atomic(D/f'{ds}_{action}_child.json',identity(c.pid));rc=c.wait()
            atomic(D/f'{ds}_{action}_exit.json',dict(code=rc));assert rc==0,'Preserve scoring/audit failure, no retry'
        atomic(D/f'{ds}_CPU_state.json',dict(status='complete_scored_audited_pending_root_analysis_curves_publication'))
    except BaseException:atomic(D/f'{ds}_CPU_state.json',dict(status='failed_preserved',error=traceback.format_exc()));raise
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('command',choices=['score','audit','domain']);a.add_argument('--dataset',choices=COUNTS,required=True);z=a.parse_args();globals()[z.command](z.dataset)
