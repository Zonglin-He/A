"""Readback of sealed TTA arrays/traces; additional groups from saved metrics.

No model, GPU, new prediction, metric recomputation or new annotation read.
"""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import csv
import json
from pathlib import Path
import sys
import numpy as np
R=Path(__file__).resolve().parents[2];sys.path[:0]=[str(R),str(R/'tools')]
from full import RUN,read,sha,atomic,check
from full_score import cluster
D=R/'outputs/paper_review_suite_v1';P=R/'results/paper_review_suite_v1/consistency_v2'


def main():
    check()
    seal=read(D/'consistency_v2_sealed.json');score=read(D/'consistency_v2_score_receipt.json')
    assert seal['counts']==dict(queries=384,photo_Source_forwards=384,steps=3840,time_decoder_calls=16896,new_expert=0)
    assert seal['GT_read']is False and score['all384sealed_before_GT']and score['checks']==768
    p=read(D/'consistency_v2_plan.json');assert seal['fingerprint']==p['fingerprint']
    seen=0;report_hashes={};readbacks=[]
    for ds,groups in read(D/'selection.json').items():
        native=read(RUN/ds/'sealed.json')
        old={x['id']:x for x in read(RUN/ds/'metrics.json')['private_rows']}
        for stage,selection in groups.items():
            scored=read(D/'consistency_v2'/ds/(stage+'_metrics.json'))
            rows=scored['private_rows'];assert len(rows)==len(selection)
            signed=[];loss=[];summaries={}
            for i,(row,r)in enumerate(zip(rows,selection),1):
                key=r['id'];assert row['id']==key and row['parent']==old[key]['parent']
                e=seal['files'][ds][stage][key]
                for f,h in [('path','sha256'),('trace_path','trace_sha256'),('flip_path','flip_sha256')]:assert sha(D/e[f])==e[h]
                t=read(D/e['trace_path']);assert t['native_bitwise']and t['episodic_reset_exact']and not t['GT_read']
                oldentry=native['files'][key];assert sha(RUN/ds/oldentry['path'])==oldentry['sha256']
                with np.load(D/e['path'])as z,np.load(RUN/ds/oldentry['path'])as oldz:
                    for field in ['Source','Source_interval','positions8','frame_ids']:assert np.array_equal(z[field],oldz[field])
                for field in ['t','v']:assert row['values']['Source'][field]==old[key]['values']['Source'][field]
                row['duration_group']=old[key]['duration_group']
                for arm in ['Entropy','PhotoConsistency','Full']:
                    dt=100*(row['values'][arm]['t']-row['values']['Source']['t'])
                    dv=100*(row['values'][arm]['v']-row['values']['Source']['v'])
                    signed.append([f'Q{i:03d}',arm,row['duration_group'],dt,dv,dt< -5,dv< -5,dt< -5 or dv< -5])
                for arm in ['Entropy','PhotoConsistency']:
                    trace=t['traces'][arm];assert [x['step']for x in trace]==list(range(6))
                    assert all(np.isfinite(x['loss'])for x in trace)
                    assert all(np.isfinite(x['gradient_norm'])for x in trace[:5])
                    loss.append([f'Q{i:03d}',arm,trace[0]['loss'],trace[-1]['loss'],trace[0]['loss']-trace[-1]['loss']])
                seen+=1
            for arm in ['Entropy','PhotoConsistency','Full']:
                deltas=np.array([[x['values'][arm][f]-x['values']['Source'][f]for f in ['t','v']]for x in rows])*100
                summaries[arm]=dict(severe_t=int((deltas[:,0]< -5).sum()),severe_v=int((deltas[:,1]< -5).sum()),severe_union_t_or_v=int((deltas< -5).any(1).sum()),severe_union_fraction=float((deltas< -5).any(1).mean()),groups={})
                for group in ['short_le5s','long_gt5s']:
                    rr=[x for x in rows if x['duration_group']==group]
                    summaries[arm]['groups'][group]=dict(queries=len(rr),paired_Source=cluster(rr,arm,'Source')if rr else None)
            pub=P/ds/stage
            result=dict(dataset=ds,stage=stage,queries=len(rows),arms=summaries,Source_all_t_v_exact_original_full=True,sealed_arrays_traces_RGBcontext_hashes_all_verified=True,scope='Reuses already official-scored metrics and saved full GT-duration groups; no new GT/metric/model work. Severe union counts a query once.',fingerprint=p['fingerprint'])
            atomic(pub/'tails_timegroups_readback.json',result)
            for filename,header,values in [
                ('severe_signed.tsv',['query','arm','duration_group','delta_t_pp','delta_v_pp','severe_t_gt5pp','severe_v_gt5pp','severe_union_once'],signed),
                ('objective_changes.tsv',['query','arm','before_loss','after_loss','loss_decrease'],loss)]:
                with(pub/filename).open('w')as f:w=csv.writer(f,delimiter='\t',lineterminator='\n');w.writerow(header);w.writerows(values)
            report_hashes.update({str(f.relative_to(R)):sha(f)for f in pub.iterdir()if f.is_file()})
            readbacks.append(dict(dataset=ds,stage=stage,queries=len(rows),allSource_equal=True))
    assert seen==384
    receipt=dict(status='all384_bound_outputs_and_traces_readback_verified',queries=seen,counts=seal['counts'],predict_seconds=seal['seconds'],official_dense_checks=score['checks'],official_dense_max_error=score['max_error'],all_predictions_seal_before_labels=True,native_and_postreset_bitwise=True,actual_clean_Source_acquisition='384 earlier original Source calls from shared support study, reused here; their time is not zero deployment cost',photometric_Source_forwards=384,optimizer_steps=3840,extra_TimeDecoder_calls=16896,additional_expert=0,new_model_or_prediction_calls=0,new_GT_or_metric_calls=0,seal_sha256=sha(D/'consistency_v2_sealed.json'),score_receipt_sha256=sha(D/'consistency_v2_score_receipt.json'),reports=report_hashes,readbacks=readbacks,code_sha256=sha(Path(__file__)))
    atomic(D/'consistency_v2_completion_readback.json',receipt);atomic(P/'completion_readback.json',receipt)
    print(json.dumps({k:receipt[k]for k in ['status','queries','predict_seconds','official_dense_checks']}))


if __name__=='__main__':main()
