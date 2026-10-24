"""Postseal reporting only; never performs model inference or new GT scoring."""
from common import *
import argparse

def run(stage):
    import csv,numpy as np
    p=validate('tta');sealpath=D/f'tta_{stage}_sealed.json';scorepath=D/f'tta_{stage}_score_receipt.json'
    assert sealpath.exists()and scorepath.exists(),'Whole stage seal + independent score required'
    seal=read(sealpath);score=read(scorepath);assert score['seal_sha256']==sha(sealpath)and not seal['GT_read'];output={};checks=0
    for ds,queries in seal['files'].items():
        trials=p['trials']if stage=='dev'else list(read(D/'tta_selected.json')[ds].values());output[ds]={};signed=[]
        for t in trials:
            traces=[]
            for key,entries in queries.items():
                e=entries[t['id']];assert sha(D/e['path'])==e['sha256']and sha(D/e['trace_path'])==e['trace_sha256'];r=read(D/e['trace_path']);assert r['reset_exact'];traces.append(r);checks+=2
                ev=r['iterations'];empty_first=sum(x.get('reliable_counts')is not None and x['reliable_counts'][0]==0 for x in ev);empty_second=sum(x.get('reliable_counts')is not None and len(x['reliable_counts'])>1 and x['reliable_counts'][1]==0 for x in ev)
                signed.append([t['id'],f'Q{list(queries).index(key)+1:03}',r['status'],r['parameter_delta'],r['seconds'],empty_first,empty_second])
            output[ds][t['id']]=dict(queries=len(traces),numerical_failures=sum(r['status']!='ok'for r in traces),norm_state_changed=sum((r['parameter_delta']or 0)>0 for r in traces),norm_state_exactly_unchanged=sum(r['parameter_delta']==0 for r in traces),seconds_sum=sum(r['seconds']for r in traces),mean_seconds=float(np.mean([r['seconds']for r in traces])),timing_scope='Actual saved-context conditional optimizer/reset/readout pipeline; excludes original Source acquisition/loading and early context. Not a full end-to-end deployment benchmark. Same-stage whole-worker wall includes loading and IO separately.',official_source_unmodified=True)
        folder=P/'tta'/stage/ds;atomic(folder/'optimizer_reliability_cost.json',output[ds])
        with(folder/'all_optimizer_signed.tsv').open('w')as f:
            w=csv.writer(f,delimiter='\t',lineterminator='\n');w.writerow(['trial','query','status','norm_delta_L2','conditional_seconds','first_empty_iterations','second_empty_iterations']);w.writerows(signed)
        if stage=='confirm':
            private={t['method']:read(D/'tta'/stage/ds/(t['id']+'_metrics.json'))['private_rows']for t in trials};a,b=private['Tent'],private['SAR'];assert len(a)==len(b);joined=[]
            for ra,rb in zip(a,b):
                assert ra['id']==rb['id']and ra['parent']==rb['parent'];assert ra['values']['Source']==rb['values']['Source']and ra['values']['Full']==rb['values']['Full'];joined.append({**ra,'values':{**ra['values'],'SAR':rb['values']['SAR']}})
            report(joined,['Source','Tent','SAR','Full'],[('Tent','Source'),('SAR','Source'),('SAR','Tent'),('Full','Tent'),('Full','SAR')],folder/'paired_all_methods')
    receipt=dict(stage=stage,seal_sha256=sha(sealpath),score_sha256=sha(scorepath),prediction_trace_SHA_checks=checks,counts=seal['counts'],whole_stage_wall_seconds=seal['seconds'],all_official_vendor_SHA_reverified_by_predict=True,source_code_sha256=sha(Path(__file__)),new_GT_scoring=0,new_model=0,new_expert=0)
    atomic(P/'tta'/stage/'reporting_receipt.json',receipt);atomic(D/f'tta_{stage}_reporting_receipt.json',receipt);print(receipt)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--stage',choices=['dev','confirm'],required=True);run(a.parse_args().stage)
