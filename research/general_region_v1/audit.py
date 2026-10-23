"""Postseal compatibility/placement/tail audit; no new labels or predictions."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from research.general_region_v1.run import *
import numpy as np,torch

def main():
    torch.set_num_threads(1);start=time.perf_counter();p=verify();s=read(D/'sealed.json');r=read(P/'report.json');receipt=read(D/'score_receipt.json');assert receipt['report_sha256']==sha(P/'report.json')
    rows={x['query']:x for d in r['domains'].values()for x in d['rows']};out=[];maxerr=0
    for item in p['items']:
        key=item['alias'];e=s['files'][key];assert sha(D/e['trace_path'])==e['trace_sha256'];tr=torch.load(D/e['trace_path'],weights_only=False,map_location='cpu');L=tr['feedback']['L'];n=np.linalg.norm(L[PRIMARY]);m=np.linalg.norm(L['K16Matched']);err=abs(n-m);maxerr=max(maxerr,err)
        corr={}
        for a in ['FullGrid','Reflected','PositionOnly']:
            if np.std(L[PRIMARY])>0 and np.std(L[a])>0:corr[a]=float(np.corrcoef(L[PRIMARY],L[a])[0,1])
            else:corr[a]=None
        row=rows[key];iv={a:row['movement'][a]for a in ARMS}
        out.append(dict(query=key,dataset=item['dataset'],matched_strength_error=err,correlations=corr,
            same_interval_as={a:iv[a]==iv[PRIMARY]for a in ['NativeG6','K16','FullGrid','Reflected','K16Matched']},
            severe_time=bool(row['values'][PRIMARY]['t']<row['values']['NativeG6']['t']-.05),
            severe_v=bool(row['values'][PRIMARY]['v']<row['values']['NativeG6']['v']-.05)))
    assert maxerr<1e-10
    domains={}
    for ds in ['hc2','vidstg']:
        rr=[x for x in out if x['dataset']==ds];values=r['domains'][ds]['rows'];cases=[]
        for sign in [1,-1]:
            selected=sorted(values,key=lambda x:sign*(x['values'][PRIMARY]['t']-x['values']['NativeG6']['t']),reverse=True)[:4]
            for row in selected:
                cases.append(dict(query=row['query'],tail='largest_gain'if sign==1 else'largest_loss',duration_group=row['duration_group'],
                    tiou_percent={a:100*row['values'][a]['t']for a in ['Source','NativeG6','K16',PRIMARY,'FullGrid','Reflected','K16Matched']},
                    length_change_seconds=row['values'][PRIMARY]['length_seconds']-row['values']['NativeG6']['length_seconds']))
        severe=[x for x in rr if x['severe_time']]
        domains[ds]=dict(count=len(rr),same_interval_counts={a:sum(x['same_interval_as'][a]for x in rr)for a in rr[0]['same_interval_as']},
            evidence_correlation_median={a:float(np.median(v))if(v:=[x['correlations'][a]for x in rr if x['correlations'][a]is not None])else None for a in ['FullGrid','Reflected','PositionOnly']},
            severe_time_count=len(severe),severe_time_same_interval_as_FullGrid=sum(x['same_interval_as']['FullGrid']for x in severe),
            mean_target_reflected_area_overlap=float(np.mean([x['target_reflection_overlap']for x in values])),fallback_frames=sum(x['fallback_frames']for x in values),cases=cases)
    result=dict(kind='Postsealed scored-output/source-only audit, no new selection or prediction',code_sha256=sha(Path(__file__)),plan_sha256=sha(D/'plan.json'),seal_sha256=sha(D/'sealed.json'),report_sha256=sha(P/'report.json'),score_receipt_sha256=sha(D/'score_receipt.json'),all256_Source_G6_K16_aliases_bitwise=True,matched_L2_max_abs_error=maxerr,domains=domains,rows=out,new_GT_read=False,new_model=0,new_predictions=0,seconds=time.perf_counter()-start)
    put(P/'POSTSEAL_AUDIT.json',result);print(json.dumps({k:{a:b for a,b in v.items()if a!='cases'}for k,v in domains.items()},indent=2))
if __name__=='__main__':main()
