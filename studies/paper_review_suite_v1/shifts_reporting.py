"""Post-seal signed tails, time groups and measured component costs.

Consumes already scored metrics only. No model, new predictions or GT access.
"""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import argparse
import csv
from pathlib import Path
import sys
import numpy as np
R=Path(__file__).resolve().parents[2];sys.path[:0]=[str(R),str(R/'tools')]
from full import RUN,read,sha,atomic
from full_score import cluster
D=R/'outputs/paper_review_suite_v1';P=R/'results/paper_review_suite_v1/shifts'


def scalar_components(value,prefix=''):
    result={}
    for key,item in value.items():
        name=prefix+key
        if isinstance(item,dict):result.update(scalar_components(item,name+'.'))
        elif isinstance(item,(int,float))and not isinstance(item,bool):result[name]=float(item)
    return result


def main(stage):
    receipt=read(D/('shifts_'+stage+'_score_receipt.json'));seal_path=D/('shifts_'+stage+'_sealed.json');seal=read(seal_path)
    assert receipt['status']=='scored'and receipt['seal_sha256']==sha(seal_path)and not seal['GT_read']
    selection=read(D/'selection.json');expected=64 if stage=='screen'else 128;assert seal['queries']==2*expected
    outputs={}
    for ds,groups in selection.items():
        old={x['id']:x for x in read(RUN/ds/'metrics.json')['private_rows']}
        for name in ['Blur','JPEG','Dim']:
            metrics=read(D/'shifts'/stage/ds/(name+'_metrics.json'));report=read(P/ds/stage/name/'report.json')
            rows=metrics['private_rows'];assert len(rows)==len(groups[stage])==expected
            for item,row in zip(rows,groups[stage]):
                assert item['parent']==old[row['id']]['parent'];item['duration_group']=old[row['id']]['duration_group']
                entry=seal['files'][ds][row['id']][name]
                assert sha(D/entry['path'])==entry['sha256']and entry['transformed_RGB_sha256']!=entry['original_RGB_sha256']
            for arm in ['Source','Full','CleanSource','CleanFull']:
                for field in ['t','v']:assert abs(np.mean([x['values'][arm][field]for x in rows])-report['means'][arm][field])<1e-14
            comparisons=[('Full','Source'),('Source','CleanSource'),('Full','CleanFull')]
            data=[];severe={};timegroups={}
            for a,b in comparisons:
                key=a+'-'+b;delta=np.array([[x['values'][a][f]-x['values'][b][f]for f in ['t','v']]for x in rows])*100
                severe[key]=dict(t=int((delta[:,0]<-5).sum()),v=int((delta[:,1]<-5).sum()),union_once=int((delta<-5).any(axis=1).sum()),queries=len(rows))
                timegroups[key]={g:dict(queries=sum(x['duration_group']==g for x in rows),paired=cluster([x for x in rows if x['duration_group']==g],a,b,fields=[('t',100),('v',100)]))for g in ['short_le5s','long_gt5s']if any(x['duration_group']==g for x in rows)}
                for i,(x,dd)in enumerate(zip(rows,delta),1):data.append([f'Q{i:03d}',x['duration_group'],key,*dd,dd[0]<-5,dd[1]<-5,bool((dd<-5).any())])
            component_rows=[scalar_components(seal['files'][ds][x['id']][name])for x in groups[stage]]
            keys=sorted(set().union(*(set(x)for x in component_rows)))
            costs={k:dict(mean=float(np.mean([x[k]for x in component_rows if k in x])),sum=float(np.sum([x[k]for x in component_rows if k in x])),recorded_queries=sum(k in x for x in component_rows))for k in keys}
            summary=dict(dataset=ds,stage=stage,shift=name,queries=len(rows),severe=severe,time_groups=timegroups,measured_component_costs=costs,cost_scope='Components may be nested; do not sum overlapping Source/capture/finish/expert totals. Cache flags/forward counts retain actual acquisition semantics.',new_predictions=0,new_GT_scoring=0,whole_stage_seal_and_score_verified=True)
            dest=P/ds/stage/name;atomic(dest/'tails_timegroups_cost.json',summary)
            with(dest/'all_signed_severe_timegroups.tsv').open('w')as out:
                writer=csv.writer(out,delimiter='\t',lineterminator='\n');writer.writerow(['query','duration_group','comparison','delta_t_pp','delta_v_pp','severe_t_gt5pp','severe_v_gt5pp','severe_union_once']);writer.writerows(data)
            outputs[ds+'/'+name]=dict(queries=len(rows),public_summary_sha256=sha(dest/'tails_timegroups_cost.json'),signed_sha256=sha(dest/'all_signed_severe_timegroups.tsv'))
    atomic(D/('shifts_'+stage+'_reporting_receipt.json'),dict(status='complete',score_receipt_sha256=sha(D/('shifts_'+stage+'_score_receipt.json')),code_sha256=sha(Path(__file__)),outputs=outputs,new_predictions=0,new_GT_scoring=0))
    print('Post-seal shift tails/timegroups/cost complete: '+stage)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['screen','confirm'],required=True);main(parser.parse_args().stage)
