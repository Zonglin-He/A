"""Independent CPU scoring only after all64 predictions are sealed."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
import sys, time, json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from run import R,D,P,OLD,ARMS,sha,put,read,verify,records
import numpy as np
import torch
from stvg.metrics import DenseTube, official, box_iou
from stvg.metrics_base import xyxy


def evaluate(boxes,row,gt,interval,positions):
    truth={int(k):v for k,v in gt['boxes'].items()}
    rr=dict(input=row['metadata'],frame_ids=row['metadata']['frame_ids'])
    a=official(boxes,rr,truth,gt['interval'],interval,'hc2')
    b=DenseTube(boxes,rr,truth,gt['interval'],clip=True).score(interval)
    error=max(abs(a[k]-b[k])for k in a);assert error<2e-10
    ids=rr['frame_ids'];obs={ids[int(i)]for i in positions};pix=np.maximum(xyxy(boxes,row['metadata']['width'],row['metadata']['height']),0)
    for name,want in [('observed_s',True),('unobserved_s',False)]:
        values=[box_iou(pix[i],truth[f])for i,f in enumerate(ids)if f in truth and (f in obs)==want]
        a[name]=float(np.mean(values))if values else None
    fps=row['metadata']['fps'];span=gt['interval']
    a.update(boundary_MAE_seconds=float((abs(interval[0]-span[0])+abs(interval[1]-span[1]))/(2*fps)),
             length_seconds=float((interval[1]-interval[0])/fps),zero_time_overlap=float(a['t']==0))
    return {k:float(v)if v is not None else None for k,v in a.items()},error


def main():
    torch.set_num_threads(1);p=verify();seal=read(D/'sealed.json');assert seal['count']==64 and not seal['GT_read'] and not seal['CUDA_initialized']
    assert seal['plan_sha256']==sha(D/'plan.json') and not (D/'score_receipt.json').exists()
    for e in seal['files'].values():
        assert sha(D/e['path'])==e['sha256'] and sha(D/e['trace_path'])==e['trace_sha256']
    put(D/'pre_GT_validation.json',dict(all64_outputs_hash_verified=True,seal_sha256=sha(D/'sealed.json'),CUDA_initialized=torch.cuda.is_initialized()))
    labels=R/'data/evaluation/development_labels.jsonl';assert sha(labels)==p['labels_sha256']
    truth={g['id']:g for g in map(json.loads,labels.read_text().splitlines())if g['dataset']=='hc2'}
    prior=read(R/'results/event_query_hc64_v1/report.json');prior={r['query']:r for r in prior['rows']}
    rows=[];maxerror=0.;checks=0;begin=time.perf_counter();arms=ARMS+['Source']
    for row in records():
        key=row['id'];gt=truth[key];e=seal['files'][key]
        with np.load(D/e['path'],allow_pickle=False)as z:
            intervals={a:z[a+'_interval'].tolist()for a in arms};base=z['NativeG6'];values={};fixed={}
            for a in arms:
                values[a],err=evaluate(z[a],row,gt,intervals[a],z['positions8']);checks+=1;maxerror=max(maxerror,err)
                fixed[a],err=evaluate(base,row,gt,intervals[a],z['positions8']);checks+=1;maxerror=max(maxerror,err)
            assert values['NativeG6']==prior[key]['values']['NativeG6']
            assert values['Source']==prior[key]['values']['Source']
            tr=torch.load(D/e['trace_path'],map_location='cpu',weights_only=False)
            pairs=tr['pairs'];ids=np.asarray(row['metadata']['frame_ids']);spans=np.stack([ids[pairs[:,0]],ids[pairs[:,1]]+1],-1)
            start,end=gt['interval']
            # Correct union, including disjoint/partially overlapping intervals.
            overlap=np.maximum(0,np.minimum(end,spans[:,1])-np.maximum(start,spans[:,0]))
            ti=overlap/((end-start)+(spans[:,1]-spans[:,0])-overlap)
            expected={a:float(tr['q'][a][pairs[:,0],pairs[:,1]]@ti)for a in ARMS}
            spread=[np.ptp(tr['score'][tr['groups']==g])for g in range(len(tr['patterns']))if (tr['groups']==g).sum()>1]
            qs=tr['q']['BoundaryCell'];q0=tr['q']['NativeG6']
            mechanics=dict(expected_t=expected,TV=float(abs(qs-q0).sum()/2),
                           max_class_mass_error=float(max(tr['mass_errors'].values())),
                           multi_candidate_classes=len(spread),classes_with_nonconstant_new_feedback=sum(v>1e-12 for v in spread),
                           changed_interval=intervals['BoundaryCell']!=intervals['NativeG6'])
        rows.append(dict(query=key,parent=f'p{len(rows)+1:03d}',duration_group='short_le5s'if(end-start)/row['metadata']['fps']<=5 else 'long_gt5s',
                         values=values,fixed_space=fixed,mechanism=mechanics))
        put(D/'score_progress.json',dict(completed=len(rows),total=64))
    # This locked panel has exactly one query per parent; retain a private proof.
    parents=[str(truth[r['id']].get('parent',r['metadata']['source']))for r in records()];assert len(set(parents))==64
    put(D/'parent_membership.json',dict(mapping=dict(zip([r['parent']for r in rows],parents)),unique=64))
    rng=np.random.default_rng(20261022);indices=rng.integers(0,64,(10000,64))
    means={a:{k:float(np.mean([r['values'][a][k]for r in rows if r['values'][a][k]is not None]))for k in rows[0]['values'][a]}for a in arms}
    comparisons=[(a,'NativeG6')for a in ARMS if a!='NativeG6']+[('BoundaryCell','ShiftCell'),('BoundaryCell','RawCell'),('BoundaryCell','BoundaryGlobal'),('BoundaryCell','Source')]
    paired={}
    for a,b in comparisons:
        entry={}
        for metric in ['v','t','unobserved_s','boundary_MAE_seconds']:
            delta=np.array([r['values'][a][metric]-r['values'][b][metric]for r in rows]);scale=1 if metric.endswith('seconds')else 100;delta*=scale
            entry[metric]=dict(mean=float(delta.mean()),CI95=np.quantile(delta[indices].mean(1),[.025,.975]).tolist(),positive=int((delta>1e-10).sum()),negative=int((delta< -1e-10).sum()),equal=int((abs(delta)<=1e-10).sum()),below_minus5pp=int((delta< -5).sum())if scale==100 else None)
        paired[a+'-'+b]=entry
    grouped={group:{a:{m:float(np.mean([100*(r['values'][a][m]-r['values']['NativeG6'][m])for r in rows if r['duration_group']==group]))for m in ['v','t']}for a in ARMS if a!='NativeG6'}for group in ['short_le5s','long_gt5s']}
    feedback={}
    for a in ARMS:
        delta=np.array([100*(r['mechanism']['expected_t'][a]-r['mechanism']['expected_t']['NativeG6'])for r in rows]);feedback[a]=dict(expected_t_gain_pp=float(delta.mean()),CI95=np.quantile(delta[indices].mean(1),[.025,.975]).tolist())
    costs=[e['feedback_seconds']for e in seal['files'].values()]
    report=dict(status='completed_scored_development',count=64,parents=64,means=means,paired=paired,groups_vs_NativeG6=grouped,feedback=feedback,rows=rows,
                official_dense_checks=checks,max_error=maxerror,score_seconds=time.perf_counter()-begin,
                mean_feedback_seconds=float(np.mean(costs)),p95_feedback_seconds=float(np.quantile(costs,.95)),cost_scope='CPU cached feedback only; not end-to-end deployment',
                prediction_seal_sha256=sha(D/'sealed.json'),plan_sha256=sha(D/'plan.json'),full_status='paused_by_user',GT_used_for_inference=False)
    put(P/'report.json',report)
    lines=['query\tmethod\tvIoU\ttIoU\tunobserved_sIoU\tdelta_v_vs_G6_pp\tdelta_t_vs_G6_pp\tboundary_MAE_seconds']
    severe=[lines[0]]
    for r in rows:
        for a in ARMS:
            v=r['values'][a];b=r['values']['NativeG6'];line=f"{r['query']}\t{a}\t{v['v']:.12g}\t{v['t']:.12g}\t{v['unobserved_s']:.12g}\t{100*(v['v']-b['v']):.12g}\t{100*(v['t']-b['t']):.12g}\t{v['boundary_MAE_seconds']:.12g}"
            lines.append(line)
            if v['v']-b['v']<-.05 or v['t']-b['t']<-.05:severe.append(line)
    (P/'signed.tsv').write_text('\n'.join(lines)+'\n');(P/'severe_declines.tsv').write_text('\n'.join(severe)+'\n')
    put(D/'score_receipt.json',dict(report_sha256=sha(P/'report.json'),seal_sha256=sha(D/'sealed.json'),official_dense_checks=checks,max_error=maxerror,CUDA_initialized=torch.cuda.is_initialized()))
    put(D/'state.json',dict(status='completed_scored',fingerprint=p['fingerprint']))
    print(json.dumps(dict(means=means,paired=paired,groups=grouped,feedback=feedback,mean_feedback_seconds=report['mean_feedback_seconds'])))


if __name__=='__main__':main()
