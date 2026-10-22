"""CPU-only scoring; opens fixed labels only after all experimental outputs seal."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import json,sys,time
from pathlib import Path
import numpy as np,torch
sys.path.insert(0,str(Path(__file__).parent));from run import R,D,P,read,put,sha,records,verify
from stvg.metrics import DenseTube,official,box_iou
from stvg.metrics_base import xyxy
from stvg.evaluation import finite_mean
def score(boxes,row,gt,interval,pos):
    truth={int(k):v for k,v in gt['boxes'].items()};rr=dict(input=row['metadata'],frame_ids=row['metadata']['frame_ids']);span=gt['interval'];a=official(boxes,rr,truth,span,interval,'hc2');b=DenseTube(boxes,rr,truth,span,clip=True).score(interval);err=max(abs(a[k]-b[k])for k in a);assert err<2e-10
    ids=rr['frame_ids'];lookup={f:i for i,f in enumerate(ids)};obs={ids[int(i)]for i in pos};pix=np.maximum(xyxy(boxes,row['metadata']['width'],row['metadata']['height']),0)
    for name,group in [('observed_s',[f for f in ids if f in truth and f in obs]),('unobserved_s',[f for f in ids if f in truth and f not in obs])]:a[name]=float(np.mean([box_iou(pix[lookup[f]],truth[f])for f in group]))if group else None
    fps=row['metadata']['fps'];a.update(boundary_MAE_seconds=(abs(interval[0]-span[0])+abs(interval[1]-span[1]))/(2*fps),length_seconds=(interval[1]-interval[0])/fps,zero_time_overlap=float(a['t']==0))
    return a,err
def run():
    torch.set_num_threads(1);plan=verify();seal=read(D/'sealed.json');assert seal['count']==64 and seal['GT_read']is False and seal['plan_sha256']==sha(D/'plan.json')
    for item in seal['files'].values():assert sha(D/item['path'])==item['sha256']and sha(D/item['trace_path'])==item['trace_sha256']
    assert not(D/'score_receipt.json').exists();assert not torch.cuda.is_initialized();put(D/'pre_GT_validation.json',dict(seal_sha256=sha(D/'sealed.json'),all64_outputs_hash_verified=True,new_predictions_after_GT=False,CUDA_initialized=False))
    labels=R/'data/evaluation/development_labels.jsonl';assert sha(labels)==plan['labels_sha256'];truth={g['id']:g for g in map(json.loads,labels.read_text().splitlines())if g['dataset']=='hc2'};rows=[];maxerr=0.;checks=0;begin=time.perf_counter()
    for row in records():
        key=row['id'];item=seal['files'][key];gt=truth[key];vals={};fixed={}
        with np.load(D/item['path'],allow_pickle=False)as z:
            arms=plan['arms']+['Source'];intervals={a:z[a+'_interval'].tolist()for a in arms};pos=z['positions8'];base=z['NativeG6']
            for a in arms:
                vals[a],e=score(z[a],row,gt,intervals[a],pos);maxerr=max(maxerr,e);checks+=1
                fixed[a],e=score(base,row,gt,intervals[a],pos);maxerr=max(maxerr,e);checks+=1
            trace=torch.load(D/item['trace_path'],map_location='cpu',weights_only=False);pairs=trace['feedback']['pairs'];rr=dict(input=row['metadata'],frame_ids=row['metadata']['frame_ids']);d=DenseTube(base,rr,{int(k):v for k,v in gt['boxes'].items()},gt['interval'],clip=True);candidate_v=np.array([d.score([rr['frame_ids'][a],rr['frame_ids'][b]+1])['v']for a,b in pairs]);quality={}
            for kind in ['Raw','Predictive']:
                q=trace['q'][kind+'Direct'];quality[kind]=dict(teacher_expected_fixed_v=float(q[pairs[:,0],pairs[:,1]]@candidate_v),query_expected_fixed_v=float(trace['q'][kind+'Query'][pairs[:,0],pairs[:,1]]@candidate_v),native_expected_fixed_v=float(trace['q']['NativeG6'][pairs[:,0],pairs[:,1]]@candidate_v),loss_first=trace['paths'][kind]['path'][0]['loss'],loss_last=trace['paths'][kind]['path'][-1]['loss'],normalized_delta_RMS=trace['paths'][kind]['path'][-1]['normalized_delta_RMS'],direct_query_same_interval=intervals[kind+'Direct']==intervals[kind+'Query'])
        rows.append(dict(query=key,parent=str(gt.get('parent',row['metadata']['source'])),values=vals,fixed_space=fixed,quality=quality,feedback=trace['feedback']['metadata']))
        put(D/'score_progress.json',dict(completed=len(rows),total=64))
    means={a:{k:finite_mean([r['values'][a][k]for r in rows])for k in rows[0]['values'][a]}for a in arms};fixedmeans={a:{k:finite_mean([r['fixed_space'][a][k]for r in rows])for k in rows[0]['fixed_space'][a]}for a in arms};paired={};parent=sorted({r['parent']for r in rows});index={p:i for i,p in enumerate(parent)};w=np.random.default_rng(20261022).multinomial(len(parent),[1/len(parent)]*len(parent),size=10000)
    comparisons=[(a,'NativeG6')for a in plan['arms']if a!='NativeG6']+[(a,'Source')for a in ['NativeG6','RawQuery','PredictiveQuery']]+[('RawQuery','RawDirect'),('PredictiveQuery','PredictiveDirect'),('PredictiveDirect','RawDirect'),('PredictiveQuery','RawQuery')]
    for a,b in comparisons:
        result={}
        for k in ['v','t','unobserved_s','boundary_MAE_seconds']:
            sums=np.zeros(len(parent));counts=np.zeros(len(parent));ds=[];scale=1 if k.endswith('seconds')else 100
            for r in rows:
                x,y=r['values'][a][k],r['values'][b][k]
                if x is None or y is None:continue
                delta=scale*(x-y);j=index[r['parent']];sums[j]+=delta;counts[j]+=1;ds.append(delta)
            den=w@counts;boot=(w@sums)[den>0]/den[den>0];d=np.array(ds);result[k]=dict(mean=float(d.mean()),CI95=np.quantile(boot,[.025,.975]).tolist(),positive=int((d>1e-10).sum()),negative=int((d< -1e-10).sum()),equal=int((abs(d)<=1e-10).sum()),below_minus5pp=int((d< -5).sum())if scale==100 else None)
        paired[a+'-'+b]=result
    mechanism={kind:dict(loss_decreased=sum(r['quality'][kind]['loss_last']<r['quality'][kind]['loss_first']-1e-10 for r in rows),same_direct_query_interval=sum(r['quality'][kind]['direct_query_same_interval']for r in rows),query_worse_than_direct_t=sum(r['values'][kind+'Query']['t']<r['values'][kind+'Direct']['t']-1e-10 for r in rows),query_better_than_direct_t=sum(r['values'][kind+'Query']['t']>r['values'][kind+'Direct']['t']+1e-10 for r in rows),teacher_expected_v_delta_pp=100*float(np.mean([r['quality'][kind]['teacher_expected_fixed_v']-r['quality'][kind]['native_expected_fixed_v']for r in rows])),query_expected_v_delta_pp=100*float(np.mean([r['quality'][kind]['query_expected_fixed_v']-r['quality'][kind]['native_expected_fixed_v']for r in rows])))for kind in ['Raw','Predictive']}
    result=dict(status='completed_scored_development',queries=64,parents=len(parent),means=means,fixed_space_means=fixedmeans,paired=paired,mechanism=mechanism,rows=rows,official_dense_checks=checks,official_dense_max_error=maxerr,plan_sha256=sha(D/'plan.json'),prediction_seal_sha256=sha(D/'sealed.json'),new_model_selection_from_GT=False,network_weights_frozen=True,full_status='paused_by_user',score_seconds=time.perf_counter()-begin)
    put(P/'report.json',result);put(P/'historical_failure_analysis.json',read(D/'historical_failure_analysis.json'));lines=['# HC64: old event feedback and learned temporal query','', 'Exposed development; one locked configuration, no GT-selected state. Spatial geometry and DP unchanged. Source/Full native controls retained. Raw/Predictive Direct and Query share exactly the same target.','', '| Method | vIoU % | tIoU % | Unobserved sIoU % | Boundary MAE s |','|---|---:|---:|---:|---:|']
    for a in arms:
        m=means[a];lines.append(f"| {a} | {100*m['v']:.4f} | {100*m['t']:.4f} | {100*m['unobserved_s']:.4f} | {m['boundary_MAE_seconds']:.4f} |")
    lines+=['','Full signed outcomes and parent-cluster intervals are in report.json. Historical token extraction/query reconstruction is not a new contribution. LOO predictive gain is a working score, not calibrated event probability; teacher observations share context. No automatic full resume.']
    (P/'REPORT.md').write_text('\n'.join(lines)+'\n');put(D/'score_receipt.json',dict(status='passed',seal_sha256=sha(D/'sealed.json'),labels_sha256=sha(labels),report_sha256=sha(P/'report.json'),official_dense_max_error=maxerr));put(D/'state.json',dict(status='completed_scored',fingerprint=plan['fingerprint'],full_status='paused_by_user'));print(json.dumps(dict(means=means,mechanism=mechanism,seconds=time.perf_counter()-begin)))
if __name__=='__main__':run()
