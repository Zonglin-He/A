"""Independent CPU labels only after all256 predictions are sealed."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import sys,json,re,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from research.general_region_v1.run import *
import numpy as np,torch
from stvg.metrics import DenseTube,official,box_iou
from stvg.metrics_base import xyxy
from stvg.prototype import inclusive_scores

def truths(path,keys):
    out={}
    for raw in path.read_bytes().splitlines():
        match=re.search(rb'"id"\s*:\s*"([^"\\]+)"',raw)
        if match and match[1].decode()in keys:
            row=json.loads(raw);out[row['id']]=row
    assert set(out)==keys;return out

def evaluate(boxes,row,gt,interval,positions,ds):
    truth={int(k):v for k,v in gt['boxes'].items()};rr=dict(input=row['metadata'],frame_ids=row['metadata']['frame_ids'])
    a=official(boxes,rr,truth,gt['interval'],interval,ds);b=DenseTube(boxes,rr,truth,gt['interval'],clip=(ds=='hc2')).score(interval)
    error=max(abs(a[k]-b[k])for k in a);assert error<2e-10
    ids=rr['frame_ids'];obs={ids[int(i)]for i in positions};pix=xyxy(boxes,row['metadata']['width'],row['metadata']['height']);pix=np.maximum(pix,0)if ds=='hc2'else pix
    for name,want in [('observed_s',True),('unobserved_s',False)]:
        values=[box_iou(pix[i],truth[f])for i,f in enumerate(ids)if f in truth and(f in obs)==want];a[name]=float(np.mean(values))if values else None
    fps=row['metadata']['fps'];span=gt['interval'];a.update(boundary_MAE_seconds=float((abs(interval[0]-span[0])+abs(interval[1]-span[1]))/(2*fps)),length_seconds=float((interval[1]-interval[0])/fps),zero_time_overlap=float(a['t']==0))
    return {k:float(v)if v is not None else None for k,v in a.items()},error

def candidate_ranks(trace,ids,truth):
    q0=trace['q']['NativeG6'];ij=np.stack(np.nonzero(q0>0),1);v=q0[ij[:,0],ij[:,1]];take=np.argsort(-v,kind='stable')[:64];ij=ij[take];v=v[take];ids=np.asarray(ids)
    lo,hi=ids[ij[:,0]],ids[ij[:,1]]+1;quality=np.maximum(0,np.minimum(hi,truth[1])-np.maximum(lo,truth[0]))/(np.maximum(hi,truth[1])-np.minimum(lo,truth[0]))
    i,j=np.triu_indices(len(ij),1);good=(ij[i,1]-ij[i,0]==ij[j,1]-ij[j,0])&(np.maximum(v[i],v[j])<=2*np.minimum(v[i],v[j]))&(abs(quality[i]-quality[j])>1e-12);i,j=i[good],j[good];direction=np.sign(quality[i]-quality[j])
    scores={a:inclusive_scores(L)[ij[:,0],ij[:,1]]for a,L in trace['feedback']['L'].items()}
    ranks={a:float(np.mean((1+direction*np.sign(s[i]-s[j]))/2))if len(i)else None for a,s in scores.items()}
    return ranks,len(i)

def summary(rows,seed):
    n=len(rows);rng=np.random.default_rng(seed);indices=rng.integers(0,n,(10000,n))
    means={a:{m:float(np.mean([r['values'][a][m]for r in rows if r['values'][a][m]is not None]))for m in rows[0]['values'][a]}for a in ARMS};paired={}
    for b in ARMS:
        if b==PRIMARY:continue
        paired[PRIMARY+'-'+b]={}
        for metric in ['t','v','unobserved_s','boundary_MAE_seconds']:
            selected=[r for r in rows if r['values'][PRIMARY][metric]is not None and r['values'][b][metric]is not None]
            d=np.array([r['values'][PRIMARY][metric]-r['values'][b][metric]for r in selected])*(1 if metric.endswith('seconds')else 100)
            ix=indices if len(selected)==n else rng.integers(0,len(selected),(10000,len(selected)))
            benefit=-d if metric.endswith('seconds')else d
            paired[PRIMARY+'-'+b][metric]=dict(mean=float(d.mean()),CI95=np.quantile(d[ix].mean(1),[.025,.975]).tolist(),valid_queries=len(selected),improved=int((benefit>1e-10).sum()),harmed=int((benefit< -1e-10).sum()),tie=int((abs(benefit)<=1e-10).sum()))
    groups={g:dict(count=sum(r['duration_group']==g for r in rows),t_Source_pp=float(np.mean([100*(r['values'][PRIMARY]['t']-r['values']['Source']['t'])for r in rows if r['duration_group']==g])))for g in ['short_le5s','long_gt5s']}
    severe={a:{m:sum(r['values'][a][m]<r['values']['NativeG6'][m]-.05 for r in rows)for m in ['t','v']}for a in ARMS}
    required=['FullGrid','Reflected','WrongTime','PositionOnly','K16Matched','Sharpen']
    checks=dict(Source_G6_t_v_positive=all(means[PRIMARY][m]>means[b][m]for b in ['Source','NativeG6']for m in ['t','v']),
        no_mean_t_v_regression_vs_K16=all(means[PRIMARY][m]>=means['K16'][m]for m in ['t','v']),Source_short_long_positive=all(x['count']>0 and x['t_Source_pp']>0 for x in groups.values()),
        time_wins_exceed_harms=paired[PRIMARY+'-NativeG6']['t']['improved']>paired[PRIMARY+'-NativeG6']['t']['harmed'],no_more_t_v_severe_than_K16=all(severe[PRIMARY][m]<=severe['K16'][m]for m in ['t','v']),
        time_beyond_matched_controls=all(paired[PRIMARY+'-'+b]['t']['mean']>0 for b in required))
    ranks={a:dict(valid_queries=len(v:=[r['ranks'][a]for r in rows if r['ranks'][a]is not None]),mean=float(np.mean(v))if v else None)for a in rows[0]['ranks']}
    return dict(queries=n,parents=n,means=means,paired=paired,groups=groups,severe=severe,checks=checks,passed=all(checks.values()),ranks=ranks,rows=rows)

def main():
    torch.set_num_threads(1);p=verify();s=read(D/'sealed.json');assert s['count']==256 and not s['GT_read']and s['plan_sha256']==sha(D/'plan.json');assert not(D/'score_receipt.json').exists()
    for e in s['files'].values():assert sha(D/e['path'])==e['sha256']and sha(D/e['trace_path'])==e['trace_sha256']
    put(D/'pre_GT_receipt.json',dict(all256_sealed_and_hash_verified=True,GT_read=False,seal_sha256=sha(D/'sealed.json')))
    put(D/'scorer.json',who());start=time.perf_counter();reports={};checks=0;maxerr=0
    for ds in ['hc2','vidstg']:
        items=[x for x in p['items']if x['dataset']==ds];assert len(items)==len({x['parent']for x in items})==128
        lf=p['labels'][ds];path=PROVIDER/lf['path'];assert sha(path)==lf['sha256'];gt=truths(path,{x['row']['id']for x in items});rows=[]
        for item in items:
            key=item['alias'];e=s['files'][key];truth=gt[item['row']['id']];values={};fixed={};tr=torch.load(D/e['trace_path'],weights_only=False,map_location='cpu')
            with np.load(D/e['path'],allow_pickle=False)as z:
                for a in ARMS:
                    values[a],err=evaluate(z[a],item['row'],truth,z[a+'_interval'].tolist(),z['positions8'],ds);checks+=1;maxerr=max(maxerr,err)
                    fixed[a],err=evaluate(z['NativeG6'],item['row'],truth,z[a+'_interval'].tolist(),z['positions8'],ds);checks+=1;maxerr=max(maxerr,err)
                ranks,pairs=candidate_ranks(tr,z['frame_ids'],truth['interval'])
                moves={a:(z[a+'_interval']-z['NativeG6_interval']).tolist()for a in ARMS}
            rows.append(dict(query=key,parent=f'p{len(rows)+1:03d}',former_split=item['split'],duration_group='short_le5s'if(truth['interval'][1]-truth['interval'][0])/item['row']['metadata']['fps']<=5 else'long_gt5s',values=values,fixed_space=fixed,ranks=ranks,matched_pairs=pairs,movement=moves,
                feedback_std=float(tr['feedback']['L'][PRIMARY].std()),k16_strength_factor=tr['feedback']['k16_strength_factor'],target_reflection_overlap=float(np.mean([x['target_reflected_soft_overlap']for x in tr['pooling']])),fallback_frames=sum(x['fallback']for x in tr['pooling'])))
            put(D/'score_progress.json',dict(dataset=ds,completed=len(rows),total=128))
        reports[ds]=summary(rows,p['bootstrap_seed'])
    qualified=all(x['passed']for x in reports.values());report=dict(status='completed_scored_exposed_development',primary=PRIMARY,domains=reports,checks=checks,max_metric_error=maxerr,development_qualified=qualified,confirmation_run=False,no_new_full_started=True,plan_sha256=sha(D/'plan.json'),seal_sha256=sha(D/'sealed.json'),prediction_seconds=s['seconds'],scoring_seconds=time.perf_counter()-start,new_model=0,new_expert=0,new_GPU=0)
    put(P/'report.json',report);put(P/'QUALIFICATION.json',dict(development_qualified=qualified,domain_checks={k:v['checks']for k,v in reports.items()},conditional_confirmation_required=True,confirmation_passed=False,restart_full_authorized_now=False,old_K16_remains_paused=True,no_control_promotion=True,report_sha256=sha(P/'report.json')))
    header='dataset\tquery\tmethod\tvIoU\ttIoU\tdelta_v_G6_pp\tdelta_t_G6_pp\tdelta_v_Source_pp\tdelta_t_Source_pp\tboundary_MAE_seconds';lines=[header];bad=[header]
    for ds,rr in reports.items():
        for row in rr['rows']:
            b=row['values']['NativeG6'];src=row['values']['Source']
            for a,v in row['values'].items():
                line=f"{ds}\t{row['query']}\t{a}\t{v['v']:.12g}\t{v['t']:.12g}\t{100*(v['v']-b['v']):.12g}\t{100*(v['t']-b['t']):.12g}\t{100*(v['v']-src['v']):.12g}\t{100*(v['t']-src['t']):.12g}\t{v['boundary_MAE_seconds']:.12g}";lines.append(line)
                if min(v['v']-b['v'],v['t']-b['t'])<-.05:bad.append(line)
    (P/'signed.tsv').write_text('\n'.join(lines)+'\n');(P/'severe.tsv').write_text('\n'.join(bad)+'\n')
    assert not torch.cuda.is_initialized();verify();put(D/'score_receipt.json',dict(report_sha256=sha(P/'report.json'),seal_sha256=sha(D/'sealed.json'),checks=checks,max_metric_error=maxerr,CUDA_initialized=False,**who()))
    put(D/'state.json',dict(status='completed_scored_qualified'if qualified else'completed_scored_not_qualified',fingerprint=p['fingerprint']));print(json.dumps({k:dict(checks=v['checks'],vsG6=v['paired'][PRIMARY+'-NativeG6'],vsK16=v['paired'][PRIMARY+'-K16'],groups=v['groups'])for k,v in reports.items()},indent=2),flush=True)
if __name__=='__main__':main()
