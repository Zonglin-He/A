"""Independent CPU evaluation of all exposed development, no parameter selection."""
import os,sys,time,json
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from research.target_semantic_final_v1.run import *
from research.target_semantic128_shrink_v1.score import truths
from research.native_event_target_semantic_v1.score import evaluate
from research.target_semantic128_rank_audit import pairs_and_quality
import numpy as np,torch

def summary(rows,seed):
    n=len(rows);ix=np.random.default_rng(seed).integers(0,n,(10000,n))
    means={a:{m:float(np.mean([r['values'][a][m] for r in rows if r['values'][a][m] is not None])) for m in rows[0]['values'][a]} for a in ARMS}
    paired={}
    for b in [a for a in ARMS if a!=PRIMARY]:
        paired[PRIMARY+'-'+b]={}
        for metric in ['t','v','unobserved_s','boundary_MAE_seconds']:
            valid=[r for r in rows if r['values'][PRIMARY][metric] is not None and r['values'][b][metric] is not None]
            delta=np.array([r['values'][PRIMARY][metric]-r['values'][b][metric] for r in valid])*(1 if metric.endswith('seconds') else 100)
            sample=ix if len(valid)==n else np.random.default_rng(seed).integers(0,len(valid),(10000,len(valid)))
            benefit=-delta if metric.endswith('seconds') else delta
            paired[PRIMARY+'-'+b][metric]=dict(mean=float(delta.mean()),CI95=np.quantile(delta[sample].mean(1),[.025,.975]).tolist(),valid_queries=len(valid),
                improved=int((benefit>1e-10).sum()),harmed=int((benefit< -1e-10).sum()),tie=int((abs(benefit)<=1e-10).sum()))
    groups={g:dict(count=sum(r['duration_group']==g for r in rows),t_Source_pp=float(np.mean([100*(r['values'][PRIMARY]['t']-r['values']['Source']['t']) for r in rows if r['duration_group']==g]))) for g in ['short_le5s','long_gt5s']}
    severe={a:{m:sum(r['values'][a][m]<r['values']['NativeG6'][m]-.05 for r in rows) for m in ['t','v']} for a in ARMS}
    required=['UniformMatched','PermutedMatched','WrongTime','RandomSemantic','ReflectedSemantic']
    checks=dict(Source_Global6_t_and_v_positive=all(means[PRIMARY][m]>means[b][m] for b in ['NativeG6','Source'] for m in ['t','v']),
        Source_short_and_long_positive=all(v['t_Source_pp']>0 for v in groups.values()),
        more_time_wins_than_harms=paired[PRIMARY+'-NativeG6']['t']['improved']>paired[PRIMARY+'-NativeG6']['t']['harmed'],
        no_more_serious_t_or_v=all(severe[PRIMARY][m]<=severe['Loop2'][m] for m in ['t','v']),
        beyond_matched_controls=all(paired[PRIMARY+'-'+b]['t']['mean']>0 for b in required))
    ranks={}
    for mode in ['evidence','posterior']:
        for arm in rows[0]['ranks'][mode]:
            v=[r['ranks'][mode][arm] for r in rows if r['ranks'][mode][arm] is not None]
            ranks[mode+'_'+arm]=dict(valid_queries=len(v),mean=float(np.mean(v)) if v else None)
    return dict(queries=n,parents=n,means=means,paired=paired,groups=groups,severe=severe,checks=checks,passed=all(checks.values()),candidate_ranks=ranks,rows=rows)

def main():
    torch.set_num_threads(1);start=time.perf_counter();p=verify();seal=read(D/'sealed.json')
    assert seal['count']==256 and seal['GT_read'] is False and seal['plan_sha256']==sha(D/'plan.json')
    assert not(D/'score_receipt.json').exists()
    for e in seal['files'].values():
        assert sha(D/e['path'])==e['sha256'] and sha(D/e['trace_path'])==e['trace_sha256']
    put(D/'pre_GT_receipt.json',dict(seal_sha256=sha(D/'sealed.json'),all256_files_verified=True,GT_read=False))
    reports={};checks=0;maxerr=0
    for ds in ['hc2','vidstg']:
        items=[x for x in p['items'] if x['dataset']==ds];lf=p['labels'][ds]
        assert sha(PROVIDER/lf['path'])==lf['sha256'];gt=truths(PROVIDER/lf['path'],{x['row']['id'] for x in items});rows=[]
        assert len(items)==len({x['parent'] for x in items})==128
        for item in items:
            key=item['alias'];e=seal['files'][key];truth=gt[item['row']['id']];values={};fixed={}
            with np.load(D/e['path'],allow_pickle=False) as z:
                for a in ARMS:
                    values[a],err=evaluate(z[a],item['row'],truth,z[a+'_interval'].tolist(),z['positions8'],ds);checks+=1;maxerr=max(maxerr,err)
                    fixed[a],err=evaluate(z['NativeG6'],item['row'],truth,z[a+'_interval'].tolist(),z['positions8'],ds);checks+=1;maxerr=max(maxerr,err)
                tr=torch.load(D/e['trace_path'],weights_only=False,map_location='cpu')
                q0=tr['q']['NativeG6'];ij,quality,i,j,v=pairs_and_quality(q0,z['frame_ids'],truth['interval']);direction=np.sign(quality[i]-quality[j]);rank=dict(evidence={},posterior={})
                for a,L in tr['feedback']['L'].items():
                    from research.native_event_semantic_prototype_v1.core import inclusive_scores
                    S=inclusive_scores(L);scores=S[ij[:,0],ij[:,1]]
                    rank['evidence'][a]=float(np.mean((1+direction*np.sign(scores[i]-scores[j]))/2)) if len(i) else None
                for a in [PRIMARY,'NativeG6','Loop1','ShrinkK16','UniformMatched','PermutedMatched']:
                    scores=np.log(tr['q'][a][ij[:,0],ij[:,1]].clip(1e-300));rank['posterior'][a]=float(np.mean((1+direction*np.sign(scores[i]-scores[j]))/2)) if len(i) else None
            rr=tr['feedback']['prototypes'];reliability=np.concatenate([x['reliability'] for x in rr])
            cos=[]
            for s in rr:
                a=s['raw_positive']-s['raw_negative'];b=s['positive']-s['negative'];den=np.linalg.norm(a)*np.linalg.norm(b)
                if den:cos.append(float(np.dot(a,b)/den))
            rows.append(dict(query=key,parent=f'p{len(rows)+1:03d}',former_split=item['split'],duration_group='short_le5s' if (truth['interval'][1]-truth['interval'][0])/item['row']['metadata']['fps']<=5 else 'long_gt5s',
                values=values,fixed_space=fixed,ranks=rank,matched_pairs=len(i),reliability_mean=float(reliability.mean()),reliability_std=float(reliability.std()),contrast_cosine=float(np.mean(cos)) if cos else None,
                uniform_factor=tr['feedback']['uniform_factor']))
            put(D/'score_progress.json',dict(dataset=ds,completed=len(rows),total=128))
        reports[ds]=summary(rows,p['bootstrap_seed'])
    reduced=sum(v['severe'][PRIMARY]['t'] for v in reports.values())<sum(v['severe']['Loop2']['t'] for v in reports.values())
    qualified=all(v['passed'] for v in reports.values()) and reduced
    report=dict(kind='Last fixed repair on fully exposed expanded DEVELOPMENT only',primary=PRIMARY,domains=reports,checks=checks,max_metric_error=maxerr,
        development_passed=bool(qualified),time_serious_total_strictly_reduced=bool(reduced),confirmation_required_before_full=True,new_confirmation_run=False,
        no_new_full_started=True,plan_sha256=sha(D/'plan.json'),seal_sha256=sha(D/'sealed.json'),seconds=time.perf_counter()-start,
        next_action='Freeze same candidate and run32newparents/domain once' if qualified else 'STOP_LAST_REPAIR_NO_FULL_NO_MORE_TUNING')
    put(P/'report.json',report)
    header='dataset\tquery\tmethod\tvIoU\ttIoU\tdelta_v_G6_pp\tdelta_t_G6_pp\tdelta_v_Source_pp\tdelta_t_Source_pp\tboundary_MAE_seconds'
    signed=[header];bad=[header]
    for ds,d in reports.items():
        for row in d['rows']:
            b=row['values']['NativeG6'];src=row['values']['Source']
            for a,v in row['values'].items():
                line=f"{ds}\t{row['query']}\t{a}\t{v['v']:.12g}\t{v['t']:.12g}\t{100*(v['v']-b['v']):.12g}\t{100*(v['t']-b['t']):.12g}\t{100*(v['v']-src['v']):.12g}\t{100*(v['t']-src['t']):.12g}\t{v['boundary_MAE_seconds']:.12g}";signed.append(line)
                if min(v['v']-b['v'],v['t']-b['t'])<-.05: bad.append(line)
    (P/'signed.tsv').write_text('\n'.join(signed)+'\n');(P/'severe.tsv').write_text('\n'.join(bad)+'\n')
    put(D/'score_receipt.json',dict(report_sha256=sha(P/'report.json'),all256_sealed_before_GT=True,checks=checks,max_metric_error=maxerr,CUDA_initialized=torch.cuda.is_initialized()))
    put(D/'state.json',dict(status='development_passed_confirmation_required' if qualified else 'completed_last_repair_failed_STOP',fingerprint=p['fingerprint'],no_full_started=True))
    print(json.dumps({ds:{k:v for k,v in d.items() if k not in ['rows','means']} for ds,d in reports.items()},indent=2))

if __name__=='__main__':main()
