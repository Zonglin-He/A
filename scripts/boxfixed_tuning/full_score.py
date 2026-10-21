"""Sealed full official evaluation, query-weighted means and parent-cluster CI."""
import argparse,json,os,sys,time,math
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.boxfixed_tuning.full import RUN,validate_plan
from scripts.tube_temporal.run import read,sha,atomic
from scripts.interval_pool.score import score_boxes,summarize

def cluster(rows,a,b):
    parents=sorted({r['parent']for r in rows});lookup={p:i for i,p in enumerate(parents)};n=len(parents)
    weights=np.random.default_rng(20261022).multinomial(n,[1/n]*n,size=10000)
    result={}
    for field,scale in [('v',100.),('t',100.),('unobserved_actual_sIoU',100.),('observed_actual_sIoU',100.),('boundary_MAE_seconds',1.)]:
        sums=np.zeros(n);counts=np.zeros(n);d=[]
        for r in rows:
            x,y=r['values'][a][field],r['values'][b][field]
            if x is None or y is None:continue
            value=scale*(x-y);d.append(value);j=lookup[r['parent']];sums[j]+=value;counts[j]+=1
        denominator=weights@counts;boot=np.divide(weights@sums,denominator,out=np.zeros(len(weights)),where=denominator>0)
        d=np.asarray(d);result[field]=dict(mean=float(d.mean()),CI95=np.quantile(boot[denominator>0],[.025,.975]).tolist(),
            unit='seconds'if scale==1 else'pp',queries=len(d),parents=n,increased=int((d>1e-10).sum()),decreased=int((d < -1e-10).sum()),unchanged=int((abs(d)<=1e-10).sum()),
            declined_over5pp=int((d < -5).sum())if scale==100 else None)
    return result

def run(ds):
    torch.set_num_threads(1);plan=validate_plan();dest=RUN/ds;seal=read(dest/'sealed.json');job=plan['jobs'][ds]
    assert seal['queries']==job['queries']and not seal['GT_read']and seal['fingerprint']==plan['fingerprint']
    if(dest/'score_receipt.json').exists():
        receipt=read(dest/'score_receipt.json');assert receipt['seal_sha256']==sha(dest/'sealed.json')
        for path,h in receipt['public_files'].items():assert sha(ROOT/path)==h
        print('Already scored and verified; no rescore');return
    for item in seal['files'].values():
        assert sha(dest/item['path'])==item['sha256']and sha(dest/item['receipt_path'])==item['receipt_sha256']
    label=ROOT/job['labels'];assert sha(label)==job['labels_sha256']
    truth={g['id']:g for g in [json.loads(s)for s in label.read_text().splitlines()]if g['dataset']==ds}
    records=[json.loads(s)for s in (ROOT/job['manifest']).read_text().splitlines()];assert set(seal['files'])=={r['id']for r in records}
    rows=[];errors=0.;begin=time.perf_counter();costs=[];dense_checks=0;aliases=0
    for i,rec in enumerate(records):
        key=rec['id'];g=truth[key];gt={int(k):v for k,v in g['boxes'].items()};values={};item=seal['files'][key]
        with np.load(dest/item['path'],allow_pickle=False)as p:
            for method in ('Source','Untuned','Final'):
                if method=='Final' and np.array_equal(p['Final'],p['Untuned']) and np.array_equal(p['Final_interval'],p['Untuned_interval']):
                    values[method]=dict(values['Untuned']);aliases+=1;continue
                values[method],e=score_boxes(p[method],rec,gt,p[method+'_interval'].tolist(),g['interval'],p['positions8'].tolist());errors=max(errors,e);dense_checks+=1
        rows.append(dict(parent=g['parent'],values=values));costs.append(read(dest/item['receipt_path']))
        if(i+1)%100==0:atomic(dest/'score_progress.json',dict(completed=i+1,total=len(records),CPU_only=True))
    groups={}
    for name,rr in [('all',costs),('complete_capture_hits',[r for r in costs if r['reused']['complete_capture_hit']]),('fresh_captures',[r for r in costs if not r['reused']['complete_capture_hit']])]:
        groups[name]=dict(queries=len(rr),mean_query_wall_seconds=float(np.mean([r['query_wall_seconds']for r in rr]))if rr else None,
            median_query_wall_seconds=float(np.median([r['query_wall_seconds']for r in rr]))if rr else None)
    hits=sum(r['visual_hits']for r in costs);misses=sum(r['visual_misses']for r in costs)
    efficiency=dict(groups=groups,visual_hits=hits,visual_misses=misses,visual_hit_fraction=hits/(hits+misses)if hits+misses else None,
        RGB_hits=sum(r.get('RGB',{}).get('hit',False)for r in costs),
        expert_forwards=sum(r['capture_cost'].get('ReferDINO_query_forwards',0)for r in costs),
        Source224_captures=len([r for r in costs if not r['reused']['complete_capture_hit']]),
        Source448_calls=0,extra_decoder_calls=0,updates=0,
        Source_peak_allocated_GPU_bytes=max([r['Source_process_peak_allocated_GPU_bytes']for r in costs if r.get('Source_process_peak_allocated_GPU_bytes')is not None],default=None),
        expert_peak_allocated_GPU_bytes=max([r['expert_process_peak_allocated_GPU_bytes']for r in costs if r.get('expert_process_peak_allocated_GPU_bytes')is not None],default=None),
        scope='Current work only, cache-hit original computation excluded. Query wall executes BOTH Untuned and Final, not a standalone method timing. Source/expert allocated peaks are separate processes, not whole-device NVML peak.')
    pairs={a+'-'+b:cluster(rows,a,b)for a,b in [('Final','Untuned'),('Final','Source'),('Untuned','Source')]}
    public=ROOT/'results/boxfixed_full_v1'/ds;public.mkdir(parents=True,exist_ok=True)
    report=dict(status='completed_full_official_evaluation',dataset=ds,queries=len(rows),parents=len({r['parent']for r in rows}),
        profile=plan['profile']['datasets'][ds],means=summarize(rows,['Source','Untuned','Final'],'values'),paired=pairs,efficiency=efficiency,
        exposure='Official complete split includes historically exposed development/tuning/recheck examples. Not a pristine zero-shot test. Profiles frozen before full results; no full-result parameter selection.',
        proof=dict(all_predictions_sealed_before_GT=True,seal_sha256=sha(dest/'sealed.json'),plan_sha256=sha(RUN/'plan.json'),
            labels_sha256=sha(label),official_dense_checks=dense_checks,identical_Final_Untuned_aliases=aliases,official_dense_max_error=errors,CPU_only=True))
    files={};atomic(public/'report.json',report);files[str((public/'report.json').relative_to(ROOT))]=sha(public/'report.json')
    columns=['query','parent']+[m+'_'+f for m in ('Source','Untuned','Final')for f in ('v','t','observed_actual_sIoU','unobserved_actual_sIoU','boundary_MAE_seconds','tIoU_ge_05')]+['Final_minus_Source_v_pp','Final_minus_Untuned_v_pp','Final_minus_Source_t_pp','Final_minus_Untuned_t_pp','severe_Final_vs_Source','severe_Final_vs_Untuned']
    pmap={p:i+1 for i,p in enumerate(sorted({r['parent']for r in rows}))}
    for offset in range(0,len(rows),1000):
        lines=['\t'.join(columns)]
        for i,r in enumerate(rows[offset:offset+1000],offset+1):
            v=r['values'];dv=100*(v['Final']['v']-v['Source']['v']);db=100*(v['Final']['v']-v['Untuned']['v'])
            vals=[f'{ds[0].upper()}{i:05d}',f'P{pmap[r["parent"]]:03d}']+[v[m][f]for m in ('Source','Untuned','Final')for f in ('v','t','observed_actual_sIoU','unobserved_actual_sIoU','boundary_MAE_seconds','tIoU_ge_05')]+[dv,db,100*(v['Final']['t']-v['Source']['t']),100*(v['Final']['t']-v['Untuned']['t']),dv < -5,db < -5]
            lines.append('\t'.join(''if x is None else format(x,'.17g')if isinstance(x,float)else str(x)for x in vals))
        path=public/f'signed_{offset//1000+1:02d}.tsv';path.write_text('\n'.join(lines)+'\n');files[str(path.relative_to(ROOT))]=sha(path)
    atomic(dest/'metrics.json',dict(report,private_rows=rows));atomic(dest/'score_receipt.json',dict(status='scored',seal_sha256=sha(dest/'sealed.json'),public_files=files,seconds=time.perf_counter()-begin))
    print(json.dumps(dict(dataset=ds,queries=len(rows),means=report['means'],paired=pairs),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['hc2','vidstg'],required=True);a=p.parse_args();run(a.dataset)
