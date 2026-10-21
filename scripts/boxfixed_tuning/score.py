"""Independent CPU score AFTER each complete sealed trial. Development-only selection."""
import argparse,json,os,sys
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.boxfixed_tuning.capture import OUT
from scripts.boxfixed_tuning.trial import inputs
from scripts.interval_pool.score import score_boxes,summarize,paired
from scripts.tube_temporal.run import read,sha,atomic

def run(ds,name):
    torch.set_num_threads(1);dest=OUT/'trials'/ds/name;lock=read(dest/'lock.json');seal=read(dest/'sealed.json');assert seal['queries']==64 and not seal['GT_read']
    assert sha(dest/'lock.json')==seal['lock_sha256']
    assert not(dest/'metrics.json').exists(),'Do not overwrite scores'
    records,_=inputs(ds);assert set(seal['files'])=={r['id']for r in records}
    for item in seal['files'].values():assert sha(dest/item['path'])==item['sha256']
    label=ROOT/'data/evaluation/development_labels.jsonl';assert sha(label)==read(OUT/'plan.json')['labels_sha256']
    truth={g['id']:g for g in [json.loads(s)for s in label.read_text().splitlines()]if g['dataset']==ds}
    rows=[];maxerr=0.;stats=[]
    for rec in records:
        key=rec['id'];g=truth[key];gt={int(k):v for k,v in g['boxes'].items()};v={}
        with np.load(dest/seal['files'][key]['path'],allow_pickle=False)as p,np.load(OUT/'trials'/ds/'base'/'predictions'/f'{key}.npz',allow_pickle=False)as b:
            for method,boxes,interval in [('Candidate',p['Candidate'],p['Candidate_interval']),('Baseline',b['Candidate'],b['Candidate_interval']),('Source',p['Source'],p['Source_interval'])]:
                v[method],e=score_boxes(boxes,rec,gt,interval.tolist(),g['interval'],p['positions8'].tolist());maxerr=max(maxerr,e)
        rows.append(dict(id=key,parent=g['parent'],values=v));stats.append(seal['files'][key]['statistics'])
    pairs=paired(rows,[('Candidate','Baseline'),('Candidate','Source'),('Baseline','Source')],'values')
    base_severe=pairs['Baseline-Source']['v']['declined_over5pp']
    delta=pairs['Candidate-Baseline'];ss=delta['unobserved_actual_sIoU']['mean'];dt=delta['t']['mean'];dv=delta['v']['mean']
    eligible=dt>=-.1-1e-9 and ss>=-.1-1e-9 and pairs['Candidate-Source']['v']['declined_over5pp']<=base_severe+2
    public=dict(dataset=ds,name=name,params=lock['params'],queries=64,parents=64,status='completed_development_tuning',
        proof=dict(predictions_sealed_before_GT=True,seal_sha256=sha(dest/'sealed.json'),official_dense_checks=192,official_dense_max_error=maxerr,GT_selection_only_development=True),
        means=summarize(rows,['Candidate','Baseline','Source'],'values'),paired=pairs,eligible=eligible,
        process_means={k:float(np.mean([s[k]for s in stats if k in s]))for k in {k for s in stats for k in s}},
        seconds=seal['seconds'],anonymous_queries=[dict(query=f'{ds[0].upper()}{i+1:03d}',values=r['values'])for i,r in enumerate(rows)])
    atomic(dest/'metrics.json',dict(public,private_rows=rows));atomic(ROOT/'results/boxfixed_tuning_v1'/ds/(name+'.json'),public)
    print(json.dumps(dict(dataset=ds,name=name,params=lock['params'],eligible=eligible,v_delta_pp=dv,t_delta_pp=dt,unobserved_s_delta_pp=ss,
        severe_vs_Source=pairs['Candidate-Source']['v']['declined_over5pp'],replacements=public['process_means'],candidate_means=public['means']['Candidate']),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True,choices=['hc2','vidstg']);p.add_argument('--name',required=True);a=p.parse_args();run(a.dataset,a.name)
