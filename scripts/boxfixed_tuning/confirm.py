"""One frozen development recheck. No profile search on these32+32 parents."""
import argparse,json,sys,os,time
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.boxfixed_tuning.capture import OUT
from scripts.boxfixed_tuning.core import BASE,temporal,spatial
from scripts.tube_temporal.run import read,sha,atomic
from scripts.interval_pool import confirmation as old
from scripts.interval_pool.score import score_boxes,summarize,paired

def predict():
    torch.set_num_threads(1);choice=read(OUT/'selected.json');dest=OUT/'confirmation'
    assert not(dest/'sealed.json').exists();selection=read(old.BASE/'confirmation_selection.json')['development_recheck_selection'];entries={}
    for ds in ('hc2','vidstg'):
        old_receipts=old.validate(ds);tr=old.BASE/('recheck_'+ds);entries[ds]={}
        for r in selection[ds]:
            key=r['query_id'];rr=old_receipts[key]
            cap=torch.load(tr/rr['capture_path'],map_location='cpu',weights_only=False)
            post=torch.load(tr/rr['posterior_path'],map_location='cpu',weights_only=False)
            cap.update(positions8=cap['planned_positions'],logits=post['logits'],fps=float(r['record']['metadata']['fps']))
            tem=temporal(cap,choice[ds]['params']['temperature']);out=spatial(cap,tem['interval'],choice[ds]['params'])
            with np.load(tr/'predictions'/f'{key}.npz',allow_pickle=False)as z:
                baseline=spatial(cap,z['Hull_MBR_interval'].tolist(),BASE)
                assert np.array_equal(baseline['final'].numpy(),z['Hull_MBR'])
                values=dict(Candidate=out['final'].numpy(),Candidate_interval=tem['interval'],Baseline=z['Hull_MBR'],Baseline_interval=z['Hull_MBR_interval'],
                    Source=z['Source_Native224'],Source_interval=z['Source_Native224_interval'],positions8=z['positions8'])
            path=dest/ds/f'{key}.npz';path.parent.mkdir(parents=True,exist_ok=True);np.savez(path,**values)
            entries[ds][key]=dict(path=str(path.relative_to(dest)),sha256=sha(path))
    atomic(dest/'sealed.json',dict(files=entries,GT_read=False,selected_sha256=sha(OUT/'selected.json'),queries=64))
    print('Both frozen32-parent outputs sealed; no new networks or GT')

def score():
    dest=OUT/'confirmation';seal=read(dest/'sealed.json');assert not seal['GT_read'] and seal['queries']==64
    assert seal['selected_sha256']==sha(OUT/'selected.json')and not(dest/'decision.json').exists()
    for files in seal['files'].values():
        for f in files.values():assert sha(dest/f['path'])==f['sha256']
    choice=read(OUT/'selected.json');selected=read(old.BASE/'confirmation_selection.json')['development_recheck_selection'];decision={}
    formal=read(ROOT/'outputs/formal/plan.json')
    for ds in ('hc2','vidstg'):
        job=next(j for j in formal['jobs']if j['id']=='tastvg-'+ds);label=ROOT/job['labels'];assert sha(label)==job['labels_sha256']
        truth={g['id']:g for g in [json.loads(s)for s in label.read_text().splitlines()]if g['dataset']==ds};rows=[];error=0.
        for item in selected[ds]:
            key=item['query_id'];rec=item['record'];g=truth[key];gt={int(k):v for k,v in g['boxes'].items()};values={}
            with np.load(dest/seal['files'][ds][key]['path'],allow_pickle=False)as p:
                for method in ('Candidate','Baseline','Source'):
                    values[method],e=score_boxes(p[method],rec,gt,p[method+'_interval'].tolist(),g['interval'],p['positions8'].tolist());error=max(error,e)
            rows.append(dict(parent=g['parent'],values=values))
        pairs=paired(rows,[('Candidate','Baseline'),('Candidate','Source'),('Baseline','Source')],'values');d=pairs['Candidate-Baseline']
        gates=dict(v_not_decreased=d['v']['mean']>=-1e-9,t_loss_at_most_point1pp=d['t']['mean']>=-.1-1e-9,
            unobserved_s_loss_at_most_point1pp=d['unobserved_actual_sIoU']['mean']>=-.1-1e-9,
            severe_increase_at_most2=pairs['Candidate-Source']['v']['declined_over5pp']<=pairs['Baseline-Source']['v']['declined_over5pp']+2)
        accepted=all(gates.values());decision[ds]=dict(accepted=accepted,gates=gates,
            proposed=choice[ds]['params'],final_params=choice[ds]['params']if accepted else dict(BASE))
        report=dict(dataset=ds,queries=32,parents=32,exposure='Historically exposed parent/video-separated DEVELOPMENT recheck, not independent test',
            means=summarize(rows,['Candidate','Baseline','Source'],'values'),paired=pairs,decision=decision[ds],
            proof=dict(all64_predictions_sealed_before_GT=True,seal_sha256=sha(dest/'sealed.json'),dense_checks=96,max_error=error),
            anonymous_queries=[dict(query=f'{ds[0].upper()}{i+1:03d}',values=r['values'])for i,r in enumerate(rows)])
        atomic(ROOT/f'results/boxfixed_tuning_v1/confirmation_{ds}.json',report)
    atomic(dest/'decision.json',decision)
    profile=dict(method='BoxFixed complete native propagation + tuned Global6 old hull + original physical MBR',
        no_network_training=True,source_resolution=224,expert_count=1,observations=8,
        production_reference='Strong unchanged; this is an explicit new full-evaluation profile',
        selected_sha256=sha(OUT/'selected.json'),confirmation_seal_sha256=sha(dest/'sealed.json'),
        datasets={ds:d['final_params']for ds,d in decision.items()},selection_decisions=decision)
    assert not(ROOT/'configs/tuned_boxfixed_v1.json').exists();atomic(ROOT/'configs/tuned_boxfixed_v1.json',profile)
    print(json.dumps(decision,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['predict','score']);a=p.parse_args();predict()if a.action=='predict'else score()
