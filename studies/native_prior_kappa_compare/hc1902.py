"""Post-hoc fixed K16/K64 comparison on already sealed official HC1902 caches.

This is descriptive full-result analysis, not parameter selection or validation.
No active queue output is written; all prediction precedes independent CPU labels.
"""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
import argparse, hashlib, json, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(Path(__file__).resolve().parent),str(ROOT),str(ROOT/'tools')]
import numpy as np
import torch
from full import read,sha,atomic,identity,check,PROVIDER,RUN
from stvg.prior import from_capture
from stvg.prototype import one_step
from stvg.temporal import final_mbr
from stvg.spatial import SpatialGeometry,evidence_dp
from run import ARMS,PAIRS,summarize
D=ROOT/'outputs/native_prior_kappa_hc1902_v1'
P=ROOT/'results/native_prior_kappa_hc1902_v1'
SOURCE=RUN/'hc2'

def verify():
    p=read(D/'plan.json')
    assert p['fingerprint']==hashlib.sha256(json.dumps({k:v for k,v in p.items() if k!='fingerprint'},sort_keys=True).encode()).hexdigest()
    for f,h in p['code'].items():assert sha(ROOT/f)==h,f
    for f,h in p['inputs'].items():assert sha(f)==h,f
    assert not torch.cuda.is_initialized()
    return p

def prepare():
    assert not (D/'plan.json').exists();prod=check();s=read(SOURCE/'sealed.json')
    assert s['queries']==1902 and not s['GT_read'] and read(SOURCE/'score_receipt.json')['status']=='scored'
    job=prod['jobs']['hc2'];manifest=PROVIDER/job['manifest'];assert sha(manifest)==job['manifest_sha256']
    records=[json.loads(x) for x in manifest.read_text().splitlines()];assert {r['id'] for r in records}==set(s['files'])
    code={f:sha(ROOT/f) for f in prod['code']}
    for f in ['studies/native_prior_kappa_compare/run.py','studies/native_prior_kappa_compare/hc1902.py']:code[f]=sha(ROOT/f)
    inputs={str(f):sha(f) for f in [RUN/'plan.json',SOURCE/'sealed.json',SOURCE/'score_receipt.json',ROOT/'results/native2d_nativeprior_full_v1/hc2/report.json',manifest]}
    p=dict(schema='native_prior_kappa_hc1902_v1',authorization='Same user-authorized fair fixed K16/K64 cached comparison; official HC1902 is fully cached and sealed, avoiding reliance on the small panels alone. Active Vid full untouched.',
        arms=ARMS,pairs=PAIRS,no_winner_selection=True,kappas=[16,64],beta=1,rounds=1,context=1.5,prior='exact native final-layer two-offset physical hull for BOTH seeds and final prior',spatial_unchanged=dict(observed=.5,guard=.1,link=.25,evidence8=True,original_MBR=True),
        exposure='POST-HOC DESCRIPTIVE official HC1902 comparison requested after K64 official results were read. Not heldout confirmation; cannot validate/reselect a method against these labels. No configuration or queue switch.',
        severe='tIoU OR complete vIoU decline >5pp; union counts a query once; Source/NativeSpatial/Global6/K64 baselines explicit.',
        duration='GT duration <=5s versus >5s, evaluation only after complete1902 seal',
        budget=dict(queries=1902,new_model=0,new_video=0,new_expert=0,new_training=0,GPU=0),
        protocol='Independent CPU prepare/predict/score. Existing exact Native2D pools/nativeprior/fullgeometry/Source/eightpositions only. Recompute two kappas with same spatial DP; K64 posterior interval/finalboxes exact oldFull. Score after all1902 seal. Preserve all signed outcomes.',
        code=code,inputs=inputs,records=records,job=job,source_seal=s,production_fingerprint=prod['fingerprint'],GT_in_prediction=False,seed=20260920)
    p['fingerprint']=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest()
    atomic(D/'plan.json',p)
    for f in code:
        t=D/'source_snapshot'/f;t.parent.mkdir(parents=True,exist_ok=True);t.write_bytes((ROOT/f).read_bytes())
    atomic(P/'protocol.json',{k:v for k,v in p.items() if k not in ('records','job','source_seal','inputs')})
    atomic(D/'state.json',dict(status='prepared',fingerprint=p['fingerprint']))
    print('prepared',p['fingerprint'],flush=True)

def predict():
    p=verify();assert not (D/'sealed.json').exists();torch.set_num_threads(1)
    atomic(D/'worker.json',identity());atomic(D/'state.json',dict(status='predicting',**identity()))
    files={};start=time.perf_counter();exact=0
    for n,r in enumerate(p['records'],1):
        e=p['source_seal']['files'][r['id']]
        for pathkey,hashkey in [('path','sha256'),('capture_path','capture_sha256'),('receipt_path','receipt_sha256')]:assert sha(SOURCE/e[pathkey])==e[hashkey]
        cap=torch.load(SOURCE/e['capture_path'],map_location='cpu',weights_only=False);q0=from_capture(cap);ids=cap['frame_ids'];g=SpatialGeometry.__new__(SpatialGeometry);g.__dict__.update(cap['geometry'])
        arr={};trace=dict(native_prior=q0,GT_read=False)
        with np.load(SOURCE/e['path'],allow_pickle=False) as old:
            for a,b in [('Source','Source'),('NativeSpatial','OursS'),('Global6','Global6')]:arr[a]=old[b].copy();arr[a+'_interval']=old[b+'_interval'].copy()
            arr['positions8']=old['positions8'].copy();arr['frame_ids']=old['frame_ids'].copy()
            for k in (16,64):
                a='K'+str(k);q,stats=one_step(cap['content_features'],q0,kappa=k,beta=1.);iv=final_mbr(ids,q)['interval'];boxes=evidence_dp(g.for_interval(iv))['final'].numpy()
                arr[a]=boxes;arr[a+'_interval']=np.asarray(iv);trace[a]=q;trace[a+'_L']=stats['L']
                if k==64:
                    assert np.array_equal(boxes,old['Full']) and np.array_equal(arr[a+'_interval'],old['Full_interval']);exact+=1
        key=f'H{n:05d}';npz=D/'predictions'/(key+'.npz');npz.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(npz,**arr)
        tf=D/'traces'/(key+'.pt');tf.parent.mkdir(parents=True,exist_ok=True);torch.save(trace,tf)
        files[r['id']]=dict(alias=key,path=str(npz.relative_to(D)),sha256=sha(npz),trace=str(tf.relative_to(D)),trace_sha256=sha(tf))
        atomic(D/'progress.json',dict(completed=n,total=1902,GT_read=False))
        if n%200==0:print('predicted',n,flush=True)
    verify();assert exact==1902
    atomic(D/'sealed.json',dict(status='sealed',queries=1902,files=files,K64_interval_finalboxes_exact=exact,GT_read=False,fingerprint=p['fingerprint'],seconds=time.perf_counter()-start))
    atomic(D/'state.json',dict(status='sealed_waiting_independent_CPU_score',fingerprint=p['fingerprint']));print('sealed',flush=True)

def score():
    from full_score import metric
    p=verify();s=read(D/'sealed.json');assert s['queries']==1902 and not s['GT_read'] and s['fingerprint']==p['fingerprint'] and not (D/'score_receipt.json').exists();torch.set_num_threads(1)
    atomic(D/'score_worker.json',identity());atomic(D/'state.json',dict(status='scoring',**identity()))
    for e in s['files'].values():assert sha(D/e['path'])==e['sha256'] and sha(D/e['trace'])==e['trace_sha256']
    lab=PROVIDER/p['job']['labels'];assert sha(lab)==p['job']['labels_sha256'];truth={x['id']:x for x in map(json.loads,lab.read_text().splitlines()) if x['dataset']=='hc2'}
    original={x['id']:x for x in read(SOURCE/'metrics.json')['private_rows']};parents=sorted({str(x['parent']) for x in truth.values()});pm={v:f'P{k+1:03d}'for k,v in enumerate(parents)}
    start=time.perf_counter();rows=[];checks=0;maxerr=0.;signed=['query\tparent\tduration_group\tmethod\tbaseline\tdelta_t_pp\tdelta_v_pp'];severe=[signed[0]]
    for n,r in enumerate(p['records'],1):
        gt=truth[r['id']];span=gt['interval'];boxes={int(k):v for k,v in gt['boxes'].items()};e=s['files'][r['id']];vals={}
        with np.load(D/e['path'],allow_pickle=False)as a:
            for arm in ARMS:vals[arm],err=metric(a[arm],r,boxes,span,a[arm+'_interval'].tolist(),a['positions8']);maxerr=max(maxerr,err);checks+=1
        for a,b in [('K64','Full'),('Source','Source'),('NativeSpatial','OursS'),('Global6','Global6')]:assert vals[a]==original[r['id']]['values'][b]
        row=dict(query=e['alias'],parent=pm[str(gt['parent'])],duration_group='short_le5s' if (span[1]-span[0])/r['metadata']['fps']<=5 else 'long_gt5s',values=vals);rows.append(row)
        for a,b in PAIRS:
            dt=100*(vals[a]['t']-vals[b]['t']);dv=100*(vals[a]['v']-vals[b]['v']);line=f"{e['alias']}\t{row['parent']}\t{row['duration_group']}\t{a}\t{b}\t{dt:.12g}\t{dv:.12g}";signed.append(line)
            if min(dt,dv)<-5:severe.append(line)
        if n%200==0:atomic(D/'score_progress.json',dict(completed=n,total=1902,CPU_only=True))
    groups=dict(all1902=summarize(rows));groups['all1902']['duration']={g:summarize([r for r in rows if r['duration_group']==g]) for g in ('short_le5s','long_gt5s')}
    assert checks==9510 and maxerr<2e-10 and len(rows)==1902 and len(parents)==227 and not torch.cuda.is_initialized()
    report=dict(status='complete_scored',dataset='hc2',groups=groups,rows=rows,checks=checks,max_metric_error=maxerr,seconds=time.perf_counter()-start,exposure=p['exposure'],plan_sha256=sha(D/'plan.json'),seal_sha256=sha(D/'sealed.json'),no_selection=True,production_unchanged=True)
    atomic(P/'report.json',report);(P/'signed.tsv').write_text('\n'.join(signed)+'\n');(P/'severe.tsv').write_text('\n'.join(severe)+'\n')
    verify();atomic(D/'score_receipt.json',dict(status='scored',report_sha256=sha(P/'report.json'),seal_sha256=sha(D/'sealed.json'),**identity()));atomic(D/'state.json',dict(status='complete_scored',fingerprint=p['fingerprint'],production_changed=False))
    g=groups['all1902'];print(json.dumps(dict(means={a:{f:100*g['means'][a][f]for f in ('t','v')}for a in ARMS},paired=g['paired']['K16-K64'],severe=g['severe'])),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=('prepare','predict','score'));args=ap.parse_args();globals()[args.command]()
