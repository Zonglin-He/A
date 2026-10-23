"""Two fixed shrinkages on sealed native-prior caches; CPU only, no selection.

Prediction never opens labels. A separate invocation scores all sealed outputs.
The active full runner, its configuration, its captures and its lock are read-only.
"""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
import argparse, hashlib, json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'tools')]
import numpy as np
import torch
from full import read, sha, atomic, identity, check, PROVIDER, RUN
from stvg.prior import from_capture
from stvg.prototype import one_step
from stvg.temporal import final_mbr
from stvg.spatial import SpatialGeometry, evidence_dp

D = ROOT/'outputs/native_prior_kappa_compare_v1'
P = ROOT/'results/native_prior_kappa_compare_v1'
PARENT = ROOT.parent/'stvg-native2d/outputs/native_prior_v1'
ARMS = ('Source', 'NativeSpatial', 'Global6', 'K16', 'K64')
PAIRS = [(a,b) for a in ('K16','K64') for b in ('Source','NativeSpatial','Global6')] + [('K16','K64')]

def verify():
    p=read(D/'plan.json')
    assert p['fingerprint']==hashlib.sha256(json.dumps({k:v for k,v in p.items() if k!='fingerprint'},sort_keys=True).encode()).hexdigest()
    for f,h in p['code'].items(): assert sha(ROOT/f)==h, f
    for f,h in p['inputs'].items(): assert sha(f)==h, f
    assert not torch.cuda.is_initialized()
    return p

def prepare():
    assert not (D/'plan.json').exists()
    prod=check(); parent=read(PARENT/'plan.json'); seal=read(PARENT/'sealed.json')
    assert seal['queries']==256 and read(PARENT/'state.json')['status']=='complete_scored'
    items=parent['items']; assert len(items)==256
    code={f:sha(ROOT/f) for f in prod['code']}
    code['studies/native_prior_kappa_compare/run.py']=sha(__file__)
    inputs={str(f):sha(f) for f in (PARENT/'plan.json',PARENT/'sealed.json',PARENT/'score_receipt.json',ROOT/'results/native_prior_v1/report.json',RUN/'plan.json')}
    for i in items:
        for pathkey,hashkey in [('path','sha256'),('feature_path','feature_sha256')]:
            assert sha(i[pathkey])==i[hashkey]
            inputs[i[pathkey]]=i[hashkey]
    p=dict(schema='native_prior_kappa_compare_v1',authorization='User requests fair cached kappa16/kappa64 accuracy, severe-harm and short/long comparison without interrupting the current full.',
        arms=ARMS,pairs=PAIRS,primary_comparison='K16-K64',no_winner_selection=True,
        fixed=dict(prior='native final-layer original two-offset physical hull, for BOTH soft seeds and update prior',kappas=[16,64],beta=1,rounds=1,context=1.5,spatial_observed=.5,spatial_guard=.1,spatial_link=.25,original_MBR=True,evidence8=True),
        exposure='Existing exposed dev96+confirm32 per direction, pooled128 descriptive only. No new confirmation. Official HC1902 was already read before this comparison; no full-result tuning or production promotion.',
        severe='tIoU OR complete vIoU falls >5pp; union counts a query once. Separate metric counts and baselines reported.',
        duration='GT event duration <=5s versus >5s; labels used ONLY after all256 predictions seal, never for inference or parameter selection.',
        budget=dict(queries=256,per_direction=128,new_model=0,new_RGB=0,new_expert=0,new_training=0,GPU=0),
        protocol='Independent CPU invocations: prepare, predict, score. K64 posterior/time/boxes and Source/NativeSpatial/Global6 exactly reproduce sealed parent. Only kappa changes. Same geometry/eight observations and spatial DP executed at each resulting time interval.',
        production_fingerprint=prod['fingerprint'],production_plan_sha256=sha(RUN/'plan.json'),code=code,inputs=inputs,items=items,jobs=parent['jobs'],seed=20260920,GT_in_prediction=False)
    p['fingerprint']=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest()
    for f in code:
        dest=D/'source_snapshot'/f; dest.parent.mkdir(parents=True,exist_ok=True); dest.write_bytes((ROOT/f).read_bytes())
    atomic(D/'plan.json',p)
    q=np.triu(np.ones((7,7)),1); q/=q.sum()
    for k in (16,64): assert np.array_equal(one_step(np.ones((7,4)),q,kappa=k,beta=1)[0],q)
    atomic(D/'preflight.json',dict(constant_representation_exact_native_noop=True,bound_production_unchanged=True,GT_read=False,models=0))
    atomic(P/'protocol.json',{k:v for k,v in p.items() if k not in ('items','jobs','inputs')})
    atomic(D/'state.json',dict(status='prepared',fingerprint=p['fingerprint']))
    print('prepared',p['fingerprint'],flush=True)

def predict():
    p=verify(); assert not (D/'sealed.json').exists(); torch.set_num_threads(1)
    atomic(D/'worker.json',identity()); atomic(D/'state.json',dict(status='predicting',**identity()))
    parent=read(PARENT/'sealed.json'); files={}; start=time.perf_counter(); exact=0
    for n,i in enumerate(p['items'],1):
        cap=torch.load(i['path'],weights_only=False,map_location='cpu')
        tr=torch.load(i['feature_path'],weights_only=False,map_location='cpu')
        z=tr['features']['Native2D'] if isinstance(tr['features'],dict) else tr['features']
        key=i['dataset']+'_'+i['alias']; e=parent['files'][key]
        assert sha(PARENT/e['path'])==e['sha256'] and sha(PARENT/e['trace'])==e['trace_sha256']
        ref=torch.load(PARENT/e['trace'],weights_only=False,map_location='cpu')
        q0=from_capture(cap); assert np.array_equal(q0,ref['native_prior'])
        ids=list(map(int,cap['frame_ids'])); g=SpatialGeometry.__new__(SpatialGeometry); g.__dict__.update(cap['geometry'])
        with np.load(PARENT/e['path'],allow_pickle=False) as old:
            arr={a:old[a].copy() for a in ('Source','NativeSpatial','Global6')}
            arr.update({a+'_interval':old[a+'_interval'].copy() for a in ('Source','NativeSpatial','Global6')})
            arr.update(frame_ids=old['frame_ids'].copy(),positions8=old['positions8'].copy())
            trace=dict(native_prior=q0,GT_read=False)
            for k in (16,64):
                a='K'+str(k); q,stats=one_step(z,q0,kappa=k,beta=1.)
                iv=final_mbr(ids,q)['interval']; boxes=evidence_dp(g.for_interval(iv))['final'].numpy()
                arr[a]=boxes; arr[a+'_interval']=np.asarray(iv); trace[a]=q; trace[a+'_L']=stats['L']
                if k==64:
                    assert np.array_equal(q,ref['NativeFeedback']) and np.array_equal(boxes,old['NativeFeedback']) and np.array_equal(arr[a+'_interval'],old['NativeFeedback_interval'])
                    exact+=1
        npz=D/'predictions'/(key+'.npz'); npz.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(npz,**arr)
        tf=D/'traces'/(key+'.pt'); tf.parent.mkdir(parents=True,exist_ok=True); torch.save(trace,tf)
        files[key]=dict(path=str(npz.relative_to(D)),sha256=sha(npz),trace=str(tf.relative_to(D)),trace_sha256=sha(tf))
        atomic(D/'progress.json',dict(completed=n,total=256,GT_read=False))
        if n%32==0: print('predicted',n,flush=True)
    verify(); assert exact==256
    atomic(D/'sealed.json',dict(status='sealed',queries=256,files=files,fingerprint=p['fingerprint'],GT_read=False,K64_exact_queries=exact,seconds=time.perf_counter()-start))
    atomic(D/'state.json',dict(status='sealed_waiting_independent_CPU_score',fingerprint=p['fingerprint']))
    print('sealed',flush=True)

def summarize(rows):
    from full_score import cluster
    fields=list(rows[0]['values']['Source'])
    means={a:{f:float(np.mean([x['values'][a][f] for x in rows if x['values'][a][f] is not None])) if any(x['values'][a][f] is not None for x in rows) else None for f in fields} for a in ARMS}
    risks={a+'-'+b:dict(time_count=sum(x['values'][a]['t']-x['values'][b]['t']<-.05 for x in rows),v_count=sum(x['values'][a]['v']-x['values'][b]['v']<-.05 for x in rows),union_count=sum(min(x['values'][a][f]-x['values'][b][f] for f in ('t','v'))<-.05 for x in rows)) for a,b in PAIRS}
    for r in risks.values(): r['union_percent']=100*r['union_count']/len(rows)
    return dict(queries=len(rows),parents=len({x['parent'] for x in rows}),means=means,paired={a+'-'+b:cluster(rows,a,b) for a,b in PAIRS},severe=risks)

def score():
    from full_score import metric
    p=verify(); s=read(D/'sealed.json'); assert s['queries']==256 and not s['GT_read'] and s['fingerprint']==p['fingerprint']
    assert not (D/'score_receipt.json').exists(); torch.set_num_threads(1)
    atomic(D/'score_worker.json',identity()); atomic(D/'state.json',dict(status='scoring',**identity()))
    for e in s['files'].values(): assert sha(D/e['path'])==e['sha256'] and sha(D/e['trace'])==e['trace_sha256']
    start=time.perf_counter(); checks=0; maxerr=0.; domains={}; parent_report=read(ROOT/'results/native_prior_v1/report.json')
    signed=['dataset\tstage\tquery\tparent\tduration_group\tmethod\tbaseline\tdelta_t_pp\tdelta_v_pp']; severe=[signed[0]]
    for ds in ('hc2','vidstg'):
        job=p['jobs'][ds]; lab=PROVIDER/job['labels']; assert sha(lab)==job['labels_sha256']
        truth={x['id']:x for x in map(json.loads,lab.read_text().splitlines()) if x['dataset']==ds}
        its=[i for i in p['items'] if i['dataset']==ds]; parents=sorted({str(truth[i['row']['id']]['parent']) for i in its}); pm={v:f'P{k+1:03d}' for k,v in enumerate(parents)}
        oldrows={x['query']:x for x in parent_report['domains'][ds]['rows']}; rows=[]
        for i in its:
            r=i['row']; gt=truth[r['id']]; boxes={int(k):v for k,v in gt['boxes'].items()}; span=gt['interval']; vals={}; e=s['files'][ds+'_'+i['alias']]
            with np.load(D/e['path'],allow_pickle=False) as a:
                for arm in ARMS:
                    vals[arm],err=metric(a[arm],r,boxes,span,a[arm+'_interval'].tolist(),a['positions8']); checks+=1; maxerr=max(maxerr,err)
            for a,b in [('K64','NativeFeedback'),('Source','Source'),('NativeSpatial','NativeSpatial'),('Global6','Global6')]: assert vals[a]==oldrows[i['alias']]['values'][b]
            row=dict(query=i['alias'],parent=pm[str(gt['parent'])],stage=i['stage'],duration_group='short_le5s' if (span[1]-span[0])/r['metadata']['fps']<=5 else 'long_gt5s',values=vals); rows.append(row)
            for a,b in PAIRS:
                dt=100*(vals[a]['t']-vals[b]['t']); dv=100*(vals[a]['v']-vals[b]['v']); line=f"{ds}\t{i['stage']}\t{i['alias']}\t{row['parent']}\t{row['duration_group']}\t{a}\t{b}\t{dt:.12g}\t{dv:.12g}"
                signed.append(line)
                if min(dt,dv)<-5: severe.append(line)
        groups={}
        for name,rr in [('all128',rows),('dev96',[x for x in rows if x['stage']=='dev']),('confirm32',[x for x in rows if x['stage']=='confirm'])]:
            assert len(rr)=={'all128':128,'dev96':96,'confirm32':32}[name]
            groups[name]=summarize(rr)
            groups[name]['duration']={g:summarize([x for x in rr if x['duration_group']==g]) for g in ('short_le5s','long_gt5s')}
        domains[ds]=dict(groups=groups,rows=rows)
    assert checks==1280 and maxerr<2e-10 and not torch.cuda.is_initialized()
    report=dict(status='complete_scored',domains=domains,checks=checks,max_metric_error=maxerr,seconds=time.perf_counter()-start,plan_sha256=sha(D/'plan.json'),seal_sha256=sha(D/'sealed.json'),exposure=p['exposure'],no_selection=True,production_unchanged=True)
    atomic(P/'report.json',report); (P/'signed.tsv').write_text('\n'.join(signed)+'\n'); (P/'severe.tsv').write_text('\n'.join(severe)+'\n')
    verify()
    atomic(D/'score_receipt.json',dict(status='scored',report_sha256=sha(P/'report.json'),seal_sha256=sha(D/'sealed.json'),**identity()))
    atomic(D/'state.json',dict(status='complete_scored',fingerprint=p['fingerprint'],new_models=0,new_expert=0,production_changed=False))
    for ds,v in domains.items():
        for name,g in v['groups'].items(): print(ds,name,{a:{f:round(g['means'][a][f]*100,6) for f in ('t','v')} for a in ARMS},g['severe']['K16-K64'],flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('command',choices=('prepare','predict','score')); args=ap.parse_args(); globals()[args.command]()
