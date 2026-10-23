"""Bounded algorithm pruning in both branches, without retuning or GPU work."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import argparse,hashlib,importlib.util,json,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path[:0]=[str(HERE),str(ROOT/'studies/algorithm_simplification'),str(ROOT),str(ROOT/'tools')]
import numpy as np
import torch
from full import read,sha,atomic,identity,check,PROVIDER
from stvg.evidence import SourceEvidence
from stvg.spatial import SpatialGeometry,evidence_dp
from stvg.prior import from_capture
from stvg.temporal import final_mbr
from stvg.prototype import one_step as current_step
from simple_spatial import build as simple_build
from simple_time import one_step as pooled_step


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

R=load('existing_simplification',ROOT/'studies/algorithm_simplification/run.py')
K=load('existing_kappa_compare',ROOT/'studies/native_prior_kappa_compare/run.py')
D=ROOT/'outputs/algorithm_pruning_v1';P=ROOT/'results/algorithm_pruning_v1'
S_ARMS=('Source','Current','BoxLink','SimplePath')
T_ARMS=('Source','NativeSpatial','Global6','Current','NoSplit')


def verify():
    p=read(D/'plan.json')
    assert p['fingerprint']==hashlib.sha256(json.dumps({k:v for k,v in p.items()if k!='fingerprint'},sort_keys=True).encode()).hexdigest()
    for f,h in p['code'].items():assert sha(ROOT/f)==h,f
    for f,h in p['parents'].items():assert sha(f)==h,f
    R.verify();K.verify();check()
    assert not torch.cuda.is_initialized()
    return p


def prepare():
    assert not(D/'plan.json').exists();prod=check();R.verify();K.verify()
    assert read(R.D/'state.json')['status']=='complete_scored'
    assert read(K.D/'state.json')['status']=='complete_scored'
    code={f:sha(ROOT/f)for f in prod['code']}
    code.update({str(f.relative_to(ROOT)):sha(f)for f in HERE.glob('*.py')})
    code.update({f:sha(ROOT/f)for f in ['studies/algorithm_simplification/run.py','studies/algorithm_simplification/lightspatial.py','studies/native_prior_kappa_compare/run.py']})
    p=dict(schema='algorithm_pruning_v1',authorization='User clarifies actual algorithm simplification in both branches; preserve core complete-box propagation/path/prototype mechanisms. Do not write an article or interrupt current full.',
           primary_spatial='SimplePath-Current',primary_temporal='NoSplit-Current',
           spatial='Keep robust shared init, BOTH nearest past/future anchors, SAME bidirectional affinity and full-box affine fit/validity. Remove cycle gates/weights, extent RBF and initialization guard; support is mean anchor appearance cosine only; adjacent link complete-box IoU. Original .5 observed/.5 unobserved/.25 link, E fallback, exact DP remain. No nearest-only arm or weight search.',
           temporal='Keep same Native2D pooled features/native prior, one shrinkage update kappa64/beta1/MBR and original spatial geometry. Remove only cross-offset fitting: one centered/L2 feature set and ONE event/background pair from all video nodes. Same seed and prior; potential self-inclusion bias explicitly tested. No other normalization/weight/round/region scan.',
           scope='Spatial67exposed availabilityHC fixed currenttime/ROI, not end-to-end altered spatial-ROI loop. Temporal128exposed parents/domain (old dev96/confirm32 all already exposed), original geometry executed at new interval. Two modular tests, not a combined final method or new full selection.',
           budget=dict(spatial=67,temporal=256,new_RGB=0,new_model=0,new_expert=0,GPU=0,training=0,parameter_scan=0),
           no_production_change=True,no_control_promotion=True,seed=20260920,code=code,
           parents={str(x):sha(x)for x in [R.D/'plan.json',R.D/'sealed.json',R.D/'score_receipt.json',K.D/'plan.json',K.D/'sealed.json',K.D/'score_receipt.json']})
    p['fingerprint']=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest()
    atomic(D/'plan.json',p);atomic(P/'protocol.json',p)
    for f in code:
        dest=D/'source_snapshot'/f;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/f).read_bytes())
    q=np.triu(np.ones((9,9)),1);q/=q.sum();assert np.array_equal(pooled_step(np.ones((9,5)),q),q)
    atomic(D/'state.json',dict(status='prepared',fingerprint=p['fingerprint']))


def predict():
    p=verify();assert not(D/'sealed.json').exists();torch.set_num_threads(1)
    atomic(D/'worker.json',identity());atomic(D/'state.json',dict(status='predicting',**identity()))
    files={};timing=[];start=time.perf_counter();sp=R.verify();ss=read(R.D/'sealed.json')
    for n,item in enumerate(sp['items'],1):
        cap=torch.load(item['capture_path'],map_location='cpu',weights_only=False)
        raw=torch.load(item['raw_path'],map_location='cpu',weights_only=False)
        g=SpatialGeometry.__new__(SpatialGeometry);g.__dict__.update(cap['geometry'])
        features,valid=R.known_grid(raw['spatial_inputs'],cap['frame_ids'],g.height,g.width)
        ev=R.C.load_evidence(item,cap)
        two=R.build(features,valid,cap['frame_ids'],cap['positions8'],ev.seeds,cap['E'],g.native_boxes,nearest=False)
        simple=simple_build(features,valid,cap['frame_ids'],cap['positions8'],ev.seeds,cap['E'],g.native_boxes)
        assert torch.equal(simple.boxes,two.boxes)and torch.equal(simple.valid,two.valid)
        assert torch.equal(simple.link,two.link)and torch.equal(simple.source_anchors,two.source_anchors)
        expected=(two.association_scores*two.side_available[:,None]).sum(-1)/two.side_available.sum(-1).clamp_min(1)[:,None]
        assert torch.equal(simple.projection,expected)and not simple.guard.any()
        e=ss['files'][item['source_key']]
        with np.load(R.D/e['path'],allow_pickle=False)as old:
            arr={a:old[a].copy()for a in S_ARMS if a!='SimplePath'}
            arr.update({a+'_interval':old[a+'_interval'].copy()for a in S_ARMS if a!='SimplePath'})
            arr.update(frame_ids=old['frame_ids'].copy(),positions8=old['positions8'].copy())
            span=old['Current_interval'].tolist();arr['SimplePath_interval']=old['Current_interval'].copy()
            arr['SimplePath']=evidence_dp(simple.for_interval(span))['final'].numpy()
        times={k:[]for k in ['Current','BoxLink','SimplePath']}
        for rep in range(3):
            for kind in (list(times)if rep%2==0 else list(reversed(times))):
                tick=time.perf_counter()
                if kind=='Current':
                    ee=SourceEvidence.from_arrays(features,valid,cap['frame_ids'],cap['positions8'],ev.seeds)
                    SpatialGeometry(ee,cap['E'],g.native_boxes)
                elif kind=='BoxLink':R.build(features,valid,cap['frame_ids'],cap['positions8'],ev.seeds,cap['E'],g.native_boxes,nearest=False)
                else:simple_build(features,valid,cap['frame_ids'],cap['positions8'],ev.seeds,cap['E'],g.native_boxes)
                times[kind].append(time.perf_counter()-tick)
        f=D/'spatial'/f'H{n:03d}.npz';f.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(f,**arr)
        files['S_'+item['source_key']]=dict(alias=f'H{n:03d}',path=str(f.relative_to(D)),sha256=sha(f))
        timing.append(dict(query=f'H{n:03d}',median_seconds={k:float(np.median(v))for k,v in times.items()}))
        if n%16==0:print('spatial',n,flush=True)
    kp=K.verify();ks=read(K.D/'sealed.json');time_cost={'Current':[],'NoSplit':[]}
    for n,item in enumerate(kp['items'],1):
        cap=torch.load(item['path'],map_location='cpu',weights_only=False)
        tr=torch.load(item['feature_path'],map_location='cpu',weights_only=False)
        z=tr['features']['Native2D']if isinstance(tr['features'],dict)else tr['features']
        q0=from_capture(cap);q=pooled_step(z,q0);ids=cap['frame_ids'];iv=final_mbr(ids,q)['interval']
        g=SpatialGeometry.__new__(SpatialGeometry);g.__dict__.update(cap['geometry'])
        key=item['dataset']+'_'+item['alias'];e=ks['files'][key]
        with np.load(K.D/e['path'],allow_pickle=False)as old:
            arr={a:old[a].copy()for a in ['Source','NativeSpatial','Global6']}
            arr.update({a+'_interval':old[a+'_interval'].copy()for a in ['Source','NativeSpatial','Global6']})
            arr.update(Current=old['K64'].copy(),Current_interval=old['K64_interval'].copy(),frame_ids=old['frame_ids'].copy(),positions8=old['positions8'].copy())
            arr.update(NoSplit=evidence_dp(g.for_interval(iv))['final'].numpy(),NoSplit_interval=np.asarray(iv))
        for kind,fn in [('Current',lambda:current_step(z,q0)),('NoSplit',lambda:pooled_step(z,q0))]:
            fn();values=[]
            for rep in range(3):
                tick=time.perf_counter();fn();values.append(time.perf_counter()-tick)
            time_cost[kind].append(float(np.median(values)))
        f=D/'temporal'/(key+'.npz');f.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(f,**arr)
        files['T_'+key]=dict(path=str(f.relative_to(D)),sha256=sha(f))
        if n%64==0:print('temporal',n,flush=True)
    verify();atomic(D/'sealed.json',dict(status='sealed',spatial=67,temporal=256,files=files,GT_read=False,
                                       timings=timing,temporal_timing=time_cost,seconds=time.perf_counter()-start))
    atomic(D/'state.json',dict(status='sealed_waiting_independent_CPU_score',fingerprint=p['fingerprint']))


def summarize(rows,arms,pairs,risk_fields):
    from full_score import cluster
    fields=rows[0]['values']['Source']
    means={a:{f:float(np.mean([r['values'][a][f]for r in rows if r['values'][a][f]is not None]))if any(r['values'][a][f]is not None for r in rows)else None for f in fields}for a in arms}
    severe={a+'-'+b:dict(counts={f:sum(r['values'][a][f]-r['values'][b][f]<-.05 for r in rows)for f in risk_fields},union=sum(any(r['values'][a][f]-r['values'][b][f]<-.05 for f in risk_fields)for r in rows))for a,b in pairs}
    return dict(queries=len(rows),means=means,paired={a+'-'+b:cluster(rows,a,b)for a,b in pairs},severe=severe)


def score():
    from full_score import metric
    p=verify();s=read(D/'sealed.json');assert s['spatial']==67 and s['temporal']==256 and not s['GT_read']
    assert not(D/'score_receipt.json').exists();torch.set_num_threads(1);atomic(D/'score_worker.json',identity())
    for e in s['files'].values():assert sha(D/e['path'])==e['sha256']
    checks=0;errmax=0.;rows=[];start=time.perf_counter();sp=R.verify();job=sp['job'];lab=PROVIDER/job['labels'];assert sha(lab)==job['labels_sha256']
    truth={x['id']:x for x in map(json.loads,lab.read_text().splitlines())if x['dataset']=='hc2'}
    old={r['query']:r for r in read(R.P/'report.json')['rows']}
    for item in sp['items']:
        e=s['files']['S_'+item['source_key']];gt=truth[item['source_key']];boxes={int(k):v for k,v in gt['boxes'].items()};vals={}
        with np.load(D/e['path'],allow_pickle=False)as a:
            for arm in S_ARMS:
                vals[arm],err=metric(a[arm],item['row'],boxes,gt['interval'],a[arm+'_interval'].tolist(),a['positions8']);checks+=1;errmax=max(errmax,err)
        for arm in ['Source','Current','BoxLink']:assert vals[arm]==old[e['alias']]['values'][arm]
        rows.append(dict(query=e['alias'],parent=old[e['alias']]['parent'],duration_group=old[e['alias']]['duration_group'],values=vals))
    pairs=[('SimplePath','Current'),('SimplePath','BoxLink'),('SimplePath','Source'),('Current','Source')]
    sg=summarize(rows,S_ARMS,pairs,('v','unobserved_actual_sIoU'));sg['duration']={tag:summarize([r for r in rows if r['duration_group']==tag],S_ARMS,pairs,('v','unobserved_actual_sIoU'))for tag in ['short_le5s','long_gt5s']}
    spatial=dict(summary=sg,rows=rows);domains={};kp=K.verify();old=read(K.P/'report.json')
    for ds in ['hc2','vidstg']:
        job=kp['jobs'][ds];lab=PROVIDER/job['labels'];assert sha(lab)==job['labels_sha256']
        truth={x['id']:x for x in map(json.loads,lab.read_text().splitlines())if x['dataset']==ds};oldrows={r['query']:r for r in old['domains'][ds]['rows']};rr=[]
        for item in kp['items']:
            if item['dataset']!=ds:continue
            e=s['files']['T_'+ds+'_'+item['alias']];r=item['row'];gt=truth[r['id']];boxes={int(k):v for k,v in gt['boxes'].items()};vals={}
            with np.load(D/e['path'],allow_pickle=False)as a:
                for arm in T_ARMS:
                    vals[arm],err=metric(a[arm],r,boxes,gt['interval'],a[arm+'_interval'].tolist(),a['positions8']);checks+=1;errmax=max(errmax,err)
            prev=oldrows[item['alias']]
            for arm,key in [('Source','Source'),('NativeSpatial','NativeSpatial'),('Global6','Global6'),('Current','K64')]:assert vals[arm]==prev['values'][key]
            rr.append(dict(query=item['alias'],parent=prev['parent'],stage=prev['stage'],duration_group=prev['duration_group'],values=vals))
        pp=[('NoSplit','Current'),('NoSplit','Source'),('NoSplit','Global6'),('Current','Source')];groups={}
        for name,rs in [('all128',rr),('exposed_dev96',[r for r in rr if r['stage']=='dev']),('exposed_confirm32',[r for r in rr if r['stage']=='confirm'])]:
            gg=summarize(rs,T_ARMS,pp,('t','v'));gg['duration']={tag:summarize([r for r in rs if r['duration_group']==tag],T_ARMS,pp,('t','v'))for tag in ['short_le5s','long_gt5s']};groups[name]=gg
        domains[ds]=dict(groups=groups,rows=rr)
    assert checks==1548 and errmax<2e-10 and not torch.cuda.is_initialized()
    report=dict(status='complete_scored',spatial=spatial,temporal=domains,checks=checks,max_metric_error=errmax,
                spatial_mean_median_seconds={k:float(np.mean([r['median_seconds'][k]for r in s['timings']]))for k in s['timings'][0]['median_seconds']},
                temporal_mean_median_seconds={k:float(np.mean(v))for k,v in s['temporal_timing'].items()},seconds=time.perf_counter()-start,
                no_production_change=True,no_combined_method_claim=True,no_parameter_or_control_selection=True)
    atomic(P/'report.json',report)
    lines=['branch\tdataset\tquery\tmethod\tbaseline\tdelta_v_pp\tdelta_secondary_pp'];negative=[lines[0]]
    for branch,ds,rs,a,b,other in [('spatial','hc2',rows,'SimplePath','Current','unobserved_actual_sIoU')]+[('temporal',ds,v['rows'],'NoSplit','Current','t')for ds,v in domains.items()]:
        for r in rs:
            dv=100*(r['values'][a]['v']-r['values'][b]['v']);dt=100*(r['values'][a][other]-r['values'][b][other]);line=f"{branch}\t{ds}\t{r['query']}\t{a}\t{b}\t{dv:.12g}\t{dt:.12g}";lines.append(line)
            if min(dv,dt)<-5:negative.append(line)
    (P/'signed.tsv').write_text('\n'.join(lines)+'\n');(P/'severe.tsv').write_text('\n'.join(negative)+'\n')
    verify();atomic(D/'score_receipt.json',dict(report_sha256=sha(P/'report.json'),**identity()));atomic(D/'state.json',dict(status='complete_scored',fingerprint=p['fingerprint']))
    print(json.dumps(dict(spatial=sg['paired']['SimplePath-Current'],temporal={ds:v['groups']['all128']['paired']['NoSplit-Current']for ds,v in domains.items()})),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['prepare','predict','score']);a=ap.parse_args();globals()[a.command]()
