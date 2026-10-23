"""One fixed CPU-only prior replacement, using sealed Native2D calibration caches.

No Source/teacher/video access and no target labels until every prediction seals.
The production runner/configuration are never edited or invoked by this study.
"""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
import argparse, hashlib, json, math, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'tools')]
import numpy as np
import torch
from full import read, sha, atomic, identity, check, PROVIDER
from stvg.global6 import posterior
from stvg.temporal import probabilities, mass_table, final_mbr
from stvg.prototype import one_step
from stvg.spatial import SpatialGeometry, evidence_dp

D = ROOT/'outputs/native_prior_v1'
P = ROOT/'results/native_prior_v1'
ARMS = ('Source', 'NativeSpatial', 'Global6', 'Full', 'NativeFeedback')
PAIRS = [('NativeFeedback', b) for b in ('Source', 'NativeSpatial', 'Global6', 'Full')]
PAIRS += [('Full', 'Global6'), ('Full', 'Source'), ('Global6', 'Source')]

def verify():
    p = read(D/'plan.json')
    assert p['fingerprint'] == hashlib.sha256(json.dumps({k:v for k,v in p.items() if k!='fingerprint'}, sort_keys=True).encode()).hexdigest()
    for f,h in p['code'].items(): assert sha(ROOT/f)==h, f
    for f,h in p['inputs'].items(): assert sha(f)==h, f
    assert sha(ROOT/'outputs/native2d_full_v2/plan.json')==p['production_plan_sha256']
    assert not torch.cuda.is_initialized()
    return p

def prepare():
    assert not (D/'plan.json').exists()
    prod=check(); items=read(ROOT/'outputs/native2d_full_v2/promotion_inputs.json')
    old=ROOT.parent/'stvg-general/outputs/native2d_risk_v2'
    study=read(old/'plan.json'); stages={}
    for stage in ('dev','confirm'):
        for x in study[stage+'_items']: stages[(x['dataset'],x['row']['id'])]=(stage,x['alias'])
    for i in items: i['stage'],i['alias']=stages[(i['dataset'],i['row']['id'])]
    assert len(items)==256 and all(sum(i['dataset']==ds for i in items)==128 for ds in ('hc2','vidstg'))
    inputs={str(ROOT/'outputs/native2d_full_v2/promotion_inputs.json'):sha(ROOT/'outputs/native2d_full_v2/promotion_inputs.json')}
    for stage in ('dev','confirm'):
        inputs[str(old/stage/'sealed.json')]=sha(old/stage/'sealed.json')
    code={f:sha(ROOT/f) for f in prod['code']}
    code['tools/full_score.py']=sha(ROOT/'tools/full_score.py')
    code[str(Path(__file__).relative_to(ROOT))]=sha(__file__)
    p=dict(schema='native_prior_v1',authorization='User requests cached native temporal prior + Native2D, fixed parameters/spatial, fair Source/Global6/current Full precision and severe-harm comparison, do not interrupt full.',
        primary='NativeFeedback',arms=ARMS,one_change='Replace Global6 six-layer averaged prior by exact native last-layer two-offset hull prior, BOTH for prototype soft seeds and posterior prior.',
        fixed=dict(kappa=64,beta=1.,rounds=1,context=1.5,spatial_observed=.5,spatial_guard=.1,spatial_link=.25,original_MBR=True,evidence8=True),
        native_definition='Native final-layer logits from both original offsets -> original physical hull -> original MBR; not removal of original offset hull or MBR.',
        budget=dict(queries=256,per_direction=128,new_model=0,new_video=0,new_expert=0,new_training=0,GPU=0),
        groups='Existing dev96 + finalconfirm32 per direction, separately reported and pooled descriptively; ALL historically exposed, no selection/promotion.',
        severe='Paired tIoU or complete vIoU decline >5 percentage points; union counts a query once. Report separately versus Source, NativeSpatial, Global6 and Full.',
        protocol='All256 predictions + traces seal before independent CPU labels. No tuning or production changes. Full q/interval/boxes + Source/Global6 reproduce exact sealed references.',
        production_plan_sha256=sha(ROOT/'outputs/native2d_full_v2/plan.json'),production_fingerprint=prod['fingerprint'],
        code=code,inputs=inputs,items=items,jobs=prod['jobs'],seed=20260920,GT_in_prediction=False)
    p['fingerprint']=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest()
    atomic(D/'plan.json',p)
    for f in code:
        dest=D/'source_snapshot'/f; dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/f).read_bytes())
    atomic(P/'protocol.json',{k:v for k,v in p.items() if k not in ('items','jobs','inputs')})
    # A constant representation has zero contrast for EITHER prior, exactly.
    q=np.triu(np.ones((7,7)),1);q/=q.sum()
    z=np.ones((7,4));assert np.array_equal(one_step(z,q,kappa=64,beta=1.)[0],q)
    atomic(D/'tests.json',dict(constant_representation_exact_noop=True,GT_read=False,model_calls=0))
    atomic(D/'state.json',dict(status='prepared',fingerprint=p['fingerprint']))
    print('prepared',p['fingerprint'],flush=True)

def predict():
    p=verify();assert not (D/'sealed.json').exists();torch.set_num_threads(1)
    atomic(D/'worker.json',identity());atomic(D/'state.json',dict(status='predicting',**identity()))
    files={};started=time.perf_counter()
    for n,i in enumerate(p['items'],1):
        for f,h in [(i['path'],i['sha256']),(i['feature_path'],i['feature_sha256']),(i['reference_prediction'],i['reference_sha256']),(i['reference_trace'],i['reference_trace_sha256'])]: assert sha(f)==h
        cap=torch.load(i['path'],weights_only=False,map_location='cpu')
        tr=torch.load(i['feature_path'],weights_only=False,map_location='cpu')
        ref=torch.load(i['reference_trace'],weights_only=False,map_location='cpu')
        z=tr['features']['Native2D'] if isinstance(tr['features'],dict) else tr['features']
        ids=list(map(int,cap['frame_ids']));pos=cap.get('positions8',cap.get('planned_positions'))
        qg=posterior(cap['records'],cap['logits'],ids)
        U,qn=mass_table(probabilities(cap['records'],[x[-1] for x in cap['logits']]))
        assert U==ids
        qfull,_=one_step(z,qg,kappa=64,beta=1.)
        assert np.array_equal(qfull,ref['q'][i['reference_q']]), (i['alias'],'Full posterior')
        qnew,stats=one_step(z,qn,kappa=64,beta=1.)
        ng=final_mbr(ids,qn)['interval']; gg=final_mbr(ids,qg)['interval']; fi=final_mbr(ids,qfull)['interval']; ni=final_mbr(ids,qnew)['interval']
        if 'native_interval' in cap: assert ng==list(map(int,cap['native_interval']))
        g=SpatialGeometry.__new__(SpatialGeometry);g.__dict__.update(cap['geometry'])
        spatial=lambda iv:evidence_dp(g.for_interval(iv))['final'].numpy()
        native_boxes=spatial(ng);new_boxes=spatial(ni);full_boxes=spatial(fi);global_boxes=spatial(gg)
        with np.load(i['reference_prediction'],allow_pickle=False) as old:
            for a,b in [(full_boxes,i['reference_arm']),(np.asarray(fi),i['reference_arm']+'_interval'),(global_boxes,'NativeG6'),(np.asarray(gg),'NativeG6_interval'),(np.asarray(cap['Source']),'Source'),(np.asarray(ng),'Source_interval'),(np.asarray(pos),'positions8')]: assert np.array_equal(a,old[b]),(i['alias'],b)
            source=old['Source'].copy();source_iv=old['Source_interval'].copy()
        arr=dict(Source=source,NativeSpatial=native_boxes,Global6=global_boxes,Full=full_boxes,NativeFeedback=new_boxes,
            Source_interval=source_iv,NativeSpatial_interval=np.asarray(ng),Global6_interval=np.asarray(gg),Full_interval=np.asarray(fi),NativeFeedback_interval=np.asarray(ni),frame_ids=np.asarray(ids),positions8=np.asarray(pos))
        key=i['dataset']+'_'+i['alias'];npz=D/'predictions'/(key+'.npz');npz.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(npz,**arr)
        trace=D/'traces'/(key+'.pt');trace.parent.mkdir(parents=True,exist_ok=True);torch.save(dict(native_prior=qn,Global6_prior=qg,Full=qfull,NativeFeedback=qnew,L=stats['L'],GT_read=False),trace)
        files[key]=dict(path=str(npz.relative_to(D)),sha256=sha(npz),trace=str(trace.relative_to(D)),trace_sha256=sha(trace))
        atomic(D/'progress.json',dict(completed=n,total=len(p['items']),GT_read=False))
        if n%32==0:print('predicted',n,flush=True)
    verify();assert len(files)==256
    atomic(D/'sealed.json',dict(status='sealed',queries=256,fingerprint=p['fingerprint'],files=files,GT_read=False,seconds=time.perf_counter()-started))
    atomic(D/'state.json',dict(status='sealed_waiting_independent_CPU_score',fingerprint=p['fingerprint']))
    print('sealed',time.perf_counter()-started,flush=True)

def score():
    from full_score import metric,cluster
    p=verify();s=read(D/'sealed.json');assert s['queries']==256 and not s['GT_read'] and s['fingerprint']==p['fingerprint']
    for e in s['files'].values():assert sha(D/e['path'])==e['sha256'] and sha(D/e['trace'])==e['trace_sha256']
    torch.set_num_threads(1);started=time.perf_counter();domains={};checks=0;maxerr=0
    signed=['dataset\tgroup\tquery\tparent\tmethod\tbaseline\tdelta_t_pp\tdelta_v_pp'];severe=[signed[0]]
    for ds in ('hc2','vidstg'):
        j=p['jobs'][ds];lab=PROVIDER/j['labels'];assert sha(lab)==j['labels_sha256']
        truth={x['id']:x for x in map(json.loads,lab.read_text().splitlines()) if x['dataset']==ds}
        its=[i for i in p['items'] if i['dataset']==ds]
        parents=sorted({str(truth[i['row']['id']]['parent']) for i in its});pm={v:f'P{k+1:03d}' for k,v in enumerate(parents)}
        rows=[]
        for i in its:
            r=i['row'];gt=truth[r['id']];span=gt['interval'];boxes={int(k):v for k,v in gt['boxes'].items()};e=s['files'][ds+'_'+i['alias']];values={}
            with np.load(D/e['path'],allow_pickle=False) as a:
                for arm in ARMS:
                    values[arm],err=metric(a[arm],r,boxes,span,a[arm+'_interval'].tolist(),a['positions8']);checks+=1;maxerr=max(maxerr,err)
            row=dict(query=i['alias'],parent=pm[str(gt['parent'])],stage=i['stage'],duration_group='short_le5s' if (span[1]-span[0])/r['metadata']['fps']<=5 else 'long_gt5s',values=values);rows.append(row)
            for a,b in PAIRS:
                dt=100*(values[a]['t']-values[b]['t']);dv=100*(values[a]['v']-values[b]['v']);line=f"{ds}\t{i['stage']}\t{i['alias']}\t{row['parent']}\t{a}\t{b}\t{dt:.12g}\t{dv:.12g}"
                signed.append(line)
                if min(dt,dv)<-5:severe.append(line)
        groups={}
        for name,rr in [('all128',rows),('dev96',[x for x in rows if x['stage']=='dev']),('confirm32',[x for x in rows if x['stage']=='confirm'])]:
            assert len(rr)=={'all128':128,'dev96':96,'confirm32':32}[name]
            means={a:{f:float(np.mean([x['values'][a][f] for x in rr if x['values'][a][f] is not None])) if any(x['values'][a][f] is not None for x in rr) else None for f in rr[0]['values'][a]} for a in ARMS}
            pairs={a+'-'+b:cluster(rr,a,b) for a,b in PAIRS}
            risks={a+'-'+b:dict(time_count=sum(x['values'][a]['t']-x['values'][b]['t']<-.05 for x in rr),v_count=sum(x['values'][a]['v']-x['values'][b]['v']<-.05 for x in rr),union_count=sum(min(x['values'][a][f]-x['values'][b][f] for f in ('t','v'))<-.05 for x in rr)) for a,b in PAIRS}
            duration={g:dict(queries=sum(x['duration_group']==g for x in rr),paired={a+'-'+b:{f:float(np.mean([100*(x['values'][a][f]-x['values'][b][f]) for x in rr if x['duration_group']==g])) if any(x['duration_group']==g for x in rr) else None for f in ('t','v')} for a,b in PAIRS}) for g in ('short_le5s','long_gt5s')}
            groups[name]=dict(queries=len(rr),parents=len({x['parent'] for x in rr}),means=means,paired=pairs,severe=risks,duration=duration)
        domains[ds]=dict(groups=groups,rows=rows)
    assert checks==1280 and maxerr<2e-10 and not torch.cuda.is_initialized()
    report=dict(status='complete_scored',primary='NativeFeedback',domains=domains,official_dense_checks=checks,max_metric_error=maxerr,plan_sha256=sha(D/'plan.json'),seal_sha256=sha(D/'sealed.json'),scoring_seconds=time.perf_counter()-started,exposure=p['groups'],no_selection=True,production_unchanged=True)
    atomic(P/'report.json',report);(P/'signed.tsv').write_text('\n'.join(signed)+'\n');(P/'severe.tsv').write_text('\n'.join(severe)+'\n')
    atomic(D/'score_receipt.json',dict(status='scored',report_sha256=sha(P/'report.json'),seal_sha256=sha(D/'sealed.json'),**identity()))
    atomic(D/'state.json',dict(status='complete_scored',fingerprint=p['fingerprint'],new_models=0,new_expert=0,production_changed=False))
    for ds,v in domains.items():
        for group,g in v['groups'].items():print(ds,group,{a:{f:round(g['means'][a][f]*100,5) for f in ('t','v')} for a in ARMS},g['severe'],flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=('prepare','predict','score'));args=ap.parse_args();globals()[args.command]()
