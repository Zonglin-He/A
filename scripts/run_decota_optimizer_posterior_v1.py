"""Finite calibrated factorial, no new backbone/expert, separate CPU posterior."""
import sys,os,time,gc,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_optimizer_posterior_common_v1 import *

def prepare():
    b=prior.verify_seal();pins=dict(prior.verify()['pins'])
    own=['vg_tta/decota_optimizer_posterior_r1_v1.py','scripts/decota_optimizer_posterior_common_v1.py',
        'scripts/run_decota_optimizer_posterior_v1.py','scripts/continue_decota_optimizer_posterior_v1.py',
        'protocols/decota_optimizer_posterior_v1.md']
    pins.update({f:sha(ROOT/f) for f in own});inputs=dict(prior.verify()['inputs'])
    for f,h in b['files'].items():
        p=prior.BASE/f;inputs[str(p.relative_to(ROOT))]=h;inputs[str(p.with_suffix('.json').relative_to(ROOT))]=sha(p.with_suffix('.json'))
    for ds in DATASETS:
        write(BASE/ds/'PLAN.json',read(prior.BASE/ds/'PLAN.json'))
        inputs[str((BASE/ds/'PLAN.json').relative_to(ROOT))]=sha(BASE/ds/'PLAN.json')
        for f in (prior.OLD/ds/'temporal').rglob('*.pt'):
            inputs[str(f.relative_to(ROOT))]=sha(f);inputs[str(f.with_suffix('.json').relative_to(ROOT))]=sha(f.with_suffix('.json'))
    write(BASE/'RUNTIME_LOCK.json',dict(version='decota_optimizer_posterior_v1',time=time.time(),pins=pins,inputs=inputs,
        protected=prior.verify()['protected_registries'],spatial_prestate='frozen P1 trajectory',
        parameters=1792,steps=10,Adam_lr=.03,proposal_temperature=1,reward_temperature=1,
        calibration='32 search clean order1 median first-step normalized cxcywh L1',
        posterior_beta=1,no_GT_online=True,arrivals=1152,new_expert_calls=0,new_backbone=0,
        historical_exposure=True,full_query_jobs_not_started=True))
    write(BASE/'ROUTE.json',dict(R1='implemented_pending_run',R2='conditional_on_R1',R3='conditional_on_R2',
        R4='conditional_on_usable_spatial_evidence',R5='conditional_online_confirmation',R6='conditional_budget_cross_domain',
        optional_LBFGS_Newton_region_critic_duration_memory='not_current_execution',
        full_dataset_scope_question='pending; default full fixed panel route',all_rounds_completed=False))
    status(BASE/'STATUS.json',dict(status='locked_pending_contracts',GT_read=False))
    archive('首轮完整factorial与无梯度posterior冻结，零新预测，后续轮次尚未运行')

def start_gpu():return __import__('scripts.run_tastvg_decota_critic_ln_p1_v1',fromlist=['start_gpu']).start_gpu()

def context(model,ds,row,cond,split,order,at):
    from scripts.run_tastvg_decota_c1_same_domain_v1 import data_input
    from scripts import tastvg_decota_c1_common_v1 as c1
    from vg_tta.tastvg_decota_c1_same_domain_v1 import NormalizedSpatialReplay
    data,rc=data_input(ds,row,cond);base=NormalizedSpatialReplay(model,data)
    pp=prior.BASE/ds/'online'/split/cond/order/f'{at:05}.pt';x=prior.checked(pp)
    assert x['parent']==row['ordinal'] and x['pixel_sha256']==rc['pixel_sha256']
    ef=prior.OLD/ds/'evidence'/cond/f"{row['ordinal']:05}.pt";ex=c1.checked(ef)
    assert ex['pixel_sha256']==rc['pixel_sha256']
    return base,x,ex['expert'],pp,ef

def calibrate(ds):
    import torch,numpy as np
    from scripts.run_tastvg_decota_c1_same_domain_v1 import model_for
    from vg_tta.decota_optimizer_posterior_r1_v1 import OBJECTIVES,fit,flat,box_displacement
    from vg_tta.c1_enabling_tricks_v1 import TrickReplay
    verify();handle=start_gpu();model=model_for(ds);p=read(BASE/ds/'PLAN.json');cases=[];tick=time.time()
    try:
        for at,parent in enumerate(p['splits']['search']['orders']['order1']):
            row=p['rows'][parent];base,x,ex,pp,ef=context(model,ds,row,'clean','search','order1',at)
            initial=x['initial'];data={}
            for name in OBJECTIVES:
                z=fit(base,initial,ex,row['frame_ids'],row['key'],name+'_adam',steps=1)
                data[name]=dict(gradient=z['path'][0]['update']['gradient'] if z['gradient_calls'] else torch.zeros(1792),
                    target=box_displacement(z['path'][-1]['boxes'],z['path'][0]['boxes']))
                if name=='all':
                    zz=x['fit'];assert torch.equal(z['path'][0]['boxes'],zz['path'][0]['boxes'])
                    if z['gradient_calls']:assert torch.equal(z['path'][1]['boxes'],zz['path'][1]['boxes'])
            cases.append((base,x,row,data))
            print('CALIBRATION_PREPARE',ds,len(cases),32,flush=True)
        result={}
        for name in OBJECTIVES:
            target=float(np.median([v[3][name]['target'] for v in cases]));trials=[]
            def evaluate(eta):
                values=[]
                for base,x,row,data in cases:
                    rp=TrickReplay(base,row['frame_ids'],row['key'],{});rp.restore(x['initial']);g=data[name]['gradient'].to(next(iter(rp.named))[1])
                    offset=0
                    with torch.no_grad():
                        for _,param in rp.named:
                            n=param.numel();param.add_(g[offset:offset+n].reshape_as(param),alpha=-eta);offset+=n
                        boxes=rp.values()['boxes'];values.append(box_displacement(boxes.cpu(),x['before']))
                    rp.restore(x['initial'])
                median=float(np.median(values));trials.append(dict(lr=float(eta),median=median,error=abs(median-target)))
                print('CALIBRATION',ds,name,len(trials),flush=True);return median
            grid=np.logspace(-6,3,25);med=[evaluate(float(e)) for e in grid]
            crossings=[i for i in range(24) if med[i]<=target<=med[i+1]]
            if crossings:
                j=crossings[0];lo,hi=float(grid[j]),float(grid[j+1])
                for _ in range(12):
                    mid=math.sqrt(lo*hi);value=evaluate(mid)
                    if value<target:lo=mid
                    else:hi=mid
            best=min(trials,key=lambda r:(r['error'],r['lr']))
            result[name]=dict(**best,target=target,relative_error=best['error']/max(target,1e-15),trials=trials,
                matching_is_approximate=True,GT_used=False,search_clean_sources=32)
        write(BASE/ds/'CALIBRATION.json',dict(status='sealed',objectives=result,time=time.time(),seconds=time.time()-tick,GT_read=False))
        print('CALIBRATION_SEALED',ds,flush=True)
    finally:handle.close()

def predict(ds):
    import torch
    from scripts.run_tastvg_decota_c1_same_domain_v1 import model_for
    from vg_tta.decota_optimizer_posterior_r1_v1 import ARMS,fit
    verify();cal=read(BASE/ds/'CALIBRATION.json');assert not cal['GT_read']
    handle=start_gpu();model=model_for(ds);p=read(BASE/ds/'PLAN.json');done=backwards=0;files={};tick=time.time()
    try:
        for split,sp in p['splits'].items():
            for cond in p['conditions']:
                for order,seq in sp['orders'].items():
                    for at,parent in enumerate(seq):
                        budget();row=p['rows'][parent];base,x,ex,pp,ef=context(model,ds,row,cond,split,order,at);fits={}
                        for arm in ARMS:
                            if arm=='all_adam':
                                fits[arm]={**x['fit'], 'arm':arm,'proposal_state':x['fit']['state'],'authority':1.,'lr':.03,'reused_from_P1':True}
                            else:
                                obj=arm.split('_',1)[0];fits[arm]=fit(base,x['initial'],ex,row['frame_ids'],row['key'],arm,cal['objectives'][obj]['lr'])
                                backwards+=fits[arm]['gradient_calls']
                        file=BASE/ds/'spatial'/split/cond/order/f'{at:05}.pt'
                        commit(file,dict(dataset=ds,split=split,condition=cond,order=order,arrival=at,parent=parent,
                            initial=x['initial'],before=x['before'],native=x['native'],fits=fits,GT_read=False,
                            prestate_path=str(pp.relative_to(ROOT)),prestate_sha256=sha(pp),evidence_path=str(ef.relative_to(ROOT)),evidence_sha256=sha(ef),
                            own_online_trajectory=False,LN_current_prestate_fixed=True,calibration_sha256=sha(BASE/ds/'CALIBRATION.json')))
                        files[str(file.relative_to(BASE))]=sha(file);done+=1
                        status(BASE/ds/'SPATIAL_STATUS.json',dict(status='running',done=done,total=576,pid=os.getpid(),backward_calls=backwards,seconds=time.time()-tick,GT_read=False))
                        print('OPTIMIZER_FACTORIAL',ds,done,576,round(time.time()-tick,1),flush=True)
                        del base,x,fits
                        if done%16==0:gc.collect()
        write(BASE/ds/'SPATIAL_BARRIER.json',dict(status='sealed',arrivals=done,files=files,backward_calls=backwards,
            seconds=time.time()-tick,peak_memory_bytes=torch.cuda.max_memory_allocated(),new_expert=0,new_backbone=0,GT_read=False))
        status(BASE/ds/'SPATIAL_STATUS.json',dict(status='completed',done=done,total=576,GT_read=False))
    finally:handle.close()

def temporal():
    import torch,numpy as np
    from scripts import tastvg_decota_c1_common_v1 as c1
    from vg_tta.decota_optimizer_posterior_r1_v1 import posterior
    from vg_tta.posterior_mass_coverage_v1 import select
    torch.set_num_threads(4);verify();sys.addaudithook(prior.guard);files={};done=0;tick=time.time()
    for ds in DATASETS:
        p=read(BASE/ds/'PLAN.json')
        for row in p['rows']:
            for cond in p['conditions']:
                f=BASE/ds/'temporal'/cond/f"{row['ordinal']:05}.pt"
                if f.exists():
                    saved=checked(f);assert saved['dataset']==ds and saved['parent']==row['ordinal'] and saved['condition']==cond
                    files[str(f.relative_to(BASE))]=sha(f);done+=1;continue
                x,rc=c1.cache(ds,row['pool_parent'],cond);tp=c1.BASE/ds/'temporal'/cond/f"{row['ordinal']:05}.pt";old=c1.checked(tp)
                evidence=old['evidence']['offsets'];offsets=[];intervals={'native':x['prediction']['raw_physical_intervals'],'hard':[o['interval'] for o in evidence],'full':[],'extent':[],'pm':[]}
                for z,o,r in zip(x['prediction']['logits'],evidence,x['records']):
                    from methods.decota_final_simplified_v1.objectives import legal_logp
                    lp,ij=legal_logp(z);assert torch.equal(ij.cpu(),o['ij'].cpu()) and torch.allclose(lp.cpu(),o['logp0'].cpu(),atol=2e-12,rtol=0)
                    out=posterior(lp.numpy(),o['cost'].numpy(),ij.numpy(),r['frame_ids'],beta=1.)
                    for arm,k in [('full',out['full_map']),('extent',out['extent_map'])]:
                        s,e=out['ij'][:,k];intervals[arm].append([r['frame_ids'][s],r['frame_ids'][e]+1])
                    pm=select(z,r['frame_ids'],tau=.9)['indices'];intervals['pm'].append([r['frame_ids'][pm[0]],r['frame_ids'][pm[1]]+1]);offsets.append(out)
                merged={a:[min(v[0] for v in rr),max(v[1] for v in rr)] for a,rr in intervals.items()}
                assert merged['native']==x['prediction']['physical_interval']
                f=BASE/ds/'temporal'/cond/f"{row['ordinal']:05}.pt";commit(f,dict(dataset=ds,parent=row['ordinal'],condition=cond,offsets=offsets,
                    intervals=merged,raw_intervals=intervals,temporal_cache_sha256=sha(tp),GT_read=False,parameters_updated=0))
                files[str(f.relative_to(BASE))]=sha(f);done+=1
        print('POSTERIOR_CPU',ds,done,576,flush=True)
    write(BASE/'TEMPORAL_BARRIER.json',dict(status='sealed',unique_inputs=576,files=files,GT_read=False,seconds=time.time()-tick))

def seal():
    verify();files=dict(read(BASE/'TEMPORAL_BARRIER.json')['files'])
    for ds in DATASETS:
        b=read(BASE/ds/'SPATIAL_BARRIER.json');assert b['arrivals']==576;files.update(b['files'])
    assert len(files)==1728
    for f,h in files.items():assert sha(BASE/f)==h
    write(BASE/'GLOBAL_PREDICTION_BARRIER.json',dict(status='sealed',files=files,spatial_arrivals=1152,temporal_unique_inputs=576,time=time.time(),GT_read=False))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','calibrate','predict','temporal','seal']);p.add_argument('dataset',nargs='?',choices=DATASETS);a=p.parse_args()
    if a.stage in ['calibrate','predict']:globals()[a.stage](a.dataset)
    else:globals()[a.stage]()
