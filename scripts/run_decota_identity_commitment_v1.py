"""Finite matched MAP stage followed by separate evolving LN streams."""
import sys,os,time,gc,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_identity_common_v1 import *

def prepare():
    assert not (BASE/'RUNTIME_LOCK.json').exists()
    previous.verify();pins=dict(previous.verify()['pins'])
    for f in sorted((previous.BASE/'revisions').glob('*.json')):pins.update(read(f)['pin_overrides'])
    own=['vg_tta/decota_identity_commitment_v1.py','scripts/decota_identity_common_v1.py','scripts/run_decota_identity_commitment_v1.py','scripts/test_decota_identity_commitment_v1.py','protocols/decota_identity_commitment_v1.md']
    pins.update({f:sha(ROOT/f) for f in own});inputs=dict(read(previous.BASE/'RUNTIME_LOCK.json')['inputs'])
    inputs.update(read(r1.BASE/'RUNTIME_LOCK.json')['inputs']);cells=[]
    def pin(f):inputs[str(f.relative_to(ROOT))]=sha(f)
    for ds in DATASETS:
        p=read(r1.BASE/ds/'PLAN.json');write(BASE/ds/'PLAN.json',p);pin(BASE/ds/'PLAN.json')
        for split,sp in p['splits'].items():
            for cond in p['conditions']:
                for order,seq in sp['orders'].items():
                    for at,parent in enumerate(seq):
                        pf=r1.prior.BASE/ds/'online'/split/cond/order/f'{at:05}.pt'
                        rf=r1.BASE/ds/'spatial'/split/cond/order/f'{at:05}.pt'
                        tf=previous.BASE/'R3'/ds/split/cond/order/f'{at:05}.pt'
                        for f in [pf,rf,tf]:pin(f);pin(f.with_suffix('.json'))
                        cells.append(dict(dataset=ds,split=split,condition=cond,order=order,arrival=at,parent=parent,prestate=str(pf.relative_to(ROOT)),top1_ref=str(rf.relative_to(ROOT)),marginal_ref=str(tf.relative_to(ROOT))))
    write(BASE/'COHORT.json',dict(cells=cells));pin(BASE/'COHORT.json')
    assert len(cells)==1152
    write(BASE/'RUNTIME_LOCK.json',dict(pins=pins,inputs=inputs,protected=read(r1.BASE/'RUNTIME_LOCK.json')['protected'],
        time=time.time(),arrivals=1152,arms=['top1','marginal','map_contrastive'],Adam_lr=.03,parameters=1792,steps=10,tau=1,
        contrastive_changes_loss=True,old_admission_preserved_top1=True,observation_positions_unchanged=True,no_GT_online=True))
    status(BASE/'STATUS.json',dict(status='prepared_pending_smoke',predictions=0,GT_read=False))
    archive('隔离三臂运行锁已实现；预测为零，纯CPU合同及无GT bitwise smoke待执行')

def references(c):
    a=r1.checked(ROOT/c['top1_ref'])['fits']['top1_adam'];b=previous.checked(ROOT/c['marginal_ref'])['fits']['track']
    return a,b

def compare(a,b):
    import torch
    assert a['selected_step']==b['selected_step'] and len(a['path'])==len(b['path'])
    assert torch.equal(a['final'],b['final'])
    for x,y in zip(a['path'],b['path']):
        assert x['loss']==y['loss'] and torch.equal(x['boxes'],y['boxes'])
        for n,v in x['state'].items():assert torch.equal(v,y['state'][n])
        if 'update' in x:
            for k in ['gradient','raw']:assert torch.equal(x['update'][k],y['update'][k])

def smoke():
    import torch
    from scripts.run_decota_optimizer_posterior_v1 import start_gpu,context
    from scripts.run_tastvg_decota_c1_same_domain_v1 import model_for
    from vg_tta.decota_identity_commitment_v1 import fit
    from methods.decota_final_simplified_v1.tensors import state_hash
    verify();lease=start_gpu();rows=[]
    try:
        for ds in DATASETS:
            model=model_for(ds);mh=state_hash(model.state_dict());p=read(BASE/ds/'PLAN.json')
            cells=[c for c in read(BASE/'COHORT.json')['cells'] if c['dataset']==ds and c['split']=='search' and c['condition']=='clean' and c['order']=='order1'][:2]
            for c in cells:
                row=p['rows'][c['parent']];base,x,ex,_,_=context(model,ds,row,c['condition'],c['split'],c['order'],c['arrival']);old=references(c)
                for arm,ref in zip(['top1','marginal'],old):compare(fit(base,x['initial'],ex,row['frame_ids'],row['key'],arm),ref)
                z=fit(base,x['initial'],ex,row['frame_ids'],row['key'],'map_contrastive')
                assert torch.equal(z['before'],x['before']) and state_hash(base.state())==state_hash(x['initial'])
                rows.append(dict(dataset=ds,parent=c['parent'],matched_control_bitwise=True,MAP_path_fixed=True,MAP_steps=z['gradient_calls']))
            assert state_hash(model.state_dict())==mh;del model,base,x;gc.collect();torch.cuda.empty_cache()
        write(BASE/'SMOKE_ROOT_ACCEPTANCE.json',dict(status='pass',records=rows,GT_read=False,new_expert=0,new_backbone=0,time=time.time()))
    finally:lease.close()

def predict(ds):
    import torch
    from scripts.run_decota_optimizer_posterior_v1 import start_gpu,context
    from scripts.run_tastvg_decota_c1_same_domain_v1 import model_for
    from vg_tta.decota_identity_commitment_v1 import fit
    from methods.decota_final_simplified_v1.tensors import state_hash
    verify();assert read(BASE/'SMOKE_ROOT_ACCEPTANCE.json')['status']=='pass';lease=start_gpu();model=model_for(ds);mh=state_hash(model.state_dict())
    p=read(BASE/ds/'PLAN.json');files={};tick=time.time();backward=0
    try:
        for c in [c for c in read(BASE/'COHORT.json')['cells'] if c['dataset']==ds]:
            budget();row=p['rows'][c['parent']];base,x,ex,pf,ef=context(model,ds,row,c['condition'],c['split'],c['order'],c['arrival'])
            z=fit(base,x['initial'],ex,row['frame_ids'],row['key'],'map_contrastive');assert torch.equal(z['before'],x['before'])
            f=BASE/'matched'/ds/c['split']/c['condition']/c['order']/f"{c['arrival']:05}.pt"
            commit(f,dict(c,fit=z,native=x['native'],initial=x['initial'],before=x['before'],evidence_path=str(ef.relative_to(ROOT)),evidence_sha256=sha(ef),prestate_sha256=sha(pf),GT_read=False))
            files[str(f.relative_to(BASE))]=sha(f);backward+=z['gradient_calls']
            status(BASE/ds/'STATUS.json',dict(status='matched_running',done=len(files),total=576,pid=os.getpid(),seconds=time.time()-tick,GT_read=False))
            if len(files)%24==0:print('IDENTITY_MATCHED',ds,len(files),576,round(time.time()-tick,1),flush=True);gc.collect()
        assert state_hash(model.state_dict())==mh
        write(BASE/ds/'MATCHED_BARRIER.json',dict(status='sealed',files=files,arrivals=576,backward_calls=backward,seconds=time.time()-tick,GT_read=False))
    finally:lease.close()

def seal():
    verify();files={}
    for ds in DATASETS:
        z=read(BASE/ds/'MATCHED_BARRIER.json');assert z['arrivals']==576 and not z['GT_read'];files.update(z['files'])
    for f,h in files.items():assert sha(BASE/f)==h
    write(BASE/'MATCHED_GLOBAL_BARRIER.json',dict(status='sealed',files=files,arrivals=1152,GT_read=False,time=time.time()))

def online_prepare():
    verify();choice=read(BASE/'SEARCH_SELECTION.json');assert choice['uses_confirmation'] is False
    schedules={};streams=['episodic','online100']+[f'seed{s}_budget{r}' for s in range(5) for r in [25,50]]
    for ds in DATASETS:
        p=read(BASE/ds/'PLAN.json');schedules[ds]={}
        for split,sp in p['splits'].items():
            z={}
            for s in range(5):
                seq=sp['orders']['order1'];rank=sorted(seq,key=lambda parent:hashlib.sha256(f'decota_identity_budget_v1:{s}:{ds}:{split}:{p["rows"][parent]["key"]}'.encode()).hexdigest())
                for r in [25,50]:z[f'seed{s}_budget{r}']=rank[:len(rank)*r//100]
            z['episodic']=sp['orders']['order1'];z['online100']=sp['orders']['order1'];schedules[ds][split]=z
    write(BASE/'ONLINE_LOCK.json',dict(arm=choice['arm'],selection_sha256=sha(BASE/'SEARCH_SELECTION.json'),streams=streams,schedules=schedules,
        arrivals=13824,query_reset=True,Adam_reset=True,LN_writeback=1/16,temporal='native',new_expert=0,new_backbone=0,time=time.time()))
    archive('三臂全封存评分与开发选择已完成；选择'+choice['arm']+'，确认不参与选择；十二独立流及五组嵌套专家roster已锁，待实际执行')

def online(ds):
    import torch
    from scripts.run_decota_optimizer_posterior_v1 import start_gpu
    from scripts.run_tastvg_decota_c1_same_domain_v1 import model_for,data_input
    from vg_tta.tastvg_decota_c1_same_domain_v1 import NormalizedSpatialReplay
    from vg_tta.c1_enabling_tricks_v1 import TrickReplay,QUERY
    from vg_tta.spatial_online_state_v1 import arrival
    from vg_tta.decota_actuation_scope_v1 import commit_state
    from vg_tta.decota_identity_commitment_v1 import fit
    from methods.decota_final_simplified_v1.tensors import detached,state_hash
    verify();lock=read(BASE/'ONLINE_LOCK.json');p=read(BASE/ds/'PLAN.json');lease=start_gpu();model=model_for(ds);mh=state_hash(model.state_dict())
    files={};tick=time.time();backward=0;episodic={}
    try:
        for stream in lock['streams']:
            for split,sp in p['splits'].items():
                chosen=lock['schedules'][ds][split][stream]
                for cond in p['conditions']:
                    for order,seq in sp['orders'].items():
                        prev=None;prevsha=None
                        for at,parent in enumerate(seq):
                            budget();row=p['rows'][parent];ep=c1.BASE/ds/'evidence'/cond/f'{parent:05}.pt';expert=parent in chosen;key=(split,cond,parent)
                            if stream=='episodic' and key in episodic:
                                ref=episodic[key];x=checked(ref);x={**x,'reused_episodic':str(ref.relative_to(ROOT)),'reuse_sha256':sha(ref),'actual_cached_backward_calls':0}
                            else:
                                data,rc=data_input(ds,row,cond);base=NormalizedSpatialReplay(model,data);source=detached(base.initial,'cpu');initial=arrival(base.initial,prev,'O-split')
                                assert torch.count_nonzero(initial[QUERY])==0;base.restore(initial);rp=TrickReplay(base,row['frame_ids'],row['key'],{})
                                with torch.no_grad():before=detached(rp.values()['boxes'],'cpu')
                                z=None;wall=time.perf_counter()
                                if expert:
                                    ex=c1.checked(ep);assert ex['pixel_sha256']==rc['pixel_sha256'];z=fit(base,initial,ex['expert'],row['frame_ids'],row['key'],lock['arm'])
                                    after=z['final'];committed=commit_state(detached(initial,'cpu'),z['state'])
                                else:after=before;committed=detached(initial,'cpu')
                                assert state_hash(base.state())==state_hash(initial)
                                x=dict(source_state=source,initial=detached(initial,'cpu'),committed=committed,before=before,after=after,fit=z,native=detached(data['prediction'],'cpu'),
                                    interval=data['prediction']['physical_interval'],expert=expert,evidence_path=str(ep.relative_to(ROOT)),evidence_sha256=sha(ep),pixel_sha256=rc['pixel_sha256'],
                                    fit_seconds=time.perf_counter()-wall,GT_read=False,actual_cached_backward_calls=0 if z is None else z['gradient_calls'])
                            x.update(dataset=ds,stream=stream,split=split,condition=cond,order=order,arrival=at,parent=parent,previous_payload_sha256=prevsha,query_reset=True,Adam_reset=True)
                            f=BASE/'online'/ds/stream/split/cond/order/f'{at:05}.pt';commit(f,x);files[str(f.relative_to(BASE))]=sha(f);backward+=x['actual_cached_backward_calls']
                            if stream=='episodic':episodic[key]=f
                            else:prev=x['committed'];prevsha=sha(f)
                            status(BASE/ds/'STATUS.json',dict(status='online_running',done=len(files),total=6912,stream=stream,pid=os.getpid(),seconds=time.time()-tick,GT_read=False))
                            if len(files)%48==0:print('IDENTITY_ONLINE',ds,stream,len(files),6912,round(time.time()-tick,1),flush=True);gc.collect()
        assert state_hash(model.state_dict())==mh
        write(BASE/ds/'ONLINE_BARRIER.json',dict(status='sealed',files=files,arrivals=6912,backward_calls=backward,seconds=time.time()-tick,GT_read=False))
    finally:lease.close()

def online_seal():
    verify();files={}
    for ds in DATASETS:
        z=read(BASE/ds/'ONLINE_BARRIER.json');assert z['arrivals']==6912;files.update(z['files'])
    for f,h in files.items():assert sha(BASE/f)==h
    write(BASE/'ONLINE_GLOBAL_BARRIER.json',dict(status='sealed',arrivals=13824,files=files,GT_read=False,time=time.time()))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','smoke','predict','seal','online_prepare','online','online_seal']);p.add_argument('dataset',nargs='?',choices=DATASETS);a=p.parse_args()
    globals()[a.action](a.dataset) if a.action in ['predict','online'] else globals()[a.action]()
