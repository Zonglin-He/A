"""Frozen selected mechanisms, actual LN trajectories and nested budgets."""
import sys,os,time,gc,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_actuation_scope_common_v1 import *

def prepare():
    verify();sel=read(BASE/'R4_SELECTION.json');plans={}
    for ds in DATASETS:
        p=read(r1.BASE/ds/'PLAN.json');schedules={}
        for split,sp in p['splits'].items():
            seq=sp['orders']['order1'];rank=sorted(seq,key=lambda parent:hashlib.sha256(('decota_budget_v1:'+ds+':'+split+':'+p['rows'][parent]['key']).encode()).hexdigest())
            schedules[split]={str(rate):rank[:int(len(rank)*rate)] for rate in [.25,.5,1.]}
            assert set(schedules[split]['0.25'])<=set(schedules[split]['0.5'])<=set(schedules[split]['1.0'])
        plans[ds]=dict(config=sel['config'][ds],scope=sel['scope'],schedules=schedules,streams=['episodic','budget_0.25','budget_0.5','budget_1.0'])
    # Posterior eligibility is a predeclared resource gate; arm selection itself uses search only.
    d=read(r1.PUB/'DECISION.json');ss=read(r1.PUB/'TEMPORAL_SUMMARY.json');qual=[a for a,v in d['temporal_posterior_qualification'].items() if v]
    temporal=max(qual,key=lambda a:sum(ss[ds]['search'][a]['corruption']['metrics']['vs_native_v']['mean'] for ds in DATASETS)) if qual else 'native'
    if qual:assert (BASE/'T1_COMPLETION.json').exists(),'Qualified temporal posterior requires root T1/occupancy implementation before final combination'
    write(BASE/'ONLINE_LOCK.json',dict(plans=plans,temporal=temporal,T1_required=bool(qual),selection_source_sha256=sha(BASE/'R4_SELECTION.json'),
        native_WHEN_if_unqualified=True,query_reset=True,optimizer_reset=True,LN_writeback=1/16,temporal_reset=True,
        predictions=4608,unique_inputs=576,no_GT_schedule=True,time=time.time(),new_backbone=0,new_DINO=0))
    archive('R5/R6同域四条独立流及嵌套25/50/100来源hash专家位置锁定，零新预测；时间采用已资格化posterior或Native回退')

def execute(model,ds,row,cond,previous,cfg,scope,expert):
    import torch
    from scripts.run_tastvg_decota_c1_same_domain_v1 import data_input
    from scripts import tastvg_decota_c1_common_v1 as c1
    from vg_tta.tastvg_decota_c1_same_domain_v1 import NormalizedSpatialReplay
    from vg_tta.c1_enabling_tricks_v1 import TrickReplay,QUERY
    from vg_tta.decota_actuation_scope_v1 import fit_scope,commit_state
    from methods.decota_final_simplified_v1.tensors import detached,state_hash
    from vg_tta.spatial_online_state_v1 import arrival
    data,rc=data_input(ds,row,cond);s=NormalizedSpatialReplay(model,data);source=detached(s.initial,'cpu');initial=arrival(s.initial,previous,'O-split')
    assert torch.count_nonzero(initial[QUERY])==0;s.restore(initial);rp=TrickReplay(s,row['frame_ids'],row['key'],{})
    with torch.no_grad():before=detached(rp.values()['boxes'],'cpu')
    ep=c1.BASE/ds/'evidence'/cond/f"{row['ordinal']:05}.pt";ex=c1.checked(ep);assert ex['pixel_sha256']==rc['pixel_sha256']
    fit=None;tick=time.perf_counter()
    if expert:
        fit=fit_scope(s,initial,ex['expert'],row['frame_ids'],row['key'],cfg,scope);after=fit['final'];committed=commit_state(detached(initial,'cpu'),fit['write_proposal'])
    else:after=before;committed=detached(initial,'cpu')
    assert state_hash(s.state())==state_hash(initial) and torch.count_nonzero(committed[QUERY])==0
    # No additional temporal network execution; episodic posterior readout sealed in R1.
    tp=r1.BASE/ds/'temporal'/cond/f"{row['ordinal']:05}.pt";time_data=r1.checked(tp);temporal=read(BASE/'ONLINE_LOCK.json')['temporal']
    interval=time_data['intervals'][temporal] if expert else time_data['intervals']['native']
    return dict(source_state=source,initial=detached(initial,'cpu'),committed=committed,before=before,after=after,fit=fit,expert=expert,
        native=detached(data['prediction'],'cpu'),interval=interval,temporal_arm=temporal if expert else 'native',fit_seconds=time.perf_counter()-tick,
        evidence_path=str(ep.relative_to(ROOT)),evidence_sha256=sha(ep),pixel_sha256=rc['pixel_sha256'],GT_read=False,
        query_reset=True,optimizer_reset=True,LN_writeback=1/16)

def reuse_preserved(ds,row,cond,split,order,at,stream):
    import torch,math
    from methods.decota_final_simplified_v1.tensors import detached
    from vg_tta.decota_actuation_scope_v1 import commit_state
    if stream=='budget_1.0':
        path=r1.prior.BASE/ds/'online'/split/cond/order/f'{at:05}.pt';old=r1.prior.checked(path);z=old['fit'];before=old['before'];initial=old['initial'];native=old['native']
    else:
        path=r1.prior.p0.BASE/ds/'predictions'/cond/f"{row['ordinal']:05}.pt";old=r1.prior.p0.checked(path);z=old['fits']['critic'];before=old['native']['boxes'];initial=old['source_state'];native=old['native']
    cfg=read(BASE/'ONLINE_LOCK.json')['plans'][ds]['config'];a=[]
    for m in z['frame_metadata']:
        a.append(1. if m['proposals']==1 else max(0.,min(1.,1-m['evidence_entropy']/math.log(m['proposals']))))
    auth=sum(a)/len(a) if a else 0.;history=z['path'];names=list(initial)
    for h in history:
        if 'update' in h:h['update'].update(names=names,optimizer='adam',lr=.03)
    fit=dict(z,scope='joint',config=cfg,before=before,initial=initial,state=z['state'],proposal_state=z['state'],write_proposal=z['state'],
        active_parameters=1792,slow_LN=None,empty=z['skipped'],authority=auth,metadata=dict(valid_frames=len(z['frame_metadata'])),
        selected_step=z['selected_step'],GT_used=False)
    tp=r1.checked(r1.BASE/ds/'temporal'/cond/f"{row['ordinal']:05}.pt");ta=read(BASE/'ONLINE_LOCK.json')['temporal'];iv=tp['intervals'][ta]
    from scripts import tastvg_decota_c1_common_v1 as c1
    ep=c1.BASE/ds/'evidence'/cond/f"{row['ordinal']:05}.pt"
    return dict(source_state=old['source_state'],initial=initial,committed=commit_state(initial,z['state']),before=before,after=z['final'],fit=fit,expert=True,
        native=native,interval=iv,temporal_arm=ta,fit_seconds=0.,evidence_path=str(ep.relative_to(ROOT)),evidence_sha256=sha(ep),
        reused_saved_stream=True,reuse_path=str(path.relative_to(ROOT)),reuse_sha256=sha(path),actual_cached_backwards=0,GT_read=False,query_reset=True,optimizer_reset=True,LN_writeback=1/16)

def predict(ds):
    import torch
    from scripts.run_decota_optimizer_posterior_v1 import start_gpu
    from scripts.run_tastvg_decota_c1_same_domain_v1 import model_for
    from methods.decota_final_simplified_v1.tensors import state_hash
    verify();lock=read(BASE/'ONLINE_LOCK.json');plan=lock['plans'][ds];p=read(r1.BASE/ds/'PLAN.json');handle=start_gpu();model=model_for(ds);mh=state_hash(model.state_dict())
    files={};done=backward=0;tick=time.time()
    preserved=plan['scope']=='joint' and plan['config']==dict(optimizer='adam',lr=.03,evidence='frame',authority=False,authority_source='frame')
    try:
        for stream in plan['streams']:
            for split,sp in p['splits'].items():
                rate=1. if stream=='episodic' else float(stream.split('_')[1]);chosen=plan['schedules'][split][str(rate)]
                for cond in p['conditions']:
                    for order,seq in sp['orders'].items():
                        prev=None;prevsha=None
                        for at,parent in enumerate(seq):
                            budget();row=p['rows'][parent];expert=parent in chosen
                            x=reuse_preserved(ds,row,cond,split,order,at,stream) if preserved and stream in ['episodic','budget_1.0'] else execute(model,ds,row,cond,prev,plan['config'],plan['scope'],expert)
                            x.update(dataset=ds,stream=stream,split=split,condition=cond,order=order,arrival=at,parent=parent,previous_payload_sha256=prevsha)
                            f=BASE/'online'/ds/stream/split/cond/order/f'{at:05}.pt';commit(f,x);files[str(f.relative_to(BASE))]=sha(f)
                            if stream!='episodic':prev=x['committed'];prevsha=sha(f)
                            done+=1;backward+=x.get('actual_cached_backwards',0 if x['fit'] is None else x['fit']['gradient_calls'])
                            status(BASE/'online'/ds/'STATUS.json',dict(status='running',done=done,total=2304,stream=stream,pid=os.getpid(),seconds=time.time()-tick,GT_read=False))
                            print('ONLINE_BUDGET',ds,stream,done,2304,round(time.time()-tick,1),flush=True)
                            del x
                            if done%32==0:gc.collect()
        assert state_hash(model.state_dict())==mh
        write(BASE/'online'/ds/'PREDICTION_BARRIER.json',dict(status='sealed',files=files,arrivals=done,backward_calls=backward,seconds=time.time()-tick,GT_read=False,new_DINO=0,new_backbone=0))
    finally:handle.close()

def seal():
    verify();files={}
    for ds in DATASETS:
        b=read(BASE/'online'/ds/'PREDICTION_BARRIER.json');assert b['arrivals']==2304 and not b['GT_read'];files.update(b['files'])
    for f,h in files.items():assert sha(BASE/f)==h
    write(BASE/'online'/'GLOBAL_PREDICTION_BARRIER.json',dict(status='sealed',arrivals=4608,files=files,time=time.time(),GT_read=False))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','predict','seal']);p.add_argument('dataset',nargs='?',choices=DATASETS);a=p.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='seal':seal()
    else:predict(a.dataset)
