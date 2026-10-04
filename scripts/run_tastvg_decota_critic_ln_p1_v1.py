"""Frozen P0 critic with exactly C1 LN1/16 inheritance; no GT in workers."""
import sys, os, time, gc
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts.tastvg_decota_critic_ln_common_v1 import *


def prepare():
    if (BASE/'RUNTIME_LOCK.json').exists(): return verify()
    prior=p0.verify_seal(); pins=dict(p0.verify()['pins'])
    for f in sorted((p0.BASE/'revisions').glob('*.json')): pins.update(read(f)['pin_overrides'])
    own=['protocols/tastvg_decota_critic_ln_p1_v1.md','scripts/tastvg_decota_critic_ln_common_v1.py',
        'scripts/run_tastvg_decota_critic_ln_p1_v1.py','scripts/continue_tastvg_decota_critic_ln_p1_v1.py',
        'scripts/score_audit_tastvg_decota_critic_ln_p1_v1.py','scripts/report_tastvg_decota_critic_ln_p1_v1.py',
        'vg_tta/spatial_online_state_v1.py','vg_tta/spatial_consolidation_v1.py']
    pins.update({f:sha(ROOT/f) for f in own})
    inputs=dict(p0.verify()['inputs'])
    for f,h in prior['files'].items():
        path=p0.BASE/f; inputs[str(path.relative_to(ROOT))]=h
        inputs[str(path.with_suffix('.json').relative_to(ROOT))]=sha(path.with_suffix('.json'))
    for f in [p0.BASE/'GLOBAL_PREDICTION_BARRIER.json',p0.BASE/'RUNTIME_LOCK.json']:
        inputs[str(f.relative_to(ROOT))]=sha(f)
    for ds in DATASETS:
        write(BASE/ds/'PLAN.json',read(p0.BASE/ds/'PLAN.json'))
        inputs[str((BASE/ds/'PLAN.json').relative_to(ROOT))]=sha(BASE/ds/'PLAN.json')
    write(BASE/'RUNTIME_LOCK.json',dict(version='tastvg_decota_critic_ln_p1_v1',pins=pins,
        inputs=inputs,protected_registries=p0.verify()['protected_registries'],time=time.time(),
        online_arrivals=1152,source_inputs=576,LN_writeback_fraction=1/16,query_reset=True,Adam_reset=True,
        proposal_temperature=1,reward_temperature=1,parameters=1792,spatial_lr=.03,steps=10,
        temporal_updates=0,new_DINO=0,new_backbone=0,expert_every_query=True,historical_exposure=True))
    status(BASE/'STATUS.json',dict(status='locked_pending_cached_smoke',predictions=0,GT_read=False))
    archive('冻结匹配P1运行锁与1152到达，零预测，待无GT缓存生命周期smoke')
    print('PREPARED_P1',flush=True)


def start_gpu():
    import torch, numpy as np
    from scripts.run_tastvg_evidence_vulnerability_v2 import install_clean_loader
    install_clean_loader(); sys.addaudithook(guard)
    torch.set_num_threads(4); torch.manual_seed(20260920); np.random.seed(20260920)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    from scripts.run_final_simplification_v1 import lease
    return lease()


def execute(model,ds,row,cond,previous):
    import torch
    from scripts.run_tastvg_decota_c1_same_domain_v1 import data_input
    from scripts import tastvg_decota_c1_common_v1 as c1
    from vg_tta.tastvg_decota_c1_same_domain_v1 import NormalizedSpatialReplay
    from vg_tta.tastvg_decota_critic_p0_v1 import fit_critic
    from vg_tta.spatial_online_state_v1 import arrival,QUERY
    from vg_tta.spatial_consolidation_v1 import consolidate
    from vg_tta.c1_enabling_tricks_v1 import TrickReplay
    from methods.decota_final_simplified_v1.tensors import detached,state_hash
    data,rc=data_input(ds,row,cond); s=NormalizedSpatialReplay(model,data)
    source=detached(s.initial,'cpu'); initial=arrival(s.initial,previous,'O-split')
    assert torch.count_nonzero(initial[QUERY])==0
    s.restore(initial); replay=TrickReplay(s,row['frame_ids'],row['key'],{})
    with torch.no_grad(): values=replay.values(); before=detached(values['boxes'],'cpu')
    assert all(torch.equal(a,b) for a,b in zip(values['logits'],data['prediction']['logits']))
    ep=OLD/ds/'evidence'/cond/f"{row['ordinal']:05}.pt"; ex=c1.checked(ep)
    assert ex['pixel_sha256']==rc['pixel_sha256']
    torch.cuda.synchronize(); tick=time.perf_counter()
    fit=fit_critic(s,initial,ex['expert'],row['frame_ids'],row['key'])
    torch.cuda.synchronize(); seconds=time.perf_counter()-tick
    assert state_hash(s.state())==state_hash(initial)
    committed=detached(consolidate(initial,fit['state'],1/16),'cpu')
    assert torch.count_nonzero(committed[QUERY])==0
    pp=p0.BASE/ds/'predictions'/cond/f"{row['ordinal']:05}.pt"; prior=p0.checked(pp)
    assert state_hash(source)==state_hash(prior['source_state'])
    assert torch.equal(data['prediction']['boxes'].cpu(),prior['native']['boxes'])
    if previous is None:
        from scripts.run_tastvg_decota_critic_p0_v1 import compare_fit
        compare_fit(fit,prior['fits']['critic'])
        assert torch.equal(before,prior['native']['boxes'])
    return dict(source_state=source,initial=detached(initial,'cpu'),before=before,fit=fit,
        committed=committed,native=detached(data['prediction'],'cpu'),fit_seconds=seconds,
        evidence_path=str(ep.relative_to(ROOT)),evidence_sha256=sha(ep),
        P0_path=str(pp.relative_to(ROOT)),P0_sha256=sha(pp),pixel_sha256=rc['pixel_sha256'],
        GT_read=False,query_reset=True,Adam_reset=True,LN_writeback_fraction=1/16,temporal_updates=0)


def smoke():
    import torch
    from scripts.run_tastvg_decota_c1_same_domain_v1 import model_for
    from methods.decota_final_simplified_v1.tensors import state_hash
    verify(); handle=start_gpu(); records=[]; backwards=0;tick=time.time()
    try:
        for ds in DATASETS:
            model=model_for(ds);mh=state_hash(model.state_dict());p=read(BASE/ds/'PLAN.json')
            seq=p['splits']['search']['orders']['order1'];prev=None
            for at,parent in enumerate(seq[:2]):
                x=execute(model,ds,p['rows'][parent],'clean',prev)
                backwards+=x['fit']['gradient_calls']
                records.append(dict(dataset=ds,arrival=at,source_id=parent,
                    P0_all_step_bitwise=at==0,query_reset=True,Adam_reset=True,native_time_bitwise=True,
                    initial_state_sha256=state_hash(x['initial']),committed_state_sha256=state_hash(x['committed'])))
                prev=x['committed']
            assert state_hash(model.state_dict())==mh
            del model,x,prev;gc.collect();torch.cuda.empty_cache()
        write(BASE/'SMOKE_ROOT_ACCEPTANCE.json',dict(status='pass',records=records,GT_read=False,
            full_backbone_forwards=0,new_expert_calls=0,cached_fits=4,
            cached_backward_calls=backwards,seconds=time.time()-tick))
        status(BASE/'STATUS.json',dict(status='smoke_pass_pending_predictions',GT_read=False))
        print('P1_CACHED_SMOKE_PASS',flush=True)
    finally: handle.close()


def predict(ds):
    import torch
    from scripts.run_tastvg_decota_c1_same_domain_v1 import model_for
    from methods.decota_final_simplified_v1.tensors import state_hash
    verify();assert read(BASE/'SMOKE_ROOT_ACCEPTANCE.json')['status']=='pass'
    handle=start_gpu();model=model_for(ds);mh=state_hash(model.state_dict())
    p=read(BASE/ds/'PLAN.json');tick=time.time();done=backward=0;files={}
    try:
        for split,sp in p['splits'].items():
            for cond in p['conditions']:
                for order,seq in sp['orders'].items():
                    prev=None;prevsha=None
                    for at,parent in enumerate(seq):
                        budget();path=BASE/ds/'online'/split/cond/order/f'{at:05}.pt'
                        assert not path.exists(),'No overwrite/replay'
                        x=execute(model,ds,p['rows'][parent],cond,prev)
                        x.update(dataset=ds,split=split,condition=cond,order=order,arrival=at,
                            parent=parent,previous_payload_sha256=prevsha)
                        commit(path,x);prev=x['committed'];prevsha=sha(path)
                        files[str(path.relative_to(BASE))]=prevsha
                        done+=1;backward+=x['fit']['gradient_calls']
                        status(BASE/ds/'PREDICTION_STATUS.json',dict(status='running',done=done,total=576,
                            split=split,condition=cond,order=order,arrival=at,pid=os.getpid(),
                            backward_calls=backward,seconds=time.time()-tick,GT_read=False))
                        print('P1_PREDICT',ds,done,576,round(time.time()-tick,1),flush=True)
                        del x
                        if done%24==0:gc.collect()
        assert done==576 and state_hash(model.state_dict())==mh
        write(BASE/ds/'PREDICTION_BARRIER.json',dict(status='sealed',files=files,arrivals=done,
            source_unchanged=True,backwards=backward,seconds=time.time()-tick,
            peak_memory_bytes=torch.cuda.max_memory_allocated(),GT_read=False))
        status(BASE/ds/'PREDICTION_STATUS.json',dict(status='completed',done=done,total=576,GT_read=False))
    finally:handle.close()


def seal():
    verify();files={}
    for ds in DATASETS:
        b=read(BASE/ds/'PREDICTION_BARRIER.json');assert b['status']=='sealed' and b['arrivals']==576 and not b['GT_read']
        files.update(b['files'])
    assert len(files)==1152
    for f,h in files.items():assert sha(BASE/f)==h,f
    write(BASE/'GLOBAL_PREDICTION_BARRIER.json',dict(status='sealed',files=files,arrivals=1152,GT_read=False,time=time.time()))
    status(BASE/'STATUS.json',dict(status='predictions_sealed_pending_CPU',GT_read=False))
    print('SEALED_P1',len(files),flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','smoke','predict','seal']);p.add_argument('dataset',nargs='?',choices=DATASETS)
    a=p.parse_args()
    if a.stage=='prepare':prepare()
    elif a.stage=='smoke':smoke()
    elif a.stage=='seal':seal()
    else:predict(a.dataset)
