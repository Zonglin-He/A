"""Finite cached track/scope interventions; no GT imports in GPU workers."""
import sys,os,time,gc
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_actuation_scope_common_v1 import *

def prepare():
    assert read(r1.BASE/'R1_COMPLETION.json')['status']=='completed_verified_publication'
    decision=read(r1.PUB/'DECISION.json');arm=decision['spatial_optimizer_search_winner'];mode=arm.split('_',1)[1]
    config={ds:dict(optimizer=mode,lr=read(r1.BASE/ds/'CALIBRATION.json')['objectives']['all']['lr'] if mode=='sgd' else .03,
        evidence='frame',authority=False,authority_source='frame') for ds in DATASETS}
    write(BASE/'R2_SELECTION.json',dict(config=config,R1_decision_sha256=sha(r1.PUB/'DECISION.json'),selection_uses_confirmation=False,one_optimizer_family=True,
        T1=decision['T1'],temporal_posterior_qualification=decision['temporal_posterior_qualification']))
    files=['vg_tta/decota_track_critic_r3_v1.py','vg_tta/decota_actuation_scope_v1.py','scripts/test_decota_track_critic_r3_v1.py',
        'scripts/decota_actuation_scope_common_v1.py','scripts/run_decota_actuation_scope_v1.py','protocols/decota_actuation_scope_v1.md',
        'scripts/continue_decota_actuation_scope_v1.py','scripts/score_decota_actuation_scope_v1.py','scripts/run_decota_online_qualification_v1.py']
    files+=['scripts/score_decota_online_qualification_v1.py']
    files+=['scripts/test_decota_actuation_scope_v1.py']
    write(BASE/'RUNTIME_LOCK.json',dict(pins={f:sha(ROOT/f) for f in files},inputs={str((BASE/'R2_SELECTION.json').relative_to(ROOT)):sha(BASE/'R2_SELECTION.json')},
        time=time.time(),R1_verified=True,no_GT_prediction=True,stage_cohort_arrivals=1152,new_DINO=0,new_backbone=0,scientific_method_promoted=False))
    write(BASE/'CPU_LOCK.json',dict(score_sha256=sha(ROOT/'scripts/score_decota_actuation_scope_v1.py'),time=time.time()))
    status(BASE/'STATUS.json',dict(status='locked_pending_R3',GT_read=False));archive('R1公开核验后锁定单一优化器与R3/R4最小具体scope定义，尚未有新预测')

def predict(stage,ds):
    import torch
    from scripts.run_decota_optimizer_posterior_v1 import context,start_gpu
    from scripts.run_tastvg_decota_c1_same_domain_v1 import model_for
    from vg_tta.decota_actuation_scope_v1 import fit_scope
    verify();selection=read(BASE/'R2_SELECTION.json') if stage=='R3' else read(BASE/'R3_FINAL_SELECTION.json') if stage=='R4' else read(BASE/'R3_SELECTION.json')
    for f,h in read(BASE/stage/'STAGE_LOCK.json')['inputs'].items():assert sha(ROOT/f)==h
    assert read(BASE/'SMOKE_ROOT_ACCEPTANCE.json')['status']=='pass'
    handle=start_gpu();model=model_for(ds);p=read(r1.BASE/ds/'PLAN.json');files={};done=backwards=0;tick=time.time()
    try:
        for split,sp in p['splits'].items():
            for cond in p['conditions']:
                for order,seq in sp['orders'].items():
                    for at,parent in enumerate(seq):
                        budget();row=p['rows'][parent];base,x,ex,pp,ef=context(model,ds,row,cond,split,order,at);fits={}
                        configs={}
                        if stage=='R3':
                            for kind in ['frame_sum','track','track_authority']:
                                configs[kind]=dict(selection['config'][ds],evidence=kind,authority=kind=='track_authority',authority_source='path' if kind=='track_authority' else 'frame')
                        elif stage=='R3G':configs={'track_giou':dict(selection['config'][ds],evidence='track_giou')}
                        else:configs={s:selection['config'][ds] for s in ['joint','u_only','small_LN']}
                        for name,cfg in configs.items():
                            scope='joint' if stage in ['R3','R3G'] else name
                            fits[name]=fit_scope(base,x['initial'],ex,row['frame_ids'],row['key'],cfg,scope);backwards+=fits[name]['gradient_calls']
                        f=BASE/stage/ds/split/cond/order/f'{at:05}.pt'
                        commit(f,dict(dataset=ds,split=split,condition=cond,order=order,arrival=at,parent=parent,initial=x['initial'],before=x['before'],native=x['native'],fits=fits,
                            prestate_path=str(pp.relative_to(ROOT)),prestate_sha256=sha(pp),evidence_path=str(ef.relative_to(ROOT)),evidence_sha256=sha(ef),GT_read=False,own_online_trajectory=False))
                        files[str(f.relative_to(BASE))]=sha(f);done+=1
                        status(BASE/stage/ds/'STATUS.json',dict(status='running',done=done,total=576,pid=os.getpid(),seconds=time.time()-tick,GT_read=False))
                        print('ACTUATION',stage,ds,done,576,round(time.time()-tick,1),flush=True)
                        del base,x,fits
                        if done%16==0:gc.collect()
        write(BASE/stage/ds/'PREDICTION_BARRIER.json',dict(status='sealed',files=files,arrivals=done,backward_calls=backwards,seconds=time.time()-tick,GT_read=False))
    finally:handle.close()

def seal(stage):
    verify();files={}
    for ds in DATASETS:
        b=read(BASE/stage/ds/'PREDICTION_BARRIER.json');assert b['arrivals']==576 and not b['GT_read'];files.update(b['files'])
    for f,h in files.items():assert sha(BASE/f)==h
    write(BASE/stage/'GLOBAL_PREDICTION_BARRIER.json',dict(status='sealed',files=files,arrivals=1152,time=time.time(),GT_read=False))

def smoke():
    import torch
    from scripts.run_decota_optimizer_posterior_v1 import context,start_gpu
    from scripts.run_tastvg_decota_c1_same_domain_v1 import model_for
    from vg_tta.decota_actuation_scope_v1 import fit_scope
    from methods.decota_final_simplified_v1.tensors import state_hash
    verify();handle=start_gpu();records=[]
    try:
        sel=read(BASE/'R2_SELECTION.json')
        for ds in DATASETS:
            model=model_for(ds);modelsha=state_hash(model.state_dict());p=read(r1.BASE/ds/'PLAN.json')
            for at,parent in enumerate(p['splits']['search']['orders']['order1'][:2]):
                row=p['rows'][parent];base,x,ex,pp,ef=context(model,ds,row,'clean','search','order1',at)
                z=fit_scope(base,x['initial'],ex,row['frame_ids'],row['key'],sel['config'][ds],'joint')
                previous=r1.checked(r1.BASE/ds/'spatial'/'search'/'clean'/'order1'/f'{at:05}.pt')['fits']['all_'+sel['config'][ds]['optimizer']]
                assert torch.equal(z['final'],previous['final']);assert state_hash(z['state'])==state_hash(previous['state'])
                records.append(dict(dataset=ds,arrival=at,final_bitwise_R1=True,state_bitwise_R1=True))
            assert state_hash(model.state_dict())==modelsha
            del model;torch.cuda.empty_cache()
        write(BASE/'SMOKE_ROOT_ACCEPTANCE.json',dict(status='pass',records=records,GT_read=False,new_backbone=0,new_DINO=0))
    finally:handle.close()

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','smoke','predict','seal']);p.add_argument('stage',nargs='?',choices=['R3','R3G','R4']);p.add_argument('dataset',nargs='?',choices=DATASETS);a=p.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='smoke':smoke()
    elif a.action=='seal':seal(a.stage)
    else:predict(a.stage,a.dataset)
