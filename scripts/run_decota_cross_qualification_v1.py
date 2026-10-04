"""Frozen R6 cross-domain qualification: fresh H/evidence, real online LN chains."""
import sys,os,time,gc
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_cross_qualification_common_v1 import *

def prepare():
    from scripts.decota_actuation_scope_common_v1 import verify as parent_verify
    from scripts.run_spatial_ssl_gpu_v1 import config
    from methods.decota_final_simplified_v1.config import EXPERT_SNAPSHOT,EXPERT_SHA256
    parent_verify();selection=read(BASE/'R4_SELECTION.json');online=read(BASE/'ONLINE_LOCK.json')
    assert selection['scope']=='joint' and online['temporal']=='native'
    expected=dict(optimizer='adam',lr=.03,evidence='frame',authority=False,authority_source='frame')
    assert all(selection['config'][ds]==expected for ds in DATASETS)
    inputs={str((BASE/'R4_SELECTION.json').relative_to(ROOT)):sha(BASE/'R4_SELECTION.json'),str((BASE/'ONLINE_LOCK.json').relative_to(ROOT)):sha(BASE/'ONLINE_LOCK.json')}
    configs={}
    from scripts import tastvg_decota_c1_common_v1 as c1
    for ds in DATASETS:
        p=read(r1.BASE/ds/'PLAN.json');write(CROSS/ds/'PLAN.json',p)
        inputs[str((CROSS/ds/'PLAN.json').relative_to(ROOT))]=sha(CROSS/ds/'PLAN.json')
        q=config('vidstg_test' if ds=='vidstg' else 'hcstvg1_test');assert sha(ROOT/q.checkpoint)==q.checkpoint_sha256
        configs[ds]=dict(source_dataset=q.source_dataset,target_dataset=ds,checkpoint=q.checkpoint,checkpoint_sha256=q.checkpoint_sha256,
            adaptation=expected,scope='joint',steps=10,parameters=1792,LN_writeback=1/16,query_reset=True,optimizer_reset=True,temporal='native',expert_budget=1.)
        inputs[q.checkpoint]=q.checkpoint_sha256
        for row in p['rows']:
            assert row['ordinal']==row['pool_parent'];assert sha(row['input']['video_path'])==row['input']['video_sha256']
            for cond in p['conditions']:
                f=c1.POOL/ds/'capture'/cond/f"{row['pool_parent']:05}.json";inputs[str(f.relative_to(ROOT))]=sha(f)
    assert sha(ROOT/EXPERT_SNAPSHOT/'model.safetensors')==EXPERT_SHA256
    own=['scripts/decota_cross_qualification_common_v1.py','scripts/run_decota_cross_qualification_v1.py',
        'scripts/continue_decota_cross_qualification_v1.py','scripts/score_decota_cross_qualification_v1.py',
        'scripts/test_decota_cross_qualification_v1.py','protocols/decota_cross_qualification_v1.md',
        'scripts/run_spatial_ssl_gpu_v1.py','scripts/run_spatial_regression_alignment_v1.py',
        'vg_tta/tastvg_decota_c1_same_domain_v1.py','vg_tta/decota_actuation_scope_v1.py']
    write(CROSS/'RUNTIME_LOCK.json',dict(version='decota_cross_qualification_v1',time=time.time(),pins={f:sha(ROOT/f) for f in own},inputs=inputs,
        protected=r1.verify()['protected'],configurations=configs,expert_checkpoint_sha256=EXPERT_SHA256,
        unique_inputs=576,new_native_two_offset_captures=576,observation_requests_cap=2304,logical_readouts=2304,
        streams=['episodic','online_100'],historical_exposure=True,no_parameter_search=True,no_GT_prediction=True,
        scope_correction='R6 cross-domain explicitly authorized; prior assistant-added same-domain resource gate superseded, not a method reselection'))
    status(CROSS/'STATUS.json',dict(status='locked_pending_smoke',GT_read=False,predictions=0))
    archive('R6跨域范围修正后独立runtime及反向checkpoint/576输入/2304 episodic与online读出已锁，尚未预测')

def acquire_one(model,expert,ds,row,cond,frames,ids):
    import torch
    from scripts.run_tastvg_full_b1_experts_v1 import observation
    from scripts.run_spatial_ssl_gpu_v1 import frozen_forward
    from scripts import tastvg_decota_c1_common_v1 as c1
    from methods.decota_final_simplified_v1.observations import observations
    from methods.decota_final_simplified_v1.tensors import detached,state_hash
    from methods.decota_final_simplified_v1.objectives import prediction
    from vg_tta.tastvg_decota_c1_same_domain_v1 import NormalizedSpatialReplay
    shifted,pixel,spec=observation(row,cond,frames);oldrc=read(c1.POOL/ds/'capture'/cond/f"{row['pool_parent']:05}.json")
    assert pixel==oldrc['pixel_sha256'] and ids==row['frame_ids']
    batch,records,raw=frozen_forward(model,shifted,row)
    with torch.no_grad():
        views=[dict(H=raw.norm(v['prefix']),info={k:a for k,a in v['info'].items() if k not in ('encoded_feature','frames_cls','videos_cls')},vis_pos=v['vis_pos']) for v in raw.views]
    native=prediction(raw.zero['logits'],raw.zero['boxes'],records,ids)
    data=dict(views=detached(views,'cpu'),records=records,frame_ids=ids,pixel_sha256=pixel,prediction=detached(native,'cpu'),
        checkpoint_state_sha256=model.cross_source_state_sha256,GT_read=False)
    from scripts.run_tastvg_evidence_vulnerability_v1 import device_tree
    s=NormalizedSpatialReplay(model,device_tree(data,'cuda'))
    assert torch.equal(s.zero['boxes'],raw.zero['boxes']) and all(torch.equal(a,b) for a,b in zip(s.zero['logits'],raw.zero['logits']))
    ex=observations(expert,row['parses'],shifted,ids,native['indices'],audit=True)
    ep=dict(expert=detached(ex,'cpu'),pixel_sha256=pixel,spec=spec,native_indices=native['indices'],GT_read=False,
        matching_checkpoint_sha256=read(CROSS/'RUNTIME_LOCK.json')['configurations'][ds]['checkpoint_sha256'])
    return data,ep,batch,records,raw,s

def smoke():
    import torch
    from scripts.run_decota_optimizer_posterior_v1 import start_gpu
    from scripts.run_tastvg_decota_c1_same_domain_v1 import decode_for
    from methods.decota_final_simplified_v1.observations import SpatialExpert
    from methods.decota_final_simplified_v1.config import EXPERT_SNAPSHOT
    from methods.decota_final_simplified_v1.tensors import state_hash
    from methods.decota_final_simplified_v1.backbone import query_subject,full_prediction
    from vg_tta.decota_actuation_scope_v1 import fit_scope
    verify();handle=start_gpu();expert=SpatialExpert(ROOT/EXPERT_SNAPSHOT);stats=[];tick=time.time()
    try:
        eh=state_hash(expert.model.state_dict())
        for ds in DATASETS:
            model=model_for(ds);mh=state_hash(model.state_dict());model.cross_source_state_sha256=mh;p=read(CROSS/ds/'PLAN.json');cfg=read(CROSS/'RUNTIME_LOCK.json')['configurations'][ds]['adaptation']
            for parent in p['splits']['search']['orders']['order1'][:2]:
                row=p['rows'][parent];frames,ids=decode_for(ds)(row['input'])
                data,ex,batch,records,raw,s=acquire_one(model,expert,ds,row,'clean',frames,ids)
                z=fit_scope(s,s.initial,ex['expert'],ids,row['key'],cfg,'joint');original=fit_scope(raw,raw.initial,ex['expert'],ids,row['key'],cfg,'joint')
                assert z['selected_step']==original['selected_step'] and len(z['path'])==len(original['path'])
                for a,b in zip(z['path'],original['path']):
                    assert a['loss']==b['loss'] and torch.equal(a['boxes'],b['boxes']) and state_hash(a['state'])==state_hash(b['state'])
                    if 'update' in a:assert torch.equal(a['update']['gradient'],b['update']['gradient'])
                with query_subject(model,batch,row['parses']['subject']):
                    full_prediction(model,batch,ids,records,z['state'],dict(boxes=z['final'].cuda(),logits=s.zero['logits']))
                assert state_hash(model.state_dict())==mh
                commit(CROSS/ds/'capture'/'clean'/f'{parent:05}.pt',data);commit(CROSS/ds/'evidence'/'clean'/f'{parent:05}.pt',ex)
                stats.append(dict(dataset=ds,parent=parent,native_bitwise=True,all_steps_states_gradients_bitwise=True,full_reinsertion_bitwise=True,new_DINO=ex['expert']['new_DINO'],GT_read=False))
                del raw,s,batch,data,ex,frames,z,original;gc.collect();torch.cuda.empty_cache()
            del model;gc.collect();torch.cuda.empty_cache()
        assert state_hash(expert.model.state_dict())==eh
        write(CROSS/'SMOKE_ROOT_ACCEPTANCE.json',dict(status='pass',records=stats,GT_read=False,seconds=time.time()-tick,peak_memory_bytes=torch.cuda.max_memory_allocated()))
    finally:handle.close()

def capture(ds):
    import torch
    from scripts.run_decota_optimizer_posterior_v1 import start_gpu
    from scripts.run_tastvg_decota_c1_same_domain_v1 import decode_for
    from methods.decota_final_simplified_v1.observations import SpatialExpert
    from methods.decota_final_simplified_v1.config import EXPERT_SNAPSHOT
    from methods.decota_final_simplified_v1.tensors import state_hash
    verify();assert read(CROSS/'SMOKE_ROOT_ACCEPTANCE.json')['status']=='pass';handle=start_gpu();model=model_for(ds);expert=SpatialExpert(ROOT/EXPERT_SNAPSHOT)
    mh=state_hash(model.state_dict());model.cross_source_state_sha256=mh;eh=state_hash(expert.model.state_dict());p=read(CROSS/ds/'PLAN.json');done=calls=0;files={};tick=time.time()
    try:
        for row in p['rows']:
            frames=None
            for cond in p['conditions']:
                budget();f=CROSS/ds/'capture'/cond/f"{row['ordinal']:05}.pt";ef=CROSS/ds/'evidence'/cond/f"{row['ordinal']:05}.pt"
                if f.exists():data=checked(f);ex=checked(ef)
                else:
                    if frames is None:frames,ids=decode_for(ds)(row['input']);assert ids==row['frame_ids']
                    data,ex,batch,records,raw,s=acquire_one(model,expert,ds,row,cond,frames,ids)
                    commit(f,data);commit(ef,ex);del raw,s,batch
                assert data['checkpoint_state_sha256']==mh and ex['pixel_sha256']==data['pixel_sha256']
                done+=1;calls+=ex['expert']['new_DINO'];files[str(f.relative_to(CROSS))]=sha(f);files[str(ef.relative_to(CROSS))]=sha(ef)
                status(CROSS/ds/'CAPTURE_STATUS.json',dict(status='running',done=done,total=288,actual_DINO=calls,pid=os.getpid(),seconds=time.time()-tick,GT_read=False))
                print('CROSS_CAPTURE',ds,done,288,round(time.time()-tick,1),flush=True);del data,ex
                gc.collect();torch.cuda.empty_cache()
            del frames
        assert state_hash(model.state_dict())==mh and state_hash(expert.model.state_dict())==eh
        write(CROSS/ds/'CAPTURE_BARRIER.json',dict(status='sealed',unique_inputs=done,files=files,actual_DINO=calls,actual_native_two_offset_captures=done,
            seconds=time.time()-tick,smoke_seconds_shared=read(CROSS/'SMOKE_ROOT_ACCEPTANCE.json')['seconds'],source_state_sha256=mh,expert_state_sha256=eh,GT_read=False))
        status(CROSS/ds/'CAPTURE_STATUS.json',dict(status='completed',done=done,total=288,GT_read=False))
    finally:handle.close()

def execute(model,ds,row,cond,previous,cfg):
    import torch
    from vg_tta.tastvg_decota_c1_same_domain_v1 import NormalizedSpatialReplay
    from vg_tta.c1_enabling_tricks_v1 import TrickReplay,QUERY
    from vg_tta.decota_actuation_scope_v1 import fit_scope,commit_state
    from vg_tta.spatial_online_state_v1 import arrival
    from methods.decota_final_simplified_v1.tensors import detached,state_hash
    data,rc=cached(ds,row['ordinal'],cond);s=NormalizedSpatialReplay(model,data);source=detached(s.initial,'cpu');initial=arrival(s.initial,previous,'O-split');assert torch.count_nonzero(initial[QUERY])==0
    s.restore(initial);rp=TrickReplay(s,row['frame_ids'],row['key'],{})
    with torch.no_grad():before=detached(rp.values()['boxes'],'cpu')
    ef=CROSS/ds/'evidence'/cond/f"{row['ordinal']:05}.pt";ex=checked(ef);assert ex['pixel_sha256']==data['pixel_sha256']
    torch.cuda.synchronize();tick=time.perf_counter();z=fit_scope(s,initial,ex['expert'],row['frame_ids'],row['key'],cfg,'joint');torch.cuda.synchronize()
    committed=commit_state(detached(initial,'cpu'),z['write_proposal']);assert state_hash(s.state())==state_hash(initial) and torch.count_nonzero(committed[QUERY])==0
    return dict(source_state=source,initial=detached(initial,'cpu'),committed=committed,before=before,after=z['final'],fit=z,expert=True,
        native=detached(data['prediction'],'cpu'),interval=data['prediction']['physical_interval'],temporal_arm='native',fit_seconds=time.perf_counter()-tick,
        evidence_path=str(ef.relative_to(ROOT)),evidence_sha256=sha(ef),capture_path=str((CROSS/ds/'capture'/cond/f"{row['ordinal']:05}.pt").relative_to(ROOT)),
        capture_sha256=rc['sha256'],pixel_sha256=data['pixel_sha256'],GT_read=False,query_reset=True,optimizer_reset=True,LN_writeback=1/16)

def predict(ds):
    import torch
    from scripts.run_decota_optimizer_posterior_v1 import start_gpu
    from methods.decota_final_simplified_v1.tensors import state_hash
    verify();assert read(CROSS/ds/'CAPTURE_BARRIER.json')['unique_inputs']==288;handle=start_gpu();model=model_for(ds);mh=state_hash(model.state_dict())
    p=read(CROSS/ds/'PLAN.json');cfg=verify()['configurations'][ds]['adaptation'];files={};done=backwards=0;tick=time.time()
    try:
        for stream in ['episodic','online_100']:
            for split,sp in p['splits'].items():
                for cond in p['conditions']:
                    for order,seq in sp['orders'].items():
                        previous=prevsha=None
                        for at,parent in enumerate(seq):
                            budget();row=p['rows'][parent];x=execute(model,ds,row,cond,previous,cfg)
                            x.update(dataset=ds,stream=stream,split=split,condition=cond,order=order,arrival=at,parent=parent,previous_payload_sha256=prevsha)
                            f=CROSS/ds/stream/split/cond/order/f'{at:05}.pt';commit(f,x);files[str(f.relative_to(CROSS))]=sha(f)
                            if stream=='online_100':previous=x['committed'];prevsha=sha(f)
                            done+=1;backwards+=x['fit']['gradient_calls'];status(CROSS/ds/'STATUS.json',dict(status='running',stream=stream,done=done,total=1152,pid=os.getpid(),seconds=time.time()-tick,GT_read=False))
                            print('CROSS_ONLINE',ds,stream,done,1152,round(time.time()-tick,1),flush=True);del x
                            if done%32==0:gc.collect()
        assert state_hash(model.state_dict())==mh
        write(CROSS/ds/'PREDICTION_BARRIER.json',dict(status='sealed',files=files,arrivals=done,backward_calls=backwards,seconds=time.time()-tick,GT_read=False,source_state_sha256=mh,peak_memory_bytes=torch.cuda.max_memory_allocated()))
        status(CROSS/ds/'STATUS.json',dict(status='completed',done=done,total=1152,GT_read=False))
    finally:handle.close()

def seal():
    verify();files={}
    for ds in DATASETS:
        c=read(CROSS/ds/'CAPTURE_BARRIER.json');p=read(CROSS/ds/'PREDICTION_BARRIER.json');assert c['unique_inputs']==288 and p['arrivals']==1152;files.update(c['files']);files.update(p['files'])
    for f,h in files.items():assert sha(CROSS/f)==h
    write(CROSS/'GLOBAL_PREDICTION_BARRIER.json',dict(status='sealed',files=files,arrivals=2304,unique_inputs=576,GT_read=False,time=time.time()))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','smoke','capture','predict','seal']);p.add_argument('dataset',nargs='?',choices=DATASETS);a=p.parse_args()
    globals()[a.action](a.dataset) if a.action in ['capture','predict'] else globals()[a.action]()
