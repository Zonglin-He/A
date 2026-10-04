"""Source-bound native captures and predict-before-update clean A streams."""
import sys,time,gc,traceback,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tastvg_cross_domain_common_v1 import *

def run(direction, stage):
    p=verify(direction);out=BASE/direction;cfg=p['params'];tick=time.monotonic()
    lease=actor=None;done=0;counts=collections.Counter()
    try:
        import torch,numpy as np
        from scripts.run_tastvg_evidence_vulnerability_v2 import install_clean_loader
        from scripts.run_tastvg_evidence_vulnerability_v1 import device_tree
        from scripts.run_final_simplification_v1 import lease as gpu_lease
        from methods.decota_final_simplified_v1.tensors import state_hash,detached
        from vg_tta.tastvg_best_full_method_v1 import (OnlineMethod,central_with_candidates,
            fast_rerank,rollout_states)
        from vg_tta.tastvg_spatial_rank_s11_v1 import SpatialActor
        from vg_tta.tastvg_native_spatial_rollout_s05_v1 import central_state,reinsert
        from scripts.run_tastvg_paper48_p5_online_v1 import source_capture,compact_prediction
        from scripts.c1_controlled_corruption_v1 import pixelhash
        for name in ['SPATIAL','TEMPORAL']:
            assert read(out/'experts'/f'{name}_BARRIER.json')['cells']==len(p['expert_needed'])
        subjects=read(out/'SUBJECT_BARRIER.json');assert subjects['count']==p['queries']
        sys.addaudithook(guard);install_clean_loader();decode=bind_decode(p['target_dataset'])
        lease=gpu_lease();torch.set_num_threads(4);torch.manual_seed(20260929)
        np.random.seed(20260929);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        source=p['target_dataset'] if stage=='target_reference' else p['source_dataset']
        model=model_load(source).eval().requires_grad_(False)
        mh=state_hash(model.state_dict());assert mh==MODEL_SHA[source]
        center=central_state(model);states,spec,basis=rollout_states(center,cfg['rho'],cfg['direction_count'])
        deltas=[{n:v-center[n] for n,v in st.items()} for st in states]
        def factory(fast=True):
            return OnlineMethod(model,deltas,fast=fast,**{k:cfg[k] for k in
                ['lr','teacher_temperature','student_temperature','steps']})
        if stage=='capture' and not (out/'SUPPORT.json').exists():
            commit(out/'PARAMETER_SUPPORT.pt',detached(dict(center=center,deltas=deltas,
                basis=basis,spec=spec),'cpu'))
            write(out/'SUPPORT.json',dict(spec=spec,center_sha256=state_hash(center),
                checkpoint_state_sha256=mh,source_dataset=source,params=cfg))
        def specialist(parent,name,pixel):
            f=out/'experts'/name/'clean'/f'{parent:05}.json';r=read(f)
            assert r['pixel_sha256']==pixel
            assert sha(out/'experts'/r['cache'])==r['cache_sha256']
            return {**load(out/'experts'/r['cache']),'pixel_sha256':pixel}
        def subject(row):
            f=out/'subjects'/f"{row['parent']:05}.json";assert sha(f)==subjects['files'][f.name]
            parsed=read(f)['parses']['subject']
            value=parsed['subject'] if isinstance(parsed,dict) else parsed
            assert isinstance(value,str) and value
            return value
        if stage in ['capture','target_reference']:
            actor=SpatialActor(model) if stage=='capture' else None
            for row in p['rows']:
                parent=row['parent'];f=out/stage/f'{parent:05}.pt'
                if f.exists(): checked(f);done+=1;counts['receipt_resumed_inputs']+=1;continue
                budget();frames,ids=decode(row['input']);assert ids==row['frame_ids'];pixel=pixelhash(frames)
                data=source_capture(model,frames,row,subject(row))
                counts['new_full_backbone_offset_forwards']+=2
                data.update(pixel_sha256=pixel,checkpoint_state_sha256=mh)
                if stage=='capture':
                    d=device_tree(data,'cuda')
                    if parent in p['expert_needed']:
                        _,_,native,layers,tc=central_with_candidates(actor,d)
                        counts['source_suffix_values_with_temporal_support']+=1
                        assert torch.equal(native['boxes'],data['prediction']['boxes'])
                        assert native['indices']==data['prediction']['indices']
                        fast,td=fast_rerank(native,tc,specialist(parent,'temporal',pixel))
                        data.update(zero_temporal=td,fast_only=compact_prediction(fast))
                    else:
                        _,_,native=actor.values(d)
                        counts['source_suffix_values']+=1
                        assert torch.equal(native['boxes'],data['prediction']['boxes'])
                        assert native['indices']==data['prediction']['indices']
                        data.update(zero_temporal=None,fast_only=compact_prediction(native))
                    commit(f,data,parent=parent,pixel_sha256=pixel,model_state_sha256=mh)
                else:
                    src=checked(out/'capture'/f'{parent:05}.pt')
                    assert src['pixel_sha256']==pixel
                    commit(f,dict(prediction=compact_prediction(data['prediction']),
                        pixel_sha256=pixel,model_state_sha256=mh,GT_read=False),parent=parent)
                done+=1
                if done%12==0 or done==1:
                    status(out/f'{stage.upper()}_STATUS.json',dict(status='running',done=done,
                        total=p['queries'],GT_read=False,seconds=time.monotonic()-tick))
                    print(stage,direction,done,p['queries'],flush=True)
                del data,frames;gc.collect();torch.cuda.empty_cache()
            if actor: actor.close();actor=None
            assert state_hash(model.state_dict())==mh
            write(out/f'{stage.upper()}_BARRIER.json',dict(count=done,GT_read=False,
                files={f.name:sha(f) for f in (out/stage).glob('*.json')},model_restored=True))
        else:
            assert stage=='online';assert read(out/'CAPTURE_BARRIER.json')['count']==p['queries']
            assert read(out/'TARGET_REFERENCE_BARRIER.json')['count']==p['queries']
            assert read(out/'SUPPORT.json')['checkpoint_state_sha256']==mh
            parity=[]
            for order,seq in p['orders'].items():
                actor=factory();previous=state_hash(actor.actor.initial);lastwrite=None
                for at,parent in enumerate(seq):
                    budget();f=out/'online'/order/f'{at:05}.pt.gz';rf=f.with_suffix('.json')
                    row=p['rows'][parent];scheduled=at%4==0
                    if rf.exists():
                        r=read(rf);assert r['pre_sha']==previous and sha(f)==r['sha256']
                        x=loadz(f);actor.actor.restore(device_tree(x['post_state'],'cuda'))
                        if x.get('arm_parity') is not None:parity.append(x['arm_parity'])
                        previous=r['post_sha'];lastwrite=at if x['updated'] else lastwrite;done+=1;counts['receipt_resumed_arrivals']+=1;continue
                    cf=out/'capture'/f'{parent:05}.pt';data=checked(cf);pixel=data['pixel_sha256']
                    frozen=compact_prediction(data['prediction']);fast=data['fast_only'] if scheduled else frozen
                    assert state_hash(actor.actor.state())==previous
                    d=device_tree(data,'cuda');calls=[]
                    def get(name):
                        assert scheduled;calls.append(name);return specialist(parent,name,pixel)
                    x,ev=actor.arrive(d,scheduled,lambda:get('temporal'),lambda:get('spatial'))
                    counts.update(x['compute'])
                    assert calls==(['temporal','spatial'] if scheduled else [])
                    assert x['pre_state_sha256']==previous
                    assert torch.equal(x['output_prediction']['boxes'],x['prediction']['boxes'])
                    control=None
                    if order=='order0' and at in [0,4]:
                        actor.close();actor=None;controlactor=factory(False)
                        controlactor.actor.restore(device_tree(x['pre_state'],'cuda'))
                        sx,_=controlactor.arrive(d,True,None,lambda:specialist(parent,'spatial',pixel))
                        counts.update({'live_parity_'+k:v for k,v in sx['compute'].items()})
                        assert sx['post_state_sha256']==x['post_state_sha256']
                        assert torch.equal(sx['prediction']['boxes'],x['prediction']['boxes'])
                        assert sx['prediction']['indices']==x['prediction']['indices']
                        assert len(sx['update_steps'])==len(x['update_steps'])
                        for a,b in zip(sx['update_steps'],x['update_steps']):
                            assert a['post_state_sha256']==b['post_state_sha256']
                            if a['update'] is not None:
                                assert a['update']['loss_before']==b['update']['loss_before']
                                for n,g in a['update']['gradients'].items():assert torch.equal(g,b['update']['gradients'][n])
                        controlactor.close();actor=factory();actor.actor.restore(device_tree(x['post_state'],'cuda'))
                        control=dict(arrival=at,box_interval_state_gradient_bitwise=True,steps=len(x['update_steps']))
                        parity.append(control)
                    rein=None
                    if at in [0,len(seq)-1]:
                        frames,ids=decode(row['input']);assert pixelhash(frames)==pixel
                        rein=reinsert(model,frames,{**row,'parses':dict(subject=subject(row))},actor.actor.state(),ev)
                        assert rein['full_pipeline_exact'];del frames
                        counts['new_full_backbone_offset_forwards']+=2
                    payload=dict(parent=parent,source=row['source'],order=order,condition='clean',arrival=at,
                        expert_scheduled=scheduled,updated=x['updated'],pre_state=x['pre_state'],post_state=x['post_state'],
                        pre_sha=x['pre_state_sha256'],post_sha=x['post_state_sha256'],update_steps=x['update_steps'],
                        compute=x['compute'],source_native=frozen,fast_only=fast,slow=compact_prediction(x['prediction']),
                        final=compact_prediction(x['output_prediction']),post_prediction=compact_prediction(x['post_prediction']),
                        temporal=x.get('temporal'),pixel_sha256=pixel,last_prior_write=lastwrite,
                        nearest_write_distance=None if lastwrite is None else at-lastwrite,
                        displacement=x['parameter_displacement'],displacement_from_source=x['displacement_from_source'],
                        arm_parity=control,reinsertion=rein,GT_read=False)
                    savez(f,payload);write(rf,dict(sha256=sha(f),parent=parent,pre_sha=previous,
                        post_sha=x['post_state_sha256'],GT_read=False))
                    previous=x['post_state_sha256'];lastwrite=at if x['updated'] else lastwrite;done+=1
                    if done%12==0 or at==0:
                        status(out/'ONLINE_STATUS.json',dict(status='running',done=done,total=p['total'],
                            order=order,arrival=at,GT_read=False,seconds=time.monotonic()-tick))
                        print('ONLINE',direction,done,p['total'],order,at,flush=True)
                    del data,d,x,ev,payload;gc.collect();torch.cuda.empty_cache()
                actor.close();actor=None;assert state_hash(model.state_dict())==mh
            assert done==p['total'] and len(parity)==2
            write(out/'ARM_TRAJECTORY_PARITY.json',dict(status='pass',controls=parity,GT_read=False,
                fast_does_not_enter_spatial_update=True))
            files={str(f.relative_to(out)):sha(f) for sub in ['online','capture','target_reference']
                   for f in (out/sub).rglob('*.json')}
            assert len(files)==p['total']+2*p['queries']
            write(out/'PREDICTION_BARRIER.json',dict(cells=done,files=files,GT_read=False,
                model_restored=True,all_arms_predict_before_spatial_write=True,time=time.time()))
        write(out/f'{stage.upper()}_RESOURCES.json',dict(counts=dict(counts),
            worker_wall_seconds=time.monotonic()-tick,wall_includes_loading_IO=True,
            peak_allocated_vram_bytes=torch.cuda.max_memory_allocated(),GT_read=False))
        status(out/f'{stage.upper()}_STATUS.json',dict(status='completed',done=done,
            seconds=time.monotonic()-tick,GT_read=False))
    except BaseException as e:
        status(out/f'{stage.upper()}_STATUS.json',dict(status='failed',done=done,error=repr(e),
            traceback=traceback.format_exc(),GT_read=False));raise
    finally:
        if actor:actor.close()
        if lease:lease.close()

if __name__=='__main__':run(*sys.argv[1:])
