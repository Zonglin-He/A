"""One HC2 development candidate. No scores, labels, or confirmation inputs."""
import collections,gc,os,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_hc2_coordinate_common_v1 import *

def run(uid):
    verify();bridge();budget()
    import torch
    from scripts.run_decota_paper_main_v1 import gpu,model_for,read_row
    import scripts.run_decota_spatial_opd_v1 as capture_module
    from methods.decota_final_simplified_v1.observations import SpatialExpert
    from methods.decota_final_simplified_v1.config import EXPERT_SNAPSHOT
    from methods.decota_final_simplified_v1.tensors import state_hash
    from vg_tta.spatial_online_state_v1 import arrival
    from vg_tta.decota_spatial_opd_tunable_v1 import fit
    from vg_tta.decota_spatial_opd_tunable_audit_v1 import audit
    dest=BASE/'trials'/uid;t=read(dest/'CONFIG.json');cfg=t['config'];qual=t['phase']=='qualification'
    if (dest/'PREDICTION_BARRIER.json').exists():return
    if not qual:assert read(BASE/'QUALIFICATION.json')['status']=='pass'
    stage=dict(dataset='hc2',source=t['source'],split='development',parents=t['parents'],orders=t['orders'])
    capture_module.BASE=BASE
    lease=gpu();model=model_for(t['source']);source_hash=state_hash(model.state_dict())
    expert=SpatialExpert(ROOT/EXPERT_SNAPSHOT);cache=collections.OrderedDict();files={};count=0;parity=[]
    started=time.time()
    try:
        for order,seq in t['orders'].items():
            previous=None;previous_hash=None
            for at,q in enumerate(seq):
                budget();p=dest/order/f'{at:05}.pt'
                if p.exists():
                    rc=read(p.with_suffix('.json'));assert sha(p)==rc['sha256'] and p.stat().st_size==rc['bytes']
                    z=load(p);assert z['config']==cfg and z['parent']==q and z['previous_payload_sha256']==previous_hash
                    previous=z['committed'];previous_hash=sha(p);files[str(p.relative_to(ROOT))]=previous_hash;count+=1;continue
                row=read_row('hc2',q);begin=time.perf_counter()
                base,native,ex,inputrc=capture_module.capture(model,expert,stage,row,'clean',cache)
                capture_seconds=time.perf_counter()-begin;initial=arrival(base.initial,previous,'O-split')
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();begin=time.perf_counter()
                result=fit(base,initial,ex,row['frame_ids'],row['key'],'on_policy',config=cfg)
                torch.cuda.synchronize();fit_seconds=time.perf_counter()-begin
                begin=time.perf_counter();check=audit(result,ex);math_seconds=time.perf_counter()-begin
                assert result['active_parameters']==1792 and result['config']==cfg
                assert result['selected_step']==cfg['steps'] and len(result['path'])==cfg['steps']+1
                state=commit(initial,result['state'],cfg['writeback'])
                if qual and cfg==START:
                    old=load(OLD/'trials/hc2/refine_09'/order/f'{at:05}.pt')
                    assert old['parent']==q and old['config']==cfg
                    assert torch.equal(old['fit']['final'],result['final'])
                    assert all(torch.equal(old['fit']['state'][n],result['state'][n]) for n in state)
                    assert all(torch.equal(old['committed'][n],state[n]) for n in state)
                    parity.append(dict(query_ordinal=q,original_selected_trial_box_and_1792_state_bitwise=True))
                z=dict(dataset='hc2',source=t['source'],trial=uid,parent=q,order=order,arrival=at,
                    fit=result,committed=state,interval=native['physical_interval'],input=inputrc,
                    previous_payload_sha256=previous_hash,math_audit=check,config=cfg,
                    query_reset=True,Adam_reset=True,GT_read=False,runtime_sha256=sha(BASE/'RUNTIME_LOCK.json'),
                    compute=dict(capture_seconds=capture_seconds,fit_GPU_seconds=fit_seconds,
                        CPU_math_seconds=math_seconds,new_DINO_calls=inputrc['new_DINO_calls'],
                        CUDA_peak_allocated=torch.cuda.max_memory_allocated(),backward_steps=result['gradient_calls']))
                save(p,z);h=sha(p);write(p.with_suffix('.json'),dict(sha256=h,bytes=p.stat().st_size,
                    GT_read=False,runtime_sha256=sha(BASE/'RUNTIME_LOCK.json'),time=time.time()))
                files[str(p.relative_to(ROOT))]=h;previous=state;previous_hash=h;count+=1
                status(dest/'STATUS.json',dict(status='running',pid=os.getpid(),trial=uid,done=count,
                    total=sum(map(len,t['orders'].values())),config=cfg,GT_read=False,time=time.time()))
                print('HC2_COORDINATE_PROGRESS',uid,count,sum(map(len,t['orders'].values())),round(time.time()-started,2),flush=True)
                base.restore(base.initial);del base,result,z,initial;gc.collect();torch.cuda.empty_cache()
        assert state_hash(model.state_dict())==source_hash and count==sum(map(len,t['orders'].values()))
        write(dest/'PREDICTION_BARRIER.json',dict(status='sealed',trial=uid,cells=count,files=files,
            input_base=str(BASE.relative_to(ROOT)),config_sha256=sha(dest/'CONFIG.json'),
            exact_history_reused=False,source_hash=source_hash,GT_read=False,time=time.time()))
        if qual:
            write(dest/'QUALIFICATION.json',dict(status='pass',queries=count,config=cfg,parity=parity,
                actual_backwards=sum(load(ROOT/p)['fit']['gradient_calls'] for p in files),
                no_GT=True,source_weights_unchanged=True,time=time.time()))
        status(dest/'STATUS.json',dict(status='sealed',done=count,total=count,GT_read=False,time=time.time()))
    finally:lease.close()

if __name__=='__main__':
    uid=sys.argv[1]
    try:run(uid)
    except BaseException:
        d=BASE/'trials'/uid/'failure';d.mkdir(parents=True,exist_ok=True)
        (d/'traceback.txt').write_text(traceback.format_exc())
        status(d.parent/'STATUS.json',dict(status='failed_preserved',pid=os.getpid(),GT_read=False,time=time.time()))
        raise
