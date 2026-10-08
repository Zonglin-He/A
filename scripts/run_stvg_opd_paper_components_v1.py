"""Later component/budget/robustness worker; not launched by preparation.
Each source/condition/order/arm is rebuilt independently. Main arithmetic is unchanged.
"""
import collections, gc, os, sys, time, traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_paper_common_v1 import *

def run(stage_name, qualify=False):
    verify();bridge()
    import torch
    from scripts.run_decota_paper_main_v1 import gpu,model_for,read_row
    import scripts.run_decota_spatial_opd_v1 as capture_module
    from methods.decota_final_simplified_v1.observations import SpatialExpert
    from methods.decota_final_simplified_v1.config import EXPERT_SNAPSHOT
    from methods.decota_final_simplified_v1.tensors import state_hash
    from vg_tta.spatial_online_state_v1 import arrival
    from vg_tta.decota_spatial_opd_tunable_v1 import mean_coordinates,action_boxes
    from vg_tta.stvg_opd_paper_ablations_v1 import fit_variant
    from vg_tta.stvg_opd_paper_component_audit_v1 import audit
    from scripts.stvg_opd_paper_inputs_v1 import capture
    design=read(BASE/'DESIGN_LOCK.json');stage=read(BASE/'LATER_DESIGN_LOCK.json')['stages'][stage_name]
    runtime=read(BASE/'COMPONENT_RUNTIME_LOCK.json')
    for f,h in runtime['pins'].items():assert sha(ROOT/f)==h,f
    assert sha(BASE/'LATER_DESIGN_LOCK.json')==runtime['design_sha256']
    ds=stage['dataset'];arms=stage['arms']
    dest=BASE/('qualification' if qualify else 'stages')/stage_name
    if (dest/'PREDICTION_BARRIER.json').exists():return
    if not qualify:
        assert read(BASE/'P1_STAGE_AUTHORIZATION.json')['status']=='completed_before_later_stages'
        assert read(BASE/'COMPONENT_QUALIFICATION.json')['status']=='pass'
    orders={f'{condition}::{order}':(seq[:2] if qualify else seq) for condition in (stage['conditions'][:1] if qualify else stage['conditions']) for order,seq in (list(stage['orders'].items())[:1] if qualify else stage['orders'].items())}
    expected=sum(map(len,orders.values()))*len(arms)
    lease=gpu();model=model_for(stage['source']);modelhash=state_hash(model.state_dict())
    expert=SpatialExpert(ROOT/EXPERT_SNAPSHOT);cache=collections.OrderedDict()
    files={};inputs={};count=0;qual_records=[];tick=time.time()
    try:
        for condition_order,seq in orders.items():
            condition,order=condition_order.split('::')
            previous={a:None for a in arms};prevhash={a:None for a in arms}
            for at,q in enumerate(seq):
                budget();paths={a:dest/condition/order/a/f'{at:05}.pt' for a in arms}
                if all(f.exists() for f in paths.values()):
                    for arm,f in paths.items():
                        cfg=stage['variant_configs'][arm]
                        h=sha(f);rc=read(f.with_suffix('.json'));assert rc['sha256']==h
                        z=load(f);assert z['query_ordinal']==q and z['config']==cfg
                        assert z['previous_payload_sha256']==prevhash[arm]
                        previous[arm]=z['committed'];prevhash[arm]=h
                        files[str(f.relative_to(BASE))]=h
                        inputs[z['input']['path']]=z['input']['sha256'];count+=1
                    continue
                row=read_row(ds,q)
                torch.cuda.synchronize();begin=time.perf_counter()
                base,native,ex,inputrc=capture(model,expert,stage,row,condition,cache)
                torch.cuda.synchronize();capture_seconds=time.perf_counter()-begin
                with torch.no_grad():
                    assert torch.equal(base.values()['boxes'],base.zero['boxes'])
                    roundtrip=float((action_boxes(mean_coordinates(base.zero['boxes']))-base.zero['boxes']).abs().max())
                    assert roundtrip<2e-7
                inputs[inputrc['path']]=inputrc['sha256']
                for arm in arms:
                    cfg=stage['variant_configs'][arm]
                    f=paths[arm]
                    if f.exists():
                        h=sha(f);z=load(f);assert read(f.with_suffix('.json'))['sha256']==h
                        assert z['previous_payload_sha256']==prevhash[arm] and z['config']==cfg
                        previous[arm]=z['committed'];prevhash[arm]=h;files[str(f.relative_to(BASE))]=h;count+=1
                        continue
                    initial=arrival(base.initial,previous[arm],'O-split')
                    assert torch.count_nonzero(initial['spatial.query_residual'])==0
                    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();begin=time.perf_counter()
                    result=fit_variant(base,initial,ex,row['frame_ids'],row['key'],arm,cfg)
                    torch.cuda.synchronize();seconds=time.perf_counter()-begin
                    assert result['selected_step']==cfg['steps'] and result['active_parameters']==({'query_only':256,'LN_only':1536}.get(arm,1792))
                    start=time.perf_counter();mathcheck=audit(result,ex);cpu_seconds=time.perf_counter()-start
                    nextstate=committed(initial,result['state'],cfg['writeback'])
                    z=dict(dataset=ds,source=stage['source'],stage=stage_name,arm=arm,condition=condition,
                        order=order,arrival=at,query_ordinal=q,parent=q,fit=result,config=cfg,
                        committed=nextstate,previous_payload_sha256=prevhash[arm],input=inputrc,
                        interval=native['physical_interval'],math_audit=mathcheck,
                        query_reset=True,Adam_reset=True,Native_WHEN_fixed=True,
                        chart_roundtrip_max_error=roundtrip,GT_read=False,
                        runtime_lock_sha256=sha(BASE/'RUNTIME_LOCK.json'),
                        compute=dict(shared_capture_seconds=capture_seconds,fit_GPU_seconds=seconds,
                            CPU_math_seconds=cpu_seconds,new_DINO_calls=inputrc['new_DINO_calls'],
                            DINO_observation_budget=stage['observation_budget'],
                            input_capture_components=inputrc,CUDA_peak_allocated=torch.cuda.max_memory_allocated(),
                            CUDA_peak_reserved=torch.cuda.max_memory_reserved(),backward_steps=result['gradient_calls']))
                    save(f,z);h=sha(f)
                    write(f.with_suffix('.json'),dict(sha256=h,bytes=f.stat().st_size,GT_read=False,
                        runtime_lock_sha256=sha(BASE/'RUNTIME_LOCK.json'),time=time.time()))
                    files[str(f.relative_to(BASE))]=h;previous[arm]=nextstate;prevhash[arm]=h;count+=1
                    if qualify:qual_records.append(dict(dataset=ds,query_ordinal=q,arm=arm,
                        updates=result['gradient_calls'],last_step=result['selected_step'],config=cfg,
                        independent_math=mathcheck,source_model_unchanged_at_end=True))
                    del z,result,initial
                base.restore(base.initial);del base,native,ex
                gc.collect();torch.cuda.empty_cache()
                status(dest/'STATUS.json',dict(status='running',pid=os.getpid(),stage=stage_name,
                    qualification=qualify,done=count,total=expected,order=order,GT_read=False,time=time.time()))
                print('OPD_PAPER_PROGRESS',stage_name,order,count,expected,round(time.time()-tick,1),flush=True)
        assert state_hash(model.state_dict())==modelhash and count==expected
        write(dest/'PREDICTION_BARRIER.json',dict(status='sealed',stage=stage_name,
            qualification=qualify,adapted_arrivals=count,Frozen_logical_arrivals=sum(map(len,orders.values())),
            files=files,inputs=inputs,orders=orders,variant_configs=stage['variant_configs'],GT_read=False,
            source_checkpoint_unchanged=True,runtime_lock_sha256=sha(BASE/'RUNTIME_LOCK.json'),time=time.time()))
        if qualify:
            assert len(qual_records)==expected
            write(dest/'QUALIFICATION.json',dict(status='pass',dataset=ds,records=qual_records,
                actual_queries=len(orders)*2,actual_fits=expected,GT_read=False,time=time.time()))
        status(dest/'STATUS.json',dict(status='sealed_pending_stage_barrier_and_root_CPU',
            done=count,total=expected,GT_read=False,time=time.time()))
    finally:lease.close()

if __name__=='__main__':
    stage_name=sys.argv[1];qualify=len(sys.argv)>2 and sys.argv[2]=='qualification'
    try:run(stage_name,qualify)
    except BaseException:
        dest=BASE/'failures'/str(time.time_ns());dest.mkdir(parents=True,exist_ok=True)
        (dest/'traceback.txt').write_text(traceback.format_exc())
        status(BASE/'FAILURE.json',dict(status='failed_preserved',stage=stage_name,
            pid=os.getpid(),evidence=str(dest),GT_read=False,time=time.time()))
        raise
