"""Isolated Full/Frozen profiling after real complete-fit qualification."""
import collections
import gc
import os
import sys
import time
import traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,status,sha,save,load
from scripts.stvg_opd_design_ablation_common_v1 import activate_old
BASE=ROOT/'artifacts/stvg_opd_claim_strengthening_v1/deployment_latency'
PUB=ROOT/'results/stvg_opd_claim_strengthening/2026-10-11/deployment_latency'
OLD=ROOT/'artifacts/stvg_opd_paper_hc2_revision_v2'


def prepare():
    if (BASE/'RUNTIME_LOCK.json').exists():return verify()
    oldlock=ROOT/'artifacts/stvg_opd_supplement_v1/cost/DESIGN_LOCK.json'
    d=read(oldlock)
    write(BASE/'DESIGN_LOCK.json',dict(status='locked_before_qualification_and_timing',
        stages=d['stages'],unique_queries=32,paired_repeats=2,methods=['Frozen','Full'],
        process_order=[['Frozen',0],['Full',0],['Full',1],['Frozen',1]],
        resident_weights_per_process=True,cold_application_decode=True,OS_page_cache_evicted=False,
        existing_deterministic_subject_parse_cache=True,source_cost_lock_sha256=sha(oldlock),
        GPU_qualification_unique_queries=32,GT_read=False,time=time.time()))
    from vg_tta.stvg_opd_deployment_profile_v1 import cpu_contracts
    write(BASE/'CPU_CONTRACTS.json',cpu_contracts())
    files=['protocols/stvg_opd_claim_strengthening_v1.md',
        'vg_tta/stvg_opd_deployment_profile_v1.py','scripts/run_stvg_opd_deployment_latency_v1.py',
        'scripts/finalize_stvg_opd_deployment_latency_v1.py','scripts/continue_stvg_opd_claim_strengthening_v1.py',
        'scripts/run_stvg_opd_end_to_end_cost_supplement_v1.py',
        'vg_tta/decota_spatial_opd_tunable_v1.py','vg_tta/c1_enabling_tricks_v1.py',
        'vg_tta/decota_spatial_opd_boundary_chart_proposal003.py',
        'methods/decota_final_simplified_v1/observations.py','scripts/stvg_opd_legacy_input_bridge003.py']
    write(BASE/'RUNTIME_LOCK.json',dict(status='locked_before_GPU',
        design_sha256=sha(BASE/'DESIGN_LOCK.json'),CPU_contracts_sha256=sha(BASE/'CPU_CONTRACTS.json'),
        pins={**read(OLD/'P4_joint_engineering_revision001/REVISION_RUNTIME.json')['pins'],
              **{p:sha(ROOT/p) for p in files}},
        protected={p:sha(ROOT/p) for p in ['methods/CURRENT_METHOD.json',
            'methods/decota_spatial_opd_v1/configs.json',str(oldlock.relative_to(ROOT)),
            str((OLD/'FINAL_COMPLETION.json').relative_to(ROOT))]},GT_read=False,time=time.time()))
    status(BASE/'STATUS.json',dict(status='prepared_pending_serial_dependencies_and_actual_GPU_qualification',
        completed_unique_queries=0,planned_unique_queries=32,GT_read=False,time=time.time()))
    return verify()


def runtime_path():
    return BASE/'preGPU_receipt_dispatch002/REVISION_RUNTIME.json'


def verify():
    r=read(runtime_path())
    assert sha(BASE/'DESIGN_LOCK.json')==r['design_sha256']
    assert sha(BASE/'CPU_CONTRACTS.json')==r['CPU_contracts_sha256']
    for p,h in {**r['pins'],**r['protected']}.items():assert sha(ROOT/p)==h,p
    return r


def expert_fields(z):
    # Admission/detection science is exact; timing and audit-only hashes differ.
    return dict(positions4=z['positions4'],anchors=z['anchors'],
        actual_observation_positions=z['actual_observation_positions'],
        observations=[dict(view=x['view'],position=x['position'],probe=x['probe']) for x in z['observations']])


def original(binding):
    p=ROOT/binding['path'];assert sha(p)==binding['sha256'];z=load(p)
    ip=OLD/z['input']['path'];assert sha(ip)==z['input']['sha256']
    from scripts.stvg_opd_legacy_input_bridge003 import normalized
    return p,z,ip,normalized(load(ip))


def qualification(ds):
    verify();activate_old()
    import torch
    from scripts.run_decota_paper_main_v1 import gpu,model_for,read_row,frames_for,unpack_expert
    from scripts.run_spatial_ssl_gpu_v1 import frozen_forward
    from scripts.stvg_opd_paper_inputs_v1 import observation
    from methods.decota_final_simplified_v1.objectives import prediction
    from methods.decota_final_simplified_v1.observations import SpatialExpert,observations
    from methods.decota_final_simplified_v1.config import EXPERT_SNAPSHOT
    from methods.decota_final_simplified_v1.tensors import state_hash
    from vg_tta.stvg_opd_deployment_profile_v1 import fit,compare_complete_trace,compare_output
    from scripts.run_stvg_opd_p3_scope_precision001 import dispatch_saved
    from scripts.run_decota_paper_main_v1 import pack_expert
    s=read(BASE/'DESIGN_LOCK.json')['stages'][ds];dest=BASE/'qualification'/ds
    if (dest/'COMPLETION.json').exists():return
    lease=gpu();model=model_for(s['source']);expert=SpatialExpert(ROOT/EXPERT_SNAPSHOT)
    mh=state_hash(model.state_dict());eh=state_hash(expert.model.state_dict());records=[]
    try:
        for i,b in enumerate(s['bindings']):
            oldpath,z,ip,inp=original(b);row=read_row(ds,b['query_ordinal'])
            frames,ids,reuse=frames_for(ds,row,collections.OrderedDict())
            shifted,pixel,spec=observation(ds,row,'clean',frames)
            batch,fr,base=frozen_forward(model,shifted,row)
            native=prediction(base.zero['logits'],base.zero['boxes'],fr,ids)
            assert pixel==inp['pixel_sha256'] and ids==inp['frame_ids'] and spec==inp['corruption_spec']
            assert native['physical_interval']==inp['interval'] and native['indices']==inp['indices']
            assert torch.equal(native['boxes'].cpu(),inp['native_boxes']) and mh==inp['source_model_state_sha256']
            ex=observations(expert,row['parses'],shifted,ids,native['indices'],audit=True)
            assert expert_fields(pack_expert(ex))==expert_fields(inp['expert'])
            # Check saved complete old mathematical dictionary independently first.
            mathcheck=dispatch_saved(z['fit'],unpack_expert(inp['expert']),z,dict(b,origin_stage=z['stage']))
            assert mathcheck==z['math_audit']
            result=fit(base,z['fit']['initial'],ex,ids,row['key'],config=s['config'],trace=True)
            f=dest/'fits'/f'{i:02}.pt'
            save(f,dict(fit=result,old_fit_sha256=sha(oldpath),input_sha256=sha(ip),GT_read=False))
            checked=compare_complete_trace(result,z['fit'])
            minimal=fit(base,z['fit']['initial'],ex,ids,row['key'],config=s['config'],trace=False)
            mf=dest/'minimal_controls'/f'{i:02}.pt'
            save(mf,dict(fit=minimal,qualified_trace_sha256=sha(f),GT_read=False))
            outputcheck=compare_output(minimal,result)
            assert state_hash(model.state_dict())==mh and state_hash(expert.model.state_dict())==eh
            rc=dict(status='pass',query_ordinal=b['query_ordinal'],sha256=sha(f),bytes=f.stat().st_size,
                complete_old_fit_comparison=checked,trace_disabled_output_state=outputcheck,
                minimal_control_path=str(mf.relative_to(BASE)),minimal_control_sha256=sha(mf),
                minimal_control_bytes=mf.stat().st_size,
                independent_saved_math_status='pass',original_input_sha256=sha(ip),
                source_expert_actual_hashes_unchanged=True,actual_new_DINO_calls=ex['new_DINO'],
                actual_GPU_fits=2,formal_predictions_accepted=0,GT_read=False,time=time.time())
            write(f.with_suffix('.json'),rc);records.append(rc)
            base.restore(base.initial);del frames,shifted,batch,base,result,minimal,ex,z
            gc.collect();torch.cuda.empty_cache()
            status(BASE/'STATUS.json',dict(status='actual_GPU_qualification',dataset=ds,done=i+1,total=16,
                worker_pid=os.getpid(),GT_read=False,time=time.time()))
            print('DEPLOYMENT_QUALIFICATION',ds,i+1,16,flush=True)
        write(dest/'COMPLETION.json',dict(status='pass',qualification_unique_queries=16,actual_GPU_fits=32,
            records=records,actual_new_DINO_calls=sum(r['actual_new_DINO_calls'] for r in records),
            formal_predictions_accepted=0,GT_read=False,runtime_sha256=sha(runtime_path()),time=time.time()))
    finally:lease.close()


def root_qualification():
    verify();activate_old()
    import torch
    torch.set_num_threads(2)
    from vg_tta.stvg_opd_deployment_profile_v1 import compare_complete_trace,compare_output
    from scripts.run_stvg_opd_p3_scope_precision001 import dispatch_saved
    from scripts.run_decota_paper_main_v1 import unpack_expert
    count=coords=0;dino=0;output_coords=0;math_rounds=0
    for ds,s in read(BASE/'DESIGN_LOCK.json')['stages'].items():
        c=read(BASE/'qualification'/ds/'COMPLETION.json');assert c['status']=='pass'
        for i,b in enumerate(s['bindings']):
            p=BASE/'qualification'/ds/'fits'/f'{i:02}.pt';rc=read(p.with_suffix('.json'))
            assert sha(p)==rc['sha256'] and not rc['GT_read']
            _,z,_,inp=original(b);new=load(p)['fit']
            checked=compare_complete_trace(new,z['fit']);assert checked==rc['complete_old_fit_comparison']
            mp=BASE/rc['minimal_control_path'];assert sha(mp)==rc['minimal_control_sha256']
            control=load(mp);assert control['qualified_trace_sha256']==sha(p) and not control['GT_read']
            outputcheck=compare_output(control['fit'],new);assert outputcheck==rc['trace_disabled_output_state']
            output_coords+=outputcheck['output_state_coordinates']
            mathcheck=dispatch_saved(z['fit'],unpack_expert(inp['expert']),z,dict(b,origin_stage=z['stage']))
            assert mathcheck==z['math_audit'];math_rounds+=len(z['fit']['rounds'])
            coords+=checked['tensor_coordinates'];dino+=rc['actual_new_DINO_calls'];count+=1
    assert count==32
    write(BASE/'ROOT_QUALIFICATION_READBACK.json',dict(status='pass',qualification_unique_queries=count,
        actual_GPU_fits=64,complete_fit_tensor_coordinates=coords,actual_new_DINO_calls=dino,
        independently_read_trace_disabled_state_coordinates=output_coords,
        independent_original_complete_math_rounds=math_rounds,
        scope='all original complete saved rounds/actions/rewards/weights/gradients/state versus deployment trace; trace-off outputs',
        formal_predictions_accepted=0,GT_read=False,independent_full_decoder_Jacobian=False,time=time.time()))


def timed(ds,method,repeat):
    verify();activate_old()
    assert method in ['Full','Frozen'] and repeat in [0,1]
    assert read(BASE/'ROOT_QUALIFICATION_READBACK.json')['status']=='pass'
    import torch
    from scripts.run_decota_paper_main_v1 import gpu,model_for,read_row,frames_for,pack_expert
    from scripts.run_spatial_ssl_gpu_v1 import frozen_forward
    from scripts.stvg_opd_paper_inputs_v1 import observation
    from methods.decota_final_simplified_v1.objectives import prediction
    from methods.decota_final_simplified_v1.observations import SpatialExpert,observations
    from methods.decota_final_simplified_v1.config import EXPERT_SNAPSHOT
    from methods.decota_final_simplified_v1.tensors import state_hash
    from vg_tta.stvg_opd_deployment_profile_v1 import fit,compare_output
    s=read(BASE/'DESIGN_LOCK.json')['stages'][ds];dest=BASE/'timing'/ds/method/str(repeat)
    if (dest/'COMPLETION.json').exists():return
    lease=gpu();start=time.perf_counter();model=model_for(s['source'])
    expert=SpatialExpert(ROOT/EXPERT_SNAPSHOT) if method=='Full' else None
    torch.cuda.synchronize();startup=time.perf_counter()-start;mh=state_hash(model.state_dict())
    eh=state_hash(expert.model.state_dict()) if expert else None
    rows=[]
    try:
        for i,b in enumerate(s['bindings']):
            _,old,ip,inp=original(b)
            qualified=load(BASE/'qualification'/ds/'fits'/f'{i:02}.pt')['fit']
            gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            tick=time.perf_counter()
            row=read_row(ds,b['query_ordinal']);frames,ids,reused=frames_for(ds,row,collections.OrderedDict())
            shifted,pixel,spec=observation(ds,row,'clean',frames);torch.cuda.synchronize();t1=time.perf_counter()
            batch,fr,base=frozen_forward(model,shifted,row)
            native=prediction(base.zero['logits'],base.zero['boxes'],fr,ids);torch.cuda.synchronize();t2=time.perf_counter()
            ex=None;result=None;last_values_seconds=[]
            if method=='Full':
                ex=observations(expert,row['parses'],shifted,ids,native['indices'],audit=False)
                torch.cuda.synchronize();t3=time.perf_counter()
                from vg_tta.stvg_opd_deployment_profile_v1 import DeploymentReplay
                values=DeploymentReplay.values
                def measured_values(replay):
                    torch.cuda.synchronize();start=time.perf_counter();v=values(replay)
                    torch.cuda.synchronize();last_values_seconds.append(time.perf_counter()-start);return v
                DeploymentReplay.values=measured_values
                try:result=fit(base,old['fit']['initial'],ex,ids,row['key'],config=s['config'],trace=False)
                finally:DeploymentReplay.values=values
                torch.cuda.synchronize();t4=time.perf_counter();final_seconds=last_values_seconds[-1]
                committed={n:(torch.zeros_like(v) if n=='spatial.query_residual' else
                    v+(result['state'][n]-v)*s['config']['writeback']) for n,v in result['initial'].items()}
                base.restore(committed)
                output=dict(boxes=result['final'].numpy().tolist(),indices=native['indices'],
                    physical_interval=native['physical_interval'],frame_ids=ids)
            else:
                t3=t4=t2;final_seconds=0.
                output=dict(boxes=native['boxes'].detach().cpu().numpy().tolist(),indices=native['indices'],
                    physical_interval=native['physical_interval'],frame_ids=ids)
            torch.cuda.synchronize();end=time.perf_counter()
            parts=dict(row_decode_preprocess_seconds=t1-tick,native_student_seconds=t2-t1,
                fresh_expert_seconds=t3-t2,adaptation_seconds=t4-t3-final_seconds,
                final_readout_seconds=final_seconds,writeback_output_dispatch_seconds=end-t4)
            total=end-tick;assert abs(sum(parts.values())-total)<1e-7
            peaks=dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
            assert not reused and pixel==inp['pixel_sha256'] and ids==inp['frame_ids']
            assert spec==inp['corruption_spec'] and mh==inp['source_model_state_sha256']
            assert torch.equal(native['boxes'].cpu(),inp['native_boxes']) and native['physical_interval']==inp['interval']
            if method=='Full':
                assert expert_fields(pack_expert(ex))==expert_fields(inp['expert'])
                checked=compare_output(result,qualified)
                actual_writeback=base.state()
                assert all(torch.equal(actual_writeback[n].cpu(),v) for n,v in committed.items())
            else:checked=dict(status='pass',exact_original_native_output=True)
            proof=dest/'outputs'/f'{i:02}.pt'
            save(proof,dict(output=output,state=result['state'] if result else None,
                committed=committed if result else None,config=s['config'],timing=parts,
                actual_writeback=actual_writeback if result else None,
                total_seconds=total,GT_read=False))
            write(proof.with_suffix('.json'),dict(status='pass',sha256=sha(proof),bytes=proof.stat().st_size,
                qualified_output_state=checked,original_input_sha256=sha(ip),GT_read=False,time=time.time()))
            rows.append(dict(dataset=ds,profile_id=i,method=method,repeat=repeat,**parts,
                total_seconds=total,**peaks,frame_count=len(ids),actual_new_DINO_calls=ex['new_DINO'] if ex else 0,
                admitted_references=len(result['positions']) if result else 0,
                backward_steps=result['gradient_calls'] if result else 0,output_sha256=sha(proof),
                independent_audit_inside_timer=False,full_trace_inside_timer=False))
            base.restore(base.initial);del frames,shifted,batch,base,result,ex,qualified,old
            gc.collect();torch.cuda.empty_cache()
            status(BASE/'STATUS.json',dict(status='actual_paired_deployment_timing',dataset=ds,method=method,
                repeat=repeat,done=i+1,total=16,worker_pid=os.getpid(),GT_read=False,time=time.time()))
            print('PAIRED_DEPLOYMENT_TIMING',ds,method,repeat,i+1,16,flush=True)
        assert state_hash(model.state_dict())==mh and (not expert or state_hash(expert.model.state_dict())==eh)
        rp=PUB/ds/method/f'repeat{repeat}.json'
        write(rp,dict(status='complete_actual_timing',rows=rows,startup_seconds=startup,
            hardware=dict(device=torch.cuda.get_device_name(0),total_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
                torch_version=torch.__version__,cuda_version=torch.version.cuda,CPU_threads=torch.get_num_threads()),
            resident_models=['student','expert'] if expert else ['student'],GT_read=False))
        write(dest/'COMPLETION.json',dict(status='complete',query_count=16,method=method,repeat=repeat,
            timing_rows_sha256=sha(rp),source_expert_hashes_unchanged=True,GT_read=False,time=time.time()))
    finally:lease.close()


if __name__=='__main__':
    try:
        job=sys.argv[1]
        if job=='prepare':prepare()
        elif job=='qualification':qualification(sys.argv[2])
        elif job=='root_qualification':root_qualification()
        elif job=='timing':timed(sys.argv[2],sys.argv[3],int(sys.argv[4]))
        else:raise ValueError(job)
    except BaseException:
        p=BASE/'failures'/str(time.time_ns());p.mkdir(parents=True,exist_ok=True)
        (p/'traceback.txt').write_text(traceback.format_exc())
        status(BASE/'STATUS.json',dict(status='failed_preserved_pending_actual_root',evidence=str(p.relative_to(ROOT)),
            GT_read=False,experiment_complete=False,time=time.time()))
        raise
