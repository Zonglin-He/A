"""Finite synchronized five-stage timing; original fits and fresh experts."""
import collections
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,status,sha,save,load
from scripts.stvg_opd_design_ablation_common_v1 import activate_old,audit,scientific
BASE=ROOT/'artifacts/stvg_opd_supplement_v1/cost'
PUB=ROOT/'results/stvg_opd_supplement/2026-10-11/cost'
OLD=ROOT/'artifacts/stvg_opd_paper_hc2_revision_v2'
ORDER=ROOT/'artifacts/stvg_opd_supplement_v1/orders'
PYTHON=ROOT/'.conda/tubedetr/bin/python'

def prepare():
    if (BASE/'RUNTIME_LOCK.json').exists():return verify()
    design=read(ORDER/'DESIGN_LOCK.json');stages={}
    for name,z in design['stages'].items():
        ds=z['dataset'];queries=sorted(z['input_order'],key=lambda q:hashlib.sha256(f'stvg-cost|{ds}|2026101104|{q}'.encode()).hexdigest())[:16]
        b=read(OLD/'stages'/z['matched_full_stage']/'PREDICTION_BARRIER.json');bindings=[]
        for q in queries:
            at=z['input_order'].index(q);p=b['logical_records'][f'clean/order1/on_policy/{at:05}']
            bindings.append(dict(query_ordinal=q,original_arrival=at,path=str((OLD/p['path']).relative_to(ROOT)),sha256=p['sha256']))
        stages[ds]=dict(source=z['source'],config=z['variant_configs']['full'],queries=queries,bindings=bindings,original_barrier_sha256=sha(OLD/'stages'/z['matched_full_stage']/'PREDICTION_BARRIER.json'))
    write(BASE/'DESIGN_LOCK.json',dict(status='locked_before_timing_GPU',stages=stages,query_count=32,seed=2026101104,resident_weights=True,cold_application_frame_cache=True,OS_page_cache_evicted=False,saved_query_parse=True,GT_read=False,parameter_selection=False,time=time.time()))
    files=['protocols/stvg_opd_end_to_end_cost_supplement_v1.md','scripts/run_stvg_opd_end_to_end_cost_supplement_v1.py','scripts/continue_stvg_opd_end_to_end_cost_supplement_v1.py']
    pins={**read(OLD/'P4_joint_engineering_revision001/REVISION_RUNTIME.json')['pins'],**{f:sha(ROOT/f) for f in files}}
    protected=['methods/CURRENT_METHOD.json','methods/decota_spatial_opd_v1/configs.json',str((ORDER/'DESIGN_LOCK.json').relative_to(ROOT)),str((OLD/'FINAL_COMPLETION.json').relative_to(ROOT))]
    write(BASE/'RUNTIME_LOCK.json',dict(status='locked_before_GPU',design_sha256=sha(BASE/'DESIGN_LOCK.json'),pins=pins,protected={f:sha(ROOT/f) for f in protected},GT_read=False,time=time.time()))
    return verify()

def verify():
    p=BASE/'preGPU_timing_scope_engineering001/REVISION_RUNTIME.json'
    r=read(p if p.exists() else BASE/'RUNTIME_LOCK.json');assert sha(BASE/'DESIGN_LOCK.json')==r['design_sha256']
    for f,h in {**r['pins'],**r['protected']}.items():assert sha(ROOT/f)==h,f
    return r

def expert_science(z):
    p=json.loads(json.dumps(z))
    for o in p['observations']:
        for k in ['seconds','forward_seconds']:o['receipt'].pop(k,None)
    return p

def execute(ds):
    verify();activate_old()
    assert read(ROOT/'artifacts/stvg_opd_supplement_v1/baselines/FORMAL_GLOBAL_BARRIER.json')['status']=='sealed'
    import torch
    from scripts.run_decota_paper_main_v1 import gpu,model_for,read_row,frames_for,pack_expert
    from scripts.run_spatial_ssl_gpu_v1 import frozen_forward
    from scripts.stvg_opd_paper_inputs_v1 import observation
    from scripts.stvg_opd_legacy_input_bridge003 import normalized
    from methods.decota_final_simplified_v1.observations import SpatialExpert,observations
    from methods.decota_final_simplified_v1.config import EXPERT_SNAPSHOT
    from methods.decota_final_simplified_v1.objectives import prediction
    from methods.decota_final_simplified_v1.tensors import state_hash
    from vg_tta.stvg_opd_p4_joint_precision001 import Installation
    from scripts.run_stvg_opd_p1_softmax_recovery_002 import equal
    from vg_tta.decota_spatial_opd_tunable_v1 import TrickReplay
    s=read(BASE/'DESIGN_LOCK.json')['stages'][ds];dest=BASE/ds
    if (dest/'COMPLETION.json').exists():return
    lease=gpu();start=time.perf_counter();model=model_for(s['source']);expert=SpatialExpert(ROOT/EXPERT_SNAPSHOT);torch.cuda.synchronize()
    startup=time.perf_counter()-start;mh=state_hash(model.state_dict());eh=state_hash(expert.model.state_dict());recorder=Installation(qualification=False)
    oldvalues=TrickReplay.values; timings=[];rows=[];files={}
    def measured_values(replay):
        torch.cuda.synchronize();t=time.perf_counter();out=oldvalues(replay);torch.cuda.synchronize();timings.append(time.perf_counter()-t);return out
    TrickReplay.values=measured_values
    try:
        for i,binding in enumerate(s['bindings']):
            original=ROOT/binding['path'];assert sha(original)==binding['sha256'];old=load(original)
            ip=OLD/old['input']['path'];assert sha(ip)==old['input']['sha256'];inp=normalized(load(ip))
            assert old['config']==s['config'];q=binding['query_ordinal'];timings.clear();gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            t0=time.perf_counter();row=read_row(ds,q);frames,ids,reuse=frames_for(ds,row,collections.OrderedDict());shifted,pixel,spec=observation(ds,row,'clean',frames);torch.cuda.synchronize();t1=time.perf_counter()
            batch,records,base=frozen_forward(model,shifted,row);native=prediction(base.zero['logits'],base.zero['boxes'],records,ids);torch.cuda.synchronize();t2=time.perf_counter()
            ex=observations(expert,row['parses'],shifted,ids,native['indices'],audit=True);torch.cuda.synchronize();t3=time.perf_counter()
            from vg_tta.decota_spatial_opd_tunable_v1 import fit
            result=fit(base,old['fit']['initial'],ex,ids,row['key'],'on_policy',config=s['config']);torch.cuda.synchronize();t4=time.perf_counter()
            assert timings and not reuse;final_seconds=timings[-1]
            parts=dict(decode_preprocess_seconds=t1-t0,native_student_prefix_seconds=t2-t1,fresh_expert_seconds=t3-t2,adaptation_seconds=t4-t3-final_seconds,final_readout_seconds=final_seconds)
            total=t4-t0;dispatcher=total-sum(parts.values());assert abs(dispatcher)<1e-7 and parts['adaptation_seconds']>=0
            peak=dict(CUDA_peak_allocated=torch.cuda.max_memory_allocated(),CUDA_peak_reserved=torch.cuda.max_memory_reserved())
            payload=dict(dataset=ds,query_ordinal=q,original_fit_sha256=binding['sha256'],original_input_sha256=sha(ip),fit=result,expert=pack_expert(ex),native=native,pixel_sha256=pixel,frame_ids=ids,timing=parts,total_seconds=total,dispatcher_seconds=dispatcher,GT_read=False)
            f=dest/'fits'/f'{i:02}.pt';save(f,payload)  # Full failed evidence exists before acceptance checks.
            assert pixel==inp['pixel_sha256'] and spec==inp['corruption_spec'] and ids==inp['frame_ids']
            assert native['indices']==inp['indices'] and native['physical_interval']==old['interval']==inp['interval'] and torch.equal(native['boxes'].cpu(),inp['native_boxes'])
            assert mh==inp['source_model_state_sha256'] and expert_science(pack_expert(ex))==expert_science(inp['expert'])
            mathcheck=audit(result,ex);count=dict(tensors=0,tensor_coordinates=0,scalar_values=0);equal(scientific(result),scientific(old['fit']),count)
            base.restore(base.initial)
            assert state_hash(model.state_dict())==mh and state_hash(expert.model.state_dict())==eh
            write(f.with_suffix('.json'),dict(status='pass',sha256=sha(f),bytes=f.stat().st_size,original_full_fit_bitwise=count,math_audit_status=mathcheck['status'] if 'status' in mathcheck else 'pass',GT_read=False,time=time.time()))
            rr=dict(dataset=ds,anonymous_profile_id=i,source=s['source'],frame_count=len(ids),config=s['config'],**parts,total_seconds=total,dispatcher_seconds=dispatcher,**peak,actual_new_DINO_calls=ex['new_DINO'],admitted_references=len(result['positions']),backward_steps=result['gradient_calls'],original_full_fit_bitwise=count,post_timing_CPU_verification_seconds=time.perf_counter()-t4)
            rows.append(rr);files[str(f.relative_to(ROOT))]=sha(f);base.restore(base.initial);del frames,shifted,batch,base,result,ex,payload;gc.collect();torch.cuda.empty_cache()
            status(BASE/'STATUS.json',dict(status='running',dataset=ds,done=i+1,total=16,GT_read=False,time=time.time()))
            print('END_TO_END_COST',ds,i+1,16,round(total,4),flush=True)
        assert len(rows)==16
        write(PUB/ds/'QUERY_TIMINGS.json',dict(status='actual_complete_matching_fits',rows=rows,resident_model_and_expert_startup_seconds=startup,source_model_state_sha256=mh,expert_model_state_sha256=eh,GT_read=False))
        write(dest/'COMPLETION.json',dict(status='complete_actual_timing_and_matched_old_fit',fits=16,new_DINO_calls=sum(r['actual_new_DINO_calls'] for r in rows),files=files,rows_sha256=sha(PUB/ds/'QUERY_TIMINGS.json'),source_expert_actual_hashes_unchanged=True,GT_read=False,time=time.time()))
    finally:TrickReplay.values=oldvalues;recorder.uninstall();lease.close()

if __name__=='__main__':
    try:
        if sys.argv[1]=='prepare':prepare();print('COST_LOCKED',sha(BASE/'RUNTIME_LOCK.json'))
        else:execute(sys.argv[1])
    except BaseException:
        p=BASE/'failures'/str(time.time_ns());p.mkdir(parents=True,exist_ok=True);(p/'traceback.txt').write_text(traceback.format_exc());status(BASE/'STATUS.json',dict(status='failed_preserved_pending_actual_root',evidence=str(p.relative_to(ROOT)),GT_read=False,time=time.time()));raise
