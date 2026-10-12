"""Conditional one-factor temporal-guided observation candidate, same science."""
import collections,gc,os,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_temporal_guided_p7_common_v1 import *
ARMS=['C_temporal_guided_OPD','D_temporal_guided_Direct']

def verify_b():
    verify();r=read(B/'RUNTIME.json')
    assert sha(A/'P7_B_BRANCH_DECISION.json')==r['branch_decision_sha256'] and read(A/'P7_B_BRANCH_DECISION.json')['P7_B_eligible']
    for f,h in r['pins'].items():assert sha(ROOT/f)==h,f
    return r

def prepare():
    verify();decision=read(A/'P7_B_BRANCH_DECISION.json')
    assert decision['P7_B_eligible'] and read(A/'ROOT_READBACK.json')['status']=='pass'
    assert not B.exists();B.mkdir();(OUT/'P7_B').mkdir(parents=True,exist_ok=False)
    d=read(BASE/'DESIGN_LOCK.json')
    own=['scripts/run_stvg_opd_temporal_guided_p7b_v1.py','scripts/score_stvg_opd_temporal_guided_p7b_v1.py',
        'vg_tta/stvg_opd_p4_joint_precision001.py','vg_tta/stvg_opd_common_state_geometry_arithmetic004.py',
        'vg_tta/stvg_opd_paper_ablations_v1.py','vg_tta/decota_spatial_opd_tunable_v1.py',
        'vg_tta/stvg_opd_paper_component_audit_v1.py','methods/decota_final_simplified_v1/observations.py']
    from methods.decota_final_simplified_v1.config import EXPERT_SNAPSHOT,EXPERT_SHA256
    assert sha(ROOT/EXPERT_SNAPSHOT/'model.safetensors')==EXPERT_SHA256
    write(B/'RUNTIME.json',dict(status='locked_before_P7_B_actual_GPU',pins={f:sha(ROOT/f) for f in own},
        parent_runtime_sha256=sha(BASE/'RUNTIME.json'),branch_decision_sha256=sha(A/'P7_B_BRANCH_DECISION.json'),
        expert_model_sha256=EXPERT_SHA256,expert_snapshot=EXPERT_SNAPSHOT,formal_cells=512,new_formal_fits=1024,
        qualification_complete_fits=16,unique_expert_packs=256,max_new_DINO_calls=1024,
        source_anchors_A_B_not_changed=True,no_GT_selection=True,GT_read=False,time=time.time()))
    status(B/'STATUS.json',dict(status='prepared_conditional_one_experiment',new_actual_GPU_fits=0,new_actual_DINO_calls=0,GT_read=False,time=time.time()))

def source_temporal(r):
    # No GT scoring/statistics are read by these actual deployment workers.
    first=next(x for x in read(BASE/'DESIGN_LOCK.json')['records'] if x['capture_index']==r['index'] and x['order']=='order1')
    p=A/'readouts'/f"{first['public_cell']:05}.pt";barrier=read(A/'GLOBAL_READOUT_BARRIER.json');assert sha(p)==barrier['files'][str(p.relative_to(ROOT))]
    z=load(p);return z['temporal_binding'],z['intervals'],z['frame_ids']

def experts():
    import torch
    from scripts.stvg_opd_design_ablation_common_v1 import activate_old
    activate_old();verify_b();handle=lease(B)
    from scripts.run_decota_paper_main_v1 import read_row,frames_for,pack_expert
    from scripts.stvg_opd_paper_inputs_v1 import observation
    from methods.decota_final_simplified_v1.observations import SpatialExpert,observations,uniform_positions
    from methods.decota_final_simplified_v1.tensors import state_hash
    expert=None;files={};calls=0;positions=0;cache=collections.OrderedDict();start=time.time()
    try:
        expert=SpatialExpert(ROOT/read(B/'RUNTIME.json')['expert_snapshot']);eh=state_hash(expert.model.state_dict())
        for r in read(BASE/'DESIGN_LOCK.json')['captures']:
            tp,iv,ids=source_temporal(r);old=load(ROOT/tp['path']);raw=read_row(r['dataset'],r['query_ordinal']);frames,actual,_=frames_for(r['dataset'],raw,cache)
            shifted,pixel,spec=observation(r['dataset'],raw,'clean',frames)
            assert actual==ids and pixel==old['pixel_sha256'] and spec is None
            extent=[ids.index(iv['T1'][0]),ids.index(iv['T1'][1]-1)];planned=uniform_positions(ids,extent,4)
            torch.cuda.reset_peak_memory_stats();tick=time.perf_counter();ex=observations(expert,raw['parses'],shifted,ids,extent,audit=True)
            assert ex['positions4']==planned and ex['new_DINO']<=4 and len(ex['observations'])==ex['new_DINO']
            p=B/'expert_I1'/f"{r['index']:05}.pt";z=dict(index=r['index'],dataset=r['dataset'],parent_id=r['parent_id'],query_ordinal=r['query_ordinal'],
                temporal_binding=tp,interval=iv['T1'],frame_ids=ids,pixel_sha256=pixel,subject_sha256=raw['subject_sha256'],
                expert=pack_expert(ex),actual_new_DINO_calls=ex['new_DINO'],planned_observations=len(planned),
                actual_seconds=time.perf_counter()-tick,actual_forward_seconds=sum(v['receipt']['forward_seconds'] for v in ex['observations'].values()),
                CUDA_peak_allocated=torch.cuda.max_memory_allocated(),same_expert_source_sha256=eh,GT_read=False,
                runtime_sha256=sha(B/'RUNTIME.json'),shared_by_both_orders_and_C_D=True)
            save(p,z);h=sha(p);write(p.with_suffix('.json'),dict(status='accepted_actual_projected_scope_expert_pack',sha256=h,bytes=p.stat().st_size,
                actual_new_DINO_calls=ex['new_DINO'],planned_observations=len(planned),GT_read=False,time=time.time()))
            files[str(p.relative_to(ROOT))]=h;calls+=ex['new_DINO'];positions+=len(planned)
            status(B/'GPU_STATUS.json',dict(status='running_actual_I1_DINO_evidence',done=len(files),total=256,actual_DINO_calls=calls,pid=os.getpid(),GT_read=False,time=time.time()))
            print('P7B_DINO_PROGRESS',len(files),256,calls,flush=True)
            del frames,shifted,ex,z;gc.collect();torch.cuda.empty_cache()
        assert len(files)==256 and calls<=1024 and state_hash(expert.model.state_dict())==eh
        write(B/'EXPERT_EVIDENCE_BARRIER.json',dict(status='sealed',files=files,unique_query_packs=256,actual_new_DINO_calls=calls,
            planned_observation_positions=positions,ordinary_per_query_budget=4,GT_read=False,source_unchanged=True,elapsed_seconds=time.time()-start,time=time.time()))
    finally:
        if expert is not None:del expert
        cache.clear();gc.collect();torch.cuda.empty_cache();handle.close()

def fit_one(base,row,expert,old,arm,qualify):
    import torch
    from vg_tta.stvg_opd_p4_joint_precision001 import Installation
    from vg_tta.stvg_opd_common_state_geometry_arithmetic004 import Geometry,mathematical_readback
    from vg_tta.stvg_opd_paper_ablations_v1 import fit_variant
    from scripts.stvg_opd_design_ablation_common_v1 import audit as OPD_audit
    from vg_tta.stvg_opd_paper_component_audit_v1 import audit as Direct_audit
    rec=Installation(qualification=qualify);hook=Geometry(rec,'P7_observation_only',old['fit'])
    try:
        result=fit_variant(base,old['fit']['initial'],expert,row['frame_ids'],row['key'],
            'on_policy' if arm==ARMS[0] else 'direct_L1_GIoU',old['fit']['config'])
        result['geometry']=dict(rounds=hook.rows,common_initial_original_Full=True,GT_read=False)
        assert all(torch.equal(v,result['initial'][k]) for k,v in old['fit']['initial'].items())
        assert torch.equal(result['before'],old['fit']['before'])
        audit=(OPD_audit if arm==ARMS[0] else Direct_audit)(result,expert)
        checked=mathematical_readback(result)
        return result,dict(status='pass',objective_audit=audit,state_Adam_audit=checked,GT_read=False,actual_native_head_VJP=rec.last_vjp)
    finally:hook.close();rec.uninstall();base.restore(base.initial)

def fits(qualification):
    import torch
    from scripts.stvg_opd_design_ablation_common_v1 import activate_old
    activate_old();verify_b();handle=lease(B)
    from scripts.run_decota_paper_main_v1 import model_for,read_row,frames_for,unpack_expert
    from scripts.stvg_opd_paper_inputs_v1 import observation
    from scripts.stvg_opd_legacy_input_bridge003 import normalized
    from vg_tta.stvg_opd_frozen_mask_cache012 import frozen_forward
    from methods.decota_final_simplified_v1.objectives import prediction
    from methods.decota_final_simplified_v1.tensors import state_hash
    from scripts.run_stvg_opd_p1_softmax_recovery_002 import equal
    from vg_tta.stvg_opd_common_state_geometry_arithmetic004 import bare
    d=read(BASE/'DESIGN_LOCK.json');qual=[r for r in d['records'] if r['capture_index'] in d['qualification_indices'] and r['order']=='order1']
    chosen=qual if qualification else d['records'];assert len(qual)==4
    eb=read(B/'EXPERT_EVIDENCE_BARRIER.json');assert eb['status']=='sealed'
    if not qualification:assert read(B/'ROOT_QUALIFICATION.json')['status']=='pass'
    files={};count=0;model=None;dataset=None;cache=collections.OrderedDict();start=time.time()
    try:
        for r in chosen:
            if dataset!=r['dataset']:
                if model is not None:
                    assert state_hash(model.state_dict())==mh;del model;cache.clear();gc.collect();torch.cuda.empty_cache()
                torch.manual_seed(20260920);np.random.seed(20260920);model=model_for(r['source']);mh=model._fixed_full_state_hash;dataset=r['dataset']
            orig=ROOT/r['original_Full']['path'];assert sha(orig)==r['original_Full']['sha256'];old=load(orig)
            ip=ROOT/r['original_Full']['input_path'];assert sha(ip)==r['original_Full']['input_sha256'];inp=normalized(load(ip))
            ep=B/'expert_I1'/f"{r['capture_index']:05}.pt";assert sha(ep)==eb['files'][str(ep.relative_to(ROOT))];exz=load(ep);expert=unpack_expert(exz['expert'])
            for repeat in ([0,1] if qualification else [0]):
                row=read_row(r['dataset'],r['query_ordinal']);frames,ids,_=frames_for(r['dataset'],row,cache);shifted,pixel,spec=observation(r['dataset'],row,'clean',frames)
                batch,records,base=frozen_forward(model,shifted,row);native=prediction(base.zero['logits'],base.zero['boxes'],records,ids)
                assert pixel==inp['pixel_sha256']==exz['pixel_sha256'] and ids==inp['frame_ids']==exz['frame_ids'] and spec is None
                assert inp['source_model_state_sha256']==mh and torch.equal(native['boxes'].cpu(),inp['native_boxes']) and native['physical_interval']==inp['interval']
                del frames,shifted,batch
                for arm in ARMS:
                    torch.cuda.reset_peak_memory_stats();tick=time.perf_counter();fit,checked=fit_one(base,row,expert,old,arm,qualification)
                    p=B/('qualification' if qualification else 'formal')/f"{r['public_cell']:05}"/(arm+f'_repeat{repeat}.pt')
                    evidence=dict(**r,arm=arm,fit=fit,math_audit=checked,expert_binding=dict(path=str(ep.relative_to(ROOT)),sha256=sha(ep)),
                        Native_interval=inp['interval'],interval=exz['interval'],frame_ids=ids,pixel_sha256=pixel,source_model_sha256=mh,
                        GT_read=False,optimizer_reset=True,runtime_sha256=sha(B/'RUNTIME.json'),compute=dict(fit_seconds=time.perf_counter()-tick,
                        CUDA_peak_allocated=torch.cuda.max_memory_allocated(),actual_DINO_calls_during_fit=0,shared_expert_pack_calls=exz['actual_new_DINO_calls']))
                    save(p.parent/('UNACCEPTED_'+p.name),evidence);counts=dict(tensors=0,tensor_coordinates=0,scalar_values=0)
                    if repeat:
                        first=load(p.parent/(arm+'_repeat0.pt'));equal(bare(fit),bare(first['fit']),counts);equal(fit['geometry'],first['fit']['geometry'],counts)
                        evidence['whole_complete_repeat_bitwise']=counts
                    elif not qualification and r in qual:
                        q=load(B/'qualification'/f"{r['public_cell']:05}"/(arm+'_repeat0.pt'))
                        equal(bare(fit),bare(q['fit']),counts);equal(fit['geometry'],q['fit']['geometry'],counts);evidence['qualified_formal_bitwise']=counts
                    save(p,evidence);h=sha(p);write(p.with_suffix('.json'),dict(status='accepted_complete_spatial_fit',sha256=h,bytes=p.stat().st_size,
                        GT_read=False,new_complete_GPU_fits=1,actual_DINO_calls_during_fit=0,time=time.time()));files[str(p.relative_to(ROOT))]=h;count+=1
                    status(B/'GPU_STATUS.json',dict(status='running_complete_qualification' if qualification else 'running_complete_formal',
                        dataset=dataset,arm=arm,done=count,total=16 if qualification else 1024,pid=os.getpid(),GT_read=False,time=time.time()))
                    print('P7B_SPATIAL_PROGRESS',qualification,count,16 if qualification else 1024,dataset,arm,flush=True)
                    del fit,checked,evidence
                del base,row;gc.collect();torch.cuda.empty_cache()
            del old,exz,expert
        assert count==(16 if qualification else 1024) and state_hash(model.state_dict())==mh
        write(B/('QUALIFICATION_BARRIER.json' if qualification else 'GLOBAL_PREDICTION_BARRIER.json'),dict(status='pass' if qualification else 'sealed',
            files=files,actual_complete_GPU_fits=count,formal_cells=4 if qualification else 512,source_state_restored=True,GT_read=False,
            actual_DINO_calls_during_fits=0,expert_barrier_sha256=sha(B/'EXPERT_EVIDENCE_BARRIER.json'),elapsed_seconds=time.time()-start,time=time.time()))
    finally:
        if model is not None:del model
        cache.clear();gc.collect();torch.cuda.empty_cache();handle.close()

def qualification_root():
    verify_b();barrier=read(B/'QUALIFICATION_BARRIER.json');assert barrier['status']=='pass' and barrier['actual_complete_GPU_fits']==16
    from vg_tta.stvg_opd_common_state_geometry_arithmetic004 import bare,mathematical_readback
    from scripts.run_stvg_opd_p1_softmax_recovery_002 import equal
    checks=0;counts=dict(tensors=0,tensor_coordinates=0,scalar_values=0)
    for f,h in barrier['files'].items():
        p=ROOT/f;assert sha(p)==h;z=load(p);ep=ROOT/z['expert_binding']['path'];assert sha(ep)==z['expert_binding']['sha256']
        first=load(p.parent/(z['arm']+'_repeat0.pt'));equal(bare(z['fit']),bare(first['fit']),counts);equal(z['fit']['geometry'],first['fit']['geometry'],counts)
        checked=mathematical_readback(z['fit']);assert checked==z['math_audit']['state_Adam_audit'];checks+=checked['scalar_checks']
        assert z['math_audit']['objective_audit']['status']=='pass' and z['fit']['config']==z['config']
    write(B/'ROOT_QUALIFICATION.json',dict(status='pass',scope='all16 actual complete C/D fresh capture repeat fits/positions/before/native invariant/objective/Adam/state',
        actual_complete_GPU_fits=16,independent_scalar_checks=checks,complete_comparison=counts,GT_read=False,CPU_only=True,time=time.time()))

def controller():
    from scripts.continue_stvg_opd_temporal_guided_p7_v1 import child
    verify_b();pause(B)
    try:
        child(B,str(Path(__file__).relative_to(ROOT)),'experts')
        child(B,str(Path(__file__).relative_to(ROOT)),'qualification')
        child(B,str(Path(__file__).relative_to(ROOT)),'qualification_root',True)
        child(B,str(Path(__file__).relative_to(ROOT)),'formal')
    finally:
        if (B/'PRIORITY_PAUSE_RECEIPT.json').exists():resume(B)
    child(B,'scripts/score_stvg_opd_temporal_guided_p7b_v1.py','score',True)
    child(B,'scripts/score_stvg_opd_temporal_guided_p7b_v1.py','root',True)
    status(BASE/'STATUS.json',dict(status='P7_B_pending_actual_PDF_view_publication_archive',CPU_only=True,GT_read=True,task_complete=False,time=time.time()))

if __name__=='__main__':
    job=sys.argv[1]
    try:{'prepare':prepare,'experts':experts,'qualification':lambda:fits(True),'formal':lambda:fits(False),'qualification_root':qualification_root,'controller':controller}[job]()
    except BaseException:
        p=B/'failures'/str(time.time_ns());p.mkdir(parents=True,exist_ok=False);(p/'traceback.txt').write_text(traceback.format_exc())
        status(B/'STATUS.json',dict(status='failed_preserved_pending_actual_root',stage=job,evidence=str(p.relative_to(ROOT)),time=time.time()));raise
