"""GT-free current-cohort exact Source/gradient/decision/full-readout comparison."""
from pathlib import Path
import collections,copy,gc,json,sys,time,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import stvg_pipeline_speed_v1 as s
from scripts.prepare_stvg_spatial_controls256_v1 import BASE as CORE,SMALL,read,put,sha,guard
BASE=s.BASE
def verify():
    for p,h in read(BASE/'EXECUTION_RUNTIME.json')['pins'].items():assert sha(ROOT/p)==h,p
def candidates():
    from scripts.prepare_stvg_lite_reclocked_inputs002 import _anchor_from_teacher_selection
    jobs=read(CORE/'TEACHER_JOB_LOCK.json')['records'];out=[]
    for rec in read(CORE/'ROSTER_LOCK.json')['records']:
        if sum(x['record']['dataset']==rec['dataset'] for x in out)>=2:continue
        jj=[j for j in jobs if j['case_id']==rec['probe_alias']];pool=read(CORE/'spatial/E1/proposals'/(rec['probe_alias']+'.json'));anchors=[]
        for j in jj:
            p=CORE/'teacher/Full_E1'/(j['job_id']+'.json');d=read(p);w=next(w for w in pool['windows'] if w['slot']==j['center_slot']);a,_=_anchor_from_teacher_selection(j,d,rec['frame_ids'],w['source_grid_position'],teacher_output_sha256=sha(p))
            if a is not None:anchors.append(a)
        if anchors:out.append(dict(record=rec,anchors=anchors))
    return out
def run():
    import torch,numpy as np
    from scripts import run_stvg_adjacent_motion_student_v1 as h
    from scripts import run_stvg_table1_source_v1 as source
    from scripts.run_tastvg_evidence_vulnerability_v2 import install_clean_loader
    from scripts.run_decota_paper_main_v1 import model_for,frames_for
    from scripts.stvg_mllm_feedback_common_v1 import raw_row
    from scripts.stvg_soft_ep_coordinate_common_v1 import load_case
    from vg_tta.stvg_opd_frozen_mask_cache012 import frozen_forward,transient_native_masks
    from vg_tta.stvg_opd_frozen_visual_host011 import frozen_visual_host
    from vg_tta.stvg_opd_frozen_attention_lifetime010 import frozen_attention_storage
    from vg_tta.stvg_opd_frozen_module_cpu008 import inactive_modules_on_host
    from methods.decota_final_simplified_v1.objectives import prediction
    from methods.decota_final_simplified_v1.tensors import state_hash
    from methods.decota_final_simplified_v1.backbone import query_subject
    from scripts import stvg_target_teacher_student_v1 as core
    from scripts.stvg_spatial_lite_fit_v1 import fit
    from scripts.run_stvg_table1_student004 import stable_fit_key
    verify();sys.addaudithook(guard);install_clean_loader();torch.set_num_threads(2);torch.manual_seed(20261015);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    assert read(CORE/'teacher/Full_E1/ROOT_READBACK.json')['status']=='pass';chosen=candidates();records=[];model=None;current=None;settings=read(SMALL/'trials/A/DESIGN_LOCK.json');start=time.perf_counter();passed=True;error=None
    put(BASE/'QUALIFICATION_PANEL_LOCK.json',dict(status='locked',count=len(chosen),records=[dict(case_id=x['record']['probe_alias'],dataset=x['record']['dataset'],parent_sha256=x['record']['parent_sha256']) for x in chosen],selection='first2 roster parents/domain with actual eligible Full_E1 references, before GT; no favorable outcome choice',GT_read=False))
    try:
        for row in chosen:
            rec=row['record'];case,p=load_case(rec);transient=dict(case,source=source.SOURCE_MODEL_NAME[rec['dataset']],_source_case_sha256=case['source_case_sha256']);_,raw=source._validate_roster_record(rec);transient['parent']=raw['source'];old_catalog=h.catalog;h.catalog=lambda _: (transient,{})
            engine=s.Engine('qualification_original',False)
            try:r=engine.capture(dict(case_id=case['case_id'],dataset=case['dataset'],source_case_sha256=case['source_case_sha256']),model,current,torch,collections,model_for,frames_for,raw_row,frozen_forward,prediction,state_hash)
            finally:h.catalog=old_catalog
            model,current=r[:2];original,batch,native,ids,records0=r[9],r[8],r[10],r[6],r[7]
            s.feature_save(transient,original,records0,native);torch.cuda.synchronize();t=time.perf_counter();cached,cached_records,cached_native=s.feature_restore(model,transient);torch.cuda.synchronize();restore_secs=time.perf_counter()-t;s.compare_tree(native,cached_native,'source_native')
            ex=h._anchors(dict(anchors=row['anchors']),ids,torch);cfg=settings['configs'][rec['dataset']]['M2'];cfgset=settings['response_settings'][rec['dataset']]['M2'];key=stable_fit_key(rec['dataset'],rec['query_key_sha256']);results=[]
            for recipe in ('lite_fullstep','lite_backtrack'):
                f=fit(original,original.initial,ex,ids,key,cfg,'M2',cfgset,recipe=recipe,response_scope='full_grid',event_support_positions=None)
                g=fit(cached,cached.initial,ex,ids,key,cfg,'M2',cfgset,recipe=recipe,response_scope='full_grid',event_support_positions=None)
                s.compare_tree(f,g,'complete_fit_'+recipe)
                torch.cuda.synchronize();t=time.perf_counter();fresh=h._readout(model,batch,ids,records0,f,raw['parses']['subject'],core,torch,transient_native_masks=transient_native_masks,frozen_visual_host=frozen_visual_host,frozen_attention_storage=frozen_attention_storage,inactive_modules_on_host=inactive_modules_on_host,query_subject=query_subject);torch.cuda.synchronize();fresh_secs=time.perf_counter()-t
                torch.cuda.synchronize();t=time.perf_counter();fast=s.full_prefix_readout(cached,ids,cached_records,g['final_state']);torch.cuda.synchronize();fast_secs=time.perf_counter()-t
                s.compare_tree(fresh,fast,'final_full_both_passes_'+recipe)
                assert torch.equal(fresh['boxes'].cpu(),f['final_cached_decoder_boxes'].cpu()) and state_hash(model.state_dict())==case['source_model_state_sha256']
                results.append(dict(recipe=recipe,full_gradient_Adam_solver_acceptance_path_exact=True,full_prediction_boxes_logits_interval_exact=True,original_full_readout_seconds=fresh_secs,prefix_full_readout_seconds=fast_secs,optimizer_steps=f['optimizer_steps'],GT_read=False))
            records.append(dict(case_id=case['case_id'],dataset=case['dataset'],source_capture_seconds=r[-1],source_profile=engine.rows[0],prefix_restore_seconds=restore_secs,tests=results,GT_read=False));del original,cached,batch,r,f,g,case;gc.collect();torch.cuda.empty_cache()
    except Exception as e:passed=False;error=dict(error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc())
    qualified=passed and all(any(r['dataset']==ds for r in records) for ds in ('hc2','vidstg'))
    put(BASE/'QUALIFICATION_COMPLETION.json',dict(status='pass' if qualified else 'failed_closed_original_path_fallback',count=len(records),engineering_adopted=qualified,records=records,error=error,GT_read=False,wall_seconds=time.perf_counter()-start,new_GT_reads=0,scientific_parameters_or_acceptance_changed=False))
    print('SPEED_QUALIFICATION',qualified,len(records),error)
if __name__=='__main__':run()
