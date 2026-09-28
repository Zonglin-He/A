"""One registered output-anchor factor; reuses already sealed matched controls."""
import argparse,fcntl,gc,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.desta3d_v2_p0 import read,sha,adapter_sha256
from scripts.score_desta3d_v2_aux_recovery import save_once
from scripts.desta3d_tta_run_v1 import scan_nested_gpu_receipts,tensor_sha256,make_mild_photometric_view
from scripts.desta3d_v2_tta8_recovery_v2 import save_pt,observe_time,CONDITIONS
from vg_tta.desta3d_v2_prediction_contract import validate_prediction
from scripts.desta3d_v2_tta8_output_anchor import ARMS as OLD_ARMS
PARENT=ROOT/'artifacts/desta3d_v2';BASE=PARENT/'tta_v2'
OLD=BASE/'target8_B1_output_anchor_v3';OUT=BASE/'target8_B1_temporal_anchor_v1';SOURCE=BASE/'source_moments_B1_v1'
PROBE=BASE/'output_anchor_probe_v1/probe005'
NEW_ARM='calibration_alignment_temporal_anchor';ARMS=OLD_ARMS+[NEW_ARM]
RECEIPT=PARENT/'receipts/tta8_temporal_anchor_v1.json'


def compact_native(native):
    result=dict(native)
    for key in ('event_injection','spatial_injection'):
        if result.get(key) is not None:result[key]={k:v for k,v in result[key].items() if k not in ('fields','updated_tokens')}
    return result


def register():
    assert not OUT.exists()
    audit=read(PROBE/'ROOT_RAW_READBACK.json');assert audit['status']=='passed' and not audit['GT_read']
    old=read(OLD/'CONFIG.json');cfg=dict(old)
    cfg.update(arms=ARMS,predictions=192,new_predictions=24,new_optimizer_steps=72,
       output_coefficient=audit['equal_gradient_source_only_coefficient'],
       output_coefficient_rule='unchanged prior source-only coefficient; remove coordinate term without renormalization',
       output_terms=['event'],coordinate_output_coefficient=0.,
       output_teacher='same observed B1 initial input; no clean counterpart in corrupted episode',
       output_student='observed input for output anchor; old photometric view for pre-gate objective',
       gates='frozen in both conditions; output objective does not expand update scope',
       missing_teacher_policy='missing event: pre-gate only; coordinate does not enter loss; all geometry failures retained',
       target_GT_read=False,comparison='time-only vs both-output anchor; no-output baseline retained; historically exposed development')
    save_once(OUT/'CONFIG.json',cfg);save_once(OUT/'INPUTS.json',read(OLD/'INPUTS.json'))
    pins=dict(read(OLD/'LOCK.json')['pins']);pins.update(read(OLD/'MEMORY_REPAIR.json')['pins'])
    pins.update(read(PROBE/'LOCK.json')['pins']);pins.update(read(PROBE/'MEMORY_REPAIR.json')['pins'])
    files=[Path(__file__),ROOT/'vg_tta/desta3d_v2_output_anchor_tta.py',
        ROOT/'vg_tta/desta3d_v2_output_anchor_layout_v5.py',ROOT/'vg_tta/desta3d_v2_output_anchor_checkpoint_v4.py',
        ROOT/'tests/test_desta3d_v2_temporal_anchor_tta.py',ROOT/'protocols/desta3d_v2_tta8_temporal_anchor_v1.md',
        ROOT/'vg_tta/desta3d_v2_temporal_anchor_tta.py',BASE/'temporal_anchor_CPU_PREFLIGHT.json',
        OLD/'ROOT_RAW_GRADIENT_AND_CASE_READBACK.json',OLD/'ROOT_SUMMARY_CROSSCHECK.json',
        OLD/'ALL_PREDICTIONS_SEAL.json',OLD/'COMPLETE.json',PROBE/'COMPLETE.json',PROBE/'ROOT_RAW_READBACK.json',
        OUT/'CONFIG.json',OUT/'INPUTS.json']
    for p,h in pins.items():assert sha(Path(p))==h,p
    pins.update({str(p):sha(p) for p in files});save_once(OUT/'LOCK.json',{'pins':pins})
    reuse={}
    oldseal=read(OLD/'ALL_PREDICTIONS_SEAL.json')['pins']
    for p,h in oldseal.items():assert sha(Path(p))==h,p
    for condition in CONDITIONS:
        for i in range(8):
            for filename in [a+'.pt' for a in OLD_ARMS]+['INPUT_IDENTITY.json']:
                src=OLD/'episodes'/condition/f'{i:02}'/filename
                dst=OUT/src.relative_to(OLD);dst.parent.mkdir(parents=True,exist_ok=True);os.link(src,dst)
                assert sha(dst)==oldseal[str(src)];reuse[str(dst)]={'source':str(src),'sha256':sha(dst)}
    save_once(OUT/'REUSE_SEAL.json',{'files':reuse,'predictions':168,'GT_read':False})
    save_once(OUT/'REGISTRATION.json',{'status':'registered_before_new_target_updates','time':time.time(),
        'GT_read':False,'GPU_used':False,'source_scale_probe':str(PROBE),'source_coefficient':cfg['output_coefficient'],
        'scope':'one branch removal; same8 historical parents24episodes; prior scores exposed; no new temporal-only target score'})


def run():
    assert not (OUT/'STARTED.json').exists() and not RECEIPT.exists()
    cfg=read(OUT/'CONFIG.json')
    for p,h in read(OUT/'LOCK.json')['pins'].items():assert sha(Path(p))==h,p
    reused=read(OUT/'REUSE_SEAL.json')['files']
    for p,r in reused.items():assert sha(Path(p))==r['sha256']==sha(Path(r['source']))
    lease=(ROOT/'artifacts/spatial_tta_research_v2/gpu.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    start=time.monotonic();status='running';failure=None;completed=0
    prior=sum(scan_nested_gpu_receipts(ROOT/'artifacts'/n)[0] for n in ['desta3d_v1','desta3d_v2'])
    try:
        save_once(OUT/'STARTED.json',{'pid':os.getpid(),'time':time.time(),'prior_seconds':prior})
        def guard():
            assert shutil.disk_usage(ROOT).free>=cfg['free_disk_bytes'],'disk reserve'
            assert time.monotonic()-start<cfg['phase_seconds'],'engineering allocation expired'
        guard();torch.set_num_threads(4);torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed']);torch.cuda.reset_peak_memory_stats()
        from scripts.ptd_spatial_adapter_ab_v1 import processor_load,model_load,frames_for,inputs_for
        from scripts.desta3d_v2_source_fit import prediction_record
        from scripts.desta3d_v2_reference_audit_cached_v3 import details
        from vg_tta.desta3d_v2 import Desta3DAdapterV2
        from vg_tta.desta3d_v2_ptd import capture_stock_fields
        from vg_tta.desta3d_v2_shared_reference_cached import decode_shared_reference_two_pass,_spatial_decode_official_cached
        from vg_tta.desta3d_v2_output_anchor import capture_teacher
        from vg_tta.desta3d_v2_temporal_anchor_tta import adapt_temporal_anchor as adapt_output_anchor,temporal_teacher_support as teacher_support
        pr=processor_load();model=model_load().eval().requires_grad_(False)
        import model.ptd_generation as pg
        a=Desta3DAdapterV2(hidden_dim=128,architecture='dual3d',p1_enabled=False).cuda().eval()
        initial=torch.load(cfg['checkpoint']['checkpoint'],map_location='cpu',weights_only=False)['adapter']
        digest=cfg['checkpoint']['adapter_sha256'];moments=read(SOURCE/'MOMENTS.json');raw=[Path(p) for p in reused]
        for condition in CONDITIONS:
            for index,row in enumerate(read(OUT/'INPUTS.json')):
                guard();e=OUT/'episodes'/condition/f'{index:02}'
                a.load_state_dict(initial);a.zero_grad(set_to_none=True);a.set_train_stage('frozen');assert adapter_sha256(a)==digest
                frames,ids=frames_for(row,condition);assert ids==row['input']['frame_ids']
                prompt,pre=inputs_for(row,pr,frames);view_prompt,view_pre=inputs_for(row,pr,make_mild_photometric_view(frames))
                f=capture_stock_fields(model,pr,prompt,row['input']['caption'],ids,row['input']['fps'])
                vf=capture_stock_fields(model,pr,view_prompt,row['input']['caption'],ids,row['input']['fps'])
                identity=read(e/'INPUT_IDENTITY.json')
                current={'preprocess':pre,'view_preprocess':view_pre,'visual_grid_sha':tensor_sha256(f['visual_grid']),
                   'query_tokens_sha':tensor_sha256(f['query_tokens']),'view_visual_grid_sha':tensor_sha256(vf['visual_grid']),
                   'view_query_tokens_sha':tensor_sha256(vf['query_tokens']),'frame_times_sha':tensor_sha256(f['frame_times'])}
                save_once(e/'REPLAY_INPUT.json',current);raw.append(e/'REPLAY_INPUT.json')
                assert all(identity[k]==v for k,v in current.items())
                native,trace=capture_teacher(model,pr,prompt,a,f)
                save_pt(e/'TEACHER.pt',{'native':compact_native(native),'trace':trace});raw.append(e/'TEACHER.pt')
                baseline=prediction_record(native,row,pre,digest)
                old=torch.load(e/'sourcefit_noTTA.pt',map_location='cpu',weights_only=False)
                for k in ['interval','positions','format_ok']:assert baseline[k]==old[k],k
                for k in ['boxes_cxcywh','geometry_valid']:assert torch.equal(baseline[k],old[k]),k
                if trace['branches'] and 'time' in trace['branches'][0]['logits']:
                    assert torch.equal(trace['branches'][0]['logits']['time'],old['time_distribution']['endpoint_logits'])
                if len(trace['branches'])==2 and 'coordinate' in trace['branches'][1]['logits']:
                    assert torch.equal(trace['branches'][1]['logits']['coordinate'],old['readout']['coordinate_logits'])
                support=teacher_support(native,trace);shared=native['event'].get('shared_reference_time')
                baseline_logits=(native.get('spatial') or {}).get('logits')
                del native,baseline,old;gc.collect();torch.cuda.empty_cache()
                torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
                def save_step(step,value):
                    p=e/f'ANCHOR_STEP_{step}.pt';save_pt(p,value);raw.append(p);guard()
                update=adapt_output_anchor(model,prompt,a,f,vf,trace,support,moments=moments,
                    coefficient=cfg['output_coefficient'],save_step=save_step)
                after=adapter_sha256(a)
                result,td=observe_time(pg,pr,len(ids),lambda:decode_shared_reference_two_pass(model,pr,prompt,a,f))
                pred=prediction_record(result,row,pre,after);pred.update(time_distribution=td,readout=details(result),
                    update=update,sourcefit_adapter_sha=digest)
                save_pt(e/'RAW_FINAL_BEFORE_VALIDATION.pt',pred);raw.append(e/'RAW_FINAL_BEFORE_VALIDATION.pt')
                validate_prediction(pred,len(ids));control={'defined':False,'reason':'missing sourcefit format/support'}
                if shared is not None and baseline_logits is not None:
                    tokenids=pg.build_ptd_token_ids(pr.tokenizer,max_time_tokens=len(ids))
                    fixed,injection=_spatial_decode_official_cached(model,pr,prompt,f,a,shared,tokenids)
                    changed=fixed.get('logits')
                    if changed is not None:
                        assert changed.shape==baseline_logits.shape
                        lp,lq=baseline_logits.float().log_softmax(-1),changed.float().log_softmax(-1)
                        control={'defined':True,'coordinate_logits':changed,'fixed_sourcefit_reference_time':True,
                            'KL_sourcefit_to_updated':float((lp.exp()*(lp-lq)).sum(-1).mean()),
                            'changed_elements':int((changed!=baseline_logits).sum()),
                            'injection':{k:injection[k] for k in ['relative_injection_norm','changed_elements']}}
                        del fixed,injection,changed
                pred['fixed_prefix_control']=control;p=e/(NEW_ARM+'.pt');save_pt(p,pred);raw.append(p)
                a.load_state_dict(initial);a.zero_grad(set_to_none=True);a.set_train_stage('frozen');assert adapter_sha256(a)==digest
                save_once(e/'EPISODE_COMPLETE.json',{'status':'complete','reset_exact':True,'sourcefit_adapter_sha':digest,
                    'arms':ARMS,'GT_read':False,'output_support':support,'new_steps':3,'baseline_replayed_exact':True})
                raw.append(e/'EPISODE_COMPLETE.json');completed+=1
                print('TEMPORAL_ANCHOR_EPISODE',completed,24,condition,row['key'],flush=True)
                del frames,prompt,view_prompt,f,vf,trace,baseline_logits,result,pred,update,control
                gc.collect();torch.cuda.empty_cache()
        assert completed==24
        save_once(OUT/'ALL_PREDICTIONS_SEAL.json',{'status':'all192 sealed','predictions':192,'new_predictions':24,'episodes':24,
            'parents':8,'queries':8,'GT_read':False,'pins':{str(p):sha(p) for p in raw}})
        save_once(OUT/'COMPLETE.json',{'status':'completed_predictions_unscored','predictions':192,'new_predictions':24,
            'episodes':24,'new_optimizer_steps':72,'GT_read':False,'seal_sha':sha(OUT/'ALL_PREDICTIONS_SEAL.json')})
        status='completed'
    except BaseException:
        status='failed';failure=traceback.format_exc();save_once(OUT/'FAILURE.json',{'failure':failure,'time':time.time(),'completed_episodes':completed});raise
    finally:
        seconds=time.monotonic()-start
        save_once(RECEIPT,{'status':status,'seconds':seconds,'prior_seconds':prior,'cumulative_seconds':prior+seconds,
            'cumulative_cap_seconds':None,'failure':failure,'completed_episodes':completed,
            'peak_bytes':torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else None});lease.close()


def launch():
    assert not (OUT/'STARTED.json').exists()
    start=time.monotonic();child=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'run'],cwd=ROOT)
    wall=time.monotonic()-start;worker=read(RECEIPT)['seconds'] if RECEIPT.exists() else 0.
    save_once(PARENT/'receipts/tta8_temporal_anchor_wrapper_v1.json',{'status':'completed' if child.returncode==0 else 'failed',
        'seconds':max(0.,wall-worker),'worker_seconds':worker,'child_wall_seconds':wall,'returncode':child.returncode})
    raise SystemExit(child.returncode)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['register','run','launch']);args=p.parse_args()
    {'register':register,'run':run,'launch':launch}[args.action]()
