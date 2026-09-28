"""One registered identity student-view factor; reuses already sealed matched controls."""
import argparse,fcntl,gc,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.desta3d_v2_p0 import read,sha,adapter_sha256
from scripts.score_desta3d_v2_aux_recovery import save_once
from scripts.desta3d_tta_run_v1 import scan_nested_gpu_receipts,tensor_sha256
from scripts.desta3d_v2_tta8_recovery_v2 import save_pt,observe_time,CONDITIONS
from vg_tta.desta3d_v2_prediction_contract import validate_prediction
from scripts.desta3d_v2_tta8_temporal_anchor import ARMS as OLD_ARMS
PARENT=ROOT/'artifacts/desta3d_v2';BASE=PARENT/'tta_v2'
OLD=BASE/'target8_B1_temporal_anchor_v1';OUT=BASE/'target8_B1_identity_view_v1';SOURCE=BASE/'source_moments_B1_v1'
PROBE=BASE/'view_alignment_signal_source_v1'
NEW_ARM='calibration_alignment_identity_view';ARMS=OLD_ARMS+[NEW_ARM]
RECEIPT=PARENT/'receipts/tta8_identity_view_v1.json'


def register():
    assert not OUT.exists()
    assert read(PROBE/'ROOT_RAW_READBACK.json')['status']=='passed'
    cfg=dict(read(BASE/'target8_B1_v2/CONFIG.json'))
    cfg.update(arms=ARMS,predictions=216,new_predictions=24,new_optimizer_steps=72,
       student_view='observed_identity',output_anchor=False,phase_seconds=600,
       target_GT_read=False,comparison='identity view vs original mild view calibration-alignment; no output anchor',
       primary_readout='paired-parent identity minus mild calibration-alignment on clean and corruption; B1/Frozen retained')
    save_once(OUT/'CONFIG.json',cfg);save_once(OUT/'INPUTS.json',read(OLD/'INPUTS.json'))
    pins=dict(read(OLD/'LOCK.json')['pins'])
    files=[Path(__file__),ROOT/'vg_tta/desta3d_v2_identity_view_tta.py',
        ROOT/'tests/test_desta3d_v2_identity_view.py',ROOT/'protocols/desta3d_v2_tta8_identity_view_v1.md',
        BASE/'identity_view_CPU_PREFLIGHT.json', PROBE/'ROOT_RAW_READBACK.json',PROBE/'REPORT.md',
        OLD/'ROOT_SUMMARY_CROSSCHECK.json',OLD/'ALL_PREDICTIONS_SEAL.json',OLD/'COMPLETE.json',
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
    save_once(OUT/'REUSE_SEAL.json',{'files':reuse,'predictions':192,'GT_read':False})
    save_once(OUT/'REGISTRATION.json',{'status':'registered_before_new_target_updates','time':time.time(),
        'GT_read':False,'GPU_used':False,'source_signal_probe':str(PROBE),
        'historical_target_exposure':True,
        'scope':'one student view substitution; same8 historical parents24episodes; prior scores exposed; no new identity-view score'})



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
        from vg_tta.desta3d_v2_identity_view_tta import adapt_identity_view
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
                prompt,pre=inputs_for(row,pr,frames)
                f=capture_stock_fields(model,pr,prompt,row['input']['caption'],ids,row['input']['fps'])
                identity=read(e/'INPUT_IDENTITY.json')
                current={'preprocess':pre,'visual_grid_sha':tensor_sha256(f['visual_grid']),
                   'query_tokens_sha':tensor_sha256(f['query_tokens']),'frame_times_sha':tensor_sha256(f['frame_times'])}
                save_once(e/'REPLAY_INPUT.json',current);raw.append(e/'REPLAY_INPUT.json')
                assert all(identity[k]==v for k,v in current.items())
                save_once(e/'STUDENT_INPUT_IDENTITY.json',{**current,'view_preprocess':pre,
                    'student_visual_grid_sha':current['visual_grid_sha'],
                    'student_query_tokens_sha':current['query_tokens_sha'],
                    'student_frame_times_sha':current['frame_times_sha'],
                    'student_view':'observed_identity','condition':condition,'clean_counterpart_read':False})
                raw.append(e/'STUDENT_INPUT_IDENTITY.json')
                native,native_td=observe_time(pg,pr,len(ids),lambda:decode_shared_reference_two_pass(model,pr,prompt,a,f))
                baseline=prediction_record(native,row,pre,digest)
                baseline_details=details(native)
                save_pt(e/'BASELINE_REPLAY.pt',{**baseline,'time_distribution':native_td,'readout':baseline_details})
                raw.append(e/'BASELINE_REPLAY.pt')
                old=torch.load(e/'sourcefit_noTTA.pt',map_location='cpu',weights_only=False)
                for k in ['interval','positions','format_ok']:assert baseline[k]==old[k],k
                for k in ['boxes_cxcywh','geometry_valid']:assert torch.equal(baseline[k],old[k]),k
                for x,y in [(native_td['endpoint_logits'],old['time_distribution']['endpoint_logits']),
                            (baseline_details['coordinate_logits'],old['readout']['coordinate_logits'])]:
                    assert (x is None and y is None) or (x is not None and y is not None and torch.equal(x,y))
                shared=native['event'].get('shared_reference_time')
                baseline_logits=(native.get('spatial') or {}).get('logits')
                del native,baseline,old,baseline_details;gc.collect();torch.cuda.empty_cache()
                torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
                update=adapt_identity_view(a,f,moments=moments)
                assert all(not p.requires_grad and p.grad is None for p in model.parameters())
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
                    'arms':ARMS,'GT_read':False,'student_view':'observed_identity','new_steps':3,'baseline_replayed_exact':True})
                raw.append(e/'EPISODE_COMPLETE.json');completed+=1
                print('IDENTITY_VIEW_EPISODE',completed,24,condition,row['key'],flush=True)
                del frames,prompt,f,native_td,baseline_logits,result,pred,update,control
                gc.collect();torch.cuda.empty_cache()
        assert completed==24
        save_once(OUT/'ALL_PREDICTIONS_SEAL.json',{'status':'all216 sealed','predictions':216,'new_predictions':24,'episodes':24,
            'parents':8,'queries':8,'GT_read':False,'pins':{str(p):sha(p) for p in raw}})
        save_once(OUT/'COMPLETE.json',{'status':'completed_predictions_unscored','predictions':216,'new_predictions':24,
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
    save_once(PARENT/'receipts/tta8_identity_view_wrapper_v1.json',{'status':'completed' if child.returncode==0 else 'failed',
        'seconds':max(0.,wall-worker),'worker_seconds':worker,'child_wall_seconds':wall,'returncode':child.returncode})
    raise SystemExit(child.returncode)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['register','run','launch']);args=p.parse_args()
    {'register':register,'run':run,'launch':launch}[args.action]()
