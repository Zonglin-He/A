"""Source-only matched unlabeled versus task-CE diagnostic, fixed three steps."""
from __future__ import annotations
import argparse,fcntl,gc,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.desta3d_v2_p0 import read,sha,adapter_sha256
from scripts.score_desta3d_v2_aux_recovery import save_once
from scripts.desta3d_tta_run_v1 import scan_nested_gpu_receipts,tensor_sha256,make_mild_photometric_view
from vg_tta.optimizer_checkpoint import cpu_clone
BASE=ROOT/'artifacts/desta3d_v2'
SOURCE=BASE/'tta_v2/source_moments_B1_v1'
OUT=BASE/'tta_v2/source_task_control_v2'
RECEIPT=BASE/'receipts/source_task_control_v2.json'
PANEL=BASE/'aux_backflow_v1/label_audit/PANEL.json'
RECORDS=ROOT/'artifacts/desta3d_v1/source_fit/SOURCE_TRAIN_RECORDS.json'
ARMS=['no_update','unlabeled','supervised']


def save_pt(p,x):
    assert not p.exists();p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix('.tmp');torch.save(cpu_clone(x),temp);os.replace(temp,p)


def register():
    assert not OUT.exists()
    panel=read(PANEL);roster=sorted(read(SOURCE/'INPUTS.json'),key=lambda r:r['key'])
    bykey={r['key']:{**r,'source_moments_index':i} for i,r in enumerate(roster)}
    rows=[bykey[k] for k in panel['keys']];records={r['key']:r for r in read(RECORDS)}
    assert len(rows)==len({r['source'] for r in rows})==16
    for r in rows:
        x=records[r['key']]
        assert r['split']==x['split']=='train' and x['response_eligible']
        assert r['source']==x['source'] and r['input']['frame_ids']==x['frame_ids']
        assert r['input']['video_sha256']==x['video_sha256']
    assert shutil.disk_usage(ROOT).free-320_000_000>8*2**30
    cfg={'checkpoint':read(SOURCE/'CONFIG.json')['checkpoint'],'queries':16,'parents':16,
        'arms':ARMS,'condition':'clean_source_training','panel':str(PANEL),'seed':20260927,
        'execution_policy':{'torch_deterministic_algorithms':True,'cudnn_deterministic':True,'CUBLAS_WORKSPACE_CONFIG':':4096:8'},
        'parameter_count':66816,'optimizer':{'name':'AdamW','lr':1e-5,'weight_decay':0.,'clip':1.,'steps':3},
        'unlabeled':{'view':'brightness1.05_contrast0.95','alignment':.01,'other_weights':1.,'joint':0.,'output_anchor':0.},
        'supervised':'event semantic/time mean CE plus spatial box mean CE; no auxiliary BCE',
        'source_GT_used':'supervised arm and task diagnostics only; no input to unlabeled update function',
        'target_inputs_read':False,'target_GT_read':False,'native_decoder':'shared reference cached v3',
        'selection':'fixed3 final; no GT or loss-based choice','phase_seconds':3600,
        'free_disk_bytes':8*2**30,'cumulative_cap_seconds':None,'resume':'preserve partials; no automatic replay',
        'predictions':48,'optimizer_steps':96,'storage_estimate_bytes':320_000_000}
    save_once(OUT/'CONFIG.json',cfg);save_once(OUT/'INPUTS.json',rows)
    save_once(OUT/'SOURCE_RECORDS.json',[records[r['key']] for r in rows])
    pre=BASE/'tta_v2/source_task_control_CPU_PREFLIGHT.json'
    assert read(pre)['status']=='passed'
    paths=[Path(__file__),ROOT/'vg_tta/desta3d_v2_source_task_control.py',
        ROOT/'tests/test_desta3d_v2_source_task_control.py',ROOT/'protocols/desta3d_v2_source_task_control_v2.md',pre,
        BASE/'tta_v2/source_task_repeat_probe_v2/ROOT_REPEATABILITY_READBACK.json',
        ROOT/'vg_tta/desta3d_v2.py',ROOT/'vg_tta/desta3d_v2_ptd.py',ROOT/'vg_tta/desta3d_v2_source.py',
        ROOT/'vg_tta/desta3d_v2_tta_pilot.py',ROOT/'vg_tta/desta3d_v2_tta_objective.py',
        ROOT/'vg_tta/desta3d_v2_prediction_contract.py',ROOT/'vg_tta/desta3d_v2_shared_reference_cached.py',
        ROOT/'vg_tta/optimizer_checkpoint.py',ROOT/'scripts/desta3d_v2_source_fit.py',
        ROOT/'scripts/desta3d_source_fit_v1.py',ROOT/'scripts/ptd_8b_teacher_feasibility_v1.py',
        ROOT/'scripts/ptd_spatial_adapter_ab_v1.py',ROOT/'scripts/desta3d_tta_run_v1.py',
        ROOT/'scripts/desta3d_v2_tta8_recovery_v2.py',ROOT/'scripts/desta3d_v2_reference_audit_cached_v3.py',
        ROOT/'external/ParallelTubeDecoding/src/model/ptd_generation.py',ROOT/'methods/CURRENT_METHOD.json',
        Path(cfg['checkpoint']['checkpoint']),PANEL,RECORDS,SOURCE/'MOMENTS.json',SOURCE/'COMPLETE.json',
        SOURCE/'INPUTS.json',OUT/'CONFIG.json',OUT/'INPUTS.json',OUT/'SOURCE_RECORDS.json']
    paths += [SOURCE/'queries'/f"{r['source_moments_index']:03}.pt" for r in rows]
    save_once(OUT/'LOCK.json',{'pins':{str(p):sha(p) for p in paths}})
    save_once(OUT/'REGISTRATION.json',{'time':time.time(),'status':'registered_before_GPU',
        'source_training_labels_read':True,'target_inputs_read':False,'target_GT_read':False,
        'scope':'same calibration interface, supervised source diagnostic only',
        'roster_queries':16,'roster_parents':16,'CE_eligible':16})
    print('REGISTERED',OUT,flush=True)


def compare(a,b):
    a=a.double();b=b.double();den=float(a.norm()*b.norm())
    return {'dot':float(a.dot(b)),'cosine':float(a.dot(b))/den if den else None,
        'left_norm':float(a.norm()),'right_norm':float(b.norm())}


def run():
    cfg=read(OUT/'CONFIG.json');assert not (OUT/'STARTED.json').exists()
    for p,h in read(OUT/'LOCK.json')['pins'].items():assert sha(Path(p))==h,p
    lease=(ROOT/'artifacts/spatial_tta_research_v2/gpu.lock').open('a')
    fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    start=time.monotonic();status='running';failure=None;steps_done=0;episodes=0
    prior=sum(scan_nested_gpu_receipts(ROOT/'artifacts'/n)[0] for n in ['desta3d_v1','desta3d_v2'])
    try:
        save_once(OUT/'STARTED.json',{'time':time.time(),'pid':os.getpid(),'prior_seconds':prior})
        def guard():
            assert shutil.disk_usage(ROOT).free>=cfg['free_disk_bytes'],'disk reserve'
            assert time.monotonic()-start<cfg['phase_seconds'],'engineering phase limit; retain partials'
        guard();torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True;torch.set_num_threads(4);torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
        torch.cuda.reset_peak_memory_stats()
        from scripts.ptd_spatial_adapter_ab_v1 import processor_load,model_load,frames_for,inputs_for
        from scripts.desta3d_source_fit_v1 import _training_inputs
        from scripts.desta3d_v2_source_fit import prediction_record
        from scripts.desta3d_v2_reference_audit_cached_v3 import details
        from scripts.desta3d_v2_tta8_recovery_v2 import observe_time
        from vg_tta.desta3d_v2 import Desta3DAdapterV2
        from vg_tta.desta3d_v2_ptd import capture_stock_fields
        from vg_tta.desta3d_v2_tta_pilot import configure,forward
        from vg_tta.desta3d_v2_tta_objective import calibration_objective,CalibrationWeights
        from vg_tta.desta3d_v2_source_task_control import active,task_signals,unlabeled_signals,adam_step
        from vg_tta.desta3d_v2_shared_reference_cached import decode_shared_reference_two_pass
        from vg_tta.desta3d_v2_prediction_contract import validate_prediction
        processor=processor_load();model=model_load().eval().requires_grad_(False)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        assert not any(isinstance(m,torch.nn.Dropout) and m.p>0 for m in model.modules())
        import model.ptd_generation as pg
        adapter=Desta3DAdapterV2(hidden_dim=128,architecture='dual3d',p1_enabled=False).cuda().eval()
        initial=torch.load(cfg['checkpoint']['checkpoint'],map_location='cpu',weights_only=False)['adapter']
        adapter.load_state_dict(initial);adapter.set_train_stage('frozen')
        digest=adapter_sha256(adapter);assert digest==cfg['checkpoint']['adapter_sha256']
        moments=read(SOURCE/'MOMENTS.json');records={r['key']:r for r in read(OUT/'SOURCE_RECORDS.json')}
        allpaths=[]
        for index,row in enumerate(read(OUT/'INPUTS.json')):
            guard();episode=OUT/'episodes'/f'{index:02}';episodepaths=[]
            adapter.load_state_dict(initial);adapter.zero_grad(set_to_none=True);adapter.set_train_stage('frozen')
            assert adapter_sha256(adapter)==digest;model.eval()
            frames,ids=frames_for(row,'clean');prompt,pre=inputs_for(row,processor,frames)
            vp,vpre=inputs_for(row,processor,make_mild_photometric_view(frames))
            args=(row['input']['caption'],ids,row['input']['fps'])
            fields=capture_stock_fields(model,processor,prompt,*args)
            view=capture_stock_fields(model,processor,vp,*args)
            data,_,dpre=_training_inputs(processor,model,row,records[row['key']])
            assert data is not None and dpre==pre
            assert ids==row['input']['frame_ids'] and pre['grid']==vpre['grid']
            assert torch.equal(fields['frame_times'],view['frame_times'])
            src=torch.load(SOURCE/'queries'/f"{row['source_moments_index']:03}.pt",map_location='cpu',weights_only=False)
            assert src['preprocess']==pre and src['key']==row['key']
            ident={'key':row['key'],'source':row['source'],'frame_ids':ids,'video_sha':row['input']['video_sha256'],
                'preprocess':pre,'view_preprocess':vpre,'initial_adapter_sha':digest,
                'source_training_labels_for_task_only':True,'target_inputs_read':False}
            for k in ['visual_grid','query_tokens','frame_times']:
                ident[k+'_sha']=tensor_sha256(fields[k]);ident['view_'+k+'_sha']=tensor_sha256(view[k])
                assert ident[k+'_sha']==src[k+'_sha'],k
            from vg_tta.desta3d_v2_source import split_source_loss_masks
            masks=split_source_loss_masks(data,processor.tokenizer)
            ident['task_token_counts']={b:int(masks[b].sum()) for b in ['event','spatial']}
            ident['teacher_forced_input_sha']=tensor_sha256(data['input_ids'])
            p=episode/'INPUT_IDENTITY.json';save_once(p,ident);episodepaths.append(p)
            def native(arm):
                model.eval();adapter.eval()
                result,td=observe_time(pg,processor,len(ids),lambda:decode_shared_reference_two_pass(model,processor,prompt,adapter,fields))
                pred=prediction_record(result,row,pre,adapter_sha256(adapter))
                pred.update(arm=arm,readout=details(result),time_distribution=td,sourcefit_adapter_sha=digest,
                    GT_read=True,decoder_uses_GT=False,source_training_labels_used_for_diagnostics=True,
                    source_labels_used_for_update=arm=='supervised',unlabeled_update_uses_GT=False,target_GT_read=False)
                p=episode/f'{arm}.pt';save_pt(p,pred);episodepaths.append(p)
                validate_prediction(pred,len(ids));return pred
            native('no_update')
            base_task=None
            for arm in ['unlabeled','supervised']:
                guard();adapter.load_state_dict(initial);adapter.zero_grad(set_to_none=True)
                assert adapter_sha256(adapter)==digest
                torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
                anchor,count=configure(adapter,'calibration');names,params=active(adapter)
                assert count==66816
                with torch.no_grad():teacher=forward(adapter,fields)
                opt=torch.optim.AdamW(params,lr=1e-5,weight_decay=0.)
                history=[]
                for step in range(1,4):
                    guard()
                    task,tinfo=task_signals(model,processor,adapter,fields,data)
                    unlabeled,uinfo=unlabeled_signals(adapter,view,teacher,anchor,moments)
                    if step==1:
                        if base_task is None:base_task={k:v.clone() for k,v in task.items()}
                        else:
                            repeat={k:torch.equal(v,base_task[k]) for k,v in task.items()}
                            if not all(repeat.values()):
                                save_pt(episode/'FAILED_INITIAL_REPEAT.pt',{'first':base_task,'second':task,'task_info':tinfo})
                            p=episode/'INITIAL_REPEAT_CHECK.json';save_once(p,repeat);episodepaths.append(p)
                            assert all(repeat.values()),'initial gradients must repeat; raw failure saved'
                    applied=unlabeled['unlabeled_total'] if arm=='unlabeled' else task['task_total']
                    delta,update=adam_step(adapter,opt,applied,step=step);steps_done+=1
                    update.update(task_dot_delta=compare(task['task_total'],delta),
                        task_vs_unlabeled=compare(task['task_total'],unlabeled['unlabeled_total']))
                    raw={'task_event':task['task_event'],'task_spatial':task['task_spatial'],**unlabeled,'actual_delta':delta}
                    payload={'key':row['key'],'source':row['source'],'arm':arm,'step':step,
                        'ordered_names':names,'parameter_count':count,'raw':raw,'task':tinfo,'unlabeled':uinfo,'update':update,
                        'applied_gradient':'unlabeled_total' if arm=='unlabeled' else 'task_event+task_spatial',
                        'backbone_frozen':all(not p.requires_grad and p.grad is None for p in model.parameters()),
                        'source_labels_used_for_update':arm=='supervised','target_inputs_read':False}
                    assert payload['backbone_frozen']
                    p=episode/arm/f'step{step}.pt';save_pt(p,payload);episodepaths.append(p)
                    history.append({k:v for k,v in payload.items() if k!='raw'})
                    del task,unlabeled,payload,raw,applied,delta
                _,after=task_signals(model,processor,adapter,fields,data,gradients=False)
                with torch.no_grad():
                    ufinal,ufinalterms=calibration_objective(adapter,forward(adapter,view),teacher,anchor,
                        weights=CalibrationWeights(alignment=.01),source_moments=moments)
                changes={n:not torch.equal(v.cpu(),initial[n]) for n,v in adapter.state_dict().items()}
                assert all(not changed or n in names for n,changed in changes.items())
                final={n:p.detach().cpu() for n,p in adapter.named_parameters() if p.requires_grad}
                p=episode/arm/'FINAL_CALIBRATION.pt';save_pt(p,final);episodepaths.append(p)
                afterhash=adapter_sha256(adapter)
                adapter.zero_grad(set_to_none=True);adapter.set_train_stage('frozen')
                native(arm)
                summary={'key':row['key'],'arm':arm,'steps':3,'history':history,'task_after':after,
                    'unlabeled_after':float(ufinal),'unlabeled_terms_after':{k:float(v) for k,v in ufinalterms.items()},
                    'changed_tensors':changes,'after_adapter_sha':afterhash,'gates_frozen':True,'initial_adapter_sha':digest}
                p=episode/arm/'SUMMARY.json';save_once(p,summary);episodepaths.append(p)
                del teacher,opt,params,anchor,history,ufinal,ufinalterms
            adapter.load_state_dict(initial);adapter.zero_grad(set_to_none=True);adapter.set_train_stage('frozen')
            assert adapter_sha256(adapter)==digest
            p=episode/'EPISODE_COMPLETE.json';save_once(p,{'key':row['key'],'source':row['source'],'reset_exact':True,
                'optimizer_steps':6,'predictions':3,'pins':{str(p):sha(p) for p in episodepaths}})
            allpaths+=episodepaths+[p];episodes+=1
            print('SOURCE_TASK_EPISODE',episodes,16,row['key'],'STEPS',steps_done,'EXACT_RESET',flush=True)
            del fields,view,prompt,vp,frames,data,src,base_task;gc.collect()
        save_once(OUT/'ALL_PREDICTIONS_SEAL.json',{'pins':{str(p):sha(p) for p in allpaths},'queries':16,'parents':16,
            'predictions':48,'optimizer_steps':steps_done,'source_labels_used_for_supervised_and_diagnostics':True,
            'target_inputs_read':False,'target_GT_read':False,'native_scoring_not_run':True})
        save_once(OUT/'COMPLETE.json',{'status':'completed_source_diagnostic','predictions':48,'optimizer_steps':96,
            'episodes':episodes,'seal_sha':sha(OUT/'ALL_PREDICTIONS_SEAL.json'),'final_reset_exact':True})
        status='completed'
    except BaseException:
        status='failed';failure=traceback.format_exc();save_once(OUT/'FAILURE.json',{'time':time.time(),'failure':failure,
            'actual_steps_executed':steps_done,'complete_episodes':episodes});raise
    finally:
        seconds=time.monotonic()-start
        save_once(RECEIPT,{'status':status,'seconds':seconds,'prior_seconds':prior,'cumulative_seconds':prior+seconds,
            'actual_steps_executed':steps_done,'complete_episodes':episodes,'failure':failure,'cumulative_cap_seconds':None,
            'peak_bytes':torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else None})
        lease.close()


def launch():
    assert not RECEIPT.exists() and not (OUT/'STARTED.json').exists()
    t=time.monotonic();child=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'run'],cwd=ROOT)
    wall=time.monotonic()-t;worker=read(RECEIPT)['seconds'] if RECEIPT.exists() else 0.
    save_once(BASE/'receipts/source_task_control_wrapper_v2.json',{'status':'completed' if child.returncode==0 else 'failed',
        'seconds':max(0.,wall-worker),'worker_seconds':worker,'child_wall_seconds':wall,'returncode':child.returncode,
        'scope':'nonoverlapping launch/import/finalization overhead','cumulative_cap_seconds':None})
    raise SystemExit(child.returncode)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['register','run','launch'])
    {'register':register,'run':run,'launch':launch}[p.parse_args().action]()
