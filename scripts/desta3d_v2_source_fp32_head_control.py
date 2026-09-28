"""Fixed source16 task-head precision contrast, one new supervised arm."""
import argparse,fcntl,gc,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.desta3d_v2_p0 import read,sha,adapter_sha256
from scripts.score_desta3d_v2_aux_recovery import save_once
from scripts.desta3d_tta_run_v1 import tensor_sha256,scan_nested_gpu_receipts
from vg_tta.desta3d_v2_cast_probe import serialize_with_guard
from vg_tta.optimizer_checkpoint import cpu_clone
BASE=ROOT/'artifacts/desta3d_v2';OLD=BASE/'tta_v2/source_task_control_v2';PROBE=BASE/'tta_v2/source_fp32_head_probe_v1'
OUT=BASE/'tta_v2/source_fp32_head_control_v1';RECEIPT=BASE/'receipts/source_fp32_head_control_v1.json'
NEW='supervised_fp32';ARMS=['no_update','unlabeled','supervised',NEW]


def put(p,x,cfg):
    assert not p.exists();p.parent.mkdir(parents=True,exist_ok=True)
    blob=serialize_with_guard(cpu_clone(x),used_bytes=sum(q.stat().st_size for q in OUT.rglob('*') if q.is_file()),
        free_bytes=shutil.disk_usage(ROOT).free,cap_bytes=cfg['storage_cap_bytes'],reserve_bytes=cfg['disk_reserve_bytes'])
    with p.with_suffix('.tmp').open('xb') as f:f.write(blob);f.flush();os.fsync(f.fileno())
    os.replace(p.with_suffix('.tmp'),p)


def same(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a.cpu(),b.cpu())
    if isinstance(a,dict):return isinstance(b,dict) and set(a)==set(b) and all(same(v,b[k]) for k,v in a.items())
    if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b


def register():
    assert not (OUT/'REGISTRATION.json').exists()
    assert read(PROBE/'ROOT_RAW_READBACK.json')['status']=='passed'
    assert read(OUT/'CPU_PREFLIGHT.json')['status']=='passed'
    for p,h in read(OLD/'ALL_PREDICTIONS_SEAL.json')['pins'].items():assert sha(Path(p))==h,p
    rows=read(OLD/'INPUTS.json');labels=read(OLD/'SOURCE_RECORDS.json');assert len(rows)==len({r['source'] for r in rows})==16
    cfg={'checkpoint':read(OLD/'CONFIG.json')['checkpoint'],'seed':20260927,'arms':ARMS,'new_arm':NEW,'queries':16,'parents':16,
        'optimizer':{'name':'AdamW','lr':1e-5,'weight_decay':0.,'clip':1.,'steps':3},'parameter_count':66816,
        'phase_seconds':3600,'disk_reserve_bytes':8*2**30,'storage_cap_bytes':220_000_000,'cumulative_cap_seconds':None,
        'new_optimizer_steps':48,'new_final_predictions':16,'scoreable_predictions':64,'additional_baseline_replays':16,
        'target_inputs_read':False,'target_GT_read':False,'source_GT':'supervised source training and diagnostic/scoring only',
        'change':'only task loss final frozen head F.linear(h.float(),weight.float()); body/native BF16 unchanged',
        'deterministic':True,'selection':'fixed3 final, no GT/loss choice','resume':'preserve partials, no automatic replay'}
    assert shutil.disk_usage(ROOT).free-cfg['storage_cap_bytes']>cfg['disk_reserve_bytes']
    save_once(OUT/'CONFIG.json',cfg);save_once(OUT/'INPUTS.json',rows);save_once(OUT/'SOURCE_RECORDS.json',labels)
    reuse={}
    for i in range(16):
        for a in ARMS[:3]:
            src=OLD/'episodes'/f'{i:02}'/(a+'.pt');dst=OUT/'episodes'/f'{i:02}'/(a+'.pt')
            dst.parent.mkdir(parents=True,exist_ok=True);os.link(src,dst);assert sha(src)==sha(dst);reuse[str(dst)]=sha(dst)
    save_once(OUT/'REUSE_SEAL.json',{'pins':reuse,'original_run':str(OLD),'predictions':48,'original_seal_sha':sha(OLD/'ALL_PREDICTIONS_SEAL.json')})
    paths=[Path(__file__),ROOT/'vg_tta/desta3d_v2_fp32_task_head.py',ROOT/'protocols/desta3d_v2_source_fp32_head_control_v1.md',
        ROOT/'scripts/score_desta3d_v2_source_fp32_head_control.py',ROOT/'scripts/crosscheck_desta3d_v2_source_fp32_control.py',
        PROBE/'ROOT_RAW_READBACK.json',PROBE/'ROOT_COMPLETION_SUMMARY.json',OUT/'CPU_PREFLIGHT.json',
        OUT/'CONFIG.json',OUT/'INPUTS.json',OUT/'SOURCE_RECORDS.json',OUT/'REUSE_SEAL.json',OLD/'ALL_PREDICTIONS_SEAL.json']
    pins=dict(read(OLD/'LOCK.json')['pins']);pins.update({str(p):sha(p) for p in paths})
    # Freeze original final states and baseline step evidence reused for dual-objective comparisons.
    for i in range(16):
        for name in ['INPUT_IDENTITY.json','supervised/FINAL_CALIBRATION.pt','supervised/SUMMARY.json','supervised/step1.pt']:
            p=OLD/'episodes'/f'{i:02}'/name;pins[str(p)]=sha(p)
    save_once(OUT/'LOCK.json',{'pins':pins});save_once(OUT/'REGISTRATION.json',{'time':time.time(),'status':'registered_before_GPU',
        'source_queries':16,'source_parents':16,'new_steps':48,'target_data':False,'fixed_source_roster':True})
    print('REGISTERED',OUT,flush=True)


def run():
    cfg=read(OUT/'CONFIG.json');assert not (OUT/'STARTED.json').exists()
    for p,h in read(OUT/'LOCK.json')['pins'].items():assert sha(Path(p))==h,p
    lease=(ROOT/'artifacts/spatial_tta_research_v2/gpu.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    start=time.monotonic();status='failed';steps=0;episodes=0
    prior=sum(scan_nested_gpu_receipts(ROOT/'artifacts'/n)[0] for n in ['desta3d_v1','desta3d_v2'])
    try:
        save_once(OUT/'STARTED.json',{'time':time.time(),'pid':os.getpid(),'prior_seconds':prior})
        def guard():
            assert time.monotonic()-start<cfg['phase_seconds'] and shutil.disk_usage(ROOT).free>=cfg['disk_reserve_bytes']
        torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True
        torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed']);torch.cuda.reset_peak_memory_stats()
        assert not torch.backends.cuda.matmul.allow_tf32
        from scripts.ptd_spatial_adapter_ab_v1 import processor_load,model_load,frames_for,inputs_for
        from scripts.desta3d_source_fit_v1 import _training_inputs
        from scripts.desta3d_v2_source_fit import prediction_record
        from scripts.desta3d_v2_reference_audit_cached_v3 import details
        from scripts.desta3d_v2_tta8_recovery_v2 import observe_time
        from vg_tta.desta3d_v2 import Desta3DAdapterV2
        from vg_tta.desta3d_v2_ptd import capture_stock_fields
        from vg_tta.desta3d_v2_tta_pilot import configure
        from vg_tta.desta3d_v2_source_task_control import active,task_signals,adam_step
        from vg_tta.desta3d_v2_fp32_task_head import task_signals_fp32
        from vg_tta.desta3d_v2_shared_reference_cached import decode_shared_reference_two_pass
        from vg_tta.desta3d_v2_prediction_contract import validate_prediction
        from vg_tta.desta3d_v2_source import split_source_loss_masks
        guard();pr=processor_load();model=model_load().eval().requires_grad_(False)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        import model.ptd_generation as pg
        initial=torch.load(cfg['checkpoint']['checkpoint'],map_location='cpu',weights_only=False)['adapter']
        adapter=Desta3DAdapterV2(hidden_dim=128,architecture='dual3d',p1_enabled=False).cuda().eval()
        adapter.load_state_dict(initial);adapter.set_train_stage('frozen');digest=adapter_sha256(adapter)
        assert digest==cfg['checkpoint']['adapter_sha256'];original_head=model.lm_head.forward.__func__
        labels={r['key']:r for r in read(OUT/'SOURCE_RECORDS.json')};paths=[]
        for index,row in enumerate(read(OUT/'INPUTS.json')):
            guard();ep=OUT/'episodes'/f'{index:02}';olde=OLD/'episodes'/f'{index:02}';epaths=[]
            adapter.load_state_dict(initial);adapter.zero_grad(set_to_none=True);adapter.set_train_stage('frozen');model.eval()
            assert adapter_sha256(adapter)==digest
            frames,ids=frames_for(row,'clean');prompt,pre=inputs_for(row,pr,frames)
            fields=capture_stock_fields(model,pr,prompt,row['input']['caption'],ids,row['input']['fps'])
            data,_,dpre=_training_inputs(pr,model,row,labels[row['key']]);oldident=read(olde/'INPUT_IDENTITY.json')
            assert pre==dpre==oldident['preprocess'] and ids==oldident['frame_ids']
            assert torch.equal(data['pixel_values_videos'],prompt['pixel_values_videos'])
            ident={k:oldident[k] for k in ['key','source','frame_ids','video_sha','preprocess','initial_adapter_sha']}
            for k in ['visual_grid','query_tokens','frame_times']:
                ident[k+'_sha']=tensor_sha256(fields[k]);assert ident[k+'_sha']==oldident[k+'_sha']
            ident['teacher_forced_input_sha']=tensor_sha256(data['input_ids']);assert ident['teacher_forced_input_sha']==oldident['teacher_forced_input_sha']
            masks=split_source_loss_masks(data,pr.tokenizer);ident['task_token_counts']={b:int(masks[b].sum()) for b in ['event','spatial']}
            assert ident['task_token_counts']==oldident['task_token_counts']
            ident['support_sha']={k:tensor_sha256(v) for k,v in data.items() if isinstance(v,torch.Tensor) and k!='pixel_values_videos'}
            save_once(ep/'INPUT_IDENTITY.json',ident);epaths.append(ep/'INPUT_IDENTITY.json')
            def native(arm):
                assert model.lm_head.forward.__func__ is original_head and 'forward' not in model.lm_head.__dict__
                model.eval();adapter.eval()
                result,td=observe_time(pg,pr,len(ids),lambda:decode_shared_reference_two_pass(model,pr,prompt,adapter,fields))
                p=prediction_record(result,row,pre,adapter_sha256(adapter));p.update(arm=arm,readout=details(result),time_distribution=td,
                    sourcefit_adapter_sha=digest,GT_read=True,decoder_uses_GT=False,source_training_labels_used_for_diagnostics=True,
                    source_labels_used_for_update=arm==NEW,unlabeled_update_uses_GT=False,target_GT_read=False)
                validate_prediction(p,len(ids));return p
            replay=native('no_update');put(ep/'REPLAY_NO_UPDATE.pt',replay,cfg);epaths.append(ep/'REPLAY_NO_UPDATE.pt')
            baseline=torch.load(ep/'no_update.pt',map_location='cpu',weights_only=False)
            assert same(replay,baseline),'B1 full native/logits replay mismatch'
            # Old BF16-trained endpoint read on the new objective, never updated and no new native.
            oldfinal=torch.load(olde/'supervised/FINAL_CALIBRATION.pt',map_location='cpu',weights_only=False)
            adapter.load_state_dict({**initial,**oldfinal});configure(adapter,'calibration')
            _,old_fp32=task_signals_fp32(model,pr,adapter,fields,data,gradients=False)
            old_native=torch.load(ep/'supervised.pt',map_location='cpu',weights_only=False)
            assert adapter_sha256(adapter)==old_native['adapter_sha']
            old_bf16=read(olde/'supervised/SUMMARY.json')['task_after']
            save_once(ep/'OLD_SUPERVISED_DUAL_CE.json',{'adapter_sha':adapter_sha256(adapter),'BF16':old_bf16,'FP32':old_fp32,'old_BF16_CE_reused':True})
            epaths.append(ep/'OLD_SUPERVISED_DUAL_CE.json')
            adapter.load_state_dict(initial);adapter.zero_grad(set_to_none=True);configure(adapter,'calibration');names,params=active(adapter)
            torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
            opt=torch.optim.AdamW(params,lr=1e-5,weight_decay=0.);history=[]
            for step in range(1,4):
                guard();_,bf16=task_signals(model,pr,adapter,fields,data,gradients=False)
                task,fp32=task_signals_fp32(model,pr,adapter,fields,data)
                if step==1:
                    oldstep=torch.load(olde/'supervised/step1.pt',map_location='cpu',weights_only=False)
                    assert bf16==oldstep['task'],'initial original task CE or injection mismatch'
                delta,update=adam_step(adapter,opt,task['task_total'],step=step);steps+=1
                update['task_dot_actual_delta']=float(task['task_total'].double().dot(delta.double()))
                payload={'key':row['key'],'source':row['source'],'step':step,'ordered_names':names,'parameter_count':66816,
                    'raw':{'task_event':task['task_event'],'task_spatial':task['task_spatial'],'task_total':task['task_total'],'actual_delta':delta},
                    'BF16':bf16,'FP32':fp32,'update':update,'backbone_frozen':all(not p.requires_grad and p.grad is None for p in model.parameters()),
                    'only_calibration_trainable':all(p.requires_grad==(n in names) for n,p in adapter.named_parameters()),
                    'source_supervised_update':True,'target_data':False}
                assert payload['backbone_frozen'] and payload['only_calibration_trainable']
                p=ep/NEW/f'step{step}.pt';put(p,payload,cfg);epaths.append(p)
                history.append({k:v for k,v in payload.items() if k!='raw'})
                del task,payload,delta
            _,bf_after=task_signals(model,pr,adapter,fields,data,gradients=False)
            _,fp_after=task_signals_fp32(model,pr,adapter,fields,data,gradients=False)
            final={n:p.detach().cpu().clone() for n,p in adapter.named_parameters() if p.requires_grad}
            changes={n:not torch.equal(v.cpu(),initial[n]) for n,v in adapter.state_dict().items()}
            assert all(not c or n in names for n,c in changes.items())
            p=ep/NEW/'FINAL_CALIBRATION.pt';put(p,final,cfg);epaths.append(p)
            p=ep/NEW/'FINAL_OPTIMIZER.pt';put(p,opt.state_dict(),cfg);epaths.append(p)
            aftersha=adapter_sha256(adapter);adapter.set_train_stage('frozen');adapter.zero_grad(set_to_none=True)
            pred=native(NEW);p=ep/(NEW+'.pt');put(p,pred,cfg);epaths.append(p)
            save_once(ep/NEW/'SUMMARY.json',{'key':row['key'],'source':row['source'],'steps':3,'history':history,
                'BF16_after':bf_after,'FP32_after':fp_after,'changed_tensors':changes,'after_adapter_sha':aftersha,'initial_adapter_sha':digest,
                'head_restored_for_native':True,'gates_frozen':True});epaths.append(ep/NEW/'SUMMARY.json')
            adapter.load_state_dict(initial);adapter.zero_grad(set_to_none=True);adapter.set_train_stage('frozen');assert adapter_sha256(adapter)==digest
            save_once(ep/'EPISODE_COMPLETE.json',{'key':row['key'],'source':row['source'],'reset_exact':True,'new_steps':3,
                'new_final_predictions':1,'original_B1_native_exact':True,'pins':{str(p):sha(p) for p in epaths}})
            paths+=epaths+[ep/'EPISODE_COMPLETE.json'];episodes+=1
            print('FP32_CONTROL_EPISODE',episodes,16,row['key'],'STEPS',steps,'EXACT_RESET',flush=True)
            del opt,params,frames,prompt,data,fields,replay,baseline,oldfinal,old_native,oldstep,history,final;gc.collect();torch.cuda.empty_cache()
        pins=dict(read(OUT/'REUSE_SEAL.json')['pins']);pins.update({str(p):sha(p) for p in paths})
        save_once(OUT/'ALL_PREDICTIONS_SEAL.json',{'pins':pins,'episodes':16,'parents':16,'scoreable_predictions':64,
            'new_optimizer_steps':steps,'baseline_replays':16,'source_supervision_disclosed':True,'target_data':False})
        save_once(OUT/'COMPLETE.json',{'status':'completed','episodes':16,'new_steps':48,'predictions':64,
            'seal_sha':sha(OUT/'ALL_PREDICTIONS_SEAL.json'),'exact_resets':True});status='completed'
    except BaseException:
        save_once(OUT/'FAILURE.json',{'time':time.time(),'error':traceback.format_exc(),'actual_steps':steps,'complete_episodes':episodes});raise
    finally:
        seconds=time.monotonic()-start
        save_once(RECEIPT,{'status':status,'seconds':seconds,'prior_seconds':prior,'cumulative_seconds':prior+seconds,
            'cap':None,'actual_steps':steps,'complete_episodes':episodes,'peak_allocated_bytes':torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else None});lease.close()


def launch():
    assert not RECEIPT.exists();start=time.monotonic()
    child=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'run'],cwd=ROOT)
    wall=time.monotonic()-start;sec=read(RECEIPT)['seconds'] if RECEIPT.exists() else 0.
    save_once(BASE/'receipts/source_fp32_head_control_wrapper_v1.json',{'status':'completed' if child.returncode==0 else 'failed',
        'seconds':max(0.,wall-sec),'child_wall_seconds':wall,'worker_seconds':sec,'cap':None,
        'scope':'nonoverlapping subprocess import/check/finalization overhead'});raise SystemExit(child.returncode)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['register','run','launch'])
    {'register':register,'run':run,'launch':launch}[p.parse_args().action]()
