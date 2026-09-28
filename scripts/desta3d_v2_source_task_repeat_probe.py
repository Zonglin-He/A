"""Isolated first-source task-gradient repeatability; no optimizer updates."""
import sys,time,traceback,fcntl,shutil
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.desta3d_v2_source_task_control import read,sha,save_once,save_pt,adapter_sha256,BASE,SOURCE
from scripts.desta3d_tta_run_v1 import scan_nested_gpu_receipts,tensor_sha256
OLD=BASE/'tta_v2/source_task_control_v1'
OUT=BASE/'tta_v2/source_task_repeat_probe_v1'
RECEIPT=BASE/'receipts/source_task_repeat_probe_v1.json'

def main():
    assert not OUT.exists()
    for p,h in read(OLD/'LOCK.json')['pins'].items():assert sha(Path(p))==h
    save_once(OUT/'REGISTRATION.json',{'time':time.time(),'status':'registered_before_GPU',
        'question':'repeat task gradients at exact B1 before/after native decoding and restoring saved calibration',
        'source_key':read(OLD/'INPUTS.json')[0]['key'],'optimizer_steps':0,'source_GT':'diagnostic task CE only',
        'target_inputs_read':False,'maximum_seconds':300,'cumulative_cap_seconds':None,
        'raw_saved_before_comparison':True,'prior_failure_second_gradient_not_saved':True,
        'pins':{str(Path(__file__)):sha(Path(__file__)),str(OLD/'LOCK.json'):sha(OLD/'LOCK.json'),
                str(OLD/'episodes/00/unlabeled/FINAL_CALIBRATION.pt'):sha(OLD/'episodes/00/unlabeled/FINAL_CALIBRATION.pt')}})
    lease=(ROOT/'artifacts/spatial_tta_research_v2/gpu.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    start=time.monotonic();prior=sum(scan_nested_gpu_receipts(ROOT/'artifacts'/n)[0] for n in ['desta3d_v1','desta3d_v2'])
    status='failed'
    try:
        assert shutil.disk_usage(ROOT).free>8*2**30
        torch.set_num_threads(4);torch.manual_seed(20260927);torch.cuda.manual_seed_all(20260927)
        from scripts.ptd_spatial_adapter_ab_v1 import processor_load,model_load,frames_for,inputs_for
        from scripts.desta3d_source_fit_v1 import _training_inputs
        from vg_tta.desta3d_v2 import Desta3DAdapterV2
        from vg_tta.desta3d_v2_ptd import capture_stock_fields
        from vg_tta.desta3d_v2_tta_pilot import configure
        from vg_tta.desta3d_v2_source_task_control import task_signals
        from vg_tta.desta3d_v2_shared_reference_cached import decode_shared_reference_two_pass
        processor=processor_load();model=model_load().eval().requires_grad_(False)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        cfg=read(OLD/'CONFIG.json');row=read(OLD/'INPUTS.json')[0];record=read(OLD/'SOURCE_RECORDS.json')[0]
        initial=torch.load(cfg['checkpoint']['checkpoint'],map_location='cpu',weights_only=False)['adapter']
        adapter=Desta3DAdapterV2(hidden_dim=128,architecture='dual3d',p1_enabled=False).cuda().eval()
        adapter.load_state_dict(initial);digest=adapter_sha256(adapter)
        frames,ids=frames_for(row,'clean');prompt,pre=inputs_for(row,processor,frames)
        fields=capture_stock_fields(model,processor,prompt,row['input']['caption'],ids,row['input']['fps'])
        data,_,dpre=_training_inputs(processor,model,row,record);assert dpre==pre
        input_hashes={k:tensor_sha256(v) for k,v in data.items() if torch.is_tensor(v)}
        saved={};summaries={}
        def measure(name):
            assert time.monotonic()-start<300
            assert adapter_sha256(adapter)==digest
            assert {k:tensor_sha256(v) for k,v in data.items() if torch.is_tensor(v)}==input_hashes
            configure(adapter,'calibration')
            before_rope=model.model.rope_deltas.clone() if model.model.rope_deltas is not None else None
            raw,info=task_signals(model,processor,adapter,fields,data)
            save_pt(OUT/(name+'.pt'),{'raw':raw,'info':info,'rope_before':before_rope,
                'rope_after':model.model.rope_deltas,'adapter_sha':adapter_sha256(adapter),'input_hashes':input_hashes})
            saved[name]=raw;summaries[name]=info
            print('MEASURED',name,info['total'],flush=True)
        def native():
            adapter.set_train_stage('frozen');model.eval()
            decode_shared_reference_two_pass(model,processor,prompt,adapter,fields)
        native();measure('after_baseline_native');measure('immediate_repeat')
        native();measure('after_same_native')
        changed=torch.load(OLD/'episodes/00/unlabeled/FINAL_CALIBRATION.pt',map_location='cpu',weights_only=False)
        adapter.load_state_dict({**initial,**changed});native()
        adapter.load_state_dict(initial);adapter.zero_grad(set_to_none=True)
        measure('after_changed_native_reset');measure('reset_immediate_repeat')
        old=torch.load(OLD/'episodes/00/unlabeled/step1.pt',map_location='cpu',weights_only=False)['raw']
        pairs=[('after_baseline_native',n) for n in saved if n!='after_baseline_native']
        old['task_total']=old['task_event']+old['task_spatial'];saved['original_v1_step1']=old
        pairs.append(('original_v1_step1','after_baseline_native'))
        comparison={}
        for a,b in pairs:
            comparison[a+'__'+b]={}
            for k in ['task_event','task_spatial','task_total']:
                x=saved[a][k].double();y=saved[b][k].double();d=y-x
                comparison[a+'__'+b][k]={'exact':torch.equal(x,y),'maxabs':float(d.abs().max()),
                    'relative_L2':float(d.norm()/x.norm()),'cosine':float(x.dot(y)/(x.norm()*y.norm())),
                    'left_norm':float(x.norm()),'right_norm':float(y.norm())}
        save_once(OUT/'REPORT.json',{'comparisons':comparison,'task_CE':summaries,
            'optimizer_steps':0,'adapter_reset_exact':adapter_sha256(adapter)==digest,'source_GT_only':True})
        save_once(OUT/'COMPLETE.json',{'pins':{str(p):sha(p) for p in OUT.iterdir() if p.is_file()},'status':'completed'})
        status='completed'
    except BaseException:
        save_once(OUT/'FAILURE.json',{'failure':traceback.format_exc()});raise
    finally:
        seconds=time.monotonic()-start
        save_once(RECEIPT,{'status':status,'seconds':seconds,'prior_seconds':prior,'cumulative_seconds':prior+seconds,
            'scope':'includes imports after registration and all model loading/probes; no child wrapper','optimizer_steps':0,'cap':None})
        lease.close()

if __name__=='__main__':main()
