"""No-update CE mode control on preselected first source, two saved states."""
import os,sys,time,traceback,fcntl,shutil
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.desta3d_v2_source_task_control import read,sha,save_once,save_pt,adapter_sha256,BASE
from scripts.desta3d_tta_run_v1 import scan_nested_gpu_receipts,tensor_sha256
OLD=BASE/'tta_v2/source_task_control_v2';OUT=BASE/'tta_v2/source_task_ce_mode_probe_v1'
RECEIPT=BASE/'receipts/source_task_ce_mode_probe_v1.json'

def main():
    assert not OUT.exists()
    for p,h in read(OLD/'LOCK.json')['pins'].items():assert sha(Path(p))==h
    save_once(OUT/'REGISTRATION.json',{'time':time.time(),'status':'registered_before_GPU','optimizer_steps':0,
        'source_key':read(OLD/'INPUTS.json')[0]['key'],'states':['B1','saved_supervised3'],
        'question':'does grad-enabled versus no-grad CE measurement change values at same state?',
        'modes':'task_signals gradients True/False, two repeats each state; no training updates',
        'source_GT':'same locked first training record, diagnostic only','target_data':False,
        'deterministic':True,'phase_seconds':300,'cap':None,
        'pins':{str(Path(__file__)):sha(Path(__file__)),str(OLD/'LOCK.json'):sha(OLD/'LOCK.json'),
                str(OLD/'episodes/00/supervised/FINAL_CALIBRATION.pt'):sha(OLD/'episodes/00/supervised/FINAL_CALIBRATION.pt')}})
    lease=(ROOT/'artifacts/spatial_tta_research_v2/gpu.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    start=time.monotonic();prior=sum(scan_nested_gpu_receipts(ROOT/'artifacts'/n)[0] for n in ['desta3d_v1','desta3d_v2']);status='failed'
    try:
        assert shutil.disk_usage(ROOT).free>8*2**30
        torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True
        torch.manual_seed(20260927);torch.cuda.manual_seed_all(20260927)
        from scripts.ptd_spatial_adapter_ab_v1 import processor_load,model_load,frames_for,inputs_for
        from scripts.desta3d_source_fit_v1 import _training_inputs
        from vg_tta.desta3d_v2 import Desta3DAdapterV2
        from vg_tta.desta3d_v2_ptd import capture_stock_fields
        from vg_tta.desta3d_v2_tta_pilot import configure
        from vg_tta.desta3d_v2_source_task_control import task_signals
        pr=processor_load();model=model_load().eval().requires_grad_(False)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        cfg=read(OLD/'CONFIG.json');row=read(OLD/'INPUTS.json')[0];record=read(OLD/'SOURCE_RECORDS.json')[0]
        initial=torch.load(cfg['checkpoint']['checkpoint'],map_location='cpu',weights_only=False)['adapter']
        final=torch.load(OLD/'episodes/00/supervised/FINAL_CALIBRATION.pt',map_location='cpu',weights_only=False)
        adapter=Desta3DAdapterV2(hidden_dim=128,architecture='dual3d',p1_enabled=False).cuda().eval()
        frames,ids=frames_for(row,'clean');prompt,pre=inputs_for(row,pr,frames)
        fields=capture_stock_fields(model,pr,prompt,row['input']['caption'],ids,row['input']['fps'])
        data,_,dpre=_training_inputs(pr,model,row,record);assert dpre==pre
        ident=read(OLD/'episodes/00/INPUT_IDENTITY.json')
        for k in ['visual_grid','query_tokens','frame_times']:assert tensor_sha256(fields[k])==ident[k+'_sha']
        result={}
        for name,state in [('B1',initial),('saved_supervised3',{**initial,**final})]:
            adapter.load_state_dict(state);configure(adapter,'calibration');h=adapter_sha256(adapter);cases=[]
            for repeat in range(2):
                for mode in [True,False]:
                    assert time.monotonic()-start<300
                    raw,info=task_signals(model,pr,adapter,fields,data,gradients=mode)
                    save_pt(OUT/f'{name}_r{repeat}_grad{int(mode)}.pt',{'raw':raw,'info':info,'adapter_sha':h,'mode':mode})
                    assert adapter_sha256(adapter)==h
                    cases.append({'repeat':repeat,'mode':mode,'info':info})
                    print(name,repeat,mode,info['total'],flush=True)
            result[name]={'cases':cases,'all_forward_infos_exact':all(c['info']==cases[0]['info'] for c in cases)}
        adapter.load_state_dict(initial);adapter.set_train_stage('frozen')
        save_once(OUT/'REPORT.json',{'states':result,'reset_sha':adapter_sha256(adapter),'optimizer_steps':0,
            'source_GT_only':True,'does_not_identify_quantization_or_optimization_root_cause':True})
        save_once(OUT/'COMPLETE.json',{'status':'completed','pins':{str(p):sha(p) for p in OUT.iterdir() if p.is_file()}})
        status='completed'
    except BaseException:
        save_once(OUT/'FAILURE.json',{'failure':traceback.format_exc()});raise
    finally:
        dt=time.monotonic()-start;save_once(RECEIPT,{'status':status,'seconds':dt,'prior_seconds':prior,
            'cumulative_seconds':prior+dt,'optimizer_steps':0,'cap':None});lease.close()

if __name__=='__main__':main()
