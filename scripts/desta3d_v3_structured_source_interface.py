"""Zero-update real structured-source engineering validation."""
import argparse,fcntl,gc,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from scripts.desta3d_v2_p0 import read,sha,adapter_sha256
from scripts.score_desta3d_v2_aux_recovery import save_once
from scripts.desta3d_tta_run_v1 import tensor_sha256,scan_nested_gpu_receipts
from vg_tta.desta3d_v2_cast_probe import serialize_with_guard
from vg_tta.desta3d_v3_source import structured_support,source_branch_objective
BASE=ROOT/'artifacts/desta3d_v3';OUT=BASE/'structured_source_interface_v1'
OLD=ROOT/'artifacts/desta3d_v2/tta_v2/source_cast_probe_v1'
RECEIPT=BASE/'receipts/structured_source_interface_v1.json'
def put_pt(name,x):
    blob=serialize_with_guard(x,used_bytes=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()),
         free_bytes=shutil.disk_usage(ROOT).free,cap_bytes=80_000_000,reserve_bytes=8*2**30)
    with (OUT/name).open('xb') as f:f.write(blob)
def register():
    assert not (OUT/'REGISTRATION.json').exists()
    assert read(OUT/'CPU_PREFLIGHT.json')['status']=='passed'
    cfg=dict(checkpoint=read(OLD/'CONFIG.json')['checkpoint'],seed=20260928,states=['B1','fresh20260928'],
         source_key='vidstg_source_query:28199',branches=['event','spatial'],phase_seconds=600,optimizer_steps=0,
         native_predictions=0,cap=None,storage_cap_bytes=80_000_000,reserve_bytes=8*2**30,scope='full adapter B integration',
         structured_loss='MTP endpoint sum, coordinate frame/coordinate mean',regularizer_per_branch=.05,evidence_per_branch=.1,
         CE_reference_tolerance=2e-5,original_CE_tolerance=1e-6,target_data=False)
    save_once(OUT/'CONFIG.json',cfg)
    pins=dict(read(OLD/'LOCK.json')['pins'])
    paths=[Path(__file__),ROOT/'vg_tta/desta3d_v3_source.py',ROOT/'tests/test_desta3d_v3_source.py',
        ROOT/'protocols/desta3d_v3_structured_source_interface_v1.md',ROOT/'protocols/desta3d_v3_full_source_route_v1.md',
        OUT/'CONFIG.json',OUT/'CPU_PREFLIGHT.json',OLD/'RAW_ENDPOINTS.pt',OLD/'INPUT.json',OLD/'SOURCE_RECORD.json']
    pins.update({str(p):sha(p) for p in paths});save_once(OUT/'LOCK.json',dict(pins=pins))
    save_once(OUT/'REGISTRATION.json',dict(time=time.time(),status='registered_before_GPU',source_queries=1,optimizer_steps=0))
def run():
    assert not (OUT/'STARTED.json').exists();cfg=read(OUT/'CONFIG.json')
    for p,h in read(OUT/'LOCK.json')['pins'].items():assert sha(p)==h,p
    lease=(ROOT/'artifacts/spatial_tta_research_v2/gpu.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    start=time.monotonic();status='failed';backwards=0
    prior=sum(scan_nested_gpu_receipts(ROOT/'artifacts'/n)[0] for n in ['desta3d_v1','desta3d_v2','desta3d_v3'])
    try:
        save_once(OUT/'STARTED.json',dict(time=time.time(),pid=os.getpid(),prior_seconds=prior))
        def guard():
            assert time.monotonic()-start<cfg['phase_seconds'];assert shutil.disk_usage(ROOT).free>cfg['reserve_bytes']
        guard();torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True
        torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
        from scripts.ptd_spatial_adapter_ab_v1 import processor_load,model_load,frames_for,inputs_for
        from scripts.desta3d_source_fit_v1 import _training_inputs
        from vg_tta.desta3d_v2 import Desta3DAdapterV2
        from vg_tta.desta3d_v2_ptd import capture_stock_fields,branch_injection
        from vg_tta.desta3d_v2_training import source_evidence_losses
        pr=processor_load();model=model_load().eval().requires_grad_(False)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        row=read(OLD/'INPUT.json');record=read(OLD/'SOURCE_RECORD.json');assert row['key']==cfg['source_key']
        old=torch.load(OLD/'RAW_ENDPOINTS.pt',map_location='cpu',weights_only=False)
        frames,ids=frames_for(row,'clean');prompt,pre=inputs_for(row,pr,frames)
        fields=capture_stock_fields(model,pr,prompt,row['input']['caption'],ids,row['input']['fps'])
        data,_,dpre=_training_inputs(pr,model,row,record);ident=old['input_identity']
        assert dpre==pre==ident['preprocess'] and ids==ident['frame_ids']
        assert torch.equal(data['pixel_values_videos'],prompt['pixel_values_videos'])
        for k in ['visual_grid','query_tokens','frame_times']:assert tensor_sha256(fields[k])==ident[k+'_sha']
        support_hash={k:tensor_sha256(v) for k,v in data.items() if isinstance(v,torch.Tensor) and k!='pixel_values_videos'}
        assert support_hash==old['task_support_sha'];support=structured_support(data,pr.tokenizer)
        save_once(OUT/'INPUT_AND_SUPPORT.json',dict(identity=ident,support_hash=support_hash,
             T=support['T'],boxes=support['boxes'],positions=support['positions'],targets=support['targets'],conditioning=support['conditioning']))
        initial=torch.load(cfg['checkpoint']['checkpoint'],map_location='cpu',weights_only=False)['adapter']
        torch.manual_seed(cfg['seed']);adapter=Desta3DAdapterV2(hidden_dim=128,architecture='dual3d',p1_enabled=False).cuda().eval()
        fresh={k:v.detach().cpu().clone() for k,v in adapter.state_dict().items()};report={}
        for state_name,state in [('B1',initial),('fresh20260928',fresh)]:
            adapter.load_state_dict(state);adapter.set_train_stage('B integration');ah=adapter_sha256(adapter)
            if state_name=='B1':assert ah==cfg['checkpoint']['adapter_sha256']
            model.train();model.model.visual.eval();adapter.eval();report[state_name]={}
            for branch in cfg['branches']:
                guard();adapter.zero_grad(set_to_none=True)
                with branch_injection(model,adapter,data,fields,branch) as cap:
                    task,details=source_branch_objective(model,data,support,branch,cfg['regularizer_per_branch'])
                    aux=source_evidence_losses(cap['fields'],record)['event' if branch=='event' else 'ref']
                    loss=task+cfg['evidence_per_branch']*aux
                    assert torch.isfinite(loss);loss.backward();backwards+=1
                    raw={n:p.grad.detach().cpu().clone() for n,p in adapter.named_parameters() if p.grad is not None}
                    assert raw and all(torch.isfinite(g).all() for g in raw.values())
                    assert sum(float(g.double().square().sum()) for g in raw.values())>0
                    assert all(p.grad is None for p in model.parameters()) and adapter_sha256(adapter)==ah
                    info=dict(total=float(loss.detach()),structured=float(details['structured'].detach()),original=float(details['original'].detach()),
                        evidence=float(aux.detach()),original_targets=details['original_targets'],structured_targets=details['structured_targets'],
                        gradient_names=list(raw),gradient_elements=sum(g.numel() for g in raw.values()),gradient_norm=sum(float(g.double().square().sum()) for g in raw.values())**.5,
                        relative_injection_norm=cap['relative_injection_norm'],cast_changed_elements=cap['changed_elements'],adapter_sha=ah,
                        frozen_backbone_grad_none=True,adapter_unchanged=True)
                    put_pt(state_name+'_'+branch+'.pt',dict(info=info,gradients=raw,
                        selected_logits=details['selected_logits'].detach().cpu(),targets=details['targets'].cpu()))
                    if state_name=='B1':assert abs(info['original']-old['states']['B1']['branches'][branch]['info']['ce'])<=cfg['original_CE_tolerance']
                    report[state_name][branch]=info;print(state_name,branch,info['structured'],info['gradient_norm'],flush=True)
                del raw,loss,task,aux,details,cap;adapter.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
        adapter.load_state_dict(initial);adapter.set_train_stage('frozen');assert adapter_sha256(adapter)==cfg['checkpoint']['adapter_sha256']
        save_once(OUT/'REPORT.json',dict(states=report,actual_backwards=backwards,optimizer_steps=0,native_predictions=0,target_data=False,
             B1_reset_exact=True,peak_allocated_bytes=torch.cuda.max_memory_allocated()))
        guard();save_once(OUT/'COMPLETE.json',dict(status='GPU_interface_completed_pending_root',pins={str(p):sha(p) for p in OUT.iterdir() if p.is_file() and not p.name.endswith('.log')}));status='completed'
    except BaseException:
        save_once(OUT/'FAILURE.json',dict(error=traceback.format_exc(),actual_backwards=backwards));raise
    finally:
        seconds=time.monotonic()-start
        save_once(RECEIPT,dict(status=status,seconds=seconds,prior_seconds=prior,cumulative_seconds=prior+seconds,cap=None,
             actual_backwards=backwards,optimizer_steps=0));lease.close()
def launch():
    start=time.monotonic();r=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'run'],cwd=ROOT)
    wall=time.monotonic()-start;worker=read(RECEIPT)['seconds'] if RECEIPT.exists() else 0.
    save_once(BASE/'receipts/structured_source_interface_wrapper_v1.json',dict(status='completed' if r.returncode==0 else 'failed',
        seconds=max(0.,wall-worker),worker_seconds=worker,child_wall_seconds=wall,cap=None));raise SystemExit(r.returncode)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['register','run','launch']);globals()[p.parse_args().action]()
