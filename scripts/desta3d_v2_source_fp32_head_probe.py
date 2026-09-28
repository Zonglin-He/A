"""No-update real FP32 task-head interface at two locked source states."""
import argparse,fcntl,gc,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.desta3d_v2_p0 import read,sha,adapter_sha256
from scripts.score_desta3d_v2_aux_recovery import save_once
from scripts.desta3d_tta_run_v1 import tensor_sha256,scan_nested_gpu_receipts
from vg_tta.desta3d_v2_head_probe import HeadEvidence
from vg_tta.desta3d_v2_fp32_task_head import task_head_fp32
from vg_tta.desta3d_v2_cast_probe import serialize_with_guard
BASE=ROOT/'artifacts/desta3d_v2';HEAD=BASE/'tta_v2/source_head_projection_v1';CAST=BASE/'tta_v2/source_cast_probe_v1'
OLD=BASE/'tta_v2/source_task_control_v2';OUT=BASE/'tta_v2/source_fp32_head_probe_v1'
RECEIPT=BASE/'receipts/source_fp32_head_probe_v1.json'


def put(name,x,cfg):
    p=OUT/name;assert not p.exists()
    blob=serialize_with_guard(x,used_bytes=sum(q.stat().st_size for q in OUT.rglob('*') if q.is_file()),
        free_bytes=shutil.disk_usage(ROOT).free,cap_bytes=cfg['storage_cap_bytes'],reserve_bytes=cfg['disk_reserve_bytes'])
    with p.with_suffix('.tmp').open('xb') as f:f.write(blob);f.flush();os.fsync(f.fileno())
    os.replace(p.with_suffix('.tmp'),p)


def register():
    assert not (OUT/'REGISTRATION.json').exists() and read(OUT/'CPU_PREFLIGHT.json')['status']=='passed'
    for p,h in read(HEAD/'COMPLETE.json')['pins'].items():assert sha(Path(p))==h,p
    cfg={'checkpoint':read(HEAD/'CONFIG.json')['checkpoint'],'source_key':'vidstg_source_query:28199','source':'5624461612',
        'states':['B1','saved_supervised3'],'branches':['event','spatial'],'seed':20260927,'scope':66816,
        'phase_seconds':600,'disk_reserve_bytes':8*2**30,'storage_cap_bytes':130_000_000,'cap':None,
        'full_vocab':152775,'reference_abs_tol':2e-4,'reference_rel_tol':0.,'optimizer_steps':0,
        'source_GT':'same training query diagnostic only','target_data':False,'native_predictions':0,
        'change':'only frozen final task-head projection FP32; original body/native BF16'}
    assert shutil.disk_usage(ROOT).free-cfg['storage_cap_bytes']>cfg['disk_reserve_bytes']
    save_once(OUT/'CONFIG.json',cfg)
    paths=[Path(__file__),ROOT/'vg_tta/desta3d_v2_fp32_task_head.py',ROOT/'tests/test_desta3d_v2_fp32_task_head.py',
        ROOT/'scripts/crosscheck_desta3d_v2_fp32_head_probe.py',ROOT/'protocols/desta3d_v2_source_fp32_head_probe_v1.md',
        OUT/'CPU_PREFLIGHT.json',OUT/'CONFIG.json',HEAD/'COMPLETE.json',HEAD/'CPU_REFERENCE.pt',HEAD/'ROOT_RAW_READBACK.json']
    pins=dict(read(HEAD/'LOCK.json')['pins']);pins.update({str(p):sha(p) for p in paths})
    save_once(OUT/'LOCK.json',{'pins':pins});save_once(OUT/'REGISTRATION.json',{'time':time.time(),'status':'registered_before_GPU',
        'source_queries':1,'source_parents':1,'optimizer_steps':0,'forward_backward_pairs':4,'free_disk_bytes':shutil.disk_usage(ROOT).free})
    print('REGISTERED',OUT,flush=True)


def run():
    cfg=read(OUT/'CONFIG.json');assert not (OUT/'STARTED.json').exists()
    for p,h in read(OUT/'LOCK.json')['pins'].items():assert sha(Path(p))==h,p
    lease=(ROOT/'artifacts/spatial_tta_research_v2/gpu.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    start=time.monotonic();status='failed';backwards=0
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
        from scripts.ptd_8b_teacher_feasibility_v1 import joint_loss
        from vg_tta.desta3d_v2 import Desta3DAdapterV2
        from vg_tta.desta3d_v2_ptd import capture_stock_fields,branch_injection
        from vg_tta.desta3d_v2_tta_pilot import configure
        from vg_tta.desta3d_v2_source_task_control import active,flat_gradient
        from vg_tta.desta3d_v2_source import split_source_loss_masks
        guard();pr=processor_load();model=model_load().eval().requires_grad_(False)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        initial=torch.load(cfg['checkpoint']['checkpoint'],map_location='cpu',weights_only=False)['adapter']
        final=torch.load(OLD/'episodes/00/supervised/FINAL_CALIBRATION.pt',map_location='cpu',weights_only=False)
        adapter=Desta3DAdapterV2(hidden_dim=128,architecture='dual3d',p1_enabled=False).cuda().eval();adapter.load_state_dict(initial)
        configure(adapter,'calibration');names,params=active(adapter);assert set(names)==set(final)
        assert adapter_sha256(adapter)==cfg['checkpoint']['adapter_sha256']
        head=model.lm_head;head_meta=read(HEAD/'HEAD_IDENTITY.json')
        assert type(head) is torch.nn.Linear and head.bias is None and head.weight.dtype==torch.bfloat16
        assert head.weight.data_ptr()==model.get_input_embeddings().weight.data_ptr()
        for p in head_meta['chunks']:assert tensor_sha256(head.weight[p['first']:p['first']+p['rows']])==p['sha256']
        original_forward=head.forward.__func__
        row=read(CAST/'INPUT.json');record=read(CAST/'SOURCE_RECORD.json')
        frames,ids=frames_for(row,'clean');prompt,pre=inputs_for(row,pr,frames)
        fields=capture_stock_fields(model,pr,prompt,row['input']['caption'],ids,row['input']['fps'])
        data,_,dpre=_training_inputs(pr,model,row,record);ident=read(HEAD/'INPUT_IDENTITY.json')
        assert dpre==pre==ident['original_identity']['preprocess'] and ids==ident['original_identity']['frame_ids']
        assert torch.equal(data['pixel_values_videos'],prompt['pixel_values_videos'])
        for k in ['visual_grid','query_tokens','frame_times']:assert tensor_sha256(fields[k])==ident['original_identity'][k+'_sha']
        support={k:tensor_sha256(v) for k,v in data.items() if isinstance(v,torch.Tensor) and k!='pixel_values_videos'}
        assert support==ident['support_sha'];save_once(OUT/'INPUT_IDENTITY.json',ident)
        masks=split_source_loss_masks(data,pr.tokenizer)
        ref=torch.load(HEAD/'CPU_REFERENCE.pt',map_location='cpu',weights_only=False);report={}
        for name,state in [('B1',initial),('saved_supervised3',{**initial,**final})]:
            adapter.load_state_dict(state);configure(adapter,'calibration');names,params=active(adapter)
            model.train();model.model.visual.eval();adapter.eval();report[name]={}
            for branch in cfg['branches']:
                guard();old=torch.load(HEAD/(name+'_'+branch+'.pt'),map_location='cpu',weights_only=False)
                assert adapter_sha256(adapter)==old['adapter_sha']
                d=dict(data);d['labels']=data['labels'].clone();d['labels'][~masks[branch].to(d['labels'].device)]=-100
                ev=HeadEvidence(d);hook=head.register_forward_hook(ev.hook)
                try:
                    with task_head_fp32(model):
                        with branch_injection(model,adapter,data,fields,branch) as cap:
                            ce,stats=joint_loss(model,d);hook.remove();raw=ev.finish()
                            raw.update(state=name,branch=branch,adapter_sha=adapter_sha256(adapter),support_sha=support,
                                precast_sha=tensor_sha256(cap['fields']['updated_tokens_'+branch]),postcast_sha=tensor_sha256(cap['updated_tokens']),
                                ce=float(ce.detach()),stats=stats,head_dtype=str(head.weight.dtype),logits_dtype=str(raw['logits'].dtype),
                                hidden_sha=tensor_sha256(raw['hidden']),logits_sha=tensor_sha256(raw['logits']))
                            put(name+'_'+branch+'_FORWARD.pt',raw,cfg)
                            assert raw['hidden_sha']==old['hidden_sha'] and raw['precast_sha']==old['precast_sha'] and raw['postcast_sha']==old['postcast_sha']
                            assert raw['logits'].dtype==torch.float32 and raw['hidden'].dtype==torch.bfloat16
                            for k in ['targets','positions','ntp']:assert torch.equal(raw['tokens'][k],old['tokens'][k])
                            diffs={}
                            for k,rk in [('target_logit','target_fp64'),('logsumexp','lse_fp64'),('cross_entropy','ce_fp64')]:
                                actual=torch.cat([x[k] for x in raw['tokens']['chunks']]).double();diff=float((actual-ref[name][branch]['token'][rk]).abs().max())
                                diffs[k]=diff;assert diff<=cfg['reference_abs_tol'],(name,branch,k,diff)
                            assert abs(raw['ce']-float(ref[name][branch]['token']['ce_fp64'].mean()))<=cfg['reference_abs_tol']
                            gradient=flat_gradient(ce,params);backwards+=1
                            assert gradient.norm()>0 and all(p.grad is None for p in adapter.parameters())
                            assert all(not p.requires_grad and p.grad is None for p in model.parameters())
                            assert all(p.requires_grad==(n in names) for n,p in adapter.named_parameters())
                            put(name+'_'+branch+'_GRADIENT.pt',{'gradient':gradient,'ordered_names':names,'adapter_sha':raw['adapter_sha'],
                                'gradient_sha':tensor_sha256(gradient),'norm':float(gradient.double().norm()),'scope':66816,'all_other_params_frozen':True},cfg)
                            report[name][branch]={'CE':raw['ce'],'CPU_ref_CE':float(ref[name][branch]['token']['ce_fp64'].mean()),
                                'max_token_errors':diffs,'gradient_norm':float(gradient.double().norm()),'hidden_exact':True,'injection_exact':True}
                            print('FP32_HEAD',name,branch,report[name][branch],flush=True)
                finally:hook.remove()
                assert head.forward.__func__ is original_forward and 'forward' not in head.__dict__
                del ce,cap,ev,raw,old,gradient;gc.collect();torch.cuda.empty_cache()
        adapter.load_state_dict(initial);adapter.set_train_stage('frozen');model.eval()
        assert adapter_sha256(adapter)==cfg['checkpoint']['adapter_sha256']
        save_once(OUT/'REPORT.json',{'states':report,'backwards':backwards,'optimizer_steps':0,'head_restored':True,
            'B1_reset_exact':True,'native_predictions':0,'target_data':False,'all_scope_checks':True,
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()})
        guard();save_once(OUT/'COMPLETE.json',{'status':'GPU_interface_passed','time':time.time(),
            'pins':{str(p):sha(p) for p in OUT.iterdir() if p.is_file() and not p.name.endswith('.log')}});status='completed'
    except BaseException:
        save_once(OUT/'FAILURE.json',{'time':time.time(),'error':traceback.format_exc(),'actual_backwards':backwards});raise
    finally:
        sec=time.monotonic()-start
        save_once(RECEIPT,{'status':status,'seconds':sec,'prior_seconds':prior,'cumulative_seconds':prior+sec,
            'cap':None,'optimizer_steps':0,'actual_backwards':backwards});lease.close()


def launch():
    assert not RECEIPT.exists();start=time.monotonic()
    child=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'run'],cwd=ROOT)
    wall=time.monotonic()-start;sec=read(RECEIPT)['seconds'] if RECEIPT.exists() else 0.
    save_once(BASE/'receipts/source_fp32_head_probe_wrapper_v1.json',{'status':'completed' if child.returncode==0 else 'failed',
        'seconds':max(0.,wall-sec),'worker_seconds':sec,'child_wall_seconds':wall,'cap':None,
        'scope':'nonoverlapping subprocess imports/check/finalization overhead'});raise SystemExit(child.returncode)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['register','run','launch'])
    {'register':register,'run':run,'launch':launch}[p.parse_args().action]()
