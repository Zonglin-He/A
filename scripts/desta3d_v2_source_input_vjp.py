"""Actual postcast merger-input VJP at one fixed B1 source point, no update."""
import argparse,fcntl,gc,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.desta3d_v2_p0 import read,sha,adapter_sha256
from scripts.score_desta3d_v2_aux_recovery import save_once
from scripts.desta3d_tta_run_v1 import scan_nested_gpu_receipts,tensor_sha256
from vg_tta.desta3d_v2_cast_probe import TokenEvidence,serialize_with_guard
from vg_tta.desta3d_v2_input_vjp import postcast_leaf,transport_summary
BASE=ROOT/'artifacts/desta3d_v2'
OLD=BASE/'tta_v2/source_cast_probe_v1'
OUT=BASE/'tta_v2/source_input_vjp_v1'
RECEIPT=BASE/'receipts/source_input_vjp_v1.json'


def register():
    assert not (OUT/'REGISTRATION.json').exists()
    assert read(OUT/'CPU_PREFLIGHT.json')['status']=='passed'
    for p,h in read(OLD/'COMPLETE.json')['pins'].items():assert sha(Path(p))==h,p
    for p,h in read(OLD/'LOCK.json')['pins'].items():assert sha(Path(p))==h,p
    cfg={'checkpoint':read(OLD/'CONFIG.json')['checkpoint'],'source_key':'vidstg_source_query:28199',
         'source':'5624461612','states':['B1'],'branches':['event','spatial'],'optimizer_steps':0,
         'seed':20260927,'phase_seconds':600,'storage_cap_bytes':100_000_000,'disk_reserve_bytes':8*2**30,
         'cumulative_cap_seconds':None,'source_GT':'same first locked source training record, diagnostic task CE only',
         'target_data':False,'parameters':'all PTD and adapter frozen; only real BF16 postcast merger leaf derivative',
         'deterministic':True,'CUBLAS_WORKSPACE_CONFIG':':4096:8','selection':'fixed B1 and prior exact endpoint deltas'}
    assert shutil.disk_usage(ROOT).free-cfg['storage_cap_bytes']>cfg['disk_reserve_bytes']
    save_once(OUT/'CONFIG.json',cfg)
    paths=[Path(__file__),ROOT/'vg_tta/desta3d_v2_input_vjp.py',ROOT/'tests/test_desta3d_v2_input_vjp.py',
           ROOT/'protocols/desta3d_v2_source_input_vjp_v1.md',OUT/'CPU_PREFLIGHT.json',OUT/'CONFIG.json',
           OLD/'RAW_ENDPOINTS.pt',OLD/'ROOT_RAW_READBACK.json',OLD/'COMPLETE.json',OLD/'LOCK.json']
    pins=dict(read(OLD/'LOCK.json')['pins']);pins.update({str(p):sha(p) for p in paths})
    save_once(OUT/'LOCK.json',{'pins':pins})
    save_once(OUT/'REGISTRATION.json',{'time':time.time(),'status':'registered_before_GPU',
         'source_queries':1,'source_parents':1,'optimizer_steps':0,'free_disk_bytes':shutil.disk_usage(ROOT).free})
    print('REGISTERED',OUT,flush=True)


def bounded_save(name,payload,cfg):
    path=OUT/name;assert not path.exists()
    used=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
    raw=serialize_with_guard(payload,used_bytes=used,free_bytes=shutil.disk_usage(ROOT).free,
         cap_bytes=cfg['storage_cap_bytes'],reserve_bytes=cfg['disk_reserve_bytes'])
    with path.with_suffix('.tmp').open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    os.replace(path.with_suffix('.tmp'),path)


def run():
    cfg=read(OUT/'CONFIG.json');assert not (OUT/'STARTED.json').exists()
    for p,h in read(OUT/'LOCK.json')['pins'].items():assert sha(Path(p))==h,p
    lease=(ROOT/'artifacts/spatial_tta_research_v2/gpu.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    start=time.monotonic();status='failed';backward_calls=0
    prior=sum(scan_nested_gpu_receipts(ROOT/'artifacts'/n)[0] for n in ['desta3d_v1','desta3d_v2'])
    try:
        save_once(OUT/'STARTED.json',{'time':time.time(),'pid':os.getpid(),'prior_seconds':prior})
        def guard():
            assert time.monotonic()-start<cfg['phase_seconds'],'engineering phase limit'
            assert shutil.disk_usage(ROOT).free>=cfg['disk_reserve_bytes'],'disk floor'
        guard();torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True
        torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
        from scripts.ptd_spatial_adapter_ab_v1 import processor_load,model_load,frames_for,inputs_for
        from scripts.desta3d_source_fit_v1 import _training_inputs
        from scripts.ptd_8b_teacher_feasibility_v1 import joint_loss
        from vg_tta.desta3d_v2 import Desta3DAdapterV2
        from vg_tta.desta3d_v2_ptd import capture_stock_fields,branch_injection
        from vg_tta.desta3d_v2_source import split_source_loss_masks
        pr=processor_load();model=model_load().eval().requires_grad_(False)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        initial=torch.load(cfg['checkpoint']['checkpoint'],map_location='cpu',weights_only=False)['adapter']
        adapter=Desta3DAdapterV2(hidden_dim=128,architecture='dual3d',p1_enabled=False).cuda().eval()
        adapter.load_state_dict(initial);adapter.set_train_stage('frozen');assert not any(p.requires_grad for p in adapter.parameters())
        assert adapter_sha256(adapter)==cfg['checkpoint']['adapter_sha256']
        row=read(OLD/'INPUT.json');record=read(OLD/'SOURCE_RECORD.json');assert row['key']==record['key']==cfg['source_key']
        old=torch.load(OLD/'RAW_ENDPOINTS.pt',map_location='cpu',weights_only=False)
        frames,ids=frames_for(row,'clean');prompt,pre=inputs_for(row,pr,frames)
        fields=capture_stock_fields(model,pr,prompt,row['input']['caption'],ids,row['input']['fps'])
        data,_,dpre=_training_inputs(pr,model,row,record)
        ident=old['input_identity'];assert dpre==pre==ident['preprocess'] and ids==ident['frame_ids']
        for k in ['visual_grid','query_tokens','frame_times']:assert tensor_sha256(fields[k])==ident[k+'_sha']
        support={k:tensor_sha256(v) for k,v in data.items() if isinstance(v,torch.Tensor) and k!='pixel_values_videos'}
        assert support==old['task_support_sha']
        save_once(OUT/'INPUT_IDENTITY.json',{'original_identity':ident,'support_sha':support,'initial_adapter_sha':adapter_sha256(adapter)})
        masks=split_source_loss_masks(data,pr.tokenizer);results={}
        model.train();model.model.visual.eval();adapter.eval()
        for branch in cfg['branches']:
            guard();d=dict(data);d['labels']=data['labels'].clone();d['labels'][~masks[branch].to(d['labels'].device)]=-100
            ev=TokenEvidence(d);head_hook=model.lm_head.register_forward_hook(ev.hook)
            try:
                with branch_injection(model,adapter,data,fields,branch) as cap:
                    with postcast_leaf(model.model.visual.merger) as derivative:
                        ce,stats=joint_loss(model,d)
                        # Stop observation before checkpoint backward replays output projection chunks.
                        head_hook.remove();tokens=ev.finish();leaf=derivative['leaf']
                        info={'ce':float(ce.detach()),'tokens':int(masks[branch].sum()),
                              'relative_injection_norm':cap['relative_injection_norm'],
                              'cast_changed_elements':cap['changed_elements'],**stats}
                        pre_sha=tensor_sha256(cap['fields']['updated_tokens_'+branch])
                        post_sha=tensor_sha256(leaf)
                        forward={'branch':branch,'info':info,'tokens':tokens,'precast_sha':pre_sha,'postcast_sha':post_sha,
                                 'leaf_dtype':str(leaf.dtype),'shape':list(leaf.shape),'support_sha':support,
                                 'all_parameters_frozen':True,'adapter_sha':adapter_sha256(adapter)}
                        bounded_save(branch+'_FORWARD.pt',forward,cfg)
                        oldbranch=old['states']['B1']['branches'][branch]
                        assert info==oldbranch['info'],('forward mismatch',branch,info)
                        assert pre_sha==oldbranch['precast_sha'] and post_sha==oldbranch['postcast_sha']
                        assert leaf.dtype==torch.bfloat16 and torch.equal(leaf.detach().cpu(),cap['updated_tokens'])
                        for k in ['positions','targets','ntp']:assert torch.equal(tokens[k],oldbranch['tokens'][k])
                        assert len(tokens['chunks'])==len(oldbranch['tokens']['chunks'])
                        for a,b in zip(tokens['chunks'],oldbranch['tokens']['chunks']):
                            assert a['chunk_size']==b['chunk_size']
                            for k in ['target_logit','logsumexp','cross_entropy']:assert torch.equal(a[k],b[k])
                        gradient=torch.autograd.grad(ce,leaf)[0].detach().cpu();backward_calls+=1
                        assert gradient.shape==leaf.shape and torch.isfinite(gradient).all()
                        assert all(not p.requires_grad and p.grad is None for m in [model,adapter] for p in m.parameters())
                        summary=transport_summary(gradient,old['differences'][branch])
                        raw={'branch':branch,'gradient':gradient,'gradient_sha':tensor_sha256(gradient),
                             'shape':list(gradient.shape),'dtype':str(gradient.dtype),'source_key':row['key'],
                             'support_sha':support,'postcast_sha':post_sha,'adapter_sha':adapter_sha256(adapter),
                             'forward_exact':True,'parameters_frozen_and_no_grad':True,'summary':summary}
                        bounded_save(branch+'_VJP.pt',raw,cfg)
                        summary.update(CE=info['ce'],actual_backward_calls=1,forward_exact=True)
                        results[branch]=summary
                        print('VJP',branch,summary,flush=True)
            finally:
                head_hook.remove()
            del cap,derivative,leaf,ce,gradient,d;gc.collect();torch.cuda.empty_cache()
        model.eval();assert adapter_sha256(adapter)==cfg['checkpoint']['adapter_sha256']
        save_once(OUT/'REPORT.json',{'source_key':row['key'],'source':row['source'],'branches':results,
              'sum_dot_precast_delta':sum(x['dot_precast_delta'] for x in results.values()),
              'sum_dot_postcast_delta':sum(x['dot_postcast_delta'] for x in results.values()),
              'existing_parameter_linearization':old['linearization'],
              'optimizer_steps':0,'backward_calls':backward_calls,'all_parameters_frozen':True,'state_unchanged':True,
              'no_target_data':True,'source_GT_diagnostic':True,'native_task_effect_not_measured':True})
        guard();assert sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())<cfg['storage_cap_bytes']
        save_once(OUT/'COMPLETE.json',{'time':time.time(),'status':'completed',
              'pins':{str(p):sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='RUN001.log'}})
        status='completed'
    except BaseException:
        save_once(OUT/'FAILURE.json',{'time':time.time(),'failure':traceback.format_exc(),'backward_calls':backward_calls});raise
    finally:
        seconds=time.monotonic()-start
        save_once(RECEIPT,{'status':status,'seconds':seconds,'prior_seconds':prior,'cumulative_seconds':prior+seconds,
              'cap':None,'optimizer_steps':0,'backward_calls':backward_calls});lease.close()


def launch():
    assert not RECEIPT.exists();start=time.monotonic()
    child=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'run'],cwd=ROOT)
    wall=time.monotonic()-start;worker=read(RECEIPT)['seconds'] if RECEIPT.exists() else 0.
    save_once(BASE/'receipts/source_input_vjp_wrapper_v1.json',{'status':'completed' if child.returncode==0 else 'failed',
         'seconds':max(0.,wall-worker),'child_wall_seconds':wall,'worker_seconds':worker,
         'scope':'nonoverlapping subprocess imports/check/finalization overhead','cap':None})
    raise SystemExit(child.returncode)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['register','run','launch'])
    {'register':register,'run':run,'launch':launch}[p.parse_args().action]()
