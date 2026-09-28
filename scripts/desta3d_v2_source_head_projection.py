"""Fixed source original-head readback, then separate full-vocabulary CPU audit."""
import argparse,fcntl,gc,os,shutil,subprocess,sys,time,traceback,resource
from pathlib import Path
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.desta3d_v2_p0 import read,sha,adapter_sha256
from scripts.score_desta3d_v2_aux_recovery import save_once
from scripts.desta3d_tta_run_v1 import scan_nested_gpu_receipts,tensor_sha256
from vg_tta.desta3d_v2_cast_probe import serialize_with_guard
from vg_tta.desta3d_v2_head_probe import HeadEvidence,full_reference
BASE=ROOT/'artifacts/desta3d_v2';OLD=BASE/'tta_v2/source_cast_probe_v1'
OUT=BASE/'tta_v2/source_head_projection_v1';CONTROL=BASE/'tta_v2/source_task_control_v2'
RECEIPT=BASE/'receipts/source_head_projection_v1.json'
WEIGHTS=ROOT/'checkpoints/ParallelTubeDecoding-Qwen3-VL-4B/model.safetensors'
WEIGHT_KEY='model.language_model.embed_tokens.weight'


def bounded_save(name,payload,cfg):
    path=OUT/name;assert not path.exists()
    used=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
    blob=serialize_with_guard(payload,used_bytes=used,free_bytes=shutil.disk_usage(ROOT).free,
                             cap_bytes=cfg['storage_cap_bytes'],reserve_bytes=cfg['disk_reserve_bytes'])
    with path.with_suffix('.tmp').open('xb') as f:f.write(blob);f.flush();os.fsync(f.fileno())
    os.replace(path.with_suffix('.tmp'),path)


def register():
    assert not (OUT/'REGISTRATION.json').exists() and read(OUT/'CPU_PREFLIGHT.json')['status']=='passed'
    for p,h in read(OLD/'LOCK.json')['pins'].items():assert sha(Path(p))==h,p
    cfg={'checkpoint':read(OLD/'CONFIG.json')['checkpoint'],'source_key':'vidstg_source_query:28199','source':'5624461612',
         'states':['B1','saved_supervised3'],'branches':['event','spatial'],'optimizer_steps':0,'backward_calls':0,
         'seed':20260927,'phase_seconds':600,'storage_cap_bytes':100_000_000,'disk_reserve_bytes':8*2**30,
         'cumulative_cap_seconds':None,'head_shape':[152775,2560],'head_dtype':'torch.bfloat16','bias':False,
         'checkpoint_weight_key':WEIGHT_KEY,'weight_chunk_rows':8192,'CPU_reference_dtype':'torch.float64',
         'CPU_phase_seconds':1800,'CPU_max_rss_bytes':4*2**30,'host_reserve_bytes':6*2**30,
         'full_vocabulary':True,'source_GT':'same first source record, task diagnostics only','target_data':False,
         'no_model_precision_change':True,'native_predictions':0,'deterministic':True}
    assert shutil.disk_usage(ROOT).free-cfg['storage_cap_bytes']>cfg['disk_reserve_bytes']
    save_once(OUT/'CONFIG.json',cfg)
    paths=[Path(__file__),ROOT/'vg_tta/desta3d_v2_head_probe.py',ROOT/'tests/test_desta3d_v2_head_probe.py',
        ROOT/'scripts/crosscheck_desta3d_v2_source_head_projection.py',ROOT/'protocols/desta3d_v2_source_head_projection_v1.md',
        OUT/'CPU_PREFLIGHT.json',OUT/'CONFIG.json',OLD/'RAW_ENDPOINTS.pt',OLD/'COMPLETE.json',
        BASE/'tta_v2/source_input_vjp_v1/ROOT_COMPLETION_SUMMARY.json']
    pins=dict(read(OLD/'LOCK.json')['pins']);pins.update({str(p):sha(p) for p in paths})
    save_once(OUT/'LOCK.json',{'pins':pins})
    save_once(OUT/'REGISTRATION.json',{'time':time.time(),'status':'registered_before_GPU','source_queries':1,
        'source_parents':1,'observed_token_rows':162,'optimizer_steps':0,'free_disk_bytes':shutil.disk_usage(ROOT).free})
    print('REGISTERED',OUT,flush=True)


def run():
    cfg=read(OUT/'CONFIG.json');assert not (OUT/'STARTED.json').exists()
    for p,h in read(OUT/'LOCK.json')['pins'].items():assert sha(Path(p))==h,p
    lease=(ROOT/'artifacts/spatial_tta_research_v2/gpu.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    start=time.monotonic();status='failed';forwards=0
    prior=sum(scan_nested_gpu_receipts(ROOT/'artifacts'/n)[0] for n in ['desta3d_v1','desta3d_v2'])
    try:
        save_once(OUT/'STARTED.json',{'time':time.time(),'pid':os.getpid(),'prior_seconds':prior})
        def guard():
            assert time.monotonic()-start<cfg['phase_seconds']
            assert shutil.disk_usage(ROOT).free>=cfg['disk_reserve_bytes']
        guard();torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True
        torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
        from safetensors import safe_open
        from scripts.ptd_spatial_adapter_ab_v1 import processor_load,model_load,frames_for,inputs_for
        from scripts.desta3d_source_fit_v1 import _training_inputs
        from scripts.ptd_8b_teacher_feasibility_v1 import joint_loss
        from vg_tta.desta3d_v2 import Desta3DAdapterV2
        from vg_tta.desta3d_v2_ptd import capture_stock_fields,branch_injection
        from vg_tta.desta3d_v2_source import split_source_loss_masks
        pr=processor_load();model=model_load().eval().requires_grad_(False)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        initial=torch.load(cfg['checkpoint']['checkpoint'],map_location='cpu',weights_only=False)['adapter']
        final=torch.load(CONTROL/'episodes/00/supervised/FINAL_CALIBRATION.pt',map_location='cpu',weights_only=False)
        adapter=Desta3DAdapterV2(hidden_dim=128,architecture='dual3d',p1_enabled=False).cuda().eval()
        adapter.load_state_dict(initial);adapter.set_train_stage('frozen')
        assert adapter_sha256(adapter)==cfg['checkpoint']['adapter_sha256']
        head=model.lm_head
        assert type(head) is torch.nn.Linear and head.bias is None and list(head.weight.shape)==cfg['head_shape']
        assert head.weight.dtype==torch.bfloat16 and head.weight.data_ptr()==model.get_input_embeddings().weight.data_ptr()
        weight_pins=[]
        with safe_open(str(WEIGHTS),framework='pt',device='cpu') as f:
            s=f.get_slice(WEIGHT_KEY);assert s.get_shape()==cfg['head_shape']
            for i in range(0,len(head.weight),cfg['weight_chunk_rows']):
                guard();w=head.weight[i:i+cfg['weight_chunk_rows']].detach().cpu()
                disk=s[i:i+len(w)];assert disk.dtype==torch.float16
                assert torch.equal(w,disk.bfloat16()),('weight mismatch',i)
                weight_pins.append({'first':i,'rows':len(w),'sha256':tensor_sha256(w)})
        save_once(OUT/'HEAD_IDENTITY.json',{'class':type(head).__module__+'.'+type(head).__name__,'bias':False,
            'shape':list(head.weight.shape),'dtype':str(head.weight.dtype),'tied_embeddings':True,
            'weight_path':str(WEIGHTS),'weight_key':WEIGHT_KEY,'disk_dtype':'torch.float16','chunks':weight_pins})
        row=read(OLD/'INPUT.json');record=read(OLD/'SOURCE_RECORD.json');assert row['key']==record['key']==cfg['source_key']
        old=torch.load(OLD/'RAW_ENDPOINTS.pt',map_location='cpu',weights_only=False)
        frames,ids=frames_for(row,'clean');prompt,pre=inputs_for(row,pr,frames)
        fields=capture_stock_fields(model,pr,prompt,row['input']['caption'],ids,row['input']['fps'])
        data,_,dpre=_training_inputs(pr,model,row,record);ident=old['input_identity']
        assert dpre==pre==ident['preprocess'] and ids==ident['frame_ids']
        assert torch.equal(data['pixel_values_videos'],prompt['pixel_values_videos'])
        for k in ['visual_grid','query_tokens','frame_times']:assert tensor_sha256(fields[k])==ident[k+'_sha']
        support={k:tensor_sha256(v) for k,v in data.items() if isinstance(v,torch.Tensor) and k!='pixel_values_videos'}
        assert support==old['task_support_sha']
        save_once(OUT/'INPUT_IDENTITY.json',{'original_identity':ident,'support_sha':support})
        masks=split_source_loss_masks(data,pr.tokenizer);reports={}
        for state_name,state in [('B1',initial),('saved_supervised3',{**initial,**final})]:
            adapter.load_state_dict(state);adapter.set_train_stage('frozen');ah=adapter_sha256(adapter)
            assert ah==old['states'][state_name]['adapter_sha']
            model.train();model.model.visual.eval();adapter.eval();reports[state_name]={}
            for branch in cfg['branches']:
                guard();d=dict(data);d['labels']=data['labels'].clone();d['labels'][~masks[branch].to(d['labels'].device)]=-100
                ev=HeadEvidence(d);hook=head.register_forward_hook(ev.hook)
                try:
                    with torch.no_grad(),branch_injection(model,adapter,data,fields,branch) as cap:
                        ce,stats=joint_loss(model,d);forwards+=1
                        info={'ce':float(ce),'tokens':int(masks[branch].sum()),'relative_injection_norm':cap['relative_injection_norm'],
                              'cast_changed_elements':cap['changed_elements'],**stats}
                        before_sha=tensor_sha256(cap['fields']['updated_tokens_'+branch]);after_sha=tensor_sha256(cap['updated_tokens'])
                    raw=ev.finish();raw.update(state=state_name,branch=branch,info=info,adapter_sha=ah,support_sha=support,
                        precast_sha=before_sha,postcast_sha=after_sha,hidden_sha=tensor_sha256(raw['hidden']),logits_sha=tensor_sha256(raw['logits']))
                    bounded_save(state_name+'_'+branch+'.pt',raw,cfg)
                    prev=old['states'][state_name]['branches'][branch];assert info==prev['info']
                    assert before_sha==prev['precast_sha'] and after_sha==prev['postcast_sha']
                    a=raw['tokens'];b=prev['tokens']
                    for k in ['positions','targets','ntp']:assert torch.equal(a[k],b[k])
                    assert len(a['chunks'])==len(b['chunks'])
                    for aa,bb in zip(a['chunks'],b['chunks']):
                        assert aa['chunk_size']==bb['chunk_size']
                        for k in ['target_logit','logsumexp','cross_entropy']:assert torch.equal(aa[k],bb[k])
                    assert list(raw['hidden'].shape)==[info['tokens'],2560] and raw['hidden'].dtype==torch.bfloat16
                    assert list(raw['logits'].shape)==[info['tokens'],152775] and raw['logits'].dtype==torch.bfloat16
                    reports[state_name][branch]={'info':info,'hidden_sha':raw['hidden_sha'],'logits_sha':raw['logits_sha'],
                        'file':state_name+'_'+branch+'.pt','original_forward_exact':True}
                    print('HEAD_CAPTURE',state_name,branch,info,flush=True)
                finally:hook.remove()
                del raw,cap,ev,ce,d;gc.collect()
            assert adapter_sha256(adapter)==ah
        adapter.load_state_dict(initial);adapter.set_train_stage('frozen');model.eval()
        assert adapter_sha256(adapter)==cfg['checkpoint']['adapter_sha256']
        assert all(not p.requires_grad and p.grad is None for m in [model,adapter] for p in m.parameters())
        save_once(OUT/'GPU_REPORT.json',{'states':reports,'forward_calls':forwards,'original_81_tokens_per_state_exact':True,
            'all_params_frozen_no_grad':True,'B1_reset_exact':True,'optimizer_steps':0,'backward_calls':0,
            'new_native_predictions':0,'target_data':False,'source_GT_diagnostic_only':True})
        guard();save_once(OUT/'GPU_COMPLETE.json',{'time':time.time(),'status':'completed',
            'pins':{str(p):sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='RUN001.log'}});status='completed'
    except BaseException:
        save_once(OUT/'GPU_FAILURE.json',{'time':time.time(),'failure':traceback.format_exc(),'forwards':forwards});raise
    finally:
        sec=time.monotonic()-start
        save_once(RECEIPT,{'status':status,'seconds':sec,'prior_seconds':prior,'cumulative_seconds':prior+sec,
            'cap':None,'optimizer_steps':0,'forward_calls':forwards});lease.close()


def cpu_reference():
    assert not (OUT/'CPU_REFERENCE.pt').exists();cfg=read(OUT/'CONFIG.json');start=time.monotonic()
    try:
        for file,h in read(OUT/'GPU_COMPLETE.json')['pins'].items():assert sha(Path(file))==h,file
        torch.set_num_threads(4)
        from safetensors import safe_open
        def guard():
            assert time.monotonic()-start<cfg['CPU_phase_seconds']
            assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<cfg['CPU_max_rss_bytes']
            mem={l.split(':')[0]:int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines()}
            assert mem['MemAvailable']>cfg['host_reserve_bytes']
        results={};head=read(OUT/'HEAD_IDENTITY.json')
        for state in cfg['states']:
            results[state]={}
            for branch in cfg['branches']:
                raw=torch.load(OUT/(state+'_'+branch+'.pt'),map_location='cpu',weights_only=False)
                def chunks():
                    with safe_open(str(WEIGHTS),framework='pt',device='cpu') as f:
                        s=f.get_slice(WEIGHT_KEY)
                        for entry in head['chunks']:
                            guard();i=entry['first'];w=s[i:i+entry['rows']].bfloat16()
                            assert tensor_sha256(w)==entry['sha256']
                            yield i,w
                r=full_reference(raw['hidden'],raw['logits'],raw['tokens']['targets'],chunks())
                r['positions']=raw['tokens']['positions'];r['targets']=raw['tokens']['targets'];r['ntp']=raw['tokens']['ntp']
                r['actual_original_CE']=raw['info']['ce'];results[state][branch]=r
                print('CPU_REFERENCE',state,branch,{k:float(v.mean()) for k,v in r['token'].items() if k.startswith('ce_')},r['full_vocab'],flush=True)
        bounded_save('CPU_REFERENCE.pt',results,cfg)
        summary={'states':{},'delta':{}}
        for state in cfg['states']:
            summary['states'][state]={b:{k:float(v.mean()) for k,v in r['token'].items() if k.startswith('ce_')} for b,r in results[state].items()}
        for b in cfg['branches']:
            summary['delta'][b]={k:summary['states']['saved_supervised3'][b][k]-summary['states']['B1'][b][k] for k in summary['states']['B1'][b]}
            summary['delta'][b]['actual_original_CE']=results['saved_supervised3'][b]['actual_original_CE']-results['B1'][b]['actual_original_CE']
        save_once(OUT/'CPU_REFERENCE.json',{'status':'completed','summary':summary,'seconds':time.monotonic()-start,
            'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'full_vocabulary':152775,'states':2,
            'offline_only':True,'dtype':'float64 dot on exact observed BF16 hidden and BF16 weights'})
    except BaseException:
        save_once(OUT/'CPU_FAILURE.json',{'time':time.time(),'failure':traceback.format_exc(),'seconds':time.monotonic()-start});raise


def launch():
    assert not RECEIPT.exists();start=time.monotonic()
    child=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'run'],cwd=ROOT)
    wall=time.monotonic()-start;worker=read(RECEIPT)['seconds'] if RECEIPT.exists() else 0.
    save_once(BASE/'receipts/source_head_projection_wrapper_v1.json',{'status':'completed' if child.returncode==0 else 'failed',
        'seconds':max(0.,wall-worker),'child_wall_seconds':wall,'worker_seconds':worker,'cap':None,
        'scope':'nonoverlapping subprocess imports/check/finalization overhead'})
    raise SystemExit(child.returncode)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['register','run','launch','cpu'])
    {'register':register,'run':run,'launch':launch,'cpu':cpu_reference}[p.parse_args().action]()
