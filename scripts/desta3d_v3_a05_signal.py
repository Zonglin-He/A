"""Frozen R16 label-free direction screen. Reference projection and scoring stay CPU-only."""
import argparse,gc,json,os,sys,time,subprocess,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from vg_tta.desta3d_v3_oracle_io import OUT,read,write,sha,check_pins,local_dependencies,allocation,tensor_sha,total_prior
from scripts.desta3d_v3_a0_fast_screen import load,ADAPTER_SHA,B1,B1_SHA
A0=OUT/'a0_fast_screen_v1';A3=OUT/'a03_shared_basis_v1';GAP=OUT/'oracle_mixer_gap_v1';D=OUT/'a05_unlabeled_signal_v1'
SIGNALS=('U-Consistency','U-Entropy');BRANCHES=(('event',0,'time'),('spatial',1,'coordinate'))
PROTOCOL=ROOT/'protocols/desta3d_v3_a05_signal_v1.md'

def prepare():
    import torch,numpy as np
    from threadpoolctl import threadpool_limits
    from vg_tta.desta3d_v3_a05_signal import select_dev16
    assert not D.exists();start=time.monotonic();torch.set_num_threads(4)
    pre=ROOT/'artifacts/desta3d_v3/a05_cpu_preflight/PASSED.json';assert read(pre)['status']=='passed'
    rows=select_dev16(read(A0/'DEV64.json'));assert len(rows)==16 and len({r['source'] for r in rows})==16
    assert read(A3/'native/ROOT_FIELD_READBACK.json')['status']=='passed'
    q=load(A0/'BASIS.pt').double().numpy();b=np.load(A3/'TRAIN_BASIS.npz')['basis'][:,:16].copy()
    with threadpool_limits(limits=4,user_api='blas'):qc=(q@b).astype(np.float32)
    orth=float(np.max(abs(qc.astype(np.float64).T@qc.astype(np.float64)-np.eye(16))));assert orth<2e-6
    cfg=dict(seed=20260928,queries=16,parents=16,signals=list(SIGNALS),radius=.13545580427763146,
      phase_seconds=3600,minimum_free_bytes=8*2**30,maximum_new_bytes=4*2**30,cap=None,prior_seconds=total_prior(),
      optimizer_steps=0,native_predictions=0,gradient_space='R16_C',balance='minus_unit(unit(gT_C)+unit(gS_C))',
      consistency_KL='teacher||student',brightness=1.05,contrast=.95,classes_spatial=152775,
      median_gate=.10,positive_descent_gate=.65,chain_relative_tolerance=2e-5,loss_absolute_tolerance=1e-5,
      source_exposed=True,fresh_read=False,target_read=False,GT_reference_only=True,gram_error=orth)
    assert shutil.disk_usage(ROOT).free>=cfg['minimum_free_bytes']+cfg['maximum_new_bytes']
    write(D/'CONFIG.json',cfg);write(D/'INPUTS.json',rows);torch.save(torch.from_numpy(qc),D/'QC.pt')
    pre=ROOT/'artifacts/desta3d_v3/a05_cpu_preflight/PASSED.json';assert read(pre)['status']=='passed'
    oldseal=read(GAP/'PREDICTIONS_SEAL.json')['files'];a3seal=read(A3/'native/PREDICTIONS_SEAL.json')['files'];a0seal=read(A0/'CACHE_SEAL.json')['files'];reuse={};refs={}
    with threadpool_limits(limits=4,user_api='blas'):
        for i,r in enumerate(rows):
            gap=GAP/'episodes'/f"{r['gap_index']:04}";ep3=A3/'native/episodes'/f"{r['dev64_index']:04}"
            for p,owner,seal in [(gap/x,GAP,oldseal) for x in ['BASE_TRACE.pt','B1.pt','RAW_DIRECTIONS.pt','INPUT.json']]+[(ep3/x,A3/'native',a3seal) for x in ['Shared-R16_FACTORS.npy','Shared-R16.pt','Shared-R16_FIELD.json']]+[(A0/'cache/dev'/f"{r['dev64_index']:04}"/'INPUT.json',A0,a0seal)]:
                assert sha(p)==seal[str(p.relative_to(owner))];reuse[str(p)]=sha(p)
            raw=load(gap/'RAW_DIRECTIONS.pt');factor=np.load(ep3/'Shared-R16_FACTORS.npy').astype(np.float64);reference=factor/np.linalg.norm(factor)
            projected={k:torch.from_numpy(v.numpy().astype(np.float64)@qc.astype(np.float64)) for k,v in raw['gradients'].items()}
            path=D/'references'/f'{i:02}.pt';path.parent.mkdir(parents=True,exist_ok=True)
            torch.save(dict(oracle_direction=torch.from_numpy(reference),GT_gradients_C=projected,
              GT_available={k:'missing' not in v for k,v in raw['objectives'].items()},original_GT_gradient_sha=sha(gap/'RAW_DIRECTIONS.pt'),
              oracle_factors_sha=sha(ep3/'Shared-R16_FACTORS.npy'),stock_shape=raw['stock_shape'],stock_sha=raw['stock_sha'],stock_norm=raw['geometry']['stock_norm']),path)
            refs[str(path)]=sha(path)
    write(D/'REUSE.json',dict(files=reuse));write(D/'REFERENCE_SEAL.json',dict(files=refs,only_CPU_readout=True,new_GT_backwards=0))
    paths=local_dependencies([Path(__file__),PROTOCOL,ROOT/'vg_tta/desta3d_v3_a05_signal.py',ROOT/'tests/test_desta3d_v3_a05_signal.py',ROOT/'scripts/audit_desta3d_v3_a05_signal.py',
      B1,A0/'BASIS.pt',A3/'TRAIN_BASIS.npz',D/'CONFIG.json',D/'INPUTS.json',D/'QC.pt',D/'REUSE.json',D/'REFERENCE_SEAL.json',pre,
      ROOT/'vg_tta/desta3d_v3_actuation_full_vocab.py',ROOT/'vg_tta/desta3d_v2_output_anchor_memory_v7.py',ROOT/'scripts/desta3d_tta_run_v1.py'])
    pins={str(p):sha(p) for p in paths};write(D/'LOCK.json',dict(pins=pins));write(D/'REGISTRATION.json',dict(status='registered_before_GPU',time=time.time(),CPU_prepare_seconds=time.monotonic()-start,selection='metadata_SHA256',GT_to_signal=False))
    for stage,indices in [('probe001',[0]),('rest001',list(range(1,16)))]:
        run=OUT/('a05_'+stage);write(run/'CONFIG.json',{**cfg,'phase_seconds':900 if stage=='probe001' else 3600,'indices':indices})
        write(run/'LOCK.json',dict(pins={**pins,str(run/'CONFIG.json'):sha(run/'CONFIG.json')}))
    print('REGISTERED A05',cfg,flush=True)

def setup(seed):
    import torch
    from scripts.ptd_spatial_adapter_ab_v1 import processor_load,model_load
    from vg_tta.desta3d_v2 import Desta3DAdapterV2
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    pr=processor_load();model=model_load().eval().requires_grad_(False)
    adapter=Desta3DAdapterV2(hidden_dim=128,architecture='dual3d',p1_enabled=False).cuda().eval()
    adapter.load_state_dict(load(B1)['adapter']);adapter.set_train_stage('frozen')
    return pr,model,adapter

def unlabeled_inputs(pr,model,row):
    import torch
    from scripts.ptd_spatial_adapter_ab_v1 import inputs_for
    from scripts.desta3d_tta_run_v1 import make_mild_photometric_view
    from vg_tta.exact_frame_decode_audit_v2 import decode
    from vg_tta.desta3d_v2_ptd import capture_stock_fields
    frames,ids=decode(row['input']);assert ids==row['input']['frame_ids']
    outputs={}
    for view,pixels in [('observed',frames),('mild',make_mild_photometric_view(frames))]:
        prompt,pre=inputs_for(row,pr,pixels)
        fields=capture_stock_fields(model,pr,prompt,row['input']['caption'],ids,row['input']['fps'])
        outputs[view]=(prompt,pre,fields)
    a,b=outputs['observed'],outputs['mild']
    assert a[1]['grid']==b[1]['grid'] and torch.equal(a[0]['input_ids'],b[0]['input_ids'])
    assert torch.equal(a[2]['frame_times'],b[2]['frame_times']) and a[2]['visual_grid'].shape==b[2]['visual_grid'].shape
    return outputs

def worker(stage):
    run=OUT/('a05_'+stage)
    if stage=='rest001':assert read(D/'ROOT_FIRST_READBACK.json')['status']=='passed'
    with allocation(run) as (cfg,guard):
        import torch
        from vg_tta.desta3d_v3_a05_signal import signal_loss,balanced_direction
        from vg_tta.desta3d_v3_actuation_full_vocab import replay_branch
        from vg_tta.desta3d_v3_decomposition import shared_fields,norm
        from vg_tta.desta3d_v2_output_anchor_memory_v7 import release_free_host_arenas
        from scripts.desta3d_v2_p0 import adapter_sha256
        pr,model,adapter=setup(cfg['seed']);qc=load(D/'QC.pt').cuda();rows=read(D/'INPUTS.json');backwards=0
        for i in cfg['indices']:
            guard();r=rows[i];ep=D/'episodes'/f'{i:02}';assert not ep.exists(),'Partial retained; no replay';ep.mkdir(parents=True)
            old=GAP/'episodes'/f"{r['gap_index']:04}";trace=load(old/'BASE_TRACE.pt');views=unlabeled_inputs(pr,model,r)
            observed=views['observed'];support={k:tensor_sha(v) for k,v in observed[2].items() if isinstance(v,torch.Tensor)}
            inp=read(A0/'cache/dev'/f"{r['dev64_index']:04}"/'INPUT.json');assert observed[1]==inp['preprocess'] and support==inp['support']
            oldbase=load(old/'B1.pt');assert oldbase['support']==support and oldbase['adapter_sha']==ADAPTER_SHA
            assert oldbase['video_sha256']==r['input']['video_sha256'] and oldbase['frame_ids']==r['input']['frame_ids']
            identity=dict(observed_preprocess=observed[1],mild_preprocess=views['mild'][1],support=support,
              mild_support={k:tensor_sha(v) for k,v in views['mild'][2].items() if isinstance(v,torch.Tensor)},
              native_trace_sha=sha(old/'BASE_TRACE.pt'),physical_replay_exact=True,stock_sha=tensor_sha(observed[2]['visual_grid']),stock_norm=norm(observed[2]['visual_grid']),
              input_token_equal=True,frame_times_equal=True,source_GT_loaded_in_worker=False)
            write(ep/'INPUT.json',identity);raw={};metadata={}
            for signal in SIGNALS:
                prompt,pre,fields=views['mild' if signal=='U-Consistency' else 'observed'];stock=fields['visual_grid'].detach()
                grads={};info={}
                for branch,j,kind in BRANCHES:
                    guard();shape=(*stock.shape[:-1],16)
                    if j>=len(trace['branches']) or kind not in trace['branches'][j]['logits']:
                        grads[branch]=torch.zeros(shape);info[branch]=dict(missing='no_native_distribution_support');continue
                    expected=trace['branches'][j]['logits'][kind]
                    assert expected.shape[-1]==(stock.shape[1] if branch=='event' else 152775)
                    c=torch.zeros(shape,device=stock.device,dtype=stock.dtype,requires_grad=True);new=stock+c@qc.T
                    assert torch.equal(new,stock);release_free_host_arenas();torch.cuda.empty_cache()
                    with shared_fields(adapter,{branch:new},[branch]):
                        logits,cache=replay_branch(model,prompt,adapter,fields,trace,branch)
                        exact=torch.equal(logits.detach().cpu(),expected)
                        if signal=='U-Entropy':assert exact,'Observed replay not exact'
                        loss=signal_loss(logits,expected,signal)
                        assert torch.isfinite(loss)
                        # Save full forward evidence before any backward.
                        if signal=='U-Consistency':torch.save(logits.detach().cpu(),ep/(signal+'_'+branch+'_LOGITS.pt'))
                        record=dict(loss=float(loss.detach()),shape=list(logits.shape),classes=logits.shape[-1],observed_native_exact=exact,
                          student_logits_sha=tensor_sha(logits.detach()),teacher_logits_sha=tensor_sha(expected),cache=cache,zero_C_exact=True)
                        write(ep/(signal+'_'+branch+'_FORWARD.json'),record)
                        if i==0:
                            grad,gf=torch.autograd.grad(loss,(c,new));torch.save(gf.cpu(),ep/(signal+'_'+branch+'_GF.pt'))
                        else:grad,=torch.autograd.grad(loss,(c,))
                        backwards+=1
                    assert torch.isfinite(grad).all();grads[branch]=grad.cpu();info[branch]={k:v for k,v in record.items() if k!='cache'}
                    info[branch]['gradient_norm']=norm(grad)
                    del c,new,logits,loss,cache,grad
                    if i==0:del gf
                    gc.collect();torch.cuda.empty_cache()
                    assert all(not p.requires_grad and p.grad is None for mod in (model,adapter) for p in mod.parameters())
                direction=balanced_direction(grads['event'],grads['spatial'])
                raw[signal]=dict(gradients=grads,direction=direction);metadata[signal]=info
            torch.save(raw,ep/'RAW_SIGNALS.pt');write(ep/'SIGNALS.json',metadata)
            assert adapter_sha256(adapter)==ADAPTER_SHA and torch.equal(qc.cpu(),load(D/'QC.pt'))
            write(ep/'COMPLETE.json',dict(index=i,files={p.name:sha(p) for p in ep.iterdir() if p.is_file()},
              backwards=sum('missing' not in v for x in metadata.values() for v in x.values()),optimizer_steps=0,new_native=0,frozen_scope=True))
            print('A05_SIGNAL',i+1,16,'BACKWARDS',backwards,flush=True)
            del trace,views,observed,oldbase,raw,metadata,grads,info,direction,prompt,fields,stock,expected
            gc.collect();torch.cuda.empty_cache()
            assert sum(p.stat().st_size for p in (D/'episodes').rglob('*') if p.is_file())<cfg['maximum_new_bytes']
        write(run/'COMPLETE.json',dict(indices=cfg['indices'],backwards=backwards,optimizer_steps=0,new_native=0,peak_GPU_bytes=torch.cuda.max_memory_allocated()))
        if stage=='rest001':
            files={str(p.relative_to(D)):sha(p) for p in (D/'episodes').rglob('*') if p.is_file()}
            write(D/'SIGNALS_SEAL.json',dict(files=files,queries=16,signals=2,optimizer_steps=0))
            write(D/'GPU_COMPLETE.json',dict(queries=16,seal_sha=sha(D/'SIGNALS_SEAL.json')))

def launch(stage):
    run=OUT/('a05_'+stage);assert not (run/'STARTED.json').exists()
    if stage=='rest001':assert read(D/'ROOT_FIRST_READBACK.json')['status']=='passed'
    check_pins(read(run/'LOCK.json')['pins']);start=time.monotonic()
    cmd=[str(ROOT/'.venv-ptd-audit/bin/python'),'-B',str(Path(__file__)),'worker','--stage',stage]
    env={**os.environ,'CUBLAS_WORKSPACE_CONFIG':':4096:8','OPENBLAS_NUM_THREADS':'4','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4'}
    with (D/(stage+'.log')).open('xb') as log:
        p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        (D/'ACTIVE.json').write_text(json.dumps(dict(status='running',stage=stage,pid=p.pid,wrapper_pid=os.getpid(),time=time.time(),cmd=cmd),indent=2)+'\n');code=p.wait()
    wall=time.monotonic()-start;receipt=read(run/'RECEIPT.json') if (run/'RECEIPT.json').exists() else dict(seconds=0)
    overhead=max(0,wall-receipt['seconds']);prior=total_prior()
    write(OUT/('a05_'+stage+'_wrapper')/'RECEIPT.json',dict(status='completed' if code==0 else 'failed',seconds=overhead,worker_seconds=receipt['seconds'],child_wall_seconds=wall,prior_seconds=prior,cumulative_seconds=prior+overhead,cap=None))
    state=dict(status='completed_pending_audit' if code==0 else 'failed',stage=stage,code=code,time=time.time());write(run/'EXIT.json',state);(D/'ACTIVE.json').write_text(json.dumps(state)+'\n')
    assert code==0,'Original failure/partial retained; no silent replay'
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','launch','worker']);p.add_argument('--stage',choices=['probe001','rest001'],default='probe001');a=p.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='launch':launch(a.stage)
    else:worker(a.stage)
