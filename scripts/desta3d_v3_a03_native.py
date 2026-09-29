"""Conditionally authorized Shared-R16/R32 oracle, frozen native two-pass PTD."""
import argparse,gc,os,sys,time,subprocess,shutil,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.desta3d_v3_a0_fast_screen import D as A0,GAP,load,setup,example,ADAPTER_SHA,labels_for
from scripts.desta3d_v3_joint_learnability import B1,B1_SHA,equal
from vg_tta.desta3d_v3_oracle_io import OUT,read,write,sha,check_pins,local_dependencies,allocation,tensor_sha,total_prior
D=OUT/'a03_shared_basis_v1';N=D/'native';RUN=OUT/'a03_native001'
PROTOCOL=ROOT/'protocols/desta3d_v3_a03_shared_channel_basis_v1.md'
ARMS=('B1','oracle','Shared-R16','Shared-R32')


def prepare():
    assert read(D/'ROOT_READBACK.json')['gate_pass'] and read(D/'ROOT_READBACK.json')['status']=='passed'
    assert not N.exists() and not RUN.exists()
    cfg=dict(seed=20260928,phase_seconds=3600,minimum_free_bytes=8*2**30,maximum_new_bytes=4*2**30,
        cumulative_cap=None,radius=read(A0/'CONFIG.json')['radius'],queries=64,parents=16,
        new_predictions=128,reused_predictions=128,optimizer_steps=0,backwards=0,native_baseline_replays=0,
        ranks=[16,32],scope='exposed source Dev64 GT analytic projected oracle; no predictor or target',fresh_read=False)
    assert shutil.disk_usage(ROOT).free>cfg['minimum_free_bytes']+cfg['maximum_new_bytes']
    rows=read(A0/'DEV64.json');assert len(rows)==64 and len({r['source'] for r in rows})==16
    assert sha(B1)==B1_SHA
    write(N/'CONFIG.json',cfg);write(N/'INPUTS.json',rows)
    oldseal=read(GAP/'PREDICTIONS_SEAL.json')['files'];cache_seal=read(A0/'CACHE_SEAL.json')['files'];reuse={}
    for i,r in enumerate(rows):
        ep=GAP/'episodes'/f"{r['gap_index']:04}"
        for name in ('B1.pt','oracle.pt','GEOMETRY.json','INPUT.json'):
            p=ep/name;assert sha(p)==oldseal[str(p.relative_to(GAP))];reuse[str(p)]=sha(p)
        for name in ('CACHE.pt','INPUT.json'):
            p=A0/'cache/dev'/f'{i:04}'/name;assert sha(p)==cache_seal[str(p.relative_to(A0))];reuse[str(p)]=sha(p)
    write(N/'REUSE.json',dict(files=reuse,old_seal_sha=sha(GAP/'PREDICTIONS_SEAL.json'),predictions=128))
    paths=local_dependencies([Path(__file__),ROOT/'scripts/score_desta3d_v3_a03_native.py',
        ROOT/'scripts/audit_desta3d_v3_a03_native.py',PROTOCOL,D/'ROOT_READBACK.json',D/'TRAIN_BASIS.npz',
        D/'TRAIN_BASIS_SEAL.json',A0/'BASIS.pt',B1,A0/'ROOT_CACHE_READBACK.json',N/'CONFIG.json',N/'INPUTS.json',N/'REUSE.json'])
    pins={str(p):sha(p) for p in paths};write(N/'LOCK.json',dict(pins=pins))
    write(N/'REGISTRATION.json',dict(time=time.time(),status='registered_before_GPU',gate=.75,
        measured_gate=read(D/'PUBLIC_REPORT.json')['dev_rank32_median_energy'],source_GT_oracle=True,
        physical_replay_required=True,old_native_reused_after_seal_and_input_identity=True))
    write(RUN/'CONFIG.json',cfg);write(RUN/'LOCK.json',dict(pins=pins))
    write(RUN/'REGISTRATION.json',dict(time=time.time(),stage='A03 projected oracle Dev64'))
    print('A03_NATIVE_REGISTERED',flush=True)


def worker():
    with allocation(RUN) as (cfg,guard):
        import numpy as np,torch
        from vg_tta.desta3d_v3_decomposition import shared_fields,norm
        from vg_tta.desta3d_v2_shared_reference_cached import decode_shared_reference_two_pass
        from vg_tta.desta3d_v2_prediction_contract import validate_prediction
        from scripts.desta3d_v2_source_fit import prediction_record
        from scripts.desta3d_v2_reference_audit_cached_v3 import details
        from scripts.desta3d_v2_p0 import adapter_sha256
        check_pins(read(N/'REUSE.json')['files'])
        pr,model,adapter,unused=setup(cfg['seed']);unused.eval().requires_grad_(False)
        q=load(A0/'BASIS.pt').double().numpy();b=np.load(D/'TRAIN_BASIS.npz')['basis']
        assert q.shape==(2560,256) and np.max(np.abs(q.T@q-np.eye(256)))<1e-7
        rows=read(N/'INPUTS.json');labels=labels_for(rows,'dev')
        for i,row in enumerate(rows):
            guard();ep=N/'episodes'/f'{i:04}';assert not ep.exists(),'Partial episode: retain; no silent replay';ep.mkdir(parents=True)
            old=GAP/'episodes'/f"{row['gap_index']:04}";data=load(A0/'cache/dev'/f'{i:04}'/'CACHE.pt');inp=read(A0/'cache/dev'/f'{i:04}'/'INPUT.json')
            prompt,pre,fields,args=example(pr,model,adapter,row,labels[row['key']]);stock=fields['visual_grid'].detach()
            support={k:tensor_sha(v) for k,v in fields.items() if isinstance(v,torch.Tensor)}
            assert pre==inp['preprocess'] and support==inp['support'] and abs(norm(stock)-data['stock_norm'])<1e-8
            for name,x in zip(('z','qT','qS','evidence8'),args[:-1]):assert torch.equal(x.cpu(),data[name]),name
            for name in ('B1','oracle'):
                prior=load(old/(name+'.pt'))
                assert prior['key']==row['key'] and prior['source']==row['source'] and prior['adapter_sha']==ADAPTER_SHA
                assert prior['preprocess']==pre and prior['support']==support
                assert prior['injection']['common_F_sha']==tensor_sha(stock)
                assert prior['frame_ids']==row['input']['frame_ids'] and prior['video_sha256']==row['input']['video_sha256']
                os.link(old/(name+'.pt'),ep/(name+'.pt'))
            write(ep/'INPUT.json',dict(**inp,physical_replay_exact=True,cache_context_exact=True,gap_index=row['gap_index']))
            a=data['oracle_coeff256'].double().numpy();an=float(np.linalg.norm(a));assert an>0
            geom=read(old/'GEOMETRY.json');assert abs(an/geom['delta_norms']['oracle']-1)<2e-6
            for k in cfg['ranks']:
                guard();arm=f'Shared-R{k}';reduced=a@b[:,:k];scale=an/np.linalg.norm(reduced)
                factors=reduced*scale;coeff=(factors@b[:,:k].T).astype(np.float32)
                delta_np=(coeff.astype(np.float64)@q.T).astype(np.float32)
                delta=torch.from_numpy(delta_np).to(stock.device);corrected=stock+delta
                assert abs(norm(delta)/geom['delta_norms']['oracle']-1)<2e-6
                assert abs(norm(delta)/norm(stock)-cfg['radius'])<2e-6
                with torch.no_grad(),shared_fields(adapter,{'event':corrected,'spatial':corrected},['event','spatial'],allow_prefix=True) as calls:
                    result=decode_shared_reference_two_pass(model,pr,prompt,adapter,fields)
                pred=prediction_record(result,row,pre,ADAPTER_SHA)
                pred.update(arm=arm,readout=details(result),support=support,source_GT_oracle=True,GT_read=True,
                    GT_purpose='existing source analytic direction projection',decoder_GT_prefix=False,target_read=False,optimizer_steps=0,
                    injection=dict(calls=calls,same_field_both_passes=True,common_F_sha=tensor_sha(stock),
                        corrected_F_sha=tensor_sha(corrected),delta_sha=tensor_sha(delta),relative_norm=norm(delta)/norm(stock)))
                validate_prediction(pred,len(row['input']['frame_ids']));torch.save(pred,ep/(arm+'.pt'))
                np.save(ep/(arm+'_FACTORS.npy'),factors)
                write(ep/(arm+'_FIELD.json'),dict(rank=k,shape=list(a.shape),original_coeff_norm=an,
                    projected_coeff_norm=float(np.linalg.norm(reduced)),scale=float(scale),actual_coeff_norm=float(np.linalg.norm(coeff.astype(np.float64))),
                    actual_delta_norm=norm(delta),old_oracle_delta_norm=geom['delta_norms']['oracle'],stock_norm=norm(stock),
                    coefficient_sha=tensor_sha(torch.from_numpy(coeff)),delta_sha=tensor_sha(delta),
                    numpy_FP64_projection_then_FP32_coeff_and_FP32_delta=True,same_field_both_passes=True,calls=calls))
                del factors,reduced,coeff,delta_np,delta,corrected,result,pred;gc.collect();torch.cuda.empty_cache()
            assert adapter_sha256(adapter)==ADAPTER_SHA and all(p.grad is None and not p.requires_grad for mod in (model,adapter,unused) for p in mod.parameters())
            assert torch.equal(unused.basis.cpu(),load(A0/'BASIS.pt'))
            write(ep/'COMPLETE.json',dict(index=i,files={p.name:sha(p) for p in ep.iterdir() if p.is_file()},
                new_native=2,reused_predictions=2,optimizer_steps=0,backwards=0,frozen_scope=True,physical_replay_exact=True))
            print('A03_NATIVE',i+1,64,flush=True)
            del data,inp,prompt,pre,fields,args,stock,a,prior;gc.collect();torch.cuda.empty_cache()
            assert sum(p.stat().st_size for p in (N/'episodes').rglob('*') if p.is_file())<cfg['maximum_new_bytes']
        write(N/'PREDICTIONS_SEAL.json',dict(files={str(p.relative_to(N)):sha(p) for p in (N/'episodes').rglob('*') if p.is_file()},predictions=256,queries=64,new_predictions=128))
        write(N/'COMPLETE.json',dict(predictions=256,queries=64,seal_sha=sha(N/'PREDICTIONS_SEAL.json')))
        write(RUN/'COMPLETE.json',dict(predictions=256,new_native=128,reused=128,optimizer_steps=0,backwards=0,peak_GPU_bytes=torch.cuda.max_memory_allocated()))


def launch():
    assert not (RUN/'STARTED.json').exists()
    start=time.monotonic();cmd=[str(ROOT/'.venv-ptd-audit/bin/python'),'-B',str(Path(__file__)),'worker']
    with (N/'RUN001.log').open('xb') as log:
        p=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        write(N/'ACTIVE.json',dict(status='running',pid=p.pid,wrapper_pid=os.getpid(),time=time.time(),cmd=cmd));code=p.wait()
    wall=time.monotonic()-start;receipt=read(RUN/'RECEIPT.json') if (RUN/'RECEIPT.json').exists() else dict(seconds=0)
    prior=total_prior();overhead=max(0.,wall-receipt['seconds'])
    write(OUT/'a03_native001_wrapper'/'RECEIPT.json',dict(status='completed' if code==0 else 'failed',seconds=overhead,
        worker_seconds=receipt['seconds'],child_wall_seconds=wall,prior_seconds=prior,cumulative_seconds=prior+overhead,cap=None))
    write(N/'EXIT.json',dict(code=code,time=time.time(),pid=p.pid))
    if code:raise RuntimeError('A03 native failed; preserve evidence, no automatic replay')
    print('A03_NATIVE_EXIT_0',flush=True)


if __name__=='__main__':
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','launch','worker']);a=p.parse_args()
    {'prepare':prepare,'launch':launch,'worker':worker}[a.action]()
