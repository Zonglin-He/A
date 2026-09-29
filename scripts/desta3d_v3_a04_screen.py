"""A0.4 sealed-cache factorized direction screen, with audited conditional continuation."""
import argparse,os,sys,time,random,subprocess,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from vg_tta.desta3d_v3_oracle_io import OUT,read,write,sha,check_pins,allocation,total_prior
A0=OUT/'a0_fast_screen_v1';A3=OUT/'a03_shared_basis_v1';D=OUT/'a04_factorized_v1'

def load(p):
    import torch
    return torch.load(p,map_location='cpu',weights_only=False)

def prepare():
    import numpy as np,torch
    from threadpoolctl import threadpool_limits
    assert not D.exists(),'write-once registration'
    assert read(A3/'ROOT_READBACK.json')['status']=='passed'
    assert read(A0/'ROOT_CACHE_READBACK.json')['status']=='passed'
    pre=ROOT/'artifacts/desta3d_v3/a04_CPU_PREFLIGHT.json';assert read(pre)['status']=='passed'
    cfg=dict(seed=20260928,feature_dim=128,hidden_dim=128,output_rank=16,state_dim=33,evidence_dim=8,
      radius=.13545580427763146,lr=.001,weight_decay=0,clip=1,batch_size=4,first_steps=200,rescue_total_steps=2000,
      loss='query_global_R16_1_minus_cosine',phase_seconds=900,minimum_free_bytes=8*2**30,
      maximum_output_bytes=2*2**30,cumulative_cap=None,prior_seconds=total_prior(),PTD_load=False,native=False,fresh_read=False,
      numpy_blas_threads=4,torch_threads=4,target_projection='FP64_CPU_then_FP32',gate_target='FP64_exact_projection')
    assert shutil.disk_usage(ROOT).free>=cfg['minimum_free_bytes']+cfg['maximum_output_bytes']
    write(D/'CONFIG.json',cfg)
    pins={};derived={};start=time.monotonic();b=np.load(A3/'TRAIN_BASIS.npz')['basis'][:,:16].copy()
    assert np.max(np.abs(b.T@b-np.eye(16)))<1e-12
    np.save(D/'B16.npy',b)
    seal=read(A0/'CACHE_SEAL.json')['files']
    with threadpool_limits(limits=4,user_api='blas'):
        for split,n in [('train',128),('dev',64)]:
            for i in range(n):
                p=A0/'cache'/split/f'{i:04}'/'CACHE.pt';assert sha(p)==seal[str(p.relative_to(A0))];pins[str(p)]=sha(p)
                a=load(p)['oracle_coeff256'].numpy().astype(np.float64);target=(a@b).astype(np.float32)
                assert np.isfinite(target).all() and np.linalg.norm(target)>0
                dest=D/'targets16'/split/f'{i:04}.npy';dest.parent.mkdir(parents=True,exist_ok=True);np.save(dest,target);derived[str(dest)]=sha(dest)
    write(D/'CACHE_REUSE.json',dict(files=pins,count=192))
    write(D/'TARGETS_SEAL.json',dict(files=derived,count=192,basis_sha=sha(D/'B16.npy')))
    paths=[Path(__file__),ROOT/'scripts/audit_desta3d_v3_a04_screen.py',ROOT/'vg_tta/desta3d_v3_a04_factorized.py',
      ROOT/'vg_tta/desta3d_v3_a0_screen.py',ROOT/'vg_tta/desta3d_v3_gap_candidates.py',ROOT/'vg_tta/optimizer_checkpoint.py',
      ROOT/'protocols/desta3d_v3_a04_factorized_v1.md',pre,A0/'BASIS.pt',A0/'FINAL.pt',A0/'TRAIN128.json',A0/'DEV64.json',
      A0/'CACHE_SEAL.json',A3/'TRAIN_BASIS.npz',A3/'TRAIN_BASIS_SEAL.json',A3/'ROOT_READBACK.json',
      D/'CONFIG.json',D/'CACHE_REUSE.json',D/'TARGETS_SEAL.json',D/'B16.npy']
    pins={str(p):sha(p) for p in paths};write(D/'LOCK.json',dict(pins=pins))
    write(D/'REGISTRATION.json',dict(status='registered_before_GPU',time=time.time(),source_GT_cached_privilege=True,
      train_queries=128,train_parents=95,dev_queries=64,dev_parents=16,CPU_prepare_seconds=time.monotonic()-start))
    for steps in [200,2000]:
        dest=D/f'S{steps}';write(dest/'CONFIG.json',{**cfg,'steps':steps})
        write(dest/'LOCK.json',dict(pins={**pins,str(dest/'CONFIG.json'):sha(dest/'CONFIG.json')}))
    print('REGISTERED A0.4',cfg,flush=True)

def evaluate(m,dest,guard):
    import torch,numpy as np
    results={};b=torch.from_numpy(np.load(D/'B16.npy'))
    m.eval()
    for split,n in [('train',128),('dev',64)]:
        rows=[]
        for i in range(n):
            guard();c=load(A0/'cache'/split/f'{i:04}'/'CACHE.pt');a=c['oracle_coeff256'].double()
            c={k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in c.items()}
            with torch.no_grad():p=m(c).cpu()
            t=a@b;pd=p.double();recon=pd@b.T
            cos=float((pd*t).sum()/pd.norm()/t.norm());full=float((recon*a).sum()/recon.norm()/a.norm())
            rounded=torch.from_numpy(np.load(D/'targets16'/split/f'{i:04}.npy')).double()
            training_cos=float((pd*rounded).sum()/pd.norm()/rounded.norm())
            assert all(np.isfinite(x) for x in [cos,full,training_cos])
            path=dest/'terminal_coefficients'/split/f'{i:04}.pt';path.parent.mkdir(parents=True,exist_ok=True);torch.save(p,path)
            rows.append(dict(index=i,cosine=cos,full_oracle_cosine=full,training_cosine=training_cos,projection_energy=float(t.square().sum()/a.square().sum()),sha=sha(path)))
        vals=np.array([x['cosine'] for x in rows]);full=np.array([x['full_oracle_cosine'] for x in rows])
        results[split]=dict(rows=rows,defined=n,mean=float(vals.mean()),median=float(np.median(vals)),
          count_gt_point1=int((vals>.1).sum()),count_gt_point3=int((vals>.3).sum()),terminal_mean_loss=float(1-vals.mean()),
          rounded_training_target_mean_loss=float(1-np.mean([x['training_cosine'] for x in rows])),
          full_oracle_mean=float(full.mean()),full_oracle_median=float(np.median(full)))
    return results

def worker(steps):
    dest=D/f'S{steps}'
    with allocation(dest) as (cfg,guard):
        import torch,numpy as np
        from vg_tta.desta3d_v3_a04_factorized import FactorizedDirectionMixer
        from vg_tta.desta3d_v3_gap_candidates import coefficient_direction_loss
        from vg_tta.optimizer_checkpoint import cpu_clone,validate_serialized_optimizer,restore_optimizer
        torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
        check_pins(read(D/'CACHE_REUSE.json')['files']);check_pins(read(D/'TARGETS_SEAL.json')['files'])
        m=FactorizedDirectionMixer(load(A0/'BASIS.pt'),torch.from_numpy(np.load(D/'B16.npy')),cfg['radius']).cuda()
        opt=torch.optim.AdamW(m.parameters(),lr=cfg['lr'],weight_decay=0)
        initial=cpu_clone(m.state_dict());old_initial=load(A0/'FINAL.pt')['initial']
        assert all(torch.equal(v,old_initial[k]) for k,v in initial.items() if k.startswith(('input.','local.','mix.')))
        assert sum(p.numel() for p in m.parameters())==76688
        rng=random.Random(cfg['seed']);order=[];history=[];start_step=0
        if steps==2000:
            audit=read(D/'S200/ROOT_READBACK.json');assert audit['status']=='passed' and audit['decision']=='continue_same_trajectory_to_2000'
            ck=load(D/'S200/FINAL.pt');m.load_state_dict(ck['mixer']);restore_optimizer(opt,ck['optimizer'])
            rng.setstate(ck['sample_rng']);order=ck['sample_order'];history=read(D/'S200/HISTORY.json');start_step=200
            torch.set_rng_state(ck['torch_rng']);torch.cuda.set_rng_state_all(ck['cuda_rng'])
            assert all(torch.equal(v,ck['mixer'][k]) for k,v in cpu_clone(m.state_dict()).items())
            assert all(int(v['step'])==200 for v in opt.state.values()) and all(isinstance(k,int) for k in ck['optimizer']['state'])
            write(dest/'RESUME_READBACK.json',dict(status='passed',source_sha=sha(D/'S200/FINAL.pt'),actual_start_counter=200,live_bound=True,
              exact_state=True,RNG_and_order_restored=True,no_new_seed=True))
        torch.save(dict(mixer=initial,seed=cfg['seed']),dest/'INITIAL.pt')
        for step in range(start_step+1,steps+1):
            guard();opt.zero_grad(set_to_none=True);indices=[];losses=[]
            for _ in range(4):
                if not order:order=list(range(128));rng.shuffle(order)
                i=order.pop();indices.append(i)
                c=load(A0/'cache/train'/f'{i:04}'/'CACHE.pt');c={k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in c.items()}
                target=torch.from_numpy(np.load(D/'targets16/train'/f'{i:04}.npy')).cuda()
                p=m(c);loss=coefficient_direction_loss(p,target)['cosine'];assert torch.isfinite(loss)
                (loss/4).backward();losses.append(float(loss.detach()));del c,p,loss,target
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
            gn=float(torch.nn.utils.clip_grad_norm_(m.parameters(),1));opt.step()
            counters=[int(v['step']) for v in opt.state.values()];assert set(counters)=={step}
            history.append(dict(step=step,indices=indices,loss=float(np.mean(losses)),gradient_norm=gn,counters=counters))
            if step%100==0:print('STEP',step,'LOSS',history[-1]['loss'],'GN',gn,flush=True)
        optimizer=cpu_clone(opt.state_dict());validate_serialized_optimizer(optimizer)
        torch.save(dict(mixer=cpu_clone(m.state_dict()),optimizer=optimizer,initial=initial,steps=steps,seed=cfg['seed'],
          sample_rng=rng.getstate(),sample_order=order,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),lock_sha=sha(dest/'LOCK.json')),dest/'FINAL.pt')
        write(dest/'HISTORY.json',history)
        test=torch.optim.AdamW(m.parameters(),lr=cfg['lr'],weight_decay=0);restore_optimizer(test,optimizer)
        assert all(int(v['step'])==steps for v in test.state.values())
        for name in ('union','channel_basis','basis'):assert torch.equal(m.state_dict()[name].cpu(),initial[name])
        results=evaluate(m,dest,guard);grad=[h['gradient_norm'] for h in history]
        write(dest/'REPORT.json',dict(status='completed_pending_independent_audit',steps=steps,new_steps=steps-start_step,results=results,
          trainable_parameters=76688,terminal_training_batch_loss=history[-1]['loss'],gradient_norm=dict(min=min(grad),max=max(grad),last=grad[-1],clipped=sum(v>1 for v in grad)),
          PTD_loaded=False,native_predictions=0,fresh_read=False,peak_GPU_bytes=torch.cuda.max_memory_allocated()))
        files=[dest/'INITIAL.pt',dest/'FINAL.pt',dest/'HISTORY.json',dest/'REPORT.json',*(dest/'terminal_coefficients').rglob('*.pt')]
        if (dest/'RESUME_READBACK.json').exists():files.append(dest/'RESUME_READBACK.json')
        write(dest/'SEAL.json',dict(files={str(p.relative_to(dest)):sha(p) for p in files}))
        size=sum(p.stat().st_size for p in D.rglob('*') if p.is_file());assert size<=cfg['maximum_output_bytes']
        write(dest/'COMPLETE.json',dict(steps=steps,new_steps=steps-start_step,seal_sha=sha(dest/'SEAL.json'),bytes=size))
        print('COMPLETE',steps,{k:{n:v for n,v in r.items() if n!='rows'} for k,r in results.items()},flush=True)

def launch(steps):
    check_pins(read(D/'LOCK.json')['pins']);dest=D/f'S{steps}';assert not (dest/'STARTED.json').exists()
    if steps==2000:assert read(D/'S200/ROOT_READBACK.json')['decision']=='continue_same_trajectory_to_2000'
    start=time.monotonic();cmd=[str(ROOT/'.venv-ptd-audit/bin/python'),'-B',str(Path(__file__)),'worker','--steps',str(steps)]
    env={**os.environ,'OPENBLAS_NUM_THREADS':'4','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','CUBLAS_WORKSPACE_CONFIG':':4096:8'}
    with (dest/'RUN001.log').open('xb') as log:
        child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        write(D/'ACTIVE.json',dict(status='running',steps=steps,pid=child.pid,wrapper_pid=os.getpid(),cmd=cmd,time=time.time()))
        code=child.wait()
    wall=time.monotonic()-start
    receipt=read(dest/'RECEIPT.json') if (dest/'RECEIPT.json').exists() else dict(status='failed_before_allocation',seconds=0)
    write(OUT/f'a04_s{steps}'/'RECEIPT.json',receipt);overhead=max(0,wall-receipt['seconds']);prior=total_prior()
    write(OUT/f'a04_s{steps}_wrapper'/'RECEIPT.json',dict(status='completed' if code==0 else 'failed',seconds=overhead,worker_seconds=receipt['seconds'],child_wall_seconds=wall,prior_seconds=prior,cumulative_seconds=prior+overhead,cap=None))
    write(D/'ACTIVE.json',dict(status='completed_pending_audit' if code==0 else 'failed',steps=steps,exit_code=code,time=time.time()))
    assert code==0,'Failure preserved, no retry'

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','launch','worker']);p.add_argument('--steps',type=int,choices=[200,2000],default=200);a=p.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='launch':launch(a.steps)
    else:worker(a.steps)
