"""A0.2 cached-only global mean screen; never loads PTD or fresh data."""
import argparse,os,sys,time,random,subprocess,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from vg_tta.desta3d_v3_oracle_io import OUT,read,write,sha,check_pins,allocation,total_prior
A0=OUT/'a0_fast_screen_v1';D=OUT/'a02_global_mean_v1'
ARMS={'GMean-S200':(128,200),'GMean-S2000':(128,2000)}
def load(p):
    import torch
    return torch.load(p,map_location='cpu',weights_only=False)

def prepare():
    assert not D.exists(),'Write-once registration'
    assert read(A0/'ROOT_CACHE_READBACK.json')['status']=='passed'
    assert read(A0/'ROOT_DIRECTION_READBACK.json')['status']=='passed'
    pre=ROOT/'artifacts/desta3d_v3/a02_CPU_PREFLIGHT.json';assert read(pre)['status']=='passed'
    cfg=dict(seed=20260928,feature_dim=128,state_dim=33,evidence_dim=8,radius=.13545580427763146,
      lr=.001,weight_decay=0,clip=1,batch_size=4,loss='query_global_coefficient_1_minus_cosine',
      phase_seconds=900,minimum_free_bytes=8*2**30,maximum_output_bytes=4*2**30,cumulative_cap=None,
      prior_seconds=total_prior(),arms={k:dict(hidden_dim=v[0],steps=v[1]) for k,v in ARMS.items()},
      global_mean=True,PTD_load=False,native=False,fresh_read=False)
    write(D/'CONFIG.json',cfg)
    paths=[Path(__file__),ROOT/'vg_tta/desta3d_v3_a02_global_mean.py',ROOT/'vg_tta/desta3d_v3_a01_triage.py',ROOT/'vg_tta/desta3d_v3_a0_screen.py',
      ROOT/'vg_tta/desta3d_v3_gap_candidates.py',ROOT/'vg_tta/optimizer_checkpoint.py',
      ROOT/'scripts/audit_desta3d_v3_a02_screen.py',ROOT/'protocols/desta3d_v3_a02_global_mean_v1.md',pre,
      A0/'BASIS.pt',A0/'CONFIG.json',A0/'CACHE_SEAL.json',A0/'TRAIN128.json',A0/'DEV64.json',
      OUT/'a01_fitability_v1/W128-S2000/REPORT.json',OUT/'a01_fitability_v1/W128-S2000/SEAL.json',A0/'DIRECTION_REPORT.json',A0/'FIT_SEAL.json',A0/'FINAL.pt',A0/'HISTORY.json',A0/'ROOT_CACHE_READBACK.json',A0/'ROOT_DIRECTION_READBACK.json']
    seal=read(A0/'CACHE_SEAL.json')['files'];cachepins={}
    for split,count in [('train',128),('dev',64)]:
        for i in range(count):
            p=A0/'cache'/split/f'{i:04}'/'CACHE.pt';rel=str(p.relative_to(A0));assert sha(p)==seal[rel]
            cachepins[str(p)]=seal[rel]
    write(D/'CACHE_REUSE.json',dict(files=cachepins,count=192,old_seal_sha=sha(A0/'CACHE_SEAL.json'),
      old_independent_audit=sha(A0/'ROOT_CACHE_READBACK.json'),new_native=0,new_oracle_backwards=0))
    paths.extend([D/'CONFIG.json',D/'CACHE_REUSE.json'])
    pins={str(p):sha(p) for p in paths};write(D/'LOCK.json',dict(pins=pins))
    write(D/'REGISTRATION.json',dict(time=time.time(),status='registered_before_GPU',prior_seconds=cfg['prior_seconds'],
      scientific_arms=2,added_parameters=0))
    for arm,(width,steps) in ARMS.items():
        write(D/arm/'CONFIG.json',{**cfg,'arm':arm,'hidden_dim':width,'steps':steps})
        write(D/arm/'LOCK.json',dict(pins={**pins,str(D/arm/'CONFIG.json'):sha(D/arm/'CONFIG.json')}))
    print('REGISTERED',cfg,flush=True)

def worker(arm):
    dest=D/arm
    with allocation(dest) as (cfg,guard):
        import torch,numpy as np
        from vg_tta.desta3d_v3_a02_global_mean import GlobalMeanDirectionMixer
        from vg_tta.desta3d_v3_a02_global_mean import coefficients
        from vg_tta.desta3d_v3_gap_candidates import coefficient_direction_loss
        from vg_tta.optimizer_checkpoint import cpu_clone,validate_serialized_optimizer,restore_optimizer
        torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True
        torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
        check_pins(read(D/'CACHE_REUSE.json')['files'])
        basis=load(A0/'BASIS.pt');m=GlobalMeanDirectionMixer(basis,feature_dim=128,hidden_dim=cfg['hidden_dim'],radius=cfg['radius']).cuda()
        opt=torch.optim.AdamW(m.parameters(),lr=cfg['lr'],weight_decay=0)
        initial=cpu_clone(m.state_dict());old=load(A0/'FINAL.pt')
        init_exact=all(torch.equal(v,old['initial'][k]) for k,v in initial.items()) if cfg['hidden_dim']==128 else None
        if cfg['hidden_dim']==128:assert init_exact
        assert m.input.in_features==425 and m.input.out_features==cfg['hidden_dim']
        torch.save(dict(mixer=initial,seed=cfg['seed']),dest/'INITIAL.pt')
        rng=random.Random(cfg['seed']);order=[];history=[]
        for step in range(1,cfg['steps']+1):
            guard();opt.zero_grad(set_to_none=True);indices=[];losses=[]
            for j in range(4):
                if not order:order=list(range(128));rng.shuffle(order)
                i=order.pop()
                indices.append(i);c=load(A0/'cache/train'/f'{i:04}'/'CACHE.pt')
                c={k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in c.items()}
                a=coefficients(m,c);obj=coefficient_direction_loss(a,c['oracle_coeff256'])['cosine']
                assert torch.isfinite(obj);(obj/4).backward();losses.append(float(obj.detach()))
                del c,a,obj
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
            gn=float(torch.nn.utils.clip_grad_norm_(m.parameters(),cfg['clip']));opt.step()
            counters=[int(s['step']) for s in opt.state.values()];assert set(counters)=={step}
            history.append(dict(step=step,indices=indices,loss=float(np.mean(losses)),gradient_norm=gn,counters=counters))
            if arm=='GMean-S2000' and step==200:
                matched=load(D/'GMean-S200/FINAL.pt')
                now=cpu_clone(m.state_dict());diff=max(float((v-matched['mixer'][k]).abs().max()) for k,v in now.items())
                write(dest/'GMEAN_STEP200_IDENTITY.json',dict(exact=diff==0,max_abs=diff,short_final_sha=sha(D/'GMean-S200/FINAL.pt'),selection_use=False))
                torch.save(dict(mixer=now,optimizer=cpu_clone(opt.state_dict())),dest/'STEP200_IDENTITY.pt')
                assert diff==0,'GMean first200 differs from short arm; preserve and audit'
            if step%100==0:print(arm,'STEP',step,'LOSS',history[-1]['loss'],'GN',gn,flush=True)
        state=cpu_clone(opt.state_dict());validate_serialized_optimizer(state)
        torch.save(dict(mixer=cpu_clone(m.state_dict()),optimizer=state,initial=initial,steps=cfg['steps'],
           seed=cfg['seed'],cache_seal_sha=sha(A0/'CACHE_SEAL.json'),lock_sha=sha(dest/'LOCK.json')),dest/'FINAL.pt')
        write(dest/'HISTORY.json',history);test=torch.optim.AdamW(m.parameters(),lr=cfg['lr'],weight_decay=0);restore_optimizer(test,state)
        assert all(int(v['step'])==cfg['steps'] for v in test.state.values()) and torch.equal(m.basis.cpu(),basis)
        results={};m.eval()
        for split,count in [('train',128),('dev',64)]:
            entries=[]
            for i in range(count):
                guard();c=load(A0/'cache'/split/f'{i:04}'/'CACHE.pt');c={k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in c.items()}
                with torch.no_grad():a=coefficients(m,c)
                p=a.cpu();t=c['oracle_coeff256'].cpu();den=p.double().norm()*t.double().norm()
                cos=float((p.double()*t.double()).sum()/den) if den>0 else None
                path=dest/'terminal_coefficients'/split/f'{i:04}.pt';path.parent.mkdir(parents=True,exist_ok=True);torch.save(p,path)
                entries.append(dict(index=i,cosine=cos,sha=sha(path)));del c,a,p,t
            vals=[x['cosine'] for x in entries if x['cosine'] is not None]
            results[split]=dict(rows=entries,defined=len(vals),undefined=count-len(vals),mean=float(np.mean(vals)),median=float(np.median(vals)),count_gt_point1=sum(v>.1 for v in vals),count_gt_point3=sum(v>.3 for v in vals),terminal_mean_loss=float(1-np.mean(vals)))
        gn=[r['gradient_norm'] for r in history]
        write(dest/'REPORT.json',dict(status='completed_pending_independent_audit',arm=arm,results=results,
          steps=cfg['steps'],terminal_training_batch_loss=history[-1]['loss'],gradient_norm=dict(min=min(gn),max=max(gn),last=gn[-1],clipped=sum(x>1 for x in gn)),
          trainable_parameters=sum(p.numel() for p in m.parameters()),initial_exact_A0=init_exact,PTD_loaded=False,new_native=0,fresh_read=False,
          peak_GPU_bytes=torch.cuda.max_memory_allocated()))
        files=[dest/'INITIAL.pt',dest/'FINAL.pt',dest/'HISTORY.json',dest/'REPORT.json',*(dest/'terminal_coefficients').rglob('*.pt')]
        files += [p for p in [dest/'GMEAN_STEP200_IDENTITY.json',dest/'STEP200_IDENTITY.pt'] if p.exists()]
        write(dest/'SEAL.json',dict(files={str(p.relative_to(dest)):sha(p) for p in files}))
        size=sum(p.stat().st_size for p in D.rglob('*') if p.is_file());assert size<=cfg['maximum_output_bytes']
        write(dest/'COMPLETE.json',dict(steps=cfg['steps'],seal_sha=sha(dest/'SEAL.json'),output_bytes=size))
        print('ARM_COMPLETE',arm,{k:{x:v for x,v in r.items() if x!='rows'} for k,r in results.items()},flush=True)

def launch():
    check_pins(read(D/'LOCK.json')['pins'])
    for arm in ARMS:
        dest=D/arm;assert not (dest/'STARTED.json').exists()
        start=time.monotonic();cmd=[str(ROOT/'.venv-ptd-audit/bin/python'),'-B',str(Path(__file__)), 'worker','--arm',arm]
        with (dest/'RUN001.log').open('xb') as log:
            child=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            (D/'ACTIVE.json').write_text(json.dumps(dict(status='running',arm=arm,pid=child.pid,wrapper_pid=os.getpid(),time=time.time(),cmd=cmd),indent=2)+'\n')
            code=child.wait()
        wall=time.monotonic()-start;worker_seconds=read(dest/'RECEIPT.json')['seconds'] if (dest/'RECEIPT.json').exists() else 0
        # Standard v3 ledger scans OUT/*/RECEIPT, so expose measured arm receipts there.
        receipt=read(dest/'RECEIPT.json') if (dest/'RECEIPT.json').exists() else dict(status='failed_before_allocation',seconds=0)
        write(OUT/('a02_'+arm)/'RECEIPT.json',receipt)
        overhead=max(0.,wall-worker_seconds);prior=total_prior()
        write(OUT/('a02_'+arm+'_wrapper')/'RECEIPT.json',dict(status='completed' if code==0 else 'failed',seconds=overhead,worker_seconds=worker_seconds,child_wall_seconds=wall,prior_seconds=prior,cumulative_seconds=prior+overhead,cap=None))
        if code:raise RuntimeError('Arm failed, retained without retry: '+arm)
    (D/'ACTIVE.json').write_text(json.dumps(dict(status='completed_pending_root_audit',time=time.time()),indent=2)+'\n')

if __name__=='__main__':
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','launch','worker']);p.add_argument('--arm',choices=list(ARMS));a=p.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='launch':launch()
    else:worker(a.arm)
