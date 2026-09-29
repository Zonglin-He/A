"""A0.3 train-only shared channel basis; cached CPU feasibility before native."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key] = '4'
import sys,time,argparse,shutil,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from vg_tta.desta3d_v3_oracle_io import OUT,read,write,sha,check_pins,total_prior
A0=OUT/'a0_fast_screen_v1';A2=OUT/'a02_global_mean_v1';S=A2/'structure_audit_v1';D=OUT/'a03_shared_basis_v1'


def prepare():
    assert not D.exists(),'Write-once registration'
    assert read(A2/'ROOT_READBACK.json')['status']=='passed'
    assert read(S/'ROOT_CROSSCHECK.json')['status']=='passed'
    pre=ROOT/'artifacts/desta3d_v3/a03_CPU_PREFLIGHT.json'
    assert read(pre)['status']=='passed'
    cfg=dict(stage='A0.3 shared channel basis feasibility',train=128,dev=64,ranks=[1,4,8,16,32,64],
        basis_fit_split='train_only',weighting='equal query normalized covariance',centered=False,
        random_seed=20260929,native_median_energy_threshold=.75,phase_seconds=900,
        minimum_free_bytes=8*2**30,maximum_new_bytes=256*2**20,GPU_seconds=0,
        prior_GPU_seconds=total_prior(),cumulative_cap=None,fresh_read=False,PTD_loaded=False)
    write(D/'CONFIG.json',cfg)
    paths=[Path(__file__),ROOT/'vg_tta/desta3d_v3_shared_channel_basis.py',
        ROOT/'tests/test_desta3d_v3_shared_channel_basis.py',ROOT/'scripts/audit_desta3d_v3_a03_shared_basis.py',
        ROOT/'protocols/desta3d_v3_a03_shared_channel_basis_v1.md',pre,A0/'BASIS.pt',A0/'CACHE_SEAL.json',
        A0/'TRAIN128.json',A0/'DEV64.json',A0/'CONFIG.json',A2/'CACHE_REUSE.json',S/'SEAL.json',S/'ROOT_CROSSCHECK.json',D/'CONFIG.json']
    write(D/'LOCK.json',dict(pins={str(p):sha(p) for p in paths}))
    write(D/'REGISTRATION.json',dict(status='registered_before_measurement',time=time.time(),source_GT_derived=True,
        new_labels=False,optimizer_steps=0,native=0,conditional_native_locked=True))
    print('A03_REGISTERED',flush=True)


def run():
    import numpy as np,torch
    from vg_tta.desta3d_v3_shared_channel_basis import normalized_covariance,fit_train_basis,random_basis,projection_statistics,aggregate,energy_gate
    torch.set_num_threads(4)
    assert not (D/'STARTED.json').exists(),'No silent replay'
    cfg=read(D/'CONFIG.json');check_pins(read(D/'LOCK.json')['pins'])
    start=time.monotonic();status='failed'
    write(D/'STARTED.json',dict(time=time.time(),pid=os.getpid(),CPU_only=True))
    def guard():
        assert time.monotonic()-start<cfg['phase_seconds']
        assert shutil.disk_usage(ROOT).free>=cfg['minimum_free_bytes']
    def field(split,index):
        guard();p=A0/'cache'/split/f'{index:04}'/'CACHE.pt'
        assert sha(p)==read(A2/'CACHE_REUSE.json')['files'][str(p)]
        a=torch.load(p,map_location='cpu',weights_only=False)['oracle_coeff256'].numpy()
        assert a.ndim==5 and a.shape[0]==1 and a.shape[-1]==256
        return a[0],sha(p)
    try:
        covs=[]
        for i in range(128):
            a,_=field('train',i);covs.append(normalized_covariance(a))
        covariance,eig,basis=fit_train_basis(covs)
        rand=random_basis(256,cfg['random_seed'])
        np.savez(D/'TRAIN_BASIS.npz',covariance=covariance,eigenvalues=eig,basis=basis,random_basis=rand)
        write(D/'TRAIN_BASIS_SEAL.json',dict(sha256=sha(D/'TRAIN_BASIS.npz'),train_count=128,
            dev_fields_opened=0,created_time=time.time(),query_trace_range=[float(np.trace(c)) for c in [min(covs,key=lambda c:np.trace(c)),max(covs,key=lambda c:np.trace(c))]],
            rank_boundary_gaps={str(k):float(eig[k-1]-eig[k]) for k in cfg['ranks']}))
        del covs
        print('TRAIN_BASIS_SEALED_BEFORE_DEV',flush=True)
        records={}
        oldseal=read(S/'SEAL.json')['files']
        for split,count in [('train',128),('dev',64)]:
            rows=[]
            for i in range(count):
                a,ch=field(split,i);old=S/'rows'/split/f'{i:04}.json'
                assert sha(old)==oldseal[str(old.relative_to(S))]
                prior=read(old);assert prior['cache_sha']==ch and prior['shape']==list(a.shape)
                energy=float(np.sum(np.asarray(a,dtype=np.float64)**2))
                assert abs(energy/prior['energy']-1)<1e-12
                optimal={str(k):dict(energy=float(sum(prior['singular_value_squared'][:k])/energy),
                    cosine=float(np.sqrt(sum(prior['singular_value_squared'][:k])/energy))) for k in cfg['ranks']}
                result={}
                for name,b in [('shared',basis),('random',rand)]:
                    result[name]=projection_statistics(a,b)
                    for k,m in result[name].items():
                        m['retention_ratio']=m['energy']/optimal[k]['energy']
                        assert -1e-10<=m['energy']<=optimal[k]['energy']+1e-10
                row=dict(index=i,shape=list(a.shape),cache_sha=ch,optimal_spectrum_sha=sha(old),optimal=optimal,**result)
                write(D/'rows'/split/f'{i:04}.json',row);rows.append(row)
                if (i+1)%32==0:print('PROJECTED',split,i+1,flush=True)
            records[split]=rows
        summary={split:aggregate(rows) for split,rows in records.items()}
        median=summary['dev']['shared']['32']['energy']['median']
        write(D/'PUBLIC_REPORT.json',dict(status='completed_pending_independent_audit',queries=192,
            train_queries=128,dev_queries=64,summary=summary,ranks=cfg['ranks'],
            decision=energy_gate(median),dev_rank32_median_energy=median,
            basis_fit='Train128 only, query normalized uncentered FP64 covariance',
            random_control_seed=cfg['random_seed'],native_executed=False,new_optimizer_steps=0,
            source_GT_derived_targets=True,PTD_loaded=False,fresh_read=False))
        paths=[D/'TRAIN_BASIS.npz',D/'TRAIN_BASIS_SEAL.json',D/'PUBLIC_REPORT.json',*(D/'rows').rglob('*.json')]
        write(D/'SEAL.json',dict(files={str(p.relative_to(D)):sha(p) for p in paths}))
        size=sum(p.stat().st_size for p in D.rglob('*') if p.is_file());assert size<=cfg['maximum_new_bytes'];guard()
        write(D/'COMPLETE.json',dict(status='completed_pending_independent_audit',queries=192,
            seal_sha=sha(D/'SEAL.json'),output_bytes=size));status='completed'
        print('A03_COMPLETE',median,energy_gate(median),flush=True)
    except BaseException as exc:
        write(D/'FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc()));raise
    finally:
        write(D/'CPU_RECEIPT.json',dict(status=status,seconds=time.monotonic()-start,GPU_seconds=0,threads=4,
            includes='cache and spectrum hashes, train covariance, all projections, serialization'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);a=p.parse_args()
    prepare() if a.action=='prepare' else run()
