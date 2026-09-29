"""Independent Torch FP64 full-cache reconstruction plus stdlib summary audit."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='4'
import sys,time,statistics,math,shutil,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from vg_tta.desta3d_v3_oracle_io import OUT,read,write,sha,check_pins
D=OUT/'a03_shared_basis_v1';A0=OUT/'a0_fast_screen_v1';A2=OUT/'a02_global_mean_v1'


def main():
    import torch,numpy as np
    torch.set_num_threads(4);start=time.monotonic();status='failed'
    assert not (D/'ROOT_READBACK.json').exists(),'No duplicate independent audit'
    try:
        assert read(D/'COMPLETE.json')['seal_sha']==sha(D/'SEAL.json')
        check_pins({str(D/p):h for p,h in read(D/'SEAL.json')['files'].items()})
        check_pins(read(D/'LOCK.json')['pins']);pins=read(A2/'CACHE_REUSE.json')['files']
        def field(split,i):
            assert time.monotonic()-start<900 and shutil.disk_usage(ROOT).free>=8*2**30
            p=A0/'cache'/split/f'{i:04}'/'CACHE.pt';assert sha(p)==pins[str(p)]
            return torch.load(p,map_location='cpu',weights_only=False)['oracle_coeff256'].double().reshape(-1,256)
        mean=torch.zeros(256,256,dtype=torch.float64)
        for i in range(128):
            x=field('train',i);mean.add_(x.T@x / x.square().sum(),alpha=1/128)
        eig,vec=torch.linalg.eigh(mean);eig=eig.flip(0);vec=vec.flip(1)
        saved=np.load(D/'TRAIN_BASIS.npz');basis=torch.from_numpy(saved['basis'].copy());rand=torch.from_numpy(saved['random_basis'].copy())
        cov_error=float((mean-torch.from_numpy(saved['covariance'])).abs().max())
        spectrum_error=float((eig-torch.from_numpy(saved['eigenvalues'])).abs().max())
        ortho_error=max(float((b.T@b-torch.eye(256,dtype=torch.float64)).abs().max()) for b in (basis,rand))
        ranks=(1,4,8,16,32,64);projector_errors={}
        for k in ranks:
            projector_errors[str(k)]=float((vec[:,:k]@vec[:,:k].T-basis[:,:k]@basis[:,:k].T).abs().max())
        assert cov_error<1e-12 and spectrum_error<1e-12 and ortho_error<1e-10 and max(projector_errors.values())<1e-8
        report=read(D/'PUBLIC_REPORT.json');maxerr=0.;direct_error=0.;rowschecked=0;scalars=0
        for split,count in [('train',128),('dev',64)]:
            rows=[]
            for i in range(count):
                x=field(split,i);energy=x.square().sum();row=read(D/'rows'/split/f'{i:04}.json');rows.append(row)
                assert row['index']==i
                spectrum=torch.linalg.eigvalsh(x.T@x).flip(0).clamp_min(0)
                for k in ranks:
                    optimal=float(spectrum[:k].sum()/energy)
                    maxerr=max(maxerr,abs(optimal-row['optimal'][str(k)]['energy']))
                    for kind,b in [('shared',basis),('random',rand)]:
                        y=x@b[:,:k];e=float(y.square().sum()/energy)
                        r=row[kind][str(k)]
                        for key,val in [('energy',e),('cosine',math.sqrt(e)),('retention_ratio',e/optimal)]:
                            maxerr=max(maxerr,abs(r[key]-val));scalars+=1
                        if kind=='shared':
                            independent_energy=float((x@vec[:,:k]).square().sum()/energy)
                            maxerr=max(maxerr,abs(e-independent_energy))
                    # Direct flattened full projection validates the sqrt identity and norm contract.
                    if k in (16,32):
                        p=(x@basis[:,:k])@basis[:,:k].T
                        cosine=float((x*p).sum()/(x.norm()*p.norm()))
                        direct_error=max(direct_error,abs(cosine-row['shared'][str(k)]['cosine']))
                        matched=p*(x.norm()/p.norm())
                        direct_error=max(direct_error,abs(float(matched.norm()/x.norm())-1))
                rowschecked+=1
            for kind in ('shared','random','optimal'):
                for k in ranks:
                    for metric,stats in report['summary'][split][kind][str(k)].items():
                        values=[r[kind][str(k)][metric] for r in rows]
                        assert stats['defined']==count
                        for key,val in [('mean',statistics.fmean(values)),('median',statistics.median(values)),('min',min(values)),('max',max(values))]:
                            maxerr=max(maxerr,abs(stats[key]-val))
            print('INDEPENDENT',split,count,flush=True)
        assert rowschecked==192 and max(maxerr,direct_error)<1e-10
        gate=report['dev_rank32_median_energy']>=.75
        assert report['decision']==('conditional_dev64_native' if gate else 'stop_fixed_shared_basis')
        assert read(D/'TRAIN_BASIS_SEAL.json')['dev_fields_opened']==0
        write(D/'ROOT_READBACK.json',dict(status='passed',queries=rowschecked,projection_scalars=scalars,
            normalized_covariance_max_abs=cov_error,eigenspectrum_max_abs=spectrum_error,
            orthogonality_max_abs=ortho_error,projector_errors=projector_errors,
            scalar_summary_max_abs=maxerr,direct_projection_cosine_norm_max_abs=direct_error,
            gate_pass=gate,CPU_seconds=time.monotonic()-start,
            scope='All192 raw caches; independently reconstructed train-only Torch basis, both controls, complete spectra, direct R16/R32 projection identities and norm matching, stdlib summaries',
            native_run=False,fresh_read=False,GPU_seconds=0));status='completed'
        print('A03_INDEPENDENT_PASS',maxerr,direct_error,'GATE',gate,flush=True)
    except BaseException as exc:
        write(D/'INDEPENDENT_FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc()));raise
    finally:
        write(D/'INDEPENDENT_CPU_RECEIPT.json',dict(status=status,seconds=time.monotonic()-start,GPU_seconds=0))


if __name__=='__main__':main()
