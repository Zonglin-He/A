"""Conditional A0.2 CPU structure audit, complete192 sealed targets, no model."""
import os,sys,time,json,traceback,shutil
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='4'
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.desta3d_v3_a02_screen import A0,D,load
from vg_tta.desta3d_v3_oracle_io import read,write,sha,check_pins
S=D/'structure_audit_v1'

def main():
    import numpy as np,torch
    from vg_tta.desta3d_v3_oracle_structure import field_statistics,torch_reference,summarize
    torch.set_num_threads(4)
    assert read(D/'ROOT_READBACK.json')['status']=='passed'
    assert read(D/'PUBLIC_REPORT.json')['decision']=='stop_architecture_tuning_cached_structure_audit'
    assert not S.exists(),'Write-once CPU audit'
    pre=ROOT/'artifacts/desta3d_v3/oracle_structure_CPU_PREFLIGHT.json';assert read(pre)['status']=='passed'
    pins={str(p):sha(p) for p in [Path(__file__),ROOT/'vg_tta/desta3d_v3_oracle_structure.py',pre,
      D/'PUBLIC_REPORT.json',D/'ROOT_READBACK.json',D/'CACHE_REUSE.json',ROOT/'protocols/desta3d_v3_a02_global_mean_v1.md']}
    write(S/'REGISTRATION.json',dict(time=time.time(),queries=192,train=128,dev=64,CPU_only=True,threads=4,phase_seconds=900,
      estimand='FP64 complete uncentered THWC field; equal query split aggregates',pins=pins))
    check_pins(pins);check_pins(read(D/'CACHE_REUSE.json')['files'])
    start=time.monotonic();status='failed';maxerr=0.;maxenergy=0.;maxspec=0.;records={}
    try:
        for split,count in [('train',128),('dev',64)]:
            rows=[]
            for i in range(count):
                assert time.monotonic()-start<900 and shutil.disk_usage(ROOT).free>=8*2**30
                p=A0/'cache'/split/f'{i:04}'/'CACHE.pt';c=load(p);a=c['oracle_coeff256'].numpy()
                assert a.ndim==5 and a.shape[0]==1 and a.shape[-1]==256
                r=field_statistics(a[0]);ref=torch_reference(a[0]);assert ref is not None
                maxenergy=max(maxenergy,abs(r['energy']-ref['energy'])/r['energy'])
                for k,v in r['metrics'].items():
                    b=ref['metrics'][k];assert (v is None)==(b is None)
                    if v is not None:maxerr=max(maxerr,abs(v-b))
                for axis in r['neighbors']:
                    for k,v in r['neighbors'][axis].items():
                        b=ref['neighbors'][axis][k];assert (v is None)==(b is None)
                        if k in ('pairs','defined','undefined'):assert v==b
                        elif v is not None:maxerr=max(maxerr,abs(v-b))
                maxspec=max(maxspec,float(np.max(np.abs(np.array(r['singular_value_squared'])-ref['singular_value_squared']))/r['energy']))
                assert max(maxerr,maxenergy,maxspec)<1e-10
                row=dict(index=i,cache_sha=sha(p),**r)
                write(S/'rows'/split/f'{i:04}.json',row);rows.append(row)
                if (i+1)%32==0:print('STRUCTURE',split,i+1,flush=True)
            records[split]=rows
        summary={split:dict(queries=len(rows),statistics=summarize(rows)) for split,rows in records.items()}
        write(S/'PUBLIC_REPORT.json',dict(status='completed_dual_implementation_verified',summary=summary,
          ranks=[1,4,8,16,32],centering=False,adjacency='observed grid, not motion-correspondence',aggregation='equal query within split',
          new_labels=False,PTD_loaded=False,GPU=False,fresh_read=False,
          numeric_max_absolute_error=maxerr,energy_max_relative_error=maxenergy,spectrum_max_relative_error=maxspec))
        write(S/'SEAL.json',dict(files={str(p.relative_to(S)):sha(p) for p in [S/'PUBLIC_REPORT.json',*(S/'rows').rglob('*.json')]}))
        write(S/'COMPLETE.json',dict(status='completed',queries=192,seal_sha=sha(S/'SEAL.json')));status='completed'
        print('STRUCTURE_COMPLETE',maxerr,maxenergy,maxspec,flush=True)
    except BaseException as exc:
        write(S/'FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc()));raise
    finally:write(S/'CPU_RECEIPT.json',dict(status=status,seconds=time.monotonic()-start,GPU_seconds=0,threads=4))

if __name__=='__main__':main()
