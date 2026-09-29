"""Seal-first full64 input/reuse/scope and 128 norm-matched fields readback."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='4'
import sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.desta3d_v3_a03_native import D,N,RUN,A0,GAP,ARMS,load,ADAPTER_SHA
from vg_tta.desta3d_v3_oracle_io import read,write,sha,check_pins,tensor_sha


def main():
    import numpy as np,torch
    torch.set_num_threads(4);start=time.monotonic();assert not (N/'ROOT_FIELD_READBACK.json').exists()
    try:
        assert read(N/'EXIT.json')['code']==0 and read(RUN/'RECEIPT.json')['status']=='completed'
        done=read(N/'COMPLETE.json');seal=read(N/'PREDICTIONS_SEAL.json')
        assert done['seal_sha']==sha(N/'PREDICTIONS_SEAL.json') and seal['predictions']==256
        check_pins({str(N/p):h for p,h in seal['files'].items()});check_pins(read(N/'LOCK.json')['pins']);check_pins(read(N/'REUSE.json')['files'])
        b=np.load(D/'TRAIN_BASIS.npz')['basis'];q=load(A0/'BASIS.pt').double().numpy();rows=read(N/'INPUTS.json')
        maxerr=0.;normerr=0.;facterr=0.;fields=0
        for i,row in enumerate(rows):
            ep=N/'episodes'/f'{i:04}';inp=read(ep/'INPUT.json');cache=load(A0/'cache/dev'/f'{i:04}'/'CACHE.pt')
            a=cache['oracle_coeff256'].double();an=float(a.norm());old=GAP/'episodes'/f"{row['gap_index']:04}"
            assert inp['physical_replay_exact'] and inp['cache_context_exact']
            comp=read(ep/'COMPLETE.json');assert comp['frozen_scope'] and comp['new_native']==2 and comp['optimizer_steps']==comp['backwards']==0
            for arm in ('B1','oracle'):assert sha(ep/(arm+'.pt'))==sha(old/(arm+'.pt'))
            for k in (16,32):
                arm=f'Shared-R{k}';r=read(ep/(arm+'_FIELD.json'));pred=load(ep/(arm+'.pt'))
                assert pred['support']==inp['support'] and pred['preprocess']==inp['preprocess'] and pred['adapter_sha']==ADAPTER_SHA
                assert pred['key']==row['key'] and pred['source']==row['source'] and pred['source_GT_oracle']
                assert not pred['decoder_GT_prefix'] and not pred['target_read'] and pred['optimizer_steps']==0
                assert r['calls'] in (['event'],['event','spatial']) and r['same_field_both_passes']
                f=np.load(ep/(arm+'_FACTORS.npy'));tb=torch.from_numpy(b[:,:k].copy());reduced=a@tb;tf=reduced*(a.norm()/reduced.norm())
                facterr=max(facterr,float((tf-torch.from_numpy(f)).abs().max()))
                coeff=(f@b[:,:k].T).astype(np.float32);delta=(coeff.astype(np.float64)@q.T).astype(np.float32)
                assert tensor_sha(torch.from_numpy(coeff))==r['coefficient_sha']
                assert tensor_sha(torch.from_numpy(delta))==r['delta_sha']==pred['injection']['delta_sha']
                independent_coeff=(tf@tb.T).float();independent_delta=(independent_coeff.double()@torch.from_numpy(q).T).float()
                maxerr=max(maxerr,float((independent_delta-torch.from_numpy(delta)).abs().max()))
                dn=float(np.linalg.norm(delta.astype(np.float64)))
                normerr=max(normerr,abs(np.linalg.norm(f)/an-1),abs(dn/r['actual_delta_norm']-1),abs(dn/r['old_oracle_delta_norm']-1))
                assert abs(dn/r['stock_norm']-read(N/'CONFIG.json')['radius'])<2e-6
                fields+=1
        assert fields==128 and facterr<1e-10 and maxerr<2e-6 and normerr<2e-6
        write(N/'ROOT_FIELD_READBACK.json',dict(status='passed',queries=64,predictions=256,fields=fields,reused_hashes=128,
            factor_max_absolute_error=facterr,independent_delta_max_absolute_error=maxerr,norm_max_relative_error=normerr,
            actual_delta_hash_reconstructed=True,CPU_seconds=time.monotonic()-start,
            limitation='Corrected-F hashes and full physical replay/frozen parameters are worker observations; full stock F not persisted again. Prior B1/Full Oracle reused after sealed hash and exact current input/context identity, no extra baseline native generation.'))
        print('A03_FIELD_AUDIT_PASS',maxerr,normerr,flush=True)
    except BaseException as e:
        write(N/'FIELD_AUDIT_FAILURE.json',dict(error=repr(e),traceback=traceback.format_exc()));raise


if __name__=='__main__':main()
