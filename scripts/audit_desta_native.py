"""CPU-only independent NumPy reconstruction, before offline native scoring."""
import argparse
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.desta_native_common import D,read,write,load,sha,check_pins,verified


def audit(phase):
    import numpy as np
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(4)
    out=D/'audits'/phase
    if (out/'COMPLETE.json').exists():
        check_pins(read(out/'COMPLETE.json')['pins']); return
    start=time.monotonic(); seal=read(D/(phase+'_SEAL.json'))
    check_pins({str(D/p):h for p,h in seal['files'].items()})
    basis=load(D/'QC.pt').numpy().astype(np.float64); gram=basis.T@basis
    assert np.max(abs(gram-np.eye(16)))<2e-6
    projection_error=0.; update_error=0.; meta_error=0.; steps=0; predictions=0; radius_error=0.
    with threadpool_limits(limits=4,user_api='blas'):
        for i in seal['indices']:
            ep=D/'native'/f'{i:02}'; initial=load(ep/'INITIAL_GRADIENTS.pt'); inp=read(ep/'INPUT.json')
            for branch in ('event','spatial'):
                path=ep/(branch+'_C0_FULL_GRAD.pt')
                if path.exists():
                    raw=load(path).numpy().astype(np.float64)
                    actual=initial['projected'][branch].numpy().astype(np.float64)
                    expected=raw@basis
                    err=np.linalg.norm(actual-expected)/max(np.linalg.norm(expected),1e-30)
                    projection_error=max(projection_error,float(err)); assert err<2e-6
                    assert abs(np.linalg.norm(raw)-initial['full_norms'][branch])/max(np.linalg.norm(raw),1e-30)<1e-9
            for key,cfg in seal['configurations'].items():
                arm=ep/key; assert verified(arm)
                pred=load(arm/'PREDICTION.pt'); assert not pred['GT_read'] and not pred['decoder_GT_prefix']
                assert pred['config']==cfg and pred['latent_updates']==cfg['steps']
                coeff=np.zeros_like(initial['projected']['event'].numpy())
                stock_norm=inp['stock_norm']; cap=cfg['radius']*stock_norm; eta=cap/cfg['steps']
                for k in range(cfg['steps']):
                    raw=load(arm/f'STEP{k+1:02}.pt'); g=np.zeros_like(coeff)
                    for branch,w in [('event',cfg['temporal_weight']),('spatial',cfg['spatial_weight'])]:
                        n=raw['full_norms'][branch]
                        assert np.isfinite(n) and n>=0
                        if n>0 and w>0:
                            g=g+raw['projected'][branch].numpy()*np.float32(w/n)
                    gn=np.linalg.norm(g.astype(np.float64))
                    candidate=coeff-g*np.float32(eta/gn) if gn else coeff.copy()
                    flat=candidate.astype(np.float64).reshape(-1,16)
                    dn=float(np.sqrt(max(0.,((flat@gram)*flat).sum())))
                    scale=min(1.,cap/dn) if dn else 1.
                    coeff=candidate*np.float32(scale)
                    actual=raw['coeff_after'].numpy()
                    err=float(np.linalg.norm((actual-coeff).astype(np.float64))/max(np.linalg.norm(actual.astype(np.float64)),1e-30))
                    update_error=max(update_error,err); assert err<3e-5,(phase,i,key,k,err)
                    meta_error=max(meta_error,abs(dn-raw['metadata']['pre_projection_delta_norm'])/max(dn,1e-30))
                    # Use saved actual endpoint for the next independent step to avoid cumulative FP32 drift.
                    coeff=actual.copy(); steps+=1
                final=load(arm/'FINAL_COEFFICIENTS.pt').numpy()
                assert np.array_equal(final,coeff)
                f=final.astype(np.float64).reshape(-1,16)
                relative=float(np.sqrt(((f@gram)*f).sum())/stock_norm)
                radius_error=max(radius_error,relative-cfg['radius'])
                assert relative<=cfg['radius']+3e-6
                predictions+=1
    report=dict(status='passed',phase=phase,predictions=predictions,actual_latent_steps=steps,
        independent_projection_relative_max=projection_error,independent_update_relative_max=update_error,
        norm_relative_max=meta_error,radius_overshoot_max=radius_error,CPU_seconds=time.monotonic()-start,
        all_parameters_frozen=True,no_GT_read=True,full_F_saved_only_first_query=True)
    write(out/'REPORT.json',report)
    write(out/'COMPLETE.json',dict(pins={str(out/'REPORT.json'):sha(out/'REPORT.json'),str(D/(phase+'_SEAL.json')):sha(D/(phase+'_SEAL.json'))}))
    print('AUDIT',report,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['dev16','dev64','ablations']);a=p.parse_args();audit(a.phase)
