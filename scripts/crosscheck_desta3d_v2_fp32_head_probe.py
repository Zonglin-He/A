"""Independent complete-vocabulary NumPy projection and raw scope readback."""
import os
os.environ['OPENBLAS_NUM_THREADS']='4';os.environ['OMP_NUM_THREADS']='4'
import time,sys,resource,traceback
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.desta3d_v2_p0 import read,sha
from scripts.desta3d_tta_run_v1 import tensor_sha256
from scripts.score_desta3d_v2_aux_recovery import save_once
OUT=ROOT/'artifacts/desta3d_v2/tta_v2/source_fp32_head_probe_v1';OLD=ROOT/'artifacts/desta3d_v2/tta_v2/source_head_projection_v1'

def run():
    from safetensors import safe_open
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=4);torch.set_num_threads(4);start=time.monotonic()
    cfg=read(OUT/'CONFIG.json');head=read(OLD/'HEAD_IDENTITY.json');report=read(OUT/'REPORT.json')
    for p,h in read(OUT/'COMPLETE.json')['pins'].items():assert sha(Path(p))==h,p
    assert not (OUT/'ROOT_RAW_READBACK.json').exists()
    result={};norm_error=0.;largest=0.
    try:
        for state in cfg['states']:
            result[state]={}
            for branch in cfg['branches']:
                r=torch.load(OUT/f'{state}_{branch}_FORWARD.pt',map_location='cpu',weights_only=False)
                g=torch.load(OUT/f'{state}_{branch}_GRADIENT.pt',map_location='cpu',weights_only=False)
                old=torch.load(OLD/f'{state}_{branch}.pt',map_location='cpu',weights_only=False)
                assert torch.equal(r['hidden'],old['hidden']) and r['adapter_sha']==old['adapter_sha']
                for k in ['targets','positions','ntp']:assert torch.equal(r['tokens'][k],old['tokens'][k])
                assert tensor_sha256(r['logits'])==r['logits_sha'] and tensor_sha256(g['gradient'])==g['gradient_sha']
                H=r['hidden'].float().numpy().astype(np.float64);A=r['logits'].numpy().astype(np.float64);t=r['tokens']['targets'].numpy();n=len(H)
                assert np.isfinite(A).all();lz=np.full(n,-np.inf);la=lz.copy();target=np.empty(n);maxerr=0.;rows=0
                with safe_open(head['weight_path'],framework='pt',device='cpu') as f:
                    w=f.get_slice(head['weight_key'])
                    for e in head['chunks']:
                        assert time.monotonic()-start<600 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<4*2**30
                        i=e['first'];j=i+e['rows'];assert i==rows
                        weight=w[i:j].bfloat16();assert tensor_sha256(weight)==e['sha256']
                        z=H@weight.float().numpy().astype(np.float64).T;a=A[:,i:j]
                        for value,which in [(z,'ref'),(a,'actual')]:
                            m=value.max(axis=1);v=m+np.log(np.exp(value-m[:,None]).sum(axis=1,dtype=np.float64))
                            if which=='ref':lz=np.logaddexp(lz,v)
                            else:la=np.logaddexp(la,v)
                        hit=np.flatnonzero((t>=i)&(t<j));target[hit]=z[hit,t[hit]-i]
                        maxerr=max(maxerr,float(np.abs(z-a).max()));rows=j
                assert rows==152775 and maxerr<=cfg['reference_abs_tol']
                refce=lz-target;ce=la-A[np.arange(n),t]
                captured=np.concatenate([x['cross_entropy'].numpy() for x in r['tokens']['chunks']])
                ce_error=float(np.max(np.abs(ce-captured)));assert ce_error<3e-6
                assert abs(float(ce.mean())-r['ce'])<3e-6
                err=abs(float(np.linalg.norm(g['gradient'].numpy().astype(np.float64)))-g['norm']);norm_error=max(norm_error,err)
                assert err<1e-10 and g['scope']==66816 and g['gradient'].numel()==66816 and np.isfinite(g['gradient'].numpy()).all()
                largest=max(largest,maxerr)
                result[state][branch]={'full_vocab_elements':A.size,'full_projection_maxabs_vs_FP64':maxerr,
                    'reference_CE':float(refce.mean()),'actual_FP32_CE_FP64_readback':float(ce.mean()),'worker_CE':r['ce'],
                    'actual_per_token_CE_vs_worker_maxabs':ce_error,'gradient_norm':g['norm'],'gradient_norm_numpy_error':err,
                    'original_hidden_and_support_exact':True,'reference_token_CE':refce.tolist(),'actual_token_CE':ce.tolist()}
                print('ROOT_FP32',state,branch,'max_projection_error',maxerr,'CE',float(ce.mean()),flush=True)
        save_once(OUT/'ROOT_RAW_READBACK.json',{'status':'passed','cases':result,'raw_gradient_norm_maxerr':norm_error,
            'all_vocab_projection_maxabs':largest,'token_rows':162,'full_logits':162*152775,'CPU_seconds':time.monotonic()-start,
            'no_model_or_GT_calls':True,'scope_checks_in_worker':report['all_scope_checks']})
    except BaseException:
        save_once(OUT/'ROOT_FAILURE.json',{'error':traceback.format_exc(),'CPU_seconds':time.monotonic()-start});raise

if __name__=='__main__':run()
