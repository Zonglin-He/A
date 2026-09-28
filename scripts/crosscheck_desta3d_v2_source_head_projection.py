"""Independent NumPy full-vocabulary dot/LSE and integer BF16 rounding readback."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','4');os.environ.setdefault('OMP_NUM_THREADS','4')
import sys,time,resource,traceback
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.desta3d_v2_p0 import read,sha
from scripts.score_desta3d_v2_aux_recovery import save_once
from scripts.desta3d_tta_run_v1 import tensor_sha256
OUT=ROOT/'artifacts/desta3d_v2/tta_v2/source_head_projection_v1'


def lse(x):
    m=np.max(x,axis=1);return m+np.log(np.sum(np.exp(x-m[:,None]),axis=1,dtype=np.float64))


def round_bf16(x):
    a=x.astype(np.float32);u=a.view(np.uint32)
    rounded=((u+np.uint32(0x7fff)+((u>>16)&1))>>16).astype(np.uint16)
    return (rounded.astype(np.uint32)<<16).view(np.float32).astype(np.float64)


def check():
    from safetensors import safe_open
    torch.set_num_threads(4)
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=4)
    cfg=read(OUT/'CONFIG.json');head=read(OUT/'HEAD_IDENTITY.json');start=time.monotonic()
    assert not (OUT/'ROOT_RAW_READBACK.json').exists()
    for p,h in read(OUT/'GPU_COMPLETE.json')['pins'].items():assert sha(Path(p))==h,p
    reference=torch.load(OUT/'CPU_REFERENCE.pt',map_location='cpu',weights_only=False)
    err=0.;nvalues=0;cases={};raws={}
    try:
        for state in cfg['states']:
            cases[state]={}
            for branch in cfg['branches']:
                r=torch.load(OUT/(state+'_'+branch+'.pt'),map_location='cpu',weights_only=False);raws[state,branch]=r
                assert tensor_sha256(r['hidden'])==r['hidden_sha'] and tensor_sha256(r['logits'])==r['logits_sha']
                H=r['hidden'].float().numpy().astype(np.float64);A=r['logits'].float().numpy().astype(np.float64)
                targets=r['tokens']['targets'].numpy();n=len(H)
                ref_lse=np.full(n,-np.inf);round_lse=ref_lse.copy();actual_lse=ref_lse.copy()
                target=np.empty(n);target_round=np.empty(n);mismatch=0;roundmax=0.;refmax=0.;squared=0.;rows=0
                with safe_open(head['weight_path'],framework='pt',device='cpu') as f:
                    w=f.get_slice(head['weight_key'])
                    for e in head['chunks']:
                        assert time.monotonic()-start<cfg['CPU_phase_seconds']
                        assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<cfg['CPU_max_rss_bytes']
                        mem={l.split(':')[0]:int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines()}
                        assert mem['MemAvailable']>cfg['host_reserve_bytes']
                        first=e['first'];end=first+e['rows'];assert first==rows
                        # Independent integer round of on-disk F16 to BF16, verify actual runtime weight identity.
                        disk=w[first:end];W=round_bf16(disk.float().numpy()).astype(np.float32)
                        assert tensor_sha256(torch.from_numpy(W).bfloat16())==e['sha256']
                        z=H@W.astype(np.float64).T;rz=round_bf16(z);az=A[:,first:end]
                        ref_lse=np.logaddexp(ref_lse,lse(z));round_lse=np.logaddexp(round_lse,lse(rz));actual_lse=np.logaddexp(actual_lse,lse(az))
                        hit=np.flatnonzero((targets>=first)&(targets<end));target[hit]=z[hit,targets[hit]-first];target_round[hit]=rz[hit,targets[hit]-first]
                        mismatch+=int(np.count_nonzero(rz!=az));roundmax=max(roundmax,float(np.abs(rz-az).max()))
                        refmax=max(refmax,float(np.abs(z-az).max()));squared+=float(np.sum((z-az)**2,dtype=np.float64));rows=end
                assert rows==152775
                values={'target_fp64':target,'lse_fp64':ref_lse,'ce_fp64':ref_lse-target,
                        'target_round':target_round,'lse_round':round_lse,'ce_round':round_lse-target_round,
                        'target_actual':A[np.arange(n),targets],'lse_actual_fp64':actual_lse,
                        'ce_actual_fp64':actual_lse-A[np.arange(n),targets]}
                q=reference[state][branch]
                for k,v in values.items():
                    difference=float(np.max(np.abs(v-q['token'][k].numpy())));err=max(err,difference);nvalues+=len(v)
                    assert difference<1e-9,(state,branch,k,difference)
                assert mismatch==q['full_vocab']['round_mismatch_elements']
                assert abs(roundmax-q['full_vocab']['round_maxabs'])<1e-12 and abs(refmax-q['full_vocab']['fp64_vs_actual_maxabs'])<1e-9
                assert abs(squared-q['full_vocab']['fp64_vs_actual_squared_error'])<1e-7
                actual_old=np.concatenate([c['cross_entropy'].numpy() for c in r['tokens']['chunks']])
                ce_diff=float(np.max(np.abs(values['ce_actual_fp64']-actual_old)))
                assert ce_diff<3e-6,ce_diff
                cases[state][branch]={'token_means':{k:float(v.mean()) for k,v in values.items()},'full_vocab_rows':rows,
                    'full_logits_checked':n*rows,'round_mismatch_elements':mismatch,'round_maxabs':roundmax,
                    'actual_FP64_CE_vs_original_FP32_maxabs':ce_diff}
                print('ROOT_PROJECTION',state,branch,'max_error',err,'round_mismatches',mismatch,flush=True)
        changes={}
        for branch in cfg['branches']:
            a,b=raws['B1',branch],raws['saved_supervised3',branch]
            h0=a['hidden'].float().numpy().astype(np.float64);dh=b['hidden'].float().numpy().astype(np.float64)-h0
            q0,q1=reference['B1'][branch],reference['saved_supervised3'][branch]
            changes[branch]={'hidden_shape':list(h0.shape),'hidden_changed_elements':int(np.count_nonzero(dh)),
                'hidden_elements':dh.size,'hidden_delta_L2':float(np.linalg.norm(dh)),
                'hidden_delta_relative_L2':float(np.linalg.norm(dh)/np.linalg.norm(h0)),
                'hidden_delta_maxabs':float(np.max(np.abs(dh))),
                'actual_original_CE_delta':b['info']['ce']-a['info']['ce'],
                'tokens':[]}
            for k in ['ce_fp64','ce_round','ce_actual_fp64']:
                delta=q1['token'][k].numpy()-q0['token'][k].numpy()
                changes[branch][k+'_delta']=float(delta.mean())
            for i in range(len(h0)):
                changes[branch]['tokens'].append({'position':int(a['tokens']['positions'][i]),'target':int(a['tokens']['targets'][i]),
                    'ntp':bool(a['tokens']['ntp'][i]),**{k+'_delta':float(q1['token'][k][i]-q0['token'][k][i]) for k in q0['token']}})
        save_once(OUT/'ROOT_RAW_READBACK.json',{'status':'passed','implementation':'independent NumPy FP64 matmul/stable full-vocabulary LSE/integer BF16 rounding',
            'full_vocabulary':152775,'token_rows':162,'scalar_values_crosschecked':nvalues,'max_token_scalar_error':err,
            'cases':cases,'changes':changes,'CPU_seconds':time.monotonic()-start,
            'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'no_model_or_GT_calls':True})
    except BaseException:
        save_once(OUT/'ROOT_FAILURE.json',{'failure':traceback.format_exc(),'CPU_seconds':time.monotonic()-start});raise

if __name__=='__main__':check()
