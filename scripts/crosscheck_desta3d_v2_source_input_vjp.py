"""Independent CPU NumPy contraction of complete input gradients and sealed deltas."""
from pathlib import Path
import sys,time,json
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.desta3d_v2_p0 import read,sha
from scripts.score_desta3d_v2_aux_recovery import save_once
from scripts.desta3d_tta_run_v1 import tensor_sha256
BASE=ROOT/'artifacts/desta3d_v2/tta_v2'
OUT=BASE/'source_input_vjp_v1';OLD=BASE/'source_cast_probe_v1';TASK=BASE/'source_task_control_v2'


def main():
    assert not (OUT/'ROOT_RAW_READBACK.json').exists()
    for manifest in ['LOCK.json','COMPLETE.json']:
        for p,h in read(OUT/manifest)['pins'].items():assert sha(Path(p))==h,p
    prior=torch.load(OLD/'RAW_ENDPOINTS.pt',map_location='cpu',weights_only=False)
    finite=read(OLD/'ROOT_RAW_READBACK.json')
    step=torch.load(TASK/'episodes/00/supervised/step1.pt',map_location='cpu',weights_only=False)
    cfg=read(TASK/'CONFIG.json')
    initial=torch.load(cfg['checkpoint']['checkpoint'],map_location='cpu',weights_only=False)['adapter']
    final=torch.load(TASK/'episodes/00/supervised/FINAL_CALIBRATION.pt',map_location='cpu',weights_only=False)
    delta_theta=np.concatenate([(final[n].numpy()-initial[n].numpy()).reshape(-1).astype(np.float64) for n in step['ordered_names']])
    result={};maxerr=0.;token_count=0;storage=0
    for branch in ['event','spatial']:
        raw=torch.load(OUT/(branch+'_VJP.pt'),map_location='cpu',weights_only=False)
        forward=torch.load(OUT/(branch+'_FORWARD.pt'),map_location='cpu',weights_only=False)
        baseline=prior['states']['B1']['branches'][branch]
        assert forward['info']==baseline['info']
        assert raw['postcast_sha']==forward['postcast_sha']==baseline['postcast_sha']
        assert forward['precast_sha']==baseline['precast_sha']
        assert raw['adapter_sha']==prior['states']['B1']['adapter_sha']
        assert raw['support_sha']==forward['support_sha']==prior['task_support_sha']
        for k in ['positions','targets','ntp']:assert torch.equal(forward['tokens'][k],baseline['tokens'][k])
        for new,old in zip(forward['tokens']['chunks'],baseline['tokens']['chunks']):
            assert new['chunk_size']==old['chunk_size']
            for k in ['target_logit','logsumexp','cross_entropy']:assert torch.equal(new[k],old[k])
        token_count+=len(forward['tokens']['targets'])
        assert raw['gradient'].dtype==torch.bfloat16
        assert raw['dtype']=='torch.bfloat16' and tensor_sha256(raw['gradient'])==raw['gradient_sha']
        g=raw['gradient'].float().numpy().reshape(-1).astype(np.float64)
        dp=prior['differences'][branch]['precast_delta'].numpy().reshape(-1).astype(np.float64)
        s=prior['differences'][branch]['sparse_postcast'];idx=s['indices'].numpy().astype(np.int64)
        db=s['after'].float().numpy().astype(np.float64)-s['before'].float().numpy().astype(np.float64)
        assert len(g)==len(dp)==9175040 and np.isfinite(g).all()
        norm=float(np.sqrt(np.sum(g*g)));pre=float(np.sum(g*dp));post=float(np.sum(g[idx]*db))
        for k,v in [('gradient_L2',norm),('dot_precast_delta',pre),('dot_postcast_delta',post)]:
            error=abs(v-raw['summary'][k]);maxerr=max(maxerr,error);assert error<1e-12
        assert int(np.count_nonzero(g))==raw['summary']['gradient_nonzero']
        pt=float(np.dot(step['raw']['task_'+branch].numpy().astype(np.float64),delta_theta))
        change=finite['CE'][branch]['CE_change']
        result[branch]={'gradient_L2':norm,'elements':len(g),'dtype':str(raw['gradient'].dtype),
                       'dot_precast_delta':pre,'dot_postcast_delta':post,'parameter_gradient_dot_delta':pt,
                       'measured_finite_CE_change':change,
                       'precast_vs_parameter_relative_difference':abs(pre-pt)/abs(pt),
                       'postcast_prediction_has_same_sign_as_finite':bool(post*change>0),
                       'finite_over_postcast_prediction_signed_ratio':change/post,
                       'finite_minus_precast_prediction':change-pre,'finite_minus_postcast_prediction':change-post,
                       'full_forward_token_and_injection_equality':True}
        storage+=(OUT/(branch+'_VJP.pt')).stat().st_size
    total={k:sum(v[k] for v in result.values()) for k in ['dot_precast_delta','dot_postcast_delta','parameter_gradient_dot_delta','measured_finite_CE_change']}
    # Branch gradients were added in FP32 in the older total; retain both aggregation definitions.
    total['old_FP32_summed_parameter_gradient_dot_delta']=finite['initial_gradient_dot_actual_three_step_parameter_delta']
    total['precast_vs_old_parameter_relative_difference']=abs(total['dot_precast_delta']-total['old_FP32_summed_parameter_gradient_dot_delta'])/abs(total['old_FP32_summed_parameter_gradient_dot_delta'])
    total['finite_over_postcast_prediction_signed_ratio']=total['measured_finite_CE_change']/total['dot_postcast_delta']
    report={'time':time.time(),'status':'passed','branches':result,'totals':total,'forward_tokens_compared':token_count,
            'max_norm_dot_absolute_error':maxerr,'full_gradient_file_bytes':storage,'optimizer_steps':0,
            'source_queries':1,'source_parents':1,'target_inputs_or_GT':False,'new_native_predictions':0,
            'checker_sha':sha(Path(__file__)),'scope':'actual BF16 input autograd gradients, existing exact endpoint differences; no precision intervention',
            'interpretation':'postcast local prediction still misses finite CE, including spatial sign; conversion-only first-order explanation not established. Full low-precision autograd is a local diagnostic, not a mathematical derivative of a continuous real-valued forward.'}
    save_once(OUT/'ROOT_RAW_READBACK.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
