"""Independent bounded CPU valid/rejection contracts; no model or GT access."""
import copy,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_opd_p2_action_reward005 import verify
from scripts.stvg_opd_paper_hc2_revision_common_v2 import BASE,write,sha
from scripts.decota_matrix_common_v1 import load
REC=BASE/'recovery/P2_matched_gradient_precision_006'


def run():
    verify()
    import numpy as np
    import torch
    from vg_tta.decota_spatial_opd_matched_gradient_precision006 import check
    from vg_tta.decota_spatial_opd_inline_gradient_precision004 import Installation
    torch.set_num_threads(2);rng=np.random.default_rng(4004);valid=[];cases=[]
    for n in [1,2,3,4,16]:
        for sigma in [.025,.1]:
            for shifted in [False,True]:
                mean=rng.normal(0,1,(n,4)).astype(np.float32)
                current=(mean+np.float32(.03)*rng.normal(0,1,(n,4))).astype(np.float32) if shifted else mean.copy()
                samples=(mean[:,None,:]+np.float32(sigma)*rng.normal(0,1,(n,32,4))).astype(np.float32)
                weights=torch.softmax(torch.tensor(rng.normal(0,1,(n,32)),dtype=torch.float32),-1).numpy()
                x=torch.tensor(current,requires_grad=True);m=torch.tensor(mean);s=torch.tensor(samples);w=torch.tensor(weights)
                inv=torch.tensor(np.float32(1)/np.float32(2*sigma*sigma));invn=torch.tensor(np.float32(1)/np.float32(n))
                loss=((x-m).square().sum(-1)*inv-((w-1/32)*(-(s-x[:,None,:]).square().sum(-1)*inv)).sum(-1)).sum()*invn
                g=torch.autograd.grad(loss,x)[0].detach().numpy()
                # Independently evaluated float32 expression used by the original guard.
                invv=np.float32(1)/np.float32(sigma*sigma)
                analytic=(((current-mean)*invv-((weights-1/32)[:,:,None]*(samples-current[:,None,:])).sum(1,dtype=np.float32)*invv)*np.float32(1/n)).astype(np.float32)
                args=[current,mean,samples,weights,g,analytic,sigma,float(abs(g-analytic).max())]
                checked=check(*args);assert checked['status']=='pass';cases.append(args)
                valid.append(dict(n=n,sigma=sigma,frozen_rollout_shifted=shifted,pass_=True))
    base=cases[0];bad=[]
    def rejects(name,mutate):
        args=copy.deepcopy(base);mutate(args)
        try:check(*args)
        except (AssertionError,ValueError,IndexError):bad.append(dict(name=name,rejected=True))
        else:raise AssertionError('Contract did not reject '+name)
    rejects('wrong_GPU_gradient',lambda a:a[4].__setitem__((0,0),a[4][0,0]+np.float32(.05)))
    rejects('wrong_GPU_gradient_and_forged_inline_error',lambda a:(a[4].__setitem__((0,0),a[4][0,0]+np.float32(.05)),a.__setitem__(7,float(abs(a[4]-a[5]).max()))))
    rejects('wrong_analytic_branch',lambda a:(a[5].__setitem__((0,0),a[5][0,0]+np.float32(.05)),a.__setitem__(7,float(abs(a[4]-a[5]).max()))))
    rejects('rewritten_original_error',lambda a:a.__setitem__(7,a[7]+1e-8))
    rejects('missing_sample',lambda a:a.__setitem__(2,a[2][:,:31,:]))
    rejects('wrong_weight_shape',lambda a:a.__setitem__(3,a[3][:,:31]))
    rejects('wrong_mean_shape',lambda a:a.__setitem__(1,a[1][:,:3]))
    rejects('wrong_gradient_shape',lambda a:a.__setitem__(4,a[4][:,:3]))
    rejects('negative_weight',lambda a:a[3].__setitem__((0,0),np.float32(-.1)))
    rejects('unnormalized_weights',lambda a:a.__setitem__(3,a[3]*np.float32(.5)))
    rejects('nonfinite_mean',lambda a:a[1].__setitem__((0,0),np.float32(np.inf)))
    rejects('nonfinite_sample',lambda a:a[2].__setitem__((0,0,0),np.float32(np.nan)))
    rejects('nonfinite_gradient',lambda a:a[4].__setitem__((0,0),np.float32(np.inf)))
    rejects('zero_sigma',lambda a:a.__setitem__(6,0.))
    rejects('mismatched_sigma',lambda a:a.__setitem__(6,.1))
    rejects('non_float32_input',lambda a:a.__setitem__(0,a[0].astype(float)+1e-12))
    rejects('non_float32_gradient',lambda a:a.__setitem__(4,a[4].astype(float)+1e-12))
    witness=load(REC/'REPRODUCED_INCOMPLETE_FIT_WITNESS.pt')
    rr=witness['rollout']
    actual=check(witness['mean'].numpy(),rr['mean'].numpy(),rr['samples'].numpy(),rr['weights'].numpy(),
        witness['autograd_mean_gradient'].numpy(),witness['analytic_mean_gradient'].numpy(),
        witness['config']['sigma'],witness['original_inline_error'])
    assert actual['original004_CPU32_absolute_guard_passed'] is False
    assert actual['original_inline_absolute_guard_passed'] is True
    assert actual['max_matched_CPU32_vs_GPU32_error']==0
    installation=Installation()
    try:assert installation.only_original_assertion_test_supplemented
    finally:installation.uninstall()
    assert len(valid)==20 and len(bad)==17
    write(REC/'CPU_CONTRACTS.json',dict(status='pass',scope='independent matched-gradient valid cases, explicit rejected malformed/wrong derivatives, and actual serialized original reproduction readback',
        valid_contracts=valid,rejection_contracts=bad,valid_count=20,rejected_count=17,
        original_reproduced_guard_failure_retained=True,actual_witness_check=actual,
        AST_only_one_original_assertion_test_supplemented=True,
        witness_sha256=sha(REC/'REPRODUCED_INCOMPLETE_FIT_WITNESS.pt'),
        new_model_calls=0,new_DINO_calls=0,GT_read=False,time=time.time()))
    print('P2_MATCHED006_CPU_CONTRACTS_PASS_20_VALID_17_REJECTED_ORIGINAL_FAILURE_RETAINED',flush=True)


if __name__=='__main__':run()
