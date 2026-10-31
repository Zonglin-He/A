"""R14 A-only real final-linear SGD, unused B readout and exact no-update control.

No model/data/G/label constructor. Caller passes audited source model/caches and
frozen R13 A target. fit_A_only has no B argument. Each fraction is a fresh source
episode; all B source queries remain fixed, without teacher/EMA or backbone fit.
"""
from dataclasses import dataclass
from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'round7'))
from regression_adapter import (SourceCache,PARAMETER_NAMES,episode,layers,raw_boxes,parameter_sha)

ARM_NAMES=('source','direct_additive_Aresidual','direct_size_normalized_Aresidual',
           'parameter_half','kernel_half','parameter_full','kernel_full')

@dataclass(frozen=True)
class TransferConfig:
    fractions:tuple=(.5,1.)
    kernel_coordinate_atol_m:float=2e-4
    fit_coordinate_atol_m:float=5e-5
    strict_FP32:bool=True

DEFAULT_CONFIG=TransferConfig()


def _finite_cache(cache):
    q=cache.query_index
    assert cache.boxes_raw6.shape==(1,256,6) and cache.base_xyz.shape==(1,256,3)
    assert cache.center_features.shape==cache.size_features.shape==(1,288,256)
    assert cache.text_scores.shape==(1,256) and q==int(cache.text_scores.argmax())
    for name in ('base_xyz','center_features','size_features','boxes_raw6','text_scores'):
        t=getattr(cache,name);assert t.dtype==torch.float32 and torch.isfinite(t).all() and not t.requires_grad


def fit_A_only(model,cache_A,target6,alpha):
    """Actual single separate-head L2 SGD step. Must run INSIDE R7 episode.

    No B cache/features/output, LR-from-B, clipping, reassociation or label API.
    Leaves selected head updated for caller's subsequent readout until episode
    exit restores source. Caller records return before engineering assertions.
    """
    assert alpha in (.5,1.)
    _finite_cache(cache_A);q=cache_A.query_index
    a,b=layers(model);params=dict(model.named_parameters())
    assert all(p.requires_grad==(n in PARAMETER_NAMES) for n,p in params.items())
    assert parameter_sha(model)==cache_A.selected_parameter_sha256
    source=cache_A.boxes_raw6.detach();target=torch.as_tensor(target6,device=source.device,dtype=torch.float32).detach().clone()
    assert target.shape==(6,) and torch.isfinite(target).all() and (target[3:]>0).all()
    hc=cache_A.center_features[0,:,q];hs=cache_A.size_features[0,:,q]
    kc=float(hc.square().sum())+1.;ks=float(hs.square().sum())+1.
    before={n:params[n].detach().clone() for n in PARAMETER_NAMES}
    initial=raw_boxes(model,cache_A);assert torch.equal(initial,source)
    optimizer=torch.optim.SGD([{'params':[a.weight,a.bias],'lr':alpha/kc},
                               {'params':[b.weight,b.bias],'lr':alpha/ks}],momentum=0,weight_decay=0)
    optimizer.zero_grad(set_to_none=True)
    current=initial[0,q]
    loss_c=.5*(current[:3]-target[:3]).square().sum()
    loss_s=.5*(current[3:]-target[3:]).square().sum()
    loss=loss_c+loss_s;assert torch.isfinite(loss)
    loss.backward()
    grads={n:float(params[n].grad.norm()) for n in PARAMETER_NAMES}
    assert all(params[n].grad is not None and torch.isfinite(params[n].grad).all() for n in PARAMETER_NAMES)
    nonzero=sum(int(torch.count_nonzero(params[n].grad)>0) for n in PARAMETER_NAMES)
    optimizer.step()
    with torch.no_grad():updated=raw_boxes(model,cache_A).detach().clone()
    delta={n:float((params[n].detach()-before[n]).abs().max()) for n in PARAMETER_NAMES}
    expected=source[0,q]+alpha*(target-source[0,q]);residual=updated[0,q]-expected
    return {'A_selected_after':updated[0,q].detach().clone(),
      'trace':{'alpha':alpha,'actual_optimizer_steps':1,'loss_before':float(loss.detach()),
        'center_loss_before':float(loss_c.detach()),'size_loss_before':float(loss_s.detach()),
        'loss_normalization':'separate .5sum center3 and size3; no division by6',
        'feature_self_kernel_center':kc,'feature_self_kernel_size':ks,
        'learning_rates_from_A_only':{'center':alpha/kc,'size':alpha/ks},
        'parameter_gradient_norm_by_name':grads,'nonzero_gradient_parameter_tensors':nonzero,
        'parameter_delta_max_by_name':delta,'observed_parameter_change':any(v>0 for v in delta.values()),
        'A_selected_before':source[0,q].cpu().tolist(),'A_target6':target.cpu().tolist(),
        'A_selected_after':updated[0,q].cpu().tolist(),
        'A_expected_fraction_target6':expected.cpu().tolist(),
        'A_coordinate_residual6':residual.cpu().tolist(),'A_fit_max_coordinate_error_m':float(residual.abs().max()),
        'no_clip_momentum_weight_decay':True,'B_in_loss_LR_gate':False}}


def run_transfer_episode(model,cache_A,cache_B,target_A6,A_gate,*,config=DEFAULT_CONFIG,
                         full_B_forward_hook=None):
    """Seven selected B arms; true SGD readout plus independent frozen kernel.

    Optional hook(model, cache_B, expected_B_pool) runs once at full fraction,
    AFTER real optimizer.step, inside adapted episode, with read-only B inference.
    Wrapper returns measurements/pass flags before caller decides to fail/seal.
    """
    assert config.fractions==(.5,1.) and config.kernel_coordinate_atol_m==2e-4 and config.fit_coordinate_atol_m==5e-5
    _finite_cache(cache_A);_finite_cache(cache_B)
    assert isinstance(A_gate,bool) and cache_A.selected_parameter_sha256==cache_B.selected_parameter_sha256==parameter_sha(model)
    sourceA=cache_A.boxes_raw6.detach().clone();sourceB=cache_B.boxes_raw6.detach().clone()
    qa=cache_A.query_index;qb=cache_B.query_index
    target=torch.as_tensor(target_A6,device=sourceA.device,dtype=torch.float32).detach().clone()
    assert target.shape==(6,) and torch.isfinite(target).all()
    srcA=sourceA[0,qa];srcB=sourceB[0,qb]
    delta=target-srcA
    invalid_A=bool((srcA[3:]<=0).any() or (target[3:]<=0).any())
    eligible=bool(A_gate and not invalid_A and torch.any(delta!=0))
    selected=srcB[None].repeat(7,1);statuses=['source']+['A_gate_or_zero_residual_source_fallback']*6
    traces={};hook_receipt=None
    norms={}
    if eligible:
        # Direct controls see exactly the same A target, without model mutation.
        selected[1]=srcB+delta
        size_ratio_BA=srcB[3:]/srcA[3:]
        selected[2]=srcB+size_ratio_BA.repeat(2)*delta
        statuses[1:3]=['same_A_additive_residual_no_update','same_A_source_size_normalized_residual_no_update']
        ac=cache_A.center_features[0,:,qa];ass=cache_A.size_features[0,:,qa]
        # Both K_BA and K_AA are frozen-source feature arithmetic; the ratio may
        # exceed1 and is NEVER clipped/reweighted based on B.
        kc=cache_B.center_features[0].transpose(0,1)@ac+1
        ks=cache_B.size_features[0].transpose(0,1)@ass+1
        selfc=ac.square().sum()+1;selfs=ass.square().sum()+1
        norms={'A_center_feature_squared_norm':float(ac.square().sum()),
          'A_size_feature_squared_norm':float(ass.square().sum()),
          'A_self_kernel_center':float(selfc),'A_self_kernel_size':float(selfs),
          'B_selected_cross_kernel_center':float(kc[qb]),'B_selected_cross_kernel_size':float(ks[qb]),
          'B_selected_kernel_ratio_center':float(kc[qb]/selfc),
          'B_selected_kernel_ratio_size':float(ks[qb]/selfs),
          'A_features_nonnegative_after_ReLU':bool((ac>=0).all() and (ass>=0).all()),
          'B_features_nonnegative_after_ReLU':bool((cache_B.center_features>=0).all() and (cache_B.size_features>=0).all()),
          'ratios_clipped_or_used_for_gate':False}
        for alpha,pi,ki in ((.5,3,4),(1.,5,6)):
            with episode(model):
                # The fitter has no B inputs. Read B only after its true SGD.
                fit=fit_A_only(model,cache_A,target,alpha)
                with torch.no_grad():parameter_B=raw_boxes(model,cache_B).detach().clone()
                kernel=sourceB.clone()
                kernel[0,:,:3]+=alpha*(kc/selfc)[:,None]*delta[:3]
                kernel[0,:,3:]+=alpha*(ks/selfs)[:,None]*delta[3:]
                difference=(parameter_B-kernel).abs()
                trace=fit['trace']
                trace.update(B_selected_after=parameter_B[0,qb].cpu().tolist(),
                    independent_kernel_B_selected=kernel[0,qb].cpu().tolist(),
                    B_parameter_kernel_all256_max_abs_error_m=float(difference.max()),
                    B_parameter_kernel_selected_max_abs_error_m=float(difference[0,qb].max()),
                    B_kernel_atol_m=config.kernel_coordinate_atol_m,
                    B_kernel_check_pass=bool(difference.max()<=config.kernel_coordinate_atol_m),
                    A_fit_check_pass=bool(trace['A_fit_max_coordinate_error_m']<=config.fit_coordinate_atol_m),
                    B_all_query_count_changed=int((parameter_B[0]!=sourceB[0]).any(-1).sum()),
                    B_nonselected_query_count_changed=int((parameter_B[0]!=sourceB[0]).any(-1).sum())-int(torch.any(parameter_B[0,qb]!=sourceB[0,qb])),
                    B_raw_nonpositive_size_count=int((parameter_B[0,:,3:]<=0).sum()))
                selected[pi]=parameter_B[0,qb];selected[ki]=kernel[0,qb]
                statuses[pi]='actual_A_only_SGD_'+str(alpha);statuses[ki]='same_information_exact_frozen_feature_kernel_'+str(alpha)
                if alpha==1. and full_B_forward_hook is not None:
                    hook_receipt=full_B_forward_hook(model,cache_B,parameter_B)
                traces[str(alpha)]=trace
            assert parameter_sha(model)==cache_A.selected_parameter_sha256
            with torch.no_grad():
                restoredA=raw_boxes(model,cache_A);restoredB=raw_boxes(model,cache_B)
            traces[str(alpha)]['source_A_head_replay_restored_bitexact']=bool(torch.equal(restoredA,sourceA))
            traces[str(alpha)]['source_B_head_replay_restored_bitexact']=bool(torch.equal(restoredB,sourceB))
    assert parameter_sha(model)==cache_A.selected_parameter_sha256 and torch.isfinite(selected).all()
    return {'arm_names':ARM_NAMES,'B_boxes_raw6':selected.detach().clone(),
      'arm_status':statuses,'q_A':qa,'q_B':qb,'A_gate':A_gate,'A_nonzero_residual':bool(torch.any(delta!=0)),
      'A_invalid_source_or_pseudo_size':invalid_A,'A_update_executed':eligible,
      'A_source6':srcA.cpu().tolist(),'A_target6':target.cpu().tolist(),'A_delta6':delta.cpu().tolist(),
      'feature_kernels':norms,'fraction_traces':traces,'full_B_forward_hook':hook_receipt,
      'B_gate_loss_LR_or_clip_used':False,'B_local_G_constructed':False,
      'B_selected_nonpositive_size_counts_by_arm':[(int((selected[j,3:]<=0).sum())) for j in range(7)],
      'size_normalized_direct_formula':'source_B + repeat2(source_B_size/source_A_size)*(target_A-source_A)',
      'same_target_or_different_instances_verified':False,'known_ego_transform_used':False,
      'restored_source_selected_parameters':True,'source_text_scores_B_immutable':cache_B.text_scores.detach().clone(),
      'engineering_checks_pass':all(t['B_kernel_check_pass'] and t['A_fit_check_pass'] and t['source_A_head_replay_restored_bitexact'] and t['source_B_head_replay_restored_bitexact'] for t in traces.values()),
      'interpretation':'Unused-input last-affine SGD is exactly a source-feature kernel correction in real arithmetic; not a unique parameter-update advantage.'}
