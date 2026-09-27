import copy
import pytest
import torch
from vg_tta.desta3d_v2 import Desta3DAdapterV2, asymmetric_event_referent_loss
from vg_tta.desta3d_v2_tta_objective import (
    CalibrationWeights,calibration_objective,configure_pre_gate_calibration,gradient_groups)


def setup():
    torch.manual_seed(7)
    m=Desta3DAdapterV2(in_channels=12,query_dim=10,hidden_dim=8).eval()
    x=torch.randn(1,4,3,3,12);q=torch.randn(1,5,10);mask=torch.ones(1,5,dtype=torch.bool)
    times=torch.tensor([[0.,.4,1.,2.]])
    return m,x,q,mask,times


def test_actual_pre_gate_loss_declared_groups_and_real_parameter_step():
    m,x,q,mask,times=setup();info=configure_pre_gate_calibration(m)
    teacher=copy.deepcopy(m).requires_grad_(False)
    with torch.no_grad():reference=teacher(x+.15*torch.randn_like(x),q,mask,frame_times=times)
    before=m(x,q,mask,frame_times=times)
    loss,terms=calibration_objective(m,before,reference,info['initial_parameters'])
    assert terms['parameter_anchor']==0
    loss.backward();groups=gradient_groups(m)
    assert groups['gates']['trainable_parameters']==0 and groups['gates']['nonzero_tensors']==0
    assert groups['branch_film']['gradient_l2']>0 and groups['norm_affine']['gradient_l2']>0
    gates=[p.detach().clone() for p in m.parameter_groups()['gates']]
    opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=1e-3,weight_decay=0)
    opt.step()
    after=m(x,q,mask,frame_times=times)
    # These are auxiliary logits and residual tokens, not PTD final logits.
    assert not torch.equal(before['event_logits'],after['event_logits'])
    assert not torch.equal(before['updated_tokens_spatial'],after['updated_tokens_spatial'])
    assert all(torch.equal(x,y) for x,y in zip(gates,m.parameter_groups()['gates']))
    assert all(p.grad is None for p in teacher.parameters())


def test_alignment_requires_new_query_conditioned_moments():
    m,x,q,mask,times=setup();info=configure_pre_gate_calibration(m);a=m(x,q,mask,frame_times=times)
    with pytest.raises(ValueError):
        calibration_objective(m,a,a,info['initial_parameters'],weights=CalibrationWeights(alignment=.01))
    moments={'feature_definition':'v2_query_conditioned_readers'}
    for b in ['spatial','event']:
        f=a['branch_features_'+b].detach().reshape(-1,8)
        moments[b]={'mean':f.mean(0),'std':f.var(0,unbiased=False).clamp_min(1e-12).sqrt()}
    _,parts=calibration_objective(m,a,a,info['initial_parameters'],weights=CalibrationWeights(alignment=.01),source_moments=moments)
    assert parts['alignment']==0


def test_full_width_declared_count_excludes_two_disconnected_gates():
    m=Desta3DAdapterV2(hidden_dim=128)
    info=configure_pre_gate_calibration(m)
    assert info['parameter_count']==66816


def test_pre_gate_loss_with_original_scope_reveals_none_gate_gradients():
    m,x,q,mask,times=setup();m.set_train_stage('tta')
    initial={n:p.detach().clone() for n,p in m.named_parameters() if p.requires_grad}
    before=m(x,q,mask,frame_times=times)
    teacher=m(x+.2,q,mask,frame_times=times)
    loss,_=calibration_objective(m,before,teacher,initial,weights=CalibrationWeights(parameter_anchor=0.))
    loss.backward()
    # Zero-coefficient anchor may create explicit zero gradients. Both mean no correction.
    assert all(p.grad is None or not p.grad.count_nonzero() for p in m.parameter_groups()['gates'])
