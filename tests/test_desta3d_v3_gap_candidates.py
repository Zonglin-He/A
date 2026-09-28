import torch,pytest
from vg_tta.desta3d_v3_gap_candidates import *
from vg_tta.desta3d_v3_joint_mixer import JointCorrectionMixer
from vg_tta.desta3d_v3_oracle_mixer_gap import analytic_joint

def inputs():
    torch.manual_seed(17);basis=torch.linalg.qr(torch.randn(8,4).double())[0].float()
    args=(torch.randn(1,3,2,2,2),torch.randn(1,2),torch.randn(1,2),torch.randn(1,3,2,2,8),torch.randn(1,3,2,2,8))
    return basis,args,torch.randn(1,3,STATE_DIM)

def test_state_full_vocab_selected_token_and_unknown_evidence():
    logits=torch.zeros(1,4,13);logits[...,12]=3;ids=torch.full((1,4),2)
    kw=dict(time_logits=torch.tensor([[2.,0.,-1.],[0.,0.,2.]]),interval=[0,2],positions=[1],
        boxes_xyxy=torch.tensor([[.1,.2,.7,.8]]),geometry_valid=torch.ones(1,dtype=torch.bool),coordinate_logits=logits,
        coordinate_token_ids=ids,evidence_boxes=torch.ones(3,4),evidence_known=torch.zeros(3,dtype=torch.bool),frames=3)
    state=compressed_native_state(**kw);assert state.shape==(3,33) and not state.requires_grad
    assert torch.all(state[1,18:22]<0) and torch.all(state[1,14:18]<.1)
    assert state[1,26]==1 and state[0,26]==0 and not state[:,28:32].any()
    with pytest.raises(ValueError):compressed_native_state(**{**kw,'coordinate_logits':None})

def test_A_fixed_radius_and_frozen_input_scope():
    basis,args,state=inputs();args=tuple(x.requires_grad_() for x in args);state.requires_grad_()
    m=StateAwareDirectionMixer(basis,hidden=2);d,a=m(*args[:4],state,args[-1]);target=torch.randn_like(a)
    loss=coefficient_direction_loss(a,target)
    assert torch.allclose(loss['coefficient'],2*loss['cosine'],atol=1e-6)
    loss['cosine'].backward();assert m.output.weight.grad is not None
    assert all(x.grad is None for x in (*args,state)) and m.basis.grad is None
    assert abs(float(d.norm()/args[-1].norm())-m.radius)<2e-6
    zero=coefficient_direction_loss(a,torch.zeros_like(a));assert not zero['valid'].any() and zero['cosine']==0

def test_B_frozen_direction_noop_and_trainable_initial_gate():
    basis,args,state=inputs();old=JointCorrectionMixer(basis,hidden=2)
    with torch.no_grad():old.output.bias.fill_(1.)
    before={k:v.clone() for k,v in old.state_dict().items()};gate=TrustGatedCorrection(old).train()
    d,alpha=gate(args,state,args[3]);d.sum().backward()
    assert all(p.grad is None and not p.requires_grad for p in old.parameters())
    assert gate.gate[-1].bias.grad is not None and gate.gate[-1].bias.grad.abs().sum()>0
    assert not old.training and all(torch.equal(v,before[k]) for k,v in old.state_dict().items())
    assert float(alpha)<.02*gate.radius
    with torch.no_grad():gate.gate[-1].bias.fill_(-.1)
    d,alpha=gate(args,state,args[3]);assert torch.count_nonzero(d)==0 and alpha==0

def test_C_first_step_matches_oracle_fixed_iterations_and_only_span_changes():
    basis,args,state=inputs();stock=args[-1];torch.manual_seed(19)
    targets={b:torch.randn_like(stock) for b in ('event','spatial')};calls=[]
    def objective(f,b):calls.append(b);return .5*(f-targets[b]).square().sum()
    g={b:stock-targets[b] for b in targets};ref=analytic_joint(g['event'],g['spatial'],basis,stock,.13545580427763146)
    one,h=rescue_ladder(objective,stock,basis,mode='union',steps=1)
    assert torch.allclose(one,ref,atol=2e-7,rtol=2e-6) and len(calls)==2
    for mode in ('union','free'):
        calls.clear();delta,history=rescue_ladder(objective,stock,basis,mode=mode)
        assert len(history)==20 and len(calls)==40 and history[-1]['step']==20
        assert float(delta.norm())<=.13545580427763146*float(stock.norm())+2e-6
        if mode=='union':assert torch.allclose(delta,delta@basis@basis.T,atol=2e-6)

def test_C_missing_support_preserves_noop_without_invented_loss():
    basis,args,state=inputs();delta,h=rescue_ladder(lambda f,b:None,args[-1],basis,mode='union')
    assert not delta.any() and len(h)==20 and all(x['loss_before']==dict(event=None,spatial=None) for x in h)
