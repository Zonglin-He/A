"""Synthetic CPU tests, no real cache/model/RGB/GT access."""
import pytest
import torch
from scripts.propagation_v0.test_bank import fixture
from scripts.box_propagation.bank import CandidateBank, fit_axis_affine, transform_native_box, extent_kernel


def test_exact_axis_affine_recovers_complete_native_extent():
    source=torch.tensor([[.1,.2],[.4,.7],[.8,.3],[.6,.9]],dtype=torch.float64)
    scale=torch.tensor([1.5,.75],dtype=torch.float64);delta=torch.tensor([.04,.1],dtype=torch.float64)
    target=source*scale+delta
    fit,info=fit_axis_affine(torch.eye(4,dtype=torch.float64),source,target)
    assert info['valid']
    torch.testing.assert_close(fit[0],scale,atol=1e-15,rtol=1e-15)
    torch.testing.assert_close(fit[1],delta,atol=1e-15,rtol=1e-15)
    native=torch.tensor([.4,.5,.6,.8],dtype=torch.float64)
    actual=transform_native_box(native,*fit)
    torch.testing.assert_close(actual,torch.cat((scale*native[:2]+delta,scale*native[2:])),atol=1e-15,rtol=1e-15)
    # Range is not cropped to visible image or any mask extent.
    assert actual[2]==pytest.approx(.9) and actual[3]==pytest.approx(.6)
    scaled,small_info=fit_axis_affine(torch.eye(4,dtype=torch.float64)*1e-8,source,target)
    assert small_info['valid'] and small_info['source_ESS']==pytest.approx(4.)
    torch.testing.assert_close(scaled[0],scale,atol=1e-15,rtol=1e-15)


def test_native_identity_exact_even_empty_masks_and_outside_event():
    evidence,cache=fixture();evidence.seeds.zero_();evidence.event[8:]=False;cache['interval']=[0,8]
    bank=CandidateBank(evidence,cache,{})
    for j,node in enumerate(cache['planned_positions']):
        assert bank.valid[node,1] and torch.equal(bank.boxes[node,1],cache['teacher']['boxes'][j])
        assert not bank.valid[node,2]
        assert bank.confidence[node]==1
    assert not bank.active[8:].any()
    assert bank.valid[8,1] and bank.valid[10,1]  # stored native, ineligible action
    with pytest.raises(ValueError):bank.score_path(torch.ones(11,dtype=torch.long))


def test_zero_mask_unobserved_fallback_only_E_and_finite_scores():
    evidence,cache=fixture();evidence.seeds.zero_();bank=CandidateBank(evidence,cache,{})
    unobserved=~bank.observed
    assert not bank.active[unobserved].any() and not bank.valid[unobserved,1:].any()
    assert torch.equal(bank.boxes[:,0],cache['E'])
    assert (bank.association_scores==0).all()
    assert torch.isfinite(bank.score_path(torch.zeros(11,dtype=torch.long)))
    assert bank.metadata['model_calls']==bank.metadata['decoder_calls']==0


def test_WLS_rejects_insufficient_ESS_variance_scale_and_offimage():
    x=torch.tensor([[.1,.2],[.4,.7],[.8,.3],[.6,.9]],dtype=torch.float64)
    joint=torch.eye(4,dtype=torch.float64);joint[1:]=0
    assert fit_axis_affine(joint,x,x)[0] is None
    thin=x.clone();thin[:,0]=.5
    assert fit_axis_affine(torch.eye(4,dtype=torch.float64),thin,thin)[0] is None
    assert fit_axis_affine(torch.eye(4,dtype=torch.float64),x,x*3)[0] is None
    assert transform_native_box(torch.tensor([.5,.5,.1,.1]),torch.ones(2),torch.tensor([2.,0.])) is None


def test_fullbox_kernel_responds_to_extent_and_complete_state_roundtrip(monkeypatch):
    ref=torch.tensor([.5,.5,.6,.8],dtype=torch.float64)
    assert extent_kernel(ref,ref)==1
    small=ref.clone();small[2:]/=2
    shifted=ref.clone();shifted[0]+=.2
    assert extent_kernel(small,ref)<1 and extent_kernel(shifted,ref)<1
    evidence,cache=fixture();bank=CandidateBank(evidence,cache,{})
    state=bank.private_state()
    def forbidden(*args,**kwargs):raise AssertionError('from_state cannot fit/repropagate')
    monkeypatch.setattr('scripts.box_propagation.bank.fit_axis_affine',forbidden)
    restored=CandidateBank.from_state(state)
    for name in ('boxes','valid','projection','guard','link','extent_scores','association_scores','side_reference_boxes','score_unary','score_pair'):
        assert torch.equal(getattr(restored,name),getattr(bank,name))
    a=torch.zeros(11,dtype=torch.long)
    assert restored.score_path(a)==bank.score_path(a)
