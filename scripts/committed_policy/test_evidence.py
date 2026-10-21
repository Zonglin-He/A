"""Synthetic-only cross-anchor checks; no real cache, labels or model."""
from types import SimpleNamespace
import copy
import torch

from scripts.committed_policy.evidence import build, _assemble


def fixture():
    T, H, W = 9, 4, 4
    raw = torch.zeros(T, H*W, 256, dtype=torch.float32)
    raw[:, torch.arange(H*W), torch.arange(H*W)] = 1.
    positions = list(range(8))
    native = torch.tensor([.5,.5,.5,.5],dtype=torch.float64)[None].repeat(8,1)
    E = torch.tensor([.4,.4,.4,.4],dtype=torch.float64)[None].repeat(T,1)
    boxes = E[:,None].repeat(1,3,1)
    boxes[positions,1] = native
    observed = torch.zeros(T,dtype=torch.bool);observed[positions] = True
    anchors = torch.stack((torch.arange(T).clamp_max(7),torch.arange(T).clamp_max(7)),-1)
    scores = torch.tensor([.1,.9,.8],dtype=torch.float64)[None,:,None].repeat(T,1,2)
    bank = dict(boxes=boxes,E=E,valid=torch.ones(T,3,dtype=torch.bool),
        active=torch.ones(T,dtype=torch.bool),observed=observed,source_anchors=anchors,
        side_reliability=torch.ones(T,2,dtype=torch.float64),extent_scores=scores,
        association_scores=scores.clone(),native_boxes=native,frame_ids=torch.arange(T),
        positions8=positions,GT_read=False)
    evidence = dict(features=raw,valid=torch.ones(T,H*W,dtype=torch.bool),
        seeds=torch.ones(8,H,W,dtype=torch.float64),frame_ids=torch.arange(T),
        positions8=positions,GT_read=False)
    return bank,evidence


def test_identity_evidence_full56_and_original_bank_unchanged():
    bank,ev = fixture()
    before = {k:v.clone() for k,v in bank.items() if torch.is_tensor(v)}
    seeds_before = ev['seeds'].clone()
    out = build(bank,ev)
    assert out['pair_valid'].sum() == 56
    assert out['cross'].diagonal().eq(0).all()
    assert out['anchor_quality'].min() > .999
    torch.testing.assert_close(out['anchor_quality'],(out['cross'].sum(0)+out['cross'].sum(1))/14,atol=0,rtol=0)
    assert torch.equal(out['compatibility'],out['compatibility'].T)
    assert out['compatibility'].diagonal().eq(1).all()
    assert out['commit'][:,1:].all() and not out['commit'][:,0].any()
    assert out['candidate_trust'][:,0].eq(0).all()
    assert out['witness_count'].eq(7).all() and out['source_verified'].all()
    for k,v in before.items():assert torch.equal(bank[k],v),k
    assert torch.equal(ev['seeds'],seeds_before)


def test_no_evidence_gives_zero_external_support_and_no_commit():
    bank,ev = fixture();ev['seeds'].zero_()
    out = build(bank,ev)
    assert not out['pair_valid'].any() and out['cross'].eq(0).all()
    assert out['anchor_quality'].eq(0).all() and out['external_support'].eq(0).all()
    assert out['delta_support'].eq(0).all() and not out['commit'].any()
    assert torch.equal(out['reset_matrices'],torch.eye(9,dtype=torch.float64)[None].repeat(9,1,1))


def test_fixed_denominator_invalid_pairs_and_no_quality_renormalization():
    bank,_ = fixture();cross=torch.zeros(8,8,dtype=torch.float64)
    cross[0,1]=.8;cross[1,0]=.2
    out=_assemble(bank,cross)
    expected=torch.tensor([1/14,1/14,0,0,0,0,0,0],dtype=torch.float64)
    torch.testing.assert_close(out['anchor_quality'],expected,atol=0,rtol=0)
    torch.testing.assert_close(out['external_support'][0],torch.tensor([.1,.9,.8],dtype=torch.float64)/14,atol=2e-17,rtol=0)
    assert out['compatibility'][0,1] == .4
    assert not out['commit'].any()  # q<.1, even though delta is positive.


def test_routes_commit_gate_and_conflict_reset_orientation():
    bank,_=fixture();cross=torch.ones(8,8,dtype=torch.float64);cross.fill_diagonal_(0)
    cross[0,1]=cross[1,0]=0.
    bank['active'][8]=False;bank['valid'][2,1]=False
    out=_assemble(bank,cross)
    assert out['slot_routes'][1].tolist()==[0,2,2]
    assert out['route_compat'][:,0].eq(0).all() and out['route_compat'][:,:,0].eq(0).all()
    assert out['route_compat'][1,2,1]==1 and out['route_compat'][1,1,1]==0
    assert not out['commit'][8].any() and not out['commit'][2,1]
    reset=out['reset_matrices'][1]
    assert reset[1,0]==1 and reset[1,1]==0  # old anchor0 -> E at anchor1.
    assert reset[2,2]==1 and reset[0,0]==1
    assert reset.sum(-1).eq(1).all()


def test_missing_source_zero_trust_delta_requires_strict_positive():
    bank,_=fixture();cross=torch.ones(8,8,dtype=torch.float64);cross.fill_diagonal_(0)
    bank['source_anchors'][8,1]=-1
    bank['side_reliability'][8,1]=0
    bank['extent_scores'][8,1]=bank['extent_scores'][8,0]
    bank['association_scores'][8,1]=bank['association_scores'][8,0]
    out=_assemble(bank,cross)
    assert out['slot_routes'][8,2]==0 and out['candidate_trust'][8,2]==0
    assert out['delta_support'][8,1]==0 and not out['commit'][8,1]


def test_attribute_containers_and_all_outputs_are_detached_CPU():
    bank,ev=fixture()
    for k,v in bank.items():
        if torch.is_tensor(v)and v.is_floating_point():bank[k]=v.requires_grad_()
    b=SimpleNamespace(**{k:v for k,v in bank.items()if k!='GT_read'},metadata={'GT_read':False})
    e=SimpleNamespace(**{k:v for k,v in ev.items()if k!='GT_read'},metadata={'GT_read':False})
    out=build(b,e)
    for v in out.values():
        if torch.is_tensor(v):assert v.device.type=='cpu' and not v.requires_grad


def test_high_quality_single_witness_cannot_self_verify_or_commit():
    bank,_=fixture();cross=torch.zeros(8,8,dtype=torch.float64)
    cross[0,1]=cross[1,0]=1.
    out=_assemble(bank,cross)
    assert out['anchor_quality'][0] >= .1
    assert out['witness_count'][0] == 1
    assert not out['source_verified'][0] and not out['commit'].any()
    assert torch.equal(out['reset_matrices'],torch.eye(9,dtype=torch.float64)[None].repeat(9,1,1))
