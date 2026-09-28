import pytest,torch
from vg_tta.desta3d_v3_gap_native_state import native_state_from_records

def test_reject_restricted_coordinate_confidence():
    pred=dict(frame_ids=[1],positions=[0],boxes_cxcywh=torch.ones(1,4),geometry_valid=torch.ones(1,dtype=torch.bool),interval=[0,0],readout=dict(raw_blocks=torch.zeros(1,6,dtype=torch.long)))
    trace=dict(branches=[dict(logits=dict(time=torch.zeros(2,1))),dict(logits=dict(coordinate=torch.zeros(1,4,1001)))])
    with pytest.raises(ValueError,match='complete152775'):native_state_from_records(pred,trace,torch.ones(1,4),torch.ones(1,dtype=torch.bool))

def test_missing_branch_state_has_explicit_validity_without_GT_prefix():
    pred=dict(frame_ids=[1,3],positions=[],boxes_cxcywh=torch.empty(0,4),geometry_valid=torch.empty(0,dtype=torch.bool),interval=None)
    result=native_state_from_records(pred,dict(branches=[]),torch.ones(2,4),torch.ones(2,dtype=torch.bool))
    assert result.shape==(2,33) and not result.any()

def test_complete_vocab_and_actual_native_block_tokens():
    logits=torch.zeros(1,4,152775);logits[...,7]=1;logits[...,8]=2
    pred=dict(frame_ids=[1,3],positions=[0],boxes_cxcywh=torch.tensor([[.4,.4,.4,.4]]),geometry_valid=torch.ones(1,dtype=torch.bool),interval=[0,1],
        readout=dict(raw_blocks=torch.tensor([[100,7,7,7,7,200]])))
    trace=dict(token_ids=dict(ordered_time_tokens=[100,101]),branches=[dict(logits=dict(time=torch.zeros(2,2))),
        dict(logits=dict(coordinate=logits),probes=[dict(kwargs=dict(query_token_ids=[100]))])])
    state=native_state_from_records(pred,trace,torch.ones(2,4),torch.tensor([True,False]))
    expected=logits.softmax(-1)[0,:,7]
    assert torch.equal(state[0,14:18],expected) and (state[0,18:22]<0).all()
    assert state[0,26]==1 and state[1,26]==0 and state[0,32]==1
