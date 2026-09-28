import copy
import pytest
import torch
from vg_tta.desta3d_v3_context_mask import temporal_context,spatial_context,context_masks
from vg_tta.desta3d_v3_latent_oracle import masks_from_source_record,matched_wrong_masks
from tests.test_desta3d_v3_latent_oracle import fixture,label
from vg_tta.desta3d_v3_location_probe import mask_at_location
from vg_tta.desta3d_v3_large_mask import large_mask_delta,RATIOS

def test_irregular_physical_half_open_neighbors_and_edges():
    ids=[1,10,11,50,90];m=torch.tensor([0,1,1,0,0.]).reshape(1,5,1,1)
    x,d=temporal_context(m,ids,[10,50]);assert x.flatten().tolist()==[1,1,1,1,0]
    assert d['added_frame_ids']==[1,50]
    m=torch.ones_like(m);x,d=temporal_context(m,ids,[0,91]);assert torch.equal(m,x) and not d['added_positions']
    with pytest.raises(ValueError):temporal_context(m,ids,[10,50])

def test_fractional_dilation_is_per_frame_one_cell_without_wrap():
    m=torch.zeros(1,3,5,5);m[0,0,0,0]=.4;m[0,0,1,1]=.7;m[0,1]=1
    x=spatial_context(m)
    assert x[0,0,0,0]==.7 and x[0,0,2,2]==.7 and x[0,0,3,3]==0
    assert torch.equal(x[0,1],m[0,1]) and not x[0,2].any() and x[0,0,-1,-1]==0
    assert torch.all(x>=m)
    with pytest.raises(ValueError):spatial_context(m+float('nan'))

def test_correct_wrong_same_rule_and_outcome_blind_eligibility():
    lab=label();m=masks_from_source_record(lab,lab['frame_ids'],2,2);w=matched_wrong_masks(lab,lab['frame_ids'],2,2)
    c=context_masks(m,w,lab['frame_ids'])
    for b,ob in [('event','temporal'),('spatial','spatial')]:
        assert c['diagnostic'][ob]['same_expansion_rule']
        for which in ['correct','wrong']:
            old=m if which=='correct' else w[ob]
            if b=='spatial':assert torch.equal(c['masks'][b][which][b],spatial_context(old[b]))
    assert torch.equal(c['masks']['spatial']['correct']['spatial'][:,1],m['spatial'][:,1])

@pytest.mark.parametrize('branch',['event','spatial'])
def test_real_hidden128_fixed_norm_branch_scope_and_state(branch):
    a,f,x=fixture();state=copy.deepcopy(a.state_dict());base=f();lab=label()
    m=masks_from_source_record(lab,lab['frame_ids'],2,2);w=matched_wrong_masks(lab,lab['frame_ids'],2,2)
    ctx=context_masks(m,w,lab['frame_ids'])['masks'][branch]['correct'][branch]
    with mask_at_location(a,branch,ctx,'late'):candidate=f()
    other='event' if branch=='spatial' else 'spatial'
    assert torch.equal(candidate['updated_tokens_'+other],base['updated_tokens_'+other])
    raw,d,log=large_mask_delta(x,base['updated_tokens_'+branch],candidate['updated_tokens_'+branch],branch)
    if log['applicable']:assert abs(float(d.double().norm()/x.double().norm())-RATIOS[branch])<1e-7
    else:assert not d.any()
    assert all(torch.equal(v,state[k]) for k,v in a.state_dict().items())
    assert all(not p.requires_grad and p.grad is None for p in a.parameters())
