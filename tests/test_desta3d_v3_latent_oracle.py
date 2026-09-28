import copy
import pytest
import torch
from vg_tta.desta3d_v2 import Desta3DAdapterV2
from vg_tta.desta3d_v3_latent_oracle import masks_from_source_record,privileged_branch_latents

def label():
    return dict(frame_ids=[10,11,90],event_interval={'begin_fid':11,'end_fid':12},event_active=[False,True,False],
        box_valid=[True,False,True],boxes_xyxy=[[.25,0,.75,1],[0,0,0,0],[0,0,1,1]])

def test_physical_temporal_and_fractional_spatial_grid_missing_neutral():
    m=masks_from_source_record(label(),[10,11,90],2,2)
    assert torch.equal(m['event'][0,:,0,0],torch.tensor([0.,1.,0.]))
    assert torch.equal(m['spatial'][0,0],torch.full((2,2),.5))
    assert torch.equal(m['spatial'][0,1:],torch.ones(2,2,2))
    assert m['support']['spatial_valid_positions']==[0,2] # boxes outside event remain independent spatial evidence
    wrong=label();wrong['event_active'][0]=True
    with pytest.raises(ValueError,match='event labels'):masks_from_source_record(wrong,[10,11,90],2,2)
    with pytest.raises(ValueError,match='identity'):masks_from_source_record(label(),[10,12,90],2,2)

def fixture():
    torch.manual_seed(20260928)
    a=Desta3DAdapterV2(in_channels=8,query_dim=8,hidden_dim=128,architecture='dual3d',p1_enabled=False)
    a.eval();a.set_train_stage('frozen');x=torch.randn(1,3,2,2,8);q=torch.randn(1,4,8);qm=torch.ones(1,4,dtype=torch.bool);times=torch.tensor([[10.,11.,90.]])
    def forward():return a(x,q,qm,frame_times=times)
    return a,forward,x

def test_identity_disabled_and_exception_restore_exact():
    a,f,x=fixture();state=copy.deepcopy(a.state_dict());base=f();m={b:torch.ones(1,3,2,2) for b in ('event','spatial')}
    with privileged_branch_latents(a,m,event=True,spatial=True) as log:
        same=f()
    assert all(torch.equal(base[k],same[k]) for k in base if isinstance(base[k],torch.Tensor))
    assert all(r['changed_elements']==0 for r in log)
    with pytest.raises(RuntimeError):
        with privileged_branch_latents(a,m,event=True):raise RuntimeError('synthetic failure')
    again=f();assert torch.equal(again['updated_tokens_event'],base['updated_tokens_event'])
    assert all(torch.equal(v,state[n]) for n,v in a.state_dict().items())
    assert all(p.grad is None and not p.requires_grad for p in a.parameters())

@pytest.mark.parametrize('branch',['event','spatial'])
def test_only_selected_branch_projection_changes_and_frozen_gate_bias_unchanged(branch):
    a,f,x=fixture();base=f();m=masks_from_source_record(label(),[10,11,90],2,2)
    with privileged_branch_latents(a,m,event=branch=='event',spatial=branch=='spatial') as log:new=f()
    other='spatial' if branch=='event' else 'event'
    assert torch.equal(new['updated_tokens_'+other],base['updated_tokens_'+other])
    z=base['branch_features_'+branch];expected=getattr(a,'out_proj_'+branch)(z*(.25+.75*m[branch]).unsqueeze(-1))
    assert torch.equal(new['proposal_delta_'+branch],expected)
    assert torch.equal(new['updated_tokens_'+branch],x+expected*base['gate_'+branch])
    assert not torch.equal(new['updated_tokens_'+branch],base['updated_tokens_'+branch])
    assert len(log)==1 and log[0]['changed_elements']>0
    assert torch.equal(new['event_logits'],base['event_logits']) # evidence head precedes intervention, distinct from policy

def test_matched_wrong_temporal_duration_count_and_spatial_area():
    from vg_tta.desta3d_v3_latent_oracle import matched_wrong_masks,masks_from_source_record
    lab=dict(frame_ids=[10,11,50,90],event_active=[False,True,True,False],event_interval={'begin_fid':11,'end_fid':51},
        box_valid=[True,False,True,True],boxes_xyxy=[[.1,.2,.4,.6],[0,0,0,0],[.2,.3,.7,.9],[0,0,1,1]])
    original=masks_from_source_record(lab,lab['frame_ids'],3,5);wrong=matched_wrong_masks(lab,lab['frame_ids'],3,5)
    assert wrong['diagnostic']['temporal']['eligible_changed_support']
    assert wrong['diagnostic']['temporal']['same_physical_duration']==40
    assert torch.equal(original['event'].sum(),wrong['temporal']['event'].sum())
    assert not torch.equal(original['event'],wrong['temporal']['event'])
    assert torch.allclose(original['spatial'].sum((-2,-1)),wrong['spatial']['spatial'].sum((-2,-1)))
    assert torch.equal(wrong['spatial']['spatial'][0,1],torch.ones(3,5))
    assert wrong['diagnostic']['spatial']['shifts'][-1]['changed'] is False
    assert lab['boxes_xyxy'][0]==[.1,.2,.4,.6]
    alltime=dict(lab,event_active=[True]*4,event_interval={'begin_fid':10,'end_fid':91})
    no=matched_wrong_masks(alltime,lab['frame_ids'],3,5)
    assert not no['diagnostic']['temporal']['eligible_changed_support']
