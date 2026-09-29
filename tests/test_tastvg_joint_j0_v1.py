import types
import torch
from methods.tastvg_dual_evidence_j0_v1.method import fast_rerank,OnlineMethod,combine_layers


def test_fast_selects_student_interval_and_preserves_pre_boxes():
    pre=dict(boxes=torch.tensor([[.5,.5,.2,.2]]),indices=[0,1],physical_interval=[0,2]);cs=[dict(indices=[0,1],physical_interval=[0,2]),dict(indices=[1,2],physical_interval=[1,4])]
    out,d=fast_rerank(pre,cs,dict(proposals=[[1.2,3.8]],proposal_confidence=[.9]));assert d['selected']==1 and out['indices']==[1,2];assert out['boxes'] is pre['boxes'];assert pre['indices']==[0,1]
    out,d=fast_rerank(pre,cs,dict(proposals=[],proposal_confidence=[]));assert d['selected']==0 and out['indices']==pre['indices']


def test_nonexpert_cannot_read_evidence_or_change_state():
    p=OnlineMethod.__new__(OnlineMethod);p.fast=p.slow=True
    state={'spatial.query_residual':torch.zeros(256)};pre=dict(boxes=torch.ones(4,4),indices=[0,2])
    p.actor=types.SimpleNamespace(state=lambda:{k:v.clone() for k,v in state.items()},initial=state,values=lambda data:([],pre['boxes'],pre))
    def forbidden():raise AssertionError('nonexpert evidence read')
    out,_=p.arrive({},False,forbidden,forbidden);assert not out['updated'];assert out['pre_state_sha256']==out['post_state_sha256'];assert torch.equal(out['output_prediction']['boxes'],pre['boxes']);assert not out['temporal_expert_read'] and not out['spatial_expert_read']


def test_all_layer_temporal_offsets_are_interleaved():
    ids=[0,1,2,3,4,5];records=[dict(frame_ids=[0,2,4]),dict(frame_ids=[1,3,5])]
    b=[torch.full((6,3,4),float(j)) for j in [1,2]];z=[torch.zeros(6,1,3,2),torch.zeros(6,1,3,2)]
    for j in range(2):z[j][:,:,0,0]=4;z[j][:,:,2,1]=4
    out=combine_layers(b,z,records,ids);assert len(out)==6
    for p in out:assert p['boxes'][:,0].tolist()==[1,2,1,2,1,2] and p['indices']==[0,5]
