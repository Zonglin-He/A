import torch
from vg_tta.desta3d_v2 import Desta3DAdapterV2
from vg_tta.desta3d_v2_training import source_evidence_losses,make_source_optimizer,warmup_cosine

def test_temporal_supervision_not_box_availability():
    outputs={'referent_logits':torch.zeros(1,3,2,2,requires_grad=True),
             'event_logits':torch.zeros(1,3,requires_grad=True)}
    record={'frame_ids':[0,2,5],'boxes_xyxy':[[0,0,1,1]]*3,
            'box_valid':[True,False,False],'event_active':[False,True,False]}
    loss=source_evidence_losses(outputs,record)
    (loss['ref']+loss['event']).backward()
    assert outputs['referent_logits'].grad[0,1:].eq(0).all()
    assert outputs['event_logits'].grad[0,1]<0
    assert outputs['event_logits'].grad[0,2]>0

def test_recipe_groups_and_schedule():
    m=Desta3DAdapterV2(in_channels=8,query_dim=8,hidden_dim=4)
    a=make_source_optimizer(m,'repaired','A')
    assert not m.gate_event.requires_grad and not m.out_proj_event.weight.requires_grad
    b=make_source_optimizer(m,'repaired','B')
    assert m.gate_event.requires_grad and m.out_proj_event.weight.requires_grad
    assert len({id(p) for g in b.param_groups for p in g['params']})==sum(len(g['params']) for g in b.param_groups)
    assert sum(p.numel() for g in b.param_groups for p in g['params'])==sum(p.numel() for p in m.parameters())
    assert warmup_cosine(0,100)==.2 and warmup_cosine(4,100)==1.
    assert warmup_cosine(99,100)==0.
