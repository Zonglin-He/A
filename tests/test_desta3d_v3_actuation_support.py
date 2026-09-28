import pytest,torch
from vg_tta.desta3d_v3_actuation_support import spatial_positions

def test_actual_probe_support_survives_empty_parsed_output():
    trace={'token_ids':{'ordered_time_tokens':[101,102,103]},'branches':[{},
      {'probes':[{'kwargs':{'query_token_ids':[77]}},{'kwargs':{'query_token_ids':[102,103]}}],
       'logits':{'coordinate':torch.zeros(2,4,1001)}}]}
    assert spatial_positions(trace)==[1,2]
    trace['branches'][1]['logits']['coordinate']=torch.zeros(1,4,1001)
    with pytest.raises(ValueError):spatial_positions(trace)

def test_gradient_arithmetic_rebuild_matches_live_adam():
    torch.manual_seed(4);p=torch.nn.Parameter(torch.randn(12));initial=p.detach().clone()
    o=torch.optim.AdamW([p],lr=.01,weight_decay=0);saved=[]
    for i in range(24):
        g=torch.randn_like(p);before=p.detach().clone();p.grad=g.clone();torch.nn.utils.clip_grad_norm_([p],1.);o.step()
        saved.append((g,p.detach()-before,p.detach().clone()))
    q=torch.nn.Parameter(initial);r=torch.optim.AdamW([q],lr=.01,weight_decay=0)
    for i,(g,d,after) in enumerate(saved):
        before=q.detach().clone();q.grad=g.clone();torch.nn.utils.clip_grad_norm_([q],1.);r.step()
        assert torch.equal(q.detach()-before,d) and torch.equal(q,after)
        assert int(r.state[q]['step'])==i+1 and next(iter(r.state)) is q
    for k in ['step','exp_avg','exp_avg_sq']:assert torch.equal(o.state[p][k],r.state[q][k])
