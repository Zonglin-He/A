import copy
import torch
from vg_tta.desta3d_v2 import Desta3DAdapterV2
from vg_tta.desta3d_v2_tta_pilot import configure,forward,adapt
from vg_tta.desta3d_v2_source_task_control import active,flat_gradient,unlabeled_signals,adam_step


def fixture():
    torch.set_num_threads(2);torch.manual_seed(19)
    a=Desta3DAdapterV2(hidden_dim=128,architecture='dual3d',p1_enabled=False).eval()
    f={'visual_grid':torch.randn(1,3,2,2,2560),'query_tokens':torch.randn(1,4,2560),
       'query_mask':torch.ones(1,4,dtype=torch.bool),'frame_times':torch.tensor([[0.,.5,1.]])}
    view=dict(f,visual_grid=f['visual_grid']*.91+.02)
    m={'feature_definition':'v2_query_conditioned_readers',
       'spatial':{'mean':[0.]*128,'std':[1.]*128},'event':{'mean':[0.]*128,'std':[1.]*128}}
    return a,f,view,m


def test_unlabeled_three_steps_exact_original_scope_and_restore():
    a,f,v,m=fixture();b=copy.deepcopy(a);state=copy.deepcopy(a.state_dict())
    expected=adapt(b,f,v,interface='calibration',alignment=.01,moments=m)
    initial,count=configure(a,'calibration');assert count==66816
    with torch.no_grad():teacher=forward(a,f)
    names,params=active(a);opt=torch.optim.AdamW(params,lr=1e-5,weight_decay=0.)
    for step in range(1,4):
        raw,loss=unlabeled_signals(a,v,teacher,initial,m)
        assert all(p.grad is None for p in a.parameters())
        assert loss['total']==expected['history'][step-1]['loss_before']
        delta,report=adam_step(a,opt,raw['unlabeled_total'],step=step)
        assert set(report['actual_Adam_counters'].values())=={step}
    assert all(torch.equal(x,b.state_dict()[n]) for n,x in a.state_dict().items())
    assert all(torch.equal(x,state[n]) for n,x in a.state_dict().items() if n not in names)
    a.load_state_dict(state);a.set_train_stage('frozen')
    assert all(torch.equal(x,state[n]) for n,x in a.state_dict().items())


def test_synthetic_task_branch_sum_in_same_calibration_subspace():
    a,f,_,_=fixture();configure(a,'calibration');_,params=active(a)
    y=forward(a,f)
    e=y['delta_event'].square().mean();s=y['delta_spatial'].square().mean()
    ge=flat_gradient(e,params,True);gs=flat_gradient(s,params,True)
    joint=flat_gradient(e+s,params)
    assert torch.allclose(ge+gs,joint,atol=1e-7,rtol=2e-4)
    assert torch.isfinite(joint).all() and joint.count_nonzero()>0
    assert all(p.grad is None for p in a.parameters())
    assert not a.gate_event.requires_grad and not a.gate_spatial.requires_grad
