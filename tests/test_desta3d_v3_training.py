import torch
from vg_tta.optimizer_checkpoint import cpu_clone,restore_optimizer
from vg_tta.desta3d_v3_training import checked_step,optimizer_counters
from vg_tta.desta3d_v2_training import warmup_cosine
def test_live_resume_two_updates_match_continuous_and_missing_gradient_counter():
    torch.manual_seed(13);a=torch.nn.Linear(3,2);o=torch.optim.AdamW(a.parameters(),lr=1e-4,weight_decay=0)
    a(torch.ones(1,3)).square().sum().backward();checked_step(a,o)
    state=cpu_clone(a.state_dict());ostate=cpu_clone(o.state_dict())
    b=torch.nn.Linear(3,2);b.load_state_dict(state);p=torch.optim.AdamW(b.parameters(),lr=1e-4,weight_decay=0);restore_optimizer(p,ostate)
    for _ in range(2):
        for m,opt in [(a,o),(b,p)]:m.weight.square().sum().backward();checked_step(m,opt)
    for x,y in zip(a.parameters(),b.parameters()):assert torch.equal(x,y)
    assert optimizer_counters(a,o)==optimizer_counters(b,p)=={'weight':3,'bias':1}
    for x,y in zip(o.state.values(),p.state.values()):
        for k in x:assert torch.equal(x[k],y[k])
def test_full_horizon_warmup_and_terminal_decay():
    total=181135;warm=9057
    assert abs(warmup_cosine(0,total)-1/warm)<1e-14
    assert warmup_cosine(warm-1,total)==1 and warmup_cosine(total-1,total)==0
