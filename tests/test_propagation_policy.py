import itertools
import torch
from tools.propagation_policy.core import initial_weights,rollout,graph_path,proxy,state


def cache():
    torch.manual_seed(77);T=4
    return dict(boxes=torch.rand(T,3,4,dtype=torch.double),valid=torch.ones(T,3,dtype=torch.bool),active=torch.ones(T,dtype=torch.bool),inside=torch.ones(T,dtype=torch.bool),descriptors=torch.nn.functional.normalize(torch.randn(T,3,6,dtype=torch.double),dim=-1),support=torch.rand(T,3,dtype=torch.double),deviation=torch.rand(T,3,dtype=torch.double),links=torch.rand(T-1,3,3,dtype=torch.double),confidence=torch.ones(T,dtype=torch.double)*.7,initial_memory=torch.nn.functional.normalize(torch.randn(6,dtype=torch.double),dim=0),positions=torch.tensor([0,3]))


def test_zero_head_strictly_returns_source_and_candidate_prior_is_point_seven():
    c=cache();r=rollout(c,initial_weights(c),greedy=True)
    assert r['choices'].count_nonzero()==0
    assert torch.allclose(r['probabilities'][:,0],torch.full((4,),.7,dtype=torch.double))


def test_real_student_history_changes_future_matching_states():
    c=cache();p=initial_weights(c);p[0]=1;noise=torch.zeros(4,3,dtype=torch.double);noise[0,0]=100
    a=rollout(c,p,noise);noise[0,0]=0;noise[0,1]=100;b=rollout(c,p,noise)
    assert a['choices'][0]!=b['choices'][0]
    assert not torch.equal(a['features'][1,:,0],b['features'][1,:,0])
    assert max(a['writes'])<=.2 and max(b['writes'])<=.2


def test_no_evidence_forces_source_even_after_shared_parameter_updates():
    c=cache();c['active'][:]=False;c['valid'][:,1:]=False
    r=rollout(c,torch.full((6,),100.,dtype=torch.double),torch.randn(4,3,dtype=torch.double)*100)
    assert r['choices'].count_nonzero()==0 and sum(r['writes'])==0


def test_full_direct_history_gradient_matches_finite_difference():
    c=cache();p=(initial_weights(c)+.1).requires_grad_(True);noise=torch.randn(4,3,dtype=torch.double)
    assert torch.autograd.gradcheck(lambda x:rollout(c,x,noise,kind='DirectSoft')['loss'],(p,),eps=1e-5,atol=1e-5)


def test_dynamic_program_exactly_maximizes_memory_free_acceptance_score():
    c=cache();best=max(itertools.product(range(3),repeat=4),key=lambda x:proxy(c,torch.tensor(x)))
    assert tuple(graph_path(c).tolist())==best


def test_fixed_pseudo_teacher_paths_do_not_depend_on_student_parameters():
    c=cache();noise=torch.randn(4,3,dtype=torch.double)
    a=rollout(c,initial_weights(c),noise,teacher_path=True);b=rollout(c,initial_weights(c)+100,noise,teacher_path=True)
    assert torch.equal(a['choices'],b['choices'])and torch.equal(a['features'],b['features'])


def test_uniform_visual_correspondence_cannot_claim_reliable_alternatives():
    from tools.propagation_policy.candidates import build
    cells=torch.tensor([[0.,0.,.5,.5],[.5,0.,1.,.5],[0.,.5,.5,1.],[.5,.5,1.,1.]])
    f=torch.ones(10,4,6);b=torch.tensor([[.4,.4,.2,.2]]*10);p=[0,1,2,3,6,7,8,9]
    c=build(f,cells,torch.arange(10.),p,torch.tensor([[1.,0.,0.,0.]]*8),b,b[p],[0,10])
    assert torch.equal(c['boxes'][:,0],b)
    assert not c['active'][4]and not c['active'][5]
    assert c['valid'][4].tolist()==[True,False,False]
