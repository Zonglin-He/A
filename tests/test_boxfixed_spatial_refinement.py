import copy
import torch
from test_boxfixed_tube_upstream import fixture
from tools.boxfixed_tube.spatial import build_geometry
from tools.boxfixed_tube.profiles import predict as original_predict
from tools.boxfixed_followup.refinement import refined_box, predict


def test_relative_residual_tracks_source_motion_and_exact_identity():
    a = torch.tensor([.2,.3,.1,.2], dtype=torch.float64)
    t = torch.tensor([.7,.6,.3,.4], dtype=torch.float64)
    teacher = a.clone(); teacher[:2] += torch.tensor([.01,-.02]); teacher[2:] *= 1.5
    r = refined_box(t,a,teacher,t,1.,0.,'relative_residual')
    torch.testing.assert_close(r[:2], t[:2]+t[2:]*torch.tensor([.1,-.1],dtype=torch.float64),atol=1e-8,rtol=0)
    assert torch.equal(r[2:],t[2:])
    assert torch.equal(refined_box(t,a,a,t,1.,1.,'relative_residual'),t)


def test_preserved_extent_and_endpoints_no_clipping():
    src = torch.tensor([.1,.5,.8,.6], dtype=torch.float64)
    prop = torch.tensor([.15,.55,.4,.3], dtype=torch.float64)
    assert torch.equal(refined_box(src,src,src,prop,0.,0.,'center_preserve'),src)
    assert torch.equal(refined_box(src,src,src,prop,1.,1.,'center_preserve'),prop)
    b = refined_box(src,src,src,prop,.5,0.,'center_preserve')
    assert torch.equal(b[2:],src[2:]) and b[0]-b[2]/2 < 0


def test_original_control_bitexact_and_refinement_leaves_base_immutable():
    e,S,n = fixture(); raw=S.numpy().copy(); g=build_geometry(e,S,n)
    old=copy.deepcopy(g.state_dict()); p=dict(gamma=0.,admission=.1,guard=.4,link=1.)
    a=original_predict(g,e,S.numpy(),raw,[0,31],p)
    b=predict(g,e,S.numpy(),raw,[0,31],dict(p,mode='original'))
    assert torch.equal(a['final'],b['final']) and a['score']==b['score']
    for mode in ['center_preserve','relative_residual']:
        r=predict(g,e,S.numpy(),raw,[0,31],dict(p,mode=mode,alpha=.5,beta=0.))
        assert torch.isfinite(r['final']).all() and (r['final'][:,2:]>0).all()
        assert r['actions'][~((g.frame_ids>=0)&(g.frame_ids<31))].eq(0).all()
    for k,v in old.items():
        if torch.is_tensor(v): assert torch.equal(v,getattr(g,k)),k


def test_empty_evidence_cannot_invent_unobserved_correction():
    e,S,n=fixture(empty=True);g=build_geometry(e,S,n)
    p=dict(gamma=0.,admission=.1,guard=.4,link=1.,mode='relative_residual',alpha=1.,beta=.5)
    r=predict(g,e,S.numpy(),S.numpy(),[0,31],p)
    assert r['actions'][~g.observed].eq(0).all()
