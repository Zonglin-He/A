import numpy as np
import torch
from test_boxfixed_tube_upstream import fixture
from tools.boxfixed_tube.spatial import build_geometry,evidence_dp
from tools.boxfixed_tube.profiles import view_geometry,predict
from stvg.predictor import output_writeback


def test_default_profile_is_exact_upstream_including_DP_and_score():
    e,E,n=fixture();g=build_geometry(e,E,n);p=dict(gamma=1.,admission=.1,guard=.1,link=.25)
    a=evidence_dp(g.for_interval([0,31]));b=predict(g,e,E.numpy(),E.numpy(),[0,31],p)
    assert torch.equal(a['final'],b['final'])and torch.equal(a['actions'],b['actions'])and a['score']==b['score']


def test_fast_gamma_view_matches_full_geometry_reconstruction_bitwise():
    e,Source,n=fixture();raw=Source.numpy().copy();raw[:,:2]+=.03;raw[:,2:]*=1.2
    baseE=torch.from_numpy(output_writeback(Source.numpy(),raw,.5));base=build_geometry(e,baseE,n)
    for gamma in [0.,.025,.25,1.]:
        p=dict(gamma=gamma,admission=.1,guard=.1,link=.25);fast=view_geometry(base,e,Source.numpy(),raw,p);full=build_geometry(e,torch.from_numpy(output_writeback(Source.numpy(),raw,gamma)),n)
        for name in ['E','boxes','valid','features','projection','guard','link','side_reference_boxes']:
            assert torch.equal(getattr(fast,name),getattr(full,name)),(gamma,name)
        a=evidence_dp(fast.for_interval([0,31]));b=evidence_dp(full.for_interval([0,31]));assert torch.equal(a['final'],b['final'])and a['score']==b['score']


def test_empty_masks_threshold_profile_does_not_invent_propagated_boxes():
    e,E,n=fixture(empty=True);g=build_geometry(e,E,n);p=dict(gamma=1.,admission=.05,guard=.1,link=.25);x=predict(g,e,E.numpy(),E.numpy(),[0,31],p)
    assert x['actions'][~g.observed].eq(0).all()
