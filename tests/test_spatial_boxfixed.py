"""Synthetic tests of standalone, label-free BoxFixed and spatial evidence."""
import itertools
import pytest
import torch
import torch.nn.functional as F
from stvg.evidence import SourceEvidence,extract_visual_grids
from stvg.spatial import build_geometry,evidence_dp,decode
from stvg._spatial_math import fit_axis_affine,transform_native_box


def fixture(empty=False):
    T,H,W=11,4,5;N=H*W
    raw=torch.zeros(T,N,256,dtype=torch.float32)
    raw[:,torch.arange(N),torch.arange(N)]=1.
    valid=torch.ones(T,N,dtype=torch.bool)
    seeds=torch.ones(8,H,W,dtype=torch.float64)
    if empty:seeds.zero_()
    positions=[0,1,2,4,6,8,9,10]
    ev=SourceEvidence.from_arrays(raw,valid,torch.arange(T)*3,positions,seeds)
    E=torch.tensor([.4,.5,.2,.3],dtype=torch.float64)[None].repeat(T,1)
    native=torch.tensor([.5,.5,.4,.5],dtype=torch.float64)[None].repeat(8,1)
    return ev,E,native


def test_geometry_independent_of_interval_and_fulltimeline_support():
    ev,E,native=fixture();g=build_geometry(ev,E,native)
    assert not hasattr(g,'event')and not hasattr(g,'active')
    original=g.boxes.clone();a=g.for_interval([0,7]);b=g.for_full_timeline()
    assert torch.equal(a.boxes,b.boxes)and torch.equal(g.boxes,original)
    assert not a.active[3:].any()and b.event.all()
    assert b.score_counts==dict(observed=8,unobserved=3,event_edges=10,all_nodes=11)
    assert a.score_counts==dict(observed=3,unobserved=0,event_edges=2,all_nodes=11)
    assert torch.equal(b.boxes[ev.positions,1],native)
    out=evidence_dp(a);assert out['actions'][~a.active].eq(0).all()
    assert out['costs']['updates']==out['costs']['decoder_calls']==0
    with pytest.raises(ValueError):decode(g,mode='full_timeline',interval=[0,7])
    with pytest.raises(ValueError):decode(g)


def test_empty_mask_native_boxes_retained_and_temporal_reliability_separate():
    ev,E,native=fixture(empty=True);g=build_geometry(ev,E,native)
    assert g.confidence[ev.positions].eq(1).all()
    assert g.temporal_reliability().eq(0).all()
    assert not g.reliable[~g.observed].any()
    assert torch.equal(g.boxes[ev.positions,1],native)
    view=g.for_full_timeline();out=evidence_dp(view)
    assert out['actions'][~g.observed].eq(0).all()
    assert torch.isfinite(view.score_unary).all()and torch.isfinite(view.score_pair).all()


def test_dp_global_optimum_and_original_tie_order():
    ev,E,native=fixture();bank=build_geometry(ev,E,native).for_interval([0,4])
    out=evidence_dp(bank)
    options=[torch.nonzero(bank.valid[t]if bank.active[t]else torch.tensor([True,False,False]),as_tuple=True)[0].tolist()for t in range(bank.frames)]
    best=max(float(bank.score_path(torch.tensor(a)))for a in itertools.product(*options))
    assert out['score']==best
    bank.score_unary.zero_();bank.score_pair.zero_()
    assert evidence_dp(bank)['actions'].eq(0).all()


def test_native_logits_conversion_preserves_official_operation_order():
    ev,_,_=fixture();g=torch.Generator().manual_seed(5)
    logits=torch.randn(8,3,5,generator=g)
    actual=SourceEvidence.from_native_masks(ev.features,ev.valid,ev.frame_ids,ev.positions,logits,(9,13),(4,5))
    full=F.interpolate(logits[:,None],size=(9,13),mode='bilinear',align_corners=False).sigmoid()
    expected=F.interpolate(full,size=(4,5),mode='area').squeeze(1).double()
    assert torch.equal(actual.seeds,expected)


def test_exact_affine_keeps_native_full_extent_without_clipping():
    x=torch.tensor([[.1,.2],[.4,.7],[.8,.3],[.6,.9]],dtype=torch.float64)
    scale=torch.tensor([1.5,.75],dtype=torch.float64);delta=torch.tensor([.04,.1],dtype=torch.float64)
    fit,info=fit_axis_affine(torch.eye(4,dtype=torch.float64),x,x*scale+delta)
    assert info['valid']
    native=torch.tensor([.4,.5,.6,.8],dtype=torch.float64)
    torch.testing.assert_close(transform_native_box(native,*fit),torch.cat((native[:2]*scale+delta,native[2:]*scale)),atol=1e-15,rtol=1e-15)


def test_source_grid_actual_offset_interleave_and_visual_prefix():
    T,H,W=9,3,4;N=H*W;S=N+3+N;views=[]
    y=torch.arange(H)[:,None].expand(H,W).reshape(-1)+1
    x=torch.arange(W)[None].expand(H,W).reshape(-1)+1
    for offset in (0,1):
        n=len(range(offset,T,2));f=torch.zeros(S,n,256);pos=torch.zeros_like(f)
        pos[:N,:,:128]=y[:,None,None];pos[:N,:,128:]=x[:,None,None]
        for local,node in enumerate(range(offset,T,2)):f[:N,local]=node+1
        f[N:]=999 # Text and motion slots must not leak into visual evidence.
        views.append(dict(encoded_feature=f,encoded_pos=pos,encoded_mask=torch.zeros(n,S,dtype=torch.bool)))
    features,valid,frames,shape,_=extract_visual_grids(views,list(range(T)))
    assert shape==(H,W)and features.shape==(T,N,256)and valid.all()
    assert all(features[t].eq(t+1).all()for t in range(T))


def test_four_thread_host_restored_and_all_CPU_arithmetic_matches_one(monkeypatch):
    import stvg.evidence as evidence_module
    import stvg.spatial as spatial_module
    from stvg._spatial_math import single_thread_cpu
    previous=torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        generator=torch.Generator().manual_seed(728)
        T,H,W=11,7,10;positions=[0,1,2,4,6,8,9,10]
        raw=torch.randn(T,H*W,256,generator=generator)
        valid=torch.ones(T,H*W,dtype=torch.bool)
        seeds=torch.rand(8,H,W,dtype=torch.float64,generator=generator)
        E=torch.tensor([.4,.5,.2,.3],dtype=torch.float64)[None].repeat(T,1)
        native=E[:8].clone();native[:,0]+=.04
        reference_evidence=SourceEvidence(raw,valid,torch.arange(T),positions,seeds)
        reference=build_geometry(reference_evidence,E,native).for_interval([1,9])
        expected=evidence_dp(reference)
        calls=[]
        old_ev,old_sp=evidence_module.affinity,spatial_module.affinity
        def spy_ev(*a,**kw):calls.append(torch.get_num_threads());return old_ev(*a,**kw)
        def spy_sp(*a,**kw):calls.append(torch.get_num_threads());return old_sp(*a,**kw)
        monkeypatch.setattr(evidence_module,'affinity',spy_ev)
        monkeypatch.setattr(spatial_module,'affinity',spy_sp)
        torch.set_num_threads(4)
        actual_evidence=SourceEvidence(raw,valid,torch.arange(T),positions,seeds)
        assert torch.get_num_threads()==4
        actual=build_geometry(actual_evidence,E,native).for_interval([1,9])
        assert torch.get_num_threads()==4
        result=evidence_dp(actual)
        assert torch.get_num_threads()==4 and calls and set(calls)=={1}
        for name in ('adjacent_forward_indices','adjacent_forward_weights'):
            assert torch.equal(getattr(actual_evidence,name),getattr(reference_evidence,name))
        for name in ('boxes','features','confidence','projection','guard','link','score_unary','score_pair'):
            assert torch.equal(getattr(actual,name),getattr(reference,name)),name
        assert torch.equal(result['actions'],expected['actions'])
        assert torch.equal(result['final'],expected['final'])and result['score']==expected['score']
        with single_thread_cpu():
            assert torch.get_num_threads()==1
            with single_thread_cpu():assert torch.get_num_threads()==1
            assert torch.get_num_threads()==1
        assert torch.get_num_threads()==4
        with pytest.raises(ValueError):
            with single_thread_cpu():raise ValueError('synthetic failure')
        assert torch.get_num_threads()==4
        with pytest.raises(ValueError):SourceEvidence(raw,valid,torch.arange(T),positions,seeds[:7])
        assert torch.get_num_threads()==4
    finally:torch.set_num_threads(previous)
