import torch,pytest
from vg_tta.desta3d_v3_actuation_full_vocab import full_coordinates
from vg_tta import desta3d_v2_output_anchor as base

def test_full_vocab_values_gradient_and_exception_restore():
    original=base._selected_logits;x=torch.randn(1,12,21,requires_grad=True);t={'ordered_time_tokens':[3,4]}
    with full_coordinates():
        kind,z=base._selected_logits(x,[3,4],t,[8,9,10]);assert kind=='coordinate'
        assert torch.equal(z,x.reshape(2,6,21)[:,1:5]);z.sum().backward()
    assert base._selected_logits is original
    assert x.grad.reshape(2,6,21)[:,[0,5]].count_nonzero()==0
    with pytest.raises(RuntimeError):
        with full_coordinates():raise RuntimeError('test')
    assert base._selected_logits is original

def test_restricted_ce_misses_noncoordinate_competitor():
    x=torch.tensor([[0.,1.,8.]],requires_grad=True);target=torch.tensor([1])
    narrow=torch.nn.functional.cross_entropy(x[:,:2],target);wide=torch.nn.functional.cross_entropy(x,target)
    ng=torch.autograd.grad(narrow,x,retain_graph=True)[0];wg=torch.autograd.grad(wide,x)[0]
    assert ng[0,2]==0 and wg[0,2]>0 and wide>narrow
