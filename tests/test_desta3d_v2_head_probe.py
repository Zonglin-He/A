import pytest
import torch
from vg_tta.desta3d_v2_head_probe import HeadEvidence, full_reference


def test_original_forward_unchanged_and_frozen():
    torch.manual_seed(91)
    m=torch.nn.Linear(7,13,bias=False).bfloat16().requires_grad_(False)
    h=torch.randn(5,7).bfloat16()
    data={'labels':torch.tensor([[-100,1,2,3,4,5]]),'ptd_prefix_lengths':torch.tensor([4])}
    before=m(h); ev=HeadEvidence(data); hook=m.register_forward_hook(ev.hook)
    after=torch.cat([m(h[:3]),m(h[3:])]); hook.remove(); raw=ev.finish()
    assert torch.equal(before,after) and torch.equal(raw['logits'],before) and torch.equal(raw['hidden'],h)
    assert raw['tokens']['positions'].tolist()==[1,2,3,4,5]
    assert all(p.grad is None and not p.requires_grad for p in m.parameters())


def test_full_vocab_streamed_lse_and_rounding():
    torch.manual_seed(73)
    h=(torch.randn(5,7)*30).bfloat16(); w=torch.randn(17,7).bfloat16(); t=torch.tensor([0,7,8,16,11])
    z=h.double()@w.double().T; actual=z.bfloat16()
    result=full_reference(h,actual,t,[(0,w[:8]),(8,w[8:16]),(16,w[16:])])
    assert torch.allclose(result['token']['ce_fp64'],torch.logsumexp(z,1)-z[range(5),t],atol=1e-12,rtol=0)
    assert torch.equal(result['token']['target_round'],actual.double()[range(5),t])
    assert result['full_vocab']['round_mismatch_elements']==0
    with pytest.raises(AssertionError,match='Incomplete vocabulary'):
        full_reference(h,actual,t,[(0,w[:8])])


def test_invalid_support_rejected():
    h=torch.zeros(1,7,dtype=torch.bfloat16); w=torch.zeros(2,7,dtype=torch.bfloat16)
    with pytest.raises(AssertionError):full_reference(h,torch.zeros(1,2,dtype=torch.bfloat16),torch.tensor([2]),[(0,w)])
    with pytest.raises(AssertionError):full_reference(h,torch.zeros(1,2,dtype=torch.bfloat16),torch.tensor([1]),[(1,w)])
