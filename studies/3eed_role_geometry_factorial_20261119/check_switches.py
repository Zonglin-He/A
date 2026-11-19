"""Verify exact historical endpoints and independent switch routing on synthetic input."""
import torch

from study import CELLS
from decoder import Decoder, objective
from factor_decoder import FactorDecoder


def main():
    torch.manual_seed(174)
    torch.set_num_threads(1)
    B,N,L = 3,96,20
    x = dict(points=torch.randn(B,N,288),rgb=torch.randn(B,N,64),xyz=torch.randn(B,N,3),
             uv=torch.rand(B,N,2), visible=torch.ones(B,N,dtype=torch.bool),
             in_image=torch.ones(B,N,dtype=torch.bool),text=torch.randn(B,L,288),
             valid=torch.ones(B,L,dtype=torch.bool),has_reference=torch.tensor([True,True,False]),
             role_mask=torch.stack([torch.arange(L)//6==k for k in range(3)])[None].expand(B,-1,-1),
             prior=torch.cat((torch.randn(B,256,3),torch.rand(B,256,3)+.25),-1),
             prior_scores=torch.randn(B,2,256),anchors=torch.arange(8)[None].expand(B,-1).reshape(B,2,4),
             gt=torch.tensor([0.,0.,0.,1.,.8,1.,.5])[None,None].expand(B,2,-1),
             known_ref=torch.tensor([True,True,False]))
    baseline = Decoder('joint')
    state = baseline.state_dict()
    first = None
    for cell in CELLS:
        m = FactorDecoder(cell)
        m.load_state_dict(state,strict=True)
        y = m(x)
        if first is None:
            first = y['passes'][0][2]
        torch.testing.assert_close(first,y['passes'][0][2],rtol=0.,atol=0.)
        if cell in ('off_off','joint'):
            b = Decoder('joint' if cell=='joint' else 'without_feedback')
            b.load_state_dict(state,strict=True)
            original = b(x)
            for a,b in zip(y['passes'],original['passes'],strict=True):
                for u,v in zip(a,b,strict=True):
                    torch.testing.assert_close(u,v,rtol=0.,atol=0.)
        objective(x,y).backward()
        is_role = cell in ('joint','role_only')
        is_geo = cell in ('joint','geometry_only')
        assert (m.pair[0].weight.grad is not None)==is_role
        assert (m.from_ref.weight.grad is not None)==is_role
        assert (m.spatial_gate.grad is not None)==is_geo
        assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    print('Both historical settings replay exactly. Gradients bind each new cell to its specified path.',flush=True)


if __name__=='__main__':
    main()
