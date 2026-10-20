import torch
from stvg.readout import generalized_iou
def corrected(boxes,source,E):
    b=boxes.double();src=source.to(b);e=E.to(b)
    assert torch.isfinite(b).all()and (b[:,2:]>0).all()and (src[:,2:]>0).all()
    return torch.cat([e[:,:2]+(b[:,:2]-src[:,:2]),e[:,2:]*torch.exp(torch.log(b[:,2:])-torch.log(src[:,2:]))],-1)


def local_cost(boxes,targets):
    return 5*(boxes-targets).abs().sum(-1)+2*(1-generalized_iou(boxes,targets))
