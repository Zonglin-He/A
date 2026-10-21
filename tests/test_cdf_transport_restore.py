import numpy as np
import torch
from tools.tubedetr_correspondence_qualification import transport
from tools.tubedetr_cdf_qualification import restored_transport


def test_sealed_correspondences_reconstruct_same_support_without_features():
    base=torch.eye(16).T.reshape(16,4,4)
    features=torch.stack([base if i%2==0 else base.flip(-1)for i in range(8)])
    boxes=torch.tensor([[.25,.5,.5,1.]]*8)
    maps,points,edges=transport(features,boxes,torch.ones(8,2),4)
    data=dict(points=points.numpy())
    for t,(a,b)in enumerate(edges):data[f'edge_{t}_source']=a;data[f'edge_{t}_target']=b
    rebuilt=restored_transport(data,boxes)
    assert np.array_equal(maps.numpy(),rebuilt.numpy())
