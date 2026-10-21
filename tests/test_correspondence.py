import torch
from tools.tubedetr_correspondence_qualification import transport


def test_mutual_correspondence_tracks_feature_permutation():
    base=torch.eye(16).T.reshape(16,4,4)
    features=torch.stack([base if i%2==0 else base.flip(-1)for i in range(8)])
    boxes=torch.tensor([[.5,.5,1.,1.]]*8)
    maps,points,edges=transport(features,boxes,torch.ones(8,2),4)
    for a,b in edges:
        assert len(a)==16
        assert torch.equal(torch.from_numpy(b),torch.arange(16).reshape(4,4).flip(-1).flatten())
    assert maps.shape==(8,1,4,4) and torch.isfinite(maps).all()


def test_transport_uses_other_frame_teacher_not_own_box():
    base=torch.eye(16).T.reshape(16,4,4);features=base[None].repeat(8,1,1,1)
    boxes=torch.tensor([[.125,.125,.1,.1]if i%2==0 else[.875,.875,.1,.1]for i in range(8)])
    maps,_,_=transport(features,boxes,torch.ones(8,2),4)
    assert maps[0,0,-1,-1]>0 and maps[0,0,0,0]==0
    assert maps[1,0,0,0]>0 and maps[1,0,-1,-1]==0
