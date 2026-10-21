import pytest
import torch
from tools.mask_reward import MaskAreaReward,box_overlap


def test_fractional_area_and_padding_are_exact():
    reward=MaskAreaReward(torch.ones(1,1,2,2,dtype=torch.float64),torch.ones(1,2),torch.zeros(1,0,dtype=torch.bool))
    b=torch.tensor([[[.5,.5,1.,1.],[.25,.5,.5,1.],[-1.,.5,1.,1.]]],dtype=torch.float64)
    assert reward.scores(b)['MaskDice'][0].tolist()==pytest.approx([1.,2/3,0.])
    padded=MaskAreaReward(torch.ones(1,1,2,2,dtype=torch.float64),torch.tensor([[.75,.5]]),torch.zeros(1,0,dtype=torch.bool))
    assert padded.scores(b[:,:1])['MaskDice'].item()==pytest.approx(1.)


def test_direct_has_same_continuous_area_gradient():
    reward=MaskAreaReward(torch.tensor([[[[.1,.7],[.8,.2]]]],dtype=torch.float64),torch.ones(1,2),torch.zeros(1,0,dtype=torch.bool))
    b=torch.tensor([[[.43,.39,.32,.24]]],dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(lambda x:reward.scores(x)['MaskDice'],(b,),eps=1e-6,atol=1e-6)


def test_duplicate_alternative_does_not_become_a_false_negative():
    masks=torch.ones(1,2,2,2,dtype=torch.float64);b=torch.tensor([[[.5,.5,.5,.5]]],dtype=torch.float64)
    overlap=box_overlap(b[:,0],b);assert overlap.item()==pytest.approx(1.)
    reward=MaskAreaReward(masks,torch.ones(1,2),overlap<.5)
    s=reward.scores(b);assert torch.equal(s['MaskDice'],s['MaskContrast'])
