import torch
from tools.boxfixed_followup.teacher_calibration import fit_query,vector


def test_zero_teacher_gain_does_not_invent_skill():
    X=torch.randn(16,4,dtype=torch.float64);y=torch.zeros(16,dtype=torch.float64);groups=torch.arange(8).repeat_interleave(2)
    m=fit_query(X,y,groups);assert not m['valid'] and m['beta'].eq(0).all()


def test_fixed_ridge_can_predict_heldout_teacher_target_without_dataset_labels():
    x=torch.linspace(-1,1,56,dtype=torch.float64);X=torch.stack((x,x.square(),x.sin(),x.cos()),-1);y=.2*x;groups=torch.arange(8).repeat_interleave(7)
    m=fit_query(X,y,groups);assert m['valid'] and m['teacher_cv_mse']<m['teacher_zero_mse']


def test_source_is_zero_feature_reference_and_no_target_pseudo_box_in_predictors():
    s=torch.tensor([.4,.5,.2,.3],dtype=torch.float64);z=torch.randn(256,dtype=torch.float64);f=torch.randn_like(z);b=torch.randn_like(z)
    assert vector(s,s,z,z,f,b).eq(0).all()


def test_held_observation_cannot_reenter_training_as_a_source_anchor():
    pairs=[(s,t)for t in range(8)for s in range(8)if s!=t];source=torch.tensor([s for s,t in pairs]);target=torch.tensor([t for s,t in pairs])
    X=torch.randn(56,4,dtype=torch.float64,generator=torch.Generator().manual_seed(41));y=.2*X[:,0]
    a=fit_query(X,y,target,source);changed=y.clone();changed[source==0]+=100
    b=fit_query(X,changed,target,source)
    assert torch.equal(a['heldout_predictions'][target==0],b['heldout_predictions'][target==0])
