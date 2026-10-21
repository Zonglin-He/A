"""ROI ordering/coordinates, exact policy substitution, differentiability/reset."""
import numpy as np
import torch
from tools.layout_trial.reward import LayoutReward,descriptors,fit_layout,fit_direct,temperature_from_unlabeled
from stvg import policy


def feature():
    torch.manual_seed(7)
    return torch.randn(8,12,24,32)


def test_ordered_roi_self_match_and_same_frame_correspondence():
    f=feature();b=torch.tensor([[.5,.5,.5,.5]]*8,dtype=torch.float64)
    reward=LayoutReward(f,b)
    assert torch.allclose(reward(b[:,None]),torch.ones(8,1,dtype=torch.float64),atol=2e-7)
    shifted=b.clone();shifted[:,0]+=.2
    assert reward(shifted[:,None]).mean()<.9
    flipped=LayoutReward(f.flip(0),b)
    assert (flipped.target*reward.target).sum(-1).mean()<.5


def test_shared_sampling_support_and_ordering_distinguish_pooled_control():
    f=feature();b=torch.tensor([[[.5,.5,.5,.5]]]*8)
    z=descriptors(f,b,3);assert z.shape==(8,1,9,12)
    assert not torch.allclose(z,z.flip(-2))
    assert descriptors(f,b,1).shape==(8,1,1,12)
    # All-zero/outside ROIs are finite zeros, never a false perfect match.
    outside=torch.tensor([[[3.,3.,.1,.1]]]*8)
    assert descriptors(f,outside,3).count_nonzero()==0
    assert descriptors(torch.zeros_like(f),b,3).count_nonzero()==0


def test_roi_box_gradients_and_direct_state_reset():
    f=feature();b=torch.tensor([[.5,.5,.4,.4]]*8,dtype=torch.float64)
    reward=LayoutReward(f,b)
    x=(b+torch.tensor([.04,0.,0.,0.])).requires_grad_()
    g=torch.autograd.grad(reward.mean_loss(x,b).mean(),x)[0]
    assert torch.isfinite(g).all()and g.count_nonzero()>0
    class Replay:
        def __init__(self):
            self.state=torch.zeros(8,256,dtype=torch.float64,requires_grad=True)
            self.E=b+torch.tensor([.04,0.,0.,0.]);self.positions=list(range(8));self.decoder=torch.nn.Identity();self.backwards=0
        def check(self):pass
        def values(self):
            boxes=torch.cat((self.E[:,:2]+.05*self.state[:,:2],self.E[:,2:]*torch.exp(.05*self.state[:,2:4])),1)
            return dict(boxes=boxes,native=boxes)
    rp=Replay();result=fit_direct(rp,b.numpy(),dict(lr=.01,steps=3),reward)
    assert result['gradient_nonzero']and rp.state.count_nonzero()==0
    rp=Replay();out=fit_layout(rp,b.numpy(),'synthetic-layout',41,dict(scales=[.05,.25],samples=32,tau=.03,lr=.01,steps=3),reward)
    assert out['actual_updates']==3 and rp.state.count_nonzero()==0


def test_layout_feedback_preserves_exact_candidates_and_uses_no_geometric_weights():
    f=feature();b=torch.tensor([[.5,.5,.4,.4]]*8,dtype=torch.float64);reward=LayoutReward(f,b)
    mu=policy.chart(b);d=policy.diagonal(b)
    old=policy.feedback(mu,d,b,'same',41,0,[.05,.25],.1)
    new=reward.feedback(mu,d,b,'same',41,0,[.05,.25],.1)
    assert torch.equal(old['actions'],new['actions'])and torch.equal(old['eps'],new['eps'])
    assert torch.equal(new['weights'],torch.softmax(reward(new['boxes'])/.1,-1))
    other=reward.feedback(mu,d,b+10,'same',41,0,[.05,.25],.1)
    assert torch.equal(new['weights'],other['weights'])
    assert not new['weights'].requires_grad


def test_temperature_calibration_is_unlabeled_and_global():
    torch.manual_seed(3);r=torch.randn(128,8,32,dtype=torch.float64)*.01;c=torch.randn_like(r)
    a=temperature_from_unlabeled(r,c);b=temperature_from_unlabeled(r.flip(0),c.flip(0))
    assert a['tau']==b['tau']and not a['target_GT_used']


def test_pilot_waits_for_current_full_before_features_or_adaptation(tmp_path,monkeypatch):
    import pytest,json
    from tools import complete_tubedetr_layout_trial as q
    (tmp_path/'runtime').mkdir()
    monkeypatch.setattr(q,'ROOT',tmp_path);monkeypatch.setattr(q,'OUT',tmp_path/'out');monkeypatch.setattr(q,'PUBLIC',tmp_path/'public')
    monkeypatch.setattr(q,'prepare',lambda:{})
    monkeypatch.setattr(q,'ready',lambda:False)
    monkeypatch.setattr(q,'launch',lambda *a:pytest.fail('Pilot launched before full terminal seal'))
    q.main(False)
    state=json.loads((q.OUT/'status.json').read_text())
    assert state['new_GPU_work_started']is False and state['adaptation_queries']==0


def test_layout_order_matters_even_when_pooled_content_is_identical():
    # Two ROIs contain the same nine channel vectors in reverse spatial order.
    f=torch.zeros(1,9,6,12)
    for y in range(3):
        for x in range(3):
            k=3*y+x;f[0,k,y*2:y*2+2,x*2:x*2+2]=1
            f[0,8-k,y*2:y*2+2,6+x*2:6+x*2+2]=1
    teacher=torch.tensor([[.25,.5,.5,1.]])
    other=torch.tensor([[[.75,.5,.5,1.]]])
    assert LayoutReward(f,teacher,1)(other).item()>.999
    assert LayoutReward(f,teacher,3)(other).item()<.2


def test_teacher_feature_hook_is_one_call_and_restores_original_interface():
    from tools.layout_trial.features import capture
    class Model(torch.nn.Module):
        def __init__(self):super().__init__();self.pixel_decoder=torch.nn.Identity();self.calls=0
        def infer(self):
            self.calls+=1;self.pixel_decoder(torch.ones(8,4,6,12))
            return {'pred_boxes':torch.full((1,8,2,4),.2)}
    class Teacher:
        def __init__(self):self.model=Model()
        def __call__(self,frames,caption):
            out=self.model.infer()
            return dict(boxes=out['pred_boxes'][0,:,0],selected_query=0,native_query_scores=torch.tensor([.8,.2]),input_shape=[8,3,24,48])
    teacher=Teacher();original=teacher.model.infer
    result=capture(teacher,np.zeros((8,24,48,3),np.uint8),'synthetic')
    assert teacher.model.calls==1 and result['features'].shape==(8,4,6,12)
    assert torch.equal(result['boxes'],torch.full((8,4),.2))
    assert teacher.model.infer==original and not teacher.model.pixel_decoder._forward_hooks


def test_new_priority_gate_does_not_wait_for_deferred_HC_or_spatial_results(tmp_path,monkeypatch):
    import json
    from tools import complete_tubedetr_layout_trial as q
    monkeypatch.setattr(q,'ROOT',tmp_path)
    path=tmp_path/'results/tubedetr_vid_development/final_seal.json';path.parent.mkdir(parents=True)
    path.write_text(json.dumps(dict(status='complete',files={})))
    (tmp_path/'runtime').mkdir();(tmp_path/'runtime/tubedetr_active_jobs.json').write_text(json.dumps(dict(jobs={})))
    assert q.ready()is True
