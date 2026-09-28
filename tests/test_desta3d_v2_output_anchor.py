import pytest
import torch
from vg_tta.desta3d_v2_output_anchor import output_kl, _selected_logits, replay_branch


def test_kl_support_teacher_detach_and_gradient():
    teacher=torch.randn(2,32,requires_grad=True)
    student=teacher.detach().clone().requires_grad_()
    assert output_kl(student,teacher).item()==0
    perturbed=(student+.1*torch.arange(32)).requires_grad_()
    loss=output_kl(perturbed,teacher)
    loss.backward()
    assert loss>0 and student.grad.norm()>0 and teacher.grad is None
    with pytest.raises(ValueError):output_kl(student,teacher[:1])


def test_exact_ptd_slot_and_vocabulary_slices():
    logits=torch.arange(2*6*20.).reshape(1,12,20).requires_grad_()
    ids={'ref_end':19,'ordered_time_tokens':[2,4,6]}
    kind,x=_selected_logits(logits,[2,4],ids,[1,3,5,7])
    assert kind=='coordinate' and x.shape==(2,4,4)
    assert torch.equal(x[1,0],logits[0,7,[1,3,5,7]])
    kind,y=_selected_logits(logits[:,:6],[19],ids,[1,3,5,7])
    assert kind=='time' and torch.equal(y[0],logits[0,1,[2,4,6]])
    (x.sum()+y.sum()).backward();assert logits.grad is not None


def test_inference_mode_and_missing_teacher_support_rejected():
    with torch.inference_mode(), pytest.raises(RuntimeError,match='inference_mode'):
        replay_branch(None,None,None,None,None,'event')
    with pytest.raises(ValueError,match='no branch'):
        replay_branch(None,{},None,None,{'branches':[]},'spatial',pg=object())
