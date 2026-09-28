from types import SimpleNamespace
import pytest
import torch
from torch import nn
from vg_tta.desta3d_v2_fp32_task_head import task_head_fp32
from scripts.ptd_8b_teacher_feasibility_v1 import joint_loss


class Core(nn.Module):
    def __init__(self):
        super().__init__();self.hidden=nn.Parameter(torch.randn(1,72,8).bfloat16())
    def forward(self,**kw):return SimpleNamespace(last_hidden_state=self.hidden)


class Toy(nn.Module):
    def __init__(self):
        super().__init__();self.model=Core();self.lm_head=nn.Linear(8,23,bias=False).bfloat16().requires_grad_(False)


def data():
    labels=torch.arange(72).remainder(23).reshape(1,-1);labels[0,0]=-100;labels[0,4]=-100
    return {'input_ids':torch.zeros_like(labels),'labels':labels,'ptd_prefix_lengths':torch.tensor([39])}


def test_disabled_exact_original_and_restore():
    torch.manual_seed(73);m=Toy();d=data();a,astat=joint_loss(m,d);ga=torch.autograd.grad(a,m.model.hidden)[0]
    with task_head_fp32(m,enabled=False):b,bst=joint_loss(m,d);gb=torch.autograd.grad(b,m.model.hidden)[0]
    assert torch.equal(a,b) and astat==bst and torch.equal(ga,gb)
    before=m.lm_head.weight.clone()
    with pytest.raises(RuntimeError):
        with task_head_fp32(m):
            assert m.lm_head(m.model.hidden).dtype==torch.float32
            raise RuntimeError('controlled')
    assert 'forward' not in m.lm_head.__dict__ and torch.equal(before,m.lm_head.weight)
    assert m.lm_head(m.model.hidden).dtype==torch.bfloat16


def test_original_chunk_loss_and_gradient_full_vocab():
    torch.manual_seed(31);m=Toy();d=data();labels=d['labels'];keep=torch.nonzero(labels[0,1:]!=-100).flatten();targets=labels[0,keep+1]
    with task_head_fp32(m):loss,stats=joint_loss(m,d);g=torch.autograd.grad(loss,m.model.hidden)[0]
    logits=torch.nn.functional.linear(m.model.hidden[0,keep].float(),m.lm_head.weight.float())
    each=torch.nn.functional.cross_entropy(logits,targets,reduction='none');ref=each.mean();rg=torch.autograd.grad(ref,m.model.hidden)[0]
    assert torch.allclose(loss,ref,atol=5e-7,rtol=0) and torch.allclose(g,rg,atol=2e-4,rtol=0)
    ntp=(keep+1)<d['ptd_prefix_lengths'][0]
    assert stats['ntp_count']==int(ntp.sum()) and stats['mtp_count']==int((~ntp).sum())
    assert abs(stats['ntp_loss']-float(each[ntp].mean()))<5e-7
    assert abs(stats['mtp_loss']-float(each[~ntp].mean()))<5e-7
    assert torch.isfinite(g).all() and g.abs().sum()>0 and m.lm_head.weight.grad is None
    assert torch.count_nonzero(g[0,-1])==0


def test_requires_frozen_output_head():
    m=Toy();m.lm_head.weight.requires_grad_(True)
    with pytest.raises(AssertionError):
        with task_head_fp32(m):pass
