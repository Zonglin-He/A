"""Isolated task-loss head precision context; original native module restored."""
from contextlib import contextmanager
import types
import torch
from torch.nn import functional as F


@contextmanager
def task_head_fp32(model, *, enabled=True):
    """Keep the context alive through checkpoint backward. No weight mutation.

    Only the final projection is computed in FP32. Hidden-state gradients still
    cross the original input dtype conversion. No vocabulary slicing or caching.
    """
    if not enabled:
        yield
        return
    head=model.lm_head
    assert type(head) is torch.nn.Linear and head.bias is None
    assert not head.weight.requires_grad and head.weight.grad is None
    assert head.weight.dtype in (torch.bfloat16,torch.float32)
    assert not torch.is_autocast_enabled(), 'task head precision requires explicit non-autocast execution'
    if head.weight.is_cuda:
        assert not torch.backends.cuda.matmul.allow_tf32, 'FP32 probe must not use TF32'
    owned='forward' in head.__dict__; previous=head.__dict__.get('forward')
    pointer=head.weight.data_ptr();dtype=head.weight.dtype
    def fp32_forward(self,h):
        with torch.autocast(device_type=h.device.type,enabled=False):
            return F.linear(h.float(),self.weight.float(),None)
    head.forward=types.MethodType(fp32_forward,head)
    try:
        yield
    finally:
        if owned:head.forward=previous
        else:del head.forward
        assert head.weight.data_ptr()==pointer and head.weight.dtype==dtype
        assert not head.weight.requires_grad and head.weight.grad is None


def task_signals_fp32(model,processor,adapter,fields,data,*,gradients=True):
    from vg_tta.desta3d_v2_source_task_control import task_signals
    with task_head_fp32(model):
        return task_signals(model,processor,adapter,fields,data,gradients=gradients)
