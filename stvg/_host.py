"""Host-stage inactive frozen source modules during the original Video Swin call."""
from contextlib import contextmanager
import gc
import torch
from stvg.capture import frozen_forward as capture007


@contextmanager
def inactive_modules_on_host(model):
    device = next(model.parameters()).device
    others = [module for module in model.children() if module is not model.vid]
    assert not model.training and not any(p.requires_grad for p in model.parameters())
    active = [False]

    def release():
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()

    def before(module, args):
        assert not active[0]
        # Those exact modules have already produced the current visual tensors,
        # or will only run after Video Swin returns. No arithmetic is replaced.
        for child in others:
            child.to('cpu')
        active[0] = True
        release()

    def after(module, args, output):
        if active[0]:
            release()
            for child in others:
                child.to(device)
            active[0] = False

    pre = model.vid.register_forward_pre_hook(before)
    post = model.vid.register_forward_hook(after, always_call=True)
    try:
        yield
    finally:
        pre.remove(); post.remove()
        after(None, None, None)


def frozen_forward(model, frames, row):
    with inactive_modules_on_host(model):
        return capture007(model, frames, row)
