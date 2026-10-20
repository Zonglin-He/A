"""Do not retain deterministic native Swin CUDA masks between captures.

The original unwrapped compute_mask body and every native forward operation
are preserved. Active layer masks stay available to all their original blocks.
"""
from contextlib import contextmanager
import gc
import sys

import torch

from stvg._visual import frozen_forward as capture011


@contextmanager
def transient_native_masks(model):
    modules = {sys.modules[type(child).__module__] for child in model.vid.modules()
               if type(child).__name__ == 'BasicLayer'}
    assert modules and not model.training and not torch.is_grad_enabled()
    saved = {module: module.compute_mask for module in modules}
    assert all(hasattr(function, '__wrapped__') and hasattr(function, 'cache_clear')
               for function in saved.values())
    try:
        for module, original in saved.items():
            original.cache_clear()
            module.compute_mask = original.__wrapped__
        gc.collect()
        torch.cuda.empty_cache()
        yield
    finally:
        for module, original in saved.items():
            module.compute_mask = original
            original.cache_clear()
            assert original.cache_info().currsize == 0
        gc.collect()
        torch.cuda.empty_cache()


@torch.no_grad()
def frozen_forward(model, frames, row):
    with transient_native_masks(model):
        return capture011(model, frames, row)
