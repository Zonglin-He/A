"""Stage already-computed frozen visual tensors on host during native Swin.

The original model forward and all arithmetic stay in their original order.
Only tensor residency and the lifetime of an unused ResNet output change.
"""
from contextlib import contextmanager
import gc
import inspect
import sys
import textwrap

import torch

from stvg._attention import frozen_forward as capture010
from stvg.capture import transfer


def residency_forward(original, namespace):
    source = textwrap.dedent(inspect.getsource(original))
    anchor = ("    with torch.no_grad():\n"
              "        vid_features_all = self.vid(videos.tensors, len(videos.tensors))\n")
    assert source.count(anchor) == 1
    assert source.count('vis_res_features') == 2
    staged = ("    assert not self.training and not torch.is_grad_enabled()\n"
              "    del vis_res_features\n"
              "    visual_device = vis_features.device\n"
              "    vis_outputs = vis_outputs.to('cpu')\n"
              "    vis_features = None\n"
              "    vis_mask = None\n"
              "    vis_pos_embed = _host_transfer(vis_pos_embed, 'cpu')\n"
              "    _host_gc.collect()\n"
              "    torch.cuda.empty_cache()\n"
              "    try:\n"
              "        with torch.no_grad():\n"
              "            vid_features_all = self.vid(videos.tensors, len(videos.tensors))\n"
              "    finally:\n"
              "        vis_outputs = vis_outputs.to(visual_device)\n"
              "        vis_features, vis_mask, vis_durations = vis_outputs.decompose()\n"
              "        vis_pos_embed = _host_transfer(vis_pos_embed, visual_device)\n")
    modified = source.replace(anchor, staged)
    # Removing the sole residency insertion reconstructs the original source.
    assert modified.replace(staged, anchor) == source
    local = dict(namespace)
    local.update(_host_transfer=transfer, _host_gc=gc)
    exec(compile(modified, __file__, 'exec'), local)
    return local['forward']


@contextmanager
def frozen_visual_host(model):
    cls = type(model)
    original = cls.forward
    assert not model.training and not any(p.requires_grad for p in model.parameters())
    try:
        cls.forward = residency_forward(original, vars(sys.modules[cls.__module__]))
        yield
    finally:
        cls.forward = original


def frozen_forward(model, frames, row):
    with frozen_visual_host(model):
        return capture010(model, frames, row)
