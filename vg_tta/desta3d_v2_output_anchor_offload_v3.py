"""Cycle-free, lossless, bounded saved-activation offload for native cached replay."""
from contextlib import contextmanager
import torch
from vg_tta.desta3d_v2_output_anchor import replay_branch as replay_original


@contextmanager
def activation_offload(model, *, limit_bytes=8*2**30, minimum_bytes=2**20):
    # Transposed frozen weights share storage and remain on the GPU. Avoid
    # copying them repeatedly into limited host RAM at every cached probe.
    weights={p.untyped_storage().data_ptr() for p in model.parameters()}
    info={'bytes_offloaded':0,'tensors_offloaded':0,'limit_bytes':limit_bytes}
    def pack(tensor):
        size=tensor.numel()*tensor.element_size()
        eligible=(tensor.device.type=='cuda' and size>=minimum_bytes and
                  tensor.untyped_storage().data_ptr() not in weights and
                  info['bytes_offloaded']+size<=limit_bytes)
        if eligible:
            info['bytes_offloaded']+=size;info['tensors_offloaded']+=1
            return tensor.device,tensor.detach().to('cpu'),True
        return tensor.device,tensor.detach(),False
    def unpack(saved):
        device,tensor,offloaded=saved
        return tensor.to(device) if offloaded else tensor
    with torch.autograd.graph.saved_tensors_hooks(pack,unpack):
        yield info


def replay_branch(*args,**kwargs):
    with activation_offload(args[0]) as info:
        logits,audit=replay_original(*args,**kwargs)
    audit['lossless_activation_offload']=dict(info)
    return logits,audit
