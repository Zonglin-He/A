"""Preserve values AND layouts required by SDPA backward in cached replay."""
from contextlib import contextmanager
from unittest.mock import patch
import torch
from vg_tta import desta3d_v2_output_anchor_checkpoint_v4 as checkpointed


def aligned_bias(mask, multiple=8):
    if mask is None or mask.ndim != 4:
        return mask
    # Only physical storage is padded; semantic shape/support and values remain.
    width=mask.shape[-1]
    padded=(width+multiple-1)//multiple*multiple
    result=torch.empty((*mask.shape[:-1],padded),dtype=mask.dtype,device=mask.device)[...,:width]
    result.copy_(mask)
    assert all(s % multiple == 0 for s in result.stride()[:-1])
    return result


def layout_storage_bytes(tensor):
    span=1
    for stride,size in sorted(zip(tensor.stride(),tensor.shape)):
        if size <= 1:continue
        if stride < span:return None  # overlapping/expanded views stay in place
        span += (size-1)*stride
    return span*tensor.element_size()


def copy_layout(tensor,device):
    result=torch.empty_strided(tensor.shape,tensor.stride(),dtype=tensor.dtype,device=device)
    result.copy_(tensor)
    return result


@contextmanager
def activation_offload(model, *, limit_bytes=8*2**30, minimum_bytes=2**20):
    weights={p.untyped_storage().data_ptr() for p in model.parameters()}
    info={'bytes_offloaded':0,'tensors_offloaded':0,'limit_bytes':limit_bytes,
          'layout_preserved':True,'noncontiguous_offloaded':0}
    def pack(tensor):
        size=layout_storage_bytes(tensor)
        if (tensor.device.type=='cuda' and size is not None and size>=minimum_bytes
                and tensor.untyped_storage().data_ptr() not in weights
                and info['bytes_offloaded']+size<=limit_bytes):
            info['bytes_offloaded']+=size;info['tensors_offloaded']+=1
            info['noncontiguous_offloaded']+=int(not tensor.is_contiguous())
            return tensor.device,copy_layout(tensor.detach(),'cpu'),True
        return tensor.device,tensor.detach(),False
    def unpack(saved):
        device,tensor,offloaded=saved
        return copy_layout(tensor,device) if offloaded else tensor
    with torch.autograd.graph.saved_tensors_hooks(pack,unpack):yield info


def replay_branch(*args,**kwargs):
    import model.ptd_generation as pg
    original=pg._run_language_model
    records=[]
    def lm(*a,**kw):
        mask=kw.get('attention_mask')
        if mask is not None:
            out=aligned_bias(mask)
            records.append({'shape':list(mask.shape),'original_stride':list(mask.stride()),
                            'aligned_stride':list(out.stride()),'values_equal':bool(torch.equal(mask,out))})
            assert records[-1]['values_equal']
            kw['attention_mask']=out
        return original(*a,**kw)
    with patch.object(pg,'_run_language_model',lm),patch.object(checkpointed,'activation_offload',activation_offload):
        logits,audit=checkpointed.replay_branch(*args,**kwargs)
    audit['bias_layout']=records
    return logits,audit
