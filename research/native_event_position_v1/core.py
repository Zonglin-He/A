"""Scoped reuse of original full-offset positional embeddings; frozen weights."""
from contextlib import contextmanager
import torch


@contextmanager
def original_positions(module,full_lengths,selected):
    """Native replay visits offset0 twice then offset1 twice. No state_dict edit."""
    assert len(full_lengths)==len(selected)==2
    with torch.no_grad():
        full=[module(int(n)).detach().clone()for n in full_lengths]
    calls=[]
    def hook(m,args,out):
        offset=len(calls)//2;assert offset<2 and len(args)==1
        ix=selected[offset];assert len(ix)==int(args[0]) and out.shape[0]==len(ix)
        value=full[offset][ix].contiguous();assert value.shape==out.shape and value.dtype==out.dtype
        calls.append(dict(offset=offset,length=len(ix),original_indices=list(ix),same_native_values=torch.equal(value,out)))
        return value
    h=module.register_forward_hook(hook)
    try:yield calls
    finally:h.remove()
    assert len(calls)==4
