"""Bounded exact memoization of frozen, text-independent visual encoders.

Keys hash the actual normalized tensor bytes, masks and durations, not a parent
name or an approximate frame list. No text-dependent fusion/decoder is cached.
The cache lives in one worker and is cleared when the Source instance changes.
"""
from collections import OrderedDict
from contextlib import contextmanager
import hashlib
import time
import torch


def signature(value, digest):
    if torch.is_tensor(value):
        x = value.detach().contiguous().cpu()
        digest.update(str((str(x.dtype), tuple(x.shape))).encode())
        digest.update(x.view(torch.uint8).numpy().tobytes())
    elif hasattr(value, 'tensors') and hasattr(value, 'mask'):
        signature((value.tensors, value.mask, value.durations), digest)
    elif isinstance(value, dict):
        for key, item in value.items():
            digest.update(str(key).encode()); signature(item, digest)
    elif isinstance(value, (list, tuple)):
        digest.update(type(value).__name__.encode())
        for item in value: signature(item, digest)
    elif value is None or isinstance(value, (str, int, float, bool)):
        digest.update(repr(value).encode())
    else:
        raise TypeError('Unsupported visual-cache input: '+str(type(value)))


def transfer(value, device, clone=False):
    if torch.is_tensor(value):
        x = value.detach().to(device)
        return x.clone() if clone else x
    if hasattr(value, 'tensors') and hasattr(value, 'mask') and hasattr(value, 'durations'):
        return type(value)(transfer(value.tensors, device, clone),
                           transfer(value.mask, device, clone), list(value.durations))
    if isinstance(value, dict): return type(value)((k, transfer(v, device, clone))for k,v in value.items())
    if isinstance(value, tuple): return tuple(transfer(v, device, clone)for v in value)
    if isinstance(value, list): return [transfer(v, device, clone)for v in value]
    if value is None or isinstance(value, (str, int, float, bool)): return value
    raise TypeError('Unsupported visual-cache output: '+str(type(value)))


def nbytes(value):
    if torch.is_tensor(value): return value.numel()*value.element_size()
    if hasattr(value, 'tensors') and hasattr(value, 'mask'): return nbytes((value.tensors,value.mask))
    if isinstance(value, dict): return sum(nbytes(v)for v in value.values())
    if isinstance(value, (list, tuple)): return sum(nbytes(v)for v in value)
    return 0


class DeviceDict(dict):
    """Materialize original dictionary values on the requested device on access.

    Video-Swin returns four layers but this frozen pipeline reads only '3'.
    All keys remain available with identical tensor bytes; unused layers do not
    consume device transfers. Every returned tensor is a private query clone.
    """
    def __init__(self, source, device):
        super().__init__(source);self.device=device;self.ready={}
    def __getitem__(self,key):
        if key not in self.ready:self.ready[key]=transfer(super().__getitem__(key),self.device,clone=True)
        return self.ready[key]
    def get(self,key,default=None):return self[key]if key in self else default
    def items(self):return ((key,self[key])for key in self.keys())
    def values(self):return (self[key]for key in self.keys())
    def copy(self):return {key:self[key]for key in self.keys()}


class VisualCache:
    def __init__(self, max_bytes=2048*2**20):
        self.max_bytes=int(max_bytes);self.values=OrderedDict();self.model=None
        self.hits=0;self.misses=0;self.bytes=0;self.seconds=0.
    def stats(self):
        return dict(hits=self.hits,misses=self.misses,entries=len(self.values),host_bytes=self.bytes,key_and_transfer_seconds=self.seconds)
    @contextmanager
    def context(self, source, frames=None, ids=None):
        if self.model is not source:
            self.values.clear();self.bytes=0;self.model=source
        saved=[]
        namespace=hashlib.sha256()
        signature((list(ids)if ids is not None else None, torch.as_tensor(frames)if frames is not None else None),namespace)
        namespace=namespace.hexdigest()
        try:
            for name in ('vis_encoder','vid'):
                module=getattr(source,name);original=module.forward
                def cached(*args,_name=name,_module=module,_original=original,**kwargs):
                    assert not _module.training and not torch.is_grad_enabled()
                    start=time.perf_counter();h=hashlib.sha256((_name+namespace+str(torch.is_autocast_enabled())+str(torch.get_autocast_dtype('cuda'))).encode());signature((args,kwargs),h);key=h.hexdigest()
                    first=args[0];device=first.device if torch.is_tensor(first)else first.tensors.device
                    if key in self.values:
                        self.hits+=1;entry=self.values.pop(key);self.values[key]=entry
                        result=DeviceDict(entry,device)if _name=='vid'and isinstance(entry,dict)else transfer(entry,device,clone=True);self.seconds+=time.perf_counter()-start;return result
                    self.seconds+=time.perf_counter()-start;self.misses+=1;result=_original(*args,**kwargs)
                    start=time.perf_counter();entry=transfer(result,'cpu',clone=True);size=nbytes(entry)
                    if size<=self.max_bytes:
                        while self.values and self.bytes+size>self.max_bytes:
                            _,old=self.values.popitem(last=False);self.bytes-=nbytes(old)
                        self.values[key]=entry;self.bytes+=size
                    self.seconds+=time.perf_counter()-start;return result
                saved.append((module,original));module.forward=cached
            yield
        finally:
            for module,original in reversed(saved):module.forward=original
