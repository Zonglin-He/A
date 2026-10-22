"""Global6: exact frozen head, per-layer two-offset hull and physical MBR."""
import time
import numpy as np
import torch
from .temporal import probabilities,mass_table,final_mbr

@torch.no_grad()
def read_logits(head,hidden,saved):
    assert len(hidden)==len(saved)==2 and not head.training and not any(p.requires_grad for p in head.parameters())
    device=next(head.parameters()).device;out=[];transfer=compute=0.
    sync=lambda:torch.cuda.synchronize(device)if device.type=='cuda'else None
    for h,z in zip(hidden,saved):
        assert h.dtype==torch.float32 and h.ndim==4 and h.shape[0]==6 and h.shape[1]==1 and h.shape[-1]==256
        tick=time.perf_counter();h=h.to(device);sync();transfer+=time.perf_counter()-tick
        tick=time.perf_counter()
        with torch.autocast(device.type,enabled=False):a=head(h)
        sync();compute+=time.perf_counter()-tick;tick=time.perf_counter();a=a.cpu();sync();transfer+=time.perf_counter()-tick
        assert torch.equal(a[-1],z),'Frozen full-batch head failed bitwise final-layer recovery'
        out.append(a)
    return out,dict(head_calls=2,head_seconds=compute,hidden_transfer_seconds=transfer)

def posterior(records,layer_logits,frame_ids):
    tables=[]
    for layer in range(6):
        ids,q=mass_table(probabilities(records,[z[layer]for z in layer_logits]))
        if ids!=list(frame_ids):raise ValueError('Physical offset grid differs from the request')
        tables.append(q)
    result=np.mean(tables,axis=0)
    if not np.isfinite(result).all()or abs(result.sum()-1)>1e-12:raise ValueError('Invalid Global6 posterior')
    return result
