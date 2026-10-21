"""Head-only readout of sealed six-layer temporal states; no decoder or labels."""
import hashlib,importlib.util,time
from pathlib import Path
import numpy as np
import torch
from stvg.temporal import probabilities,mass_table,final_mbr
from stvg.spatial import SpatialGeometry,evidence_dp

def tensor_digest(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):
        h.update(k.encode());h.update(str(v.dtype).encode());h.update(str(tuple(v.shape)).encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def load_head(root,checkpoint):
    """Instantiate the original upstream class only, not the Source network."""
    path=Path(root)/'external/TA-STVG/models/net_utils.py'
    spec=importlib.util.spec_from_file_location('frozen_ta_temporal_net_utils',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    ckpt=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
    key='model_ema'if'model_ema'in ckpt else'model'
    state={k.removeprefix('temp_embed.'):v.clone()for k,v in ckpt[key].items()if k.startswith('temp_embed.')}
    assert set(state)=={'layers.0.weight','layers.0.bias','layers.1.weight','layers.1.bias'}
    head=module.MLP(256,256,2,2,dropout=.3).eval().requires_grad_(False)
    head.load_state_dict(state,strict=True);del ckpt
    return head,dict(checkpoint_state=key,weights_sha256=tensor_digest(state),parameters=sum(p.numel()for p in head.parameters()),upstream_class='external/TA-STVG/models/net_utils.py:MLP')

@torch.no_grad()
def head_logits(head,hidden,saved):
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

def layer_posteriors(records,logits,frame_ids):
    assert len(logits)==2 and all(z.shape[0]==6 for z in logits)
    tables=[]
    for layer in range(6):
        ids,q=mass_table(probabilities(records,[z[layer]for z in logits]))
        assert ids==list(frame_ids);tables.append(q)
    mean=np.mean(tables,axis=0)
    assert mean.dtype==np.float64 and np.isfinite(mean).all()and abs(mean.sum()-1)<1e-12
    return mean,np.stack(tables)

def fuse(global_q,conditional_q):
    assert global_q.shape==conditional_q.shape
    return .5*global_q+.5*conditional_q

def restore_geometry(state):
    assert state['metadata']['schema']=='boxfixed_geometry_v1'and state['metadata']['GT_read']is False
    g=SpatialGeometry.__new__(SpatialGeometry);g.__dict__.update(state);return g

def final_prediction(geometry,ids,q):
    interval=final_mbr(ids,q)['interval'];out=evidence_dp(geometry.for_interval(interval))
    return out['final'].numpy(),interval,out
