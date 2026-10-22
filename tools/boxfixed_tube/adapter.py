"""Native TubeDETR interfaces only; spatial BoxFixed implementation is upstream."""
import numpy as np
import torch
from stvg.tubedetr import native_forward,endpoint_posterior
from stvg.temporal import final_mbr
from .evidence import _valid_grid_shape


@torch.no_grad()
def capture(model,frames,frame_ids,caption):
    native=native_forward(model,frames,frame_ids,caption,224)
    args,kw=native['decoder_inputs'];memory=args[1].detach().float().cpu();pos=kw['pos'].detach().float().cpu();mask=kw['memory_key_padding_mask'].detach().bool().cpu()
    assert memory.shape==pos.shape and memory.shape[1:]==(len(frame_ids),256)and mask.shape==(len(frame_ids),len(memory))
    flags=pos.abs().sum(-1).any(-1);N=int(flags.sum());assert N>0 and flags[:N].all()and not flags[N:].any()
    shape=_valid_grid_shape([pos],[mask],N);valid=~mask[:,:N];features=memory[:N].transpose(0,1);features=torch.where(valid[:,:,None],features,torch.zeros_like(features))
    assert len(native['layers'])==6 and torch.equal(native['layers'][-1],native['logits'])
    q=np.mean([endpoint_posterior(z)for z in native['layers']],axis=0);assert abs(q.sum()-1)<1e-12
    interval=final_mbr(frame_ids,q)['interval']
    return dict(Source=native['source'],Native_interval=native['interval'],native_extent=native['extent'],Global6_interval=interval,frame_ids=list(frame_ids),features=features,valid=valid,grid_shape=shape,Global6_posterior=q,logits6=torch.stack(native['layers']),feature_interface='actual TubeDETR post-fusion img_memory spatial prefix; all native physical frames, excluding text; 256 channels',temporal_interface='six original native224 full-grid head logits; arithmetic endpoint-posterior mean; physical MBR; one native pass, no invented TA offsets/no448')
