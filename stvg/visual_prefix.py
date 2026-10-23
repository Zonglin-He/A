"""Exact native appearance prefix, only when frozen prior+geometry already exist.

Delegates to the original vis_encoder and input_proj, on original224 offset batches
under the same CUDA FP16 autocast. No text/motion/fusion/time/word head is evaluated.
Native Source/Global6 and complete geometry must be hash-bound cached inputs.
"""
import gc,torch
from .inputs import make_batch,offset_batch

@torch.no_grad()
def native_appearance_grid(model,frames,frame_ids,metadata):
    assert not model.training and not any(p.requires_grad for p in model.parameters())
    model.cpu();q=dict(metadata);q.update(height=frames.shape[1],width=frames.shape[2]);batch=make_batch(frames,frame_ids,q,model);batch['videos']=batch['videos'].to('cpu');model.vis_encoder.cuda();model.input_proj.cuda();out=[]
    for offset in [0,1]:
        view=offset_batch(batch,offset);videos=view['videos'].to('cuda')
        with torch.autocast('cuda',dtype=torch.float16):
            original,position=model.vis_encoder(videos)
            values,mask,durations=original.decompose();projected=model.input_proj(values)
        assert projected.shape[0]==len(frame_ids[offset::2]);out.append((projected.detach().cpu().clone(),(~mask).detach().cpu().clone()))
        del view,videos,original,position,values,mask,projected
    return out
