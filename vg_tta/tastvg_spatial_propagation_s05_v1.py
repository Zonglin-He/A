"""GT-blind fixed prototype/nearest-reference propagation from frozen H_app.

This is the user-specified heuristic, not a reproduction of a trained RVOS memory.
Sparse references retain their original pixel-resolution boxes. Only unobserved
frames receive thresholded affinity boxes; the S0 optimizer remains untouched.
"""
import numpy as np
import torch
from torch.nn import functional as F
from vg_tta.tastvg_spatial_expansion_s0_v1 import mask_boxes


def affinity(h, reference, foreground):
    pos=reference[foreground];neg=reference[~foreground]
    assert len(pos) and len(neg)
    z=F.normalize(h,dim=-1);pp=F.normalize(pos.mean(0),dim=0);pn=F.normalize(neg.mean(0),dim=0)
    global_e=z@pp-z@pn
    bank=F.normalize(pos,dim=-1)
    local_e=torch.cat([(q@bank.T).max(-1).values for q in z.reshape(-1,z.shape[-1]).split(2048)]).reshape(h.shape[:-1])
    return .5*global_e+.5*local_e


@torch.no_grad()
def propagate(data,expert):
    h,w=map(int,data['views'][0]['info']['fea_map_size']);n=h*w;t=len(data['frame_ids'])
    assert all(tuple(v['info']['fea_map_size'])==(h,w) for v in data['views'])
    # A single unpadded video is resized anisotropically by native make_batch.
    assert all(not v['info']['encoded_mask'][:,:n].any() for v in data['views'])
    tokens=torch.stack([data['views'][i%2]['H'][:n,i//2,:].detach().cpu().float() for i in range(t)])
    pos=list(expert['positions']);valid,boxes=mask_boxes(expert['masks'],pos,t)
    result=dict(valid=valid,boxes=boxes,positions=pos,parent=expert['parent'],condition=expert['condition'],pixel_sha256=expert['pixel_sha256'],GT_read=False)
    diag=dict(grid=[h,w],frames=t,reference_positions=len(pos),reference_tokens=len(pos)*n,original_nonempty_references=int(valid.sum()),foreground_tokens=0,background_tokens=0,propagated_frames=0,threshold=None,reason='no_masks')
    if expert['masks'] is not None:
        occupancy=F.interpolate(torch.as_tensor(expert['masks'].copy()).float()[:,None],size=(h,w),mode='area')[:,0].reshape(len(pos),n)
        fg=occupancy>=.5;ref=tokens[pos].reshape(-1,tokens.shape[-1]);labels=fg.reshape(-1)
        diag.update(foreground_tokens=int(labels.sum()),background_tokens=int((~labels).sum()),quantized_nonempty_references=int(fg.any(-1).sum()))
        result['reference_occupancy']=occupancy
        if labels.any() and (~labels).any():
            evidence=affinity(tokens,ref,labels)
            ref_e=evidence[pos].reshape(-1)
            # Fixed class-balanced midpoint, learned only from existing references.
            threshold=(ref_e[labels].mean()+ref_e[~labels].mean())/2
            masks=(evidence>=threshold).reshape(t,h,w).numpy()
            dense_valid,dense_boxes=mask_boxes(masks,list(range(t)),t)
            unseen=np.ones(t,bool);unseen[pos]=False
            valid[unseen]=dense_valid[unseen];boxes[unseen]=dense_boxes[unseen]
            diag.update(reason='propagated',threshold=float(threshold),foreground_score_mean=float(ref_e[labels].mean()),background_score_mean=float(ref_e[~labels].mean()),propagated_frames=int((valid&unseen).sum()),unobserved_frames=int(unseen.sum()),foreground_fraction=float(masks[unseen].mean()) if unseen.any() else None)
            result['evidence']=evidence;result['propagated_masks']=masks
        else:diag['reason']='missing_foreground_or_background_after_projection'
    result['diagnostics']=diag
    return result
