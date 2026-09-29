"""User-specified symmetric foreground/background affinity + soft box moments.

Evidence computed once from native H0, temperature1, no evidence threshold.
Reference mask-to-token projection retains fixed S0.5 area>=.5 semantics.
"""
import numpy as np
import torch
from torch.nn import functional as F
from vg_tta.tastvg_spatial_expansion_s0_v1 import mask_boxes,alignment as sparse_alignment
from vg_tta.tastvg_causal_round2_v1 import forward,masks
from vg_tta.tastvg_evidence_attack_v1 import norm
from methods.decota_final_simplified_v1.tensors import detached


def foreground_probability(h,reference,foreground):
    pos,neg=reference[foreground],reference[~foreground]
    assert len(pos) and len(neg)
    z=F.normalize(h,dim=-1);scores=[]
    for bank in [pos,neg]:
        global_e=z@F.normalize(bank.mean(0),dim=0)
        unit=F.normalize(bank,dim=-1)
        local_e=torch.cat([(q@unit.T).max(-1).values for q in z.reshape(-1,z.shape[-1]).split(2048)]).reshape(h.shape[:-1])
        scores.append(.5*global_e+.5*local_e)
    logits=torch.stack(scores,-1)
    return logits.softmax(-1)[...,0],logits


def moment_boxes(probability,height,width):
    assert probability.shape[-1]==height*width
    yy,xx=torch.meshgrid((torch.arange(height,device=probability.device,dtype=probability.dtype)+.5)/height,(torch.arange(width,device=probability.device,dtype=probability.dtype)+.5)/width,indexing='ij')
    xy=torch.stack([xx.flatten(),yy.flatten()],-1)
    weights=probability/probability.sum(-1,keepdim=True)
    center=weights@xy
    variance=(weights[...,None]*(xy-center[...,None,:]).square()).sum(-2)
    size=(12*variance.clamp_min(0)).sqrt()
    lo=(center-size/2).clamp(0,1);hi=(center+size/2).clamp(0,1)
    return torch.cat([(lo+hi)/2,hi-lo],-1)


@torch.no_grad()
def propagate(data,expert):
    h,w=map(int,data['views'][0]['info']['fea_map_size']);n=h*w;t=len(data['frame_ids'])
    assert h>1 and w>1
    assert all(tuple(v['info']['fea_map_size'])==(h,w) for v in data['views'])
    assert all(not v['info']['encoded_mask'][:,:n].any() for v in data['views'])
    tokens=torch.stack([data['views'][i%2]['H'][:n,i//2,:].detach().cpu().float() for i in range(t)])
    pos=list(expert['positions']);valid,boxes=mask_boxes(expert['masks'],pos,t)
    result=dict(valid=valid,boxes=boxes,positions=pos,parent=expert['parent'],condition=expert['condition'],pixel_sha256=expert['pixel_sha256'],GT_read=False)
    diag=dict(grid=[h,w],frames=t,reference_positions=len(pos),reference_tokens=len(pos)*n,original_nonempty_references=int(valid.sum()),foreground_tokens=0,background_tokens=0,propagated_frames=0,reason='no_masks',temperature=1)
    if expert['masks'] is not None:
        occupancy=F.interpolate(torch.as_tensor(expert['masks'].copy()).float()[:,None],size=(h,w),mode='area')[:,0].reshape(len(pos),n)
        fg=occupancy>=.5;ref=tokens[pos].reshape(-1,tokens.shape[-1]);labels=fg.reshape(-1)
        diag.update(foreground_tokens=int(labels.sum()),background_tokens=int((~labels).sum()),quantized_nonempty_references=int(fg.any(-1).sum()))
        result['reference_occupancy']=occupancy
        if labels.any() and (~labels).any():
            prob,logits=foreground_probability(tokens,ref,labels);dense_boxes=moment_boxes(prob,h,w)
            unseen=np.ones(t,bool);unseen[pos]=False
            valid[unseen]=True;boxes[unseen]=dense_boxes.numpy()[unseen]
            diag.update(reason='propagated',propagated_frames=int(unseen.sum()),unobserved_frames=int(unseen.sum()),probability_min=float(prob.min()),probability_max=float(prob.max()),probability_mean=float(prob.mean()),mean_unobserved_box_area=float(dense_boxes[unseen,2:].prod(-1).mean()) if unseen.any() else None)
            result['probability']=prob;result['affinity_logits']=logits
        else:diag['reason']='missing_foreground_or_background_after_projection'
    result['diagnostics']=diag
    return result


def alignment(boxes,target,valid,l1_coef,giou_coef):
    # Exactly 1/T; undefined empty-reference targets are omitted, not invented.
    return sparse_alignment(boxes,target,valid,l1_coef,giou_coef)*valid.sum()/len(boxes)


def expand(model,data,expert):
    base=[v['H'].detach() for v in data['views']];fields=base
    allowed=masks(data,'S');visual=norm([h*m for h,m in zip(base,masks(data,'ST'))])
    target=torch.as_tensor(expert['boxes'],device=base[0].device,dtype=torch.float32)
    valid=torch.as_tensor(expert['valid'],device=base[0].device,dtype=torch.bool)
    coeff=[model.cfg.SOLVER.BBOX_COEF,model.cfg.SOLVER.GIOU_COEF]
    trajectory=[dict(prediction=detached(data['prediction'],'cpu'))];diagnostics=[];final=None
    for k in range(3):
        fields=[h.detach().requires_grad_(True) for h in fields]
        ev,boxes,pred=forward(model,data,fields)
        if k==0:assert torch.equal(boxes.detach(),data['prediction']['boxes'])
        loss=alignment(boxes,target,valid,*coeff);before=float(loss.detach())
        grad=torch.autograd.grad(loss,fields);gg=[g*m for g,m in zip(grad,allowed)];gn=norm(gg)
        assert torch.isfinite(gn) and torch.isfinite(loss)
        shifted=[h.detach()-.004*visual/gn*g.detach() if float(gn)>0 else h.detach().clone() for h,g in zip(fields,gg)]
        for h,z,m in zip(base,shifted,allowed):assert torch.equal(h[m==0],z[m==0])
        relative=float(norm([z-h for z,h in zip(shifted,base)])/visual);assert relative<=.004*(k+1)+1e-7
        step_size=float(norm([z-h.detach() for z,h in zip(shifted,fields)])/visual);assert step_size==0 or abs(step_size-.004)<1e-7
        del ev,boxes,pred,loss,grad,gg,fields
        with torch.no_grad():
            final,boxes,pred=forward(model,data,shifted);after=float(alignment(boxes,target,valid,*coeff));assert np.isfinite(after)
        diagnostics.append(dict(step=k+1,loss_before=before,loss_after=after,gradient_norm=float(gn),relative_step=step_size,relative_to_native=relative,appearance_only_exact=True))
        trajectory.append(dict(prediction=detached(pred,'cpu')));fields=shifted
    return dict(trajectory=trajectory,diagnostics=diagnostics,coefficients=coeff,visual_norm=float(visual),valid_evidence_frames=int(valid.sum()),loss_denominator=len(target)),fields,final
