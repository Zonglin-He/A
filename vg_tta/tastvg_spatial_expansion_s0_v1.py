"""Fixed appearance-only native tube trajectory; expert evidence is not final output."""
import numpy as np
import torch
from vg_tta.tastvg_causal_round2_v1 import forward,masks
from vg_tta.tastvg_evidence_attack_v1 import norm
from methods.decota_final_simplified_v1.tensors import detached


def mask_boxes(mask_arrays,positions,count):
    valid=np.zeros(count,dtype=bool);boxes=np.zeros((count,4),dtype=np.float32)
    if mask_arrays is None:return valid,boxes
    assert len(mask_arrays)==len(positions)
    for pos,mask in zip(positions,mask_arrays):
        assert mask.ndim==2
        yy,xx=np.nonzero(mask)
        if not len(xx):continue
        h,w=mask.shape;x0,x1=xx.min()/w,(xx.max()+1)/w;y0,y1=yy.min()/h,(yy.max()+1)/h
        boxes[pos]=[(x0+x1)/2,(y0+y1)/2,x1-x0,y1-y0];valid[pos]=True
    return valid,boxes


def alignment(boxes,target,valid,l1_coef,giou_coef):
    if not valid.any():return boxes.sum()*0
    p,q=boxes[valid],target[valid]
    a,b=p[:,:2]-p[:,2:]/2,p[:,:2]+p[:,2:]/2
    c,d=q[:,:2]-q[:,2:]/2,q[:,:2]+q[:,2:]/2
    inter=(torch.minimum(b,d)-torch.maximum(a,c)).clamp_min(0).prod(-1)
    union=(b-a).prod(-1)+(d-c).prod(-1)-inter
    enclosing=(torch.maximum(b,d)-torch.minimum(a,c)).prod(-1)
    giou=inter/union-(enclosing-union)/enclosing
    return l1_coef*(p-q).abs().sum()/len(p)+giou_coef*(1-giou).mean()


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
        relative=float(norm([z-h for z,h in zip(shifted,base)])/visual)
        assert relative <= .004*(k+1)+1e-7
        step_size=float(norm([z-h.detach() for z,h in zip(shifted,fields)])/visual)
        assert step_size==0 or abs(step_size-.004)<1e-7
        del ev,boxes,pred,loss,grad,gg,fields
        with torch.no_grad():
            final,boxes,pred=forward(model,data,shifted)
            after=float(alignment(boxes,target,valid,*coeff));assert np.isfinite(after)
        diagnostics.append(dict(step=k+1,loss_before=before,loss_after=after,gradient_norm=float(gn),relative_step=step_size,relative_to_native=relative,appearance_only_exact=True))
        trajectory.append(dict(prediction=detached(pred,'cpu')));fields=shifted
    return dict(trajectory=trajectory,diagnostics=diagnostics,coefficients=coeff,visual_norm=float(visual),valid_evidence_frames=int(valid.sum())),fields,final
