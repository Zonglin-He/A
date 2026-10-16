"""Synthetic positive and corruption controls for a skipped-gradient reader."""
from pathlib import Path
import copy
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def contracts():
    import torch
    from scripts import audit_stvg_spatial_lite_fit002 as audit
    from scripts import audit_stvg_spatial_lite_fit_v1 as original
    from scripts.stvg_spatial_lite_fit_v1 import _loss_per_anchor
    from vg_tta.decota_spatial_opd_tunable_v1 import mean_coordinates,rollout
    from vg_tta.tastvg_decota_critic_p0_v1 import aligned_iou
    assert not torch.cuda.is_initialized()
    boxes=torch.tensor([[.2,.2,.02,.02]],dtype=torch.float32)
    evidence=torch.tensor([[.9,.9,.01,.01]],dtype=torch.float32)
    cfg=dict(sigma=.025,tau=.1);key='synthetic-flat-reader002'
    mean=mean_coordinates(boxes).requires_grad_()
    p=rollout(mean,evidence,key,0,config=cfg)
    assert not p['informative']
    per=_loss_per_anchor(mean,p,cfg['sigma'])
    row=dict(boxes_before=boxes,rollout=p,per_anchor_loss=per.detach(),loss=float(per.mean().detach()),mean_gradient=None,reference_iou_before=aligned_iou(boxes,evidence),anchor_weights=torch.ones(1))
    result=audit._rollout_and_loss(row,evidence.numpy(),cfg,[0],key,0)
    assert result['informative']is False
    n=1
    for module,case in [(original,row),(audit,dict(row,mean_gradient=torch.zeros(1,4))),(audit,dict(row,loss=row['loss']+1))]:
        try:module._rollout_and_loss(case,evidence.numpy(),cfg,[0],key,0)
        except AssertionError:n+=1
        else:raise AssertionError('corruption or original missing-gradient error was not detected')
    # An informative score still uses the original, unchanged 8e-5 test.
    evidence=boxes.clone()
    p=rollout(mean,evidence,key,0,config=cfg);assert p['informative']
    per=_loss_per_anchor(mean,p,cfg['sigma']);grad=torch.autograd.grad(per.mean(),mean)[0]
    row=dict(row,rollout=p,per_anchor_loss=per.detach(),loss=float(per.mean().detach()),mean_gradient=grad,reference_iou_before=aligned_iou(boxes,evidence))
    assert audit._rollout_and_loss(row,evidence.numpy(),cfg,[0],key,0)['informative']
    n+=1
    for case in [dict(row,mean_gradient=None),dict(row,mean_gradient=grad+1)]:
        try:audit._rollout_and_loss(case,evidence.numpy(),cfg,[0],key,0)
        except AssertionError:n+=1
        else:raise AssertionError('informative gradient corruption was accepted')
    assert not torch.cuda.is_initialized()
    return dict(status='pass',count=n,CPU_only=True,GT_read=False,GPU_used=False,actual_model_called=False,original_failure_reproduced=True,gradient_threshold_unchanged=True)


if __name__=='__main__':print(contracts())
