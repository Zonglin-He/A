"""GT-controlled coordinate-capacity probe, NEVER an unlabeled TTA method.
Same cached q0 + source weights, real final-linear SGD/backward. No GT file I/O.
"""
from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'round7'))
from regression_adapter import layers,raw_boxes,episode,parameter_sha,PARAMETER_NAMES

def exact_fit_same_head(model,cache,target6,mask6):
    source=cache.boxes_raw6.detach().clone();q=cache.query_index
    target=torch.as_tensor(target6,device=source.device,dtype=source.dtype).detach().clone()
    mask=torch.as_tensor(mask6,device=source.device,dtype=source.dtype).detach().clone()
    assert target.shape==mask.shape==(6,) and torch.isfinite(target).all() and (target[3:]>0).all()
    assert mask.tolist() in ([1.,1.,1.,0.,0.,0.],[0.,0.,0.,1.,1.,1.],[1.,1.,1.,1.,1.,1.])
    assert parameter_sha(model)==cache.selected_parameter_sha256
    a,b=layers(model);before_head=parameter_sha(model)
    expected=torch.where(mask>0,target,source[0,q]);active=[]
    knorm={}
    if mask[0]>0:
        h=cache.center_features[0,:,q];k=float(h.square().sum())+1.;knorm['center']=k;active.append({'params':[a.weight,a.bias],'lr':1/k})
    if mask[3]>0:
        h=cache.size_features[0,:,q];k=float(h.square().sum())+1.;knorm['size']=k;active.append({'params':[b.weight,b.bias],'lr':1/k})
    original={n:p.detach().clone() for n,p in model.named_parameters() if n in PARAMETER_NAMES}
    with episode(model):
        initial=raw_boxes(model,cache);assert torch.equal(initial,source)
        opt=torch.optim.SGD(active,momentum=0,weight_decay=0);opt.zero_grad(set_to_none=True)
        # Separate .5sum squared losses; never divide by6 or share maxnorm LR.
        loss=source.new_tensor(0.)
        if mask[0]>0:loss=loss+.5*(initial[0,q,:3]-target[:3]).square().sum()
        if mask[3]>0:loss=loss+.5*(initial[0,q,3:]-target[3:]).square().sum()
        assert torch.isfinite(loss);loss.backward()
        grads={n:None if p.grad is None else float(p.grad.norm()) for n,p in model.named_parameters() if n in PARAMETER_NAMES}
        assert all(v is None or torch.isfinite(torch.tensor(v)) for v in grads.values())
        opt.step()
        with torch.no_grad():updated=raw_boxes(model,cache).detach().clone()
        pd={n:float((dict(model.named_parameters())[n].detach()-v).abs().max()) for n,v in original.items()}
        residual=(updated[0,q]-expected).detach();supervised=residual[mask>0]
        inactive_bitexact=bool(torch.equal(updated[0,q][mask==0],source[0,q][mask==0]))
        other=torch.ones(source.shape[1],device=source.device,dtype=torch.bool);other[q]=False
        result={'boxes_raw6':updated,'q_source':q,'coordinate_residual6':residual.cpu().tolist(),'supervised_max_abs_coordinate_residual_m':float(supervised.abs().max()),'inactive_coordinates_bitexact_source':inactive_bitexact,'separate_head_feature_kernel':knorm,'actual_head_learning_rates':{k:1/v for k,v in knorm.items()},'squared_loss_before':float(loss.detach()),'parameter_gradient_norm_by_name':grads,'parameter_delta_max_by_name':pd,'real_backward_and_optimizer_step_executed':True,'observed_parameter_change':any(v>0 for v in pd.values()),'other_query_max_abs_change':float((updated[0,other]-source[0,other]).abs().max()) if other.any() else 0.,'no_output_override_used':True,'target_components_mask':mask.cpu().tolist(),'capability_scope':'one cached fixed query coordinate fitting only, no GTyaw or instance identification; GTIoU is not an upperbound'}
    assert parameter_sha(model)==before_head
    result['selected_source_parameters_restored']=True
    return result
