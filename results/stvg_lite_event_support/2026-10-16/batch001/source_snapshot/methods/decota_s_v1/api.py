"""Label-free inference API. No local residual, expert TTA, or label reader."""
import copy
import time
import numpy as np
import torch
from torchvision.ops import box_convert
from vg_tta.tg_spatial_tta_v1 import detached_tree, replay_spatial
from vg_tta.metrics import generalized_box_iou_aligned_cxcywh

VERSION='decota_s_v1'
DEFAULT_CONFIG=dict(lr=.1,steps=3,anchor=1e-4)
BACKTRACKS=(1.,.5,.25,.125)


def valid_boxes(b):
    return bool(torch.isfinite(b).all() and (b>=0).all() and (b<=1).all() and (b[...,2:]>0).all())


def fit_spatial(model,caches,base,pseudo,*,lr=.1,steps=3,anchor=1e-4):
    """Equal-frame loss: 5*sum-coordinate L1 + 2*(1-GIoU) + anchor.

    Parameters of model must already be frozen/eval. Gradients pass through
    its spatial decoder. Each call owns a new residual and optimizer.
    """
    started=time.perf_counter();delta=torch.zeros(256,device='cuda',requires_grad=True)
    audit=dict(lr=lr,steps=steps,anchor=anchor,weighting='equal_accepted_frames',
               trainable_parameters=256,gradient_calls=0,forward_calls=0,history=[],
               fallback=None,accepted_steps=0,pseudo_count=len(pseudo),GT_used=False)

    def finish(boxes,reason=None):
        torch.cuda.synchronize();audit.update(fallback=reason,seconds=time.perf_counter()-started,
             delta=(torch.zeros_like(delta) if reason else delta.detach()).cpu(),
             delta_norm=0. if reason else float(delta.detach().norm()))
        return boxes.detach().cpu().clone(),audit

    if not pseudo:return finish(base,'empty_supervision')
    positions=torch.tensor([p['position'] for p in pseudo],device='cuda')
    target=torch.tensor([p['box'] for p in pseudo],dtype=torch.float32,device='cuda')
    if not valid_boxes(target) or len(set(positions.tolist()))!=len(pseudo) or int(positions.min())<0 or int(positions.max())>=len(base):
        return finish(base,'invalid_pseudo_boxes_or_positions')
    opt=torch.optim.Adam([delta],lr=lr,eps=1e-8)

    def objective():
        audit['forward_calls']+=1
        boxes=replay_spatial(model,caches,delta,len(base))
        pred=boxes[positions]
        per_frame=5*(pred-target).abs().sum(-1)+2*(1-generalized_box_iou_aligned_cxcywh(pred,target))
        return per_frame.mean()+anchor*delta.square().sum(),boxes,per_frame

    try:
        with torch.no_grad():initial,zero,parts=objective()
        assert torch.equal(zero.cpu(),base),'Zero residual differs from native boxes'
        if not torch.isfinite(initial):return finish(base,'nonfinite_initial_objective')
        audit.update(initial_loss=float(initial),initial_frame_losses=parts.cpu())
        for step in range(steps):
            opt.zero_grad(set_to_none=True);loss,_,_=objective()
            if not torch.isfinite(loss):return finish(base,'nonfinite_training_loss')
            loss.backward();audit['gradient_calls']+=1
            if delta.grad is None or not torch.isfinite(delta.grad).all():return finish(base,'nonfinite_gradient')
            before=delta.detach().clone();state=copy.deepcopy(opt.state_dict());gradnorm=float(delta.grad.norm())
            opt.step();direction=delta.detach()-before;trials=[];accepted=False
            for alpha in BACKTRACKS:
                with torch.no_grad():
                    delta.copy_(before+alpha*direction);candidate,boxes,_=objective()
                finite=bool(torch.isfinite(candidate)) and valid_boxes(boxes)
                trials.append(dict(alpha=alpha,loss=float(candidate) if torch.isfinite(candidate) else None,valid=finite))
                if finite and float(candidate)<float(loss.detach())-1e-7:
                    accepted=True;break
            if not accepted:
                with torch.no_grad():delta.copy_(before)
                opt.load_state_dict(state)
            audit['history'].append(dict(step=step,loss=float(loss.detach()),gradient_norm=gradnorm,
                 accepted=accepted,trials=trials,accepted_step_norm=float((delta.detach()-before).norm())))
            audit['accepted_steps']+=int(accepted)
        with torch.no_grad():final,boxes,parts=objective()
        if not torch.isfinite(final) or not valid_boxes(boxes):return finish(base,'invalid_final_prediction')
        assert float(final)<=float(initial)+1e-6
        audit.update(final_loss=float(final),final_frame_losses=parts.cpu(),
                     native_query_norm=float(torch.stack([c['query_tgt'].float().norm(dim=-1).mean() for c in caches]).mean()))
        if steps>0 and audit['accepted_steps']==0:return finish(base,'no_accepted_update')
        return finish(boxes)
    except torch.cuda.OutOfMemoryError:
        opt.zero_grad(set_to_none=True);torch.cuda.empty_cache()
        return finish(base,'spatial_solver_out_of_memory')


def interpolate_corrections(base,pseudo,frame_ids):
    """Baseline only: interpolate normalized cxcywh *corrections* in real time.

    Between outermost accepted anchors use linear interpolation. Outside
    their hull, leave native boxes unchanged. One anchor = sparse replacement.
    Clip modified boxes to the image; invalid modified boxes revert per frame.
    """
    output=base.clone();invalid=[]
    if not pseudo:return output,dict(interpolated_frames=0,invalid_frames=[])
    pp=sorted(pseudo,key=lambda p:p['position']);pos=np.array([p['position'] for p in pp]);ids=np.asarray(frame_ids)
    target=torch.tensor([p['box'] for p in pp]);correction=(target-base[pos]).numpy()
    changed=pos if len(pos)==1 else np.arange(pos[0],pos[-1]+1)
    values=np.stack([np.interp(ids[changed],ids[pos],correction[:,k]) for k in range(4)],axis=1)
    candidate=base[changed]+torch.tensor(values,dtype=base.dtype)
    candidate=box_convert(box_convert(candidate,'cxcywh','xyxy').clamp(0,1),'xyxy','cxcywh')
    for i,idx in enumerate(changed):
        if valid_boxes(candidate[i]):output[idx]=candidate[i]
        else:invalid.append(int(idx))
    return output,dict(interpolated_frames=len(changed),invalid_frames=invalid,outside_hull='native_unchanged')


def native_episode(model,batch,temporal_config):
    """Live temporal fit + native spatial caches; original model never mutates."""
    from vg_tta.decota_tastvg_episode_v1 import forward,fitted_merge
    from methods.decota_v1 import fit,decode
    captures=[];h=model.ground_decoder.decoder.register_forward_pre_hook(
        lambda _m,args,kw:captures.append(detached_tree(kw)),with_kwargs=True)
    start=time.perf_counter()
    try:base,inputs,records=forward(model,batch)
    finally:h.remove()
    assert len(captures)==4;torch.cuda.synchronize();native_seconds=time.perf_counter()-start
    ids=list(batch['targets'][0]['frame_ids']);t=time.perf_counter()
    private,zs,temporal_audit=fit(model.temp_embed,inputs,backbone='tastvg',**temporal_config)
    extent=fitted_merge(zs,records,ids);decoded=decode(base['temporal_logits'],extent,ids,native_indices=base['predicted_indices'])
    torch.cuda.synchronize();temporal_seconds=time.perf_counter()-t
    del private,zs
    # Exact pre-existing selection evidence, including native FP16 arithmetic.
    starts=[];ends=[];full=np.asarray(ids)
    for record in records:
        z=record['temporal_logits'].cuda();vi=np.asarray(record['frame_ids'])
        score=(z[:,0].log_softmax(0)[:,None]+z[:,1].log_softmax(0)[None]).double().cpu()
        legal=torch.triu(torch.ones_like(score,dtype=torch.bool),diagonal=1)
        joint=torch.zeros_like(score);joint[legal]=torch.softmax(score[legal],0)
        ps=joint.sum(1).numpy();pe=joint.sum(0).numpy()
        starts.append(np.array([ps[vi>t].sum() for t in full]));ends.append(np.array([pe[vi<t].sum() for t in full]))
    evidence=np.clip(1-np.prod(starts,axis=0)-np.prod(ends,axis=0),0,1)
    return dict(boxes=base['raw_boxes'].float().cpu(),native_indices=list(base['predicted_indices']),
        indices=list(decoded['indices']),frame_ids=ids,evidence=evidence.tolist(),caches=[captures[1],captures[3]],
        temporal_audit=temporal_audit,timing=dict(native_seconds=native_seconds,temporal_seconds=temporal_seconds))
