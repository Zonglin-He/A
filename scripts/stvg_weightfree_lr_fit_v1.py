"""Fixed loss-level evidence weighting with optional label-free step rejection.

Reward remains original IoU. No GT in fitting; bounded separate development LR comparison, no new
teacher calls. Validation tests agreement with the same imperfect reference;
it is not semantic verification or a guarantee of generalization.
"""
import copy
import math
import time
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

def loss_per_anchor(mean, policy, sigma):
    kl=(mean-policy['mean']).square().sum(-1)/(2*sigma**2)
    logp=-(policy['samples']-mean[:,None,:]).square().sum(-1)/(2*sigma**2)-2*math.log(2*math.pi*sigma**2)
    return kl-((policy['weights']-1/32)*logp).sum(-1)

def accept_reference_step(before, after, weights, tolerance=1e-7):
    import torch
    if not bool(torch.isfinite(after).all()):return False
    return float((weights*(after-before)).sum()/weights.sum()) >= -tolerance

def fit_validated(base, initial, anchors, frame_ids, key, config, *, validate,
                  reference_sha256, checkpoint_callback=None):
    import torch
    from methods.decota_final_simplified_v1.tensors import detached
    from scripts import stvg_target_teacher_student_v1 as core
    from scripts import stvg_native_instance_spatial_fit_v1 as native
    from vg_tta.decota_spatial_opd_tunable_v1 import rollout,mean_coordinates,optimizer_step
    from vg_tta.tastvg_decota_critic_p0_v1 import aligned_iou
    native.verify_source_bindings()
    matched=[]
    for ds in ['hc2','vidstg']:
        original=core.load_original_opd_config(ds);original['steps']=5
        for scale in [.25,.5,1.]:
            proposed=dict(original);proposed['lr']*=scale
            if proposed==dict(config):matched.append((original,scale))
    if not matched:raise ValueError('Only prelocked quarter/half/original learning rate allowed')
    native._validate_key_and_config(key, matched[0][0], 'joint', 'teacher', reference_sha256)
    checked=native._validate_anchors(anchors, frame_ids, torch)
    active=[a for a in checked if a['weight']>0]
    positions=[a['position'] for a in active]
    started=time.perf_counter();base.restore(initial)
    replay,selected=core._trainable_replay(base,list(frame_ids),key,'joint')
    origin=replay.state();params=[p for _,p in selected]
    assert sum(p.numel() for p in params)==1792
    optimizer=torch.optim.Adam(params,lr=config['lr'],betas=(.9,.999),eps=1e-8,weight_decay=0.)
    evidence=torch.stack([a['box'].to(base.zero['boxes']) for a in active]) if active else base.zero['boxes'].new_empty((0,4))
    weights=base.zero['boxes'].new_tensor([a['weight'] for a in active]).detach()
    path=[];rounds=[];proposals=accepted=0;stopped=False
    try:
        for step in range(config['steps']):
            before_state=replay.state();before=replay.values()['boxes']
            before_boxes=before.detach().clone();opt_before=core._copy_cpu(optimizer.state_dict())
            entry=dict(step=step,boxes=before_boxes.cpu(),state=detached(before_state,'cpu'),optimizer_before=opt_before)
            if not active or stopped:
                entry.update(skipped=True,reason='no_evidence' if not active else 'stopped_after_rejected_step')
                path.append(entry);continue
            mean=mean_coordinates(before[positions]);policy=rollout(mean,evidence,key,step,config=config)
            per_anchor=loss_per_anchor(mean,policy,config['sigma'])
            loss=(weights*per_anchor).mean()
            assert torch.isfinite(loss)
            before_iou=aligned_iou(before_boxes[positions],evidence).detach()
            update=optimizer_step(optimizer,loss,params,policy['informative'])
            proposed_state=replay.state()
            with torch.no_grad():proposed=replay.values()['boxes'].detach().clone()
            proposed_iou=aligned_iou(proposed[positions],evidence).detach()
            opt_proposed=core._copy_cpu(optimizer.state_dict())
            approved=update is not None and (not validate or accept_reference_step(before_iou,proposed_iou,weights))
            if update is not None:
                proposals+=1
                if approved:accepted+=1
                else:
                    replay.restore(before_state);optimizer.load_state_dict(copy.deepcopy(opt_before));stopped=True
                    assert core._tree_equal(core._copy_cpu(optimizer.state_dict()),opt_before)
                    assert core._tree_equal(detached(replay.state(),'cpu'),detached(before_state,'cpu'))
            after_state=replay.state()
            with torch.no_grad():after=replay.values()['boxes'].detach().clone()
            entry.update(skipped=False,loss=float(loss.detach()),anchor_loss=per_anchor.detach().cpu(),
                reliability=weights.cpu(),rollout=detached(policy,'cpu'),
                update=None if update is None else detached(update,'cpu'),
                proposed_state=detached(proposed_state,'cpu'),proposed_boxes=proposed.cpu(),
                optimizer_proposed=opt_proposed,optimizer_after=core._copy_cpu(optimizer.state_dict()),
                update_state=detached(after_state,'cpu'),after_boxes=after.cpu(),
                reference_iou_before=before_iou.cpu(),reference_iou_proposed=proposed_iou.cpu(),
                proposed=update is not None,accepted=approved,validation_enabled=validate)
            path.append(entry)
            rounds.append(dict(step=step,proposed=update is not None,accepted=approved,
                no_information=not policy['informative'],reference_mean_before=float((weights*before_iou).sum()/weights.sum()),
                reference_mean_proposed=float((weights*proposed_iou).sum()/weights.sum())))
            if checkpoint_callback:
                checkpoint_callback(dict(next_step=step+1,path=path,rounds=rounds,
                    state=detached(after_state,'cpu'),optimizer_state=core._copy_cpu(optimizer.state_dict()),
                    accepted_optimizer_steps=accepted,proposed_optimizer_steps=proposals,GT_read=False))
        with torch.no_grad():final=replay.values()['boxes'].detach().cpu()
        return dict(schema='stvg-instance-loss-weighted-fit-v1',scope='joint',active_parameters=1792,
            algorithm='validated_loss_weighted_iou' if validate else 'loss_weighted_iou',
            initial_state=detached(origin,'cpu'),final_state=detached(replay.state(),'cpu'),
            final_cached_decoder_boxes=final,path=path,rounds=rounds,
            optimizer_steps=accepted,proposed_optimizer_steps=proposals,
            rejected_optimizer_steps=proposals-accepted,declared_steps=config['steps'],
            config=dict(config),seed_key=key,observation_positions=positions,
            reliability=[float(x) for x in weights.cpu()],reference_sha256=reference_sha256,
            reference_role='teacher',reward='original_iou',loss='mean_r_times_original_per_anchor_forward_KL',
            validation='weighted_reference_IoU_non_decrease_1e-7_then_stop' if validate else 'none',
            reliability_is_calibrated=False,GT_read=False,wall_seconds=time.perf_counter()-started)
    finally:
        replay.restore(origin);base.restore(initial)

def contracts():
    import torch
    from scripts import stvg_target_teacher_student_v1 as core
    from vg_tta.decota_spatial_opd_tunable_v1 import rollout,mean_coordinates,likelihood_loss
    base=core._synthetic_student_base(8);cfg=core.load_original_opd_config('hc2');cfg['steps']=5
    mu=mean_coordinates(base.zero['boxes'][[1,3]]);e=base.zero['boxes'][[1,3]].clone();e[:,0]+=.03
    p=rollout(mu,e,'caption_query_fixed_space|synthetic',0,config=cfg)
    assert torch.equal(loss_per_anchor(mu,p,cfg['sigma']).mean(),likelihood_loss(mu,p['mean'],p['samples'],p['weights'],cfg['sigma']))
    v=loss_per_anchor(mu,p,cfg['sigma']);w=torch.tensor([.2,.8]);assert torch.equal((v*w).mean(),.1*v[0]+.4*v[1])
    assert accept_reference_step(torch.tensor([.4,.5]),torch.tensor([.5,.5]),w)
    assert not accept_reference_step(torch.tensor([.4,.5]),torch.tensor([.3,.5]),w)
    assert not accept_reference_step(torch.tensor([.4,.5]),torch.tensor([float('nan'),.5]),w)
    a=[dict(position=1,physical_frame_id=1,box=e[0],weight=.7)]
    for gate in [False,True]:
        f=fit_validated(base,base.initial,a,list(range(8)),'caption_query_fixed_space|synthetic',cfg,validate=gate,reference_sha256='a'*64)
        assert f['proposed_optimizer_steps']==f['optimizer_steps']+f['rejected_optimizer_steps']
        for row in f['rounds']:
            if gate and row['accepted']:assert row['reference_mean_proposed']>=row['reference_mean_before']-1e-7
    assert not torch.cuda.is_initialized()
    return dict(status='pass',scope='synthetic CPU loss and acceptance contracts',GT_read=False)

if __name__=='__main__':
    import argparse,json
    p=argparse.ArgumentParser();p.add_argument('--contracts',action='store_true');a=p.parse_args()
    assert a.contracts;print(json.dumps(contracts()))
