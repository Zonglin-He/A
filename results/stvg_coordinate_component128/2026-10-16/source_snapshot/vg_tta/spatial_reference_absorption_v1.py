"""Fixed-supervision Adam acceptance controls; no GT inputs or deployment edits."""
import copy,math,time
import torch
from methods.decota_final_simplified_v1.tensors import detached
from methods.decota_final_simplified_v1.objectives import SpatialLoss,generalized_iou

ALPHAS=[1.,.5,.25,.125,.0625,.03125]
PROBE_ALPHAS=[0.,.03125,.0625,.125,.25,.5,1.]
LOSS_TOL=1e-9
BOX_TOL=1e-7


def sync(replay):
    if next(iter(replay.initial.values())).is_cuda:torch.cuda.synchronize()


def blend(before,proposal,alpha):
    if alpha==0:return detached(before)
    if alpha==1:return detached(proposal)
    return {k:v+alpha*(proposal[k]-v) for k,v in before.items()}


def output_allowed(current,trial,anchors):
    if not anchors:return True,0.
    pos=[a['position'] for a in anchors];target=current.new_tensor([a['box'] for a in anchors])
    scale=target[:,[2,3,2,3]]
    bound=.5*torch.maximum((current[pos]-target).abs(),scale)
    excess=(trial[pos]-current[pos]).abs()-bound
    return bool((excess<=BOX_TOL).all()),float(excess.max())


def parts(boxes,anchors,planned):
    if not anchors:return dict(l1_coordinates=[],giou=[],expert_iou=[],anchor_boxes=[],loss=0.)
    pred=boxes[[a['position'] for a in anchors]];target=pred.new_tensor([a['box'] for a in anchors])
    from methods.decota_final_simplified_v1.objectives import xyxy
    a,b=xyxy(pred),xyxy(target);inter=(torch.minimum(a[:,2:],b[:,2:])-torch.maximum(a[:,:2],b[:,:2])).clamp_min(0).prod(-1)
    union=(pred[:,2:].prod(-1)+target[:,2:].prod(-1)-inter).clamp_min(1e-7)
    l1=(pred-target).abs();g=generalized_iou(pred,target)
    return dict(l1_coordinates=l1.detach().cpu().tolist(),giou=g.detach().cpu().tolist(),
        expert_iou=(inter/union).detach().cpu().tolist(),anchor_boxes=pred.detach().cpu().tolist(),
        loss=float((5*l1.sum()+2*(1-g).sum())/planned))


def adam(replay,lr):
    return torch.optim.Adam([p for _,p in replay.named],lr=lr,betas=(.9,.999),eps=1e-8,weight_decay=0.)


@torch.enable_grad()
def probe(replay,anchor,lr):
    replay.restore(replay.initial);opt=adam(replay,lr);loss_fn=SpatialLoss([anchor],replay.zero['boxes'])
    try:
        opt.zero_grad(set_to_none=True);v=replay.values();loss=4*loss_fn(v['boxes']);loss.backward();opt.step()
        proposal=replay.state();states=[]
        for alpha in PROBE_ALPHAS:
            st=blend(replay.initial,proposal,alpha);replay.restore(st)
            with torch.no_grad():value=replay.values()
            states.append(dict(alpha=alpha,state=detached(st,'cpu'),boxes=detached(value['boxes'],'cpu'),
                details=parts(value['boxes'],[anchor],1)))
        return dict(states=states,proposal=detached(proposal,'cpu'),optimizer=detached(opt.state_dict(),'cpu'),
            backwards=1,values_calls=8,GT_online=False)
    finally:replay.restore(replay.initial)


@torch.enable_grad()
def fit(replay,anchors,lr,arm,planned=4,steps=10):
    assert arm in ['O0','O1','O2','O3'];assert planned in [1,4]
    replay.restore(replay.initial);opt=adam(replay,lr/4 if arm=='O1' else lr)
    objective=SpatialLoss(anchors,replay.zero['boxes']);factor=4/planned
    paths=[];attempts=[];best=None;best_loss=math.inf;selected=0;backwards=0;calls=0;forward_seconds=0.
    sync(replay);start=time.perf_counter()
    def evaluate(grad):
        nonlocal calls,forward_seconds
        sync(replay);begin=time.perf_counter()
        with torch.set_grad_enabled(grad):v=replay.values();loss=factor*objective(v['boxes'])
        sync(replay);forward_seconds+=time.perf_counter()-begin;calls+=1
        assert torch.isfinite(loss),'nonfinite loss'
        return v,loss
    def record(value,loss,step):
        nonlocal best,best_loss,selected
        row=dict(step=step,loss=float(loss.detach()),state=detached(replay.state(),'cpu'),
                 boxes=detached(value['boxes'],'cpu'),details=parts(value['boxes'],anchors,planned))
        paths.append(row)
        if row['loss']<best_loss:best=row;best_loss=row['loss'];selected=step
    try:
        value,loss=evaluate(True);record(value,loss,0)
        for step in range(1,(steps if anchors else 0)+1):
            opt.zero_grad(set_to_none=True);loss.backward();backwards+=1
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for _,p in replay.named)
            before=replay.state();ob=copy.deepcopy(opt.state_dict());oldboxes=value['boxes'].detach();oldloss=float(loss.detach())
            opt.step();proposal=replay.state();oa=copy.deepcopy(opt.state_dict());trials=[];accepted=False
            for alpha in (ALPHAS if arm in ['O2','O3'] else [1.]):
                state=blend(before,proposal,alpha);replay.restore(state);trial,tl=evaluate(True);lv=float(tl)
                allowed,excess=output_allowed(oldboxes,trial['boxes'],anchors)
                decrease=lv<oldloss-LOSS_TOL
                ok=(arm in ['O0','O1']) or (decrease and (allowed or arm=='O2'))
                trials.append(dict(alpha=alpha,loss=lv,loss_decrease=decrease,output_allowed=allowed,max_output_excess=excess,
                    accepted=ok,state=detached(state,'cpu'),boxes=detached(trial['boxes'],'cpu'),details=parts(trial['boxes'],anchors,planned)))
                if ok:
                    accepted=True;record(trial,tl,step);value,loss=trial,tl;break
            if not accepted:replay.restore(before);opt.load_state_dict(ob)
            attempts.append(dict(step=step,before=detached(before,'cpu'),proposal=detached(proposal,'cpu'),
                optimizer_before=detached(ob,'cpu'),optimizer_proposed=detached(oa,'cpu'),
                optimizer_committed=detached(opt.state_dict(),'cpu'),accepted=accepted,trials=trials))
            if not accepted:break
        sync(replay);seconds=time.perf_counter()-start
        return dict(arm=arm,anchors=anchors,planned=planned,lr=lr/4 if arm=='O1' else lr,selected_step=selected,
            state=best['state'],boxes=best['boxes'],path=paths,attempts=attempts,backwards=backwards,values_calls=calls,
            offset_suffix_forwards=2*calls,trial_forwards=sum(len(a['trials']) for a in attempts),fit_seconds=seconds,
            suffix_seconds=forward_seconds,accepted_updates=sum(a['accepted'] for a in attempts),
            final_optimizer=detached(opt.state_dict(),'cpu'),skipped=not anchors,GT_online=False)
    finally:replay.restore(replay.initial)
