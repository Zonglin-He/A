"""E-retaining QR256, two-step conditional endpoint Gaussian distillation."""
import numpy as np
import torch
from methods.decota_final_simplified_v1.tensors import detached,state_hash
from vg_tta.stvg_dense_e_opd64_v1 import NativeReplay
from vg_tta.stvg_error_state_correction_v1 import encode,decode
from vg_tta.stvg_dino_conditional_opd32_v1 import legal
QUERY='spatial.query_residual'
FIT_ARMS=['Fixed_center_KL','Fixed_sample_KL','On_policy_KL']
ARMS=['E_MBR','Center_full_local']+FIT_ARMS

def corrected(boxes,source,E):
    b=boxes.double();src=source.to(b);e=E.to(b)
    assert torch.isfinite(b).all()and (b[:,2:]>0).all()and (src[:,2:]>0).all()
    return torch.cat([e[:,:2]+(b[:,:2]-src[:,:2]),e[:,2:]*torch.exp(torch.log(b[:,2:])-torch.log(src[:,2:]))],-1)

def coordinates(b):return torch.cat([b[:,:2],torch.log(b[:,2:])],-1)

def loss_fn(mu,targets,mask,scale):
    assert targets.shape==(4,4,4)and mask.shape==(4,4)and not targets.requires_grad
    return ((((mu[:,None]-targets)/scale)**2).sum(-1)*mask).sum()/32

def properties():
    s=torch.tensor([[.2,.4,.1,.2],[.8,.7,.3,.1]],dtype=torch.float32);e=s.double()+.01;u=s.clone().requires_grad_(True)
    out=corrected(u,s,e);assert torch.equal(out,e);torch.autograd.grad(out.sum(),u)
    mu=torch.zeros(4,4,dtype=torch.float64,requires_grad=True);t=torch.ones(4,4,4,dtype=torch.float64);m=torch.zeros(4,4,dtype=torch.float64);scale=torch.ones(4)
    assert torch.equal(torch.autograd.grad(loss_fn(mu,t,m,scale),mu)[0],mu.detach())
    m[0,0]=1;g=torch.autograd.grad(loss_fn(mu,t,m,scale),mu)[0];expected=((mu.detach()[:,None]-t)*m[:,:,None]).sum(1)/16
    assert torch.equal(g,expected)
    return dict(status='pass',zero_QR_exact_E=True,no_feedback_zero_gradient=True,masked16_normalization=True,analytic_mean_gradient=True,QR_grad_through_frozen_decoder_required=True)

def cached_details(old,round_index,arm):
    if arm=='Fixed_center_KL':return [old['center_feedback']for _ in range(4)]
    return old['sample_feedback']['details'][round_index*4:round_index*4+4]

def query_feedback(caches,reference_boxes,positions):
    valid=legal(reference_boxes[:,positions]).all(1);details=[dict(legal=bool(v),anchors={})for v in valid];traces={}
    ids=np.flatnonzero(valid)
    for pos,c in zip(positions,caches):
        if not len(ids):continue
        unique,inverse=np.unique(reference_boxes[ids,pos],axis=0,return_inverse=True);tr=c.run_trace(unique,True);traces[pos]=dict(candidate_indices=ids,inverse=inverse,trace=tr)
        for j,k in enumerate(ids):
            h=inverse[j];score=float(tr['scores'][-1,h]);details[k]['anchors'][pos]=dict(box=tr['boxes'][-1,h],score=score,accepted=score>=.35,slot=int(tr['content_slots'][h]),mask_slot=int(tr['mask_slots'][h]))
    return details,traces

@torch.enable_grad()
def fit(base,ids,key,source,E,old,deltaE,caches,arm,lr,check):
    assert arm in FIT_ARMS;base.restore(base.initial);rp=NativeReplay(base,ids,key,{})
    flags=[(v,v.requires_grad)for _,v in rp.named]
    for n,v in rp.named:v.requires_grad_(n==QUERY)
    u=base.delta;assert u.numel()==256 and torch.count_nonzero(u)==0
    initial=detached(rp.state(),'cpu');src=source.to(u);et=E.to(device=u.device,dtype=torch.float64);pos=old['planned_positions'];scale=np.asarray(old['scale']);sc=torch.as_tensor(scale,device=u.device,dtype=torch.float64)
    opt=torch.optim.Adam([u],lr=lr,betas=(.9,.999),eps=1e-8,weight_decay=0);assert not opt.state
    fixedhash=state_hash(base.decoder.state_dict());path=[];rounds=[];active=old['active'];count=dict(forwards=0,backwards=0,conditional_logical_queries=0,conditional_decoder_seconds=0.)
    try:
        for step in range(3):
            check();v=rp.values();count['forwards']+=1;b=corrected(v['boxes'],src,et);bc=b.detach().cpu();state=detached(rp.state(),'cpu')
            if step==0:assert torch.equal(bc,E.double())and torch.equal(v['boxes'].detach().cpu(),source)
            path.append(dict(step=step,boxes=bc,state=state,native_boxes=v['boxes'].detach().cpu()))
            if step==2:break
            refs=None;traces={};details=[]
            if active:
                if arm in ['Fixed_sample_KL','On_policy_KL']and step==1:
                    initial_refs=np.stack([decode(encode(source.double().numpy())+a*scale)for a in old['actions'][4:8]])
                    shift=encode(bc.numpy())-encode(E.double().numpy())if arm=='On_policy_KL'else np.zeros_like(encode(E.double().numpy()));refs=np.stack([decode(encode(q)+shift)for q in initial_refs])
                    before=sum(c.decoder_seconds for c in caches);details,traces=query_feedback(caches,refs,pos);count['conditional_decoder_seconds']+=sum(c.decoder_seconds for c in caches)-before
                    count['conditional_logical_queries']+=sum(len(t['candidate_indices'])for t in traces.values())
                else:details=cached_details(old,step,arm)
            targets=torch.zeros((4,4,4),device=u.device,dtype=torch.float64);mask=torch.zeros((4,4),device=u.device,dtype=torch.float64)
            for k,d in enumerate(details):
                for j,i in enumerate(pos):
                    fb=d['anchors'].get(i)
                    if fb and fb['accepted']:
                        assert d['legal']and fb['score']>=.35;targets[j,k]=torch.as_tensor(encode(np.asarray(fb['box'])[None])[0],device=u.device);mask[j,k]=1
            mu=coordinates(b)[pos];loss=loss_fn(mu,targets,mask,sc);assert torch.isfinite(loss)
            gm=torch.autograd.grad(loss,mu,retain_graph=True)[0];expected=((mu.detach()[:,None]-targets)/sc.square()*mask[:,:,None]).sum(1)/16
            err=float((gm-expected).abs().max());assert err<1e-10
            opt.zero_grad(set_to_none=True);updated=bool(mask.any());beforeu=u.detach().clone();grad=None
            if updated:
                grad=torch.autograd.grad(loss,u)[0];assert torch.isfinite(grad).all();u.grad=grad.detach();opt.step();count['backwards']+=1
            assert torch.isfinite(u).all()
            rounds.append(dict(round=step,updated=updated,loss=float(loss.detach()),mask=mask.cpu(),targets=targets.cpu(),scale=scale,
                feedback=details,live_traces=traces,live_reference_boxes=refs,mean=mu.detach().cpu(),mean_gradient=gm.cpu(),analytic_gradient_error=err,
                QR_gradient=None if grad is None else grad.cpu(),QR_step=(u.detach()-beforeu).cpu(),Adam_after=detached(opt.state_dict(),'cpu'),
                fixed_denominator=16,teacher_refreshed=arm=='On_policy_KL'and step==1,noise_indices=list(range(step*4,step*4+4))if arm!='Fixed_center_KL'else None))
            del v,b,loss,mu,gm,targets,mask
        with torch.no_grad():
            base.restore(path[-1]['state']);full=base.reference_values(capture=False)['boxes'];assert torch.equal(full.cpu(),path[-1]['native_boxes']);count['full_two_pass_checks']=1
        assert all(torch.equal(v['state'][n],initial[n])for v in path for n in initial if n!=QUERY)
        assert state_hash(base.decoder.state_dict())==fixedhash and not any(p.grad is not None for p in base.decoder.parameters())
        return dict(arm=arm,path=path,rounds=rounds,final=path[-1]['boxes'],state=path[-1]['state'],count=count,
            pretrained_and_LN_frozen=True,query_only256=True,selected_step=2,E_retained=True,full_two_pass_exact=True,lr=lr,GT_used=False)
    finally:
        base.restore(base.initial)
        for v,f in flags:v.requires_grad_(f)
