"""Round2 native-Jacobian oracle and frozen-context ASA/query interventions."""
import math
import torch
from torch.nn import functional as F
from vg_tta.tastvg_evidence_capture_v1 import combined
from vg_tta.tastvg_evidence_attack_v1 import norm,project,preservation
from methods.decota_final_simplified_v1.objectives import prediction,generalized_iou
from methods.decota_final_simplified_v1.tensors import detached


def masks(data,branch):
    out=[]
    for v in data['views']:
        h=v['H'];n=v['info']['fea_map_size'][0]*v['info']['fea_map_size'][1];m=torch.zeros_like(h)
        if branch in ('S','ST'):m[:n]=1
        if branch in ('T','ST'):m[-n:]=1
        out.append(m)
    return out


def forward(model,data,fields):
    return combined(model,data['views'],fields,data['records'],data['frame_ids'],evidence_grad=False)


def make_targets(data,label,device):
    ids=data['frame_ids'];valid=torch.tensor(label['valid'],device=device,dtype=torch.bool)
    boxes=torch.tensor(label['boxes'],device=device,dtype=torch.float32)
    assert len(valid)==len(ids)==len(boxes)
    bounds=[];exceptions=[]
    for j,r in enumerate(data['records']):
        frames=r['frame_ids'];active=[i for i,f in enumerate(frames) if label['interval'][0]<=f<label['interval'][1]]
        if active:bound=[active[0],active[-1]]
        else:
            bound=[min(range(len(frames)),key=lambda i:abs(frames[i]-label['interval'][0])),min(range(len(frames)),key=lambda i:abs(frames[i]-(label['interval'][1]-1)))]
            exceptions.append(dict(offset=j,reason='event_not_sampled_in_offset',nearest_indices=bound))
        bounds.append(bound)
    return dict(valid=valid,boxes=boxes,bounds=bounds,exceptions=exceptions)


def temporal_native(z,bound,sigma=2.):
    # Exact VideoSTGLoss.loss_sted formula, including its epsilon and time mean.
    eps=1e-6;terms=[]
    for k,target in enumerate(bound):
        distribution=(-(torch.arange(z.shape[1],device=z.device)[None,:]-target)**2/(2*sigma**2)).exp()
        distribution=F.normalize(distribution+eps,p=1,dim=1)
        p=z[:,:,k].softmax(1);terms.append(p*((p+eps)/distribution).log())
    return (terms[0]+terms[1]).mean()


def losses(model,ev,boxes,target):
    keep=target['valid'];p=boxes[keep];q=target['boxes'][keep]
    # Exact official no-clamp GIoU for positive-size normalized boxes.
    if keep.any():
        a,b=p[:,:2]-p[:,2:]/2,p[:,:2]+p[:,2:]/2
        c,d=q[:,:2]-q[:,2:]/2,q[:,:2]+q[:,2:]/2
        inter=(torch.minimum(b,d)-torch.maximum(a,c)).clamp_min(0).prod(-1)
        union=(b-a).prod(-1)+(d-c).prod(-1)-inter
        enclosing=(torch.maximum(b,d)-torch.minimum(a,c)).prod(-1)
        giou=inter/union-(enclosing-union)/enclosing
        s=model.cfg.SOLVER.BBOX_COEF*(p-q).abs().sum()/len(p)+model.cfg.SOLVER.GIOU_COEF*(1-giou).mean()
    else:s=boxes.sum()*0
    t=torch.stack([temporal_native(e['pred_sted'],b,model.cfg.SOLVER.SIGMA) for e,b in zip(ev,target['bounds'])]).mean()*model.cfg.SOLVER.TEMP_COEF
    return dict(S=s,T=t,ST=s+t)


def metric_loss(ls):return {k:float(v.detach()) for k,v in ls.items()}


def relative_vector(a,b):
    a=a.double().flatten();b=b.double().flatten()
    return dict(relative_norm=float((b-a).norm()/a.norm().clamp_min(1e-30)),cosine_drift=float(1-F.cosine_similarity(a,b,dim=0,eps=1e-30)))


def pool(H,A,chosen,count,branch):
    h=H[:count] if branch=='app' else H[-count:]
    return (h.permute(1,0,2)[chosen]*A.unsqueeze(2)).mean((0,1))


def decode_one(model,view,H,qs,qt,stage):
    info=dict(view['info']);info.update(encoded_feature=H,frames_cls=H.mean(0),videos_cls=H.mean(0).mean(0))
    handle=None
    if stage==2:
        def zero(m,args,kwargs):
            kw=dict(kwargs);q=kw['query_tgt'];kw['query_tgt']=q+torch.zeros(256,device=q.device,dtype=q.dtype)[None,None,:];return args,kw
        handle=model.ground_decoder.decoder.register_forward_pre_hook(zero,with_kwargs=True)
    try:boxes,hidden=model.ground_decoder(encoded_info=info,vis_pos=view['vis_pos'],isq=qs,itq=qt)
    finally:
        if handle:handle.remove()
    return boxes.flatten(1,2)[-1],model.temp_embed(hidden)[-1]


def decoder_pair(model,data,queries,stage):
    zz=[];bb=[]
    for v,e,q in zip(data['views'],data['evidence'],queries):
        b,z=decode_one(model,v,v['H'],q['s'],q['t'],stage);zz.append(z);bb.append(b)
    boxes=torch.stack([bb[i%2][i//2] for i in range(len(data['frame_ids']))])
    return prediction(zz,boxes,data['records'],data['frame_ids'])


def route_pair(model,data,queries,branch):
    counter=[0]
    def replace(m,args,kwargs):
        i=counter[0];counter[0]+=1;kw=dict(kwargs)
        if i%2==0:kw['isq' if branch=='app' else 'itq']=queries[i//2]
        return args,kw
    handle=model.ground_decoder.register_forward_pre_hook(replace,with_kwargs=True)
    try:ev,boxes,pred=forward(model,data,[v['H'] for v in data['views']])
    finally:handle.remove()
    assert counter[0]==4
    return pred


@torch.no_grad()
def causal(model,data,attack):
    changed=attack['selected'];hp=[v['H']+d.to(v['H'].device) for v,d in zip(data['views'],changed['delta'])]
    output=[];eligibility={}
    for stage in (1,2):
        eligibility[stage]=all(b[f'selected_stage{stage}']==n[f'selected_stage{stage}'] for b,n in zip(data['evidence'],changed['evidence']))
        if not eligibility[stage]:continue
        q0=[dict(s=e[f'Qs{stage}'],t=e[f'Qt{stage}']) for e in data['evidence']]
        basepred=decoder_pair(model,data,q0,stage)
        if stage==2:
            assert torch.equal(basepred['boxes'].cpu(),data['prediction']['boxes'].cpu())
            assert all(torch.equal(a.cpu(),b.cpu()) for a,b in zip(basepred['logits'],data['prediction']['logits']))
        for branch,short in (('app','s'),('motion','t')):
            vectors=[];descriptions=[]
            for j,(v,b,n) in enumerate(zip(data['views'],data['evidence'],changed['evidence'])):
                count=v['info']['fea_map_size'][0]*v['info']['fea_map_size'][1];chosen=b[f'selected_stage{stage}'];A=b[f'ASA{stage}_{branch}'];Ap=n[f'ASA{stage}_{branch}'].to(A.device)
                q=dict(base=pool(v['H'],A,chosen,count,branch),A=pool(v['H'],Ap,chosen,count,branch),H=pool(hp[j],A,chosen,count,branch),AH=pool(hp[j],Ap,chosen,count,branch))
                assert torch.equal(q['base'],b[f'Q{short}{stage}'])
                assert torch.equal(q['AH'].cpu(),n[f'Q{short}{stage}'].cpu())
                da=(q['A']-q['base']).double();dh=(q['H']-q['base']).double();dah=(q['AH']-q['base']).double()
                descriptions.append(dict(effects={k:relative_vector(q['base'],q[k]) for k in ('A','H','AH')},cancellation_ratio=float(dah.norm()/(da.norm()+dh.norm()).clamp_min(1e-30)),interaction_relative_norm=float((dah-da-dh).norm()/q['base'].double().norm().clamp_min(1e-30)),A_H_cosine=float(F.cosine_similarity(da,dh,dim=0,eps=1e-30))))
                vectors.append(q)
            conditions={}
            for mode in ('A','H','AH'):
                qs=[{**q0[j],short:vectors[j][mode]} for j in range(2)];pred=decoder_pair(model,data,qs,stage)
                conditions[mode]=dict(prediction=pred,preservation=preservation(basepred,pred,data['frame_ids']))
                if stage==1:
                    routed=route_pair(model,data,[v[mode] for v in vectors],branch)
                    conditions[mode].update(final_routed_prediction=routed,final_routed_preservation=preservation(data['prediction'],routed,data['frame_ids']))
            output.append(dict(stage=stage,branch=branch,vectors=vectors,decomposition=descriptions,baseline_stage_prediction=basepred,conditions=conditions))
    return detached(dict(eligibility=eligibility,rows=output,GT_read=False),'cpu')


def accepts(current,candidate,task,pred,reference,protect,joint=False):
    tol=lambda k:1e-8*max(1.,abs(current[k]))
    descend=candidate[task]<current[task]-tol(task)
    if joint:descend=descend and candidate['ST']<current['ST']-tol('ST')
    p=preservation(reference,pred,[])
    allowed=True if protect is None else p['interval_exact'] if protect=='T' else p['mean_native_interval_box_iou']>=.95
    return bool(descend and allowed),bool(descend),bool(allowed),p


def run_oracle(model,data,target,arm,cfg,guard=lambda:None):
    fields=[v['H'] for v in data['views']];visual=masks(data,'ST');stock=float(norm([h*m for h,m in zip(fields,visual)]));cap=cfg['rho']*stock
    allowed=masks(data,'ST' if arm in ('OST','Oselective') else 'S' if arm.startswith('OS') else 'T')
    ds=[torch.zeros_like(h) for h in fields];paths=[];reference=data['prediction'];grad_calls=0
    for k in range(cfg['steps']):
        guard()
        if arm=='Oselective':task='S' if k<cfg['steps']//2 else 'T';protect='T' if task=='S' else 'S'
        else:task='ST' if arm=='OST' else 'S' if arm.startswith('OS') else 'T';protect='T' if arm=='OS_PT' else 'S' if arm=='OT_PS' else None
        hs=[(h+d).detach().requires_grad_(True) for h,d in zip(fields,ds)]
        ev,boxes,curpred=forward(model,data,hs);ls=losses(model,ev,boxes,target);current=metric_loss(ls)
        if arm=='Oselective' and k==cfg['steps']//2:reference=curpred
        grads=torch.autograd.grad(ls[task],hs);grad_calls+=1;gm=masks(data,task);grads=[g*m for g,m in zip(grads,gm)];length=float(norm(grads))
        assert all(torch.isfinite(g).all() for g in grads)
        record=dict(step=k,task=task,protect=protect,current_loss=current,gradient_norm=length,accepted_alpha=0.,trials=[])
        old=ds
        with torch.no_grad():
            for alpha in cfg['alphas']:
                proposal=project([d-(alpha*2*cap/cfg['steps'])*g/max(length,1e-30) for d,g in zip(old,grads)],allowed,cap)
                ee,bb,pp=forward(model,data,[h+d for h,d in zip(fields,proposal)]);ll=metric_loss(losses(model,ee,bb,target))
                ok,descent,preserved,pm=accepts(current,ll,task,pp,reference,protect,arm=='Oselective')
                record['trials'].append(dict(alpha=alpha,loss=ll,descent=descent,preserved=preserved,accepted=ok,preservation=pm,prediction=pp,delta_norm=float(norm(proposal))))
                if ok:ds=proposal;record['accepted_alpha']=alpha;break
        record['delta_norm']=float(norm(ds));paths.append(record)
        assert float(norm(ds))<=cap*(1+1e-6)
        assert all(torch.count_nonzero(d*(1-m))==0 for d,m in zip(ds,allowed))
        del ev,boxes,ls,grads,hs
    with torch.no_grad():ev,boxes,pred=forward(model,data,[h+d for h,d in zip(fields,ds)]);ls=metric_loss(losses(model,ev,boxes,target))
    return detached(dict(arm=arm,delta=ds,prediction=pred,evidence=ev,loss=ls,path=paths,stock_visual_norm=stock,cap=cap,backwards=grad_calls,GT_oracle=True),'cpu')


def baseline_gradients(model,data,target):
    hs=[v['H'].detach().requires_grad_(True) for v in data['views']];ev,boxes,pred=forward(model,data,hs);ls=losses(model,ev,boxes,target)
    result={}
    for k in ('S','T'):
        gg=torch.autograd.grad(ls[k],hs,retain_graph=k=='S');result[k]=[g*m for g,m in zip(gg,masks(data,'ST'))]
    return detached(dict(gradients=result,losses=metric_loss(ls),prediction=pred),'cpu')


@torch.no_grad()
def swapped_probe(model,data,target,gradients,task,cfg):
    fields=[v['H'] for v in data['views']];scope='T' if task=='S' else 'S';mask=masks(data,scope);visual=masks(data,'ST');cap=cfg['rho']*float(norm([h*m for h,m in zip(fields,visual)]))
    grad=[g.to(h.device)*m for g,h,m in zip(gradients[task],fields,mask)];length=float(norm(grad));ev,b,p=forward(model,data,fields);before=metric_loss(losses(model,ev,b,target));trials=[]
    for alpha in cfg['alphas']:
        delta=project([-alpha*2*cap/cfg['steps']*g/max(length,1e-30) for g in grad],mask,cap);ee,bb,pp=forward(model,data,[h+d for h,d in zip(fields,delta)]);ll=metric_loss(losses(model,ee,bb,target));ok,descent,_,pm=accepts(before,ll,task,pp,p,None)
        trials.append(dict(alpha=alpha,loss=ll,prediction=pp,accepted=ok,delta_norm=float(norm(delta))))
        if ok:return detached(dict(task=task,scope=scope,prediction=pp,loss=ll,trials=trials,gradient_norm=length),'cpu')
    return detached(dict(task=task,scope=scope,prediction=p,loss=before,trials=trials,gradient_norm=length),'cpu')


def native_contract_audit(model,data,target,input_payload):
    """Full original pipeline Jacobian equivalence; no relaxed forward tolerance."""
    from types import SimpleNamespace
    from methods.decota_final_simplified_v1._tastvg_load import official_imports
    from methods.decota_final_simplified_v1.backbone import make_batch,query_subject,inserted_state,offset_batch
    from vg_tta.exact_frame_decode_audit_v2 import decode
    official_imports()
    from models.criterion import VideoSTGLoss
    hs=[v['H'].detach().requires_grad_(True) for v in data['views']]
    flags=[];hooks=[]
    for module in (model.t_temporal_clas,model.s_temporal_clas,model.t_spatial_clas,model.s_spatial_clas):
        hooks.append(module.register_forward_pre_hook(lambda m,args:flags.append([x.requires_grad for x in args if torch.is_tensor(x)])))
    try:ev,boxes,pred=forward(model,data,hs)
    finally:
        for h in hooks:h.remove()
    assert flags and not any(any(f) for f in flags)
    assert all(torch.equal(a['pred_sted'],b['pred_sted']) and torch.equal(a['pred_boxes'],b['pred_boxes']) for a,b in zip(ev,data['evidence']))
    ls=losses(model,ev,boxes,target);cached=torch.autograd.grad(ls['ST'],hs)
    criterion=VideoSTGLoss(model.cfg,[])
    check=[]
    if target['valid'].any():
        valid=target['valid'];gt=SimpleNamespace(bbox=target['boxes'][valid]);parts=criterion.loss_boxes({'pred_boxes':boxes[valid]},[{'boxs':gt}],int(valid.sum()))
        ss=parts['loss_bbox']*model.cfg.SOLVER.BBOX_COEF+parts['loss_giou']*model.cfg.SOLVER.GIOU_COEF
        check.append(abs(float(ss)-float(ls['S'])))
    ts=[]
    for e,b in zip(ev,target['bounds']):
        mask=torch.ones(e['pred_sted'].shape[:2],dtype=torch.bool,device=boxes.device)
        ts.append(criterion.loss_sted({'pred_sted':e['pred_sted']},0,[b],None,mask)['loss_sted'])
    check.append(abs(float(torch.stack(ts).mean()*model.cfg.SOLVER.TEMP_COEF)-float(ls['T'])))
    assert max(check)<2e-6,check
    frames,ids=decode(input_payload['input']);batch=make_batch(frames,ids,input_payload['input'],model);counter=[]
    full_fields=[v['H'].detach().requires_grad_(True) for v in data['views']]
    def replace(m,args,out):
        h=full_fields[len(counter)];counter.append(1);out=dict(out);out.update(encoded_feature=h,frames_cls=h.mean(0),videos_cls=h.mean(0).mean(0));return out
    hook=model.ground_encoder.register_forward_hook(replace)
    full=[]
    try:
        with query_subject(model,batch,input_payload['subject']),inserted_state(model,{}):
            for j in (0,1):
                b=offset_batch(batch,j)
                with torch.autocast('cuda',dtype=torch.float16):z=model(b['videos'],b['texts'],b['targets'],iteration_rate=-1)
                for k in ('pred_boxes','pred_sted'):assert torch.equal(z[k],ev[j][k]),('native_forward',j,k)
                full.append(z)
    finally:hook.remove()
    bb=torch.stack([full[i%2]['pred_boxes'][i//2] for i in range(len(ids))]);fl=losses(model,full,bb,target)
    fg=torch.autograd.grad(fl['ST'],full_fields)
    error=float(norm([a-b for a,b in zip(fg,cached)])/norm(list(cached)).clamp_min(1e-30))
    assert error<1e-5,error
    return dict(forward_exact=True,full_H_joint_gradient_relative_error=error,official_loss_max_absolute_error=max(check),all_evidence_inputs_detached=True,offsets=2)
