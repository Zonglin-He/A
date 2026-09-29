"""One fixed motion-only gradient step for hard and pairwise student targets."""
import torch
from torch.nn import functional as F
from vg_tta.tastvg_causal_round2_v1 import temporal_native,forward,masks
from vg_tta.tastvg_evidence_attack_v1 import norm
from methods.decota_final_simplified_v1.tensors import detached


def endpoint_map(records,intervals):
    out=[]
    for r in records:
        ff=r['frame_ids'];bb=[]
        for start,end in intervals:
            active=[i for i,f in enumerate(ff) if start<=f<end]
            bb.append([active[0],active[-1]] if active else [min(range(len(ff)),key=lambda i:abs(ff[i]-start)),min(range(len(ff)),key=lambda i:abs(ff[i]-(end-1)))])
        out.append(bb)
    return out


def candidate_logits(evidence,bounds):
    vals=[]
    for ev,bb in zip(evidence,bounds):
        z=ev['pred_sted'].log_softmax(1)
        vals.append(torch.stack([z[0,s,0]+z[0,e,1] for s,e in bb]))
    return torch.stack(vals).mean(0)


def ordered_pairs(scores):
    return [(i,j) for i in range(len(scores)) for j in range(len(scores)) if scores[i]>scores[j]+1e-12]


def loss(model,evidence,bounds,scores,selected,arm):
    if arm=='Hard':
        return torch.stack([temporal_native(e['pred_sted'],b[selected],model.cfg.SOLVER.SIGMA) for e,b in zip(evidence,bounds)]).mean()*model.cfg.SOLVER.TEMP_COEF
    assert arm=='OPD'
    ell=candidate_logits(evidence,bounds);pairs=ordered_pairs(scores)
    return torch.stack([F.softplus(-(ell[i]-ell[j])) for i,j in pairs]).mean() if pairs else ell.sum()*0


def step(model,data,critic,arm):
    bounds=endpoint_map(data['records'],critic['candidate_intervals'])
    fields=[v['H'].detach().requires_grad_(True) for v in data['views']]
    ev,_,_=forward(model,data,fields);initial=loss(model,ev,bounds,critic['scores'],critic['selected'],arm)
    gradient=torch.autograd.grad(initial,fields);mm=masks(data,'T');gg=[g*m for g,m in zip(gradient,mm)]
    gn=norm(gg);visual=norm([h*m for h,m in zip(fields,masks(data,'ST'))]);radius=.004*visual
    assert torch.isfinite(gn) and torch.isfinite(initial)
    delta=[-radius/gn*g if float(gn)>0 else torch.zeros_like(g) for g in gg]
    shifted=[h.detach()+d.detach() for h,d in zip(fields,delta)]
    # Verify actual numerical edits, not just requested masks.
    for h,z,m in zip(fields,shifted,mm):assert torch.equal(h.detach()[m==0],z[m==0])
    initial_value=float(initial.detach());del ev,initial,gradient,gg,fields
    with torch.no_grad():
        final,_,pred=forward(model,data,shifted)
        after=float(loss(model,final,bounds,critic['scores'],critic['selected'],arm))
    realized=[z-v['H'] for z,v in zip(shifted,data['views'])]
    mappings=[tuple(tuple(b[i]) for b in bounds) for i in range(len(critic['scores']))]
    diag=dict(initial_loss=initial_value,final_loss=after,loss_decreased=after<initial_value,gradient_norm=float(gn),visual_norm=float(visual),nominal_relative_step=.004,
        realized_relative_step=float(norm(realized)/visual),gradient_steps=1,no_ordered_pairs=not ordered_pairs(critic['scores']),ordered_pairs=len(ordered_pairs(critic['scores'])),
        mapped_candidate_count=len(set(mappings)),candidate_count=len(mappings),motion_only_exact=True,backtracking=False)
    return dict(prediction=detached(pred,'cpu'),diagnostics=diag,bounds=bounds),shifted,final


@torch.no_grad()
def reinsert(model,frames,row,fields,expected):
    from methods.decota_final_simplified_v1.backbone import make_batch,query_subject,inserted_state,offset_batch
    batch=make_batch(frames,row['frame_ids'],row['input'],model);calls=[]
    def replace(module,args,out):
        h=fields[len(calls)];calls.append(1);z=dict(out);z.update(encoded_feature=h,frames_cls=h.mean(0),videos_cls=h.mean(0).mean(0));return z
    hook=model.ground_encoder.register_forward_hook(replace)
    try:
        with query_subject(model,batch,row['parses']['subject']),inserted_state(model,{}):
            for j in (0,1):
                b=offset_batch(batch,j)
                with torch.autocast('cuda',dtype=torch.float16):out=model(b['videos'],b['texts'],b['targets'],iteration_rate=-1)
                for key in ('pred_boxes','pred_sted'):assert torch.equal(out[key],expected[j][key]),('reinsert',j,key)
        assert len(calls)==2
    finally:hook.remove()
    return dict(full_pipeline_exact=True,offsets=2)
