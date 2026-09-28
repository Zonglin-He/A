"""F41 exact F39 optimization prefixes; no GT or changed temporal objective.

A rejected step restores weights AND Adam state. If the next attempted update
is exactly identical and rejected, the remaining deterministic eval-mode steps
repeat that fixed point. We record this execution shortcut, not fake backwards.
"""
import copy
import math
import time

import torch

from vg_tta.dense_support_temporal_v1 import objective, norm
from vg_tta.dense_support_tuning_v1 import state_hash
from vg_tta.time_space_repair_v1 import legal_logp
from vg_tta.decota_tastvg_episode_v1 import fitted_merge, native_view_indices


def fit_prefixes(it, records, ids, teacher, *, lr, beta, budgets, output_control=False):
    budgets=sorted(set(int(n) for n in budgets))
    assert budgets and budgets[0]>=0
    maximum=budgets[-1]
    it.restore(it.initial)
    params=[p for _,p in it.named]
    cls=torch.optim.Adam if output_control else torch.optim.AdamW
    opt=cls(params,lr=lr,betas=(.9,.999),eps=1e-4,weight_decay=0.)
    with torch.no_grad():zero=it.values()
    refs=[legal_logp(z)[0].detach().clone() for z in zero['logits']]
    teacher_copy=[t['a'].detach().clone() for t in teacher]
    path=[];states={};prefixes={};best=math.inf;selected=0;saved=it.state()
    frozen_tail=None;previous_rejection=None;backwards=0;start=time.perf_counter()
    def retain(n):
        h=path[selected]['state_sha256']
        states.setdefault(h,{k:v.detach().cpu().clone() for k,v in saved.items()})
        prefixes[str(n)]=dict(best_step=selected,state_sha256=h,final=path[selected],steps=n)
    for step in range(maximum+1):
        opt.zero_grad(set_to_none=True)
        value=it.values();loss,parts=objective(value['logits'],refs,teacher,beta)
        lv=float(loss.detach());assert math.isfinite(lv)
        if lv<best:best,selected,saved=lv,step,it.state()
        assert torch.equal(value['boxes'],zero['boxes'])
        st=it.state();h=state_hash(st)
        row=dict(step=step,loss=lv,parts=parts,logits=[z.detach().cpu().clone() for z in value['logits']],
                 indices=list(fitted_merge(value['logits'],records,ids)),
                 raw_offset_indices=[list(native_view_indices(z)) for z in value['logits']],
                 state_sha256=h,state_delta=norm([p-it.initial[n] for n,p in it.named]))
        row['physical_interval']=[ids[row['indices'][0]],ids[row['indices'][1]]+1]
        row['raw_physical_intervals']=[[r['frame_ids'][s],r['frame_ids'][e]+1]
                                      for r,(s,e) in zip(records,row['raw_offset_indices'])]
        path.append(row)
        if step in budgets:retain(step)
        if step==maximum:break
        loss.backward();backwards+=1
        row['gradient_norm']=norm([p.grad for p in params])
        row['unused_parameters']=[n for n,p in it.named if p.grad is None]
        assert not row['unused_parameters']
        assert all(bool(torch.isfinite(p.grad).all()) for p in params)
        before=it.state();opt_before=copy.deepcopy(opt.state_dict())
        opt.step();after=it.state();trials=[];accepted=False
        for alpha in (1.,.5,.25,.125):
            it.restore({n:v+alpha*(after[n]-v) for n,v in before.items()})
            with torch.no_grad():ll,_=objective(it.values()['logits'],refs,teacher,beta)
            val=float(ll);trials.append(dict(alpha=alpha,loss=val if math.isfinite(val) else None))
            if math.isfinite(val) and val<lv-1e-9:accepted=True;break
        row.update(accepted=accepted,trials=trials,optimizer_restored_on_reject=not accepted,
                   proposal_norm=norm([after[n]-before[n] for n in before]))
        if accepted:
            previous_rejection=None
            continue
        it.restore(before);opt.load_state_dict(opt_before)
        assert all(torch.equal(p,before[n]) for n,p in it.named)
        restored=opt.state_dict();assert restored['param_groups']==opt_before['param_groups']
        for k,v in opt_before['state'].items():
            assert all(torch.equal(restored['state'][k][n],x) if torch.is_tensor(x)
                       else restored['state'][k][n]==x for n,x in v.items())
        signature=(h,lv,row['gradient_norm'],row['proposal_norm'],trials)
        if previous_rejection==signature:
            frozen_tail=dict(first_omitted_state=step+1,through=maximum,
                reason='two identical rejected proposals; parameters and optimizer restored; fixed eval-mode inputs',
                repeated_step=step,unchanged_state_sha256=h,unchanged_loss=lv)
            for n in budgets:
                if n>step:retain(n)
            break
        previous_rejection=signature
    it.restore(saved)
    with torch.no_grad():end=it.values()
    assert all(torch.equal(a.cpu(),b) for a,b in zip(end['logits'],path[selected]['logits']))
    assert all(torch.equal(t['a'],a) for t,a in zip(teacher,teacher_copy))
    assert set(prefixes)==set(map(str,budgets))
    out=dict(path=path,states=states,prefixes=prefixes,initial_state={n:v.detach().cpu() for n,v in it.initial.items()},
        lr=float(lr),beta=float(beta),budgets=budgets,requested_steps=maximum,backwards=backwards,
        path_states=len(path),frozen_tail=frozen_tail,parameters=sum(p.numel() for p in params),
        optimizer=cls.__name__,seconds=time.perf_counter()-start,GT_online=False,
        teacher_frozen=True,spatial_invariant=True,actual_best_state_verified=True,
        parameter_TTA=not output_control,output_optimization=output_control,failure=None)
    it.restore(it.initial)
    assert all(torch.equal(p,it.initial[n]) for n,p in it.named)
    return out


def selected_fit(trajectory,budget,boxes):
    p=trajectory['prefixes'][str(budget)]
    z=dict(p['final']);z['boxes']=boxes
    return dict(state=trajectory['states'][p['state_sha256']],final=z,best_step=p['best_step'],
        state_delta=z['state_delta'],parameters=trajectory['parameters'],steps=int(budget),
        lr=trajectory['lr'],beta=trajectory['beta'],GT_online=False,parameter_TTA=trajectory['parameter_TTA'])
