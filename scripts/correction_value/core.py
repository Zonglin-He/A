"""Single-decision interventions in the frozen, cached eight-weight policy.

No GT, optimizer or neural model. Every continuation is the same current
student, with common random numbers and its own causally updated state.
"""
from __future__ import annotations
import hashlib
import numpy as np
import torch
import torch.nn.functional as F
from scripts.evidence_policy.policy import _prepare,_prior,_log_probs,_settings,_weights,rollout,gumbel_noise

BRANCHES=('Student','Teacher','Random')


def query_noise(key,T):
    return gumbel_noise('correction_value_v1:'+key,0,0,T)


def random_alternative(key,node,valid,student):
    choices=torch.nonzero(valid,as_tuple=True)[0].tolist()
    choices=[a for a in choices if a!=int(student)]
    assert choices
    seed=int.from_bytes(hashlib.sha256(('correction_value_random_v1:'+key+':'+str(node)).encode()).digest()[:8],'big')%(2**63-1)
    generator=torch.Generator(device='cpu').manual_seed(seed)
    return choices[int(torch.randint(len(choices),(1,),generator=generator))]


def select_points(path,frame_ids,max_points=4):
    """One eligible disagreement nearest each fixed event-quartile center.

    Eligibility depends only on valid actions, actual sampled student action,
    frozen teacher argmax, and whether a later active non-observation exists.
    Physical frame IDs define coverage; no annotation is inspected.
    """
    assert max_points==4
    active=np.asarray(path['active'],bool);observed=np.asarray(path['observed'],bool)
    valid=np.asarray(path['effective_valid'],bool);actual=np.asarray(path['actions'],int)
    scores=np.asarray(path['teacher_scores'],float);teacher=np.where(valid,scores,-np.inf).argmax(-1)
    student_argmax=np.asarray(path['student_log_probs']).argmax(-1)
    frames=np.asarray(frame_ids,dtype=float);assert frames.shape==active.shape and np.all(np.diff(frames)>0)
    candidate=[]
    for t in np.flatnonzero(active):
        if teacher[t]==actual[t] or not (active[t+1:]&~observed[t+1:]).any():continue
        assert valid[t,teacher[t]] and valid[t,actual[t]] and valid[t].sum()>=2
        candidate.append(int(t))
    if not active.any():return [],[]
    event_mask=np.asarray(path.get('event',active),bool);assert event_mask.shape==active.shape
    event=frames[event_mask];assert len(event)>0
    low,high=event[0],event[-1]
    if high==low:return [],candidate
    edges=np.linspace(low,high,5);points=[]
    for j in range(4):
        eligible=[t for t in candidate if frames[t]>=edges[j]and (frames[t]<edges[j+1]or(j==3 and frames[t]<=edges[j+1]))]
        if not eligible:continue
        center=(edges[j]+edges[j+1])/2
        node=min(eligible,key=lambda t:(abs(frames[t]-center),frames[t],t))
        points.append(dict(node=node,time_bin=j,frame_id=int(frames[node]),student_action=int(actual[node]),
            teacher_action=int(teacher[node]),student_argmax=int(student_argmax[node]),
            valid_actions=np.flatnonzero(valid[node]).tolist(),
            disagreement_type='policy_disagreement'if teacher[node]!=student_argmax[node]else'noise_only_disagreement'))
    assert len(points)<=4 and len({r['node']for r in points})==len(points)
    return points,candidate


def continue_student(prepared,phi,cfg,noise,forced_node=None,forced_action=None):
    """Execute one full student path, optionally replacing exactly one action.

    Prefix equality is asserted externally against the immutable original path.
    After the replacement no teacher acts again; student weights and random
    numbers remain identical while memory, previous action and feature evolve.
    """
    b=prepared;settings=_settings(cfg);phi=_weights(phi);T=b['T']
    assert noise.shape==(T,3) and torch.isfinite(noise).all()
    assert (forced_node is None)==(forced_action is None)
    memory=b['initial_memory'];previous=torch.tensor([1.,0.,0.],dtype=torch.float64)
    previous_feature=b['features'][0,0];is_E=previous.clone()
    fields=('features','student_log_probs','effective_valid','memory_before','memory_after',
        'previous_action_probs','previous_features','actions','boxes','omega')
    rows={k:[]for k in fields}
    for t in range(T):
        valid=b['valid'][t]if bool(b['active'][t])else torch.tensor([True,False,False])
        prior=_prior(valid,settings['prior_E']);logprior=prior.log()
        memcos=(memory[None]*b['features'][t]).sum(-1)
        prevcos=(F.normalize(previous_feature,dim=-1,eps=1e-12)[None]*b['features'][t]).sum(-1)
        link=previous@b['link'][t-1]if t else torch.zeros(3,dtype=torch.float64)
        ecos=(b['features'][t,0,None]*b['features'][t]).sum(-1)
        delta=b['u'][t]-b['u'][t,0];c=b['confidence'][t]*b['active'][t]
        features=torch.stack((memcos,prevcos,link,ecos,-b['guard'][t],is_E,delta,c*delta),dim=-1)
        logits=logprior+features@phi;logp=_log_probs(logits,valid)
        action=0 if not bool(b['active'][t])else int((logits+noise[t]).masked_fill(~valid,-torch.inf).argmax())
        if t==forced_node:
            assert bool(b['active'][t]) and bool(valid[forced_action])
            action=int(forced_action)
        probs=F.one_hot(torch.tensor(action),3).double()
        selected=probs@b['features'][t];u=probs@b['u'][t]
        omega=settings['memory_rate']*c*u;before=memory
        if bool(b['active'][t])and bool(omega.detach()!=0):
            memory=F.normalize((1-omega)*memory+omega*selected,dim=-1,eps=1e-12)
        values=dict(features=features,student_log_probs=logp,effective_valid=valid,memory_before=before,
            memory_after=memory,previous_action_probs=previous,previous_features=previous_feature,
            actions=torch.tensor(action),boxes=probs@b['boxes'][t],omega=omega)
        for name in fields:rows[name].append(values[name])
        previous,previous_feature=probs,selected
    path={k:torch.stack(v)for k,v in rows.items()}
    path.update(active=b['active'],observed=b['observed'],noise=noise,phi=phi.clone(),
        forced_node=forced_node,forced_action=forced_action,student_continuation=True)
    return path


def verify_branch(base,path,point,action):
    t=point['node']
    for name in ('features','student_log_probs','effective_valid','memory_before','memory_after',
                 'previous_action_probs','previous_features','actions','boxes','omega'):
        assert torch.equal(path[name][:t],base[name][:t]),name
    for name in ('memory_before','previous_action_probs','previous_features','features','student_log_probs'):
        assert torch.equal(path[name][t],base[name][t]),name
    assert int(path['actions'][t])==action and torch.equal(path['noise'],base['noise'])
    assert torch.equal(path['phi'],base['phi']) and path['student_continuation']
    assert path['actions'][~path['active']].eq(0).all()
    assert torch.equal(path['memory_before'][~path['active']],path['memory_after'][~path['active']])
    return dict(prefix_and_incoming_state_identical=True,common_random_numbers=True,
        same_frozen_student_weights=True,single_forced_action=True,inactive_E_no_memory_write=True,
        later_changed_actions=int((path['actions'][t+1:]!=base['actions'][t+1:]).sum()),
        later_changed_active_unobserved_actions=int(((path['actions'][t+1:]!=base['actions'][t+1:])&
            path['active'][t+1:]&~path['observed'][t+1:]).sum()))


def intervene(bank,phi,cfg,key,frame_ids):
    b=_prepare(bank);noise=query_noise(key,b['T'])
    with torch.no_grad():
        original=rollout(bank,phi,cfg,'student_sample',noise=noise,variant='B',prepared=b)
        original['phi']=phi.clone()
        original['event']=bank.event if hasattr(bank,'event')else b['active']
        base=continue_student(b,phi,cfg,noise)
        for field in ('features','student_log_probs','effective_valid','memory_before','memory_after',
                      'previous_action_probs','previous_features','actions','boxes','omega'):
            assert torch.equal(original[field],base[field]),'Original-policy replay differs: '+field
        points,candidates=select_points(original,frame_ids)
        branches=[]
        for point in points:
            t=point['node'];point={**point,'random_action':random_alternative(key,t,base['effective_valid'][t],point['student_action'])}
            paths={};proof={}
            for name,action in [('Student',point['student_action']),('Teacher',point['teacher_action']),('Random',point['random_action'])]:
                path=continue_student(b,phi,cfg,noise,t,action);proof[name]=verify_branch(base,path,point,action)
                paths[name]=path
            for field in ('actions','boxes','memory_before','memory_after','features','student_log_probs'):
                assert torch.equal(paths['Student'][field],base[field])
                if point['teacher_action']==point['random_action']:
                    assert torch.equal(paths['Teacher'][field],paths['Random'][field])
            branches.append(dict(point=point,paths=paths,proof=proof))
    return dict(base=base,original_feedback=original,branches=branches,eligible_disagreement_nodes=candidates,
        selected_points=points,parameter_updates=0,heavy_forwards=0,decoder_calls=0,expert_calls=0,GT_read=False)
