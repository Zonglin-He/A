"""Offline finite-candidate IoU headroom and E-relative reward ranking only."""
import numpy as np

def xy(boxes,row,ds):
    b=np.asarray(boxes);v=np.concatenate((b[...,:2]-b[...,2:]/2,b[...,:2]+b[...,2:]/2),axis=-1)*np.array([row['input']['width'],row['input']['height']]*2)
    return np.maximum(v,0)if ds=='hc2'else v

def iou(a,b):
    inter=np.maximum(0,np.minimum(a[...,2:],b[...,2:])-np.maximum(a[...,:2],b[...,:2])).prod(-1)
    union=np.maximum(0,a[...,2:]-a[...,:2]).prod(-1)+np.maximum(0,b[...,2:]-b[...,:2]).prod(-1)-inter
    return np.divide(inter,union,out=np.zeros_like(inter),where=union>0)

def auc(good,bad,score):
    a,b=score[good],score[bad]
    if not len(a)or not len(b):return None
    d=a[:,None]-b[None];return float(((d>1e-12)+.5*(abs(d)<=1e-12)).mean())

def oracle(b,valid,E,ids,planned,row,ds,gt,span,interval):
    """Exact DP over independent node candidate choices, including official interpolation.

    All labels used only for this diagnostic. E always included, planned nodes fixed.
    Never a legal TTA algorithm or an unconstrained grounding ceiling.
    """
    b=np.asarray(b);N,K=b.shape[:2];allowed=np.asarray(valid).copy();allowed[:,0]=True
    for t in planned:allowed[t]=False;allowed[t,0]=True
    xx=xy(b,row,ds);ids=np.asarray(ids);gf=np.array(sorted(gt));truth=np.array([gt[int(f)]for f in gf]);inside=(gf>=max(interval[0],span[0]))&(gf<min(interval[1],span[1]));den=max(max(interval[1],span[1])-min(interval[0],span[0]),1)
    values=np.zeros(K);values[~allowed[0]]=-np.inf;parents=[];costs=[]
    for t in range(N-1):
        ix=np.flatnonzero(inside&(gf>=ids[t])&(gf<ids[t+1]));cost=np.zeros((K,K))
        for j in ix:
            alpha=(gf[j]-ids[t])/(ids[t+1]-ids[t]);pred=(1-alpha)*xx[t,:,None]+alpha*xx[t+1,None,:];cost+=iou(pred,truth[j])
        costs.append(cost);matrix=values[:,None]+cost;parent=matrix.argmax(0);values=matrix[parent,np.arange(K)];values[~allowed[t+1]]=-np.inf;parents.append(parent)
    end=np.zeros(K)
    for j in np.flatnonzero(inside&(gf==ids[-1])):end+=iou(xx[-1],truth[j])
    last=int((values+end).argmax());best=float((values+end)[last]/den);choice=[last]
    for parent in reversed(parents):choice.append(int(parent[choice[-1]]))
    choice=np.array(choice[::-1]);out=b[np.arange(N),choice].copy();assert allowed[np.arange(N),choice].all()
    # A separately ordered backward recursion must yield the same optimum.
    back=end.copy();back[~allowed[-1]]=-np.inf
    for t in range(N-2,-1,-1):
        back=(costs[t]+back[None]).max(1);back[~allowed[t]]=-np.inf
    assert abs(back.max()/den-best)<1e-12
    # Every shared-index complete candidate, with E at disallowed/observed nodes,
    # is in this larger product pool and cannot beat the exact optimum.
    return dict(boxes=out,choice=choice,allowed=allowed,optimum_v=best,costs=costs,last_cost=end,denominator=den)

def summarize_frame(quality,reward,prob,valid,base,active,on_quality,step1_quality):
    gain=quality-base;good=valid&(gain>1e-10);bad=valid&(gain< -1e-10);strong=valid&(gain>=.05);harm=valid&(gain<=-.05)
    raw=reward.copy();post=prob.copy();raw[~valid]=-np.inf;post[~valid]=-np.inf
    rawtop=int(raw.argmax())if active else 0;pttop=int(post.argmax())if active else 0
    return dict(E_IoU=float(base),best_gain_pp=100*max(float(gain[valid].max()),0),has_gain_1pp=bool((valid&(gain>=.01)).any()),has_gain_5pp=bool(strong.any()),has_gain_10pp=bool((valid&(gain>=.1)).any()),
        good_candidates=int(good.sum()),bad_candidates=int(bad.sum()),meaningful_good_candidates=int(strong.sum()),meaningful_bad_candidates=int(harm.sum()),valid_candidates=int(valid.sum()),
        raw_auc=auc(good,bad,reward)if active else None,posterior_auc=auc(good,bad,prob)if active else None,
        meaningful_raw_auc=auc(strong,harm,reward)if active else None,meaningful_posterior_auc=auc(strong,harm,prob)if active else None,
        raw_top_gain_pp=100*float(gain[rawtop])if active else None,posterior_top_gain_pp=100*float(gain[pttop])if active else None,
        any_good_above_E_raw=bool((good&(reward>reward[0]+1e-12)).any())if active else None,
        any_good_above_E_posterior=bool((good&(prob>prob[0]+1e-12)).any())if active else None,
        any_strong_above_E_raw=bool((strong&(reward>reward[0]+1e-12)).any())if active else None,
        any_strong_above_E_posterior=bool((strong&(prob>prob[0]+1e-12)).any())if active else None,
        strong_pool_raw_top_good=bool(gain[rawtop]>1e-10)if active and strong.any()else None,
        strong_pool_posterior_top_good=bool(gain[pttop]>1e-10)if active and strong.any()else None,
        strong_pool_raw_top_harmful=bool(gain[rawtop]<-1e-10)if active and strong.any()else None,
        strong_pool_posterior_top_harmful=bool(gain[pttop]<-1e-10)if active and strong.any()else None,
        strong_pool_no_good_outranks_E_raw=not bool((good&(reward>reward[0]+1e-12)).any())if active and strong.any()else None,
        strong_pool_no_good_outranks_E_posterior=not bool((good&(prob>prob[0]+1e-12)).any())if active and strong.any()else None,
        on_gain_pp=100*float(on_quality-base),step1_gain_pp=100*float(step1_quality-base),
        posterior_good_but_On_nonpositive=bool(gain[pttop]>1e-10 and on_quality-base<=1e-10)if active and strong.any()else None)
