"""Exact observable-interval prior/target/likelihood; original512 optimizer/trust."""
import inspect,math
import numpy as np
from scipy.special import logsumexp
from scripts import stvg_temporal_hypothesis_fit32_v1 as old
native=old.native;original=old.original;CFG=dict(old.CFG)
def groups(off,interval):
    """Four disjoint endpoint-supply events, covering every merged latent state."""
    a,b=interval;assert a<b and len(off)==2;ss=[];ee=[]
    for o in off:
        ids=np.asarray(o['ids']);ss.append(ids[o['ij'][0]]);ee.append(ids[o['ij'][1]]+1)
    inside=[(s>=a)&(e<=b)for s,e in zip(ss,ee)]
    masks0=[inside[0]&(ss[0]==a)&(ee[0]==b),inside[0]&(ss[0]==a)&(ee[0]<b),inside[0]&(ss[0]>a)&(ee[0]==b),inside[0]&(ss[0]>a)&(ee[0]<b)]
    masks1=[inside[1],inside[1]&(ee[1]==b),inside[1]&(ss[1]==a),inside[1]&(ss[1]==a)&(ee[1]==b)]
    result=[(np.flatnonzero(m0),np.flatnonzero(m1))for m0,m1 in zip(masks0,masks1)if m0.any()and m1.any()];assert result,'fixed interval must have Native support';return result
def event_logweight(off,scores,event,gradient=True):
    terms=[];means=[]
    for pair in event:
        value=0.;g=np.zeros((256,2))
        for o,z,ix in zip(off,scores,pair):
            v=z[ix];norm=float(logsumexp(v));value+=norm
            if gradient:
                w=np.exp(v-norm);s=np.bincount(o['ij'][0,ix],weights=w,minlength=len(o['hidden']));e=np.bincount(o['ij'][1,ix],weights=w,minlength=len(o['hidden']));g[:,0]+=o['hidden'].T@s;g[:,1]+=o['hidden'].T@e
        terms.append(value);means.append(g)
    total=float(logsumexp(terms));g=sum((math.exp(t-total)*v for t,v in zip(terms,means)),np.zeros((256,2)))if gradient else None
    return total,g
def prepare(off,hypotheses,rewards):
    events=[groups(off,h['interval_frames'])for h in hypotheses];scores=[o['base']for o in off];lw=np.array([event_logweight(off,scores,e,False)[0]for e in events]);p0,lp0=original._prob(lw)if len(lw)else(np.empty(0),np.empty(0));valid=len(hypotheses)==2 and len(rewards)==2 and all(r is not None and math.isfinite(r)for r in rewards)and rewards[0]!=rewards[1];q,lq=original._prob(lw+np.asarray(rewards)/CFG['tau_T'])if valid else(p0.copy(),lp0.copy());Z=sum(float(logsumexp(o['base']))for o in off)
    return dict(offsets=off,events=events,candidate_base=lw,q=q,lq=lq,no_information=not valid,Native_conditional=p0,Native_interval_probabilities=np.exp(lw-Z))
def loss_gradient(pre,theta_flat,variant,cfg):
    th=np.asarray(theta_flat).reshape(256,2);scores=[original._scores_for_offset(o,th)for o in pre['offsets']];pairs=[event_logweight(pre['offsets'],scores,e)for e in pre['events']];g=np.zeros_like(th);cand=0.;trust=0.
    if pairs:
        p,lp=original._prob(np.array([v for v,_ in pairs]));cand=original._kl(pre['q'],pre['lq'],lp)
        for d,(_,v)in zip(p-pre['q'],pairs):g+=d*v
    for o,z in zip(pre['offsets'],scores):
        p0,lp0=original._prob(o['base']);p,lp=original._prob(z);trust+=original._kl(p0,lp0,lp);d=p-p0;s=np.bincount(o['ij'][0],weights=d,minlength=len(o['hidden']));e=np.bincount(o['ij'][1],weights=d,minlength=len(o['hidden']));g[:,0]+=o['hidden'].T@s;g[:,1]+=o['hidden'].T@e
    L=cand+trust;assert math.isfinite(L)and np.isfinite(g).all();return L,g.reshape(-1),dict(candidate_kl=cand,native_grid_trust_kl=trust)
def fit(off,hypotheses,rewards):
    pre=prepare(off,hypotheses,rewards);ns=dict(original.__dict__);ns['_loss_gradient']=loss_gradient;exec(compile(inspect.getsource(original._lbfgs_armijo),__file__,'exec'),ns);s=ns['_lbfgs_armijo'](pre,'marginal_interval',CFG);iv,logits=native.native_decode(off,s['x']);path=[dict(theta=x.reshape(256,2),loss=L,terms=v,step_size=step)for x,L,v,step in s['path']];assert all(path[i+1]['loss']<path[i]['loss']for i in range(len(path)-1))
    if pre['no_information']:assert s['accepted']==0 and np.count_nonzero(s['x'])==0
    return dict(status='updated'if s['accepted']else'exact_zero_native_noop',theta=s['x'].reshape(256,2),path=path,optimizer_steps=s['accepted'],rejected_line_search_trials=s['rejected'],termination=s['termination'],final_interval=iv,final_logits=logits,no_information=pre['no_information'],hypotheses=hypotheses,rewards=rewards,Teacher_conditional=pre['q'],Native_conditional=pre['Native_conditional'],Native_interval_probabilities=pre['Native_interval_probabilities'],candidate_probability_scope='sum of every latent offset-state pair merging to each fixed physical interval',config=CFG,GT_read=False)
def brute_pairs(off,hypotheses):
    s=[np.asarray(o['ids'])[o['ij'][0]]for o in off];e=[np.asarray(o['ids'])[o['ij'][1]]+1 for o in off];start=np.minimum(s[0][:,None],s[1][None,:]);end=np.maximum(e[0][:,None],e[1][None,:]);return [tuple(np.nonzero((start==h['interval_frames'][0])&(end==h['interval_frames'][1])))for h in hypotheses]
def independent_torch(off,theta,q,pairs):
    """Enumerated merged pairs, no factorization shared with the fitter."""
    import torch
    th=torch.tensor(theta,dtype=torch.float64,requires_grad=True);scores=[];L=th.sum()*0
    for o in off:
        h=torch.tensor(o['hidden']);ij=torch.tensor(o['ij']);z=torch.tensor(o['base']);v=z+(h@th[:,0])[ij[0]]+(h@th[:,1])[ij[1]];scores.append(v);L+=(z.softmax(0)*(z.log_softmax(0)-v.log_softmax(0))).sum()
    lw=torch.stack([torch.logsumexp(scores[0][torch.tensor(a)]+scores[1][torch.tensor(b)],0)for a,b in pairs])if pairs else torch.empty(0,dtype=torch.float64)
    if len(q):
        qt=torch.tensor(q);pos=qt>0;L+=(qt[pos]*(qt[pos].log()-lw.log_softmax(0)[pos])).sum()
    gradient=torch.autograd.grad(L,th)[0].detach().numpy();return float(L.detach()),gradient,lw.detach().numpy()
def audit(off,result):
    pre=prepare(off,result['hypotheses'],result['rewards']);pairs=brute_pairs(off,result['hypotheses']);count=0
    for row in result['path']:
        L,g,lw=independent_torch(off,row['theta'],pre['q'],pairs);actual,grad,_=loss_gradient(pre,row['theta'].reshape(-1),'marginal_interval',CFG);assert abs(L-actual)<1e-10 and abs(actual-row['loss'])<1e-10 and np.max(abs(g.reshape(-1)-grad))<1e-9
        if count==0 and len(lw):assert np.max(abs(original._prob(lw)[0]-result['Native_conditional']))<1e-12
        count+=1
    return count
