"""Detached selective offset preferences and Strong-constrained joint policy.
No model, labels, scorer, dataset roster or experiment directory dependency.
"""
import numpy as np
from scipy.special import logsumexp, expit
import torch
import torch.nn.functional as F


def native_logp(records, logits):
    out=[]
    for rec,z in zip(records,logits):
        if hasattr(z,'detach'): z=z.detach().cpu().numpy()
        z=np.asarray(z);n=len(rec['frame_ids'])
        assert z.dtype==np.float32 and z.shape==(1,n,2) and np.isfinite(z).all()
        ii,jj=np.triu_indices(n,1);v=z[0].astype(np.float64)
        s=v[ii,0]+v[jj,1];out.append(s-logsumexp(s))
    assert len(out)==2
    return out


def evidence(old,stage,hires):
    records=old['records'];assert records==stage['records']==hires['observation']['records']
    dense=native_logp(records,old['logits']['DenseLocal'])
    high=native_logp(records,hires['observation']['logits'])
    layers=[native_logp(records,[z[k]for z in stage['pass_logits'][1]])for k in range(6)]
    ref=[np.logaddexp(high[o]-np.log(2),logsumexp(np.stack([v[o]for v in layers]),axis=0)-np.log(12))for o in range(2)]
    B=old['outputs']['HiResAuxMix']['q'];support=np.argwhere(B>0)
    ids=np.asarray(old['frame_ids']); intervals=np.stack([ids[support[:,0]],ids[support[:,1]]+1],axis=-1)
    mappings=[]
    for rec in records:
        f=np.asarray(rec['frame_ids']);i,j=np.triu_indices(len(f),1)
        a,b=f[i],f[j]+1
        overlap=np.maximum(0,np.minimum(intervals[:,1,None],b)-np.maximum(intervals[:,0,None],a))
        union=intervals[:,1,None]-intervals[:,0,None]+b-a-overlap
        mappings.append((overlap/union).argmax(axis=1))
    return dict(logD=dense,logB=ref,mapping=np.stack(mappings),support=support,
                local_reference_is_not_global_marginal=True)


def feedback(ev,pairs):
    i,j=pairs.T;maps=ev['mapping'];h=[];d=[];collision=[]
    for o in range(2):
        a,b=maps[o,i],maps[o,j];v=ev['logD'][o][a]-ev['logD'][o][b]
        h.append(v);d.append(v-(ev['logB'][o][a]-ev['logB'][o][b]));collision.append(a==b)
    terms=np.stack([h[0],h[1],d[0],d[1]],axis=-1)
    positive=(terms>0).all(axis=-1);negative=(terms<0).all(axis=-1)
    accepted=positive|negative;m=np.where(positive,terms.min(-1),np.where(negative,terms.max(-1),0.))
    # Binary weight is the user-revised protocol; weak evidence is softened once by sigmoid.
    return dict(terms=terms,t=expit(m),w=accepted.astype(np.float64),m=m,
                mapped=np.stack([maps[:,i],maps[:,j]],axis=-1).transpose(1,0,2),
                collision=np.stack(collision,axis=-1))


def probability(B,r):
    if not np.any(r):return B.copy()
    logp=np.log(B)+r;return np.exp(logp-logsumexp(logp))


def sample(p,u):
    c=np.cumsum(p);c[-1]=1.;return np.searchsorted(c,u,side='right')


def loss_gradient(B,r,pairs,fb,kl_coefficient=1.):
    """Autodiff objective; support and targets are detached by construction."""
    base=torch.from_numpy(B);eta=torch.tensor(r,dtype=torch.float64,requires_grad=True)
    ij=torch.from_numpy(pairs);t=torch.from_numpy(fb['t']);w=torch.from_numpy(fb['w'])
    logpi=torch.log_softmax(base.log()+eta,0);pi=logpi.exp();v=eta[ij[:,0]]-eta[ij[:,1]]
    ent=torch.special.xlogy(t,t)+torch.special.xlogy(1-t,1-t)
    pref=(w*(F.softplus(v)-t*v+ent)).mean()
    kl=(pi*eta).sum()-torch.logsumexp(base.log()+eta,0)+torch.logsumexp(base.log(),0);loss=pref+kl_coefficient*kl
    loss.backward();g=eta.grad.detach().numpy().copy()
    return dict(gradient=g,loss=float(loss.detach()),preference_KL=float(pref.detach()),global_KL=float(kl.detach()))


def manual(B,r,pairs,fb,kl_coefficient=1.):
    p=probability(B,r);g=kl_coefficient*p*(r-p@r)
    a=(expit(r[pairs[:,0]]-r[pairs[:,1]])-fb['t'])*fb['w']/len(pairs)
    np.add.at(g,pairs[:,0],a);np.add.at(g,pairs[:,1],-a)
    return g


def fit(Bgrid,ev,mode,seed,steps=40,lr=.1,check=lambda:None,kl_coefficient=1.):
    support=ev['support'];B=Bgrid[tuple(support.T)];assert abs(B.sum()-1)<1e-12
    rng=np.random.default_rng(seed);uniforms=rng.random((steps,32,2))
    bank=np.stack([sample(B,uniforms[:,:,0]),sample(B,uniforms[:,:,1])],axis=-1)
    bankfb=feedback(ev,bank.reshape(-1,2));r=np.zeros(len(B));trace=[];m1=r.copy();m2=r.copy();maxerror=0.
    for k in range(steps):
        check();before=r.copy();p=probability(B,r)
        if mode=='Direct':pairs=bank.reshape(-1,2);fb=bankfb
        else:
            pairs=bank[k].copy()
            if mode=='Live':pairs[:,0]=sample(p,uniforms[k,:,0])
            fb=feedback(ev,pairs)
        calc=loss_gradient(B,r,pairs,fb,kl_coefficient);g=calc['gradient']
        err=float(abs(g-manual(B,r,pairs,fb,kl_coefficient)).max());assert err<1e-12;maxerror=max(maxerror,err)
        m1=.9*m1+.1*g;m2=.999*m2+.001*g*g
        update=lr*(m1/(1-.9**(k+1)))/(np.sqrt(m2/(1-.999**(k+1)))+1e-8)
        r=r-update;r-=r.mean()
        trace.append(dict(step=k,eta_before=before,eta_after=r.copy(),pi_before=p,pairs=pairs.copy(),
            uniforms=uniforms[k].copy(),feedback={x:y.copy()for x,y in fb.items()},
            adam_first=m1.copy(),adam_second=m2.copy(),**calc))
    q=np.zeros_like(Bgrid);q[tuple(support.T)]=probability(B,r)
    return dict(q=q,eta=r,trace=trace,bank=bank,bank_feedback=bankfb,mode=mode,seed=seed,
                actual_updates=steps,scheduled_updates=steps,max_manual_gradient_error=maxerror,
                full_support_size=len(B),new_network_updates=0,kl_coefficient=kl_coefficient)


def properties():
    B=np.array([.2,.3,.5]);r=np.array([.1,-.05,.02]);pairs=np.array([[0,1],[1,2],[0,0]])
    ev=dict(mapping=np.array([[0,1,2],[0,1,2]]),logD=[np.log([.1,.3,.6])]*2,logB=[np.log(B)]*2)
    fb=feedback(ev,pairs);g=loss_gradient(B,r,pairs,fb)['gradient'];assert np.max(abs(g-manual(B,r,pairs,fb,kl_coefficient)))<1e-12
    reverse=feedback(ev,pairs[:,::-1]);assert np.allclose(reverse['t'],1-fb['t'])and np.array_equal(reverse['w'],fb['w'])
    assert not fb['w'][-1] and np.array_equal(probability(B,np.zeros(3)),B)
    equal=dict(ev,logD=ev['logB']);f0=feedback(equal,pairs);assert not f0['w'].any()
    assert np.max(abs(loss_gradient(B,np.zeros(3),pairs,f0)['gradient']))<1e-15
    epsilon=1e-6;numeric=[]
    for i in range(3):
        delta=np.eye(3)[i]*epsilon
        numeric.append((loss_gradient(B,r+delta,pairs,fb)['loss']-loss_gradient(B,r-delta,pairs,fb)['loss'])/(2*epsilon))
    assert max(abs(g-numeric))<1e-9
    return dict(status='pass',binary_weights=True,reversal=True,zero_state_exact=True,same_reference_abstains=True,
                same_native_interval_abstains=True,autograd_manual_maxerror=float(max(abs(g-manual(B,r,pairs,fb)))),
                finite_difference_maxerror=float(max(abs(g-numeric))))
