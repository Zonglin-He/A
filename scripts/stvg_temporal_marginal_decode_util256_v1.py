"""Exact physical-interval MAP from actual full-grid FP32 endpoint logits."""
import numpy as np
from scipy.special import logsumexp

def probabilities(records,logits):
    grids=[]
    for rec,z in zip(records,logits):
        if hasattr(z,'detach'):z=z.detach().cpu().numpy()
        z=np.asarray(z);assert z.dtype==np.float32 and z.shape==(1,len(rec['frame_ids']),2)and np.isfinite(z).all();v=z[0].astype(np.float64);ij=np.asarray(np.triu_indices(len(v),1));score=v[ij[0],0]+v[ij[1],1];p=np.exp(score-logsumexp(score));assert abs(float(p.sum())-1)<1e-12;grids.append(dict(ids=list(map(int,rec['frame_ids'])),ij=ij,p=p))
    assert len(grids)==2;return grids

def mass_table(grids):
    ids=sorted({x for g in grids for x in g['ids']});idx={x:i for i,x in enumerate(ids)};n=len(ids);s=[];e=[]
    for g in grids:s.append(np.array([idx[g['ids'][i]]for i in g['ij'][0]]));e.append(np.array([idx[g['ids'][i]]for i in g['ij'][1]]))
    key=np.minimum(s[0][:,None],s[1][None,:])*n+np.maximum(e[0][:,None],e[1][None,:]);w=grids[0]['p'][:,None]*grids[1]['p'][None,:];M=np.bincount(key.ravel(),weights=w.ravel(),minlength=n*n).reshape(n,n);assert np.isfinite(M).all()and (M>=0).all()and abs(float(M.sum())-1)<1e-12;return ids,M

def mode(ids,M):
    # Fixed physical row-major ties, no GT, clamp, top-k approximation or support restriction.
    q=int(np.argmax(M));a,b=divmod(q,len(ids));assert a<b and M[a,b]>0;return [ids[a],ids[b]+1],float(M[a,b])

def independent_cdf(grids,ids):
    idx={v:i for i,v in enumerate(ids)};n=len(ids);cs=[]
    for g in grids:
        H=np.zeros((n,n))
        for (i,j),p in zip(g['ij'].T,g['p']):H[idx[g['ids'][i]],idx[g['ids'][j]]]+=p
        cs.append(H[::-1,:].cumsum(0)[::-1,:].cumsum(1))
    q=np.pad(cs[0]*cs[1],((0,1),(1,0)));return q[:-1,1:]-q[1:,1:]-q[:-1,:-1]+q[1:,:-1]

def decode(records,logits):
    grids=probabilities(records,logits);ids,M=mass_table(grids);iv,value=mode(ids,M);return iv,value,ids,M,grids
