"""Common valid-area and ID safeguards; fixed-budget A/B/C token policies.

A implements SemVID-style query/object-MMR, transition and context allocation
using paper equations and released constants. It is an STVG adaptation, not a
bitwise reproduction of the separate upstream model. B/C share exact quotas,
ID/context/object selections; C changes only transition-slot selection.
"""
import math
import numpy as np

def norm(a):
    a=np.asarray(a,dtype=np.float32);return a/np.maximum(np.linalg.norm(a,axis=-1,keepdims=True),1e-6)
def rank(x,valid):
    out=np.full(len(x),-np.inf);ix=np.flatnonzero(valid);s=np.asarray(x)[ix];order=np.argsort(s,kind='stable');i=0
    while i<len(ix):
        j=i+1
        while j<len(ix)and s[order[j]]==s[order[i]]:j+=1
        out[ix[order[i:j]]]=(i+j-1)/2/max(1,len(ix)-1);i=j
    return out
def top(score,k,valid,used=()):
    ix=[int(i)for i in np.flatnonzero(valid)if int(i)not in set(map(int,used))]
    return sorted(ix,key=lambda i:(-float(score[i]),i))[:k]
def mmr(score,z,k,valid,used=()):
    selected=[];eligible=set(map(int,np.flatnonzero(valid)))-set(map(int,used))
    while eligible and len(selected)<k:
        old=list(map(int,used))+selected
        i=min(eligible,key=lambda i:(-(.5*float(score[i])-.5*(float((z[i]@z[old].T).max())if old else 0)),i));selected.append(i);eligible.remove(i)
    return selected
def budget(weights,floor,cap,total):
    f=np.asarray(floor,dtype=int).copy();assert f.sum()<=total<=sum(cap)
    w=np.maximum(weights,0)
    if w.sum()<1e-8:w=np.ones_like(w)
    while f.sum()<total:
        eligible=np.flatnonzero(f<cap);i=min(eligible,key=lambda i:(-(w[i]/(f[i]-floor[i]+1)),int(i)));f[i]+=1
    return f
def text_rows(z):
    # Match the upstream default's >50 token mean-pooling branch.
    z=np.asarray(z,dtype=np.float32)
    return norm(z.mean(0,keepdims=True)if len(z)>50 else z)
def select(features,q,c,valid,mandatory,groups):
    T,P,D=features.shape;assert valid.shape==(T,P)and mandatory.shape==valid.shape and groups.shape==valid.shape
    z=norm(features);qt,ct=text_rows(q),text_rows(c)
    def similarity(text):
        a=z@text.T;k=min(2,len(text));return np.sort(a,axis=-1)[...,-k:].mean(-1)
    aq,ac=similarity(qt),similarity(ct);rq=np.stack([rank(aq[t],valid[t])for t in range(T)]);rc=np.stack([rank(ac[t],valid[t])for t in range(T)])
    consensus=np.minimum(rq,rc);difference=np.full_like(rq,-np.inf);difference[valid]=np.abs(rq[valid]-rc[valid]);joint=np.full_like(rq,-np.inf);joint[valid]=.75*consensus[valid]+.25*difference[valid]
    global_z=np.stack([norm(features[t,valid[t]].mean(0))for t in range(T)]);frame=np.sort(global_z@qt.T,axis=-1)[...,-min(2,len(qt)):].mean(-1);change=np.linalg.norm(global_z[1:]-global_z[:-1],axis=-1);ec=np.r_[change[:1],change]if T>1 else np.zeros(1)
    floor=mandatory.sum(1)+2;total=int(math.ceil(.125*T*P));caps=valid.sum(1);assert (floor<=caps).all(),(floor,caps)
    quotas=budget(.7*frame+.3*ec,floor,caps,total)
    # State/transition score, after excluding padding from normalization.
    motion=np.zeros((T,P),np.float32)
    for t in range(T):
        vals=[np.linalg.norm(features[t]-features[u],axis=-1)for u in (t-1,t+1)if 0<=u<T];motion[t]=np.mean(vals,axis=0)if vals else 0
    xy=np.stack(np.meshgrid((np.arange(16)+.5)/16,(np.arange(16)+.5)/16),-1).reshape(P,2);connection=np.zeros((T,P),np.float32);edges=[]
    for t in range(T-1):
        if float(global_z[t]@global_z[t+1])<.5:continue
        sim=z[t]@z[t+1].T;allowed=valid[t,:,None]&valid[t+1,None,:]&(groups[t,:,None]==groups[t+1,None,:])&(np.linalg.norm(xy[:,None]-xy[None,:],axis=-1)<=.25)
        score=np.where(allowed,sim,-np.inf);a=score.argmax(1);b=score.argmax(0)
        for i,j in enumerate(a):
            if not np.isfinite(score[i,j])or b[j]!=i:continue
            value=max(float(score[i,j]),0)*(1-float(np.linalg.norm(xy[i]-xy[j])));connection[t,i]=max(connection[t,i],value);connection[t+1,j]=max(connection[t+1,j],value);edges.append([t,i,int(j),value,int(groups[t,i])])
    result={a:[]for a in ('A','B','C')};quota_roles=[];records=[]
    for t in range(T):
        fixed=list(map(int,np.flatnonzero(mandatory[t])));proto=top(z[t]@global_z[t],1,valid[t],fixed);base=fixed+proto;remaining=int(quotas[t])-len(base);ko=int(round(.8*remaining));km=remaining-ko
        # Same spatial extent/ID integrity safeguards in all arms. B/C add
        # per-candidate evidence diversity without changing candidate geometry.
        qa=rank(aq[t],valid[t]);mo=rank(motion[t],valid[t]);mot_a=np.full(P,-np.inf);mot_a[valid[t]]=.5*mo[valid[t]]+.5*qa[valid[t]]
        obj_a=mmr(qa,z[t],ko,valid[t],base);mot_a_ix=top(mot_a,km,valid[t],base+obj_a)
        obj_b=[]
        for gid in sorted(set(map(int,groups[t,valid[t]]))):
            if gid<=0 or len(obj_b)>=ko:continue
            eligible=valid[t]&(groups[t]==gid);pick=top(joint[t],1,eligible,base+obj_b);obj_b+=pick
        obj_b+=mmr(joint[t],z[t],ko-len(obj_b),valid[t],base+obj_b)
        state=np.full(P,-np.inf);state[valid[t]]=.5*mo[valid[t]]+.25*consensus[t,valid[t]]+.25*difference[t,valid[t]]
        mot_b=top(state,km,valid[t],base+obj_b)
        cr=rank(connection[t],valid[t]);bridge=np.full(P,-np.inf);bridge[valid[t]]=.5*state[valid[t]]+.5*cr[valid[t]];mot_c=top(bridge,km,valid[t],base+obj_b)
        for arm,ob,mt in [('A',obj_a,mot_a_ix),('B',obj_b,mot_b),('C',obj_b,mot_c)]:
            ix=sorted(base+ob+mt);assert len(ix)==int(quotas[t])==len(set(ix))and valid[t,ix].all()and set(fixed)<=set(ix);result[arm].append(ix)
        quota_roles.append(dict(merged_frame=t,total=int(quotas[t]),ID=len(fixed),context=len(proto),object=ko,transition=km))
        records.append(dict(B_C_object_indices_equal=True,B_C_fixed_indices_equal=True,transition_changed=mot_b!=mot_c))
    return dict(indices={k:np.concatenate([np.asarray(ix,dtype=int)+t*P for t,ix in enumerate(v)])for k,v in result.items()},quotas=quotas.tolist(),role_quotas=quota_roles,edges=edges,comparability=records,total_tokens_before=T*P,total_tokens_kept=int(quotas.sum()),padding_excluded=True)

def contracts():
    rng=np.random.default_rng(199);z=rng.normal(size=(3,256,24)).astype(np.float32);q=rng.normal(size=(8,24));c=rng.normal(size=(60,24));valid=np.ones((3,256),bool);valid[:,:32]=False;mandatory=np.zeros_like(valid);mandatory[:,[40,120]]=True;groups=np.tile(np.arange(256)//32,(3,1));a=select(z,q,c,valid,mandatory,groups);b=select(z,q,c,valid,mandatory,groups)
    assert all(np.array_equal(a['indices'][k],b['indices'][k])for k in a['indices']);assert a['total_tokens_kept']==96
    for k in a['indices']:
        assert all(valid.reshape(-1)[a['indices'][k]])and set(np.flatnonzero(mandatory))<=set(a['indices'][k])
    assert sum(x['total']for x in a['role_quotas'])==96;assert all(x['B_C_object_indices_equal']for x in a['comparability'])
    assert text_rows(c).shape==(1,24);assert text_rows(q).shape==(8,24)
    return dict(status='pass',count=7,CPU_only=True,GT_read=False,scope='determinism, exact equal budgets, valid-area exclusion, ID inclusion, B/C shared roles, long/short text branches')
