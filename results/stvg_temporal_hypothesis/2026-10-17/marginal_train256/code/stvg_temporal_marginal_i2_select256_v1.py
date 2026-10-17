"""Only I2 ranking changes: physical Native interval mass, original latent tie."""
import numpy as np
from scripts import stvg_temporal_hypothesis_fit32_v1 as f
def select(off,available):
    import torch
    old=f.select(off,available)
    if not old['hypotheses']:return dict(old,marginal_no_update_interval=old['Native_interval'])
    ids=sorted({int(x)for o in off for x in o['ids']});idx={v:i for i,v in enumerate(ids)};n=len(ids)
    ps=[f.original._prob(o['base'])[0]for o in off]
    starts=[np.array([idx[o['ids'][i]]for i in o['ij'][0]])for o in off];ends=[np.array([idx[o['ids'][i]]for i in o['ij'][1]])for o in off]
    keys=np.minimum(starts[0][:,None],starts[1][None,:])*n+np.maximum(ends[0][:,None],ends[1][None,:]);weights=ps[0][:,None]*ps[1][None,:]
    mass=np.bincount(keys.ravel(),weights=weights.ravel(),minlength=n*n);assert abs(float(mass.sum())-1)<1e-12
    q=int(mass.argmax());a,b=divmod(q,n);mode=[ids[a],ids[b]+1]
    chosen=None
    for q in np.argsort(-mass,kind='stable'):
        if mass[q]<=0:continue
        a,b=divmod(int(q),n)
        if a>=b:continue
        iv=[ids[a],ids[b]+1];sf=f.sampled(iv,available)
        if sf is not None and sf!=old['hypotheses'][0]['sampled_physical_frame_ids']:chosen=(int(q),iv,sf);break
    one=dict(old['hypotheses'][0]);hs=[one]
    if chosen is not None:
        # Representative state uses exactly the original FP32 endpoint log-softmax
        # ranking/tie rule. It is NOT replaced by the marginal probability in fit.
        scores=[]
        for o in off:
            z=torch.tensor(o['native_logits'],dtype=torch.float32);z=z[:,0].log_softmax(0)[:,None]+z[:,1].log_softmax(0)[None,:];a,b=o['ij'];scores.append(z[a,b].numpy().astype(float))
        joint=scores[0][:,None]+scores[1][None,:];flat=int(np.argmax(np.where(keys==chosen[0],joint,-np.inf)));i,j=divmod(flat,len(scores[1]));pairs=[o['ij'][:,k].tolist()for o,k in zip(off,[i,j])]
        h=dict(pairs=pairs,interval_frames=chosen[1],sampled_physical_frame_ids=chosen[2]);h['native_joint_score']=sum(float(o['native_logits'][p[0],0]+o['native_logits'][p[1],1])for o,p in zip(off,pairs));assert f.merge(off,pairs)==h['interval_frames'];hs.append(h)
    return dict(status='two_distinct_observations'if len(hs)==2 else'no_distinct_sampled_hypothesis',hypotheses=hs,Native_interval=old['Native_interval'],marginal_no_update_interval=mode,probability_scope=old['probability_scope'],selection='only I2 physical interval ranking uses exact pushed-forward Native mass; descending mass then row-major physical ties; representative latent state uses original FP32 ranking',marginal_mode_mass=float(mass.max()),I2_marginal_mass=float(mass[chosen[0]])if chosen else None)
