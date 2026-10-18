"""Profiled assignment working distribution, NOT a calibrated identity posterior.

All computation is outside the frozen backbone. A latent state selects at most
one retained candidate per admitted frame, or pays the existing outlier cost.
Its conditional least-squares correction and joint E energy define a discrete
working distribution. Exact duplicate corrections marginalize assignment mass.
"""
import itertools
import numpy as np
from vg_tta.stvg_dino_supported_correction32_v1 import relative_cost
from vg_tta.stvg_joint_robust_correction32_v1 import apply


def hypotheses(pools,scores,scale,cutoff,beta=1/16):
    pools=[np.asarray(v,np.float64).reshape(-1,4)for v in pools]
    costs=[relative_cost(s)for s in scores];scale=np.asarray(scale,np.float64)
    assert len(pools)==len(costs)and len(pools)<=4 and all(0<len(v)<=3 for v in pools)
    assert scale.shape==(4,)and(scale>0).all()and cutoff>0 and beta==1/16
    rows=[]
    for assignment in itertools.product(*[range(-1,len(v))for v in pools]):
        active=[v[k]for v,k in zip(pools,assignment)if k>=0]
        delta=np.mean(active,0)if active else np.zeros(4)
        energy=sum(cutoff if k<0 else float(np.square((v[k]-delta)/scale).sum()+beta*cutoff*c[k])for v,c,k in zip(pools,costs,assignment))
        rows.append(dict(assignment=assignment,delta=delta,energy=float(energy)))
    energies=np.array([v['energy']for v in rows]);logw=-(energies-energies.min())/cutoff
    w=np.exp(logw);w/=w.sum()
    indices={};deltas=[];weights=[];membership=[]
    for i,(r,mass)in enumerate(zip(rows,w)):
        d=r['delta'].copy();d[d==0]=0.;key=d.tobytes()
        if key not in indices:indices[key]=len(deltas);deltas.append(d);weights.append(0.);membership.append([])
        j=indices[key];weights[j]+=float(mass);membership[j].append(i)
    weights=np.asarray(weights);weights/=weights.sum()
    return dict(deltas=np.array(deltas),weights=weights,assignment_weights=w,assignments=rows,memberships=membership,
        effective_unique_hypotheses=float(1/np.square(weights).sum()),energy_temperature=float(cutoff),
        all_outlier_canonical_Source=True,duplicate_correction_mass_summed=True)


def time_weights(frame_ids,interval):
    ids=np.asarray(frame_ids,np.float64);assert len(ids)>=2 and (np.diff(ids)>0).all()
    edges=np.r_[ids[0],(ids[:-1]+ids[1:])/2,ids[-1]+1]
    weights=np.maximum(0,np.minimum(edges[1:],interval[1])-np.maximum(edges[:-1],interval[0]))
    assert weights.sum()>0;return weights/weights.sum()


def tube_utilities(tubes,frame_ids,interval):
    """Physical-cell-weighted spatial IoU on the shared MBR event support.

All candidate tubes have the same temporal interval. This is an actual-sample
grid approximation to complete-tube overlap, without GT or label visibility.
Boxes are not clamped or merged across identities. No benchmark interpolation
or unobserved annotation support is used to define this decision utility.
"""
    b=np.asarray(tubes,np.float64);assert b.ndim==3 and b.shape[2]==4
    assert np.isfinite(b).all()and(b[:,:,2:]>0).all()
    weights=time_weights(frame_ids,interval);n=len(b);out=np.zeros((n,n))
    xy=np.concatenate((b[:,:,:2]-b[:,:,2:]/2,b[:,:,:2]+b[:,:,2:]/2),2)
    for t,wt in enumerate(weights):
        if wt==0:continue
        x=xy[:,t];a=np.maximum(0,x[:,2:]-x[:,:2]).prod(1)
        inter=np.maximum(0,np.minimum(x[:,None,2:],x[None,:,2:])-np.maximum(x[:,None,:2],x[None,:,:2])).prod(2)
        union=a[:,None]+a[None,:]-inter
        out+=wt*np.divide(inter,union,out=np.zeros_like(inter),where=union>0)
    assert np.max(abs(out-out.T))<1e-12 and np.max(abs(np.diag(out)-1))<1e-10
    return out


def decide(base,pools,scores,scale,cutoff,frame_ids,interval):
    h=hypotheses(pools,scores,scale,cutoff)
    tubes=np.stack([apply(base,d)for d in h['deltas']]);utility=tube_utilities(tubes,frame_ids,interval)
    expected=utility@h['weights'];best=int(np.argmax(expected))
    mean_delta=h['weights']@h['deltas']
    return dict(distribution=h,candidate_tubes=tubes,utility=utility,expected_utilities=expected,
        selected_index=best,mean_delta=mean_delta,mean_output=apply(base,mean_delta),MBR_output=tubes[best].copy())


def contract():
    from vg_tta.stvg_dino_supported_correction32_v1 import estimate
    b=np.array([[.3,.4,.15,.25],[.4,.45,.16,.26],[.5,.5,.17,.27]])
    pools=[np.array([[.1,.0,.2,.1],[-.1,.0,.0,-.1]]),np.array([[.1,.0,.2,.1],[-.1,.0,.0,-.1]])]
    scores=[np.array([.8,.2]),np.array([.8,.2])];s=np.ones(4);cap=.5
    h=hypotheses(pools,scores,s,cap);e=estimate(pools,scores,s,cap,1/16)
    assert abs(min(x['energy']for x in h['assignments'])-e['objective'])<1e-12
    assert len(h['assignments'])==9 and abs(h['weights'].sum()-1)<1e-14
    d=decide(b,pools,scores,s,cap,[0,3,8],[0,9]);assert np.array_equal(d['MBR_output'],d['candidate_tubes'][d['selected_index']])
    empty=decide(b,[],[],s,cap,[0,3,8],[0,9]);assert np.array_equal(empty['mean_output'],b)and np.array_equal(empty['MBR_output'],b)
    # Independent scalar xyxy IoU and physical weights [1.5,4,3.5]/9.
    weights=np.array([1.5,4,3.5])/9;mx=0.
    for i,x in enumerate(d['candidate_tubes']):
        for j,y in enumerate(d['candidate_tubes']):
            val=0.
            for t,wt in enumerate(weights):
                p=np.r_[x[t,:2]-x[t,2:]/2,x[t,:2]+x[t,2:]/2];q=np.r_[y[t,:2]-y[t,2:]/2,y[t,:2]+y[t,2:]/2]
                area=float(max(0,min(p[2],q[2])-max(p[0],q[0]))*max(0,min(p[3],q[3])-max(p[1],q[1])))
                val+=wt*area/(x[t,2]*x[t,3]+y[t,2]*y[t,3]-area)
            mx=max(mx,abs(val-d['utility'][i,j]))
    assert mx<1e-12
    return dict(status='pass',count=5,E_global_energy_exact=True,marginal_mass_sums_one=True,actual_candidate_MBR=True,empty_exact_noop=True,independent_utility_max_error=mx)
