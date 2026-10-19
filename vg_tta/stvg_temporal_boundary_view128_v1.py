"""Full-span global anchors plus boundary-associated-frame observations.
Frozen weights; student posterior determines WHICH real frames the teacher sees.
"""
import numpy as np
from vg_tta.stvg_posterior_decision_tree_v1 import final_mbr


def sampling(ids,q,phase=.25):
    ids=np.asarray(ids,int);q=np.asarray(q,float);assert len(ids)==65 and q.shape==(65,65)and(q>=0).all()and abs(q.sum()-1)<1e-10 and np.all(np.diff(ids)>0)
    edges=np.r_[float(ids[0]),(ids[:-1]+ids[1:])/2,float(ids[-1]+1)];mass=(q.sum(1)+q.sum(0))/2;mass=mass/mass.sum();cdf=np.cumsum(mass);cdf[-1]=1.;global_ids=ids[::2].tolist();available=np.arange(ids[0],ids[-1]+1);free=~np.isin(available,global_ids);extra=[];targets=[]
    for u in (np.arange(32)+phase)/32:
        k=int(np.searchsorted(cdf,u,side='right'));left=cdf[k-1]if k else 0.;fraction=(u-left)/mass[k];t=edges[k]+fraction*(edges[k+1]-edges[k]);pool=available[free];chosen=int(pool[np.argmin(abs(pool-t))]);extra.append(chosen);targets.append(float(t));free[chosen-ids[0]]=False
    selected=sorted(global_ids+extra);assert len(selected)==len(set(selected))==65 and selected[0]==ids[0]and selected[-1]==ids[-1]
    nearest=np.array([int(np.argmin(abs(ids-t)))for t in selected]);return selected,dict(global_anchors=global_ids,adaptive_frames=extra,quantile_targets=targets,phase=phase,domain=[int(ids[0]),int(ids[-1])+1],new_frames_vs_original=int(sum(t not in set(ids)for t in selected)),nearest_original_indices=nearest.tolist(),boundary_frame_mass=mass.tolist(),end_boundary_is_associated_last_included_frame=True)


def project_logits(records,logits,old_records):
    """Diagnostic teacher feedback on old endpoint states; not probability transport."""
    import torch
    result=[]
    for rec,ll,old in zip(records,logits,old_records):
        x=np.array(rec['frame_ids']);y=ll.detach().cpu().numpy()[0];t=np.array(old['frame_ids']);v=np.stack([np.interp(t,x,y[:,k])for k in range(2)],-1).astype(np.float32);result.append(torch.from_numpy(v[None]))
    return result


def cross_utility(decision_ids,hypothesis_ids,q):
    """Same temporal MBR loss; old actions and teacher hypotheses can differ."""
    di=np.array(decision_ids);hi=np.array(hypothesis_ids);a,b=np.triu_indices(len(di),1);c,d=np.triu_indices(len(hi),1);S=di[a];E=di[b]+1;HS=hi[c];HE=hi[d]+1;prob=np.asarray(q)[c,d];assert abs(prob.sum()-1)<1e-10
    out=np.empty(len(S))
    for k in range(0,len(S),64):
        s=S[k:k+64,None];e=E[k:k+64,None];inter=np.maximum(0,np.minimum(e,HE)-np.maximum(s,HS));union=e-s+HE-HS-inter;out[k:k+64]=(inter/union)@prob
    u=np.full((len(di),len(di)),-np.inf);u[a,b]=out;return u


def choose(ids,utility):
    a,b=np.unravel_index(int(np.argmax(utility)),utility.shape);return [int(ids[a]),int(ids[b])+1]


def properties():
    from vg_tta.stvg_posterior_decision_tree_v1 import expected_iou_fast
    import torch
    rng=np.random.default_rng(41);ids=np.rint(np.linspace(0,499,65)).astype(int);q=np.triu(rng.random((65,65)),1);q/=q.sum();new,meta=sampling(ids,q);again,_=sampling(ids,q);assert new==again and all(x in new for x in ids[::2]);assert len(set(meta['adaptive_frames'])&set(meta['global_anchors']))==0
    small=[0,4,8,14,20];p=np.triu(rng.random((5,5)),1);p/=p.sum();u=cross_utility(small,small,p);a,b=np.triu_indices(5,1);assert abs(u[a,b]-expected_iou_fast(small,p)[a,b]).max()<1e-12
    rec=[dict(frame_ids=ids[o::2].tolist())for o in [0,1]];ll=[torch.tensor(rng.normal(size=(1,len(r['frame_ids']),2)),dtype=torch.float32)for r in rec];assert all(torch.equal(x,y)for x,y in zip(ll,project_logits(rec,ll,rec)))
    return dict(status='pass',fixed65_unique=True,global33_full_span=True,deterministic_quantiles=True,same_grid_cross_utility_recovers_original_MBR=True,identity_logit_projection_exact=True,student_space_and_expert_positions_unchanged=True)
