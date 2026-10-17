"""Native joint-offset hypotheses and prior-tilted semantic preference KL."""
import inspect,math
import numpy as np
from scripts import stvg_temporal_semantic_view_fit32_v1 as native
original=native.original;CFG=dict(native.CFG,tau_T=1.)
def merge(off,pairs):return [min(o['ids'][p[0]] for o,p in zip(off,pairs)),max(o['ids'][p[1]]+1 for o,p in zip(off,pairs))]
def sampled(interval,available):
    from scripts.stvg_table1_temporal_adapter_v1 import _sample_candidate_ids
    try:return _sample_candidate_ids(dict(interval_frames=interval),available)[0]
    except ValueError:return None

def select(off,available):
    from methods.decota_final_simplified_v1.objectives import native_view_indices
    import torch
    # Actual Native decoder uses FP32 endpoint log-softmax/mask tie semantics.
    first=[list(native_view_indices(torch.tensor(o['native_logits'],dtype=torch.float32)[None])) for o in off];iv=merge(off,first);sf=sampled(iv,available)
    if sf is None:return dict(status='Native_fewer_than_eight_frames',hypotheses=[],Native_interval=iv)
    ss=[]
    for o in off:
        z=torch.tensor(o['native_logits'],dtype=torch.float32);z=z[:,0].log_softmax(0)[:,None]+z[:,1].log_softmax(0)[None,:];a,b=o['ij'];ss.append(z[a,b].numpy().astype(float))
    # Joint Native state probability = independent original offset probabilities.
    # Each hypothesis keeps its two latent endpoint pairs; displayed interval is
    # native merge. This is not interval-marginal probability after merging.
    totals=ss[0][:,None]+ss[1][None,:];order=np.argsort(-totals.reshape(-1),kind='stable');n1=len(ss[1]);chosen=None
    for flat in order:
        ix=[int(flat)//n1,int(flat)%n1];pp=[o['ij'][:,i].tolist() for o,i in zip(off,ix)];vv=merge(off,pp);sf2=sampled(vv,available)
        if sf2 is not None and sf2!=sf:chosen=dict(pairs=pp,interval_frames=vv,sampled_physical_frame_ids=sf2);break
    one=dict(pairs=first,interval_frames=iv,sampled_physical_frame_ids=sf)
    if chosen is None:return dict(status='no_distinct_sampled_hypothesis',hypotheses=[one],Native_interval=iv)
    result=[one,chosen]
    for h in result:
        h['native_joint_score']=sum(float(o['native_logits'][p[0],0]+o['native_logits'][p[1],1]) for o,p in zip(off,h['pairs']))
    return dict(status='two_distinct_observations',hypotheses=result,Native_interval=iv,probability_scope='two retained joint offset states, not merged-interval marginal',selection='highest native FP32 joint score with distinct actual sampled frame IDs; stable row-major tie')
def prepare(off,hypotheses,rewards):
    features=[];base=[]
    for h in hypotheses:
        feat=np.zeros((256,2));value=0.
        for o,p in zip(off,h['pairs']):feat[:,0]+=o['hidden'][p[0]];feat[:,1]+=o['hidden'][p[1]];value+=o['native_logits'][p[0],0]+o['native_logits'][p[1],1]
        features.append(feat);base.append(value)
    informative=len(hypotheses)==2 and len(rewards)==2 and all(x is not None and math.isfinite(x) for x in rewards) and rewards[0]!=rewards[1]
    p0,lp0=original._prob(np.asarray(base)) if len(base) else (np.empty(0),np.empty(0));q,lq=original._prob(np.asarray(base)+np.asarray(rewards)/CFG['tau_T']) if informative else (p0.copy(),lp0.copy())
    return dict(offsets=off,features=np.asarray(features).reshape(-1,256,2),candidate_base=np.asarray(base),q=q,lq=lq,no_information=not informative)
def loss_gradient(pre,theta_flat,variant,cfg):
    th=np.asarray(theta_flat).reshape(256,2);g=np.zeros_like(th);L=0.;cand=0.;trust=0.
    if len(pre['q']):
        p,lp=original._prob(pre['candidate_base']+np.einsum('kdc,dc->k',pre['features'],th));cand=original._kl(pre['q'],pre['lq'],lp);g+=np.einsum('k,kdc->dc',p-pre['q'],pre['features']);L+=cand
    for o in pre['offsets']:
        p0,lp0=original._prob(o['base']);p,lp=original._prob(original._scores_for_offset(o,th));v=original._kl(p0,lp0,lp);L+=v;trust+=v;d=p-p0;s=np.bincount(o['ij'][0],weights=d,minlength=len(o['hidden']));e=np.bincount(o['ij'][1],weights=d,minlength=len(o['hidden']));g[:,0]+=o['hidden'].T@s;g[:,1]+=o['hidden'].T@e
    return L,g.reshape(-1),dict(candidate_kl=cand,native_grid_trust_kl=trust)
def fit(off,hypotheses,rewards):
    pre=prepare(off,hypotheses,rewards);ns=dict(original.__dict__);ns['_loss_gradient']=loss_gradient;exec(compile(inspect.getsource(original._lbfgs_armijo),__file__,'exec'),ns);s=ns['_lbfgs_armijo'](pre,'hypothesis',CFG);iv,logits=native.native_decode(off,s['x']);path=[dict(theta=x.reshape(256,2),loss=L,terms=v,step_size=step) for x,L,v,step in s['path']]
    assert all(path[i+1]['loss']<path[i]['loss'] for i in range(len(path)-1))
    if pre['no_information']:assert s['accepted']==0 and np.count_nonzero(s['x'])==0
    return dict(status='updated' if s['accepted'] else 'exact_zero_native_noop',theta=s['x'].reshape(256,2),path=path,optimizer_steps=s['accepted'],rejected_line_search_trials=s['rejected'],termination=s['termination'],final_interval=iv,final_logits=logits,no_information=pre['no_information'],hypotheses=hypotheses,rewards=rewards,Teacher_conditional=pre['q'],Native_conditional=original._prob(pre['candidate_base'])[0] if len(hypotheses) else [],GT_read=False)
def audit(off,f):
    import torch
    pre=prepare(off,f['hypotheses'],f['rewards']);n=0
    for row in f['path']:
        th=torch.tensor(row['theta'],dtype=torch.float64,requires_grad=True);L=th.sum()*0
        if len(pre['q']):
            v=torch.tensor(pre['candidate_base'])+torch.einsum('kdc,dc->k',torch.tensor(pre['features']),th);q=torch.tensor(pre['q']);pos=q>0;L+=(q[pos]*(q[pos].log()-v.log_softmax(0)[pos])).sum()
        for o in off:
            h=torch.tensor(o['hidden']);ij=torch.tensor(o['ij']);z=torch.tensor(o['base']);v=z+(h@th[:,0])[ij[0]]+(h@th[:,1])[ij[1]];L+=(z.softmax(0)*(z.log_softmax(0)-v.log_softmax(0))).sum()
        g=torch.autograd.grad(L,th)[0].detach().numpy();v,a,_=loss_gradient(pre,row['theta'].reshape(-1),'hypothesis',CFG);assert abs(float(L.detach())-v)<1e-10 and abs(v-row['loss'])<1e-10 and np.max(abs(g.reshape(-1)-a))<1e-9;n+=1
    return n
