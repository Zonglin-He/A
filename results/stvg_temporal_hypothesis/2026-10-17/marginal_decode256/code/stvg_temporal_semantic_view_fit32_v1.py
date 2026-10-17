"""Native-grid512 endpoint distillation; no Actionness/Projection/candidate intervals."""
import inspect,math
import numpy as np
from scripts import stvg_evidence_temporal_fit_v1 as original
CFG=dict(original.OFFSET_TRUST_CONFIG,l2=0.,lambda_grid=1.)

def offsets(case,hidden):
    import torch
    out=[]
    for z,h,r in zip(case['zero']['logits'],hidden,case['records']):
        z=torch.as_tensor(z).detach().cpu();h=torch.as_tensor(h).detach().double().cpu().numpy();n=z.shape[1];assert z.shape==(1,n,2) and h.shape==(n,256)
        ij=np.asarray(torch.triu_indices(n,n,1));base=z[0].double().numpy();out.append(dict(hidden=h,ij=ij,base=base[ij[0],0]+base[ij[1],1],native_logits=base.copy(),ids=list(map(int,r['frame_ids']))))
    assert len(out)==2;return out

def native_decode(off,theta):
    import torch
    from methods.decota_final_simplified_v1.objectives import native_view_indices
    intervals=[];all_logits=[]
    for o in off:
        # Source logits and actual inserted residual use the native float32 head.
        z=torch.tensor(o['native_logits'],dtype=torch.float32)[None]+torch.tensor(o['hidden']@np.asarray(theta).reshape(256,2),dtype=torch.float32)[None]
        i,j=native_view_indices(z);intervals.append([o['ids'][i],o['ids'][j]+1]);all_logits.append(z)
    return [min(x[0] for x in intervals),max(x[1] for x in intervals)],all_logits

def boundary_targets(off,interval):
    a,b=interval;assert math.isfinite(a) and math.isfinite(b) and a<b
    result=[]
    for o in off:
        ids=np.asarray(o['ids'],float);stride=float(np.median(np.diff(ids)));assert stride>0
        s,t=o['ij'];scores=-.5*((ids[s]-a)/stride)**2-.5*((ids[t]+1-b)/stride)**2
        result.append(original._prob(scores)[0])
    return result

def distribution_targets(logits,off):
    import torch
    out=[]
    for z,o in zip(logits,off):
        a=torch.as_tensor(z).detach().double().cpu().numpy()[0];assert a.shape==o['native_logits'].shape
        out.append(original._prob(a[o['ij'][0],0]+a[o['ij'][1],1])[0])
    return out

def loss_gradient(pre,theta_flat,variant,cfg):
    theta=np.asarray(theta_flat,dtype=float).reshape(256,2);loss=0.;gradient=np.zeros_like(theta);teacher_loss=trust_loss=0.
    for o,q in zip(pre['offsets'],pre['targets']):
        p0,lp0=original._prob(o['base']);scores=original._scores_for_offset(o,theta);p,lp=original._prob(scores);positive=q>0;kl=float(np.sum(q[positive]*(np.log(q[positive])-lp[positive])));trust=original._kl(p0,lp0,lp)
        loss+=kl+cfg['lambda_grid']*trust;teacher_loss+=kl;trust_loss+=trust
        d=(p-q)+cfg['lambda_grid']*(p-p0);s=np.bincount(o['ij'][0],weights=d,minlength=len(o['hidden']));e=np.bincount(o['ij'][1],weights=d,minlength=len(o['hidden']));gradient[:,0]+=o['hidden'].T@s;gradient[:,1]+=o['hidden'].T@e
    assert math.isfinite(loss) and np.isfinite(gradient).all()
    return loss,gradient.reshape(-1),dict(teacher_kl=teacher_loss,native_trust_kl=trust_loss)

def fit(off,targets,admitted=True):
    off=[dict(o) for o in off];p0=[original._prob(o['base'])[0] for o in off]
    if not admitted:targets=[p.copy() for p in p0]
    assert len(targets)==2
    for q,p in zip(targets,p0):assert q.shape==p.shape and np.isfinite(q).all() and (q>=0).all() and abs(float(q.sum())-1)<1e-12
    pre=dict(offsets=off,targets=targets,no_information=not admitted or all(np.array_equal(a,b) for a,b in zip(p0,targets)))
    ns=dict(original.__dict__);ns['_loss_gradient']=loss_gradient;exec(compile(inspect.getsource(original._lbfgs_armijo),__file__,'exec'),ns);solution=ns['_lbfgs_armijo'](pre,'semantic_view',CFG)
    path=[]
    for theta,L,terms,step in solution['path']:
        iv,_=native_decode(off,theta);path.append(dict(theta=np.asarray(theta).reshape(256,2),loss=L,terms=terms,step_size=step,interval=iv))
    final,logits=native_decode(off,solution['x']);assert all(path[i+1]['loss']<path[i]['loss'] for i in range(len(path)-1))
    if pre['no_information']:assert len(path)==1 and np.count_nonzero(solution['x'])==0
    return dict(status='updated' if solution['accepted'] else 'exact_zero_native_noop',theta=solution['x'].reshape(256,2),path=path,optimizer_steps=solution['accepted'],rejected_line_search_trials=solution['rejected'],termination=solution['termination'],final_interval=final,final_logits=logits,targets=targets,no_information=pre['no_information'],active_parameters=512,config=CFG,GT_read=False)

def audit_fit(off,f):
    # Independent Torch autograd reconstructs objective and endpoint gradients.
    import torch
    checks=0
    for row in f['path']:
        theta=torch.tensor(row['theta'],dtype=torch.float64,requires_grad=True);L=theta.sum()*0
        for o,q0 in zip(off,f['targets']):
            h=torch.tensor(o['hidden']);ij=torch.tensor(o['ij']);base=torch.tensor(o['base']);z=base+(h@theta[:,0])[ij[0]]+(h@theta[:,1])[ij[1]];lp=z.log_softmax(0);q=torch.tensor(q0);p0=base.softmax(0);L+=(q[q>0]*(q[q>0].log()-lp[q>0])).sum()+(p0*(base.log_softmax(0)-lp)).sum()
        g=torch.autograd.grad(L,theta)[0].detach().numpy();lv,ga,_=loss_gradient(dict(offsets=off,targets=f['targets']),row['theta'].reshape(-1),'semantic_view',CFG)
        assert abs(float(L.detach())-lv)<1e-10 and np.max(np.abs(g.reshape(-1)-ga))<1e-9 and abs(lv-row['loss'])<1e-10
        assert native_decode(off,row['theta'])[0]==row['interval'];checks+=1
    assert np.array_equal(f['theta'],f['path'][-1]['theta']) and f['final_interval']==f['path'][-1]['interval']
    return checks
