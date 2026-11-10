"""Fixed arithmetic detector -> one column patch -> native matching replay.

No autograd, solver, pseudo-target, task label, description/scene special case,
per-input search or score-driven step acceptance. Existing source spans only.
"""
import numpy as np
import torch

ALPHA=.25
TOPK=8

def sparse_gate(score,strength):
    idx=np.argsort(-score,kind='stable')[:TOPK]
    gate=np.zeros(288);idx=idx[score[idx]>1e-12]
    gate[idx]=np.clip(strength[idx],0,1)
    return ALPHA*gate

def detect(f,g,W,b):
    # Unit entity-versus-other token direction is a diagnostic, not a new label.
    # The final replay still uses the original logsumexp log-odds, not this proxy.
    evidence={};diag={};raw={};hidden=[];energy=[];role=[]
    for v in 'ABC':
        x=f[v+'_X'].astype(float);t=f[v+'_tokens'].astype(float);mask=f[v+'_mask'].astype(bool)
        y=x@W.T+b;norm=np.maximum(np.linalg.norm(y,axis=1),1e-12)
        direction=t[mask].mean(0)-t[~mask].mean(0)
        direction/=max(np.linalg.norm(direction),1e-12)
        evidence[v]=(x/norm[:,None])*(W.T@direction)[None]
        raw[v]=y;hidden.append(x);energy.append((x/norm[:,None])**2*np.square(W).sum(0)[None])
        role.append(np.square(W.T@direction)/np.maximum(np.square(W).sum(0),1e-20))
    aligned={'A':evidence['A']};confidence={'A':np.ones(256)}
    for v in 'BC':
        transport=g[v+'_W'].astype(float);mass=transport.sum(1)
        aligned[v]=np.divide(transport@evidence[v],mass[:,None],out=np.zeros_like(evidence['A']),where=mass[:,None]>0)
        confidence[v]=g[v+'_affinity'].max(1)*(mass>0)
    # Require both existing B and C support for cross-view noise diagnosis.
    support=np.minimum(confidence['B'],confidence['C']);den=float(support.sum())
    if den>1e-12:
        weights=support/den;stack=np.stack([aligned[v] for v in 'ABC']);mean=stack.mean(0)
        noise=(weights[:,None]*np.square(stack-mean[None]).mean(0)).sum(0)
        center=(weights[:,None]*mean).sum(0)
        signal=(weights[:,None]*np.square(mean-center)).sum(0)
        excess=np.maximum(noise-signal,0)
        strength=excess/(noise+signal+1e-12)
        gn=sparse_gate(excess,strength)
    else:
        noise=np.zeros(288);signal=np.zeros(288);gn=np.zeros(288)
    # A candidate-common response with little entity-direction energy may
    # dominate the normalized vector. This is distinct from variance pruning.
    x=np.concatenate(hidden);common=np.square(x.mean(0))/np.maximum(np.square(x).mean(0),1e-20)
    relevance=np.mean(role,0);e=np.mean(np.concatenate(energy),0)
    common_excess=np.clip((common-.5)/.5,0,1)*(1-relevance)
    gc=sparse_gate(e*common_excess,common_excess)
    gates={'noise':gn,'common':gc}
    diag={'support_mass':den,'supported_anchor_count':int((support>0).sum()),'noise_energy':noise,'between_candidate_signal':signal,'common_fraction':common,'entity_direction_fraction':relevance,'normalized_channel_energy':e}
    return gates,diag

def random_norm_matched(gate,W,rng):
    # Match count, per-selected-column edit norm, and total Frobenius norm.
    # Choose random destinations among the 16 closest column norms, excluding
    # all mechanism-selected channels. No model score is consulted.
    norms=np.linalg.norm(W,axis=0);idx=np.flatnonzero(gate);out=np.zeros(288);used=set(idx.tolist())
    for c in idx:
        ordered=np.argsort(abs(norms-norms[c]),kind='stable')
        eligible=[j for j in ordered if j not in used and norms[j]>0 and gate[c]*norms[c]/norms[j]<=.5]
        pool=np.array(eligible[:16]);assert len(pool)>0
        j=int(rng.choice(pool));used.add(j);out[j]=gate[c]*norms[c]/norms[j]
    assert len(np.flatnonzero(out))==len(idx)
    assert abs(np.linalg.norm(W*out)-np.linalg.norm(W*gate))<1e-10
    return out

def native(raw,f,g,return_diagnostics=False):
    probabilities=[];logits=[];coverage=[]
    with torch.no_grad():
        for v in 'ABC':
            y=raw[v];y=y if isinstance(y,torch.Tensor) else torch.tensor(np.asarray(y,np.float32))
            q=torch.nn.functional.normalize(y,dim=-1);t=torch.as_tensor(f[v+'_tokens']);mask=torch.as_tensor(f[v+'_mask']).bool()
            l=q@t.T/.07;z=torch.logsumexp(l[:,mask],-1)-torch.logsumexp(l[:,~mask],-1);p=z.double().softmax(-1).numpy()
            if v=='A':aligned=z.double().numpy();cover=np.ones(256)
            else:
                T=g[v+'_W'];mass=T.sum(1);zz=T@z.double().numpy();aligned=np.divide(zz,mass,out=np.zeros(256),where=mass>0)
                cover=g[v+'_affinity'].max(1)*(mass>0)
                p=T@p;fixed=float(g[v+'_fixed_mass']);p=fixed*p/max(float(p.sum()),1e-300)+(1-fixed)*g['base']
            probabilities.append(p);logits.append(aligned);coverage.append(cover)
    scores=sum(probabilities)/3
    if not return_diagnostics:return scores
    z=np.stack(logits);support=np.minimum(coverage[1],coverage[2]);s=float(support.sum())
    if s>1e-12:
        w=support/s;centered=z-(z*w).sum(1)[:,None];mean=centered.mean(0)
        disagreement=float((np.square(centered-mean[None]).mean(0)*w).sum())
        separation=float((np.square(mean-(mean*w).sum())*w).sum())
    else:disagreement=0.;separation=0.
    order=np.sort(scores)
    return scores,{'matched_logodds_disagreement':disagreement,'matched_logodds_candidate_separation':separation,'ensemble_Top1_probability_margin':float(order[-1]-order[-2])}

def replay(model,f,g,gate,activation=False):
    original=model[4].weight.detach().clone();bias=model[4].bias.detach().clone()
    with torch.no_grad():
        if not activation:model[4].weight.copy_(original*torch.tensor(1-gate,dtype=original.dtype)[None])
        raw={v:model[4](torch.tensor(f[v+'_X'])*(torch.tensor(1-gate,dtype=torch.float32)[None] if activation else 1)) for v in 'ABC'}
        changed=not torch.equal(model[4].weight,original)
        output,stats=native(raw,f,g,True)
        model[4].weight.copy_(original)
        assert torch.equal(model[4].weight,original) and torch.equal(model[4].bias,bias)
    stats.update(actual_weight_changed=changed,edited_columns=int(np.count_nonzero(gate)),relative_weight_edit_norm=float(np.linalg.norm(original.numpy()*gate)/np.linalg.norm(original.numpy())),max_column_attenuation=float(gate.max()),source_restored_exact=True)
    return output,stats
