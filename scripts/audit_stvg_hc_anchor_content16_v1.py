"""Independent CPU replay of saved state/policy/optimizer provenance, no model."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.run_stvg_joint_event_critic_v1 import read,put,load
from scripts.run_stvg_hc_anchor_content16_v1 import D,P,PRE,CACHE,verify


def cost(b,t):
    b=np.asarray(b,dtype=np.float64);t=np.asarray(t,dtype=np.float64)
    bl,bh=b[...,:2]-b[...,2:]/2,b[...,:2]+b[...,2:]/2
    tl,th=t[...,:2]-t[...,2:]/2,t[...,:2]+t[...,2:]/2
    inter=np.maximum(np.minimum(bh,th)-np.maximum(bl,tl),0).prod(-1)
    union=np.maximum(np.maximum(bh-bl,0).prod(-1)+np.maximum(th-tl,0).prod(-1)-inter,1e-7)
    enclosing=np.maximum(np.maximum(np.maximum(bh,th)-np.minimum(bl,tl),0).prod(-1),1e-7)
    giou=inter/union-(enclosing-union)/enclosing
    return 5*np.abs(b-t).sum(-1)+2*(1-giou)


def main():
    import torch
    torch.set_num_threads(2);p=verify(True);bar=read(D/'GLOBAL_PREDICTION_BARRIER.json')
    assert bar['count']==16 and bar['all16_outputs_before_new_GT'] and not torch.cuda.is_initialized()
    cfg=p['configuration'];checks=dict(zero_states=0,physical_state_maps=0,state_steps=0,
        candidate_feedback=0,Gaussian_mean_gradients=0,first_round_equal=0,unanchored_offset_paths=0)
    maximum=dict(E_mapping_error=0.,cost_error=0.,weight_error=0.,mean_gradient_error=0.,Adam_state_error=0.)
    for r in p['records']:
        a=r['alias'];z=load(D/'predictions'/(a+'.pt'));s=load(PRE/'source'/(a+'.pt'));e=load(PRE/'E'/(a+'.pt'));rc=load(CACHE/(a+'.pt'))
        pos=z['positions'];target=np.array([f['box']for f in z['targets']],float).reshape(-1,4)
        for v in range(2):
            assert s['records'][v]['frame_ids']==z['frame_ids'][v::2]
            assert len(rc['spatial_inputs'][v]['query_tgt'])==len(z['frame_ids'][v::2])
        for i,t in enumerate(pos):
            assert t in z['planned_positions'] and t//2<len(rc['spatial_inputs'][t%2]['query_tgt']);checks['physical_state_maps']+=1
        for arm,f in z['fits'].items():
            assert f['parameters']==len(pos)*256 and not torch.count_nonzero(f['path'][0]['state'])
            assert np.array_equal(f['path'][0]['boxes'].numpy(),e['E']);checks['zero_states']+=1
            u=torch.zeros_like(f['path'][0]['state'],requires_grad=True)
            opt=torch.optim.Adam([u],lr=cfg['lr'],betas=(.9,.999),eps=1e-8,weight_decay=0.)
            for k,entry in enumerate(f['path']):
                raw=entry['native'].double().numpy();src=s['Source'].double().numpy()
                mapped=np.concatenate((e['E'][:,:2]+raw[:,:2]-src[:,:2],e['E'][:,2:]*np.exp(np.log(raw[:,2:])-np.log(src[:,2:]))),-1)
                err=float(np.max(abs(mapped-entry['boxes'].numpy())));maximum['E_mapping_error']=max(maximum['E_mapping_error'],err);assert err<5e-13
                if pos:
                    assert abs(float(cost(entry['boxes'].numpy()[pos],target).mean())-entry['anchor_loss'])<1e-10
                for v in range(2):
                    if not any(t%2==v for t in pos):
                        assert torch.equal(entry['native'][v::2],s['Source'][v::2]);checks['unanchored_offset_paths']+=1
                if k==len(f['rounds']):continue
                rr=f['rounds'][k];policy=rr['rollout']
                if policy is not None:
                    mu=np.log(entry['native'][pos].numpy().astype(np.float64)/(1-entry['native'][pos].numpy().astype(np.float64)))
                    # Match the model's float32 inverse-sigmoid policy mean.
                    mean=torch.logit(entry['native'][pos]).numpy()
                    proposal=policy['proposal_mean'].numpy();actions=policy['actions'].numpy()
                    if arm=='Live':assert np.allclose(proposal,mu,rtol=0,atol=8*np.finfo(np.float32).eps*max(1.,float(np.max(abs(mu)))))
                    else:assert torch.equal(policy['proposal_mean'],f['rounds'][0]['rollout']['proposal_mean'])
                    seed=int(hashlib.sha256(f'anchor-content16-v1|{a}|{k if arm=="Live" else 0}'.encode()).hexdigest()[:15],16)
                    assert policy['seed']==seed
                    eps=torch.randn(len(pos),16,4,generator=torch.Generator(device='cpu').manual_seed(seed),dtype=torch.float32)
                    eps=torch.cat((eps,-eps),1).numpy()
                    assert np.array_equal(actions,proposal[:,None]+cfg['sigma']*eps)
                    bb=1/(1+np.exp(-actions.astype(np.float64)))
                    # Production action sigmoid is evaluated float32 before the
                    # explicit float64 E map, so verify that exact saved chart.
                    bb=torch.sigmoid(torch.from_numpy(actions)).double().numpy()
                    cand=np.concatenate((e['E'][pos,None,:2]+bb[...,:2]-src[pos,None,:2],
                        e['E'][pos,None,2:]*np.exp(np.log(bb[...,2:])-np.log(src[pos,None,2:]))),-1)
                    amplitude=max(1.,float(np.max(e['E'][pos,2:]/src[pos,2:])))
                    assert np.max(abs(cand-policy['boxes'].numpy()))<3e-7*amplitude
                    cc=cost(policy['boxes'].numpy(),target[:,None]);err=float(np.max(abs(cc-policy['cost'].numpy())));maximum['cost_error']=max(maximum['cost_error'],err);assert err<1e-10
                    logits=-cc/cfg['tau'];w=np.exp(logits-logits.max(-1,keepdims=True));w/=w.sum(-1,keepdims=True)
                    err=float(np.max(abs(w-policy['weights'].numpy())));maximum['weight_error']=max(maximum['weight_error'],err);assert err<1e-10
                    # Double-precision evaluation of the original analytic CV
                    # gradient; saved float32 rounding has relative tolerance.
                    ww=policy['weights'].numpy();aa=actions.astype(float);mm=proposal.astype(float) if arm=='Live' else mu;pp=proposal.astype(float)
                    gm=((mm-pp)/cfg['sigma']**2-((ww-1/32)[...,None]*(aa-mm[:,None])).sum(1)/cfg['sigma']**2)/len(pos)
                    err=float(np.max(abs(gm-rr['policy_mean_gradient'].numpy())));maximum['mean_gradient_error']=max(maximum['mean_gradient_error'],err)
                    rounding=0. if arm=='Live' else 8*np.finfo(np.float32).eps*max(1.,float(np.max(abs(mu))))/(len(pos)*cfg['sigma']**2)
                    assert err<2e-5+rounding+2e-6*max(1.,float(np.max(abs(gm))))
                    if arm=='Fixed':
                        for name in ['actions','weights','cost']:assert torch.equal(policy[name],f['rounds'][0]['rollout'][name])
                    checks['candidate_feedback']+=len(pos)*32;checks['Gaussian_mean_gradients']+=1
                assert torch.allclose(u.detach(),entry['state'],rtol=0,atol=5e-7)
                if rr['updated']:
                    opt.zero_grad(set_to_none=True);u.grad=rr['content_gradient'].clone();opt.step()
                err=float((u.detach()-f['path'][k+1]['state']).abs().max());maximum['Adam_state_error']=max(maximum['Adam_state_error'],err);assert err<5e-7
                checks['state_steps']+=int(rr['updated'])
        ff,ll=z['fits']['Fixed'],z['fits']['Live']
        if ff['rounds']:
            assert torch.equal(ff['path'][1]['state'],ll['path'][1]['state'])
            assert torch.equal(ff['rounds'][0]['content_gradient'],ll['rounds'][0]['content_gradient'])
            for name in ['actions','cost','weights']:assert torch.equal(ff['rounds'][0]['rollout'][name],ll['rounds'][0]['rollout'][name])
        checks['first_round_equal']+=1
    verify(True);assert not torch.cuda.is_initialized()
    out=dict(status='pass',count=16,checks=checks,max_errors=maximum,
        scope='saved physical alignment, zero E, output map, common localization costs, native policy samples, Fixed/Live provenance, mean gradients, independent CPU Adam replay; no new model replay',
        GT_annotation_or_new_task_scoring=False,model_or_GPU=False)
    put(P/'ROOT_MECHANISM_AUDIT.json',out);print(out,flush=True)


if __name__=='__main__':main()
