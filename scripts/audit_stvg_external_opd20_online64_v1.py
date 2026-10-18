"""Independent online history, policy/Adam math, input separation and metrics."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,collections
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
from scripts.run_stvg_external_opd20_online64_v1 import D,P,verify
from vg_tta.stvg_external_opd20_v1 import ARMS


def transform(b,mu,s):return np.c_[b[:,:2]+(s*mu)[:2],b[:,2:]*np.exp((s*mu)[2:])]


def overlap(x,y):
    p=np.concatenate((x[...,:2]-x[...,2:]/2,x[...,:2]+x[...,2:]/2),-1);q=np.concatenate((y[...,:2]-y[...,2:]/2,y[...,:2]+y[...,2:]/2),-1)
    inter=np.maximum(0,np.minimum(p[...,2:],q[...,2:])-np.maximum(p[...,:2],q[...,:2])).prod(-1);u=x[...,2:].prod(-1)+y[...,2:].prod(-1)-inter
    return inter/u,p,q,u,inter


def main():
    from scripts.score_stvg_opd_p1_v1 import truths
    from vg_tta.tastvg_oracle_event5_v1 import official,box_iou
    p=verify();bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');assert bar['count']==64
    assert sha(P/'RESULTS.json')==read(D/'CPU_COMPLETION.json')['result_sha256'];rs=read(P/'RESULTS.json');by={r['alias']:r for r in rs['rows']};counter=collections.Counter();maxerror=0.
    targets={ds:truths(ds,dict(parents=[r['query_ordinal']for r in p['records']if r['dataset']==ds]))for ds in ['hc2','vidstg']};plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in targets}
    prev={};last={};role_counts=collections.Counter();steps=0
    for r in p['records']:
        a=r['alias'];ds=r['dataset'];z=load(D/'predictions'/(a+'.pt'));b=z['Source_boxes'];s=z['scale'];phi=np.r_[1.,b.mean(0)]
        assert np.array_equal(phi,z['phi']);role_counts[ds,r['role']]+=1
        if r['domain_arrival']==0:prev={a:np.zeros((4,5))for a in ARMS};last[ds]=None
        assert z['predecessor_sha256']==last[ds]
        for arm in ARMS:assert np.array_equal(z['W_before'][arm].numpy(),prev[arm]);counter['state_links']+=1
        if r['role']=='no_expert':
            assert z['teacher']is None and z['expert_input_sha256']is None and not z['fits']and not r['expert_path']
            assert not(D/'expert_only'/(a+'.npz')).exists()
            for arm in ARMS:
                expected=transform(b,prev[arm]@phi,s);error=float(abs(expected-z['outputs'][arm]).max());assert error<1e-12;maxerror=max(maxerror,error)
                assert np.array_equal(prev[arm],z['W_after'][arm].numpy());counter['no_expert_no_update']+=1
        else:
            target=z['teacher'];assert all(np.array_equal(z['outputs'][arm],target)for arm in ARMS)
            for arm in ARMS[1:]:
                f=z['fits'][arm];assert f['parameter_count']==20 and f['optimizer_initial_empty'];assert np.array_equal(f['W_before'].numpy(),prev[arm]);m=np.zeros((4,5));v=m.copy();t=0;fixed=None
                assert len(f['trace'])==5
                for item in f['trace']:
                    W=item['W_before'].numpy();W1=item['W_after'].numpy();mu=W@phi
                    if item.get('no_valid_expert'):
                        assert not item['updated']and np.array_equal(W,W1);continue
                    rr=item['rollout'];pb=transform(b,mu,s)
                    if arm=='B_Direct':
                        ov,x,y,union,inter=overlap(pb,target);enclose=np.maximum(0,np.maximum(x[:,2:],y[:,2:])-np.minimum(x[:,:2],y[:,:2])).prod(-1);giou=inter/np.maximum(union,1e-7)-(np.maximum(enclose,1e-7)-np.maximum(union,1e-7))/np.maximum(enclose,1e-7)
                        loss=float((5*np.abs(pb-target).sum(1)+2*(1-giou)).mean());assert abs(loss-item['loss'])<1e-10;counter['Direct_objectives']+=1
                    else:
                        actions=rr['actions'].numpy();noise=rr['noise'].numpy();assert np.array_equal(noise[:16],-noise[16:]);assert np.array_equal(actions,rr['mean'].numpy()[None]+noise)
                        pos=f['positions'];candidate=np.stack([transform(b,act,s)for act in actions]);rewards=overlap(candidate[:,pos],target[None,pos])[0].mean(1);assert np.max(abs(rewards-rr['rewards'].numpy()))<1e-12
                        ew=np.exp((rewards-rewards.max())/.25);ew/=ew.sum();assert np.max(abs(ew-rr['weights'].numpy()))<1e-12
                        gradmu=mu-rr['mean'].numpy()-((ew-1/32)[:,None]*(actions-mu[None])).sum(0)
                        if item['updated']:assert np.max(abs(item['gradient'].numpy()-gradmu[:,None]*phi[None]))<1e-12
                        if arm=='F_fixed_rollout':
                            if fixed is None:fixed=actions.copy()
                            assert np.array_equal(actions,fixed);counter['fixed_action_reuse']+=1
                        else:
                            assert np.array_equal(rr['mean'].numpy(),item['mu'].numpy())
                            assert np.max(abs(rr['mean'].numpy()-mu))<1e-12
                            counter['current_policy_centers']+=1
                        counter['OPD_reward_gradient_rounds']+=1
                    if item['updated']:
                        t+=1;steps+=1;g=item['gradient'].numpy();m=.9*m+.1*g;v=.999*v+.001*g*g
                        expected=-.01*(m/(1-.9**t))/(np.sqrt(v/(1-.999**t))+1e-8);err=float(np.max(abs(W1-W-expected)));assert err<1e-12;maxerror=max(maxerror,err)
                        state=list(item['Adam_after']['state'].values());assert len(state)==1
                        assert np.max(abs(state[0]['exp_avg'].numpy()-m))<1e-12 and np.max(abs(state[0]['exp_avg_sq'].numpy()-v))<1e-12;assert int(state[0]['step'])==t;counter['Adam_steps']+=1
                    else:assert np.array_equal(W,W1)
                assert np.array_equal(f['trace'][-1]['W_after'].numpy(),z['W_after'][arm].numpy());counter['expert_fits']+=1
        for arm in ARMS:prev[arm]=z['W_after'][arm].numpy()
        last[ds]=sha(D/'predictions'/(a+'.pt'))
        q=r['query_ordinal'];row=plans[ds][q];gt,span=targets[ds][0][q],targets[ds][1][q];ids=np.array(z['frame_ids']);known=np.array([int(i)in gt for i in ids])
        for arm,bx in z['outputs'].items():
            met=official(bx,row,gt,span,z['interval'],ds);saved=by[a]['values'][arm]
            for ep in ['v','t','s']:assert abs(met[ep]-saved[ep])<2e-10;counter['metrics']+=1
            xy=np.c_[bx[:,:2]-bx[:,2:]/2,bx[:,:2]+bx[:,2:]/2]*np.array([row['input']['width'],row['input']['height']]*2)
            if ds=='hc2':xy=np.maximum(xy,0)
            if known.any():assert abs(float(box_iou(xy[known],np.array([gt[int(fid)]for fid in ids[known]])).mean())-saved['actual_decoder_s'])<2e-10
            else:assert saved['actual_decoder_s']is None
            counter['actual_frame_support']+=1
    assert steps==bar['external_optimizer_steps'];assert all(role_counts[ds,role]==16 for ds in targets for role in ['expert','no_expert'])
    access=read(D/'ACCESS_READBACK.json');experts={r['alias']for r in p['records']if r['role']=='expert'}
    assert {v['alias']for v in access['expert_open_events']}==experts and all(v['role']=='expert'for v in access['expert_open_events']);assert access['nonexpert_teacher_opens']==access['original_mixed_payload_opens']==0
    for ds in targets:
        for role,g in rs['groups'][ds].items():
            rr=[r for r in rs['rows']if r['dataset']==ds and(role=='all'or r['role']==role)]
            for arm,eps in g['metrics'].items():
                for ep,x in eps.items():
                    values=[r['values'][arm][ep]for r in rr if r['values'][arm][ep]is not None];assert x is None and not values or x is not None and abs(sum(values)/len(values)-x)<1e-12;counter['aggregates']+=1
            for pair,eps in g['paired'].items():
                aa,bb=pair.split('_minus_')
                for ep,saved in eps.items():
                    values=np.array([100*(r['values'][aa][ep]-r['values'][bb][ep])for r in rr if r['values'][aa][ep]is not None and r['values'][bb][ep]is not None]);assert np.max(abs(values-saved['per_parent_pp']))<1e-12
                    ix=np.random.default_rng(20261018).integers(len(values),size=(10000,len(values)));ci=np.quantile(values[ix].mean(1),[.025,.975]);assert np.max(abs(ci-saved['CI95']))<1e-12;counter['conditional_CIs']+=1
    put(P/'ROOT_INDEPENDENT_AUDIT.json',dict(status='pass',count=64,checks=dict(counter),max_formula_error=maxerror,actual_external_optimizer_steps=steps,
        nonexpert_current_teacher_reads=0,expert_input_read_aliases_exact=True,source_only_inputs_no_expert_payload=True,model_asset_reads=0,backbone_updates=0,uncertainty_is_saved_history_conditional=True))
    put(D/'ROOT_READBACK_COMPLETION.json',dict(status='pass',count=64,actual_pixels_privacy_remote_archive_pending=True))


if __name__=='__main__':main()
