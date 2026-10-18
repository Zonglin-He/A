"""CPU audit of saved QR-only paths, original objectives, Adam and metrics."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,collections
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
from scripts.run_stvg_query_e_opd64_v1 import D,P,ARMS,verify
from scripts.audit_stvg_external_opd20_online64_v1 import overlap
QUERY='spatial.query_residual'


def main():
    import torch
    from scripts.score_stvg_opd_p1_v1 import truths
    from vg_tta.tastvg_oracle_event5_v1 import official,box_iou
    torch.set_num_threads(4);p=verify();bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');assert bar['count']==64 and bar['all_outputs_before_GT']
    rs=read(P/'RESULTS.json');assert sha(P/'RESULTS.json')==read(D/'CPU_COMPLETION.json')['result_sha256'];by={r['alias']:r for r in rs['rows']}
    targets={ds:truths(ds,dict(parents=[r['query_ordinal']for r in p['records']if r['dataset']==ds]))for ds in ['hc2','vidstg']}
    plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in targets};c=collections.Counter();errors=collections.defaultdict(float)
    def near(a,b,tol,label):
        err=float(np.max(abs(np.array(a)-np.array(b))));errors[label]=max(errors[label],err);assert err<tol,(label,err)
    def moment_near(a,b,label):
        av=np.array(a,dtype=float);bv=np.array(b,dtype=float);err=float(np.max(abs(av-bv)));rel=float(np.max(abs(av-bv)/(1+abs(bv))));errors[label+'_absolute']=max(errors[label+'_absolute'],err);errors[label+'_normalized']=max(errors[label+'_normalized'],rel)
        assert np.allclose(av,bv,rtol=2e-6,atol=2e-5),(label,err,rel)
    for r in p['records']:
        z=load(D/'predictions'/(r['alias']+'.pt'));assert not z['GT_read']and z['new_DINO_calls']==0
        ds=r['dataset'];teacher=z['E_training_teacher_float32'].numpy();source=z['outputs']['Source_MBR'];n=len(source)
        prior=load(ROOT/r['prior_prediction']);assert np.array_equal(source,prior['outputs']['Source_no_correction'])and np.array_equal(z['outputs']['E_MBR'],prior['outputs']['E_DINO_support'])
        for arm,f in z['fits'].items():
            assert f['active_parameters']==256 and f['parameter_names']==[QUERY]and f['optimizer_initial_empty']and f['selected_step']==p['configs'][ds]['steps']
            assert f['full_two_pass_readout_exact']and f['pretrained_weights_frozen_exact']
            assert not torch.count_nonzero(f['initial'][QUERY])and np.array_equal(f['before'].double().numpy(),source)
            for it in f['path']:
                for k,v in f['initial'].items():
                    if k!=QUERY:assert torch.equal(it['state'][k],v);c['unchanged_LN_tensors']+=1
            m=np.zeros(256);v=np.zeros(256);t=0;frozen=None;fixedmean=None
            pos=f['feedback_positions'];assert pos==(z['planned_positions']if arm=='Sparse_OPD'else list(range(n)))
            for k,it in enumerate(f['path'][:-1]):
                step=f['rounds'][k];nxt=f['path'][k+1];b=it['boxes'].numpy();b1=nxt['boxes'].numpy();u=it['state'][QUERY].numpy();u1=nxt['state'][QUERY].numpy()
                near(np.linalg.norm(b1-b),step['output_displacement_l2'],1e-4,'movement');c['path_steps']+=1
                if f['empty']:
                    assert not step['updated']and np.array_equal(b,b1)and np.array_equal(u,u1);continue
                if arm=='Dense_Direct':
                    ov,x,y,union,inter=overlap(b,teacher);enc=np.maximum(0,np.maximum(x[:,2:],y[:,2:])-np.minimum(x[:,:2],y[:,:2])).prod(-1)
                    giou=inter/np.maximum(union,1e-7)-(np.maximum(enc,1e-7)-np.maximum(union,1e-7))/np.maximum(enc,1e-7)
                    near((5*abs(b-teacher).sum(1)+2*(1-giou)).mean(),step['loss'],3e-5,'Direct_loss');c['Direct_objectives']+=1
                else:
                    rr=step['rollout'];mu=it['chart']['mean'].numpy();act=rr['samples'].numpy();center=rr['mean'].numpy();w=rr['weights'].numpy();cfg=f['config'];sig=cfg['sigma']
                    # Verify seed, antithetic candidate noise and per-frame support.
                    gen=torch.Generator(device='cpu').manual_seed(rr['seed']);eps=torch.randn(n,16,4,generator=gen,dtype=torch.float32);eps=torch.cat((eps,-eps),1)[pos].numpy()
                    near(act,center[:,None]+sig*eps,2e-6,'actions')
                    candidate=1/(1+np.exp(-act.astype(np.float64)));reward=overlap(candidate,teacher[pos,None])[0]
                    near(reward,rr['rewards'].numpy(),2e-5,'reward')
                    ww=np.exp((reward-reward.max(1,keepdims=True))/cfg['tau']);ww/=ww.sum(1,keepdims=True);near(ww,w,2e-5,'softmax')
                    expected=((mu-center)-((w-1/32)[:,:,None]*(act-mu[:,None])).sum(1))/(sig**2*len(pos))
                    near(expected,it['chart']['autograd_gradient'].numpy(),5e-5,'policy_gradient')
                    kl=((mu-center)**2).sum(1)/(2*sig**2);logp=-((act-mu[:,None])**2).sum(2)/(2*sig**2)-2*np.log(2*np.pi*sig**2)
                    near((kl-((w-1/32)*logp).sum(1)).mean(),step['loss'],1e-4,'likelihood_loss')
                    if arm=='Fixed_rollout':
                        if frozen is None:frozen=act.copy();fixedmean=center.copy()
                        assert np.array_equal(act,frozen)and np.array_equal(center,fixedmean);c['fixed_rollout_reuse']+=1
                    else:assert np.array_equal(mu,center);c['current_policy_centers']+=1
                    c['OPD_objectives']+=1
                if step['updated']:
                    t+=1;g=it['update']['gradient'].numpy().astype(float);m=.9*m+.1*g;v=.999*v+.001*g*g
                    expected=-f['config']['lr']*(m/(1-.9**t))/(np.sqrt(v/(1-.999**t))+1e-8)
                    near(u1-u,expected,2e-6,'Adam_delta');state=list(it['Adam_after']['state'].values());assert len(state)==1 and int(state[0]['step'])==t
                    moment_near(state[0]['exp_avg'].numpy(),m,'Adam_m');moment_near(state[0]['exp_avg_sq'].numpy(),v,'Adam_v');c['Adam_steps']+=1
                else:assert np.array_equal(u,u1)
            assert t==f['gradient_calls'];assert np.array_equal(f['final'].double().numpy(),z['outputs'][arm]);c['fits']+=1
        # Independent metric readback on exactly the same physical support.
        q=r['query_ordinal'];row=plans[ds][q];gt,span=targets[ds][0][q],targets[ds][1][q];ids=np.array(z['frame_ids']);mask=np.array([int(i)in gt for i in ids])&~np.isin(np.arange(n),z['planned_positions'])
        for arm,b in z['outputs'].items():
            saved=by[r['alias']]['values'][arm];met=official(b,row,gt,span,z['interval'],ds)
            for ep in ['v','t','s']:near(met[ep],saved[ep],2e-10,'metric');c['metrics']+=1
            xy=np.c_[b[:,:2]-b[:,2:]/2,b[:,:2]+b[:,2:]/2]*np.array([row['input']['width'],row['input']['height']]*2)
            if ds=='hc2':xy=np.maximum(xy,0)
            if mask.any():near(box_iou(xy[mask],np.array([gt[int(i)]for i in ids[mask]])).mean(),saved['unobserved_actual_s'],2e-10,'unobserved_metric')
            else:assert saved['unobserved_actual_s']is None
        assert len({x['t']for x in by[r['alias']]['values'].values()})==1;c['temporal_invariants']+=1
    for ds,g in rs['summary'].items():
        rr=[r for r in rs['rows']if r['dataset']==ds]
        for arm,eps in g['query_mean'].items():
            for ep,x in eps.items():
                vv=[r['values'][arm][ep]for r in rr if r['values'][arm][ep]is not None]
                if vv:near(np.mean(vv),x,1e-12,'aggregate')
                else:assert x is None
                c['aggregates']+=1
        for pair,eps in g['paired'].items():
            aa,bb=pair.split('_minus_')
            for ep,saved in eps.items():
                vals=np.array([100*(r['values'][aa][ep]-r['values'][bb][ep])for r in rr if r['values'][aa][ep]is not None and r['values'][bb][ep]is not None]);near(vals,saved['per_parent_pp'],1e-12,'paired')
                ix=np.random.default_rng(20261018).integers(len(vals),size=(10000,len(vals)));near(np.quantile(vals[ix].mean(1),[.025,.975]),saved['CI95'],1e-12,'CI');c['paired_CIs']+=1
    put(P/'ROOT_INDEPENDENT_AUDIT.json',dict(status='pass',count=64,checks=dict(c),max_errors=dict(errors),active_query_variables=256,LN_updates=0,backbone_updates=0,full_two_pass_outputs_exact=True,audit_moment_precision=dict(actual_saved_dtype="FP32",independent_recurrence_dtype="FP64",rtol=2e-6,atol=2e-5,original_failure_preserved=True,no_predictions_updates_or_scores_changed=True)))
    put(D/'ROOT_READBACK_COMPLETION.json',dict(status='pass',count=64,actual_pixel_privacy_remote_archive_pending=True))


if __name__=='__main__':main()
