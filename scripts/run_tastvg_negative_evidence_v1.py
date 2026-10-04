"""Same-A-state local writes, isolated transfer, then a separate reset-u stream."""
import os,sys,time,functools,gc,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
from scripts.tastvg_negative_evidence_common_v1 import *

def run(ds,stage):
    import torch,numpy as np
    from scripts.run_tastvg_evidence_vulnerability_v2 import install_clean_loader
    from scripts.run_tastvg_evidence_vulnerability_v1 import device_tree
    from scripts.run_final_simplification_v1 import lease
    from scripts.run_tastvg_paper48_p5_online_v1 import compact_prediction
    from vg_tta.tastvg_spatial_online_opd_s1_v1 import SpatialActor
    from vg_tta.tastvg_selected_rollout_v1 import central_with_candidates,rollout_states,target_loss
    from vg_tta.tastvg_native_spatial_rollout_s05_v1 import predict,central_state
    from vg_tta.tastvg_negative_evidence_v1 import make_anchor,negative_loss
    from vg_tta.tastvg_spatial_critic_s06_v1 import rewards
    from methods.decota_final_simplified_v1.tensors import detached,state_hash
    verify();budget();install_clean_loader();sys.addaudithook(guard);lh=lease();tick=time.monotonic()
    torch.set_num_threads(4);torch.manual_seed(20260929);np.random.seed(20260929)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    from scripts.run_spatial_regression_alignment_v1 import model_load
    model=model_load('hcstvg1_test' if ds=='vidstg' else 'vidstg_test').eval().requires_grad_(False)
    bar=read(POOL/ds/'CAPTURE_BARRIER.json');mh=state_hash(model.state_dict());assert mh==bar['checkpoint_state_sha256']
    center=central_state(model);states,spec,basis=rollout_states(center,.05,4)
    deltas=[{n:v-center[n] for n,v in z.items()} for z in states]
    actor=SpatialActor(model);coeff=[model.cfg.SOLVER.BBOX_COEF,model.cfg.SOLVER.GIOU_COEF]
    counts=dict(suffix_replays=0,backward_calls=0,new_backbone_calls=0,new_expert_calls=0,block_replays=0,future_replays=0)
    cells=[c for c in read(BASE/'COHORT.json')['cells'] if c['dataset']==ds]
    bykey={key(c):c for c in cells}
    @functools.lru_cache(maxsize=16)
    def cached(parent,cond):
        f=POOL/ds/'capture'/cond/f'{parent:05}.json';assert sha(f)==bar['files'][str(f.relative_to(POOL/ds))]
        r=read(f);cf=POOL/ds/r['cache'];assert sha(cf)==r['sha256'];return load(cf),r
    def data_at(c):
        d,r=cached(c['parent'],c['condition']);assert r['pixel_sha256']==c['pixel_sha256'];return device_tree(d,'cuda')
    def values(data):counts['suffix_replays']+=1;return actor.values(data)
    def pred(data,state):
        counts['suffix_replays']+=1;return compact_prediction(predict(model,data,device_tree(state,'cuda'))[2])
    def fixed(z,c):
        ids=plan(ds)['rows'][c['parent']]['frame_ids'];idx=oldcell(c)['final_indices']
        return dict(z,indices=list(idx),physical_interval=[ids[idx[0]],ids[idx[1]]+1])
    def fit(data,start,e,cfg,mode,old=None):
        actor.restore(device_tree(start,'cuda'));steps=[]
        for k in range(cfg['steps']):
            pre=detached(actor.state(),'cpu')
            with torch.no_grad():_,_,cp=values(data)
            cp=compact_prediction(cp)
            if old is not None and k<len(old['update_steps']) and old['update_steps'][k].get('candidates'):
                cs=[compact_prediction(z['prediction']) for z in old['update_steps'][k]['candidates']]
                assert torch.equal(cp['boxes'],cs[0]['boxes'])
            else:
                cs=[pred(data,{n:pre[n]+delta[n].cpu() for n in pre}) for delta in deltas]
            assert torch.equal(cs[0]['boxes'],cp['boxes'])
            target=torch.stack([z['boxes'] for z in cs]).cuda().detach()
            score=rewards([z['boxes'].numpy() for z in cs],e['boxes'],e['valid'])
            _,bg,pg=values(data);assert torch.equal(bg.detach().cpu(),cp['boxes'])
            anchor=make_anchor(bg,target,e['boxes'],e['valid'],coeff,mode='local' if mode=='negative_local' else 'global')
            if mode.startswith('rank') and score is not None:
                loss,pi,qi,dist,lq,meta=target_loss(bg,target,torch.tensor(score,device='cuda',dtype=torch.float64),coeff,cfg['teacher_temperature'],1.,'rank',1.)
                info=dict(logp=pi.log(),logp0=pi.detach().log(),logq=lq.detach(),distance=dist,
                          e_jk=anchor['e_jk'],framewise_rewards=anchor['framewise_rewards'],valid_positions=anchor['valid_positions'])
            else:
                info=negative_loss(bg,anchor);loss=info['loss'];lq=info['logq']
            gs=torch.autograd.grad(loss,[bg]+[v for _,v in actor.named]);counts['backward_calls']+=1
            assert all(torch.isfinite(g).all() for g in gs)
            grads={n:g.detach().cpu() for (n,_),g in zip(actor.named,gs[1:])}
            with torch.no_grad():
                for (n,p),g in zip(actor.named,gs[1:]):p.add_(g,alpha=-cfg['lr'])
                _,ba,pa=values(data)
                if mode.startswith('rank') and score is not None:
                    from vg_tta.tastvg_spatial_online_opd_s1_v1 import geometry
                    lp=(-geometry(ba,target,*coeff)).log_softmax(0);after=(lp.exp()*(lp-lq)).sum()
                else:ai=negative_loss(ba,anchor);lp=ai['logp'];after=ai['loss']
            post=detached(actor.state(),'cpu');q=dict(step=k,mode=mode,lr=cfg['lr'],teacher_temperature=cfg['teacher_temperature'],coefficients=coeff,
                pre_state=pre,post_state=post,pre_state_sha256=state_hash(pre),post_state_sha256=state_hash(post),
                candidates=cs,rewards=None if score is None else score.tolist(),pre_prediction=cp,post_prediction=compact_prediction(pa),
                gradients=grads,output_gradient=gs[0].detach().cpu(),loss=float(loss.detach()),loss_after=float(after),
                logp_after=lp.detach().cpu(),eta_gradient_norm=cfg['lr']*float(torch.sqrt(sum(g.double().square().sum() for g in gs[1:]))),
                gradient_block_norms={n:float(g.double().norm()) for n,g in grads.items()},
                **{n:detached(info[n],'cpu') for n in ['logp','logp0','logq','distance','e_jk','framewise_rewards','valid_positions']})
            if old is not None:
                prior=old['update_steps'][k];assert q['pre_state_sha256']==prior['pre_state_sha256'] and q['post_state_sha256']==prior['post_state_sha256']
                if prior['update']:
                    for n,g in grads.items():assert torch.equal(g,prior['update']['gradients'][n]),(n,k)
                if prior.get('post_prediction'):assert torch.equal(q['post_prediction']['boxes'],prior['post_prediction']['boxes'])
            steps.append(q)
            if score is None:break
        return dict(steps=steps,post_state=detached(actor.state(),'cpu'),post_prediction=steps[-1]['post_prediction'],config=cfg,mode=mode)
    if stage=='local':
        jobs=[c for c in cells if c['scheduled']]
        for done,c in enumerate(jobs,1):
            budget();path=local_payload_path(c)
            if Path(str(path)+'.json').exists():checked(path);continue
            old=oldcell(c);data=data_at(c);st=old['pre_state'];actor.restore(device_tree(st,'cuda'))
            with torch.no_grad():_,_,pre=values(data)
            assert torch.equal(pre['boxes'],old['slow']['boxes']);e,receipt=expert(c)
            arms={}
            future=dict(c,arrival=c['arrival']+1);future=bykey[key(future)];assert not future['scheduled']
            fd=data_at(future);fb=pred(fd,st);fb=fixed(fb,future);counts['future_replays']+=1
            for arm in ARMS:
                cfg=BUNDLES[('hc2' if ds=='vidstg' else 'vidstg') if arm=='rank_cross' else ds]
                z=fit(data,st,e,cfg,arm,old if arm=='rank_native' else None)
                z['post_prediction']=fixed(z['post_prediction'],c);z['future_prediction']=fixed(pred(fd,z['post_state']),future);counts['future_replays']+=1
                blocks={}
                for b in ['query','norm1','norm3','norm4']:
                    names=[n for n in st if (n=='spatial.query_residual' if b=='query' else f'.{b}.' in n)]
                    partial={n:z['post_state'][n] if n in names else v for n,v in st.items()}
                    blocks[b]=fixed(pred(data,partial),c);counts['block_replays']+=1
                z['block_counterfactual_predictions']=blocks;arms[arm]=z
            payload=dict(cell=c,pre_state=st,pre_prediction=fixed(compact_prediction(pre),c),arms=arms,
                future_cell=future,future_baseline=fb,evidence=dict(boxes=e['boxes'],valid=e['valid'],receipt=receipt),
                old_payload_sha256=sha(ROOT/c['old_payload']),GT_read=False,persistent_unchanged=True,probe_spec=spec)
            commit(path,payload);actor.restore(actor.initial)
            status(BASE/ds/'local_STATUS.json',dict(status='running',done=done,total=len(jobs),worker_pid=os.getpid(),GT_read=False,time=time.time()))
            print('LOCAL',ds,done,len(jobs),flush=True);del data,fd,payload,arms;gc.collect();torch.cuda.empty_cache()
    else:
        assert stage=='reset_u';from vg_tta.tastvg_event_support_v1 import OnlineMethod
        actor.close();cfg=BUNDLES[ds];p=plan(ds)
        from scripts.tastvg_correction_views_common_v1 import OLD
        method=OnlineMethod(model,deltas,**cfg,student_temperature=1.,basis=basis,**read(OLD/ds/'A/REQUEST.json')['method'])
        done=0
        for split in ['search','confirm']:
            for cond in p['conditions']:
                for order,seq in p['splits'][split]['orders'].items():
                    method.reset()
                    for at,parent in enumerate(seq):
                        c=bykey[key(dict(dataset=ds,split=split,condition=cond,order=order,arrival=at))];path=reset_payload_path(c)
                        budget()
                        if Path(str(path)+'.json').exists():
                            prior=checked(path);method.actor.restore(device_tree(prior['result']['post_state'],'cuda'));done+=1;continue
                        inherited=detached(method.actor.state(),'cpu')
                        with torch.no_grad():method.actor.query.zero_()
                        data=data_at(c)
                        def ep(stage):
                            from scripts.tastvg_correction_views_common_v1 import expert as oldexpert
                            return dict(**oldexpert(ds,stage,parent,cond,c['pixel_sha256'])[0],pixel_sha256=c['pixel_sha256'])
                        z,_=method.arrive(data,c['scheduled'],lambda:ep('temporal'),lambda:ep('spatial'))
                        counts['suffix_replays']+=z['compute']['native_replays'];counts['backward_calls']+=z['compute']['backward_calls']
                        commit(path,dict(cell=c,result=z,inherited_before_reset=inherited,GT_read=False,lifecycle='query_zero_LN_persistent'))
                        done+=1;status(BASE/ds/'reset_u_STATUS.json',dict(status='running',done=done,total=len(cells),worker_pid=os.getpid(),GT_read=False,time=time.time()))
                        print('RESET_U',ds,done,len(cells),flush=True);del data,z;gc.collect();torch.cuda.empty_cache()
        method.close()
    if stage=='local':actor.close()
    assert state_hash(model.state_dict())==mh;verify()
    write(BASE/ds/(stage+'_RESOURCES.json'),dict(**counts,worker_wall_seconds=time.monotonic()-tick,wall_includes_loading_IO=True,peak_allocated_vram_bytes=torch.cuda.max_memory_allocated(),GT_read=False))
    status(BASE/ds/(stage+'_STATUS.json'),dict(status='completed',done=len(jobs) if stage=='local' else len(cells),GT_read=False,time=time.time()));lh.close()

if __name__=='__main__':
    try:run(*sys.argv[1:])
    except BaseException as e:
        status(BASE/'WORKER_FAILURE.json',dict(status='failed',error=repr(e),traceback=traceback.format_exc(),time=time.time()));raise
