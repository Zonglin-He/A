"""Sparse-expert online spatial RKL; current-policy rollouts and persistent state."""
import sys,time,gc,traceback,subprocess,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,load,save,sha,status
from scripts.run_tastvg_spatial_expansion_s0_v1 import OUT as S0,OLD,CONDS,observation
NATIVE=ROOT/'artifacts/tastvg_native_spatial_rollout_s05_v1';CRITIC=ROOT/'artifacts/tastvg_spatial_critic_s06_v1';BASE=ROOT/'artifacts/tastvg_spatial_rank_s11_v1';ARM=sys.argv[2] if len(sys.argv)>2 else 'rank';assert ARM in ['rank','norm'];OUT=BASE/ARM;RAW=ROOT/'artifacts/tastvg_spatial_online_opd_s1_v1'

def prepare():
    p=read(NATIVE/'LOCK.json');paths=['protocols/tastvg_spatial_rank_s11_v1.md','vg_tta/tastvg_spatial_rank_s11_v1.py','vg_tta/tastvg_native_spatial_rollout_s05_v1.py','vg_tta/tastvg_spatial_critic_s06_v1.py','scripts/run_tastvg_spatial_rank_s11_v1.py','methods/CURRENT_METHOD.json','vg_tta/tastvg_spatial_online_opd_s1_v1.py']
    write(OUT/'LOCK.json',dict(rows=p['rows'],conditions=CONDS,pins={f:sha(ROOT/f) for f in paths},input_hashes={str(f.relative_to(ROOT)):sha(f) for f in [S0/'H_BARRIER.json',S0/'EXPERT_BARRIER.json',NATIVE/'PARAMETER_SUPPORT.pt',CRITIC/'REWARD_BARRIER.json',RAW/'PREDICTION_BARRIER.json',RAW/'LOCK.json']},arm=ARM,rho_u=.01*read(NATIVE/'PARAMETER_SUPPORT.json')['radius'],rank_tolerance=1e-12,lr=.005,expert_indices=[0,4,8,12],steps=1,tauE=1,tauS=1,cap_seconds=900,time=time.time()))

def verify():
    p=read(OUT/'LOCK.json')
    for f,h in {**p['pins'],**p['input_hashes']}.items():assert sha(ROOT/f)==h,f
    return p

def run():
    p=verify();tick=time.monotonic();done=0;updates=0;failure=None;state='failed';lease=None;actor=None
    try:
        import torch,numpy as np
        from methods.decota_final_simplified_v1.tensors import state_hash,detached
        from scripts.run_tastvg_evidence_vulnerability_v2 import install_clean_loader
        from scripts.run_tastvg_evidence_vulnerability_v1 import device_tree
        from scripts.run_final_simplification_v1 import lease as gpu_lease
        from vg_tta.tastvg_spatial_rank_s11_v1 import SpatialActor,reverse_kl,average_ranks,update_scale
        from vg_tta.tastvg_native_spatial_rollout_s05_v1 import predict,reinsert
        from vg_tta.tastvg_spatial_critic_s06_v1 import rewards
        from vg_tta.exact_frame_decode_audit_v2 import decode
        procs=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True);assert all('/opt/todesk/' in l for l in procs.splitlines() if l.strip()),procs;lease=gpu_lease()
        torch.set_num_threads(4);torch.manual_seed(20260929);np.random.seed(20260929);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        def guard(event,args):
            if event=='open' and args and isinstance(args[0],(str,bytes)) and any(x in str(args[0]) for x in ['GT_SUBSET','/ROWS.json','/SUMMARY.json']):raise PermissionError('S1 prediction worker forbids GT-derived metrics')
        sys.addaudithook(guard);install_clean_loader()
        from scripts.run_spatial_regression_alignment_v1 import model_load
        model=model_load('hcstvg1_test').eval().requires_grad_(False);mh=state_hash(model.state_dict());assert mh==read(OLD/'CAPTURE_BARRIER.json')['model_state_sha256'];actor=SpatialActor(model)
        support=load(NATIVE/'PARAMETER_SUPPORT.pt');deltas=[{n:(v-support['center'][n]).cuda() for n,v in s.items()} for s in support['states']];assert all(torch.equal(v.cpu(),support['center'][n]) for n,v in actor.initial.items())
        hb=read(S0/'H_BARRIER.json');eb=read(S0/'EXPERT_BARRIER.json');coeff=[model.cfg.SOLVER.BBOX_COEF,model.cfg.SOLVER.GIOU_COEF];rein=[];streams=[]
        for cond in CONDS:
            actor.restore(actor.initial);previous=state_hash(actor.state());assert previous==state_hash(actor.initial);stream_start=previous;local_updates=0
            for arrival,row in enumerate(p['rows']):
                assert time.monotonic()-tick<890 and shutil.disk_usage(ROOT).free>8*2**30
                name=f"{row['ordinal']:03}.pt";hr=f'capture/{cond}/{name}';assert sha(S0/hr)==hb['files'][hr];data=device_tree(load(S0/hr),'cuda');before=actor.state();prehash=state_hash(before);assert prehash==previous
                with torch.no_grad():ev,boxes,pre=actor.values(data)
                if arrival==0:assert torch.equal(boxes,data['prediction']['boxes']) and all(torch.equal(a,b.cpu()) for a,b in zip(pre['logits'],data['prediction']['logits']))
                scheduled=arrival in p['expert_indices'];result=dict(parent=row['ordinal'],condition=cond,arrival=arrival,expert_scheduled=scheduled,prediction=detached(pre,'cpu'),pre_state=detached(before,'cpu'),pre_state_sha256=prehash,pixel_sha256=data['pixel_sha256'],updated=False,GT_read=False)
                post_ev=ev;post=pre
                if scheduled:
                    candidates=[]
                    for k,delta in enumerate(deltas):
                        state_k={n:before[n]+delta[n] for n in before}
                        with torch.no_grad():cev,cb,cp=predict(model,data,state_k)
                        if k==0:assert torch.equal(cp['boxes'],pre['boxes'])
                        candidates.append(dict(prediction=detached(cp,'cpu')));del cev,cb,cp
                    # Evidence is read only after current-policy candidates exist.
                    er=f'expert/{cond}/{name}';assert sha(S0/er)==eb['files'][er];expert=load(S0/er);assert expert['pixel_sha256']==data['pixel_sha256']
                    score=rewards([c['prediction']['boxes'].numpy() for c in candidates],expert['boxes'],expert['valid']);result.update(candidates=candidates,rewards=score.tolist() if score is not None else None,valid_expert_frames=int(expert['valid'].sum()))
                    if score is not None:
                        target=torch.stack([c['prediction']['boxes'] for c in candidates]).cuda().detach();rt=torch.tensor(score,device='cuda',dtype=torch.float64).detach()
                        evg,bg,pg=actor.values(data);assert torch.equal(bg.detach().cpu(),pre['boxes'])
                        loss,pi,qi,dist=reverse_kl(bg,target,rt,coeff);grad=torch.autograd.grad(loss,[v for _,v in actor.named]);assert all(torch.isfinite(g).all() for g in grad)
                        scale,global_norm=update_scale(grad,ARM,p['lr'],p['rho_u'])
                        result['update']=dict(arm=ARM,rank=average_ranks(score).tolist(),update_scale=scale,global_gradient_norm=global_norm,requested_step_norm=(p['rho_u'] if ARM=='norm' and global_norm else p['lr']*global_norm),loss_before=float(loss.detach()),p=pi.detach().cpu(),q=qi.detach().cpu(),distances=dist.detach().cpu(),gradients={n:g.detach().cpu() for (n,_),g in zip(actor.named,grad)},gradient_norm=float(torch.sqrt(sum(g.square().sum() for g in grad))),query_gradient_norm=float(grad[0].norm()),LN_gradient_norm=float(torch.sqrt(sum(g.square().sum() for g in grad[1:]))),coefficients=coeff,lr=p['lr'],candidate_targets_detached=not target.requires_grad,reward_detached=not rt.requires_grad)
                        with torch.no_grad():
                            for (_,param),g in zip(actor.named,grad):param.add_(g,alpha=-scale)
                            post_ev,bpost,post=actor.values(data);after_loss,_,_,_=reverse_kl(bpost,target,rt,coeff)
                        result['update']['loss_after']=float(after_loss);result['updated']=global_norm>0;updates+=int(result['updated']);local_updates+=int(result['updated'])
                        del evg,bg,pg,loss,pi,qi,dist,grad,target,rt,bpost
                    del expert,candidates
                after=actor.state();posthash=state_hash(after);result.update(post_prediction=detached(post,'cpu'),post_state=detached(after,'cpu'),post_state_sha256=posthash,parameter_displacement=float(torch.sqrt(sum((after[n]-before[n]).square().sum() for n in before))),displacement_from_source=float(torch.sqrt(sum((after[n]-actor.initial[n]).square().sum() for n in before))))
                if cond in ['clean','frame_drop_5'] and ((result['updated'] and local_updates==1) or arrival==15):
                    frames,ids=decode(row['input']);assert ids==row['frame_ids'];shifted,_=observation(row,cond,frames);audit=reinsert(model,shifted,row,after,post_ev);rein.append(dict(parent=row['ordinal'],condition=cond,arrival=arrival,learned_state=result['displacement_from_source']>0,**audit));del frames,shifted
                assert state_hash(actor.state())==posthash;previous=posthash;save(OUT/'online'/cond/name,result);done+=1;print(done,96,cond,arrival,'expert',scheduled,'updated',result['updated'],'state_delta',result['displacement_from_source'],flush=True)
                status(OUT/'STATUS.json',dict(status='running',done=done,updates=updates));del data,ev,boxes,pre,post,post_ev,before,after,result;gc.collect();torch.cuda.empty_cache()
            streams.append(dict(condition=cond,start_sha256=stream_start,final_sha256=previous,updates=local_updates));save(OUT/'final_states'/f'{cond}.pt',detached(actor.state(),'cpu'))
        actor.close();actor=None;assert state_hash(model.state_dict())==mh;verify();assert len(rein)==4 and all(r['learned_state'] for r in rein)
        write(OUT/'REINSERTION_AUDIT.json',dict(status='pass',cells=rein));write(OUT/'STATE_CHAIN.json',dict(status='pass',streams=streams));write(OUT/'PREDICTION_BARRIER.json',dict(cells=96,updates=updates,files={str(f.relative_to(OUT)):sha(f) for f in (OUT/'online').rglob('*.pt')},final_states={str(f.relative_to(OUT)):sha(f) for f in (OUT/'final_states').glob('*.pt')},GT_read=False,model_restored=True,time=time.time()));state='completed'
    except BaseException as e:failure=dict(error=repr(e),traceback=traceback.format_exc());raise
    finally:
        if actor:actor.close()
        r=dict(status=state,done=done,updates=updates,seconds=time.monotonic()-tick,failure=failure,time=time.time());write(OUT/'allocations'/f'{time.time_ns()}.json',r);status(OUT/'STATUS.json',r)
        if lease:lease.close()

if __name__=='__main__':prepare() if sys.argv[1]=='prepare' else run()
