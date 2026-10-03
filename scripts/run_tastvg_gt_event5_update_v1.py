"""One query-scoped ordinary SGD with GT-event masks; never write persistent A."""
import os,sys,time,gc,traceback,functools
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
from scripts.tastvg_oracle_event5_common_v1 import *
def run(ds):
    import torch,numpy as np
    from scripts.run_tastvg_evidence_vulnerability_v2 import install_clean_loader
    from scripts.run_tastvg_evidence_vulnerability_v1 import device_tree
    from scripts.run_final_simplification_v1 import lease
    from scripts.run_tastvg_paper48_p5_online_v1 import compact_prediction
    from vg_tta.tastvg_spatial_online_opd_s1_v1 import SpatialActor
    from vg_tta.tastvg_selected_rollout_v1 import target_loss
    from vg_tta.tastvg_native_spatial_rollout_s05_v1 import predict
    from vg_tta.tastvg_ur_write_decomposition_v1 import sgd
    from vg_tta.tastvg_spatial_critic_s06_v1 import rewards
    from vg_tta.tastvg_current_correction_views_v1 import select
    from methods.decota_final_simplified_v1.tensors import state_hash
    verified();p=plan(ds);cfg=p['params'];budget();install_clean_loader();sys.addaudithook(guard)
    lh=lease();tick=time.monotonic();torch.set_num_threads(4);torch.manual_seed(20260929);np.random.seed(20260929)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    from scripts.run_spatial_regression_alignment_v1 import model_load
    model=model_load('hcstvg1_test' if ds=='vidstg' else 'vidstg_test').eval().requires_grad_(False)
    bar=read(POOL/ds/'CAPTURE_BARRIER.json');mh=state_hash(model.state_dict());assert mh==bar['checkpoint_state_sha256']
    actor=SpatialActor(model);coeff=[model.cfg.SOLVER.BBOX_COEF,model.cfg.SOLVER.GIOU_COEF]
    counts=dict(suffix_replays=0,backwards=0,uniform_gradient_controls=0,new_backbone_calls=0,probe_regenerations=0)
    @functools.lru_cache(maxsize=12)
    def cached(parent,cond):
        f=POOL/ds/'capture'/cond/f'{parent:05}.json';assert sha(f)==bar['files'][str(f.relative_to(POOL/ds))]
        r=read(f);cf=POOL/ds/r['cache'];assert sha(cf)==r['sha256'];return load(cf),r
    jobs=[c for c in read(BASE/'EVENT_PLAN.json')['events'] if c['dataset']==ds];out=BASE/ds/'updates'
    for done,c in enumerate(jobs,1):
        budget();path=out/'predictions'/f'{prefix(c)}.pt'
        if path.with_suffix('.json').exists():checked(path);continue
        x=cached_payload(c);er=read(BASE/ds/'expert_receipts'/f'{prefix(c)}.json')
        value=dict(cell=c,eligible=c['eligible'],A=x['A'],pre_sha=x['persistent_pre_sha'],persistent_post_sha=x['persistent_post_sha'],
            persistent_unchanged=True,GT_assisted_observation=True,raw_GT_read=False,cached_probes_sha256=sha(payload_path(c)))
        if not c['eligible']:
            value.update(reason='sampling_support_insufficient',post_prediction=x['A'],selected=0,updated=False)
            commit(path,value);continue
        assert er['eligible'] and sha(ROOT/er['cache'])==er['cache_sha256'];ev=load(ROOT/er['cache'])
        reward=rewards([z['boxes'].numpy() for z in x['probes']],ev['boxes'],ev['valid']);sel=select(reward)
        data,cr=cached(c['parent'],c['condition']);assert cr['pixel_sha256']==c['pixel_sha256'];data=device_tree(data,'cuda')
        state=x['pre_state'];actor.restore(device_tree(state,'cuda'));_,bg,pre=actor.values(data);counts['suffix_replays']+=1
        assert torch.equal(pre['boxes'],x['A']['boxes'])
        target=torch.stack([z['boxes'] for z in x['probes']]).cuda().detach();grad={n:torch.zeros_like(v) for n,v in state.items()};meta=None
        # Reproduce cached Uniform VJP bitwise on this exact central graph.
        if x['rewards']['U'] is not None:
            loss,*_=target_loss(bg,target,torch.tensor(x['rewards']['U'],device='cuda',dtype=torch.float64),
                coeff,cfg['teacher_temperature'],cfg['student_temperature'],'rank',1.)
            gs=torch.autograd.grad(loss,[v for _,v in actor.named],retain_graph=reward is not None)
            for (n,_),g in zip(actor.named,gs):assert torch.equal(g.detach().cpu(),x['gradients']['U'][n])
            counts['backwards']+=1;counts['uniform_gradient_controls']+=1
        if reward is not None:
            loss,pi,qi,d,_,_=target_loss(bg,target,torch.tensor(reward,device='cuda',dtype=torch.float64),
                coeff,cfg['teacher_temperature'],cfg['student_temperature'],'rank',1.)
            gs=torch.autograd.grad(loss,[v for _,v in actor.named]);assert all(torch.isfinite(g).all() for g in gs)
            grad={n:g.detach().cpu() for (n,_),g in zip(actor.named,gs)};counts['backwards']+=1
            meta=dict(loss=float(loss.detach()),p=pi.detach().cpu(),q=qi.detach().cpu(),distances=d.detach().cpu())
        temporary=sgd(state,grad,cfg['lr']) if reward is not None else state
        if reward is not None:
            _,_,cp=predict(model,data,device_tree(temporary,'cuda'));counts['suffix_replays']+=1
            post=compact_prediction(cp)
        else:post=x['A']
        value.update(evidence=er,rewards=reward.tolist() if reward is not None else None,selection=sel,
            selected=sel['selected'],selected_prediction=x['probes'][sel['selected']],updated=reward is not None,
            post_prediction=post,pre_state=state,temporary_state=temporary,gradients=grad,metadata=meta,coefficients=coeff,
            temporary_sha=state_hash(temporary),params=cfg,inner_steps=1,
            displacement=float(torch.sqrt(sum((temporary[n]-v).double().square().sum() for n,v in state.items()))))
        actor.restore(actor.initial);assert state_hash(model.state_dict())==mh
        commit(path,value);status(out/'STATUS.json',dict(status='running',done=done,total=len(jobs),worker_pid=os.getpid(),time=time.time()))
        print('GT_EVENT_TEMP_UPDATE',ds,done,len(jobs),flush=True);del data,bg,pre,value;gc.collect();torch.cuda.empty_cache()
    actor.close();assert state_hash(model.state_dict())==mh;verified()
    write(out/'PREDICTION_BARRIER.json',dict(status='sealed',donors=len(jobs),raw_GT_read=False,GT_assisted_observation=True,
        model_restored=True,files={str(f.relative_to(out)):sha(f) for f in (out/'predictions').glob('*.json')},time=time.time()))
    write(out/'RESOURCES.json',dict(**counts,worker_wall_seconds=time.monotonic()-tick,
        peak_allocated_vram_bytes=torch.cuda.max_memory_allocated(),wall_includes_loading_IO=True,time=time.time()))
    status(out/'STATUS.json',dict(status='completed',done=len(jobs),total=len(jobs),time=time.time()));lh.close()
if __name__=='__main__':
    try:run(sys.argv[1])
    except BaseException as e:
        status(BASE/'UPDATE_FAILURE.json',dict(status='failed',dataset=sys.argv[1],error=repr(e),traceback=traceback.format_exc(),time=time.time()));raise
