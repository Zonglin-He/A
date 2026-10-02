"""Actual A states, same nine probes, query-only readout and temporary writes."""
import os,sys,time,gc,traceback,functools
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
from scripts.tastvg_correction_views_common_v1 import *

def run(ds,split,stage):
    import torch,numpy as np
    from scripts.run_tastvg_evidence_vulnerability_v2 import install_clean_loader
    from scripts.run_tastvg_evidence_vulnerability_v1 import device_tree
    from scripts.run_final_simplification_v1 import lease
    from scripts.run_tastvg_paper48_p5_online_v1 import compact_prediction
    from vg_tta.tastvg_event_support_v1 import OnlineMethod
    from vg_tta.tastvg_spatial_online_opd_s1_v1 import SpatialActor
    from vg_tta.tastvg_selected_rollout_v1 import central_with_candidates,rollout_states,target_loss
    from vg_tta.tastvg_native_spatial_rollout_s05_v1 import predict,central_state
    from vg_tta.tastvg_ur_write_decomposition_v1 import sgd,difference,gradient_geometry
    from vg_tta.tastvg_spatial_critic_s06_v1 import rewards
    from vg_tta.tastvg_current_correction_views_v1 import select
    from methods.decota_final_simplified_v1.tensors import state_hash
    verify();p=plan(ds);cfg=p['params'];budget();install_clean_loader();sys.addaudithook(guard)
    lh=lease();tick=time.monotonic();torch.set_num_threads(4);torch.manual_seed(20260929);np.random.seed(20260929)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    from scripts.run_spatial_regression_alignment_v1 import model_load
    model=model_load('hcstvg1_test' if ds=='vidstg' else 'vidstg_test').eval().requires_grad_(False)
    bar=read(POOL/ds/'CAPTURE_BARRIER.json');mh=state_hash(model.state_dict());assert mh==bar['checkpoint_state_sha256']
    center=central_state(model);states,spec,basis=rollout_states(center,cfg['rho'],cfg['direction_count'])
    deltas=[{n:v-center[n] for n,v in z.items()} for z in states]
    counts=dict(suffix_replays=0,backward_calls=0,new_backbone_calls=0,persistent_live_control_arrivals=0)
    @functools.lru_cache(maxsize=32)
    def cached(parent,cond):
        f=POOL/ds/'capture'/cond/f'{parent:05}.json';assert sha(f)==bar['files'][str(f.relative_to(POOL/ds))]
        r=read(f);cf=POOL/ds/r['cache'];assert sha(cf)==r['sha256'];return load(cf),r
    def data_at(parent,cond):
        d,r=cached(parent,cond);return device_tree(d,'cuda'),r
    # Verify the original persistent A writer AND a subsequent nonexpert live.
    if stage=='round1' and split=='search':
        a=OnlineMethod(model,deltas,lr=cfg['lr'],teacher_temperature=cfg['teacher_temperature'],
            student_temperature=cfg['student_temperature'],steps=cfg['steps'],basis=basis,
            **read(OLD/ds/'A/REQUEST.json')['method'])
        seq=p['splits']['search']['orders']['order1'];parity=[]
        for at,parent in enumerate(seq[:2]):
            data,cr=data_at(parent,'clean')
            x,_=a.arrive(data,at%4==0,
                lambda:dict(**expert(ds,'temporal',parent,'clean',cr['pixel_sha256'])[0],pixel_sha256=cr['pixel_sha256']),
                lambda:dict(**expert(ds,'spatial',parent,'clean',cr['pixel_sha256'])[0],pixel_sha256=cr['pixel_sha256']))
            old=oldcell(ds,'search','clean','order1',at)
            assert x['pre_state_sha256']==old['pre_sha'] and x['post_state_sha256']==old['post_sha']
            assert torch.equal(x['prediction']['boxes'],old['slow']['boxes']) and x['output_prediction']['indices']==old['final_indices']
            for new,prior in zip(x['update_steps'],old['update_steps']):
                assert new['post_state_sha256']==prior['post_state_sha256']
                if new['update']:
                    for n,g in new['update']['gradients'].items():assert torch.equal(g,prior['update']['gradients'][n])
            parity.append(dict(arrival=at,exact_prediction_state_gradient=True));counts['persistent_live_control_arrivals']+=1
        a.close();assert state_hash(model.state_dict())==mh
        write(BASE/ds/'A_LIVE_PARITY.json',dict(status='pass',arrivals=parity,GT_read=False,time=time.time()))
    actor=SpatialActor(model);coeff=[model.cfg.SOLVER.BBOX_COEF,model.cfg.SOLVER.GIOU_COEF]
    out=BASE/ds/stage/split;jobs=[r for r in read(BASE/'COHORT.json')['cells'] if r['dataset']==ds and r['split']==split]
    def timed(pred,idx,ids):
        return dict(pred,indices=list(idx),physical_interval=[ids[idx[0]],ids[idx[1]]+1])
    for done,cell in enumerate(jobs,1):
        budget();cond,order,at,parent=[cell[k] for k in ['condition','order','arrival','parent']]
        path=out/'predictions'/cond/order/f'{at:05}.pt'
        if path.with_suffix('.json').exists():checked(path);continue
        old=oldcell(ds,split,cond,order,at);ids=p['rows'][parent]['frame_ids'];base=timed(old['slow'],old['final_indices'],ids)
        value=dict(cell=cell,A=base,persistent_pre_sha=old['pre_sha'],persistent_post_sha=old['post_sha'],
            persistent_payload_sha256=sha(ROOT/cell['old_payload']),persistent_unchanged=True,GT_read=False)
        if not cell['scheduled']:
            value['predictions']={b:base for b in ARMS};commit(path,value);continue
        data,cr=data_at(parent,cond);assert cr['pixel_sha256']==old['pixel_sha256']
        state=device_tree(old['pre_state'],'cuda');actor.restore(state)
        with torch.no_grad():_,_,pre,_,tc=central_with_candidates(actor,data)
        counts['suffix_replays']+=1
        assert torch.equal(pre['boxes'],old['slow']['boxes']) and pre['indices']==old['slow']['indices']
        assert tc==old['temporal']['candidates'];prefix=f'{split}_{cond}_{order}_{at:05}'
        targets=[]
        for j,delta in enumerate(deltas):
            # Exact new replay also checks the historical nine candidates, when saved.
            _,_,cp=predict(model,data,{n:state[n]+delta[n] for n in state});counts['suffix_replays']+=1
            z=compact_prediction(cp);targets.append(z)
            saved=old['update_steps'][0].get('candidates')
            if saved:assert torch.equal(z['boxes'],saved[j]['prediction']['boxes'])
        assert torch.equal(targets[0]['boxes'],old['slow']['boxes'])
        branches=['R'] if stage=='round1' else ['Rnew','U2']
        evidence={};receipts={}
        evidence['U'],receipts['U']=expert(ds,'spatial',parent,cond,cr['pixel_sha256'])
        for branch in branches:
            r=read(BASE/ds/'expert_receipts'/f'{prefix}_{branch}.json');assert r['pixel_sha256']==cr['pixel_sha256']
            assert sha(ROOT/r['cache'])==r['cache_sha256'];evidence[branch]=load(ROOT/r['cache']);receipts[branch]=r
        rew={b:rewards([z['boxes'].numpy() for z in targets],e['boxes'],e['valid']) for b,e in evidence.items()}
        grads={};meta={};available=[b for b in evidence if rew[b] is not None]
        if available:
            actor.restore(state);_,bg,_=actor.values(data);counts['suffix_replays']+=1
            target=torch.stack([z['boxes'] for z in targets]).cuda().detach()
            for j,b in enumerate(available):
                loss,pi,qi,d,_,_=target_loss(bg,target,torch.tensor(rew[b],device='cuda',dtype=torch.float64),
                    coeff,cfg['teacher_temperature'],cfg['student_temperature'],'rank',1.)
                gs=torch.autograd.grad(loss,[v for _,v in actor.named],retain_graph=j<len(available)-1)
                counts['backward_calls']+=1;assert all(torch.isfinite(g).all() for g in gs)
                grads[b]={n:g.detach().cpu() for (n,_),g in zip(actor.named,gs)}
                meta[b]=dict(loss=float(loss.detach()),p=pi.detach().cpu(),q=qi.detach().cpu(),distances=d.detach().cpu(),coefficients=coeff)
        for b in evidence:
            if b not in grads:grads[b]={n:torch.zeros_like(v).cpu() for n,v in state.items()};meta[b]=None
        if old['update_steps'][0]['update']:
            for n,g in grads['U'].items():assert torch.equal(g,old['update_steps'][0]['update']['gradients'][n])
        outputs={};temporary={};specific={};common={n:v.cpu() for n,v in state.items()}
        for b in branches:
            sel=select(rew[b]);z=timed(targets[sel['selected']],old['final_indices'],ids);outputs[b+'_select']=z
            g=grads[b];eligible=rew[b] is not None
            for name,gradient,ok in [(b+'_temp',g,eligible),(b+'_specific',difference(g,grads['U']),eligible and rew['U'] is not None)]:
                st=sgd(common,gradient,cfg['lr']) if ok else common
                _,_,cp=predict(model,data,device_tree(st,'cuda'));counts['suffix_replays']+=1
                z=timed(compact_prediction(cp),old['final_indices'],ids);outputs[name]=z;temporary[name]=st
            specific[b]=gradient_geometry(grads['U'],grads[b])
        us=select(rew['U']);u=timed(targets[us['selected']],old['final_indices'],ids);outputs['U_select']=u
        if stage=='round1':
            predictions=dict(A=base,U_select=outputs['U_select'],R_select=outputs['R_select'],R_temp=outputs['R_temp'],Specific_temp=outputs['R_specific'])
        else:predictions=dict(A=base,**outputs)
        value.update(predictions=predictions,probes=targets,pre_state=common,temporary_states=temporary,
            gradients=grads,metadata=meta,rewards={b:r.tolist() if r is not None else None for b,r in rew.items()},
            selections={b:select(r) for b,r in rew.items()},evidence=receipts,gradient_geometry=specific,
            temporal_candidates=tc,coefficients=coeff,params=cfg,probe_spec=spec)
        actor.restore(actor.initial);assert state_hash(model.state_dict())==mh
        commit(path,value);status(out/'STATUS.json',dict(status='running',done=done,total=len(jobs),worker_pid=os.getpid(),GT_read=False,time=time.time()))
        print('CORRECTION',stage,split,ds,done,len(jobs),flush=True);del data,targets,value;gc.collect();torch.cuda.empty_cache()
    actor.close();assert state_hash(model.state_dict())==mh;verify()
    write(out/'PREDICTION_BARRIER.json',dict(status='sealed',cells=len(jobs),GT_read=False,model_restored=True,
        files={str(f.relative_to(out)):sha(f) for f in (out/'predictions').rglob('*.json')},time=time.time()))
    write(out/'RESOURCES.json',dict(**counts,worker_wall_seconds=time.monotonic()-tick,
        peak_allocated_vram_bytes=torch.cuda.max_memory_allocated(),wall_includes_loading_IO=True,GT_read=False))
    status(out/'STATUS.json',dict(status='completed',done=len(jobs),total=len(jobs),GT_read=False,time=time.time()));lh.close()
if __name__=='__main__':
    try:run(*sys.argv[1:])
    except BaseException as e:
        status(BASE/'WORKER_FAILURE.json',dict(status='failed',error=repr(e),traceback=traceback.format_exc(),time=time.time()));raise
