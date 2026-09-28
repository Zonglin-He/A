"""Fixed source cases: free merger control, then gated frozen-output-span control."""
import argparse,os,sys,time,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from vg_tta.desta3d_v3_oracle_io import OUT,PANEL,read,write,sha,local_dependencies,check_pins,allocation,seal,tensor_sha
from scripts.desta3d_v3_privileged_ptd_qualification import B1,B1_SHA,ADAPTER_SHA,equal
PROTOCOL=ROOT/'protocols/desta3d_v3_free_actuation_v2.md'
CASES=[(7,'event'),(3,'spatial')]


def register(name,mode):
    dest=OUT/name;assert not dest.exists();assert mode in ('free','span')
    pre=read(OUT/'FREE_ACTUATION_CPU_PREFLIGHT.json');check_pins(pre['pins']);assert pre['status']=='passed'
    check_pins(read(OUT/'FREE_ACTUATION_CPU_PREFLIGHT_V2.json')['pins'])
    if mode=='span':
        d=read(OUT/'actuation_free002'/'ROOT_DECISION.json')
        assert d['proceed_span'] and d['fixed_endpoint_only'],'Free native control has not qualified span comparison'
    rows=read(PANEL/'INPUTS.json');selected=[dict(rows[i],diagnostic_branch=b,original_index=i) for i,b in CASES]
    cfg=dict(mode=mode,steps=30,lr=.01,wd=0.,clip=1.,seed=20260927,phase_seconds=3600,minimum_free_bytes=8*2**30,
       maximum_new_bytes=10*2**30,cumulative_cap=None,source_GT_in_worker=True,target_input=False,target_GT=False,
       source_GT_purpose='explicit per-source native endpoint/coordinate supervision; not TTA',
       checkpoint=str(B1),checkpoint_sha=B1_SHA,adapter_sha=ADAPTER_SHA,queries=2,parents=2,
       selection='prior oracle outcomes: sole correct-T flip19368; most negative S-correct-minus-wrong28546',
       optimizer='fresh AdamW for one free tensor per query; all PTD/adapter parameters frozen; no scheduler',
       objective='mean endpoint CE over observed time classes or mean coordinate CE on native anchors with known referent boxes',
       support='Each step fresh native trace of current intervention; native reference/time/anchors, never GT prefix',
       span_parameterization='orthonormal QR(W) times sqrt(C/d); equivalent g W delta_z, stored gate unchanged; token-space units',
       success='fixed step30: T tIoU improves AND both GT endpoint argmax correct; S fixed-support sIoU improves AND CE decreases; no best-step selection',
       limitations='two exposed outcome-selected train cases; failure does not prove mathematical non-controllability')
    cfg.update(recovery_from='actuation_free001' if mode=='free' else None, support_contract='actual native box-probe query IDs, independent of parsed format validity')
    write(dest/'CONFIG.json',cfg);write(dest/'INPUTS.json',selected)
    extra=[]
    if mode=='free':
        import shutil
        old=OUT/'actuation_free001';assert read(old/'RECEIPT.json')['status']=='failed'
        reused={}
        for ep in sorted((old/'episodes').iterdir()):
            dst=dest/'episodes'/ep.name;dst.mkdir(parents=True)
            for f in ep.iterdir():
                if ep.name=='01' and f.name=='NATIVE_24.pt':continue
                if f.is_file():os.link(f,dst/f.name);reused[str((dst/f.name).relative_to(dest))]=sha(f)
        write(dest/'REUSE_SEAL.json',{'source':str(old),'files':reused,'actual_steps_already_done':54,'remaining_new_steps':6,'failed_native24_replay_required':True})
        extra=[dest/'REUSE_SEAL.json',old/'RECEIPT.json',old/'episodes/01/NATIVE_24.pt']
    paths=[*extra,OUT/'FREE_ACTUATION_CPU_PREFLIGHT_V2.json',Path(__file__),PROTOCOL,OUT/'FREE_ACTUATION_CPU_PREFLIGHT.json',dest/'CONFIG.json',dest/'INPUTS.json',B1,PANEL/'SOURCE_RECORDS.json',
           OUT/'diagnosis_cpu_v1/REPORT.json',ROOT/'external/ParallelTubeDecoding/src/model/ptd_generation.py',
           ROOT/'checkpoints/ParallelTubeDecoding-Qwen3-VL-4B/model.safetensors']
    for i,b in CASES:paths.append(OUT/'oracle001/episodes'/f'{i:02}'/'original.pt')
    write(dest/'LOCK.json',{'pins':{str(p):sha(p) for p in local_dependencies(paths)}})
    write(dest/'REGISTRATION.json',{'time':time.time(),'status':'registered_before_GPU','proposed_total_steps':60,'mode':mode})


def run(name):
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';dest=OUT/name
    with allocation(dest) as (cfg,guard):
        import torch,gc
        from scripts.ptd_spatial_adapter_ab_v1 import processor_load,model_load,inputs_for
        from scripts.desta3d_v2_source_fit import prediction_record
        from scripts.desta3d_v2_reference_audit_cached_v3 import details
        from scripts.desta3d_v2_p0 import adapter_sha256
        from vg_tta.desta3d_v2 import Desta3DAdapterV2
        from vg_tta.desta3d_v2_ptd import capture_stock_fields
        from vg_tta.desta3d_v2_output_anchor import capture_teacher
        from vg_tta.desta3d_v2_output_anchor_memory_v7 import replay_branch,release_free_host_arenas
        from vg_tta.desta3d_v3_actuation_support import spatial_positions
        from vg_tta.desta3d_v3_free_actuation import inject_delta,token_delta,span_basis,native_targets,native_ce,margin
        from vg_tta.exact_frame_decode_audit_v2 import decode
        from vg_tta.optimizer_checkpoint import cpu_clone
        from vg_tta.desta3d_v2_prediction_contract import validate_prediction
        torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True
        torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
        pr=processor_load();model=model_load().eval().requires_grad_(False)
        adapter=Desta3DAdapterV2(hidden_dim=128,architecture='dual3d',p1_enabled=False).cuda().eval()
        adapter.load_state_dict(torch.load(B1,map_location='cpu',weights_only=False)['adapter']);adapter.set_train_stage('frozen')
        assert adapter_sha256(adapter)==ADAPTER_SHA
        labels={r['key']:r for r in read(PANEL/'SOURCE_RECORDS.json')};count=0;new_steps=0;reconstruction=[]
        if cfg.get('recovery_from'):check_pins({str(dest/k):v for k,v in read(dest/'REUSE_SEAL.json')['files'].items()})
        for index,row in enumerate(read(dest/'INPUTS.json')):
            guard();ep=dest/'episodes'/f'{index:02}';ep.mkdir(parents=True,exist_ok=True)
            if (ep/'COMPLETE.json').exists():
                assert cfg.get('recovery_from') and index==0;count+=31;continue
            branch=row['diagnostic_branch'];lab=labels[row['key']]
            frames,ids=decode(row['input']);prompt,pre=inputs_for(row,pr,frames)
            fields=capture_stock_fields(model,pr,prompt,row['input']['caption'],ids,row['input']['fps'])
            old=torch.load(OUT/'oracle001/episodes'/f"{row['original_index']:02}"/'original.pt',map_location='cpu',weights_only=False)
            supports={k:tensor_sha(v) for k,v in fields.items() if isinstance(v,torch.Tensor)}
            supports.update({k:tensor_sha(prompt[k]) for k in ['input_ids','pixel_values_videos','video_grid_thw']})
            assert supports==old['support'] and equal(pre,old['preprocess'])
            if not (ep/'INPUT.json').exists():write(ep/'INPUT.json',dict(key=row['key'],branch=branch,support=supports,original_replay_pending=True))
            basis=rr=None;shape=list(fields['visual_grid'].shape)
            if cfg['mode']=='span':
                basis,rr=span_basis(getattr(adapter,'out_proj_'+branch).weight);shape[-1]=basis.shape[1]
                torch.save({'basis':basis.cpu(),'R':rr.cpu(),'gate':getattr(adapter,'gate_'+branch).sigmoid().cpu(),
                            'scale':math.sqrt(basis.shape[0]/basis.shape[1])},ep/'BASIS.pt')
            parameter=torch.nn.Parameter(torch.zeros(shape,device='cuda',dtype=torch.float32))
            optimizer=torch.optim.AdamW([parameter],lr=cfg['lr'],weight_decay=cfg['wd'])
            history=[];initial_readout=None;start_step=0
            if cfg.get('recovery_from'):
                # Replay only saved gradients through the identical CUDA Adam arithmetic;
                # assert every saved actual delta and before/after parameter hash.
                for k in range(24):
                    r=torch.load(ep/f'STEP_{k+1:02}.pt',map_location='cpu',weights_only=False)
                    prev=torch.load(ep/f'NATIVE_{k:02}.pt',map_location='cpu',weights_only=False)
                    tr=torch.load(ep/f'TRACE_{k:02}.pt',map_location='cpu',weights_only=False)
                    pos=spatial_positions(tr);proxy={**prev,'positions':pos};tar,val=native_targets(lab,proxy,branch)
                    ex=tr['branches'][1]['logits']['coordinate'];ce=native_ce(ex,tar,val)
                    history.append({'step':k,'CE':float(ce),'GT_margin':margin(ex,tar,val),'interval':prev['interval'],
                        'native_argmax':ex.argmax(-1).tolist(),'valid_targets':int(val.sum()),'trace_sha':sha(ep/f'TRACE_{k:02}.pt'),
                        'parameter_norm':float(parameter.detach().double().norm())})
                    assert tensor_sha(parameter)==r['parameter_before_sha']
                    parameter.grad=r['gradient'].cuda();before=parameter.detach().clone()
                    norm=torch.nn.utils.clip_grad_norm_([parameter],cfg['clip']);optimizer.step()
                    assert torch.equal((parameter.detach()-before).cpu(),r['actual_delta'])
                    assert tensor_sha(parameter)==r['parameter_after_sha'] and int(optimizer.state[parameter]['step'])==k+1
                    reconstruction.append({'step':k+1,'exact_delta':True,'exact_before_after_hash':True,'counter':k+1})
                    parameter.grad=None;del r,prev,tr,ex,before;count+=1
                start_step=24
                write(ep/'ADAM_RECONSTRUCTION.json',{'source':'saved raw gradients, same CUDA AdamW/clip arithmetic','rows':reconstruction,
                    'optimizer_forward_replay_updates':24,'new_model_gradient_steps':0,'all_live_parameter_keys':all(x is parameter for x in optimizer.state)})
            for step in range(start_step,cfg['steps']+1):
                guard();release_free_host_arenas();torch.cuda.empty_cache()
                with inject_delta(adapter,branch,parameter,cfg['mode'],basis):native,trace=capture_teacher(model,pr,prompt,adapter,fields)
                p=prediction_record(native,row,pre,ADAPTER_SHA);p.update(readout=details(native),source_GT_update=True,target_read=False,
                    actuation=cfg['mode'],actuation_branch=branch,step=step,base_adapter_unchanged=True)
                p['time_distribution']={'endpoint_logits':trace['branches'][0]['logits'].get('time')}
                torch.save(trace,ep/f'TRACE_{step:02}.pt')
                if cfg.get('recovery_from') and step==24:
                    failed=torch.load(OUT/cfg['recovery_from']/'episodes/01/NATIVE_24.pt',map_location='cpu',weights_only=False)
                    keys=['interval','positions','boxes_cxcywh','geometry_valid','event_completion','spatial_completion','format_ok','readout','time_distribution']
                    checks={k:equal(p[k],failed[k]) for k in keys};write(ep/'FAILED_NATIVE_EXACT_REPLAY.json',checks);assert all(checks.values())
                torch.save(p,ep/f'NATIVE_{step:02}.pt');validate_prediction(p,len(ids));count+=1
                if step==0:
                    checks={k:equal(p[k],old[k]) for k in ['interval','positions','boxes_cxcywh','geometry_valid','event_completion','spatial_completion','format_ok','preprocess','readout']}
                    checks['time_logits']=torch.equal(p['time_distribution']['endpoint_logits'],old['time_distribution']['endpoint_logits'])
                    write(ep/'ZERO_NATIVE_REPLAY.json',checks);assert all(checks.values()),checks
                    initial_readout=p
                if branch=='spatial':
                    assert p['event_completion']==old['event_completion'] and p['interval']==old['interval']
                    assert torch.equal(p['time_distribution']['endpoint_logits'],old['time_distribution']['endpoint_logits'])
                    assert spatial_positions(trace)==old['positions']
                    assert equal(p['readout']['spatial_reference_token_ids'],old['readout']['spatial_reference_token_ids'])
                target_support={**p,'positions':spatial_positions(trace)} if branch=='spatial' else p
                targets,valid=native_targets(lab,target_support,branch)
                kind='time' if branch=='event' else 'coordinate';bi=0 if branch=='event' else 1
                expected=trace['branches'][bi]['logits'][kind]
                targets=targets.cuda();valid=valid.cuda()
                value=native_ce(expected.cuda(),targets,valid)
                native_entry={'step':step,'CE':float(value),'GT_margin':margin(expected,targets,valid),'interval':p['interval'],
                     'native_argmax':expected.argmax(-1).tolist(),'valid_targets':int(valid.sum()),'trace_sha':None,
                     'parameter_norm':float(parameter.detach().double().norm())}
                native_entry['trace_sha']=sha(ep/f'TRACE_{step:02}.pt')
                history.append(native_entry)
                del native,value,p;gc.collect();torch.cuda.empty_cache()
                if step==cfg['steps']:break
                optimizer.zero_grad(set_to_none=True)
                with inject_delta(adapter,branch,parameter,cfg['mode'],basis):
                    logits,audit=replay_branch(model,prompt,adapter,fields,trace,branch)
                    # Check every state, not just zero: differentiable replay is the same native policy on the saved native support.
                    exact=torch.equal(logits.detach().cpu(),expected)
                    torch.save({'native':expected,'replay':logits.detach().cpu(),'exact':exact,'targets':targets.cpu(),'valid':valid.cpu()},ep/f'FORWARD_{step:02}.pt')
                    assert exact,'Differentiable replay differs from current native distribution'
                    loss=native_ce(logits,targets,valid);loss.backward()
                grad=parameter.grad.detach().clone();assert torch.isfinite(grad).all() and grad.norm()>0
                assert all(x.grad is None and not x.requires_grad for x in model.parameters())
                assert all(x.grad is None and not x.requires_grad for x in adapter.parameters())
                before=parameter.detach().clone();norm=torch.nn.utils.clip_grad_norm_([parameter],cfg['clip']);optimizer.step();new_steps+=1
                actual=parameter.detach()-before;st=optimizer.state[parameter]
                assert len(optimizer.state)==1 and int(st['step'])==step+1 and next(iter(optimizer.state)) is parameter
                torch.save({'gradient':grad.cpu(),'actual_delta':actual.cpu(),'loss':float(loss),'gradient_norm':float(norm),
                    'counter':int(st['step']),'parameter_before_sha':tensor_sha(before),'parameter_after_sha':tensor_sha(parameter),
                    'replay_audit':audit,'live_parameter_binding':True,'clip_triggered':float(norm)>cfg['clip'],
                    'target':targets.cpu(),'valid':valid.cpu()},ep/f'STEP_{step+1:02}.pt')
                print('ACTUATION',row['key'],branch,cfg['mode'],step+1,'CE',float(loss),'grad',float(norm),flush=True)
                del before,actual,grad,logits,loss,trace,audit;parameter.grad=None;gc.collect()
                assert adapter_sha256(adapter)==ADAPTER_SHA
                assert sum(p.stat().st_size for p in dest.rglob('*') if p.is_file())<cfg['maximum_new_bytes']
            torch.save({'parameter':parameter.detach().cpu(),'optimizer':cpu_clone(optimizer.state_dict()),
                        'token_delta':token_delta(parameter,cfg['mode'],basis).detach().cpu()},ep/'FINAL.pt')
            write(ep/'TRAJECTORY.json',history)
            write(ep/'COMPLETE.json',{'key':row['key'],'branch':branch,'actual_steps':cfg['steps'],'counter':int(optimizer.state[parameter]['step']),
                'native_predictions':len(history),'adapter_sha':adapter_sha256(adapter),'all_model_parameters_frozen':True})
            del parameter,optimizer,fields,prompt,old,initial_readout,trace,expected;gc.collect();torch.cuda.empty_cache()
        write(dest/'COMPLETE.json',{'status':'completed_fixed_endpoint_not_scored','queries':2,'actual_steps':60,'predictions':count,
             'seal_sha':seal(dest,count),'new_model_gradient_steps':new_steps,'reconstructed_saved_Adam_updates':len(reconstruction),'GPU_peak_allocated':torch.cuda.max_memory_allocated(),'target_read':False})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['register','run']);p.add_argument('--name',required=True);p.add_argument('--mode',choices=['free','span'],default='free')
    a=p.parse_args();register(a.name,a.mode) if a.action=='register' else run(a.name)
