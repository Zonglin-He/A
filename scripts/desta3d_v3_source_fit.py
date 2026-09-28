"""Full mixed-source structured fit with actual complete-epoch scheduling.

Explicit register/run actions. Safe complete-window checkpoints, integer Adam
keys, physical input readback, separate sealed source-val scoring. No target.
"""
import argparse,fcntl,gc,hashlib,json,math,os,random,shutil,sqlite3,subprocess,sys,time,traceback
from pathlib import Path
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.desta3d_v2_p0 import read,sha,adapter_sha256
from scripts.score_desta3d_v2_aux_recovery import save_once
from scripts.desta3d_tta_run_v1 import tensor_sha256,scan_nested_gpu_receipts
from vg_tta.optimizer_checkpoint import cpu_clone,restore_optimizer,validate_serialized_optimizer
from vg_tta.desta3d_v3_data import SourcePool,balanced_epoch
from vg_tta.desta3d_v3_source import structured_support,source_branch_objective
from vg_tta.desta3d_v3_training import training_inputs,checked_step,optimizer_counters
BASE=ROOT/'artifacts/desta3d_v3';ROSTER=BASE/'full_source_roster_v1';OUT=BASE/'full_source_fit_v1'

def register():
    assert not (OUT/'REGISTRATION.json').exists()
    for p,h in read(ROSTER/'COMPLETE.json')['pins'].items():assert sha(p)==h,p
    assert read(OUT/'CPU_PREFLIGHT.json')['status']=='passed'
    for p,h in read(OUT/'CPU_PREFLIGHT.json')['pins'].items():assert sha(p)==h,p
    assert read(ROSTER/'ROOT_ROSTER_READBACK.json')['media_hash_overlap']==0
    assert read(BASE/'structured_source_interface_v1/ROOT_RAW_READBACK.json')['status']=='independently_audited'
    s=read(ROSTER/'SUMMARY.json')
    cfg=dict(seed=20260928,hidden_dim=128,architecture='dual3d',P1=False,joint=0,offset=False,
        source_counts=s,epochs_max=5,accum=4,total_horizon=s['total_horizon'],warmup_fraction=.05,
        optimizer='AdamW',reader_lr=3e-5,head_out_gate_lr=1e-4,weight_decay=0,clip=1,
        initialize='fresh full adapter seed20260928; not B1 continuation',
        objective=dict(endpoint_sum=1,coordinate_mean=1,referent_BCE=.1,event_BCE=.1,original_each_branch_CE=.05),
        augmentation='stateless per occurrence: 50% clean; otherwise independent brightness/contrast uniform[.95,1.05], uint8 round/clip; no temporal/geometry change or target severity',
        epoch_sampling='complete shuffled Vid once; shuffled HC cycles to same query count; interleave exactly50/50',
        state_selection='max equal-domain parent-macro clean source-val vIoU; ties earlier; patience2 only after at least3 completeepochs',
        source_gate='selected clean source mean >= physically matched Frozen; full CI/tails retained; not target success',
        validation='full fixed roster, trained plus Frozen gate0 on identical physical pixels; all seal before label SELECT/scoring',
        shared_referent_pooling=False,phase_seconds=3600,reserve_bytes=8*2**30,cap=None,
        target_inputs=False,target_GT=False,source_validation_exposure='internal official-train holdout; historically developed, not unseen claim')
    save_once(OUT/'CONFIG.json',cfg)
    pins=dict(read(BASE/'structured_source_interface_v1/LOCK.json')['pins'])
    files=[Path(__file__),ROOT/'vg_tta/desta3d_v3_data.py',ROOT/'vg_tta/desta3d_v3_training.py',ROOT/'vg_tta/optimizer_checkpoint.py',
        ROOT/'scripts/score_desta3d_v3_source_fit.py',ROOT/'protocols/desta3d_v3_source_fit_v1.md',
        ROOT/'vg_tta/desta3d_v2_shared_reference_cached.py',OUT/'CONFIG.json',OUT/'CPU_PREFLIGHT.json',ROSTER/'COMPLETE.json',ROSTER/'ROOT_ROSTER_READBACK.json']
    files += [ROOT/'vg_tta/desta3d_v2_training.py',ROOT/'scripts/ptd_8b_teacher_feasibility_v1.py',
        ROOT/'scripts/desta3d_source_fit_v1.py',ROOT/'vg_tta/metrics.py',ROOT/'scripts/score_desta3d_v2_reference_audit.py',
        ROOT/'vg_tta/exact_frame_decode_audit_v2.py',ROOT/'vg_tta/unanchored_dense_shift_data_v1.py']
    pins.update(read(ROSTER/'COMPLETE.json')['pins']);pins.update({str(p):sha(p) for p in files});pins.update(read(OUT/'CPU_PREFLIGHT.json')['pins'])
    for p,h in read(ROSTER/'REGISTRATION.json')['pins'].items():assert sha(p)==h,p
    pins.update(read(ROSTER/'REGISTRATION.json')['pins'])
    save_once(OUT/'LOCK.json',dict(pins=pins));save_once(OUT/'REGISTRATION.json',dict(time=time.time(),status='registered_before_fit',cap=None))

def save_state(state,adapter,optimizer):
    opt=cpu_clone(optimizer.state_dict());validate_serialized_optimizer(opt)
    payload=dict(**state,adapter=cpu_clone(adapter.state_dict()),optimizer=opt,cpu_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all(),lock_sha=sha(OUT/'LOCK.json'),schema='live_parameter_integer_Adam_v3')
    temp=OUT/'LATEST.tmp.pt';torch.save(payload,temp);os.replace(temp,OUT/'LATEST.pt')
    last=state.get('last_window')
    if last:
        p=OUT/'history'/f"E{last['epoch']}_S{last['step']:07d}.json"
        if p.exists():assert read(p)==last
        else:save_once(p,last)

def load_example(db,pool,key,training):
    cols='row_json,provenance_json'+(',labels_json' if training else '')
    values=db.execute('SELECT '+cols+' FROM examples WHERE key=?',(key,)).fetchone();assert values
    row=json.loads(values[0]);provenance=json.loads(values[1]);path,media=pool.media(provenance,materialize=True)
    assert media['sha256']==row['input']['video_sha256'];row['input']['video_path']=str(path)
    return row,json.loads(values[2]) if training else None

def observed_frames(row,epoch,occurrence,seed,training):
    from scripts.ptd_spatial_adapter_ab_v1 import frames_for
    frames,ids=frames_for(row,'clean');aug=dict(clean=True,brightness=1.,contrast=1.)
    if training:
        rs=int(hashlib.sha256(f'{seed}|{epoch}|{occurrence}|{row["key"]}'.encode()).hexdigest()[:16],16);rng=random.Random(rs)
        if rng.random()<.5:
            b,c=rng.uniform(.95,1.05),rng.uniform(.95,1.05);x=frames.astype(np.float32);mean=x.mean((1,2),keepdims=True)
            frames=np.rint((mean+(x-mean)*c)*b).clip(0,255).astype(np.uint8);aug=dict(clean=False,brightness=b,contrast=c)
    return frames,ids,aug

def prediction(res,row,pre,adapter_hash):
    sp=res.get('spatial') or {};inj=res['event_injection']['fields']
    return dict(key=row['key'],source=row['source'],domain=row['domain'],frame_ids=row['input']['frame_ids'],
        video_sha256=row['input']['video_sha256'],positions=sp.get('positions',[]),boxes_cxcywh=sp.get('boxes',torch.empty(0,4)).cpu(),
        geometry_valid=sp.get('geometry_valid',torch.empty(0,dtype=torch.bool)).cpu(),interval=res['interval'],format_ok=res['format_ok'],
        event_completion=res['event']['completion'],spatial_completion=sp.get('completion'),event_logits=inj['event_logits'].detach().float().cpu(),
        preprocess=pre,adapter_sha=adapter_hash,validation_GT_used=False)

def run(allocation,phase_seconds=3600):
    cfg=read(OUT/'CONFIG.json');receipt=BASE/'receipts'/f'full_source_{allocation}.json'
    assert 120<=phase_seconds<=cfg['phase_seconds']
    assert not receipt.exists() and not (OUT/'starts'/f'{allocation}.json').exists()
    for p,h in read(OUT/'LOCK.json')['pins'].items():assert sha(p)==h,p
    lease=(ROOT/'artifacts/spatial_tta_research_v2/gpu.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    start=time.monotonic();status='failed';state=None;prior=sum(scan_nested_gpu_receipts(ROOT/'artifacts'/n)[0] for n in ['desta3d_v1','desta3d_v2','desta3d_v3'])
    try:
        save_once(OUT/'starts'/f'{allocation}.json',dict(pid=os.getpid(),time=time.time(),prior_seconds=prior,phase_seconds=phase_seconds))
        assert shutil.disk_usage(ROOT).free>cfg['reserve_bytes']+500*2**20
        torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True
        from vg_tta.desta3d_v2 import Desta3DAdapterV2
        from vg_tta.desta3d_v2_training import make_source_optimizer,set_source_lr,source_evidence_losses
        from vg_tta.desta3d_v2_ptd import capture_stock_fields,branch_injection
        from vg_tta.desta3d_v2_shared_reference_cached import decode_shared_reference_two_pass
        from scripts.ptd_spatial_adapter_ab_v1 import processor_load,model_load,inputs_for
        pr=processor_load();model=model_load().eval().requires_grad_(False)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
        adapter=Desta3DAdapterV2(hidden_dim=128,architecture='dual3d',p1_enabled=False).cuda()
        optimizer=make_source_optimizer(adapter,'repaired','B')
        latest=OUT/'LATEST.pt'
        if latest.exists():
            entry=OUT/'resume_points'/f'{allocation}.pt';entry.parent.mkdir(parents=True,exist_ok=True);os.link(latest,entry)
            payload=torch.load(entry,map_location='cpu',weights_only=False);assert payload['lock_sha']==sha(OUT/'LOCK.json')
            adapter.load_state_dict(payload['adapter']);restore_optimizer(optimizer,payload['optimizer'])
            torch.set_rng_state(payload['cpu_rng']);torch.cuda.set_rng_state_all(payload['cuda_rng'])
            state={k:v for k,v in payload.items() if k not in ['adapter','optimizer','cpu_rng','cuda_rng','lock_sha','schema']}
            save_once(OUT/'restore_entries'/f'{allocation}.json',dict(entry_sha=sha(entry),adapter_sha=adapter_sha256(adapter),
                live_actual_counters=optimizer_counters(adapter,optimizer),exact_state_match=True,epoch=state['epoch'],cursor=state['cursor']))
            # Complete the immutable history record if a previous process died
            # after atomic LATEST replacement but before its JSON sidecar.
            save_state(state,adapter,optimizer)
        else:
            state=dict(stage='fit',epoch=0,cursor=0,step=0,val_cursor=0,last_window=None)
            save_state(state,adapter,optimizer);os.link(latest,OUT/'INITIAL.pt')
            save_once(OUT/'INITIAL.json',dict(adapter_sha=adapter_sha256(adapter),active_parameters=adapter.active_parameter_count(),fresh_optimizer=True,seed=cfg['seed']))
        db=sqlite3.connect(f'file:{ROSTER/"SOURCE.sqlite"}?mode=ro',uri=True);pool=SourcePool(load_annotations=False)
        rows=read(ROSTER/'INPUTS.json');train=[r for r in rows if r['split']=='train'];val=[r for r in rows if r['split']=='validation']
        if state['stage']=='await_score':
            decision=read(OUT/'scores'/f"E{state['epoch']}"/'DECISION.json')
            if decision['stop']:state['stage']='complete';save_state(state,adapter,optimizer)
            else:
                state.update(stage='fit',epoch=state['epoch']+1,cursor=0,val_cursor=0,last_window=None);save_state(state,adapter,optimizer)
        order=balanced_epoch(train,state['epoch'],cfg['seed'])
        while state['stage'] not in ['await_score','complete']:
            # Leave at a fully committed accumulation-window/prediction pair.
            if time.monotonic()-start>phase_seconds-90 or shutil.disk_usage(ROOT).free<cfg['reserve_bytes']+200*2**20:
                status='safe_pause';break
            if state['stage']=='fit':
                assert len(order)==cfg['source_counts']['epoch_queries']
                if state['cursor']==len(order):
                    p=OUT/'epoch_states'/f"E{state['epoch']}.pt";p.parent.mkdir(parents=True,exist_ok=True)
                    if p.exists():
                        prior_epoch=torch.load(p,map_location='cpu',weights_only=False)
                        assert prior_epoch['step']==state['step'] and prior_epoch['cursor']==state['cursor']
                        assert all(torch.equal(prior_epoch['adapter'][n],v.detach().cpu()) for n,v in adapter.state_dict().items())
                        del prior_epoch
                    else:os.link(latest,p)
                    state.update(stage='validation',val_cursor=0);save_state(state,adapter,optimizer);continue
                stop=min(state['cursor']+cfg['accum'],len(order));divisor=stop-state['cursor'];optimizer.zero_grad(set_to_none=True)
                lrscale=set_source_lr(optimizer,state['step'],cfg['total_horizon'],'repaired');details=[]
                for occurrence in range(state['cursor'],stop):
                    row,rec=load_example(db,pool,train[order[occurrence]]['key'],True)
                    frames,ids,aug=observed_frames(row,state['epoch'],occurrence,cfg['seed'],True)
                    prompt,pre=inputs_for(row,pr,frames);model.eval();adapter.train()
                    fields=capture_stock_fields(model,pr,prompt,row['input']['caption'],ids,row['input']['fps'])
                    data=training_inputs(pr,prompt,rec);item=dict(key=row['key'],domain=row['domain'],parent=row['source'],preprocess=pre,augmentation=aug,ce_available=data is not None)
                    if data is None:
                        output=adapter(fields['visual_grid'],fields['query_tokens'],fields['query_mask'],frame_times=fields['frame_times'])
                        aux=source_evidence_losses(output,rec);loss=.1*(aux['ref']+aux['event'])/divisor;loss.backward()
                        item['evidence']={k:float(v.detach()) for k,v in aux.items()};del loss,aux,output
                    else:
                        assert torch.equal(data['pixel_values_videos'],prompt['pixel_values_videos']);support=structured_support(data,pr.tokenizer)
                        model.train();model.model.visual.eval()
                        for branch in ['event','spatial']:
                            with branch_injection(model,adapter,data,fields,branch) as cap:
                                task,terms=source_branch_objective(model,data,support,branch,.05)
                                aux=source_evidence_losses(cap['fields'],rec)['event' if branch=='event' else 'ref']
                                loss=(task+.1*aux)/divisor;assert torch.isfinite(loss);loss.backward()
                                item[branch]=dict(structured=float(terms['structured'].detach()),original=float(terms['original'].detach()),aux=float(aux.detach()),
                                    structured_targets=terms['structured_targets'],actual_T=support['T'],actual_boxes=support['boxes'],injection_norm=cap['relative_injection_norm'],cast_changed=cap['changed_elements'])
                            del task,terms,aux,loss,cap
                    assert all(p.grad is None for p in model.parameters());details.append(item)
                    del data,prompt,fields,frames;gc.collect()
                checked=checked_step(adapter,optimizer);state['step']+=1;state['cursor']=stop
                state['last_window']=dict(epoch=state['epoch'],cursor=stop,step=state['step'],divisor=divisor,lr_scale=lrscale,
                    learning_rates={g['name']:g['lr'] for g in optimizer.param_groups},queries=details,**checked)
                save_state(state,adapter,optimizer);print('WINDOW',state['epoch'],state['cursor'],state['step'],checked['norm'],flush=True)
            else:
                epoch=state['epoch'];ah=adapter_sha256(adapter);adapter.eval();model.eval()
                if state['val_cursor']==len(val):
                    files=list((OUT/'predictions'/f'E{epoch}').glob('*.pt'))
                    frozen=list((OUT/'predictions'/'Frozen').glob('*.pt'));assert len(files)==len(frozen)==len(val)
                    seal=dict(epoch=epoch,queries=len(val),adapter_sha=ah,step=state['step'],lock_sha=sha(OUT/'LOCK.json'),
                        pins={str(p):sha(p) for p in files+frozen},validation_GT_used=False)
                    sealpath=OUT/'seals'/f'E{epoch}.json'
                    if sealpath.exists():assert read(sealpath)==seal
                    else:save_once(sealpath,seal)
                    state['stage']='await_score';save_state(state,adapter,optimizer);break
                i=state['val_cursor'];row,_=load_example(db,pool,val[i]['key'],False)
                frames,ids,_=observed_frames(row,epoch,i,cfg['seed'],False);prompt,pre=inputs_for(row,pr,frames)
                fields=capture_stock_fields(model,pr,prompt,row['input']['caption'],ids,row['input']['fps'])
                for arm,override in [('Frozen',0.),(f'E{epoch}',None)]:
                    p=OUT/'predictions'/arm/f'{i:05d}.pt';p.parent.mkdir(parents=True,exist_ok=True)
                    if p.exists():
                        old=torch.load(p,map_location='cpu',weights_only=False);assert old['key']==row['key'] and old['preprocess']==pre and old['video_sha256']==row['input']['video_sha256']
                        if arm!='Frozen':assert old['adapter_sha']==ah
                    else:
                        res=decode_shared_reference_two_pass(model,pr,prompt,fields,adapter,gate_override=override)
                        pred=prediction(res,row,pre,'frozen_gate0' if override==0 else ah)
                        temp=p.with_suffix('.tmp');torch.save(pred,temp);os.replace(temp,p);del pred,res
                state['val_cursor']+=1;save_state(state,adapter,optimizer)
                del frames,prompt,fields;gc.collect();torch.cuda.empty_cache()
        if state['stage']=='await_score':status='predictions_sealed_await_source_score'
        elif state['stage']=='complete':status='completed'
        save_once(OUT/'stops'/f'{allocation}.json',dict(time=time.time(),status=status,epoch=state['epoch'],cursor=state['cursor'],step=state['step'],stage=state['stage'],
           latest_sha=sha(latest),adapter_sha=adapter_sha256(adapter),actual_counters=optimizer_counters(adapter,optimizer),free_bytes=shutil.disk_usage(ROOT).free))
    except BaseException:
        save_once(OUT/'failures'/f'{allocation}.json',dict(error=traceback.format_exc(),in_memory_progress_may_be_uncommitted=state,
             committed_checkpoint_sha=sha(OUT/'LATEST.pt') if (OUT/'LATEST.pt').exists() else None));raise
    finally:
        seconds=time.monotonic()-start;save_once(receipt,dict(status=status,seconds=seconds,prior_seconds=prior,cumulative_seconds=prior+seconds,cap=None));lease.close()

def launch(allocation,phase_seconds=3600):
    start=time.monotonic();p=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'run','--allocation',allocation,'--phase-seconds',str(phase_seconds)],cwd=ROOT)
    wall=time.monotonic()-start;r=BASE/'receipts'/f'full_source_{allocation}.json';worker=read(r)['seconds'] if r.exists() else 0.
    save_once(BASE/'receipts'/f'full_source_wrapper_{allocation}.json',dict(status='completed' if p.returncode==0 else 'failed',seconds=max(0.,wall-worker),worker_seconds=worker,child_wall_seconds=wall,cap=None));raise SystemExit(p.returncode)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['register','run','launch']);p.add_argument('--allocation');p.add_argument('--phase-seconds',type=int,default=3600);a=p.parse_args()
    if a.action=='register':register()
    else:
        assert a.allocation and a.allocation.replace('_','').isalnum();globals()[a.action](a.allocation,a.phase_seconds)
