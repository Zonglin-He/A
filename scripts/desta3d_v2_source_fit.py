"""Locked four-arm source sanity/recipe comparison. Never reads target labels.

Repaired arms: one evidence epoch followed by five integration epochs, unless
all arms meet the registered source-only patience rule. Current-recipe dual:
six integration epochs. Exposure, pixel input and query order are paired.
"""
from __future__ import annotations
import argparse, copy, fcntl, gc, json, math, os, random, shutil, sys, time, traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from scripts.desta3d_v2_p0 import adapter_sha256,sha,read,write
from scripts.desta3d_tta_run_v1 import cpu_copy,scan_nested_gpu_receipts,tensor_sha256
from vg_tta.desta3d_v2 import Desta3DAdapterV2
from vg_tta.desta3d_v2_training import source_evidence_losses,make_source_optimizer,set_source_lr
PARENT=ROOT/'artifacts/desta3d_v2';OUT=PARENT/'source_fit'
V1=ROOT/'artifacts/desta3d_v1'
ARMS={'early_repaired':('early_factorized','repaired'),'shared_repaired':('shared3d','repaired'),
      'dual_repaired':('dual3d','repaired'),'dual_current':('dual3d','current')}
EPOCHS=6;ACCUM=4;SEED=20260927

def register():
    assert read(PARENT/'p0/p0_real_v3/COMPLETE.json')['real_gradients_pass']
    assert not OUT.exists(),'registered configuration is immutable'
    rows=read(V1/'SOURCE_INPUTS.json');train=[r for r in rows if r['split']=='train'];val=[r for r in rows if r['split']=='validation']
    assert len(train)==618 and len(val)==198 and len({r['source'] for r in train})==95 and len({r['source'] for r in val})==31
    assert not {r['source'] for r in train}&{r['source'] for r in val}
    assert not {r['input']['video_sha256'] for r in train}&{r['input']['video_sha256'] for r in val}
    target_metadata=read(V1/'tta_run/target64_v1/INPUTS.json')
    assert not {r['source'] for r in rows}&{r['source'] for r in target_metadata}
    assert not {r['input']['video_sha256'] for r in rows}&{r['input']['video_sha256'] for r in target_metadata}
    for r in rows:assert r['input']['kind']=='vidstg'
    config={'seed':SEED,'arms':ARMS,'hidden_dim':128,'p1_enabled':False,
      'train_queries':618,'train_parents':95,'validation_queries':198,'validation_parents':31,
      'target64_metadata_overlap':{'parents':0,'video_sha256':0,'target_labels_read':False,'scope':'only existing target64 manifest; not universal unseen-source claim'},
      'cohort':'existing isolated VidSTG source-only; no HC mixed-source claim; future HC is cross-dataset unless separately re-prepared',
      'epochs_max':EPOCHS,'stages':'repaired A=epoch0 evidence; B=epochs1..5 integration; current B throughout0..5',
      'source_patience':'after at least 2 B validation epochs, stop only if every arm has no strict parent-vIoU improvement for 2 B epochs',
      'state_selection':'best source-validation parent-macro vIoU among B epochs only; ties keep earlier; all epoch predictions retained',
      'current_recipe':{'lr':1e-4,'accumulation':1,'warmup':0.,'scheduler':'constant'},
      'repaired_recipe':{'reader_lr':3e-5,'head_out_gate_lr':1e-4,'accumulation':4,'warmup':.05,'scheduler':'cosine per stage; fresh optimizer on A to B only'},
      'optimizer':'AdamW','weight_decay':0.,'clip_norm':1.,
      'loss_A':'L_ref + L_frame_event; source labels only; missing boxes masked only for referent',
      'loss_B':'L_reference_time_NTP_MTP + L_box_NTP_MTP + .1 L_ref + .1 L_frame_event; structural separator CE omitted explicitly; joint=0; no identity loss',
      'source_teacher_forcing':'official GT reference/time prefix for spatial loss; free validation uses event-predicted I* and separate spatial semantic reference',
      'current_control_scope':'same v2 nonzero gated architecture and split objectives, old optimizer recipe; not old v1 architecture',
      'backbone':'frozen official PTD4B; no backbone optimizer parameters',
      'initialization':'common semantic modules copied from dual_repaired to architecture controls where names/shapes agree; dual_current exact entire dual_repaired state',
      'target_GT_read':False,'validation_GT':'read only after all 4x198 current-epoch prediction seals; source development selection',
      'TTA_steps':0,'source_gate':'dual_repaired selected B parent vIoU >= exact matched Frozen before later target TTA; CI and negative tails separately reported',
      'stage_review_seconds':3600,'cumulative_cap_seconds':None,'minimum_free_bytes':8*2**30,
      'checkpoint':'all-arm complete accumulation-window boundary; states/optimizer/RNG/cursor atomically saved; failure replay counted; validation predictions write-once and resume verified'}
    write(OUT/'CONFIG.json',config);write(OUT/'INPUTS.json',rows)
    paths=[Path(__file__),ROOT/'vg_tta/desta3d_v2.py',ROOT/'vg_tta/desta3d_v2_ptd.py',ROOT/'vg_tta/desta3d_v2_source.py',ROOT/'vg_tta/desta3d_v2_training.py',
      ROOT/'scripts/desta3d_v2_p0.py',ROOT/'scripts/desta3d_source_fit_v1.py',ROOT/'scripts/ptd_spatial_adapter_ab_v1.py',
      ROOT/'scripts/ptd_8b_teacher_feasibility_v1.py',ROOT/'vg_tta/metrics.py',
      ROOT/'vg_tta/exact_frame_decode_audit_v2.py',ROOT/'scripts/corruption_route_retest_v1.py',ROOT/'scripts/c1_controlled_corruption_v1.py',
      ROOT/'external/ParallelTubeDecoding/src/dataset/data_utils.py',ROOT/'external/ParallelTubeDecoding/src/dataset/sft_dataset.py',
      ROOT/'external/ParallelTubeDecoding/src/train/monkey_patch_forward.py',ROOT/'external/ParallelTubeDecoding/src/model/ptd_generation.py',
      ROOT/'methods/CURRENT_METHOD.json',V1/'SOURCE_INPUTS.json',V1/'source_fit/SOURCE_TRAIN_RECORDS.json',V1/'SOURCE_LABELS_TRAINING_ONLY.json',
      V1/'tta_run/target64_v1/INPUTS.json',PARENT/'SOURCE_MASK_CPU_CHECK.json',PARENT/'SOURCE_INVENTORY.json',
      V1/'source_fit/FROZEN_SOURCE_VAL_BASELINE.json',V1/'SOURCE_FULL_FEATURE_BARRIER.json',
      ROOT/'checkpoints/ParallelTubeDecoding-Qwen3-VL-4B/OFFICIAL_RECEIPT.json',ROOT/'checkpoints/ParallelTubeDecoding-Qwen3-VL-4B/model.safetensors',OUT/'CONFIG.json',OUT/'INPUTS.json']
    write(OUT/'LOCK.json',{'time':time.time(),'pins':{str(p):sha(p) for p in paths},
        'weights_sha':read(PARENT/'p0/p0_real_v3/LOCK.json')['pins'][str(ROOT/'checkpoints/ParallelTubeDecoding-Qwen3-VL-4B/model.safetensors')]})
    print('REGISTERED',str(OUT),flush=True)

def stage(epoch,recipe):return 'A' if epoch==0 and recipe=='repaired' else 'B'

def save_state(state,adapters,optimizers):
    payload={**state,'adapters':{a:cpu_copy(m.state_dict()) for a,m in adapters.items()},
       'optimizers':{a:cpu_copy(o.state_dict()) for a,o in optimizers.items()},
       'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
       'lock_sha':sha(OUT/'LOCK.json')}
    temp=OUT/'LATEST.tmp.pt';torch.save(payload,temp);os.replace(temp,OUT/'LATEST.pt')

def commit_window_history(state):
    """Recover the last committed window if interrupted between save and log."""
    payload=state.get('last_window_commit')
    if payload is None:return
    path=OUT/'history_windows'/f"E{payload['epoch']}_C{payload['cursor']:04}.json"
    if path.exists():assert read(path)==payload,'history differs from committed checkpoint'
    else:write(path,payload)

def norm_step(adapter,optimizer):
    params=[p for p in adapter.parameters() if p.requires_grad]
    norm=float(torch.nn.utils.clip_grad_norm_(params,1.))
    if not math.isfinite(norm):raise RuntimeError('nonfinite source gradient')
    optimizer.step();optimizer.zero_grad(set_to_none=True)
    return {'norm':norm,'clip_triggered':norm>1.}

def train_query(model,pr,adapter,row,record,fields,data,recipe,epoch,divisor):
    from vg_tta.desta3d_v2_ptd import branch_injection
    from vg_tta.desta3d_v2_source import split_source_loss_masks
    from scripts.ptd_8b_teacher_feasibility_v1 import joint_loss
    adapter.train();details={}
    if stage(epoch,recipe)=='A' or data is None:
        outputs=adapter(fields['visual_grid'],fields['query_tokens'],query_mask=fields['query_mask'],frame_times=fields['frame_times'])
        aux=source_evidence_losses(outputs,record)
        factor=1. if stage(epoch,recipe)=='A' else .1
        loss=factor*(aux['ref']+aux['event'])/divisor
        assert torch.isfinite(loss);loss.backward()
        details={'ref':float(aux['ref'].detach()),'event':float(aux['event'].detach()),'ce_available':False,'stage':stage(epoch,recipe)}
        del loss,aux,outputs
    else:
        masks=split_source_loss_masks(data,pr.tokenizer)
        model.train();model.model.visual.eval()
        for branch,auxkey in [('event','event'),('spatial','ref')]:
            d=dict(data);d['labels']=data['labels'].clone();d['labels'][~masks[branch].to(d['labels'].device)]=-100
            with branch_injection(model,adapter,data,fields,branch) as capture:
                ce,ce_stats=joint_loss(model,d)
                aux=source_evidence_losses(capture['fields'],record)[auxkey]
                loss=(ce+.1*aux)/divisor
                assert torch.isfinite(loss);loss.backward()
                details[branch]={'ce':float(ce.detach()),'aux':float(aux.detach()),'tokens':int(masks[branch].sum()),**ce_stats}
            del ce,aux,loss,capture,d
        details['ce_available']=True;details['stage']='B'
    assert all(p.grad is None for p in model.parameters())
    return details

def prediction_record(result,row,preprocess,adapter_hash):
    sp=result.get('spatial');sp={} if sp is None else sp
    evidence=result['event_injection']['fields']
    return {'key':row['key'],'source':row['source'],'frame_ids':row['input']['frame_ids'],
      'positions':sp.get('positions',[]),'boxes_cxcywh':sp.get('boxes',torch.empty(0,4)).cpu(),
      'geometry_valid':sp.get('geometry_valid',torch.empty(0,dtype=torch.bool)).cpu(),
      'interval':result['interval'],'format_ok':result['format_ok'],
      'event_completion':result['event']['completion'],'spatial_completion':sp.get('completion'),
      'event_logits':evidence['event_logits'].detach().float().cpu(),
      'preprocess':preprocess,'adapter_sha':adapter_hash,'GT_read':False,
      'video_sha256':row['input']['video_sha256']}

def verify_prediction_identity(saved,row,adapter_hash):
    assert saved['key']==row['key'] and saved['adapter_sha']==adapter_hash
    assert saved['source']==row['source'] and saved['frame_ids']==row['input']['frame_ids']
    assert saved['video_sha256']==row['input']['video_sha256']
    assert len(saved['preprocess']['pixel_sha'])==64

def score_epoch(epoch,rows,hashes,state):
    from scripts.desta3d_source_fit_v1 import _score_predictions
    paths={a:[OUT/'predictions'/f'E{epoch}_{a}'/f'{i:03}.pt' for i in range(len(rows))] for a in ARMS}
    pins={str(p):sha(p) for paths_a in paths.values() for p in paths_a}
    seal=OUT/f'E{epoch}_ALL_PREDICTIONS_SEAL.json'
    if not seal.exists():write(seal,{'pins':pins,'adapter_hashes':hashes,'query_count':len(rows),'arms':list(ARMS),'validation_GT_read':False})
    assert read(seal)['pins']==pins and read(seal)['adapter_hashes']==hashes
    scorepath=OUT/f'E{epoch}_SOURCE_METRICS.json'
    if scorepath.exists():return read(scorepath)
    labels=read(V1/'SOURCE_LABELS_TRAINING_ONLY.json')
    frozen=read(V1/'source_fit/FROZEN_SOURCE_VAL_BASELINE.json')['summary']
    assert frozen['queries']==len(rows) and frozen['parents']==len({r['source'] for r in rows})
    assert {r['key'] for r in frozen['rows']}=={r['key'] for r in rows}
    scores={};tails={}
    for a in ARMS:
        predictions={r['key']:torch.load(p,map_location='cpu',weights_only=False) for r,p in zip(rows,paths[a])}
        scores[a]=_score_predictions(predictions,rows,labels)
        fm={r['source']:r for r in frozen['parent_rows']}
        diff=[{'source':r['source'],'delta_v_pp':100*(r['vIoU']-fm[r['source']]['vIoU'])} for r in scores[a]['parent_rows']]
        tails[a]={'severe_gt5pp':sum(r['delta_v_pp']< -5 for r in diff),'worst':min(diff,key=lambda r:r['delta_v_pp']),'all_parent_deltas':diff}
    result={'epoch':epoch,'source_development_only':True,'target_GT_read':False,
      'prediction_seal_sha':sha(seal),'frozen':frozen,'arms':scores,'tails':tails,
      'source_gate_dual_current_epoch':epoch>=1 and scores['dual_repaired']['parent_macro']['vIoU']>=frozen['parent_macro']['vIoU']}
    write(scorepath,result);return result

def fit(allocation,phase_seconds):
    config=read(OUT/'CONFIG.json');lock=read(OUT/'LOCK.json')
    for p,h in lock['pins'].items():assert sha(p)==h,f'changed active dependency {p}'
    assert not (OUT/'COMPLETE.json').exists(),'source run already complete'
    allocation_path=OUT/'allocations'/f'{allocation}.json'
    write(allocation_path,{'time':time.time(),'phase_seconds':phase_seconds,'scope':'same max6-epoch locked source experiment; no scientific changes','cumulative_cap_seconds':None})
    lease=(ROOT/'artifacts/spatial_tta_research_v2/gpu.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    began=time.monotonic();status='running';failure=None;used=None
    def guard():
        if shutil.disk_usage(ROOT).free<config['minimum_free_bytes']:raise RuntimeError('source fit disk reserve below8GiB')
        return time.monotonic()-began>phase_seconds-30
    try:
        used=sum(scan_nested_gpu_receipts(p)[0] for p in [V1,PARENT])
        write(OUT/'starts'/f'{allocation}.json',{'pid':os.getpid(),'time':time.time(),'prior_seconds':used})
        if guard():raise RuntimeError('phase allocation too short')
        torch.set_num_threads(4);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED);torch.cuda.reset_peak_memory_stats()
        from scripts.ptd_spatial_adapter_ab_v1 import processor_load,model_load,frames_for,inputs_for
        from scripts.desta3d_source_fit_v1 import _training_inputs
        from vg_tta.desta3d_v2_ptd import capture_stock_fields,decode_two_pass
        pr=processor_load();model=model_load();model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        rows=read(OUT/'INPUTS.json');train=sorted([r for r in rows if r['split']=='train'],key=lambda r:r['key']);val=sorted([r for r in rows if r['split']=='validation'],key=lambda r:r['key'])
        records={r['key']:r for r in read(V1/'source_fit/SOURCE_TRAIN_RECORDS.json')}
        adapters={}
        for a,(arch,_) in ARMS.items():
            torch.manual_seed(SEED);adapters[a]=Desta3DAdapterV2(hidden_dim=128,architecture=arch,p1_enabled=False).cuda()
        common_prefixes=('input_proj.','shared_stem.','norm_stem.','query_pool_', 'film_', 'norm_spatial.', 'norm_event.',
                         'referent_head.','event_presence_head.','out_proj_','gate_','event_temporal_reader.')
        reference=adapters['dual_repaired'].state_dict()
        for a in ('early_repaired','shared_repaired'):
            weights=adapters[a].state_dict()
            for name,value in reference.items():
                if name.startswith(common_prefixes):
                    assert name in weights and weights[name].shape==value.shape
                    weights[name]=value.clone()
            adapters[a].load_state_dict(weights)
        # Exact paired initial state for the optimization contrast.
        adapters['dual_current'].load_state_dict(adapters['dual_repaired'].state_dict())
        state={'epoch':0,'cursor':0,'steps':{a:0 for a in ARMS},'stage_steps':{a:0 for a in ARMS},
          'best':{a:None for a in ARMS},'stale':{a:0 for a in ARMS},'optimizer_reset_on_B':False,'epoch_scored':[]}
        loaded=None
        if (OUT/'LATEST.pt').exists():
            loaded=torch.load(OUT/'LATEST.pt',map_location='cpu',weights_only=False);assert loaded['lock_sha']==sha(OUT/'LOCK.json')
            for a in ARMS:adapters[a].load_state_dict(loaded['adapters'][a])
            state={k:v for k,v in loaded.items() if k not in {'adapters','optimizers','cpu_rng','cuda_rng','lock_sha'}}
        optimizers={a:make_source_optimizer(adapters[a],recipe,stage(state['epoch'],recipe)) for a,(_,recipe) in ARMS.items()}
        if loaded is not None:
            for a in ARMS:optimizers[a].load_state_dict(loaded['optimizers'][a])
            torch.set_rng_state(loaded['cpu_rng']);torch.cuda.set_rng_state_all(loaded['cuda_rng']);del loaded
            commit_window_history(state)
        else:
            write(OUT/'INITIAL.json',{'adapters':{a:adapter_sha256(m) for a,m in adapters.items()},'parameter_counts':{a:m.parameter_count_by_group() for a,m in adapters.items()},
              'backbone_frozen':all(not p.requires_grad for p in model.parameters()),
              'runtime':{'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0)},
              'single_optimization_seed':True})
            save_state(state,adapters,optimizers)
        while state['epoch']<EPOCHS:
            epoch=state['epoch'];order=list(train);random.Random(SEED+epoch).shuffle(order)
            while state['cursor']<len(order):
                if guard():status='paused_safe_boundary';save_state(state,adapters,optimizers);return
                window=order[state['cursor']:state['cursor']+ACCUM];history=[]
                for a in ARMS:optimizers[a].zero_grad(set_to_none=True)
                for row in window:
                    model.eval();frames,ids=frames_for(row,'clean');prompt,prep=inputs_for(row,pr,frames)
                    fields=capture_stock_fields(model,pr,prompt,row['input']['caption'],ids,row['input']['fps'])
                    record=records[row['key']];data,_,_=_training_inputs(pr,model,row,record)
                    item={'epoch':epoch,'key':row['key'],'source':row['source'],'pixel_sha':prep['pixel_sha'],'arms':{}}
                    for a,(_,recipe) in ARMS.items():
                        divisor=len(window) if recipe=='repaired' else 1
                        losses=train_query(model,pr,adapters[a],row,record,fields,data,recipe,epoch,divisor)
                        item['arms'][a]={'losses':losses}
                        if recipe=='current':
                            item['arms'][a]['optimizer']=norm_step(adapters[a],optimizers[a]);state['steps'][a]+=1;state['stage_steps'][a]+=1
                    history.append(item)
                    del data,prompt,fields,frames;gc.collect()
                for a,(_,recipe) in ARMS.items():
                    if recipe=='repaired':
                        total=math.ceil(len(train)/ACCUM)*(1 if epoch==0 else EPOCHS-1)
                        lrscale=set_source_lr(optimizers[a],state['stage_steps'][a],total,recipe)
                        update=norm_step(adapters[a],optimizers[a]);state['steps'][a]+=1;state['stage_steps'][a]+=1
                        history[-1]['arms'][a]['optimizer']={**update,'lr_scale':lrscale,'window_queries':len(window)}
                state['cursor']+=len(window)
                state['last_window_commit']={'epoch':epoch,'cursor':state['cursor'],'allocation':allocation,'rows':history}
                save_state(state,adapters,optimizers)
                commit_window_history(state)
                # Convenience stream; authoritative per-window JSON is recoverable
                # from LATEST even if this secondary append is interrupted.
                with (OUT/'TRAIN_HISTORY.jsonl').open('a') as f:
                    for item in history:f.write(json.dumps({**item,'allocation':allocation,'committed_cursor':state['cursor']})+'\n')
                if state['cursor']%100==0 or state['cursor']==len(order):print('TRAIN',epoch,state['cursor'],state['steps'],flush=True)
            # All source-val predictions precede any label read in this epoch.
            hashes={a:adapter_sha256(m) for a,m in adapters.items()};model.eval()
            for i,row in enumerate(val):
                required={a:OUT/'predictions'/f'E{epoch}_{a}'/f'{i:03}.pt' for a in ARMS}
                if all(p.exists() for p in required.values()):
                    for a,p in required.items():
                        saved=torch.load(p,map_location='cpu',weights_only=False);verify_prediction_identity(saved,row,hashes[a])
                    continue
                if guard():status='paused_safe_validation';save_state(state,adapters,optimizers);return
                frames,ids=frames_for(row,'clean');prompt,prep=inputs_for(row,pr,frames)
                fields=capture_stock_fields(model,pr,prompt,row['input']['caption'],ids,row['input']['fps'])
                for a,path in required.items():
                    if path.exists():
                        saved=torch.load(path,map_location='cpu',weights_only=False);verify_prediction_identity(saved,row,hashes[a]);continue
                    adapters[a].eval();result=decode_two_pass(model,pr,prompt,adapters[a],fields)
                    path.parent.mkdir(parents=True,exist_ok=True)
                    temp=path.with_suffix('.tmp.pt');torch.save(prediction_record(result,row,prep,hashes[a]),temp);os.replace(temp,path)
                    del result
                del frames,prompt,fields;gc.collect()
            scores=score_epoch(epoch,val,hashes,state)
            if epoch>=1:
                for a in ARMS:
                    score=scores['arms'][a]['parent_macro']['vIoU']
                    if state['best'][a] is None or score>state['best'][a]['score']:
                        state['best'][a]={'epoch':epoch,'score':score,'adapter_sha':hashes[a]};state['stale'][a]=0
                        path=OUT/f'{a}_BEST.pt';tmp=path.with_suffix('.tmp.pt');torch.save({'adapter':cpu_copy(adapters[a].state_dict()),'selection':state['best'][a]},tmp);os.replace(tmp,path)
                    else:state['stale'][a]+=1
            state['epoch_scored'].append(epoch)
            print('SCORED',epoch,{a:scores['arms'][a]['parent_macro']['vIoU'] for a in ARMS},flush=True)
            if epoch>=2 and all(v>=2 for v in state['stale'].values()):
                state['early_stop_reason']='all arms two B-epochs source-vIoU patience';state['epoch']=EPOCHS
            else:state['epoch']+=1
            state['cursor']=0
            if state['epoch']==1:
                for a,(_,recipe) in ARMS.items():
                    if recipe=='repaired':optimizers[a]=make_source_optimizer(adapters[a],recipe,'B');state['stage_steps'][a]=0
                state['optimizer_reset_on_B']=True
            save_state(state,adapters,optimizers)
        final_scores=read(OUT/f"E{state['epoch_scored'][-1]}_SOURCE_METRICS.json")
        write(OUT/'COMPLETE.json',{'state':state,'status':'source_development_complete_requires_independent_readback',
          'best_dual_source_gate':state['best']['dual_repaired']['score']>=final_scores['frozen']['parent_macro']['vIoU'],
          'target_GT_read':False,'TTA_done':False})
        status='completed'
    except BaseException:
        status='failed';failure=traceback.format_exc();write(OUT/'failures'/f'{allocation}.json',{'traceback':failure,'time':time.time(),'replay_policy':'restore last all-arm complete window; all elapsed counted'});raise
    finally:
        elapsed=time.monotonic()-began
        write(PARENT/'receipts'/f'source_{allocation}.json',{'stage':'v2_source_fit','status':status,'seconds':elapsed,'failure':failure,
          'prior_seconds':used,'cumulative_seconds':used+elapsed if used is not None else None,'cumulative_cap_seconds':None,
          'peak_bytes':torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else None})
        fcntl.flock(lease,fcntl.LOCK_UN);lease.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['register','fit']);p.add_argument('--allocation');p.add_argument('--phase-seconds',type=int,default=3600)
    args=p.parse_args()
    if args.action=='register':register()
    else:
        assert args.allocation and args.phase_seconds>=120
        fit(args.allocation,args.phase_seconds)
