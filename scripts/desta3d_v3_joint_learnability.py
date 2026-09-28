"""Locked source mixer preparation, native interface probe, fit, and inference.

Only C_theta is optimized. This is GT-privileged source research, not TTA.
"""
import argparse,gc,hashlib,json,math,os,random,sqlite3,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from vg_tta.desta3d_v3_oracle_io import OUT,allocation,read,write,sha,check_pins,local_dependencies,tensor_sha
from scripts.desta3d_v3_privileged_ptd_qualification import B1,B1_SHA,ADAPTER_SHA,equal
D=OUT/'joint_learnability_v1'
V1=ROOT/'artifacts/desta3d_v1'
ROSTER=ROOT/'artifacts/desta3d_v3/full_source_roster_v1'
AUDIT=OUT/'joint_components_v1'
PROTOCOL=ROOT/'protocols/desta3d_v3_joint_learnability_v1.md'

def load(p):
    import torch
    return torch.load(p,map_location='cpu',weights_only=False)

def prepare():
    import torch,unittest,shutil
    from tests.test_desta3d_v3_joint_mixer import Contracts
    assert not D.exists()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts));assert result.wasSuccessful()
    rows=read(V1/'SOURCE_INPUTS.json');train=[r for r in rows if r['split']=='train']
    target=read(V1/'tta_run/target64_v1/INPUTS.json')
    panel=read(ROOT/'artifacts/desta3d_v2/tta_v2/source_task_control_v2/INPUTS.json')
    excluded=rows+target+panel
    parents={r['source'] for r in excluded};media={r['input']['video_sha256'] for r in excluded}
    db=sqlite3.connect(f'file:{ROSTER/"SOURCE.sqlite"}?mode=ro',uri=True)
    candidates=[]
    for key,parent,rr,pp in db.execute("select key,parent,row_json,provenance_json from examples where split='validation' and domain='Vid'"):
        r=json.loads(rr)
        if parent not in parents and r['input']['video_sha256'] not in media:candidates.append((r,json.loads(pp)))
    selected=sorted({r['source'] for r,p in candidates},key=lambda s:hashlib.sha256(('joint-learnability-v1|'+s).encode()).hexdigest())[:31]
    val=[r for r,p in candidates if r['source'] in selected];val.sort(key=lambda r:(r['source'],r['key']))
    provenance={r['key']:p for r,p in candidates if r['source'] in selected}
    assert len(train)==618 and len({r['source'] for r in train})==95 and len(selected)==31
    assert all(Path(r['input']['video_path']).is_file() for r in train)
    assert not {r['source'] for r in train}&set(selected) and sha(B1)==B1_SHA
    basis=load(AUDIT/'AUDIT_BASIS.pt')['U'].float();assert basis.shape==(2560,256)
    assert shutil.disk_usage(ROOT).free>8*2**30+12*2**30
    cfg=dict(stage='GT_evidence_joint_correction_learnability',train_queries=618,train_parents=95,
        validation_queries=len(val),validation_parents=31,seeds=[20260928,20260929],epochs=1,accum=4,windows=155,
        lr=.001,weight_decay=0,clip=1,warmup_windows=8,phase_seconds=3600,minimum_free_bytes=8*2**30,
        radius=math.sqrt((.087687**2+.170316**2)/2),cap=None,prior_GPU_seconds=43107.875212573104,
        validation_label_use='GT privileged evidence; offline scores only after all seals',target_read=False,
        union_sha=sha(AUDIT/'AUDIT_BASIS.pt'),source_validation_scope='newly locked relative to enumerated prior manifests; annotations prepared earlier; not globally untouched claim')
    write(D/'CONFIG.json',cfg);write(D/'TRAIN_INPUTS.json',train);write(D/'VALIDATION_INPUTS.json',val);write(D/'VALIDATION_PROVENANCE.json',provenance)
    torch.save(basis,D/'BASIS.pt')
    write(D/'SPLIT_AUDIT.json',dict(train_queries=618,train_parents=95,confirmation_queries=len(val),confirmation_parents=31,
        selected_parents=selected,eligible_parents=len({r['source'] for r,p in candidates}),parent_overlap=0,media_overlap=0,
        exclusion_manifest_paths=[str(V1/'SOURCE_INPUTS.json'),str(V1/'tta_run/target64_v1/INPUTS.json'),str(ROOT/'artifacts/desta3d_v2/tta_v2/source_task_control_v2/INPUTS.json')],
        selection='SHA256 fixed salt parent order; every query; no labels/metrics SELECT during selection',
        unproven='global historical and PTD-pretraining nonmembership',source_validation_GT_prepared_previously=True))
    write(D/'CPU_PREFLIGHT.json',dict(status='passed',tests=result.testsRun,GPU=False))
    paths=local_dependencies([Path(__file__),PROTOCOL,ROOT/'vg_tta/desta3d_v3_joint_mixer.py',ROOT/'tests/test_desta3d_v3_joint_mixer.py',
        B1,D/'CONFIG.json',D/'TRAIN_INPUTS.json',D/'VALIDATION_INPUTS.json',D/'VALIDATION_PROVENANCE.json',D/'BASIS.pt',
        V1/'source_fit/SOURCE_TRAIN_RECORDS.json',ROSTER/'SOURCE.sqlite',AUDIT/'AUDIT_BASIS.pt',
        ROOT/'checkpoints/ParallelTubeDecoding-Qwen3-VL-4B/model.safetensors'])
    write(D/'LOCK.json',dict(pins={str(p):sha(p) for p in paths}));write(D/'REGISTRATION.json',dict(time=time.time(),status='registered_before_GPU',protocol_sha=sha(PROTOCOL)))
    print('REGISTERED',cfg,flush=True)

def setup(seed):
    import torch
    from scripts.ptd_spatial_adapter_ab_v1 import processor_load,model_load
    from vg_tta.desta3d_v2 import Desta3DAdapterV2
    from vg_tta.desta3d_v3_joint_mixer import JointCorrectionMixer
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    pr=processor_load();model=model_load().eval().requires_grad_(False)
    adapter=Desta3DAdapterV2(hidden_dim=128,architecture='dual3d',p1_enabled=False).cuda().eval()
    adapter.load_state_dict(load(B1)['adapter']);adapter.set_train_stage('frozen')
    # Independent mixer seed is reset after frozen model construction.
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    mixer=JointCorrectionMixer(load(D/'BASIS.pt'),radius=read(D/'CONFIG.json')['radius']).cuda()
    return pr,model,adapter,mixer

def example(pr,model,adapter,row,label):
    from scripts.ptd_spatial_adapter_ab_v1 import inputs_for
    from vg_tta.exact_frame_decode_audit_v2 import decode
    from vg_tta.desta3d_v2_ptd import capture_stock_fields
    from vg_tta.desta3d_v3_joint_mixer import frozen_context,source_evidence
    frames,ids=decode(row['input']);prompt,pre=inputs_for(row,pr,frames)
    fields=capture_stock_fields(model,pr,prompt,row['input']['caption'],ids,row['input']['fps'])
    f=fields['visual_grid'];context=frozen_context(adapter,fields)
    evidence,pos=source_evidence(label,ids,f.shape[2],f.shape[3])
    return prompt,pre,fields,(*context,evidence.to(f.device),f)

def infer(model,pr,adapter,mixer,prompt,fields,args,*,capture=False,base=False):
    import torch
    from vg_tta.desta3d_v3_decomposition import shared_fields
    from vg_tta.desta3d_v2_shared_reference_cached import decode_shared_reference_two_pass
    from vg_tta.desta3d_v3_actuation_full_vocab import capture_teacher
    with torch.no_grad():
        delta=torch.zeros_like(args[-1]) if base else mixer(*args)
        new=args[-1]+delta
        ratio=float(delta.double().norm()/args[-1].double().norm())
        assert ratio<=mixer.radius+2e-6
        with shared_fields(adapter,{'event':new,'spatial':new},['event','spatial'],allow_prefix=True) as calls:
            output=capture_teacher(model,pr,prompt,adapter,fields) if capture else decode_shared_reference_two_pass(model,pr,prompt,adapter,fields)
    return output,dict(delta_norm=float(delta.double().norm()),relative_norm=ratio,common_F_sha=tensor_sha(args[-1]),
        corrected_F_sha=tensor_sha(new),calls=calls,same_field_both_passes=True)

def backward_query(model,pr,adapter,mixer,prompt,fields,args,label,divisor,*,verify=False):
    import torch
    from vg_tta.desta3d_v3_joint_mixer import native_supervision
    from vg_tta.desta3d_v3_actuation_full_vocab import replay_branch
    from vg_tta.desta3d_v3_free_actuation import native_ce
    from vg_tta.desta3d_v3_decomposition import shared_fields
    from vg_tta.desta3d_v2_output_anchor_memory_v7 import release_free_host_arenas
    (native,trace),inj=infer(model,pr,adapter,mixer,prompt,fields,args,capture=True)
    # Save a small native record in the query audit; never persist full vocab
    # teacher matrices for every training occurrence.
    result=dict(injection=inj,native_format=native['format_ok'],interval=native['interval'],branches={})
    del native
    for i,b in enumerate(['event','spatial']):
        supervision,missing=native_supervision(label,trace,b)
        if missing:result['branches'][b]=dict(missing=missing);continue
        release_free_host_arenas();torch.cuda.empty_cache()
        target,valid=supervision;new=args[-1]+mixer(*args)
        with shared_fields(adapter,{b:new},[b]):
            logits,cache=replay_branch(model,prompt,adapter,fields,trace,b)
            kind='time' if b=='event' else 'coordinate'
            exact=torch.equal(logits.detach().cpu(),trace['branches'][i]['logits'][kind])
            if verify:assert exact,'current native/replay logits differ'
            ce=native_ce(logits,target.cuda(),valid.cuda());(ce/divisor).backward()
        result['branches'][b]=dict(CE=float(ce.detach()),actions=int(valid.sum()),classes=logits.shape[-1],native_replay_exact=exact,cache=cache)
        del ce,logits,new,cache;gc.collect();torch.cuda.empty_cache()
    assert all(p.grad is None for p in model.parameters()) and all(p.grad is None for p in adapter.parameters())
    assert mixer.basis.grad is None
    del trace;return result

def save_state(path,mixer,opt,state):
    import torch
    from vg_tta.optimizer_checkpoint import cpu_clone,validate_serialized_optimizer
    osd=cpu_clone(opt.state_dict());validate_serialized_optimizer(osd)
    payload=dict(**state,mixer=cpu_clone(mixer.state_dict()),optimizer=osd,lock_sha=sha(D/'LOCK.json'),
        cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp');torch.save(payload,temp);os.replace(temp,path)

def stage_register(name,mode,seed):
    check_pins(read(D/'LOCK.json')['pins']);d=OUT/name;assert not d.exists()
    cfg=read(D/'CONFIG.json');cfg.update(mode=mode,seed=seed,phase_seconds=900 if mode=='probe' else 3600)
    write(d/'CONFIG.json',cfg);write(d/'LOCK.json',dict(pins={str(D/'LOCK.json'):sha(D/'LOCK.json'),**read(D/'LOCK.json')['pins']}))
    write(d/'REGISTRATION.json',dict(time=time.time(),mode=mode,source_GT_privileged=True,no_target=True))
    return d

def probe(d,cfg,guard):
    import torch
    from scripts.desta3d_v2_p0 import adapter_sha256
    from vg_tta.optimizer_checkpoint import cpu_clone
    pr,model,adapter,mixer=setup(cfg['seed']);rows=sorted(read(D/'TRAIN_INPUTS.json'),key=lambda r:r['key'])
    row=rows[0];label={r['key']:r for r in read(V1/'source_fit/SOURCE_TRAIN_RECORDS.json')}[row['key']]
    prompt,pre,fields,args=example(pr,model,adapter,row,label)
    initial=cpu_clone(mixer.state_dict());opt=torch.optim.AdamW(mixer.parameters(),lr=cfg['lr'],weight_decay=0)
    assert torch.equal(mixer(*args),torch.zeros_like(args[-1]))
    normal,_=infer(model,pr,adapter,mixer,prompt,fields,args,base=True)
    zero,_=infer(model,pr,adapter,mixer,prompt,fields,args)
    from scripts.desta3d_v2_source_fit import prediction_record
    from scripts.desta3d_v2_reference_audit_cached_v3 import details
    pp=[prediction_record(x,row,pre,ADAPTER_SHA) for x in [normal,zero]]
    assert equal(pp[0],pp[1]) and equal(details(normal),details(zero))
    del normal,zero
    rr=backward_query(model,pr,adapter,mixer,prompt,fields,args,label,1,verify=True)
    grads={n:p.grad.detach().cpu().clone() for n,p in mixer.named_parameters() if p.grad is not None}
    assert grads and all(torch.isfinite(g).all() for g in grads.values()) and sum(float(g.double().square().sum()) for g in grads.values())>0
    torch.save(dict(initial=initial,gradients=grads,query=row['key'],input={k:tensor_sha(v) for k,v in fields.items() if isinstance(v,torch.Tensor)}),d/'RAW.pt')
    norm=float(torch.nn.utils.clip_grad_norm_(mixer.parameters(),cfg['clip']));opt.step()
    assert any(not torch.equal(initial[n],x.cpu()) for n,x in mixer.state_dict().items())
    counters=[int(x['step']) for x in opt.state.values()];assert set(counters)=={1}
    mixer.load_state_dict(initial);opt.zero_grad(set_to_none=True)
    assert all(torch.equal(initial[n],x.cpu()) for n,x in mixer.state_dict().items()) and adapter_sha256(adapter)==ADAPTER_SHA
    write(d/'REPORT.json',dict(status='passed_real_native_interface',key=row['key'],branches=rr,gradient_norm=norm,
        trainable_parameters=sum(p.numel() for p in mixer.parameters()),zero_native_exact=True,frozen_scope=True,exact_reset=True,
        temporary_steps=1,optimizer_counters=counters,peak_GPU_bytes=torch.cuda.max_memory_allocated(),utility_not_measured=True))
    write(d/'COMPLETE.json',dict(status='probe_passed',report_sha=sha(d/'REPORT.json'),raw_sha=sha(d/'RAW.pt')))

def fit(d,cfg,guard):
    import torch
    from scripts.desta3d_v2_p0 import adapter_sha256
    from vg_tta.optimizer_checkpoint import restore_optimizer
    assert read(OUT/'joint_learnability_probe001/COMPLETE.json')['status']=='probe_passed'
    seed=cfg['seed'];pr,model,adapter,mixer=setup(seed);opt=torch.optim.AdamW(mixer.parameters(),lr=cfg['lr'],weight_decay=0)
    dest=D/f'seed{seed}';latest=dest/'LATEST.pt'
    rows=read(D/'TRAIN_INPUTS.json');order=list(range(len(rows)));random.Random(seed).shuffle(order)
    labels={r['key']:r for r in read(V1/'source_fit/SOURCE_TRAIN_RECORDS.json')}
    state=dict(cursor=0,windows=0,steps=0,seed=seed,empty_windows=0)
    if latest.exists():
        p=load(latest);assert p['lock_sha']==sha(D/'LOCK.json');mixer.load_state_dict(p['mixer']);restore_optimizer(opt,p['optimizer'])
        torch.set_rng_state(p['cpu_rng']);torch.cuda.set_rng_state_all(p['cuda_rng'])
        state={k:p[k] for k in state};write(d/'RESTORE.json',dict(cursor=state['cursor'],steps=state['steps'],checkpoint_sha=sha(latest),live_counters=[int(v['step']) for v in opt.state.values()]))
        if p.get('last_history'):
            hp=dest/'history'/f"{state['windows']:04}.json"
            if hp.exists():assert read(hp)==p['last_history']
            else:write(hp,p['last_history'])
    else:save_state(latest,mixer,opt,state);os.link(latest,dest/'INITIAL.pt')
    start=time.monotonic();status='safe_pause'
    while state['cursor']<len(order):
        guard()
        if time.monotonic()-start>cfg['phase_seconds']-180:break
        stop=min(state['cursor']+cfg['accum'],len(order));divisor=stop-state['cursor'];opt.zero_grad(set_to_none=True)
        window=state['windows']+1;lr=cfg['lr']*(window/8 if window<=8 else .5*(1+math.cos(math.pi*(window-8)/(155-8))))
        for g in opt.param_groups:g['lr']=lr
        details=[]
        for pos in range(state['cursor'],stop):
            row=rows[order[pos]];label=labels[row['key']];prompt,pre,fields,args=example(pr,model,adapter,row,label)
            result=backward_query(model,pr,adapter,mixer,prompt,fields,args,label,divisor,verify=(pos==0))
            details.append(dict(position=pos,key=row['key'],source=row['source'],physical_pixel=pre,**result))
            del prompt,pre,fields,args;gc.collect();torch.cuda.empty_cache()
        active=any(p.grad is not None for p in mixer.parameters())
        norm=float(torch.nn.utils.clip_grad_norm_(mixer.parameters(),cfg['clip'])) if active else 0.
        assert math.isfinite(norm)
        if active:opt.step();state['steps']+=1
        else:state['empty_windows']+=1
        state.update(cursor=stop,windows=window)
        assert adapter_sha256(adapter)==ADAPTER_SHA and all(p.grad is None for p in model.parameters())
        counters=[int(v['step']) for v in opt.state.values()];assert not counters or set(counters)=={state['steps']}
        hist=dict(**state,lr=lr,gradient_norm=norm,clip=norm>cfg['clip'],counters=counters,queries=details,
            frozen_adapter_sha=ADAPTER_SHA,union_sha=tensor_sha(mixer.basis))
        # Durable checkpoint contains the corresponding history to recover an
        # interruption between atomic state replacement and history publication.
        save_state(latest,mixer,opt,{**state,'last_history':hist})
        hp=dest/'history'/f'{window:04}.json'
        if hp.exists():assert read(hp)==hist
        else:write(hp,hist)
        print('JOINT_FIT',seed,stop,len(order),'steps',state['steps'],'norm',norm,flush=True)
    if state['cursor']==len(order):
        status='complete';os.link(latest,dest/'FINAL.pt');write(dest/'COMPLETE.json',dict(**state,final_sha=sha(dest/'FINAL.pt'),frozen_scope=True))
    write(d/'COMPLETE.json',dict(status=status,**state,latest_sha=sha(latest)))

def run(name,mode,seed):
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';d=stage_register(name,mode,seed)
    with allocation(d) as (cfg,guard):globals()[mode](d,cfg,guard)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','probe','fit']);p.add_argument('--name');p.add_argument('--seed',type=int,default=20260928)
    a=p.parse_args()
    if a.action=='prepare':prepare()
    else:assert a.name;run(a.name,a.action,a.seed)
