"""Finite, separately pinned common-state GPU control, qualification and resume."""
import collections,fcntl,gc,inspect,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,status,sha,load,save
BASE=ROOT/'artifacts/stvg_opd_optimization_geometry_v1'
OLD=ROOT/'artifacts/stvg_opd_paper_hc2_revision_v2'
ARM=['Full','Direct','Direct_capped']
PYTHON=ROOT/'.conda/tubedetr/bin/python'

def prepare():
    if (BASE/'GPU_RUNTIME.json').exists():return verify()
    import gzip
    with gzip.open(ROOT/'results/stvg_opd_mechanism_review/2026-10-11/CELL_ROWS.jsonl.gz','rt') as f:cells=[json.loads(s) for s in f]
    binds=read(ROOT/'artifacts/stvg_opd_mechanism_review_v1/PRIVATE_INPUT_BINDINGS.json')['records'];records=[];seen=collections.Counter()
    for i,c in enumerate(cells):
        b=binds[2*i];p=ROOT/b['path'];assert sha(p)==b['sha256'];z=load(p)
        group=(z['stage'],z['condition'],z['order']);seen[group]+=1
        records.append(dict(index=i,dataset=z['dataset'],source=z['source'],stage=z['stage'],condition=z['condition'],order=z['order'],arrival=z['arrival'],
          query_ordinal=z['query_ordinal'],source_id=c['source_id'],development=c['development'],setting=c['setting'],config=z['config'],
          original_Full=b,qualify=seen[group]<=2))
    assert len(records)==1792 and sum(r['qualify'] for r in records)==28
    write(BASE/'COMMON_STATE_COHORT.json',dict(status='fixed_entire_original_cohort_no_score_selection',records=records,cells=1792,qualification_cells=28,
      actual_qualification_GPU_fits=168,actual_formal_GPU_fits=5376,new_formal_predictions=3584,exact_Full_references=1792,GT_read=False,time=time.time()))
    pins={**read(OLD/'P4_joint_engineering_revision001/REVISION_RUNTIME.json')['pins']}
    files=['scripts/run_stvg_opd_common_state_geometry_v1.py','vg_tta/stvg_opd_common_state_geometry_v1.py',
      'protocols/stvg_opd_optimization_geometry_v1.md','protocols/stvg_opd_optimization_geometry_priority_pause001.md',
      'scripts/pause_stvg_opd_for_geometry_v1.py','scripts/run_stvg_opd_expanded_state_v1.py',
      'scripts/analyze_stvg_opd_optimization_geometry_readback002.py','methods/CURRENT_METHOD.json',
      'methods/decota_spatial_opd_v1/configs.json']
    pins.update({p:sha(ROOT/p) for p in files})
    write(BASE/'GPU_RUNTIME.json',dict(status='locked_before_actual_GPU',cohort_sha256=sha(BASE/'COMMON_STATE_COHORT.json'),pins=pins,
      priority_authorization_sha256=sha(BASE/'PRIORITY_AUTHORIZATION.json'),baseline_pause_sha256=sha(BASE/'BASELINE_PAUSE_RECEIPT.json'),
      allocator='expandable_segments:True',GT_read=False,time=time.time()))
    return verify()

def verify():
    r=read(BASE/'GPU_RUNTIME.json');assert sha(BASE/'COMMON_STATE_COHORT.json')==r['cohort_sha256']
    assert sha(BASE/'PRIORITY_AUTHORIZATION.json')==r['priority_authorization_sha256']
    assert sha(BASE/'BASELINE_PAUSE_RECEIPT.json')==r['baseline_pause_sha256']
    for p,h in r['pins'].items():assert sha(ROOT/p)==h,p
    return r

def priority_lease():
    from scripts.pause_stvg_opd_for_geometry_v1 import process
    pause=read(BASE/'BASELINE_PAUSE_RECEIPT.json');assert pause['status']=='paused_original_processes_preserved'
    for role in ['controller','worker']:
        p=process(pause[role]['pid']);assert p['start_ticks']==pause[role]['start_ticks'] and p['state']=='T'
    f=(BASE/'priority_GPU.lock').open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
    import torch,numpy as np
    from scripts.run_tastvg_evidence_vulnerability_v2 import install_clean_loader
    from scripts.decota_paper_common_v1 import guard
    install_clean_loader();sys.addaudithook(guard)
    torch.set_num_threads(4);torch.manual_seed(20260920);np.random.seed(20260920)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    write(BASE/'PRIORITY_GPU_LEASE_RECEIPT.json',dict(status='exclusive_active_diagnostic_with_authorized_original_workers_stopped',
      scope='separate reservation while original shared exclusion remains held',pid=os.getpid(),original_worker=pause['worker']['pid'],
      original_processes_preserved=True,GT_read=False,time=time.time()))
    return f

def capture(model,r,old,cache):
    import torch
    from scripts.run_decota_paper_main_v1 import read_row,frames_for,unpack_expert
    from scripts.run_spatial_ssl_gpu_v1 import frozen_forward
    from scripts.stvg_opd_paper_inputs_v1 import observation
    from scripts.stvg_opd_legacy_input_bridge003 import normalized
    from methods.decota_final_simplified_v1.objectives import prediction
    row=read_row(r['dataset'],r['query_ordinal']);frames,ids,reuse=frames_for(r['dataset'],row,cache)
    shifted,pixel,spec=observation(r['dataset'],row,r['condition'],frames)
    batch,records,base=frozen_forward(model,shifted,row)
    native=prediction(base.zero['logits'],base.zero['boxes'],records,ids)
    b=r['original_Full'];ip=ROOT/b['input_path'];assert sha(ip)==b['input_sha256'];inp=normalized(load(ip))
    assert inp['source_model_state_sha256']==model._fixed_full_state_hash and inp['pixel_sha256']==pixel
    assert inp['corruption_spec']==spec and inp['frame_ids']==ids and inp['indices']==native['indices']
    assert inp['interval']==old['interval']==native['physical_interval'] and torch.equal(inp['native_boxes'],native['boxes'].cpu())
    del batch,frames,shifted
    return base,row,unpack_expert(inp['expert'])

def fit_arm(base,row,expert,old,arm,qualify):
    import torch
    from vg_tta.stvg_opd_p3_scope_precision001 import Installation
    from vg_tta.stvg_opd_common_state_geometry_v1 import Geometry,mathematical_readback,bare
    from scripts.run_stvg_opd_p1_softmax_recovery_002 import equal
    from scripts.stvg_opd_design_ablation_common_v1 import audit
    reference=old['fit'];rec=Installation(qualification=qualify);hook=Geometry(rec,arm,reference)
    try:
        from vg_tta.stvg_opd_paper_ablations_v1 import fit_variant
        result=fit_variant(base,reference['initial'],expert,row['frame_ids'],row['key'],
            'on_policy' if arm=='Full' else 'direct_L1_GIoU',reference['config'])
        result['geometry']=dict(arm=arm,rounds=hook.rows,extra_native_forwards=hook.extra,
          common_initial_original_Full=True,GT_read=False)
        assert all(torch.equal(result['initial'][n],reference['initial'][n]) for n in reference['initial'])
        assert torch.equal(result['before'],reference['before']) and result['positions']==reference['positions']
        if arm=='Full':
            # Replay has probes, but the actual full scientific trajectory must
            # remain exactly the previously sealed original, not just the boxes.
            counts=dict(tensors=0,tensor_coordinates=0,scalar_values=0);equal(bare(result),bare(reference),counts)
            result['geometry']['original_Full_complete_bitwise']=counts
            audit_result=audit(result,expert)
        elif arm=='Direct':audit_result=audit(result,expert)
        else:audit_result=None
        independent=mathematical_readback(result)
        return result,dict(status='pass',original_component=audit_result,independent=independent,
                           actual_native_head_VJP=rec.last_vjp,GT_read=False)
    finally:hook.close();rec.uninstall();base.restore(base.initial)

def gpu_run(qualify):
    verify();assert os.environ.get('PYTORCH_CUDA_ALLOC_CONF')=='expandable_segments:True'
    from scripts.stvg_opd_design_ablation_common_v1 import activate_old
    activate_old();import torch
    from scripts.run_decota_paper_main_v1 import model_for
    from methods.decota_final_simplified_v1.tensors import state_hash
    from scripts.run_stvg_opd_p1_softmax_recovery_002 import equal
    from vg_tta.stvg_opd_common_state_geometry_v1 import bare
    from scripts.stvg_opd_paper_common_v1 import budget
    records=read(BASE/'COMMON_STATE_COHORT.json')['records']
    selected=[r for r in records if r['qualify']] if qualify else records
    dest=BASE/('qualification' if qualify else 'common_state');dest.mkdir(exist_ok=True)
    if not qualify:assert read(BASE/'ROOT_QUALIFICATION_READBACK.json')['status']=='pass'
    lease=priority_lease();model=None;stage=None;cache=collections.OrderedDict();files={};count=0;start=time.time()
    try:
        for r in selected:
            budget();b=r['original_Full'];p=ROOT/b['path'];assert sha(p)==b['sha256'];old=load(p)
            if stage!=r['stage']:
                if model is not None:
                    assert state_hash(model.state_dict())==mh;del model;gc.collect();torch.cuda.empty_cache()
                stage=r['stage'];model=model_for(r['source']);mh=state_hash(model.state_dict());cache.clear()
            paths={a:dest/f"{r['index']:05}"/(a+'.pt') for a in ARM}
            needs=any(not f.with_suffix('.json').exists() for f in paths.values())
            if needs:base,row,expert=capture(model,r,old,cache)
            for arm,path in paths.items():
                receipt=path.with_suffix('.json')
                if receipt.exists():
                    rc=read(receipt);assert sha(path)==rc['sha256'] and not rc['GT_read'];z=load(path)
                else:
                    torch.cuda.synchronize();tick=time.perf_counter()
                    fit,checked=fit_arm(base,row,expert,old,arm,qualify)
                    torch.cuda.synchronize();seconds=time.perf_counter()-tick
                    z=dict(**r,arm=arm,fit=fit,math_audit=checked,input_sha256=b['input_sha256'],interval=old['interval'],
                      common_initial_original_Full_sha256=b['sha256'],Adam_reset=True,GT_read=False,runtime_sha256=sha(BASE/'GPU_RUNTIME.json'),
                      compute=dict(actual_new_DINO_calls=0,fit_and_probe_seconds=seconds,extra_native_forwards=fit['geometry']['extra_native_forwards']))
                    # Preserve actual complete unaccepted evidence before any
                    # repeat or qualification/formal comparison.
                    save(path.parent/('UNACCEPTED_'+arm+'.pt'),z)
                    if qualify:
                        again,ac=fit_arm(base,row,expert,old,arm,qualify)
                        repeat=path.parent/('REPEAT_'+arm+'.pt');save(repeat,again)
                        compared=dict(tensors=0,tensor_coordinates=0,scalar_values=0)
                        equal(bare(fit),bare(again),compared);equal(fit['geometry'],again['geometry'],compared)
                        z['qualification']=dict(actual_complete_GPU_fits=2,complete_repeat_bitwise=compared,repeat_sha256=sha(repeat))
                    elif r['qualify']:
                        qf=BASE/'qualification'/f"{r['index']:05}"/(arm+'.pt');qz=load(qf)
                        compared=dict(tensors=0,tensor_coordinates=0,scalar_values=0)
                        equal(bare(fit),bare(qz['fit']),compared);equal(fit['geometry'],qz['fit']['geometry'],compared)
                        z['qualified_formal_bitwise']=compared
                    save(path,z);write(receipt,dict(sha256=sha(path),bytes=path.stat().st_size,GT_read=False,time=time.time()))
                files[str(path.relative_to(ROOT))]=sha(path);count+=1
                status(BASE/'GPU_STATUS.json',dict(status='running_qualification' if qualify else 'running_common_state_formal',
                  dataset=r['dataset'],stage=r['stage'],condition=r['condition'],order=r['order'],arrival=r['arrival'],arm=arm,
                  done=count,total=len(selected)*3,pid=os.getpid(),GT_read=False,time=time.time()))
                print('COMMON_STATE_PROGRESS',qualify,count,len(selected)*3,r['dataset'],arm,flush=True)
            if needs:del base,row,expert
            del old,z;gc.collect();torch.cuda.empty_cache()
        assert count==len(selected)*3 and state_hash(model.state_dict())==mh
        output=dest/'GPU_QUALIFICATION.json' if qualify else BASE/'GLOBAL_PREDICTION_BARRIER.json'
        write(output,dict(status='pass' if qualify else 'sealed',files=files,logical_fits=count,
          actual_complete_GPU_fits=count*(2 if qualify else 1),new_DINO_calls=0,GT_read=False,
          original_source_state_restored=True,elapsed_seconds=time.time()-start,time=time.time()))
        print('COMMON_STATE_GLOBAL',qualify,count,flush=True)
    finally:
        if model is not None:del model
        gc.collect();torch.cuda.empty_cache();lease.close()

def qualification_root():
    verify();q=read(BASE/'qualification/GPU_QUALIFICATION.json');assert q['status']=='pass' and q['logical_fits']==84
    from vg_tta.stvg_opd_common_state_geometry_v1 import mathematical_readback,bare
    from scripts.run_stvg_opd_p1_softmax_recovery_002 import equal
    checks=0;rounds=0;counts=dict(tensors=0,tensor_coordinates=0,scalar_values=0)
    for p,h in q['files'].items():
        path=ROOT/p;assert sha(path)==h;z=load(path);fit=z['fit'];r=load(path.parent/('REPEAT_'+z['arm']+'.pt'))
        equal(bare(fit),bare(r),counts);equal(fit['geometry'],r['geometry'],counts)
        assert mathematical_readback(fit)==z['math_audit']['independent'];checks+=z['math_audit']['independent']['scalar_checks'];rounds+=len(fit['rounds'])
    write(BASE/'ROOT_QUALIFICATION_READBACK.json',dict(status='pass',scope='all 28 cells/three arms, actual repeats, common initial states and independent Adam/cap arithmetic',
      qualification_cells=28,qualified_complete_fits=84,actual_GPU_fits=168,complete_rounds=rounds,
      independent_scalar_checks=checks,complete_repeat_bitwise=counts,GT_read=False,formal_predictions_accepted=0,time=time.time()))

def resume_original():
    from scripts.pause_stvg_opd_for_geometry_v1 import process
    pause=read(BASE/'BASELINE_PAUSE_RECEIPT.json')
    for role in ['worker','controller']:
        p=process(pause[role]['pid']);assert p['state']=='T' and p['start_ticks']==pause[role]['start_ticks']
    for p,r in pause['prefix'].items():
        assert sha(ROOT/p)==r['sha256'] and sha((ROOT/p).with_suffix('.json'))==r['receipt_sha256']
    for p,h in pause['protected'].items():assert sha(ROOT/p)==h
    os.kill(pause['worker']['pid'],signal.SIGCONT);os.kill(pause['controller']['pid'],signal.SIGCONT)
    write(BASE/'BASELINE_RESUMPTION_RECEIPT.json',dict(status='original_preserved_processes_resumed',
      scope='exact same original process memory and prefix; no optimizer reconstruction',worker_pid=pause['worker']['pid'],controller_pid=pause['controller']['pid'],
      original_prefix_predictions=len(pause['prefix']),original_prefix_reverified=True,GT_read=False,time=time.time()))

def controller():
    verify();env={**os.environ,'PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}
    for job in ['qualify','qualification_root','formal']:
        if job=='qualify' and (BASE/'qualification/GPU_QUALIFICATION.json').exists():continue
        if job=='qualification_root' and (BASE/'ROOT_QUALIFICATION_READBACK.json').exists():continue
        if job=='formal' and (BASE/'GLOBAL_PREDICTION_BARRIER.json').exists():continue
        e=dict(env)
        if job=='qualification_root':e['CUDA_VISIBLE_DEVICES']=''
        with (BASE/(job+'.log')).open('a') as f:
            p=subprocess.Popen([str(PYTHON),'-B',str(Path(__file__)),job],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env=e)
            status(BASE/'CONTROLLER_STATUS.json',dict(status='running',stage=job,controller_pid=os.getpid(),worker_pid=p.pid,GT_read=False,time=time.time()))
            assert p.wait()==0,'Actual new common-state failure preserved; no formal acceptance after failed qualification'
    resume_original()
    status(BASE/'CONTROLLER_STATUS.json',dict(status='pending_actual_postseal_CPU_root_view_publication_archive',GPU_complete=True,experiment_complete=False,GT_read=False,time=time.time()))

if __name__=='__main__':
    job=sys.argv[1]
    try:
        if job=='prepare':prepare()
        elif job in ['qualify','formal']:gpu_run(job=='qualify')
        elif job=='qualification_root':qualification_root()
        elif job=='controller':controller()
        else:raise ValueError(job)
    except BaseException:
        p=BASE/'GPU_failures'/str(time.time_ns());p.mkdir(parents=True,exist_ok=True);(p/'traceback.txt').write_text(traceback.format_exc())
        status(BASE/'CONTROLLER_STATUS.json',dict(status='failed_preserved_pending_actual_root',stage=job,evidence=str(p.relative_to(ROOT)),experiment_complete=False,time=time.time()));raise
