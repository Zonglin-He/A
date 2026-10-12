"""Fixed P7 rosters, source bindings and serial GPU pause/resume metadata."""
import collections,fcntl,gc,os,signal,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,status,sha,load,save
BASE=ROOT/'artifacts/stvg_opd_temporal_guided_spatial_candidate_v1'
A=BASE/'P7_A';B=BASE/'P7_B'
OUT=ROOT/'results/stvg_opd_temporal_guided_candidate/2026-10-12'
O=ROOT/'artifacts/stvg_opd_optimization_geometry_v1'
OLD=ROOT/'artifacts/stvg_opd_paper_hc2_revision_v2'
P6=OLD/'P6_existing_temporal_revision001'
S=ROOT/'artifacts/stvg_opd_supplement_v1/baselines'
PYTHON=ROOT/'.conda/tubedetr/bin/python'

def contracts():
    from methods.decota_final_simplified_v1.observations import uniform_positions
    assert uniform_positions([0,2,5,9,12],[0,4],4)==uniform_positions([0,2,5,9,12],[0,4],4)
    assert len(uniform_positions([0,2,5,9,12],[1,2],4))==2
    from methods.decota_final_simplified_v1.objectives import project
    from scripts.stvg_opd_p6_existing_common001 import config,check_projection
    import torch
    z=[torch.tensor([[[.2,.3],[.5,.9],[-.1,.1],[.8,.2]]])]*2
    acts=[torch.tensor([-.3,.5,.4,-.2])]*2;rec=[dict(frame_ids=[0,3,8,11])]*2
    for ds in ['hc2','vidstg']:
        c=config(ds);ev=project(z,acts,rec,c);check_projection(z,acts,rec,ev,c)
    assert not (A/'GLOBAL_READOUT_BARRIER.json').exists()
    return dict(status='pass',scope='original Uniform4 and limited support/two-target unchanged projection/independent formula/preseal absence',
        contracts=4,CPU_only=True,GT_read=False,new_GPU_fits=0)

def prepare():
    import torch
    torch.set_num_threads(2);assert not BASE.exists();A.mkdir(parents=True);(OUT/'P7_A').mkdir(parents=True,exist_ok=False)
    assert read(O/'FINAL_COMPLETION.json')['status']=='complete'
    cohort=read(O/'COMMON_STATE_COHORT.json')['records'];cells=[r for r in cohort if r['condition']=='clean']
    assert len(cells)==512 and all(not r['development'] for r in cells)
    p6={};unique={};parents={};keys=collections.Counter()
    for ds in ['hc2','vidstg']:
        for p in sorted((P6/'formal'/ds).glob('*.pt')):
            z=load(p);rc=read(p.with_suffix('.json'));assert sha(p)==rc['sha256'] and not rc['GT_read']
            p6[ds,z['query_ordinal']]=dict(path=str(p.relative_to(ROOT)),sha256=sha(p))
    from scripts.stvg_opd_legacy_input_bridge003 import normalized
    records=[]
    for r in cells:
        key=(r['dataset'],r['query_ordinal']);ip=ROOT/r['original_Full']['input_path'];assert sha(ip)==r['original_Full']['input_sha256']
        inp=normalized(load(ip));nativekey=(inp['pixel_sha256'],tuple(inp['frame_ids']),inp['source_model_state_sha256'])
        if key not in unique:
            parents[key]=len(unique);unique[key]=dict(index=len(unique),dataset=r['dataset'],source=r['source'],query_ordinal=r['query_ordinal'],
                parent_id=parents[key],original_Full=r['original_Full'],input_identity=nativekey,previous_P6=key in p6,P6_binding=p6.get(key))
        else:assert unique[key]['input_identity']==nativekey and unique[key]['original_Full']['input_sha256']==r['original_Full']['input_sha256']
        records.append(dict(**r,public_cell=len(records),parent_id=parents[key],capture_index=unique[key]['index'],previous_P6=key in p6))
        keys[r['dataset'],r['order']]+=1
    assert len(unique)==256 and len(p6)==64 and sum(v['previous_P6'] for v in unique.values())==64
    assert set(keys.values())=={128}
    qualify=[]
    for ds in ['hc2','vidstg']:
        for prev in [True,False]:qualify.append(next(v['index'] for v in unique.values() if v['dataset']==ds and v['previous_P6']==prev))
    d=dict(status='locked_original_clean_roster_before_GPU_and_new_GT',records=records,captures=list(unique.values()),
        cells=512,parents=256,previous_P6_per_target=32,added_per_target=96,reused_temporal_captures=64,new_formal_temporal_captures=192,
        logical_outputs=4096,qualification_indices=qualify,actual_qualification_GPU_captures=8,
        temporal_configs={ds:__import__('scripts.stvg_opd_p6_existing_common001',fromlist=['config']).config(ds).to_dict() for ds in ['hc2','vidstg']},
        spatial_configs={ds:next(r['config'] for r in records if r['dataset']==ds) for ds in ['hc2','vidstg']},
        branch_P7_B=dict(rule='mean_delta_T_OPD_v>0 in added96 and all128 for both datasets after complete root',
            CI_not_a_fitted_gate=True,roster_not_selected_by_GT=True,cells=512,new_formal_spatial_fits=1024,actual_qualification_spatial_fits=16),
        GT_read=False,new_spatial_fits_P7_A=0,new_DINO_P7_A=0,history_exposed=True,time=time.time())
    write(BASE/'DESIGN_LOCK.json',d);write(BASE/'CPU_CONTRACTS.json',contracts())
    own=['protocols/stvg_opd_temporal_guided_candidate_p7_v1.md','scripts/stvg_opd_temporal_guided_p7_common_v1.py',
        'scripts/run_stvg_opd_temporal_capture_p7a_v1.py','scripts/continue_stvg_opd_temporal_guided_p7_v1.py',
        'scripts/score_stvg_opd_temporal_capture_p7a_v1.py']
    pins={f:sha(ROOT/f) for f in own}
    for f in ['methods/CURRENT_METHOD.json','methods/decota_spatial_opd_v1/configs.json','scripts/run_stvg_opd_baseline_supplement_v1.py',
        'scripts/continue_stvg_opd_baseline_supplement_v1.py','scripts/publish_stvg_opd_supplement_v1.py',
        'scripts/run_spatial_ssl_gpu_v1.py','scripts/run_decota_paper_main_v1.py','scripts/stvg_opd_legacy_input_bridge003.py',
        'methods/decota_final_simplified_v1/objectives.py','methods/decota_final_simplified_v1/observations.py']:
        pins[f]=sha(ROOT/f)
    for runtime in [O/'frozen_mask_cache012/REVISION_RUNTIME.json',OLD/'P6_existing_temporal_revision001/REVISION_RUNTIME_revision002.json']:
        pins.update(read(runtime)['pins'])
    write(BASE/'RUNTIME.json',dict(status='pre_registered_P7_A_and_conditional_P7_B',pins=pins,design_sha256=sha(BASE/'DESIGN_LOCK.json'),
        contracts_sha256=sha(BASE/'CPU_CONTRACTS.json'),original_O_root_sha256=sha(O/'FINAL_COMPLETION.json'),
        bootstrap_replicates=10000,seed=20261012,CUDA_allocator='expandable_segments:True',GT_before_global=False,
        production_promotion=False,time=time.time()))
    status(BASE/'STATUS.json',dict(status='prepared_pending_actual_GPU_qualification',stage='P7_A',GT_read=False,time=time.time()))
    print('P7_PREPARED',256,192,512,4096,flush=True)

def verify():
    r=read(BASE/'RUNTIME.json');assert sha(BASE/'DESIGN_LOCK.json')==r['design_sha256'] and sha(BASE/'CPU_CONTRACTS.json')==r['contracts_sha256']
    for p,h in r['pins'].items():assert sha(ROOT/p)==h,p
    assert sha(O/'FINAL_COMPLETION.json')==r['original_O_root_sha256'];return r

def pause(folder):
    from scripts.pause_stvg_opd_for_geometry_v1 import process
    verify();s=read(S/'STATUS.json');assert s['status']=='running' and s['job']=='formal'
    cp=int(s['controller_pid']);wp=int(s['worker_pid']);identities=[process(wp),process(cp)]
    assert 'run_stvg_opd_baseline_supplement_v1.py formal' in identities[0]['command']
    assert 'continue_stvg_opd_baseline_supplement_v1.py' in identities[1]['command']
    dest=S/'formal'/s['dataset']/s['candidate'];old=read(dest/'STATUS.json')
    os.kill(cp,signal.SIGSTOP)
    try:
        deadline=time.monotonic()+90
        while time.monotonic()<deadline:
            now=read(dest/'STATUS.json')
            if now['done']>old['done']:break
            time.sleep(.01)
        else:raise RuntimeError('No next saved query within bounded90s; no worker pause')
        os.kill(wp,signal.SIGSTOP);time.sleep(.5)
        assert all(process(p['pid'])['state']=='T' and process(p['pid'])['start_ticks']==p['start_ticks'] for p in identities)
        prefix={}
        for p in sorted(dest.glob('*/*.pt')):
            rc=p.with_suffix('.json')
            if rc.exists():
                h=sha(p);assert read(rc)['sha256']==h and not read(rc)['GT_read'];prefix[str(p.relative_to(ROOT))]=dict(sha256=h,receipt_sha256=sha(rc),bytes=p.stat().st_size)
        assert len(prefix)>=now['done']>old['done'];verify()
        write(folder/'PRIORITY_PAUSE_RECEIPT.json',dict(status='paused_same_original_processes_at_saved_query',
            scope='latest human P7 priority within prior saved-query pause authorization',resume_roles=identities,must_remain_stopped=identities,
            prefix=prefix,accepted_before=old['done'],accepted_stage_snapshot=now,original_process_optimizer_memory_preserved=True,
            original_shared_GPU_lease_preserved=True,GT_read=False,time=time.time()))
    except BaseException:
        for identity in identities:
            p=process(identity['pid'])
            if p['state']=='T':os.kill(identity['pid'],signal.SIGCONT)
        raise
    print('P7_BASELINE_PAUSED',now['done'],wp,cp,flush=True)

def resume(folder):
    from scripts.pause_stvg_opd_for_geometry_v1 import process
    if (folder/'PRIORITY_RESUMPTION_RECEIPT.json').exists():return
    p=read(folder/'PRIORITY_PAUSE_RECEIPT.json');verify()
    for name,b in p['prefix'].items():assert sha(ROOT/name)==b['sha256'] and sha((ROOT/name).with_suffix('.json'))==b['receipt_sha256']
    for identity in p['resume_roles']:
        actual=process(identity['pid']);assert actual['state']=='T' and actual['start_ticks']==identity['start_ticks'];os.kill(identity['pid'],signal.SIGCONT)
    write(folder/'PRIORITY_RESUMPTION_RECEIPT.json',dict(status='same_original_processes_resumed_after_actual_GPU_exit',
        prefix_files_rehashed=len(p['prefix']),original_optimizer_memory_preserved=True,
        worker_pid=p['resume_roles'][0]['pid'],controller_pid=p['resume_roles'][1]['pid'],GT_read=False,time=time.time()))

def lease(folder):
    from scripts.pause_stvg_opd_for_geometry_v1 import process
    verify();p=read(folder/'PRIORITY_PAUSE_RECEIPT.json')
    for old in p['must_remain_stopped']:
        actual=process(old['pid']);assert actual['state']=='T' and actual['start_ticks']==old['start_ticks']
    f=(folder/'priority_GPU.lock').open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
    import torch
    from scripts.run_tastvg_evidence_vulnerability_v2 import install_clean_loader
    from scripts.decota_paper_common_v1 import guard
    install_clean_loader();sys.addaudithook(guard);torch.set_num_threads(2);torch.manual_seed(20260920);np.random.seed(20260920)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    return f

def projection(z):
    from methods.decota_final_simplified_v1.objectives import project
    from scripts.stvg_opd_p6_existing_common001 import config,check_projection
    cfg=config(z['dataset']);c=z['capture'];e=project(c['zero']['logits'],c['zero']['actions'],c['records'],cfg)
    error=check_projection(c['zero']['logits'],c['zero']['actions'],c['records'],e,cfg)
    iv=[x['interval'] for x in e['offsets']]
    return [min(v[0] for v in iv),max(v[1] for v in iv)],dict(status='pass',offsets=2,max_error=error)

if __name__=='__main__':prepare()
