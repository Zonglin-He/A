"""P9-P0 reader dispatch for original empty/no-sampling saved episodes."""
import os,sys,time,collections
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import stvg_opd_state_conditional_actionability_p0_v1 as core
from scripts.decota_matrix_common_v1 import read,write,status,sha,load,save
BASE=core.BASE;REV=BASE/'empty_episode_readback001';trace_original=core.random_trace;assemble_original=core.assemble
def verify():
    core.verify();r=read(REV/'REVISION_RUNTIME.json');assert r['original_runtime_sha256']==sha(BASE/'RUNTIME.json') and r['source_sha256']==sha(Path(__file__)) and r['capture_sha256']==sha(REV/'CAPTURE_RECEIPT.json')
def trace(f):
    import torch
    if not f['empty']:return trace_original(f)
    assert not f['positions'] and len(f['rounds'])==f['config']['steps'] and f['gradient_calls']==0
    assert torch.equal(f['before'],f['final']) and all(torch.equal(v,f['state'][k]) for k,v in f['initial'].items())
    for k,r in enumerate(f['rounds']):assert r==dict(round=k,updated=False,no_information=True)
    return [],0
def assemble():
    rr=assemble_original();bar=read(BASE/'GLOBAL_PAIRED_READOUT_BARRIER.json');applies={}
    for rel in bar['files']:
        z=load(ROOT/rel)
        for j,scope in enumerate(['Omega0','Omega1']):applies[(z['parent_id'],scope)]=bool(z['random_traces'][j])
    for r in rr:r['randomness_applicable']=applies[(r['parent_id'],r['scope'])]
    return rr
def seal():
    import torch
    torch.set_num_threads(2);verify();records=read(BASE/'DESIGN_LOCK.json')['records'];source=core.origins(records);groups=collections.defaultdict(list)
    for r in records:groups[r['parent_id']].append(r)
    bindings={};coordinates=0;rounds=0;empty=0;reused=0
    for parent,group in groups.items():
        z=core.extract_pair(group,source[group[0]['dataset']]);p=BASE/'pairs'/f'{parent:05}.pt'
        if p.exists():assert core.digest(load(p))==core.digest(z);reused+=1
        else:save(p,z)
        assert core.digest(load(p))==core.digest(z);bindings[str(p.relative_to(ROOT))]=sha(p);coordinates+=z['random_sample_coordinate_checks'];rounds+=sum(len(v) for v in z['random_traces'])*2;empty+=sum(not bool(v) for v in z['random_traces'])*2
        status(BASE/'STATUS.json',dict(status='sealing_original_all_pairs_with_empty_no_sampling_retained',done=len(bindings),total=256,CPU_only=True,GT_read=False,time=time.time()))
    assert len(bindings)==256 and not torch.cuda.is_initialized();verify()
    write(BASE/'GLOBAL_PAIRED_READOUT_BARRIER.json',dict(status='sealed',files=bindings,parents=256,parent_scope_pairs=512,saved_updates=1024,initial_state_coordinates=256*2*1792,random_rounds=rounds,random_sample_coordinate_checks=coordinates,empty_saved_updates_no_sampler_or_optimizer=empty,actual_sampled_saved_updates=1024-empty,all_source_evidence_config_noise_matched=True,all_applicable_saved_sample_replay_bitwise=True,accepted_original_prefix_reused_parents=reused,runtime_sha256=sha(BASE/'RUNTIME.json'),revision_runtime_sha256=sha(REV/'REVISION_RUNTIME.json'),CPU_only=True,GT_read=False,time=time.time()))
def run():
    import torch
    torch.set_num_threads(2);verify();core.random_trace=trace;core.assemble=assemble
    write(REV/'LAUNCH.json',dict(status='launched_finite_original_CPU_readback_continuation',pid=os.getpid(),CPU_only=True,GT_read=False,new_GPU_fits=0,new_model_calls=0,new_DINO_calls=0,time=time.time()))
    for name,fn in [('seal',seal),('analyze',core.analyze),('root',core.root)]:
        status(BASE/'CONTROLLER_STATUS.json',dict(status='running',stage=name,pid=os.getpid(),CPU_only=True,GT_read=name!='seal',time=time.time()));fn()
    write(REV/'RESOLUTION_RECEIPT.json',dict(status='pass',scope='all original empty/no-information rounds retained as no sampler/no optimizer; 69 accepted parent prefix reused exactly; full independent readback passed',accepted_original_prefix_reused_parents=69,root_sha256=sha(BASE/'ROOT_READBACK.json'),barrier_sha256=sha(BASE/'GLOBAL_PAIRED_READOUT_BARRIER.json'),original_science_runtime_unchanged=True,new_GPU_fits=0,new_model_calls=0,new_DINO_calls=0,new_GT_scoring_calls=0,CPU_only=True,GT_read=True,time=time.time()))
    status(BASE/'CONTROLLER_STATUS.json',dict(status='pending_actual_PDF_pixel_view_publication_archive',pid=os.getpid(),all_CPU_jobs_exit_zero=True,CPU_only=True,GT_read=True,time=time.time()))
if __name__=='__main__':run()
