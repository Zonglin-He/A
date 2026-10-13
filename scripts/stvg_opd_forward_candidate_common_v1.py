"""Fixed eight-query custody and immutable real-forward screen authority."""
import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,status,sha,load,save
from scripts import stvg_opd_observation_response_projection_common_v1 as prior
BASE=ROOT/'artifacts/stvg_opd_forward_candidate_update_v1'
OUT=ROOT/'results/stvg_opd_forward_candidate_update/2026-10-13'
SCIENCE=['protocols/stvg_opd_forward_candidate_update_v1.md','vg_tta/stvg_opd_forward_candidate_update_v1.py',
 'scripts/stvg_opd_forward_candidate_common_v1.py','scripts/run_stvg_opd_forward_candidate_v1.py',
 'scripts/audit_stvg_opd_forward_candidate_v1.py']

def verify():
    prior.verify();r=read(BASE/'RUNTIME.json')
    assert sha(BASE/'DESIGN_LOCK.json')==r['design_sha256']
    pins=dict(r['pins'])
    for p in sorted(BASE.glob('*/REVISION_RUNTIME.json'),key=lambda p:read(p)['time']):
        v=read(p);assert v['original_runtime_sha256']==sha(BASE/'RUNTIME.json');pins.update(v['pin_overrides'])
    for n,h in {**pins,**r['bindings']}.items():assert sha(ROOT/n)==h,n
    return r

def prepare():
    import torch
    from vg_tta.stvg_opd_forward_candidate_update_v1 import contracts
    torch.set_num_threads(2);assert not torch.cuda.is_initialized() and not BASE.exists()
    prior.verify();assert read(prior.BASE/'FINAL_COMPLETION.json')['status']=='complete'
    assert read(prior.BASE/'TECHNICAL_DECISION.json')['status']=='stop_current_projection_recipe_before_formal'
    d=read(prior.BASE/'DESIGN_LOCK.json');rr=[r for r in d['records'] if r['index'] in d['qualification_indices']]
    assert [r['index'] for r in rr]==[0,1,2,3,32,33,34,36]
    bindings={str((prior.BASE/n).relative_to(ROOT)):sha(prior.BASE/n) for n in ['DESIGN_LOCK.json','RUNTIME.json','QUALIFICATION_BARRIER.json','FINAL_COMPLETION.json','TECHNICAL_DECISION.json']}
    for r in rr:
        assert sha(ROOT/r['A_path'])==r['A_sha256'];bindings[r['A_path']]=r['A_sha256']
        for repeat in [0,1]:
            for k in [0,r['config']['steps']//2,r['config']['steps']-1]:
                p=prior.BASE/'qualification'/f"{r['index']:05}"/f'P_k{k}_repeat{repeat}.pt'
                bindings[str(p.relative_to(ROOT))]=sha(p)
    BASE.mkdir();OUT.mkdir(parents=True)
    write(BASE/'DESIGN_LOCK.json',dict(status='locked_before_new_forward',records=rr,query_count=8,
      repeats=2,boundaries='0,floor(K/2),K-1',count=48,candidate_readouts=192,
      candidates=['A','LN','Query','Half'],rho=.9,acceptance_fraction_per_domain=.5,
      actual_new_training_fits=0,new_optimizer_steps=0,new_backward_calls=0,new_DINO_calls=0,
      GT_read=False,historical_exposure=True,future_task_validation_protocol_complete=False,time=time.time()))
    write(BASE/'CPU_CONTRACTS.json',dict(contracts(),time=time.time()))
    write(BASE/'BASELINE_STANDBY_IDENTITY.json',read(prior.BASE/'BASELINE_STANDBY_IDENTITY.json'))
    write(BASE/'RUNTIME.json',dict(status='locked_before_new_forward',pins={n:sha(ROOT/n) for n in SCIENCE},bindings=bindings,
      design_sha256=sha(BASE/'DESIGN_LOCK.json'),prior_runtime_sha256=sha(prior.BASE/'RUNTIME.json'),GT_read=False,time=time.time()))
    write(BASE/'SOURCE_STATE_CUSTODY_READBACK.json',dict(status='pass',scope='same exact eight fixed original states/Adam proposals/sealed noise and source/pixel inputs; prior negative technical result preserved',
      count=8,total=8,bindings=len(bindings),CPU_only=True,GT_read=False,time=time.time()))
    status(BASE/'STATUS.json',dict(status='prepared',done=0,total=48,CPU_only=True,GT_read=False,time=time.time()))
    verify()
