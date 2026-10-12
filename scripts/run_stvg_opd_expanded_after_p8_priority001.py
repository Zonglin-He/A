"""Original expanded-state worker under a saved-query priority/memory wrapper."""
import os,sys,time,json,fcntl,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,status,sha
from scripts.pause_stvg_opd_for_geometry_v1 import process
EXP=ROOT/'artifacts/stvg_opd_claim_strengthening_v1/expanded_state'
BASE=ROOT/'artifacts/stvg_opd_claim_strengthening_v1/after_P8_priority001'
class HostOnlyExpert:
    def __init__(self,snapshot):
        from transformers import AutoProcessor,AutoModelForZeroShotObjectDetection
        self.processor=AutoProcessor.from_pretrained(snapshot,local_files_only=True)
        self.model=AutoModelForZeroShotObjectDetection.from_pretrained(snapshot,local_files_only=True,disable_custom_kernels=True).cpu().eval().requires_grad_(False)
        assert all(not p.is_cuda for p in self.model.parameters())
    def __call__(self,*args,**kwargs):raise RuntimeError('Expanded state reuses sealed expert evidence; no new DINO call authorized here')
def cpu_contract():
    from scripts.stvg_opd_expanded_state_common_v1 import verify as original_verify
    original_verify()
    import torch
    from methods.decota_final_simplified_v1.config import EXPERT_SNAPSHOT,EXPERT_SHA256
    from methods.decota_final_simplified_v1.tensors import state_hash
    from scripts.decota_matrix_common_v1 import load
    torch.set_num_threads(2);assert not torch.cuda.is_initialized()
    expert=HostOnlyExpert(ROOT/EXPERT_SNAPSHOT)
    old=load(ROOT/'artifacts/stvg_opd_temporal_guided_spatial_candidate_v1/P7_B/expert_I1/00000.pt')
    assert state_hash(expert.model.state_dict())==old['same_expert_source_sha256']
    assert sha(ROOT/EXPERT_SNAPSHOT/'model.safetensors')==EXPERT_SHA256
    try:expert(None,None,None)
    except RuntimeError:pass
    else:raise AssertionError('Inactive expert call was not rejected')
    assert not torch.cuda.is_initialized()
    write(BASE/'CPU_STORAGE_CONTRACTS.json',dict(status='pass',scope='real same frozen expert checkpoint CPU storage/state hash matches previously sealed actual GPU expert; accidental calls rejected; original expanded scientific pins exact; no model forward or GPU qualification',contracts=4,expert_state_sha256=old['same_expert_source_sha256'],original_scientific_runtime_sha256=sha(EXP/'RUNTIME_LOCK.json'),CPU_only=True,GT_read=False,new_GPU_fits=0,new_model_forward_calls=0,time=time.time()))
def verify():
    from scripts.stvg_opd_expanded_state_common_v1 import verify as original_verify
    original_verify();r=read(BASE/'REVISION_RUNTIME.json')
    for n,h in r['pins'].items():assert sha(ROOT/n)==h,n
    assert sha(BASE/'PRIORITY_PAUSE_RECEIPT.json')==r['pause_sha256']
    assert read(ROOT/'artifacts/stvg_opd_support_factorization_v1/CPU_FACTORIAL_FINAL_COMPLETION.json')['status']=='complete'
    return r
def stopped():
    for p in read(BASE/'PRIORITY_PAUSE_RECEIPT.json')['baseline_processes']:
        actual=process(p['pid']);assert actual['state']=='T' and actual['start_ticks']==p['start_ticks'],p['pid']
def gpu():
    verify();stopped()
    from scripts.run_tastvg_evidence_vulnerability_v2 import install_clean_loader
    from scripts.decota_paper_common_v1 import guard
    import torch,numpy as np
    install_clean_loader();sys.addaudithook(guard);torch.set_num_threads(4);torch.manual_seed(20260920);np.random.seed(20260920)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    f=(BASE/'priority_GPU.lock').open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
    return f
def run(mode,name):
    verify()
    import torch
    from scripts import run_decota_paper_main_v1 as native
    from scripts import decota_paper_common_v1 as budget_module
    native.gpu=gpu
    original_budget=budget_module.budget
    def protected_budget():stopped();return original_budget()
    budget_module.budget=protected_budget
    # Import the qualified capture chain before replacing the original entry
    # point so its saved original forward reference cannot recurse.
    from vg_tta.stvg_opd_frozen_mask_cache012 import frozen_forward as memory_forward
    from scripts import run_spatial_ssl_gpu_v1 as capture_module
    capture_module.frozen_forward=memory_forward
    from methods.decota_final_simplified_v1 import observations as expert_module
    expert_module.SpatialExpert=HostOnlyExpert
    from scripts.run_stvg_opd_expanded_state_v1 import run_gpu
    run_gpu(name,mode=='qualification')
    verify();stopped()
    write(BASE/'worker_exit'/f'{mode}_{name}.json',dict(status='pass',scope='original complete expanded-state stage under priority/O012 frozen storage/inactive expert host storage',stage=name,qualification=mode=='qualification',new_DINO_calls=0,original_scientific_runtime_sha256=sha(EXP/'RUNTIME_LOCK.json'),CPU_only=False,GT_read=False,pid=os.getpid(),time=time.time()))
if __name__=='__main__':
    try:
        if sys.argv[1]=='cpu_contract':cpu_contract()
        else:run(*sys.argv[1:])
    except BaseException:
        p=BASE/'failures'/str(time.time_ns());p.mkdir(parents=True,exist_ok=False);(p/'traceback.txt').write_text(traceback.format_exc())
        status(BASE/'STATUS.json',dict(status='failed_preserved_pending_root',evidence=str(p.relative_to(ROOT)),GT_read=False,time=time.time()));raise
