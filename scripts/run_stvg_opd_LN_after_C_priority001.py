"""Original missing LN-only streams under exact frozen-storage priority scheduling."""
import os,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,sha,status
from scripts.stvg_opd_mechanism_LN_order_common_v1 import BASE as SCIENCE
from scripts.run_stvg_opd_expanded_after_p8_priority001 import HostOnlyExpert,stopped,gpu as priority_gpu
BASE=SCIENCE/'after_C_priority001'
def verify():
    from scripts.stvg_opd_mechanism_LN_order_common_v1 import verify as original_verify
    original_verify();r=read(BASE/'REVISION_RUNTIME.json')
    for p,h in {**r['pins'],**r['protected']}.items():assert sha(ROOT/p)==h,p
    c=read(ROOT/'artifacts/stvg_opd_claim_strengthening_v1/FINAL_COMPLETION.json')
    assert c['status']=='complete' and c['original_three_C_items_complete']
    assert read(ROOT/'artifacts/stvg_opd_supplement_v1/baselines/USER_CANCEL_REMAINING_FORMAL_RECEIPT.json')['status']=='remaining_formal_cancelled_by_explicit_human_instruction'
    stopped();return r
def cpu_contract():
    from scripts.stvg_opd_mechanism_LN_order_common_v1 import verify as original_verify
    original_verify()
    import torch
    from methods.decota_final_simplified_v1.config import EXPERT_SNAPSHOT,EXPERT_SHA256
    from methods.decota_final_simplified_v1.tensors import state_hash
    torch.set_num_threads(2);assert not torch.cuda.is_initialized()
    expert=HostOnlyExpert(ROOT/EXPERT_SNAPSHOT)
    reference=read(ROOT/'artifacts/stvg_opd_claim_strengthening_v1/after_P8_priority001/CPU_STORAGE_CONTRACTS.json')
    assert reference['status']=='pass' and state_hash(expert.model.state_dict())==reference['expert_state_sha256']
    assert sha(ROOT/EXPERT_SNAPSHOT/'model.safetensors')==EXPERT_SHA256
    assert all(not p.is_cuda for p in expert.model.parameters())
    try:expert(None,None,None)
    except RuntimeError:pass
    else:raise AssertionError('Unexpected expert call accepted')
    assert not torch.cuda.is_initialized()
    write(BASE/'CPU_STORAGE_CONTRACTS.json',dict(status='pass',scope='actual same original frozen expert CPU checkpoint/state hash equals previously qualified GPU hash; accidental forward rejected; exact original LN scientific runtime; no GPU/model forward qualification',
        contracts=4,expert_state_sha256=reference['expert_state_sha256'],original_scientific_runtime_sha256=sha(SCIENCE/'RUNTIME_LOCK.json'),
        original_qualified_CPU_storage_receipt_sha256=sha(ROOT/'artifacts/stvg_opd_claim_strengthening_v1/after_P8_priority001/CPU_STORAGE_CONTRACTS.json'),
        CPU_only=True,GT_read=False,new_GPU_fits=0,new_DINO_calls=0,new_model_forward_calls=0,time=time.time()))
    print('LN_ORIGINAL_EXPERT_CPU_STORAGE_CONTRACT_PASS')
def run(mode,name):
    verify()
    from scripts import run_decota_paper_main_v1 as native
    from scripts import decota_paper_common_v1 as budget_module
    native.gpu=priority_gpu;original_budget=budget_module.budget
    def budget():stopped();return original_budget()
    budget_module.budget=budget
    from scripts import stvg_opd_paper_common_v1 as paper_common
    paper_common.budget=budget
    from vg_tta.stvg_opd_frozen_mask_cache012 import frozen_forward as memory_forward
    from scripts import run_spatial_ssl_gpu_v1 as capture_module
    capture_module.frozen_forward=memory_forward
    from methods.decota_final_simplified_v1 import observations as expert_module
    expert_module.SpatialExpert=HostOnlyExpert
    from scripts.run_stvg_opd_mechanism_LN_order_v1 import run_gpu
    run_gpu(name,mode=='qualification');verify();stopped()
    write(BASE/'worker_exit'/f'{mode}_{name}.json',dict(status='pass',scope='original fixed missing LN-only stage under existing priority/O012 frozen storage/exact original inactive expert CPU checkpoint; no new DINO',
        stage=name,qualification=mode=='qualification',new_DINO_calls=0,original_scientific_runtime_sha256=sha(SCIENCE/'RUNTIME_LOCK.json'),CPU_only=False,GT_read=False,pid=os.getpid(),time=time.time()))
if __name__=='__main__':
    try:cpu_contract() if sys.argv[1]=='cpu_contract' else run(*sys.argv[1:])
    except BaseException:
        p=BASE/'failures'/str(time.time_ns());p.mkdir(parents=True,exist_ok=False);(p/'traceback.txt').write_text(traceback.format_exc())
        status(BASE/'STATUS.json',dict(status='failed_preserved_pending_actual_root',evidence=str(p.relative_to(ROOT)),CPU_only=False,GT_read=False,time=time.time()));raise
