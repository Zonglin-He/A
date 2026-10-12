"""Original complete deployment path under reversible inactive-storage scheduling."""
import gc,os,sys,time,weakref,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,sha,status
from scripts.run_stvg_opd_expanded_after_p8_priority001 import stopped,gpu as priority_gpu
BASE=ROOT/'artifacts/stvg_opd_claim_strengthening_v1/deployment_latency/after_expanded_priority002'
LAT=BASE.parent
EXPERTS=[]
def verify():
    from scripts.run_stvg_opd_deployment_latency_v1 import verify as original_verify
    original_verify();r=read(BASE/'REVISION_RUNTIME.json')
    for p,h in {**r['pins'],**r['protected']}.items():assert sha(ROOT/p)==h,p
    assert read(ROOT/'artifacts/stvg_opd_claim_strengthening_v1/expanded_state/FINAL_COMPLETION.json')['status']=='complete'
    assert read(ROOT/'artifacts/stvg_opd_supplement_v1/baselines/USER_CANCEL_REMAINING_FORMAL_RECEIPT.json')['status']=='remaining_formal_cancelled_by_explicit_human_instruction'
    stopped();return r
def stored_capture(forward,model,frames,row,expert):
    if expert is None:return forward(model,frames,row)
    import torch
    device=next(expert.model.parameters()).device
    expert.model.to('cpu');gc.collect()
    if torch.cuda.is_initialized():torch.cuda.empty_cache()
    try:return forward(model,frames,row)
    finally:expert.model.to(device)
def cpu_contract():
    import torch
    from types import SimpleNamespace
    torch.set_num_threads(2);assert not torch.cuda.is_initialized()
    checks=0;coordinates=0
    for dtype in [torch.float32,torch.float64]:
        module=torch.nn.Linear(3,2,bias=True).to(dtype).eval().requires_grad_(False)
        expert=SimpleNamespace(model=torch.nn.Sequential(torch.nn.Linear(3,2),torch.nn.BatchNorm1d(2)).to(dtype).eval().requires_grad_(False))
        original={n:v.clone() for n,v in expert.model.state_dict().items()}
        frames=torch.arange(12,dtype=dtype).reshape(4,3);row={'ordinal':3};calls=[]
        expected=module(frames).clone()
        def forward(m,f,r):
            assert m is module and f is frames and r is row
            assert all(not p.is_cuda for p in expert.model.parameters());calls.append(1);return m(f)
        output=stored_capture(forward,module,frames,row,expert)
        assert calls==[1] and torch.equal(output,expected)
        assert all(torch.equal(original[n],v) for n,v in expert.model.state_dict().items())
        coordinates+=output.numel()+sum(v.numel() for v in original.values());checks+=3
        calls.clear();output=stored_capture(forward,module,frames,row,None)
        assert calls==[1] and torch.equal(output,expected);checks+=1;coordinates+=output.numel()
    assert not torch.cuda.is_initialized()
    write(BASE/'CPU_STORAGE_CONTRACTS.json',dict(status='pass',scope='synthetic CPU identity forwarding/optional inactive expert exact weights and buffers/device restoration; not actual GPU qualification or OOM guarantee',
        contracts=checks,tensor_coordinates=coordinates,CUDA_initialized=False,CPU_only=True,GT_read=False,
        new_model_inference_calls=0,actual_GPU_fits=0,new_DINO_calls=0,time=time.time()))
    print('DEPLOYMENT_STORAGE_CPU_CONTRACT_PASS',checks,coordinates)
def run(job,args):
    verify()
    from scripts import run_decota_paper_main_v1 as native
    from scripts import decota_paper_common_v1 as budget_module
    native.gpu=priority_gpu
    original_budget=budget_module.budget
    def budget():stopped();return original_budget()
    budget_module.budget=budget
    # Bind the saved native capture chain before installing the entry wrapper.
    from vg_tta.stvg_opd_frozen_mask_cache012 import frozen_forward as memory_capture
    from scripts import run_spatial_ssl_gpu_v1 as capture_module
    from methods.decota_final_simplified_v1 import observations as expert_module
    original_expert=expert_module.SpatialExpert
    class ResidentExpert(original_expert):
        def __init__(self,*a,**k):
            super().__init__(*a,**k);EXPERTS.append(weakref.ref(self))
    expert_module.SpatialExpert=ResidentExpert
    def frozen_forward(model,frames,row):
        alive=[r() for r in EXPERTS if r() is not None];assert len(alive)<=1
        return stored_capture(memory_capture,model,frames,row,alive[0] if alive else None)
    capture_module.frozen_forward=frozen_forward
    from scripts import run_stvg_opd_deployment_latency_v1 as original
    if job=='qualification':original.qualification(args[0])
    elif job=='timing':original.timed(args[0],args[1],int(args[2]))
    else:raise ValueError(job)
    verify();stopped()
    write(BASE/'worker_exit'/('_'.join([job]+args)+'.json'),dict(status='pass',scope='original deployment science under priority/native O012 storage/inactive expert host staging; all storage overhead remains within complete timer',
        stage=job,dataset=args[0],original_runtime_sha256=sha(LAT/'preGPU_receipt_dispatch002/REVISION_RUNTIME.json'),
        CPU_only=False,GT_read=False,pid=os.getpid(),time=time.time()))
if __name__=='__main__':
    try:
        if sys.argv[1]=='cpu_contract':cpu_contract()
        else:run(sys.argv[1],sys.argv[2:])
    except BaseException:
        p=BASE/'failures'/str(time.time_ns());p.mkdir(parents=True,exist_ok=False);(p/'traceback.txt').write_text(traceback.format_exc())
        status(BASE/'STATUS.json',dict(status='failed_preserved_pending_actual_root',evidence=str(p.relative_to(ROOT)),CPU_only=False,GT_read=False,time=time.time()));raise
