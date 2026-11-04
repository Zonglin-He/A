"""Independent post-CUDA-recovery entry; original failed preflight stays immutable."""
from pathlib import Path
import os,sys,json,hashlib,shutil,ctypes
R=Path(__file__).resolve().parents[2];S=Path(__file__).parent;ROOT_D=R/'outputs/expert_guided_ttft_v1';D=ROOT_D/'source_gradient_preflight_v2';P=R/'results/expert_guided_ttft_v1/source_gradient_preflight_v2'
sys.path[:0]=[str(S),str(R)]
import preflight_gpu as original
read=original.read;sha=original.sha;atomic=original.atomic;digest=original.digest
def old_bound_exact():
 p=read(ROOT_D/'source_gradient_preflight_plan.json')
 for name,h in p['code_and_input_sha256'].items():assert sha(R/name)==h,name
 return p
def prepare():
 assert not(D/'source_gradient_preflight_plan.json').exists();old=old_bound_exact();assert read(ROOT_D/'source_gradient_preflight_actual_exit.json')['exit_code']==1
 # Exact copies of previously verified preparation receipts, not new CPU doctors.
 for name in ['CPU_doctor_complete.json','calibration_intent_v1.json']:
  dest=D/name;dest.parent.mkdir(parents=True,exist_ok=True);assert not dest.exists();shutil.copy2(ROOT_D/name,dest);assert sha(dest)==sha(ROOT_D/name)
 original.D=D;original.P=P;original.prepare()
 cfg=read(R/'configs/default.json');files=[Path(__file__),ROOT_D/'source_gradient_preflight_plan.json',ROOT_D/'source_gradient_preflight_actual_exit.json',D/'source_gradient_preflight_plan.json',R/'configs/default.json']
 files+=list((R/'stvg').glob('*.py'));files+=list((R/'external/TA-STVG/models').rglob('*.py'))
 for ds in ['hc2','vidstg']:files.append(R/cfg['datasets'][ds]['source_checkpoint'])
 manifest=dict(schema='independent_runtime_namespace_after_original_CUDA_startup_failure_v2',old_science_fp=old['fingerprint'],new_preflight_fp=read(D/'source_gradient_preflight_plan.json')['fingerprint'],old_failure_counts=dict(Source=0,expert=0,optimizer_steps=0),preservation=str(ROOT_D/'preserved/CUDA_initialization_before_forward_v1'),changes='Namespace/launch gate only; original Source extraction, observer, all spatial-gradient math, expert loss/config and reset functions executed unchanged. No old owner restored.',reuse_CPU_doctors='Exact byte copies of original verified doctor receipt; no new model construction implied',GPU_gate='Fresh working cuInit required before any model construction, original protected-service/foreign-compute check and global GPU lease; cannot reboot/reset device from this entry.',code_and_weights_sha256={str(f.relative_to(R)):sha(f)for f in files})
 manifest['fingerprint']=digest(manifest);atomic(D/'execution_v2.json',manifest);atomic(P/'execution_protocol.json',manifest);print(manifest['fingerprint'])
def run():
 old_bound_exact();plan=read(D/'execution_v2.json');assert digest({k:v for k,v in plan.items()if k!='fingerprint'})==plan['fingerprint']
 for name,h in plan['code_and_weights_sha256'].items():assert sha(R/name)==h,name
 assert not(D/'source_gradient_preflight_worker.json').exists();assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0';original.foreign();driver=ctypes.CDLL('libcuda.so.1');code=driver.cuInit(0)
 if code!=0:
  atomic(D/'CUDA_gate_failed_before_model.json',dict(status='CUDA_not_recovered_do_not_launch_model',cuInit=code,new_Source=0,new_expert=0,new_model=0,optimizer_steps=0));raise RuntimeError(f'cuInit={code}; requires working host CUDA before this preflight')
 original.D=D;original.P=P;original.run()
if __name__=='__main__':
 assert sys.argv[1]in ['prepare','run'];globals()[sys.argv[1]]()
