"""One isolated CPU model constructor; never GPU/model forward or GT."""
from pathlib import Path
import os,sys,json,hashlib,time,traceback
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
R=Path(__file__).resolve().parents[2];S=Path(__file__).parent;D=R/'outputs/edgetam_comparison_v1';P=R/'results/edgetam_comparison_v1';sys.path.insert(0,str(R))
from stvg.runtime import sha,guard
def atomic(path,value):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');os.replace(tmp,path)
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()

def prepare():
 assert not(D/'CPU_model_plan_v1.json').exists()
 code=[Path(__file__),S/'model_adapter_v1.py',S/'adapter.py'];inputs=[D/'preparation_plan.json',D/'preflight_receipt.json',D/'source_checkpoint_lock.json',D/'extension_reuse_receipt.json']
 p=dict(schema='official_EdgeTAM_complete_state_CPU_constructor_v1',scope='CPU model construction only, full official checkpoint key/tensor exact, scoped redundant Timm preload disabled. Official default video postprocessing, no forward or CUDA. Strict fill helper preserves original successful connected-component/where equations and raises errors instead of silent fallback; GPU support still requires actual exclusive pilot.',code_sha256={str(f.relative_to(R)):sha(f)for f in code},input_sha256={str(f.relative_to(R)):sha(f)for f in inputs});p['fingerprint']=digest(p);atomic(D/'CPU_model_plan_v1.json',p);atomic(P/'CPU_model_protocol_v1.json',{k:v for k,v in p.items()if k!='input_sha256'});print(p['fingerprint'])

def run():
 import torch
 from model_adapter_v1 import build_cpu,strict_fill
 sys.addaudithook(guard);torch.set_num_threads(1);p=json.loads((D/'CPU_model_plan_v1.json').read_text());assert p['fingerprint']==digest({k:v for k,v in p.items()if k!='fingerprint'})
 for field in ['code_sha256','input_sha256']:
  for f,h in p[field].items():assert sha(R/f)==h,f
 assert not(D/'CPU_model_attempt_v1.json').exists();start=time.perf_counter();atomic(D/'CPU_model_attempt_v1.json',dict(before_constructor=True,PID=os.getpid(),GT_read=False))
 try:
  model,load=build_cpu();assert all(not p.is_cuda for p in model.parameters())and not torch.cuda.is_initialized()
  # Model-free CPU check of the successful official postprocessing equations.
  import sam2.utils.misc as misc
  from types import SimpleNamespace
  mask=torch.tensor([[[[-1.,2.],[-2.,3.]]]]);labels=torch.tensor([[[[1,0],[2,0]]]]);areas=torch.tensor([[[[1,0],[9,0]]]])
  old=misc.get_connected_components;misc.get_connected_components=lambda x:(labels,areas)
  try:expected=misc.fill_holes_in_mask_scores(mask,8)
  finally:misc.get_connected_components=old
  actual=strict_fill(SimpleNamespace(get_connected_components=lambda x:(labels,areas)))(mask,8);assert torch.equal(expected,actual)
  def fail(x):raise RuntimeError('synthetic operator error')
  try:strict_fill(SimpleNamespace(get_connected_components=fail))(mask,8)
  except RuntimeError as e:assert str(e)=='synthetic operator error'
  else:raise AssertionError('Silent fallback forbidden')
  stat=Path('/proc',str(os.getpid()),'stat').read_text().rsplit(') ',1)[1].split();rec=dict(status='passed_CPU_complete_official_EdgeTAM_state_no_forward',fingerprint=p['fingerprint'],load=load,worker=dict(pid=os.getpid(),startticks=int(stat[19])),elapsed_wall_seconds=time.perf_counter()-start,new_model_constructor=1,new_model_forward=0,new_Source=0,new_original_expert=0,new_GT_scoring=0,CUDA_initialized=False,strict_postprocess_success_CPU_exact=True,synthetic_failure_raises_not_fallback=True,actual_CUDA_kernel_tested=False,actual_propagation=False);atomic(D/'CPU_model_bridge_receipt_v1.json',rec);atomic(P/'CPU_model_bridge_receipt_v1.json',rec);print(rec['status'])
 except BaseException:atomic(D/'CPU_model_failure_preserved_v1.json',dict(error=traceback.format_exc(),elapsed_wall_seconds=time.perf_counter()-start,new_model_forward=0,new_Source=0,new_original_expert=0,new_GT_scoring=0));raise
if __name__=='__main__':
 import argparse
 a=argparse.ArgumentParser();a.add_argument('command',choices=['prepare','run']);globals()[a.parse_args().command]()
