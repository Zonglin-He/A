"""Import official code and inspect checkpoint on CPU; never build a model."""
import os,sys,json,hashlib,time,shutil,importlib.metadata as md
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
R=Path(__file__).resolve().parents[2];D=R/'outputs/expert_generalization_v1';P=R/'results/expert_generalization_v1';V=D/'vendor/ReferFormer';overlay=D/'runtime/python'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+'\n')
def run():
 import torch
 torch.set_num_threads(1);assert not torch.cuda.is_initialized();download=json.loads((D/'checkpoint_download.json').read_text());cp=D/'vendor/r50_refytvos_finetune.pth';assert cp.stat().st_size==download['bytes']and sha(cp)==download['sha256']
 # This extension's original seven C++/CUDA source files are byte identical
 # between both official projects. Existing ABI-compatible compiled artifact is
 # reused read-only for CPU import, never claimed kernel-tested for ReferFormer.
 source=R/'external/ReferDINO/models/GroundingDINO/ops/src';exact={}
 for p in (V/'models/ops/src').rglob('*'):
  if p.is_file():rel=p.relative_to(V/'models/ops/src');q=source/rel;assert p.read_bytes()==q.read_bytes();exact[str(rel)]=sha(p)
 pycoco=R/'runtime/referdino/python/pycocotools';target=overlay/'pycocotools'
 if not target.exists():shutil.copytree(pycoco,target)
 sys.path[:0]=[str(V),str(overlay),str(R/'runtime/referdino/cuda_build')]
 started=time.perf_counter();from models import build_model
 import MultiScaleDeformableAttention as extension
 assert callable(build_model)and callable(extension.ms_deform_attn_forward);assert not torch.cuda.is_initialized()
 # Load state data only; no class/model/optimizer constructors are invoked.
 ck=torch.load(cp,map_location='cpu',weights_only=False);state=ck['model'];assert isinstance(state,dict)and state;assert all(not t.is_cuda for t in state.values()if torch.is_tensor(t))
 keys=list(state);assert any(k.startswith('backbone')for k in keys)and any(k.startswith('text_encoder')for k in keys)and any('mask' in k for k in keys)
 v=dict(status='official_model_builder_imported_checkpoint_state_inspected_CPU_only',checkpoint_sha256=download['sha256'],checkpoint_bytes=download['bytes'],checkpoint_state_tensors=len(state),checkpoint_state_elements=sum(t.numel()for t in state.values()if torch.is_tensor(t)),state_includes_backbone_and_text_encoder_and_masks=True,checkpoint_actual_model_strict_load_pending=True,official7original_CUDA_source_files_byte_exact_to_existing_ReferDINO=exact,compiled_extension_sha256=sha(R/'runtime/referdino/cuda_build/MultiScaleDeformableAttention.so'),existing_extension_readonly_imported=True,actual_CUDA_kernel_tested=False,actual_ReferFormer_model_constructed=False,new_model=0,new_Source=0,new_expert=0,new_optimizer=0,new_GT_scoring=0,CUDA_initialized=torch.cuda.is_initialized(),original_environment_unchanged=True,dependency_overlay_files_sha256={str(p.relative_to(overlay)):sha(p)for p in overlay.rglob('*')if p.is_file()and '__pycache__'not in p.parts},elapsed_wall_seconds=time.perf_counter()-started,code_sha256=sha(Path(__file__)))
 assert not v['CUDA_initialized'];save(D/'runtime_doctor_receipt.json',v);save(P/'runtime_doctor_receipt.json',{k:x for k,x in v.items()if k!='dependency_overlay_files_sha256'});print(v['status'],v['checkpoint_state_tensors'])
if __name__=='__main__':run()
