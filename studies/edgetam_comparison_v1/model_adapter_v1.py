"""Strict complete official EdgeTAM state; no vendor or original-environment edit."""
from pathlib import Path
import sys,json,hashlib,time
R=Path(__file__).resolve().parents[2];S=Path(__file__).parent;D=R/'outputs/edgetam_comparison_v1';V=R/'outputs/review_strengthening_v1/vendor/EdgeTAM'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def build_cpu():
 import torch,importlib.util
 lock=json.loads((D/'source_checkpoint_lock.json').read_text())
 for f,h in lock['source_files_SHA256'].items():assert sha(V/f)==h,f
 checkpoint=V/'checkpoints/edgetam.pt';assert sha(checkpoint)==lock['checkpoint']['sha256']
 sys.path.insert(0,str(V));import sam2
 assert Path(sam2.__file__).resolve().is_relative_to(V.resolve());sam2.__path__.append(str(D/'extensions/sam2'))
 import sam2._C as extension
 assert sha(extension.__file__)==json.loads((D/'extension_reuse_receipt.json').read_text())['module_sha256']
 from sam2.build_sam import build_sam2_video_predictor
 import sam2.modeling.backbones.timm as backbone
 spec=importlib.util.spec_from_file_location('_fixed_edgetam_adapter_v1',S/'adapter.py');adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)
 tick=time.perf_counter()
 with adapter.checkpoint_only_backbone(backbone):model=build_sam2_video_predictor('configs/edgetam.yaml',str(checkpoint),device='cpu',mode='eval',apply_postprocessing=True)
 state=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)['model'];current=model.state_dict();assert set(current)==set(state)
 for k,v in current.items():assert torch.equal(v.detach().cpu(),state[k]),k
 model.eval().requires_grad_(False);assert model.fill_hole_area==8
 return model,dict(all_official_state_keys_exact=True,all_official_state_tensors_exact=True,state_tensors=len(state),parameters=sum(p.numel()for p in model.parameters()),elapsed_CPU_model_construction_and_load_seconds=time.perf_counter()-tick,official_postprocessing=True,fill_hole_area=8,redundant_Timm_pretraining_suppressed_then_full_state_exact=True,source_commit=lock['commit'],checkpoint_sha256=lock['checkpoint']['sha256'],actual_forward=0)

def strict_fill(module):
 import torch
 def fill(mask,max_area):
  assert max_area>0
  labels,areas=module.get_connected_components(mask<=0)
  return torch.where((labels>0)&(areas<=max_area),0.1,mask)
 return fill
