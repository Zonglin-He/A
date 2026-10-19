"""Isolated official ReferDINO runtime: dependency ABI compatibility, no training."""
import os,sys,json,hashlib,shutil,time,difflib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
RT=ROOT/'.cache/referdino_runtime_v1';SRC=ROOT/'external/ReferDINO'
def sh(p):return hashlib.file_digest(open(p,'rb'),'sha256').hexdigest()
def main():
 import torch
 from transformers import BertModel,BertConfig
 from methods.decota_final_simplified_v1.config import EXPERT_SNAPSHOT
 RT.mkdir(exist_ok=True);bd=RT/'bert-from-frozen-teacher';bd.mkdir(exist_ok=True)
 checkpoint=torch.load(ROOT/'checkpoints/referdino/ryt_mevis_swinb.pth',map_location='cpu',weights_only=False);state=checkpoint['model_state_dict'];bs={k[5:]:v for k,v in state.items()if k.startswith('bert.')and not k.endswith('position_ids')}
 bert=BertModel(BertConfig());bert.load_state_dict(bs,strict=True);bert.save_pretrained(bd,safe_serialization=True)
 for n in ['vocab.txt','tokenizer.json','tokenizer_config.json','special_tokens_map.json']:shutil.copy2(ROOT/EXPERT_SNAPSHOT/n,bd/n)
 del bert,checkpoint,state,bs
 dest=RT/'cuda_src';shutil.copytree(SRC/'models/GroundingDINO/ops/src',dest,dirs_exist_ok=True);changes=[]
 for f in dest.rglob('*'):
  if f.is_file():
   before=f.read_text();after=before.replace('.type().is_cuda()', '.is_cuda()').replace('value.type(),', 'value.scalar_type(),').replace('.data<','.data_ptr<').replace('#include <THC/THCAtomics.cuh>','#include <ATen/cuda/Atomic.cuh>')
   if before!=after:changes.append(dict(path=str(f.relative_to(dest)),diff=''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='official/'+str(f.relative_to(dest)),tofile='torch27-ABI/'+str(f.relative_to(dest))))));f.write_text(after)
 os.environ['TORCH_CUDA_ARCH_LIST']='12.0';os.environ['MAX_JOBS']='2'
 import torch.utils.cpp_extension as ce
 import shlex
 original_ld=ce._prepare_ldflags
 def quoted_ld(*args,**kwargs):return ['-L'+shlex.quote(v[2:])if v.startswith('-L')and ' 'in v else v for v in original_ld(*args,**kwargs)]
 ce._prepare_ldflags=quoted_ld
 from torch.utils.cpp_extension import load
 build=RT/'cuda_build';build.mkdir(exist_ok=True);t=time.perf_counter()
 ext=load(name='MultiScaleDeformableAttention',sources=[str(dest/'vision.cpp'),str(dest/'cpu/ms_deform_attn_cpu.cpp'),str(dest/'cuda/ms_deform_attn_cuda.cu')],extra_include_paths=[str(dest)],extra_cflags=['-O2','-DWITH_CUDA'],extra_cuda_cflags=['-O2','-DWITH_CUDA','-DCUDA_HAS_FP16=1','-D__CUDA_NO_HALF_OPERATORS__','-D__CUDA_NO_HALF_CONVERSIONS__','-D__CUDA_NO_HALF2_OPERATORS__'],build_directory=str(build),verbose=True)
 result=dict(status='built',official_code_revision='3cfc01f57dff97f7d801b1bd54c251e0f34fcef8',native_model_decoder_unchanged=True,linker_path_space_quoted=True,changes_only_CPP_ABI=changes,build_seconds=time.perf_counter()-t,extension=ext.__file__,extension_sha256=sh(ext.__file__),BERT_initialized_from_same_ReferDINO_final_checkpoint=True,BERT_missing_learned_keys=0,original_external_checkout_unchanged=True,torch_version=torch.__version__,cuda=torch.version.cuda)
 (RT/'SETUP_RECEIPT.json').write_text(json.dumps(result,indent=2)+'\n');print('SETUP BUILT',flush=True)
if __name__=='__main__':main()
