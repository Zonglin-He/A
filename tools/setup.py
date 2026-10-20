"""Prepare existing model assets and, only when requested, the frozen CUDA operator."""
import argparse,json,sys,os,shutil,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');parser.add_argument('--prepare',action='store_true');parser.add_argument('--build',action='store_true');parser.add_argument('--tokenizer',type=Path);args=parser.parse_args();runtime=ROOT/'runtime/referdino'
 if args.prepare:
  for name in ['hc-stvg2','vidstg']:
   dest=ROOT/'runtime/source'/name/'manifest.json';dest.parent.mkdir(parents=True,exist_ok=True)
   if not dest.exists():dest.write_text(json.dumps(dict(dataset=name,label_free_construction=True))+'\n')
  dest=runtime/'bert-from-frozen-teacher'
  if not(dest/'model.safetensors').exists():
   import torch
   from transformers import BertModel,BertConfig
   ck=torch.load(ROOT/'checkpoints/referdino/ryt_mevis_swinb.pth',map_location='cpu',weights_only=False)['model_state_dict'];state={k[5:]:v for k,v in ck.items()if k.startswith('bert.')and not k.endswith('position_ids')};bert=BertModel(BertConfig());bert.load_state_dict(state,strict=True);dest.mkdir(parents=True,exist_ok=True);bert.save_pretrained(dest,safe_serialization=True)
  for name in ['vocab.txt','tokenizer.json','tokenizer_config.json','special_tokens_map.json']:
   if not(dest/name).exists():
    if args.tokenizer is None:raise FileNotFoundError('Supply the BERT tokenizer files with --tokenizer; no automatic model download')
    shutil.copyfile(args.tokenizer/name,dest/name)
 if args.build:
  import torch.utils.cpp_extension as ce,shlex
  src=ROOT/'external/ReferDINO/compat';build=runtime/'cuda_build';build.mkdir(parents=True,exist_ok=True);os.environ.setdefault('MAX_JOBS','2');original=ce._prepare_ldflags
  def quoted(*a,**kw):return ['-L'+shlex.quote(s[2:])if s.startswith('-L')and' 'in s else s for s in original(*a,**kw)]
  ce._prepare_ldflags=quoted
  try:ce.load(name='MultiScaleDeformableAttention',sources=[str(src/'vision.cpp'),str(src/'cpu/ms_deform_attn_cpu.cpp'),str(src/'cuda/ms_deform_attn_cuda.cu')],extra_include_paths=[str(src)],extra_cflags=['-O2','-DWITH_CUDA'],extra_cuda_cflags=['-O2','-DWITH_CUDA','-DCUDA_HAS_FP16=1','-D__CUDA_NO_HALF_OPERATORS__','-D__CUDA_NO_HALF_CONVERSIONS__','-D__CUDA_NO_HALF2_OPERATORS__'],build_directory=str(build),verbose=True)
  finally:ce._prepare_ldflags=original
 if args.check:
  assets=json.loads((ROOT/'docs/assets.json').read_text());missing=[]
  for rel,h in assets['required_files'].items():
   f=ROOT/rel
   if not f.is_file():missing.append(rel);continue
   assert hashlib.sha256(f.read_bytes()).hexdigest()==h,rel
  assert not missing,missing
  assert (ROOT/'runtime/referdino/cuda_build/MultiScaleDeformableAttention.so').is_file(),'Build the local CUDA operator with --build'
  print('Required frozen assets present; hashes match. No forward or training was performed.')
 if not any([args.check,args.prepare,args.build]):parser.print_help()
if __name__=='__main__':main()
