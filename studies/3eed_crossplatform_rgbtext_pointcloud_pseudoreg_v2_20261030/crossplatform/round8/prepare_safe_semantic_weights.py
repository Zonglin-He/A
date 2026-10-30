"""One-time pinned tensor-only conversion; no unrestricted pickle or CUDA."""
from pathlib import Path
import hashlib,json,time
import torch
from safetensors.torch import save_file,load_file
from transformers import CLIPConfig,CLIPModel
ROOT=Path(__file__).resolve().parent
A=ROOT/'assets/clip-vit-base-patch32'
EXPECTED='a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f'
REVISION='3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def main():
 torch.set_num_threads(2);started=time.monotonic();source=A/'pytorch_model.bin';assert source.stat().st_size==605247071 and sha(source)==EXPECTED
 state=torch.load(source,map_location='cpu',weights_only=True,mmap=True)
 assert isinstance(state,dict) and all(isinstance(k,str) and isinstance(v,torch.Tensor) and v.device.type=='cpu' for k,v in state.items())
 config=CLIPConfig.from_pretrained(str(A),local_files_only=True);model=CLIPModel(config).eval();model.requires_grad_(False)
 generated=model.state_dict();removed=[]
 for name in list(state):
  if name not in generated:
   assert name in ('text_model.embeddings.position_ids','vision_model.embeddings.position_ids'),('Unexpected official tensor',name)
   n=config.text_config.max_position_embeddings if name.startswith('text') else (config.vision_config.image_size//config.vision_config.patch_size)**2+1
   assert state[name].dtype==torch.int64 and state[name].shape==(1,n) and torch.equal(state[name],torch.arange(n)[None])
   removed.append({'name':name,'shape':list(state[name].shape),'dtype':str(state[name].dtype),'reason':'legacy deterministic arange position_ids; current Transformers registers this buffer nonpersistently; exact values verified, no learned weight omitted'})
   del state[name]
 assert set(state)==set(generated)
 assert all(state[k].shape==generated[k].shape and state[k].dtype==generated[k].dtype for k in state)
 model.load_state_dict(state,strict=True)
 target=A/'model.safetensors';save_file({k:v.contiguous() for k,v in state.items()},str(target),metadata={'format':'pt','source_repo_id':'openai/clip-vit-base-patch32','source_revision':REVISION,'source_pytorch_sha256':EXPECTED,'conversion':'torch.load(weights_only=True,map_location=cpu); strict CLIPModel; no learned key omissions'})
 safe=load_file(str(target),device='cpu');assert set(safe)==set(state) and all(torch.equal(safe[k],state[k]) for k in safe);model.load_state_dict(safe,strict=True)
 names=['model.safetensors','config.json','preprocessor_config.json','merges.txt','special_tokens_map.json','tokenizer.json','tokenizer_config.json','vocab.json']
 receipt={'state':'safe_weights_only_official_to_local_safetensors_strict_model_verified','repo_id':'openai/clip-vit-base-patch32','revision':REVISION,'source_format':'official pytorch_model.bin; repo has no safetensors at fixed revision','original_pytorch_sha256':EXPECTED,'original_bytes':source.stat().st_size,'derived_safetensors_sha256':sha(target),'derived_bytes':target.stat().st_size,'tensor_key_count':len(state),'tensor_key_order_sha256':hashlib.sha256('\n'.join(sorted(state)).encode()).hexdigest(),'parameter_count':sum(p.numel() for p in model.parameters()),'all_loaded_tensor_values_bitexact_after_safetensors_roundtrip':True,'strict_model_load_all_learned_keys_and_shapes':True,'legacy_deterministic_nonpersistent_buffers':removed,'runtime_files_sha256':{name:sha(A/name) for name in names},'torch_version':torch.__version__,'no_CUDA_initialization':not torch.cuda.is_initialized(),'no_unrestricted_pickle_no_remote_code':True,'seconds':time.monotonic()-started}
 (A/'SAFE_CONVERSION_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
