"""Freeze assets/config and verify complete checkpoint on CPU, zero forward."""
from pathlib import Path
import urllib.request,json,hashlib,time,os,sys,argparse
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
R=Path(__file__).resolve().parents[2];S=Path(__file__).parent;D=R/'outputs/expert_generalization_v1';P=R/'results/expert_generalization_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+'\n')
def prepare():
 assert not(D/'CPU_model_plan.json').exists();assets=D/'vendor/roberta-base';assets.mkdir(exist_ok=True);revision='e2da8e2f811d1448a5b465c236feacd80ffbac7b';files={}
 for name in ['config.json','merges.txt','tokenizer.json','tokenizer_config.json','vocab.json']:
  dest=assets/name;assert not dest.exists();url=f'https://huggingface.co/FacebookAI/roberta-base/resolve/{revision}/{name}';data=urllib.request.urlopen(url,timeout=40).read();assert len(data)>10;dest.write_bytes(data);files[name]=dict(sha256=sha(dest),bytes=len(data),url=url)
 inputs=[D/'intent.json',D/'source_lock.json',D/'checkpoint_download.json',D/'input_contract_receipt.json',D/'runtime_doctor_receipt.json']
 v=dict(schema='ReferFormer_official_checkpoint_CPU_load_bridge_v1',scope='CPU model construction and strict learned/full-shared-state tensor coverage only, zero forward/optimizer/GT/CUDA. No actual expert output/model GPU bridge.',source_lock_sha256=sha(D/'source_lock.json'),checkpoint=json.loads((D/'checkpoint_download.json').read_text()),text_assets=dict(repository='FacebookAI/roberta-base',revision=revision,files=files),constructors='Official build_model defaultresnet50/5queries/4levels/with_box_refine/binary/masks/freeze_text_encoder. Skip redundant ImageNet and text-weight initialization, instantiate exact pinned Roberta config/tokenizer, then every final shared tensor equal official complete model state. Only verified deterministic index buffers may be absent/unexpected; any learned mismatch hardfails.',code_sha256={str(f.relative_to(R)):sha(f)for f in [Path(__file__),S/'referformer_adapter.py',S/'interface.py']},input_sha256={str(f.relative_to(R)):sha(f)for f in inputs});v['fingerprint']=hashlib.sha256(json.dumps(v,sort_keys=True).encode()).hexdigest();save(D/'CPU_model_plan.json',v);save(P/'CPU_model_protocol.json',{k:x for k,x in v.items()if k!='input_sha256'});print(v['fingerprint'])
def run():
 import torch
 torch.set_num_threads(1);p=json.loads((D/'CPU_model_plan.json').read_text());assert p['fingerprint']==hashlib.sha256(json.dumps({k:v for k,v in p.items()if k!='fingerprint'},sort_keys=True).encode()).hexdigest()
 for field in ['code_sha256','input_sha256']:
  for path,h in p[field].items():assert sha(R/path)==h,path
 for name,e in p['text_assets']['files'].items():assert sha(D/'vendor/roberta-base'/name)==e['sha256']
 sys.path.insert(0,str(S));from referformer_adapter import build_cpu
 model,tokenizer,proof=build_cpu();assert not torch.cuda.is_initialized();v=dict(status='passed_actual_CPU_model_constructor_full_official_state_coverage_no_forward',fingerprint=p['fingerprint'],load=proof,new_model_constructor=1,new_model_forward=0,new_Source=0,new_expert_forward=0,new_optimizer=0,new_GT_scoring=0,CUDA_initialized=False,actual_GPU_model_bridge=False);save(D/'CPU_model_bridge_receipt.json',v);save(P/'CPU_model_bridge_receipt.json',v);print(v['status'])
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('command',choices=['prepare','run']);globals()[a.parse_args().command]()
