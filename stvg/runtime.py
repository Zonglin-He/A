import os,sys,time,json,hashlib,fcntl,gc,subprocess
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]


def sha(path):
 h=hashlib.sha256()
 with open(path,'rb')as f:
  for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
 return h.hexdigest()

def guard(event,args):
 if event=='open' and args and isinstance(args[0],(str,bytes,Path)):
  raw=os.fsdecode(args[0]);paths={raw,str(Path(raw).resolve())}
  denied=['/annos/','/annotations/','_labels.jsonl','labels_diagnostic','GT_LABEL','test_annotations.json','valv2_proc.json']
  if any(x in s for s in paths for x in denied):raise PermissionError('Inference forbids annotation access: '+raw)

def setup():
 os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
 torch.set_num_threads(4);torch.manual_seed(20260920);np.random.seed(20260920)
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 sys.addaudithook(guard)

def lease():
 f=(ROOT/'runtime/gpu.lock').open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);return f

def load_source(config):
    from stvg.source_loader import official_imports, load_model_on_device
    official_imports();pipeline=sys.modules['models.pipeline'];old=pipeline.load_json;seen=[]
    def empty(path):
        assert str(path).endswith(('/annos/train.json','/annos/test.json'));seen.append(str(path));return {}
    pipeline.load_json=empty
    try:
        src='vidstg'if config['source_dataset']=='vidstg'else'hcstvg2';runtime='vidstg'if src=='vidstg'else'hc-stvg2'
        model,_,_=load_model_on_device(src,ROOT/'runtime/source'/runtime,
            checkpoint=ROOT/config['source_checkpoint'],device='cuda',source_dataset=src)
    finally:pipeline.load_json=old
    assert len(seen)==2 and model.verb_label=={}and model.verb_label2=={}
    return model.eval().requires_grad_(False)
