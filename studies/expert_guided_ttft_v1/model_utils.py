"""Source model access and exact spatial-branch parameter scope; no training entry."""
from pathlib import Path
import os,sys,json,hashlib,gc
import torch
R=Path(__file__).resolve().parents[2]

def cpu_source(target):
 from stvg.source_loader import official_imports,load_model_on_device
 official_imports();pipeline=sys.modules['models.pipeline'];original=pipeline.load_json;opened=[]
 def no_labels(path):
  assert str(path).endswith(('/annos/train.json','/annos/test.json'));opened.append(str(path));return {}
 pipeline.load_json=no_labels
 try:
  cfg=json.loads((R/'configs/default.json').read_text())['datasets'][target];src='vidstg'if cfg['source_dataset']=='vidstg'else'hcstvg2';runtime='vidstg'if src=='vidstg'else'hc-stvg2'
  model,_,meta=load_model_on_device(src,R/'runtime/source'/runtime,checkpoint=R/cfg['source_checkpoint'],device='cpu',source_dataset=src)
 finally:pipeline.load_json=original
 assert len(opened)==2 and model.verb_label=={}and model.verb_label2=={}
 return model.eval().requires_grad_(False),meta

def spatial_modules(model):
 assert model.ground_decoder.decoder.bbox_embed is model.bbox_embed
 return {'pos_fc':model.ground_decoder.pos_fc,'decoder':model.ground_decoder.decoder}

def spatial_parameters(model):
 ids={id(p)for m in spatial_modules(model).values()for p in m.parameters()}
 result=[(n,p)for n,p in model.named_parameters()if id(p)in ids]
 assert len(result)==len(ids)and any('bbox_embed' in n for n,p in result)and any(n.startswith('ground_decoder.pos_fc.')for n,p in result)
 return result

def state_snapshot(model):return {name:{k:v.detach().cpu().clone()for k,v in mod.state_dict().items()}for name,mod in spatial_modules(model).items()}
def restore(model,state):
 for name,mod in spatial_modules(model).items():mod.load_state_dict(state[name],strict=True)
 model.eval().requires_grad_(False)
 for p in model.parameters():p.grad=None
 for name,mod in spatial_modules(model).items():assert all(torch.equal(v.cpu(),state[name][k])for k,v in mod.state_dict().items())
