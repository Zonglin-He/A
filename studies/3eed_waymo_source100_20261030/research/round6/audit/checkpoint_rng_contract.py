"""CPU-only serialization/RNG fixture; no task model/data, CUDA explicitly mocked."""
import ast, hashlib, io, json, random
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[3]
CODE=ROOT/'crossplatform/round6/checkpointing.py'
source=CODE.read_text();tree=ast.parse(source)
funcs=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in ('rng_state','restore_rng')]
ns={'torch':torch,'random':random,'np':np}
get_original=torch.cuda.get_rng_state_all;set_original=torch.cuda.set_rng_state_all
try:
 torch.cuda.get_rng_state_all=lambda: []
 torch.cuda.set_rng_state_all=lambda states: None
 exec(compile(ast.Module(body=funcs,type_ignores=[]),str(CODE),'exec'),ns)
 g=torch.Generator().manual_seed(17);random.seed(13);np.random.seed(19);torch.manual_seed(23)
 state=ns['rng_state'](g)
 def draw():return {'python':[random.random() for _ in range(3)],'numpy':np.random.rand(3).tolist(),'torch':torch.rand(3).tolist(),'loader_shared_generator':torch.rand(3,generator=g).tolist()}
 expected=draw();payload=io.BytesIO();torch.save({'rng':state},payload);payload.seek(0);loaded=torch.load(payload,weights_only=True,map_location='cpu');draw();ns['restore_rng'](loaded['rng'],g);actual=draw();assert actual==expected
 result={'status':'pass_CPU_only_RNG_serialization','checkpointing_sha256':hashlib.sha256(CODE.read_bytes()).hexdigest(),'weights_only_true_roundtrip':True,'python_numpy_torchCPU_sharedloader_exact':actual==expected,'cuda_state_path':'source-reviewed get_rng_state_all/set_rng_state_all only; deliberately mocked here, no CUDA initialization','no_task_model_dataset_or_GPU':True}
 (ROOT/'research/round6/audit/CHECKPOINT_RNG_CONTRACT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
finally:
 torch.cuda.get_rng_state_all=get_original;torch.cuda.set_rng_state_all=set_original
