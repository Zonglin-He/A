"""Metadata-only synthetic epoch100 finalization fixture; no actual model/GT/CUDA."""
import ast,hashlib,json,tempfile
from pathlib import Path
import torch
R=Path(__file__).resolve().parents[3];code=R/'crossplatform/round6/train_source.py';A=R/'research/round6/audit'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
fn=[x for x in ast.parse(code.read_text()).body if isinstance(x,ast.FunctionDef) and x.name=='recover_completed_epoch100'];assert len(fn)==1
ns={'Path':Path,'state_fingerprint':lambda ignored:'synthetic_restored_model','sha':sha,'json':json,'torch':torch};exec(compile(ast.Module(body=fn,type_ignores=[]),str(code),'exec'),ns)
with tempfile.TemporaryDirectory(dir=A) as td:
 root=Path(td);ck=root/'epoch100.pth';vd=root/'val';vd.mkdir();torch.save({'epoch':100,'trained_platforms':['waymo'],'source_provenance':{'plan_sha256':'synthetic_plan'}},ck)
 seal={'epoch':100,'n':3782,'checkpoint_sha256':sha(ck),'model_state_fingerprint':'synthetic_restored_model'};(vd/'sealed.json').write_text(json.dumps(seal));entry={'epoch':100,'checkpoint_path':str(ck),'checkpoint_sha256':sha(ck),'val_prediction_dir':str(vd),'val_prediction_seal_sha256':sha(vd/'sealed.json')};(root/'MILESTONES.jsonl').write_text(json.dumps(entry)+'\n');assert ns['recover_completed_epoch100'](root,None,'synthetic_plan')==entry
 ns['state_fingerprint']=lambda ignored:'different_restored_model'
 try:ns['recover_completed_epoch100'](root,None,'synthetic_plan');raise RuntimeError('mismatch incorrectly accepted')
 except AssertionError:pass
result={'status':'pass_metadata_only_synthetic_fixture','train_source_sha256':sha(code),'matching_sameplan_epoch100_checkpoint_and_predictionseal_recovers':True,'different_restored_state_fingerprint_rejected':True,'no_model_GT_CUDA_or_real_training_checkpoint_access':True};(A/'CHECKPOINT_FINALIZE_CONTRACT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
