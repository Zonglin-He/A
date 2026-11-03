"""Independent expert subset namespace; old roots and frozen numerical math read-only."""
from pathlib import Path
import os,sys,time,json,hashlib,functools,shutil,ast
R=Path(__file__).resolve().parents[2];S=Path(__file__).parent;D=R/'outputs/expert_subset_expansion_v1';P=R/'results/expert_subset_expansion_v1'
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
sys.path[:0]=[str(R/'studies/fresh_external_comparison_v1'),str(R/'studies/expert_generalization_v1'),str(R),str(R/'tools')]
import refer_common_v1 as old
import math_endpoint_contract_v2 as legacy
import numpy as np,torch
read=old.read;sha=old.sha;atomic=old.atomic;identity=old.identity;digest=old.digest
EXPERTS=['referformer_r101','mttr'];REUSE=['referdino','referformer_r50'];ARMS=['Source','PureSource','PureOwn','Current'];PAIRS=[('Current','PureSource'),('Current','PureOwn'),('PureOwn','PureSource'),('Current','Source')]
# Reuse the exact already audited pure interpolation function without executing
# its CLI/module-level CUDA environment changes.
_pure_path=R/'studies/pure_expert_interpolation_v1/run.py';_txt=_pure_path.read_text();_node=next(x for x in ast.parse(_txt).body if isinstance(x,ast.FunctionDef)and x.name=='pure_interpolation');_namespace={'np':np};exec(compile(ast.get_source_segment(_txt,_node),str(_pure_path),'exec'),_namespace);pure=_namespace['pure_interpolation']
def rows(ds):return read(D/'selection.json')[ds]
def validate():
 plan=read(D/'plan.json');assert plan['fingerprint']==digest({k:v for k,v in plan.items()if k!='fingerprint'})
 for group in ['code_sha256','input_sha256','mttr_source_files','referformer_source_files']:
  for f,h in plan[group].items():assert sha(R/f)==h,f
 return plan
def guard():old.rc.gt_guard()
def resources(start):
 assert time.perf_counter()-start<72*3600;assert shutil.disk_usage(R).free>=80*2**30
 assert next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines()if x.startswith('MemAvailable:'))>=8*2**30
 assert sum(p.stat().st_size for p in D.rglob('*')if p.is_file())<160*2**30
 return True
def save(path,value):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');torch.save(value,tmp);os.replace(tmp,path)
def cpu_tree(x):
 if torch.is_tensor(x):return x.detach().cpu()
 if isinstance(x,dict):return{k:cpu_tree(v)for k,v in x.items()}
 if isinstance(x,(list,tuple)):return type(x)(cpu_tree(v)for v in x)
 return x
def build(cap,raw,teacher,ds):
 arrays,compact=legacy.build(cap,raw,teacher,ds);return from_compact(compact),compact

def from_compact(c):
 if c.get('schema')=='reuse_old_Current_only_derive_new_pure_v1':return reuse_arrays(c)
 original=legacy.replay(c);pure_boxes=pure(c['frame_ids'],c['positions8'],c['native_boxes'])
 return dict(Source=original['Source'],Source_interval=original['Source_interval'],PureSource=pure_boxes.copy(),PureSource_interval=original['Source_interval'].copy(),PureOwn=pure_boxes.copy(),PureOwn_interval=original['Ours_interval'].copy(),Current=original['Ours'],Current_interval=original['Ours_interval'],E=original['E'],frame_ids=original['frame_ids'],positions8=original['positions8'],expert_boxes=c['native_boxes'].copy())
def read_inputs(ds,row):
 cap,raw,frames,t,refs,proof=old.payload(ds,row);assert len(cap['positions8'])==8
 return cap,raw,frames,t,refs,proof

def reuse_payload(expert,ds,row):
 cap,raw,frames,t,refs,proof=read_inputs(ds,row)
 if expert=='referdino':return cap,raw,frames,t,refs,proof,None
 assert expert=='referformer_r50';B=R/'outputs/fresh_external_comparison_v1/referformer_v4';e=read(B/'pilot_v1'/ds/'sealed.json')['files'][row['id']];assert e['input_binding']==proof;info=e['expert_payload'];tp=B/info['path'];assert sha(tp)==info['sha256'];teacher=torch.load(tp,map_location='cpu',weights_only=False,mmap=True)
 return cap,raw,frames,teacher,refs,proof,dict(path=str(tp.relative_to(R)),sha256=info['sha256'],receipt_sha256=e['receipt_sha256'],original_expert_forward_cost_inherited_nonzero=True)
def commit(folder,key,arrays,compact,record):
 cp=folder/'compact'/(key+'.pt');save(cp,compact);cold=torch.load(cp,map_location='cpu',weights_only=False,mmap=True);again=from_compact(cold);assert set(arrays)==set(again)and all(np.array_equal(v,again[k])for k,v in arrays.items())
 pp=folder/'predictions'/(key+'.npz');pp.parent.mkdir(parents=True,exist_ok=True)
 with pp.with_suffix('.tmp').open('wb')as f:np.savez_compressed(f,**arrays)
 os.replace(pp.with_suffix('.tmp'),pp)
 with np.load(pp)as z:assert set(z.files)==set(arrays)and all(np.array_equal(v,z[k])for k,v in arrays.items())
 rec={**record,'path':str(pp.relative_to(D)),'sha256':sha(pp),'compact_path':str(cp.relative_to(D)),'compact_sha256':sha(cp),'all_cold_arrays_exact':True};rp=folder/'receipts'/(key+'.json');atomic(rp,rec);return{**rec,'receipt_path':str(rp.relative_to(D)),'receipt_sha256':sha(rp)}

def reuse_arrays(c):
 p=pure(c['frame_ids'],c['positions8'],c['native_boxes'])
 return dict(Source=c['Source'].copy(),Source_interval=c['Source_interval'].copy(),PureSource=p.copy(),PureSource_interval=c['Source_interval'].copy(),PureOwn=p.copy(),PureOwn_interval=c['Current_interval'].copy(),Current=c['Current'].copy(),Current_interval=c['Current_interval'].copy(),E=c['E'].copy(),frame_ids=c['frame_ids'].copy(),positions8=c['positions8'].copy(),expert_boxes=c['native_boxes'].copy())
def cached_reuse_derive(expert,ds,row):
 assert expert in REUSE
 B=R/'outputs/fresh_external_comparison_v1';base=B if expert=='referdino'else B/'referformer_v4';seal=base/ds/'sealed.json'if expert=='referdino'else base/'pilot_v1'/ds/'sealed.json';e=read(seal)['files'][row['id']]
 pp=base/e['path'];assert sha(pp)==e['sha256']
 if expert=='referdino':
  info=e['current_teacher'];tp=Path(info['payload_path']);th=info['payload_sha256']
 else:info=e['expert_payload'];tp=base/info['path'];th=info['sha256']
 assert sha(tp)==th;teacher=torch.load(tp,map_location='cpu',weights_only=False,mmap=True)
 with np.load(pp)as z:
  c=dict(schema='reuse_old_Current_only_derive_new_pure_v1',Source=z['Source'].copy(),Source_interval=z['Source_interval'].copy(),Current=z['Ours'].copy(),Current_interval=z['Ours_interval'].copy(),E=z['E'].copy(),frame_ids=z['frame_ids'].copy(),positions8=z['positions8'].copy(),native_boxes=teacher['boxes'].numpy().copy())
 proof=dict(original_prediction_path=str(pp.relative_to(R)),original_prediction_sha256=e['sha256'],original_seal_sha256=sha(seal),expert_payload_path=str(tp.relative_to(R)),expert_payload_sha256=th,Current_copied_without_rerunning_inference=True,prior_full_input_and_array_audit_reused=True,GT_read=False)
 return reuse_arrays(c),c,proof
