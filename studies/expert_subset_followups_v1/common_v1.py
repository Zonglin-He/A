"""Independent two-expert followup; original arrays/scientific files immutable."""
from pathlib import Path
import os,sys,time,json,hashlib,shutil,ast,importlib.util
R=Path(__file__).resolve().parents[2];S=Path(__file__).parent;D=R/'outputs/expert_subset_followups_v1';P=R/'results/expert_subset_followups_v1'
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
sys.path.insert(0,str(R/'studies/expert_subset_expansion_v1'))
spec=importlib.util.spec_from_file_location('_immutable_previous_expert_common',R/'studies/expert_subset_expansion_v1/common_v1.py');base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
# Restore this namespace first after loading immutable parent helpers.
sys.path.insert(0,str(S))
import numpy as np,torch
old=base.old;legacy=base.legacy;read=base.read;sha=base.sha;atomic=base.atomic;identity=base.identity;digest=base.digest;pure=base.pure;save=base.save;cpu_tree=base.cpu_tree
EXPERTS=['onlinerefer_r50','sgmg_videoswin_t'];REUSE=[];ARMS=base.ARMS;PAIRS=base.PAIRS
def rows(ds):return read(D/'selection.json')[ds]
def validate():
 base.validate();plan=read(D/'plan.json');assert plan['fingerprint']==digest({k:v for k,v in plan.items()if k!='fingerprint'})
 for group in ['code_sha256','input_sha256','official_source_files','tokenizer_source_files']:
  for f,h in plan[group].items():assert sha(R/f)==h,f
 return plan
def guard():base.guard()
def resources(start):
 assert time.perf_counter()-start<72*3600 and shutil.disk_usage(R).free>=80*2**30
 assert next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines()if x.startswith('MemAvailable:'))>=8*2**30
 assert sum(p.stat().st_size for p in D.rglob('*')if p.is_file())<160*2**30
def read_inputs(ds,row):return base.read_inputs(ds,row)
def from_compact(c):return base.from_compact(c)
# Exact original numerical build, extending only its expert-identity interface.
# OnlineRefer has causal native tracking rather than one global query index.
_path=R/'studies/fresh_external_comparison_v1/math_endpoint_contract_v2.py';_txt=_path.read_text();_node=next(n for n in ast.parse(_txt).body if isinstance(n,ast.FunctionDef)and n.name=='build');_original=ast.get_source_segment(_txt,_node)
_edits=[("teacher['nested_mask_all_false']and teacher['native_same_track_all8']","teacher['nested_mask_all_false']and (teacher['native_same_track_all8']or (teacher.get('native_causal_state_carried_all8')and teacher.get('episodic_reset')and teacher.get('model_steps')==8 and teacher.get('expert_model')=='OnlineRefer-R50-RefYTVOS-sparse-eight-causal-adapter'))"),("selected_query=teacher['selected_query']","selected_query=teacher.get('selected_query',teacher.get('selected_queries'))")]
_source=_original
for a,b in _edits:assert _source.count(a)==1;_source=_source.replace(a,b)
_reverse=_source
for a,b in reversed(_edits):_reverse=_reverse.replace(b,a)
assert _reverse==_original
_namespace=dict(vars(legacy));exec(compile(_source,str(S/'common_v1.py')+'::identity-interface-only','exec'),_namespace);_build=_namespace['build']
def build(cap,raw,teacher,ds):
 arrays,compact=_build(cap,raw,teacher,ds);return from_compact(compact),compact
def commit(folder,key,arrays,compact,record):
 cp=folder/'compact'/(key+'.pt');save(cp,compact);cold=torch.load(cp,map_location='cpu',weights_only=False,mmap=True);again=from_compact(cold);assert set(arrays)==set(again)and all(np.array_equal(v,again[k])for k,v in arrays.items())
 pp=folder/'predictions'/(key+'.npz');pp.parent.mkdir(parents=True,exist_ok=True)
 with pp.with_suffix('.tmp').open('wb')as f:np.savez_compressed(f,**arrays)
 os.replace(pp.with_suffix('.tmp'),pp)
 with np.load(pp)as z:assert set(z.files)==set(arrays)and all(np.array_equal(v,z[k])for k,v in arrays.items())
 rec={**record,'path':str(pp.relative_to(D)),'sha256':sha(pp),'compact_path':str(cp.relative_to(D)),'compact_sha256':sha(cp),'all_cold_arrays_exact':True};rp=folder/'receipts'/(key+'.json');atomic(rp,rec);return{**rec,'receipt_path':str(rp.relative_to(D)),'receipt_sha256':sha(rp)}
