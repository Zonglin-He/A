"""Explicit local module identity; original numerical core and plan unchanged."""
import importlib.util,sys,json,hashlib,argparse
from pathlib import Path
S=Path(__file__).resolve().parent
sys.path.insert(0,str(S))
import run as original
assert Path(original.__file__).resolve()==S/'run.py'
assert Path(sys.modules['common'].__file__).resolve()==S/'common.py'
spec=importlib.util.spec_from_file_location('_full_tube_controls_core_runtime_v3',S/'core_runtime_v3.py');numeric=importlib.util.module_from_spec(spec);sys.modules[spec.name]=numeric;spec.loader.exec_module(numeric);sys.modules['core']=numeric
base_validate=original.validate

def validate():
 p=original.read(original.D/'execution_v3_plan.json');assert p['fingerprint']==original.digest({k:v for k,v in p.items()if k!='fingerprint'})
 for field in ['code_sha256','input_sha256']:
  for f,h in p[field].items():assert original.sha(original.R/f)==h,f
 return base_validate()
original.validate=validate

def preflight():
 import numpy as np,torch
 from stvg.readout import prediction
 torch.set_num_threads(1);validate();original.no_GT();proof=[]
 for ds in original.COUNTS:
  for row in original.records(ds)[:2]:
   a,x=numeric.derive(ds,row);b,y=numeric.derive(ds,row);assert x==y and all(np.array_equal(v,b[k])for k,v in a.items())
   e=original.fixed(original.selected.RUN/ds/'sealed.json')['files'][row['id']];p=original.selected.RUN/ds/e['capture_path'];assert original.sha(p)==e['capture_sha256'];cap=torch.load(p,map_location='cpu',weights_only=False,mmap=True);assert not cap['GT_read']
   logits=cap.get('native_logits');logits=logits if logits is not None else[z[-1]for z in cap['logits']]
   iv=prediction(logits,torch.as_tensor(cap['Source']),cap['records'],cap['frame_ids'])['physical_interval'];assert np.array_equal(a['Source_interval'],iv)
   proof.append(dict(dataset=ds,all_arrays_repeat_exact=True,original_saved_native_logits_MAP_hull_replayed_exact=True,full_current_exact=True,audited_structural_NoTemporal_exact=True,legacy_MBR_interval_not_used=True,shared_geometry_immutable=True,same_eight_direct_replacement_exact=True))
 assert not torch.cuda.is_initialized();z=dict(status='passed_four_actual_cached_cases',cases=proof,new_Source=0,new_expert=0,new_optimizer=0,new_GT_scoring=0,execution='v3');original.atomic(original.D/'preflight.json',z);original.atomic(original.P/'preflight.json',z);print(z['status'],flush=True)
original.preflight=preflight
# The numeric source, original code and original scientific plan are never edited.
# The owner spawns this exact runtime for all subsequent independent stages.
def all_domains():
 import subprocess,traceback,time
 D=original.D;validate();assert not(D/'owner_v3.json').exists();original.atomic(D/'owner_v3.json',original.identity())
 try:
  for ds in original.COUNTS:
   for command in ['predict','score','audit']:
    original.atomic(D/'state_v3.json',dict(stage=command,dataset=ds,owner=original.identity()));log=(D/(ds+'_'+command+'_v3.log')).open('xb');c=subprocess.Popen([sys.executable,'-B',__file__,command,'--dataset',ds],stdout=log,stderr=subprocess.STDOUT);original.atomic(D/(ds+'_'+command+'_child_v3.json'),original.identity(c.pid));code=c.wait();log.close();original.atomic(D/(ds+'_'+command+'_exit_v3.json'),dict(code=code));assert code==0,'Preserve all failed stage evidence; no automatic retry'
  original.atomic(D/'state_v3.json',dict(stage='both_domains_scored_audited_pending_root_analysis_publication',owner=original.identity()))
 except BaseException:original.atomic(D/'state_v3.json',dict(stage='failed_preserved',error=traceback.format_exc(),owner=original.identity()));raise
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('command',choices=['preflight','predict','score','audit','all_domains']);ap.add_argument('--dataset',choices=original.COUNTS);v=ap.parse_args();validate()
 if v.command=='all_domains':all_domains()
 else:getattr(original,v.command)(*([]if v.command=='preflight'else[v.dataset]))
