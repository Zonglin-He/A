"""Freeze one finite original-expert reference-swap stage before model use."""
from pathlib import Path
import sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_native_instance_common_v1 import read,write,sha
from scripts.lock_stvg_ecld_runtime002 import closure
from scripts import run_stvg_reference_swap_teacher_v1 as t
from scripts.stvg_reference_recovery_fit_v1 import source_hashes
BASE=t.BASE

def main():
 stage=sys.argv[1];assert stage in ['teacher','student'];runtime=BASE/('RUNTIME.json'if stage=='teacher'else'STUDENT_RUNTIME.json');assert not runtime.exists()
 seeds=[ROOT/'scripts'/s for s in ['launch_stvg_reference_swap_v1.py','lock_stvg_reference_swap_v1.py','run_stvg_reference_swap_'+stage+'_v1.py']]
 sources=closure(seeds);pins={str(p.relative_to(ROOT)):sha(p)for p in sorted(sources)};pins.update(source_hashes())
 for rel in t.REQUIRED_PIN_FILES:pins[rel]=sha(ROOT/rel)
 for p in [BASE/'DESIGN_LOCK.json',BASE/'ROSTER_LOCK.json',BASE/'TEACHER_JOB_LOCK.json',t.ROSTER_IN,t.SOURCE_BARRIER,t.PARENT_REFERENCE]:pins[str(p.relative_to(ROOT))]=sha(p)
 assets=t._asset_bindings()
 for name in list(t.ASSET_FILES)+list(t.TOKENIZER_FILES):
  p=ROOT/t.SNAPSHOT_REL/name;pins[str(p.relative_to(ROOT))]=sha(p)
 for rec in read(t.ROSTER_IN)['records']:
  for key in ['plan_path','subject_path']:
   p=ROOT/rec[key];pins[str(p.relative_to(ROOT))]=sha(p)
 if stage=='student':
  for p in [BASE/'RUNTIME.json',BASE/'TEACHER_GLOBAL_BARRIER.json',BASE/'REFERENCE_LOCK.json']:
   pins[str(p.relative_to(ROOT))]=sha(p)
  for item in read(BASE/'TEACHER_GLOBAL_BARRIER.json')['files']:pins[item['path']]=item['sha256']
 protected={p:sha(ROOT/p)for p in ['methods/CURRENT_METHOD.json','artifacts/stvg_evidence_constrained_v1/FROZEN_METHOD_LOCK.json']}
 import torch,torchvision,transformers,numpy
 environment={k:str(v.__version__)for k,v in [('torch',torch),('torchvision',torchvision),('transformers',transformers),('numpy',numpy)]}
 bindings=dict(design_lock_sha256=sha(BASE/'DESIGN_LOCK.json'),roster_sha256=sha(BASE/'ROSTER_LOCK.json'),source_roster_sha256=sha(t.ROSTER_IN),teacher_job_lock_sha256=sha(BASE/'TEACHER_JOB_LOCK.json'),source_capture_barrier_sha256=sha(t.SOURCE_BARRIER),parent_wf_reference_lock_sha256=sha(t.PARENT_REFERENCE),**{k:assets[k]for k in ['expert_checkpoint_sha256','expert_config_sha256','processor_config_sha256']})
 write(runtime,dict(status='locked',scope='reference_swap_'+stage,pins=pins,protected=protected,environment=environment,bindings=bindings,local_static_python_closure_count=len(sources),GT_read=False,time=time.time()))
 print('locked',stage,len(sources),len(pins))
if __name__=='__main__':main()
