"""Lock a finite reference-recovery prediction stage before its GPU execution."""
from pathlib import Path
import sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_native_instance_common_v1 import read,write,sha
from scripts.lock_stvg_ecld_runtime002 import closure
from scripts.stvg_reference_recovery_fit_v1 import source_hashes

def main():
 stage=sys.argv[1];assert stage in ['optimizer','carry']
 base=ROOT/'artifacts/stvg_reference_recovery_v1'/('Optimizer_replay'if stage=='optimizer'else'LN_carry')
 assert not(base/'RUNTIME.json').exists()
 seeds=[ROOT/'scripts'/x for x in ['launch_stvg_reference_recovery_v1.py','lock_stvg_reference_recovery_v1.py','stvg_reference_recovery_common_v1.py','stvg_reference_recovery_fit_v1.py',('run_stvg_reference_optimizer_v1.py'if stage=='optimizer'else'run_stvg_reference_carry_v1.py')]]
 sources=closure(seeds);pins={str(p.relative_to(ROOT)):sha(p)for p in sorted(sources)};pins.update(source_hashes())
 for name in ['DESIGN_LOCK.json','ROSTER_LOCK.json','REFERENCE_LOCK.json']+(['ORDERS.json','ROOT_CARRY_CPU_PROOF.json']if stage=='carry'else[]):
  p=base/name;pins[str(p.relative_to(ROOT))]=sha(p)
 parent=ROOT/'artifacts/stvg_evidence_constrained_v1/round2_prompt009'
 for name in ['TEACHER_GLOBAL_BARRIER.json','TEACHER_JOB_LOCK.json','CANDIDATE_INPUT_BARRIER.json','REFERENCE_LOCK.json','STUDENT_GLOBAL_BARRIER.json','GLOBAL_PREDICTION_BARRIER.json']:
  p=parent/name;pins[str(p.relative_to(ROOT))]=sha(p)
 for p in [ROOT/'artifacts/stvg_final_table1_v1/PREFLIGHT_ROSTER_LOCK.json',ROOT/'artifacts/stvg_reference_recovery_v1/CPU_FIT_CONTRACTS.json',ROOT/'artifacts/stvg_reference_recovery_v1/ROOT_CURRENT_FITTER_EQUIVALENCE.json']:
  pins[str(p.relative_to(ROOT))]=sha(p)
 if stage=='carry':
  for ds in ['hc2','vidstg']:
   p=ROOT/'artifacts/decota_paper_experiments_v1'/ds/'PLAN.json';pins[str(p.relative_to(ROOT))]=sha(p)
 protected={str(p.relative_to(ROOT)):sha(p)for p in [ROOT/'methods/CURRENT_METHOD.json',ROOT/'artifacts/stvg_evidence_constrained_v1/FROZEN_METHOD_LOCK.json']}
 write(base/'RUNTIME.json',dict(status='locked',scope=stage,pins=pins,protected=protected,local_static_python_closure_count=len(sources),GT_read=False,bootstrap_executed=False,full_started=False,time=time.time()))
 print('locked',stage,len(sources),len(pins))
if __name__=='__main__':main()
