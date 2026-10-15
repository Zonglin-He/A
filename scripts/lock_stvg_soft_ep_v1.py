"""Pin the finite direction-only experiment and its exact cached controls."""
from pathlib import Path
import sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_native_instance_common_v1 import read,write,sha
from scripts.lock_stvg_ecld_runtime002 import closure
from scripts.stvg_reference_recovery_fit_v1 import source_hashes
BASE=ROOT/'artifacts/stvg_soft_ep_opd_v1'
def main():
 assert not(BASE/'RUNTIME.json').exists()
 seeds=[ROOT/'scripts'/n for n in ['run_stvg_soft_ep_v1.py','stvg_soft_ep_fit_v1.py','stvg_soft_ep_response_v1.py','launch_stvg_soft_ep_v1.py','lock_stvg_soft_ep_v1.py']]
 src=closure(seeds);pins={str(p.relative_to(ROOT)):sha(p)for p in src};pins.update(source_hashes())
 for n in ['DESIGN_LOCK.json','ROSTER_LOCK.json','REFERENCE_LOCK.json','CPU_CONTRACTS.json']:pins[str((BASE/n).relative_to(ROOT))]=sha(BASE/n)
 parent=ROOT/'artifacts/stvg_evidence_constrained_v1/round2_prompt009'
 for n in ['TEACHER_GLOBAL_BARRIER.json','REFERENCE_LOCK.json','STUDENT_GLOBAL_BARRIER.json','GLOBAL_PREDICTION_BARRIER.json']:pins[str((parent/n).relative_to(ROOT))]=sha(parent/n)
 for rec in read(BASE/'ROSTER_LOCK.json')['records']:
  for key in ['plan_path','subject_path']:pins[rec[key]]=sha(ROOT/rec[key])
 protected={n:sha(ROOT/n)for n in ['methods/CURRENT_METHOD.json','artifacts/stvg_evidence_constrained_v1/FROZEN_METHOD_LOCK.json']}
 write(BASE/'RUNTIME.json',dict(status='locked',pins=pins,protected=protected,scope='four-case actual qualification then matched32 M0/M1/M2, noGTbeforeseal',source_closure_count=len(src),GT_read=False,time=time.time()))
 print('locked',len(src),len(pins))
if __name__=='__main__':main()
