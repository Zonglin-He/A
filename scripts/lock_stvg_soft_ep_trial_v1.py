"""Lock one declared mu trial; M1 is an exact fixed-rule saved reuse."""
from pathlib import Path
import sys,copy,time,shutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_native_instance_common_v1 import read,write,sha
from scripts.lock_stvg_ecld_runtime002 import closure
PRIMARY=ROOT/'artifacts/stvg_soft_ep_opd_v1'
def main():
 label=sys.argv[1];assert label in ['mu025','mu4'];b=PRIMARY/label;b.mkdir(exist_ok=False);mu={'mu025':.25,'mu4':4.}[label]
 for n in ['ROSTER_LOCK.json','REFERENCE_LOCK.json']:shutil.copy2(PRIMARY/n,b/n)
 d=copy.deepcopy(read(PRIMARY/'DESIGN_LOCK.json'));d['response_settings']['mu']=mu;d['scope']='single-factor development trial '+label;d['M1_reuse_from']=str(PRIMARY.relative_to(ROOT));d['primary_design_sha256']=sha(PRIMARY/'DESIGN_LOCK.json');d['GT_read']=False;write(b/'DESIGN_LOCK.json',d)
 q=read(PRIMARY/'ROOT_QUALIFICATION.json');q.update(scope='same operator and native JVP/response/readout ABI qualified in parent; new mu is the only active M2 factor, actual child full decoder outputs still audited',parent_qualification_sha256=sha(PRIMARY/'ROOT_QUALIFICATION.json'),actual_fresh_child_qualification=False,mu=mu);write(b/'ROOT_QUALIFICATION.json',q)
 seeds=[ROOT/'scripts'/n for n in ['run_stvg_soft_ep_trial_v1.py','run_stvg_soft_ep_trial_worker_v1.py','launch_stvg_soft_ep_trial_v1.py','lock_stvg_soft_ep_trial_v1.py']];src=closure(seeds);pins={str(p.relative_to(ROOT)):sha(p)for p in src};pins.update(read(PRIMARY/'RUNTIME.json')['pins'])
 for p in [b/n for n in ['DESIGN_LOCK.json','ROSTER_LOCK.json','REFERENCE_LOCK.json','ROOT_QUALIFICATION.json']]+[PRIMARY/n for n in ['DESIGN_LOCK.json','ROOT_QUALIFICATION.json','STUDENT_GLOBAL_BARRIER.json','ROOT_READBACK.json','GLOBAL_PREDICTION_BARRIER.json','CPU_CONTRACTS.json']]:pins[str(p.relative_to(ROOT))]=sha(p)
 for f in read(PRIMARY/'STUDENT_GLOBAL_BARRIER.json')['files']:pins[f['path']]=f['sha256'];p=(ROOT/f['path']).with_suffix('.receipt.json');pins[str(p.relative_to(ROOT))]=sha(p)
 for row in read(b/'ROSTER_LOCK.json')['records']:
  for key in ['plan_path','subject_path']:pins[row[key]]=sha(ROOT/row[key])
 for n in ['REFERENCE_LOCK.json','STUDENT_GLOBAL_BARRIER.json','GLOBAL_PREDICTION_BARRIER.json']:
  p=ROOT/'artifacts/stvg_evidence_constrained_v1/round2_prompt009'/n;pins[str(p.relative_to(ROOT))]=sha(p)
 write(b/'RUNTIME.json',dict(status='locked',scope='fixed-reference single-mu development with exact M1 reuse',pins=pins,protected=read(PRIMARY/'RUNTIME.json')['protected'],GT_read=False,time=time.time()))
 print('locked',label,mu,len(pins))
if __name__=='__main__':main()
