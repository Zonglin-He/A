"""One finite audited-feature -> fit/seal -> score chain; never starts capture or another experiment."""
from pathlib import Path
import sys,os,time,subprocess
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_fine_dataset_formal64_v1 import D,P,verify
from scripts.run_stvg_joint_event_critic_v1 import read,put,sha

def main():
 p=verify();assert read(D/'FEATURE_BARRIER.json')['count']==64
 codes=['scripts/continue_stvg_fine_dataset_formal64_v1.py','scripts/audit_stvg_fine_dataset_formal64_inputs_v1.py','scripts/audit_stvg_fine_dataset_formal64_v1.py','scripts/score_stvg_opd_p1_v1.py','scripts/run_stvg_dino_local_residual32_v1.py','scripts/score_stvg_joint_robust_correction32_v1.py','scripts/audit_stvg_discriminative_dense32_v1.py']
 pins={f:sha(ROOT/f)for f in codes};put(D/'CPU_PHASE_LOCK.json',dict(status='locked_before_fit_and_GT',pins=pins))
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='');py=sys.executable
 def run(script,*args,timeout=900):
  for f,h in pins.items():assert sha(ROOT/f)==h
  subprocess.run([py,'-B',str(ROOT/script),*args],check=True,timeout=timeout,env=env,cwd=ROOT)
 run('scripts/audit_stvg_fine_dataset_formal64_inputs_v1.py')
 assert read(D/'INPUT_DEPENDENCY_AUDIT.json')['status']=='pass'and read(D/'INPUT_DEPENDENCY_AUDIT.json')['count']==64
 assert time.time()<p['deadline_epoch'];started=time.time();run('scripts/run_stvg_fine_dataset_formal64_v1.py','measure',timeout=min(900,p['deadline_epoch']-time.time()));assert time.time()<p['deadline_epoch'];put(D/'FIT_SCHEDULE_GUARD.json',dict(status='pass',input_audit_pass_before_fit=True,started_epoch=started,finished_epoch=time.time(),within_locked_deadline=True))
 bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');assert bar['count']==64 and bar['all_outputs_before_NEW_GT']
 run('scripts/run_stvg_fine_dataset_formal64_v1.py','score')
 put(D/'CPU_CONTROLLER_COMPLETION.json',dict(status='complete',count=64,all_partials_preserved=True,no_new_GPU_DINO_training=True))
if __name__=='__main__':main()
