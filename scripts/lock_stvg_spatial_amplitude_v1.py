"""Freeze finite alpha schedule and existing source dependencies before new readout."""
import sys,json,hashlib,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_spatial_update_amplitude_v1 import PHASE_REQUIRED_PIN_PATHS,_preflight
B=ROOT/'artifacts/stvg_spatial_causal_decomposition_v1/phase1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):
 assert not p.exists();p.write_text(json.dumps(d,indent=2)+'\n')
old=json.loads((ROOT/'artifacts/stvg_motion_guided_instance_opd_v1/METHOD_STUDENT_RUNTIME002.json').read_text())
paths=set(PHASE_REQUIRED_PIN_PATHS)|set(old['pins'])|{'scripts/launch_stvg_spatial_amplitude_v1.py','scripts/lock_stvg_spatial_amplitude_v1.py','protocols/stvg_spatial_causal_decomposition_v1.md'}
bar=json.loads((ROOT/'artifacts/stvg_motion_guided_instance_opd_v1/METHOD_STUDENT_GLOBAL_BARRIER.json').read_text());paths|={r['path']for r in bar['files']}
pins={p:sha(ROOT/p)for p in sorted(paths)}
lock=dict(status='locked',phase='phase1',GT_read=False,alpha_values=[0,.25,.5,1],pins=pins,scope='same sealed 16; 3 nonzero directions; 3 source captures + 9 additional full readouts',new_fits=0,new_teacher_calls=0,time=time.time())
save(B/'PHASE1_LOCK.json',lock);pins={**pins,str((B/'PHASE1_LOCK.json').relative_to(ROOT)):sha(B/'PHASE1_LOCK.json')}
save(B/'RUNTIME.json',dict(status='locked',phase='phase1',GT_read=False,pins=pins,protected=old['protected'],phase1_lock_sha256=sha(B/'PHASE1_LOCK.json'),cpu_contract_sha256=sha(B/'CPU_AMPLITUDE_CONTRACTS.json'),time=time.time()))
_preflight();print('preflight pass')
