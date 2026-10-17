"""Resume eight sealed queries; preserve raw DINO geometry and float32 boundary tolerance."""
from pathlib import Path
import inspect,sys,types
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_r1_geometry32_v2 import BASE,read,sha,put,verify
T=BASE/'arms/DINO/trials/G32';LOCK=BASE/'recovery/DINO_FLOAT_BOUNDARY003/RESUME_LOCK.json'
def state():
 d=read(LOCK)
 for f in d['files']:
  assert sha(ROOT/f['path'])==f['sha256'] and sha(ROOT/f['receipt_path'])==f['receipt_sha256']
 return d
def precheck(trial,roster):
 import torch
 d=state();assert trial==T and len(roster)==32 and not (T/'STUDENT_GLOBAL_BARRIER.json').exists();seen={f['case_id'] for f in d['files']};assert set(p.stem for p in (T/'student').glob('*.pt'))==seen
 for f in d['files']:
  p=torch.load(ROOT/f['path'],map_location='cpu',weights_only=False);assert p['runtime_sha256']==sha(T/'RUNTIME.json') and p['reference_lock_sha256']==sha(T/'REFERENCE_LOCK.json') and p['source_restored'] and not p['GT_read']
def patched(text,enabled):
 from scripts.run_stvg_controls256_speed_student_v1 import source_patch
 text=source_patch(text,enabled)
 changes={
  '    _no_partial_outputs(trial_dir, roster)':'    from scripts.resume_stvg_r1_geometry32_dino_v1 import state as resume_state,precheck as resume_precheck\n    resume_precheck(trial_dir, roster)',
  '    final_files = []':'    resume=resume_state()\n    final_files = list(resume["files"])',
  '    checkpoint_files = []':'    checkpoint_files = list(resume["checkpoint_files"])',
  '    source_capture_count = fit_invocations = final_readout_calls = reused_fit_count = 0':'    source_capture_count=0\n    fit_invocations=final_readout_calls=8\n    reused_fit_count=0',
  '    mode_invocations = {mode: 0 for mode in MODES}':'    mode_invocations = {"M2":8}',
  '    mode_steps = {mode: 0 for mode in MODES}':'    mode_steps = {"M2":40}',
  '    for number, rec in enumerate(roster, 1):\n        cid = _case_id(rec)':'    for number, rec in enumerate(roster, 1):\n        cid = _case_id(rec)\n        if cid in {f["case_id"] for f in resume["files"]}: continue',
 }
 for a,b in changes.items():assert text.count(a)==1,(a,text.count(a));text=text.replace(a,b)
 return text
def anchors():
 from scripts import run_stvg_adjacent_motion_student_v1 as h
 s=inspect.getsource(h._anchors);a='assert 0 <= cx0-w0/2 < cx0+w0/2 <= 1 and 0 <= cy0-h0/2 < cy0+h0/2 <= 1';b='assert -1e-7 <= cx0-w0/2 < cx0+w0/2 <= 1+1e-7 and -1e-7 <= cy0-h0/2 < cy0+h0/2 <= 1+1e-7';assert s.count(a)==1;ns=dict(h.__dict__);exec(compile(s.replace(a,b),__file__,'exec'),ns);h._anchors=ns['_anchors']
def main():
 from scripts import stvg_r1_geometry32_measure_v1 as m
 from scripts import stvg_overnight_student_v1 as s
 from scripts import stvg_pipeline_speed_v1 as speed
 verify();state();anchors();s.source_patch=patched
 if sys.argv[1]=='worker':
  m.configure('DINO');ns=dict(s.__dict__);ns['speed']=types.SimpleNamespace(BASE=speed.BASE,Engine=lambda label,enabled:speed.Engine('r1_geometry32_resume003_'+label,enabled));exec(compile(inspect.getsource(s.worker),__file__,'exec'),ns);ns['worker']()
 else:m.student('root','DINO')
if __name__=='__main__':main()
