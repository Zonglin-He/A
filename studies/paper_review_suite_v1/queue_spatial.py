import sys,time,subprocess
from pathlib import Path
R=Path(__file__).resolve().parents[2];sys.path[:0]=[str(R/'tools')]
from full import read,atomic,identity,sha
D=R/'outputs/paper_review_suite_v1';p=read(D/'spatial_controls_plan.json');assert sha(Path(__file__))==p['queue_sha256'];atomic(D/'spatial_controls_controller.json',identity());atomic(D/'spatial_controls_state.json',dict(status='waiting_shared_native_grids',controller=identity()))
while not(D/'supports_sealed.json').exists():
 who=read(D/'supports_launch.json')
 try:st=Path(f"/proc/{who['pid']}/stat").read_text().rsplit(')',1)[1].split();assert int(st[19])==who['startticks']and st[0]!='Z'
 except (OSError,AssertionError):raise RuntimeError('Shared grid owner failed; preserve log')
 time.sleep(10)
for command in ['predict','score']:
 atomic(D/'spatial_controls_state.json',dict(status=command,controller=identity()));c=subprocess.Popen([sys.executable,'-B',str(R/'studies/paper_review_suite_v1/spatial_controls.py'),command]);atomic(D/('spatial_controls_'+command+'_child.json'),identity(c.pid));code=c.wait()
 if code:atomic(D/'spatial_controls_state.json',dict(status='failed_preserved',command=command,exit=code));raise RuntimeError('Spatial controls failed before completion, no automatic rerun')
atomic(D/'spatial_controls_state.json',dict(status='complete_scored_pending_publication',fingerprint=p['fingerprint']))
