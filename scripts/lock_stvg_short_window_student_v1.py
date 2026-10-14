"""Root-reviewed frozen-source OPD runtime; no model or GT execution."""
import sys,json,hashlib,time
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from scripts import run_stvg_short_window_student_v1 as worker
from scripts.stvg_video_global_spatial_fit_fp32_geometry001 import SOURCE_BINDINGS
B=R/'artifacts/stvg_short_event_tracklets_v1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
def save(p,d):
 assert not p.exists(),p
 p.write_text(json.dumps(d,indent=2)+'\n')
def main():
 assert sha(R/'scripts/run_stvg_short_window_student_v1.py')=='c57db36a636090f9fc8c486a5ee9fd0fae669166d06bda7137b17a31943db233'
 assert sha(R/'scripts/test_stvg_short_window_student_cpu_v1.py')=='b424b2e72c707709d0d0dd2289f0d90bf55c28c4459cbaeaf04dde9d1d59513f'
 oldroster=R/'artifacts/stvg_adjacent_motion_opd_v1/ROSTER_LOCK.json'
 roster=read(oldroster);assert roster['status']=='locked' and roster['count']==8 and roster['GT_read'] is False
 save(B/'ROSTER_LOCK.json',{**roster,'exact_source_roster_sha256':sha(oldroster),'reuse_scope':'same fixed historically exposed eight sources'})
 refs=read(B/'METHOD_REFERENCE_LOCK.json');assert refs['status']=='locked'and refs['GT_read']is False
 fits=sum(bool(r['references'][role]['anchors']) for r in refs['records'] for role in worker.ROLES)
 paths=set(worker.REQUIRED_STATIC_PINS)|set(worker._old_t_pin_paths())|set(SOURCE_BINDINGS)|{'scripts/lock_stvg_short_window_student_v1.py','scripts/test_stvg_short_window_student_cpu_v1.py','scripts/lock_stvg_short_window_references_v1.py','artifacts/stvg_short_event_tracklets_v1/FEATURE_CAPTURE_LOCK.json','artifacts/stvg_short_event_tracklets_v1/RUNTIME.json','artifacts/stvg_short_event_tracklets_v1/ROOT_FEATURE_READBACK.json','artifacts/stvg_short_event_tracklets_v1/ROOT_TRACKLET_READBACK.json','artifacts/stvg_adjacent_motion_opd_v1/ROSTER_LOCK.json'}
 old=read(B/'FEATURE_RUNTIME.json');paths|=set(old['pins']);protected=old['protected'];assert protected['methods/CURRENT_METHOD.json']=='146a2caf1633544a844ea33106c83c022c0bebad5f3f49a87bb5f81e6f8568d4'
 assert all(sha(R/p)==h for p,h in protected.items())
 save(B/'STUDENT_RUNTIME.json',dict(status='locked',GT_read=False,steps=5,active_parameters=1792,expected_fit_count=fits,reference_lock_sha256=sha(B/'METHOD_REFERENCE_LOCK.json'),pins={p:sha(R/p)for p in sorted(paths)},protected=protected,time=time.time()))
 worker._read_locked_inputs()
 save(B/'ROOT_STUDENT_PREFLIGHT.json',dict(status='pass',count=8,expected_fit_count=fits,GT_read=False,GPU_used=False,runtime_sha256=sha(B/'STUDENT_RUNTIME.json'),time=time.time()))
 print(json.dumps(dict(status='pass',count=8,expected_fit_count=fits)))
if __name__=='__main__':main()
