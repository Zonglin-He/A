"""Lock this finite scope's actual code/input bindings, without starting workers."""
import sys,time,ast
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_temporal_caption_query_v1 import read,write,sha
B=ROOT/'artifacts/stvg_adjacent_motion_opd_v1';O=ROOT/'artifacts/stvg_motion_guided_instance_opd_v1'
def main(phase):
 assert phase in ('teacher','student')
 proto='protocols/stvg_adjacent_motion_opd_v1.md';assert sha(ROOT/proto)==read(B/'DESIGN_LOCK.json')['protocol_sha256']
 old=read(O/('RUNTIME.json' if phase=='teacher' else 'METHOD_STUDENT_RUNTIME002.json'))
 pins={p:h for p,h in old['pins'].items() if not p.startswith(('artifacts/stvg_motion_guided_instance_opd_v1/','scripts/run_stvg_motion_guided','scripts/launch_stvg_motion_guided','protocols/stvg_motion_guided'))}
 paths=[proto,'scripts/lock_stvg_adjacent_motion_runtime_v1.py','scripts/prepare_stvg_adjacent_motion_observation001.py','scripts/stvg_adjacent_affine_tracklet_v1.py','scripts/prepare_stvg_motion_guided_v1.py','scripts/prepare_stvg_motion_order_layout001.py','scripts/run_stvg_sparse_r1_v2.py','scripts/stvg_sparse_r1_prompt_v1.py','scripts/review_stvg_adjacent_motion_inputs_v1.py']
 paths += [str((B/n).relative_to(ROOT)) for n in ['ROSTER_LOCK.json','DESIGN_LOCK.json','CANDIDATE_INPUT_BARRIER.json','TEACHER_JOB_LOCK.json','CPU_TRACKLET_CONTRACTS.json','ROOT_INPUT_READBACK.json','ROOT_CPU_MOTION_CONTRACTS.json']]
 if phase=='teacher':paths += ['scripts/run_stvg_adjacent_motion_teacher_v1.py','scripts/launch_stvg_adjacent_motion_teacher_v1.py','scripts/review_stvg_adjacent_motion_teacher_v1.py']
 else:
  paths += ['scripts/run_stvg_adjacent_motion_student_v1.py','scripts/launch_stvg_adjacent_motion_student_v1.py','scripts/stvg_adjacent_motion_fit_v1.py','scripts/stvg_video_global_spatial_fit_fp32_geometry001.py','scripts/review_stvg_adjacent_motion_teacher_v1.py']
  paths += [str((B/n).relative_to(ROOT)) for n in ['METHOD_REFERENCE_LOCK.json','ROOT_TEACHER_READBACK.json','TEACHER_GLOBAL_BARRIER.json']]
  paths += ['artifacts/stvg_temporal_caption_query_spatial_opd_v1/STUDENT_GLOBAL_BARRIER.json']+[f'artifacts/stvg_temporal_caption_query_spatial_opd_v1/student/{r["case_id"]}.pt' for r in read(B/'ROSTER_LOCK.json')['records']]
 for p,h in pins.items():assert sha(ROOT/p)==h,p
 for p in paths:
  assert (ROOT/p).is_file(),p
  if p.endswith('.py'):ast.parse((ROOT/p).read_text())
  pins[p]=sha(ROOT/p)
 for p,h in old['protected'].items():assert sha(ROOT/p)==h,p
 rt=dict(status='locked',phase='method' if phase=='student' else 'teacher',scope='joint' if phase=='student' else 'adjacent_motion_P1',count=8,GT_read=False,pins=pins,protected=old['protected'],time=time.time())
 if phase=='teacher':rt['teacher_requests']=48
 else:rt.update(steps=5,expected_fit_count=read(B/'METHOD_REFERENCE_LOCK.json')['expected_fit_count'],roster_lock_sha256=sha(B/'ROSTER_LOCK.json'),reference_lock_sha256=sha(B/'METHOD_REFERENCE_LOCK.json'))
 dst=B/('RUNTIME.json' if phase=='teacher' else 'STUDENT_RUNTIME.json');write(dst,rt);print(phase,'locked',len(pins),sha(dst))
if __name__=='__main__':main(sys.argv[1])
